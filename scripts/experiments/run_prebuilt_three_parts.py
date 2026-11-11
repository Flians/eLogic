#!/usr/bin/env python3
"""Freeze per-configuration libraries, then evaluate all designs with private deltas.

Primary: 416 native prebuilt observations + 32 fresh ABC observations.
Separate construction: 416 cold source observations; private warm cache controls:
80 full-configuration observations. Total 944, without rewriting previous data.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time
import traceback

import run_prebuffer_three_parts as base
from run_joint_three_parts import ARMS, FLAGS
from run_isolated_mem_eth import measured_call
from build_readonly_profile_library import build, snapshot, sha
from preserve_cold_profile_pools import preserve

PROJECT=Path(__file__).resolve().parents[2]
CONTROL=PROJECT/'results_epmapping_joint_prebuffer_v1_20261111'


def save(path,data):
    tmp=path.with_suffix('.new')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def specs():
    return ([(stage,'ours',design,{}) for stage in ['main','mfs3','strong_common','main_r1']
             for design in base.DESIGNS]
            +[('ablation',arm,design,flags) for design in base.DESIGNS for arm,flags in ARMS.items()])


def family(stage,arm):
    return arm if stage=='ablation' else 'full'


def initialize(root,engine,cpus):
    assert not root.exists(),('use a new campaign root',root)
    root.mkdir(parents=True)
    # Original physical inputs, native source snapshots and standard auditors.
    m=base.initialize(root,engine,len(cpus))
    old=json.loads((CONTROL/'manifest.json').read_text())
    for rel,digest in old['input_sha256'].items():
        if rel.startswith('benchmarks/') or rel in ['abc','asap7.genlib','asap7_6in.genlib','database.super']:
            assert sha(root/'inputs'/rel)==digest,rel
    m['goal'].update(FLAGS)
    m['goal'].pop('MAP_EGG_PROFILE_READONLY_DIR',None)
    for name in ['normalized','normalization_abc_blif']:
        (root/name).symlink_to((CONTROL/name).resolve(),target_is_directory=True)
    shutil.copy2(CONTROL/'abc_import_proofs.json',root/'abc_import_proofs.json')
    source=root/'inputs/sources'
    for name in ['run_prebuilt_three_parts.py','build_readonly_profile_library.py',
                 'run_joint_three_parts.py','run_isolated_mem_eth.py',
                 'run_prebuilt_profile_pilot.py','audit_prebuilt_profile_pilot.py',
                 'preserve_cold_profile_pools.py',
                 'tests/test_prebuilt_profile_pool.py','tests/test_readonly_profile_library_builder.py']:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PROJECT/'scripts/experiments'/name,target)
    context=root/'build_context';context.mkdir()
    for name,src in [('cxx_flags.make','build/epmapping/CMakeFiles/epmapping_check.dir/flags.make'),
                     ('Cargo.lock','Cargo.lock'),('map_egg_Cargo.toml','map_egg/Cargo.toml')]:
        shutil.copy2(PROJECT/src,context/name)
    m.update(protocol='shared-readonly-prebuilt-three-parts-v1',primary_mode='prebuilt',
             control_campaign=str(CONTROL),worker_cpus=cpus,host_exclusive=False,
             cold='empty private source pool; construction cost, not main post runtime',
             warm='new process using its paired cold private pool; full configurations only',
             prebuilt='frozen per-arm shared library; empty private delta for every run',
             library_training_scope='all sixteen evaluated designs; repeated-workload evaluation, not held-out generalization',
             library_isolation='separate frozen library for each ablation arm; full shared across primary stages',
             expected_observations=944,normalization_sha256=sha(root/'normalized/manifest.json'))
    m['three_part_plan'].update(arms=ARMS,expected_method_runs=912,expected_abc_runs=32,
        expected_observations=944,primary_method_runs=416,primary_abc_runs=32,
        construction_runs=416,cache_warm_runs=80,jobs=len(cpus))
    m['normalized_sha256']={str(p.relative_to(root/'normalized')):sha(p)
                           for p in (root/'normalized').rglob('*') if p.is_file()}
    m['input_sha256']={str(p.relative_to(root/'inputs')):sha(p)
                       for p in (root/'inputs').rglob('*') if p.is_file()}
    save(root/'manifest.json',m)
    return m


def execute(root,m,cpus,tasks,phase):
    work=queue.Queue()
    for task in tasks:work.put(task)
    lock=threading.Lock();finished=[];errors=[];cold_receipts={};start=time.monotonic()
    def status():
        save(root/(phase+'_status.json'),dict(phase=phase,complete=len(finished)==len(tasks) and not errors,
             finished=len(finished),expected=len(tasks),errors=errors,elapsed_seconds=time.monotonic()-start,
             worker_cpus=cpus,host_exclusive=False))
    def worker(cpu):
        os.sched_setaffinity(0,{cpu})
        while True:
            try:stage,arm,design,flags=work.get_nowait()
            except queue.Empty:return
            try:
                if phase=='construction':
                    cache=root/'cache_sources'/stage/arm/design
                    base.native(root,m,stage,arm,design,'cold',cache,flags)
                    if arm in ['ours','full']:
                        base.native(root,m,stage,arm,design,'warm',cache,flags)
                        receipt=preserve(root,stage,arm,design)
                        with lock:
                            cold_receipts['/'.join([stage,arm,design])]=receipt
                            save(root/'cold_pool_receipts.json',dict(complete=len(cold_receipts)==80,
                                expected=80,records=cold_receipts))
                else:
                    library=root/'shared_libraries'/family(stage,arm)
                    options=dict(flags,MAP_EGG_PROFILE_READONLY_DIR=str(library))
                    private=root/'private_deltas'/stage/arm/design
                    base.native(root,m,stage,arm,design,'prebuilt',private,options)
                    if stage in ['mfs3','strong_common']:
                        cohort='direct6' if stage=='mfs3' else 'strong6'
                        base.abc(root,m,stage,design,'prebuilt',root/'normalized'/cohort/(design+'.blif'))
                with lock:finished.append((stage,arm,design));status()
            except Exception as error:
                traceback.print_exc()
                with lock:errors.append(dict(spec=[stage,arm,design],error=repr(error),cpu=cpu));status()
            finally:work.task_done()
    status()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        list(pool.map(worker,cpus))
    assert not errors,errors
    return dict(phase=phase,elapsed_seconds=time.monotonic()-start,finished=len(finished))


def freeze_libraries(root,m,tasks):
    receipt=json.loads((root/'cold_pool_receipts.json').read_text())
    assert receipt['complete'] and len(receipt['records'])==80
    excluded={name:row for name,row in receipt['records'].items() if name.startswith(('main/','main_r1/'))}
    assert len(excluded)==32
    save(root/'training_pool_selection.json',dict(complete=True,excluded=excluded,
        full_bank_training='mfs3/ours + strong_common/ours + ablation/full: 48 cold runs; full47 namespace uses only own 16 cold sources',
        other_banks_training='own 16 ablation cold sources per configuration',all_cold_controls_retained=True))
    library_root=root/'shared_libraries';library_root.mkdir()
    groups={name:{} for name in ARMS};costs={name:[] for name in ARMS}
    for stage,arm,design,_ in tasks:
        if stage in ['main','main_r1']:continue  # duplicate full47 cache controls, not training
        group=family(stage,arm)
        groups[group][stage+'/'+arm+'/'+design]=root/'cache_sources'/stage/arm/design
        record=json.loads((root/stage/arm/('cold.'+design)/'result.json').read_text())
        resources=json.loads((root/stage/arm/('cold.'+design)/'log.resources.json').read_text())
        costs[group].append(dict(stage=stage,arm=arm,design=design,status=record['status'],
                                 post_seconds=record['optimizer_seconds'],**resources))
    reports={};hashes={}
    for group in ARMS:
        rows=costs[group]
        provenance=dict(engine_sha256=m['engine_sha256'],manifest_sha256=sha(root/'manifest.json'),
             configuration=group,overrides=ARMS[group],source_cases=rows,
             construction_post_seconds=sum(r['post_seconds'] for r in rows),
             construction_cpu_seconds=sum(r['cpu_seconds'] for r in rows),
             construction_wall_sum=sum(r['wall_seconds'] for r in rows),
             source_timeouts=sum(r['status']=='timeout' for r in rows),
             timing='Cold source runs only; whole-child resources include setup and CEC. Warm controls excluded.')
        report=build(library_root/group,groups[group],provenance)
        reports[group]={k:v for k,v in report.items() if k!='records'}
        hashes[group]=snapshot(library_root/group)
        print('FROZEN',group,report['unique_queries'],'queries',flush=True)
    save(root/'library_build_cost.json',reports)
    save(root/'library_frozen_sha256.json',hashes)
    library_root.chmod(0o555)
    return hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--engine',type=Path,default=PROJECT/'build/epmapping/epmapping_check')
    parser.add_argument('--cpus',type=int,nargs='+',default=list(range(8,16)))
    args=parser.parse_args();assert len(set(args.cpus))==len(args.cpus)
    assert set(args.cpus)<=os.sched_getaffinity(0)
    root=args.root.resolve();m=initialize(root,args.engine.resolve(),args.cpus)
    base.common.call=measured_call;tasks=specs();assert len(tasks)==416
    start=time.monotonic()
    construction=execute(root,m,args.cpus,tasks,'construction')
    frozen=freeze_libraries(root,m,tasks)
    before_primary=time.monotonic()
    primary=execute(root,m,args.cpus,tasks,'primary')
    for name,digests in frozen.items():assert snapshot(root/'shared_libraries'/name)==digests,('library changed',name)
    records=list(root.glob('*/*/*/result.json'));assert len(records)==944,len(records)
    save(root/'execution_status.json',dict(complete=True,observations=944,construction=construction,
        primary=primary,elapsed_seconds=time.monotonic()-start,
        prebuild_and_controls_elapsed_seconds=before_primary-start,all_libraries_unchanged=True))
    print('COMPLETE: 944 observations; primary frozen libraries byte-unchanged',flush=True)


if __name__=='__main__':main()
