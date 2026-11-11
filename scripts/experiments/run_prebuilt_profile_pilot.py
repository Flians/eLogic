#!/usr/bin/env python3
"""Build once, freeze, and audit read-only candidate reuse on three designs.

Nine unchanged inputs cover direct6, strong6 and strong47. Every evaluation is a
fresh process with an empty private delta. Neither evaluation order modifies the
shared library. Cold construction is reported separately, never hidden in warm.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import shutil
import time

import run_prebuffer_three_parts as campaign
from run_isolated_mem_eth import measured_call
from build_readonly_profile_library import build, snapshot, sha
from run_joint_three_parts import FLAGS

PROJECT = Path(__file__).resolve().parents[2]
CONTROL = PROJECT / 'results_epmapping_joint_prebuffer_v1_20261111'
DESIGNS = ['mem_ctrl', 'ethernet', 'usb_phy']
STAGES = ['mfs3', 'strong_common', 'main']


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def metrics(result, root):
    wd = root / result['selected_netlist']
    log = (wd.parent/'log').read_text()
    line = re.findall(r'egraph_profile (.*)', log)[-1]
    values = {k:float(v) if any(c in v for c in '.eE') else int(v)
              for k,v in re.findall(r'(\w+)=([0-9.eE+-]+)', line)}
    return dict(stage=result['stage'], design=result['design'], arm=result['arm'],
                status=result['status'], area=result['final']['area'],
                delay=result['final']['delay'], seconds=result['optimizer_seconds'],
                selected_sha256=result['selected_sha256'], initial_sha256=result['initial_sha256'],
                CEC=result['cec'], profile=values,
                resource=json.loads((wd.parent/'log.resources.json').read_text()))


def initialize(root, engine, cpus):
    assert not root.exists(), ('use a fresh result root', root)
    root.mkdir(parents=True)
    inputs = root/'inputs'; inputs.mkdir()
    for name in ['abc', 'benchmarks', 'asap7.genlib', 'asap7_6in.genlib', 'database.super']:
        (inputs/name).symlink_to((CONTROL/'inputs'/name).resolve(), target_is_directory=name=='benchmarks')
    shutil.copy2(engine, inputs/'engine')
    (root/'normalized').symlink_to((CONTROL/'normalized').resolve(), target_is_directory=True)
    sources = inputs/'sources'; sources.mkdir()
    for name in ['epmapping','map_egg/src']:
        source = PROJECT/name
        for file in source.rglob('*'):
            if file.is_file() and file.suffix in ['.hpp','.cpp','.rs','.json']:
                target=sources/name/file.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(file,target)
    for name in ['run_prebuilt_profile_pilot.py', 'build_readonly_profile_library.py',
                 'run_prebuffer_three_parts.py','run_isolated_mem_eth.py','run_joint_three_parts.py',
                 'audit_mapped_ppa.py','recover_checkpoint.py','run_buffer_core_validation.py','report.py']:
        shutil.copy2(PROJECT/'scripts/experiments'/name,sources/name)
    m=json.loads((CONTROL/'manifest.json').read_text())
    m['goal'].update(FLAGS);m['engine_sha256']=sha(inputs/'engine')
    manifest=dict(protocol='shared-readonly-profile-pilot-v1', control=str(CONTROL),
                  design_order=DESIGNS, stages=STAGES, cpus=cpus, jobs=len(cpus), host_exclusive=False,
                  engine_sha256=m['engine_sha256'], goal=m['goal'],
                  primary_mode='prebuilt', private_delta='empty at every evaluation start; never shared',
                  build_source='the same three pilot designs; repeated-workload, not unseen-design generalization',
                  preprocessing='reuse byte-identical frozen mapped inputs; bypass buffer before fixed D0',
                  input_sha256={str(p.relative_to(inputs)):sha(p) for p in inputs.rglob('*') if p.is_file()},
                  normalized_sha256={str(p.relative_to(root/'normalized')):sha(p) for p in (root/'normalized').rglob('*') if p.is_file()},
                  timeout=1000, whole_window_synthesis=False,
                  timing='native post-only; whole-child CPU/RSS includes setup and CEC; OS caches retained')
    save(root/'manifest.json',manifest)
    return m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--engine',type=Path,default=PROJECT/'build/epmapping/epmapping_check')
    parser.add_argument('--cpus',type=int,nargs=3,default=[8,9,10])
    args=parser.parse_args();root=args.root.resolve();engine=args.engine.resolve()
    assert len(set(args.cpus))==3 and set(args.cpus)<=os.sched_getaffinity(0)
    m=initialize(root,engine,args.cpus);campaign.common.call=measured_call
    old=json.loads((CONTROL/'analysis.json').read_text());previous={(r['stage'],r['design']):r for r in old['records'] if r['arm']=='ours' and r['mode']=='cold'}
    records=[];start=time.monotonic()
    library=root/'shared_library';library_hash=None

    def run_design(design, cpu, arm, stages):
        os.sched_setaffinity(0,{cpu});rows=[]
        for stage in stages:
            cache=root/'private'/arm/stage/design
            flags={} if arm=='build' else {'MAP_EGG_PROFILE_READONLY_DIR':str(library)}
            r=campaign.native(root,m,stage,arm,design,'cold',cache,flags)
            assert r['status']=='complete',(stage,design,arm,r['status'])
            target=previous[stage,design]
            assert r['initial_sha256']==target['initial_sha256']
            assert r['final']['area']<=target['final']['area']+.0001,('area regression',stage,design,arm)
            assert r['final']['delay']<=target['final']['delay']+.001,('delay regression',stage,design,arm)
            assert r['initial_buffer_area']==0 and r['final_buffer_area']==0
            row=metrics(r,root);row['quality_pass']=True;row['previous_area']=target['final']['area'];row['previous_delay']=target['final']['delay']
            rows.append(row)
        return rows

    def phase(arm,reverse=False):
        nonlocal records
        pairs=list(zip(DESIGNS,args.cpus))
        if reverse:pairs.reverse()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(run_design,d,c,arm,list(reversed(STAGES)) if reverse else STAGES) for d,c in pairs]
            for future in concurrent.futures.as_completed(futures):
                records.extend(future.result());save(root/'summary.json',records)
        save(root/'progress.json',dict(phase=arm,observed=len(records),elapsed_seconds=time.monotonic()-start))
        if library_hash is not None:
            assert snapshot(library)==library_hash,('shared library changed',arm)
        print('PHASE COMPLETE',arm,flush=True)

    phase('build')
    build_rows=list(records)
    provenance=dict(engine_sha256=m['engine_sha256'],manifest_sha256=sha(root/'manifest.json'),
                    source_cases=[dict(stage=r['stage'],design=r['design']) for r in build_rows],
                    construction_post_seconds=sum(r['seconds'] for r in build_rows),
                    construction_cpu_seconds=sum(r['resource']['cpu_seconds'] for r in build_rows),
                    construction_profile_seconds=sum(r['profile']['profile_build_seconds'] for r in build_rows),
                    construction_elapsed_seconds=time.monotonic()-start)
    report=build(library,{r['stage']+'/'+r['design']:root/'private/build'/r['stage']/r['design'] for r in build_rows},provenance)
    save(root/'library_build_cost.json',{k:v for k,v in report.items() if k!='records'})
    library_hash=snapshot(library);save(root/'library_frozen_sha256.json',library_hash)
    phase('prebuilt')
    phase('prebuilt_reverse',reverse=True)
    idx={(r['arm'],r['stage'],r['design']):r for r in records};comparisons=[]
    for stage in STAGES:
        for design in DESIGNS:
            cold=idx['build',stage,design];warm=idx['prebuilt',stage,design];reverse=idx['prebuilt_reverse',stage,design]
            assert cold['selected_sha256']==warm['selected_sha256']==reverse['selected_sha256'],('changed netlist',stage,design)
            assert warm['profile']['builds']==reverse['profile']['builds']==0,('unexpected cache miss',stage,design)
            comparisons.append(dict(stage=stage,design=design,area=warm['area'],delay=warm['delay'],
                cold_seconds=cold['seconds'],prebuilt_seconds=warm['seconds'],reverse_seconds=reverse['seconds'],
                readonly_hits=warm['profile']['readonly_hits'],order_independent=True,byte_identical=True))
    save(root/'validation.json',dict(complete=True,observations=len(records),quality_pass=True,completion_pass=True,
        readonly_library_unchanged=snapshot(library)==library_hash,reverse_order_netlists_identical=True,
        primary_mode='prebuilt',previous_cold_campaign_preserved=str(CONTROL),comparisons=comparisons,
        library_build_cost={k:v for k,v in report.items() if k!='records'},
        timing='three CPU-pinned workers; shared host; single observations, not statistical speedups'))
    print(json.dumps(dict(complete=True,observations=len(records),quality_pass=True,readonly_library_unchanged=True)),flush=True)


if __name__=='__main__':
    main()
