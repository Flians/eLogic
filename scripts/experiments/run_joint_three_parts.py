#!/usr/bin/env python3
"""Frozen buffer-free three-part campaign with per-worker CPU affinity."""
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
from run_isolated_mem_eth import measured_call

PROJECT = Path(__file__).resolve().parents[2]
CONTROL = PROJECT / 'results_epmapping_prebuffer_v3_20261111'
FLAGS = dict(EPMAPPING_SHARED_SUM_FACTOR='1', EPMAPPING_PHASE_TIMING='1',
             MAP_EGG_PROFILE_TIMING='1', MAP_EGG_PROFILE_MEMORY_ENTRIES='2048',
             MAP_EGG_PROFILE_MEMORY_BYTES='67108864', MAP_EGG_COVER_INDEXED='0')
ARMS = dict(base.ARMS, noJOINT={'EPMAPPING_SHARED_SUM_FACTOR': '0'},
            noLAZYVARIANTS={'MAP_EGG_COVER_INDEXED': '1'})


def initialize(root, binary, cpus):
    root.mkdir(parents=True, exist_ok=True)
    ready = root / 'joint_protocol.json'
    if ready.exists():
        protocol = json.loads(ready.read_text())
        assert protocol['cpus'] == cpus
        return base.common.initialize(root, binary)
    assert not (root / 'execution_status.json').exists()
    m = base.initialize(root, binary, len(cpus))
    previous = json.loads((CONTROL / 'manifest.json').read_text())
    assert {k:v for k,v in m['goal'].items() if k != 'EPMAPPING_SUPER'} == {
        k:v for k,v in previous['goal'].items() if k != 'EPMAPPING_SUPER'}
    for rel in previous['input_sha256']:
        if rel.endswith(('.aig', '.genlib')) or rel in ['abc', 'database.super']:
            assert base.common.sha(root/'inputs'/rel) == previous['input_sha256'][rel], rel
    # Reuse the exact already-audited normalization, not a new preprocessing run.
    for name in ['normalized', 'normalization_abc_blif']:
        (root/name).symlink_to(CONTROL/name, target_is_directory=True)
    shutil.copy2(CONTROL/'abc_import_proofs.json', root/'abc_import_proofs.json')
    m['goal'].update(FLAGS)
    m['three_part_plan'].update(arms=ARMS, expected_method_runs=832,
                                expected_abc_runs=64, expected_observations=896)
    m['worker_cpus'] = cpus
    m['host_exclusive'] = False
    m['control_campaign'] = str(CONTROL)
    m['normalization_reused_sha256'] = base.common.sha(CONTROL/'normalized/manifest.json')
    sources = root/'inputs/sources'
    for name in ['run_joint_three_parts.py', 'run_isolated_mem_eth.py',
                 'analyze_isolated_mem_eth.py']:
        shutil.copy2(PROJECT/'scripts/experiments'/name, sources/name)
    m['input_sha256'] = {str(p.relative_to(root/'inputs')):base.common.sha(p)
                        for p in (root/'inputs').rglob('*') if p.is_file()}
    base.common.save(root/'manifest.json', m)
    base.common.save(ready, dict(cpus=cpus, jobs=len(cpus), new_arms=ARMS,
                                flags=FLAGS, own_cpu_oversubscription=False,
                                cold='empty private persistent pool',
                                warm='new process using paired completed cold pool',
                                normalized_input_source=str(CONTROL)))
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--cpus', nargs='+', type=int, default=list(range(8,16)))
    a = parser.parse_args()
    assert len(set(a.cpus)) == len(a.cpus)
    assert set(a.cpus) <= os.sched_getaffinity(0)
    root = a.root.resolve()
    m = initialize(root, a.binary.resolve(), a.cpus)
    base.common.call = measured_call
    specs = [(stage,'ours',d,{}) for stage in ['main','mfs3','strong_common','main_r1']
             for d in base.DESIGNS]
    specs += [('ablation',arm,d,flags) for d in base.DESIGNS for arm,flags in ARMS.items()]
    work = queue.Queue()
    for spec in specs:
        work.put(spec)
    lock = threading.Lock()
    errors = []
    finished = []
    started = time.monotonic()

    def status():
        base.common.save(root/'execution_status.json',dict(
            complete=len(finished)==len(specs) and not errors,
            pairs_finished=len(finished), pairs_expected=len(specs), errors=errors,
            expected_observations=896, elapsed_seconds=time.monotonic()-started,
            worker_cpus=a.cpus, host_exclusive=False))

    def worker(cpu):
        os.sched_setaffinity(0,{cpu})  # Linux thread affinity, inherited by its children.
        while True:
            try:
                stage,arm,design,flags = work.get_nowait()
            except queue.Empty:
                return
            try:
                cache = root/'cache'/stage/arm/design
                for mode in ['cold','warm']:
                    base.native(root,m,stage,arm,design,mode,cache,flags)
                    if stage in ['mfs3','strong_common']:
                        cohort = 'direct6' if stage=='mfs3' else 'strong6'
                        base.abc(root,m,stage,design,mode,root/'normalized'/cohort/(design+'.blif'))
                with lock:
                    finished.append((stage,arm,design))
                    status()
            except Exception as error:
                traceback.print_exc()
                with lock:
                    errors.append(dict(spec=[stage,arm,design],error=repr(error),cpu=cpu))
                    status()
            finally:
                work.task_done()

    status()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.cpus)) as pool:
        list(pool.map(worker,a.cpus))
    assert not errors, errors
    print('COMPLETE: 896 observations, original shared buffer-free inputs',flush=True)


if __name__ == '__main__':
    main()
