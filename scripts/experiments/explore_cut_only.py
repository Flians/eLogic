#!/usr/bin/env python3
"""Frozen, resumable cut-only search exploration with a declared tuning/validation split."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

PROJECT = Path(__file__).resolve().parents[2]
TRAIN = ['aes_core', 'iwls05_i2c', 'spi', 'systemcdes', 'tv80', 'usb_phy']
ALL = 'aes_core arbiter div ethernet iwls05_i2c mem_ctrl RISC sasc simple_spi spi sqrt systemcaes systemcdes tv80 usb_funct usb_phy'.split()
COMMON = dict(EPMAPPING_OBJECTIVE='area_delay', EPMAPPING_WINDOW_SYN='0',
              EPMAPPING_GLOBAL_OBJECTIVE_GUARD='1', EPMAPPING_PRESERVE_DEPTH='1',
              EPMAPPING_PRESERVE_DELAY='1', EPMAPPING_NO_DELAY_REGRESSION='1',
              EPMAPPING_EGG_REQ='1', EPMAPPING_WRITE_BASELINE='1')
PROFILES = {
    'bounded': dict(EPMAPPING_OBJECTIVE='area_delay_bound', EPMAPPING_AREA_UNDER_DELAY='1',
                    EPMAPPING_PATH_OBJECTIVE='0', EPMAPPING_RANK_AREA_GAIN='1',
                    EPMAPPING_PRESERVE_DEPTH='0', EPMAPPING_PRESERVE_DELAY='0',
                    EPMAPPING_NO_DELAY_REGRESSION='0', EPMAPPING_USE_REQUIRED_TIMES='1',
                    EPMAPPING_MAX_DELAY_GROWTH='0', EPMAPPING_MAX_AREA_GROWTH='0',
                    EPMAPPING_LOCAL_AREA_GROWTH='0'),
    'control': {},
    'gain': dict(EPMAPPING_RANK_AREA_GAIN='1'),
    'nodepth': dict(EPMAPPING_RANK_AREA_GAIN='1', EPMAPPING_PRESERVE_DEPTH='0'),
    'required': dict(EPMAPPING_RANK_AREA_GAIN='1', EPMAPPING_PRESERVE_DEPTH='0',
                     EPMAPPING_USE_REQUIRED_TIMES='1', EPMAPPING_MAX_DELAY_GROWTH='0'),
    'pareto_delay': dict(EPMAPPING_OBJECTIVE='delay_area', EPMAPPING_RANK_AREA_GAIN='1',
                         EPMAPPING_PRESERVE_DEPTH='0', EPMAPPING_NO_DELAY_REGRESSION='0',
                         EPMAPPING_USE_REQUIRED_TIMES='1', EPMAPPING_MAX_DELAY_GROWTH='0',
                         EPMAPPING_LOCAL_AREA_GROWTH='0', EPMAPPING_MAX_AREA_GROWTH='0'),
    'path': dict(EPMAPPING_OBJECTIVE='delay_area', EPMAPPING_PATH_OBJECTIVE='1',
                 EPMAPPING_RANK_AREA_GAIN='1', EPMAPPING_PRESERVE_DEPTH='0',
                 EPMAPPING_PRESERVE_DELAY='0', EPMAPPING_NO_DELAY_REGRESSION='0',
                 EPMAPPING_USE_REQUIRED_TIMES='1', EPMAPPING_MAX_DELAY_GROWTH='0',
                 EPMAPPING_LOCAL_AREA_GROWTH='-1', EPMAPPING_MAX_AREA_GROWTH='-1'),
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--phase', choices=['tuning', 'validation', 'rounds', 'all', 'converged'], default='tuning')
    ap.add_argument('--profiles', default='bounded')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=1500)
    ap.add_argument('--rounds', type=int, default=1)
    args = ap.parse_args()
    root = args.root.resolve()
    names = args.profiles.split(',')
    assert names and set(names) <= PROFILES.keys()
    designs = ALL if args.phase in ('all','converged') else TRAIN if args.phase in ('tuning', 'rounds') else [d for d in ALL if d not in TRAIN]
    root.mkdir(parents=True, exist_ok=True)
    frozen = root/'inputs'
    manifest_path = root/'manifest.json'
    if not manifest_path.exists():
        frozen.mkdir()
        for relative in ['build/epmapping/epmapping_check', 'epmapping/main.cpp',
                         'epmapping/epmapping.hpp', 'epmapping/objective.hpp',
                         'scripts/experiments/common.sh', 'scripts/experiments/explore_cut_only.py']:
            shutil.copy2(PROJECT/relative, frozen/Path(relative).name)
        shutil.copy2(PROJECT/'results_v5_objectives/objectives/asap7_6in.genlib', frozen/'library.genlib')
        paths = [PROJECT/'tools/mockturtle/experiments/benchmarks'/f'{d}.aig' for d in ALL]
        paths += [PROJECT/'epmapping/db/asap7_4in_3lvl_a050.super']
        manifest = dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),
                        whole_window_synthesis=False, tuning_designs=TRAIN,
                        validation_designs=[d for d in ALL if d not in TRAIN],
                        profiles={n: dict(COMMON, **p) for n, p in PROFILES.items()},
                        archived_engine='inputs/epmapping_check',
                        input_sha256={str(p.relative_to(PROJECT)): sha(p) for p in paths},
                        frozen_sha256={p.name: sha(p) for p in frozen.iterdir()})
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    manifest = json.loads(manifest_path.read_text())
    assert all(sha(frozen/n) == h for n,h in manifest['frozen_sha256'].items())
    assert all(sha(PROJECT/n) == h for n,h in manifest['input_sha256'].items())
    phase = root/args.phase
    phase.mkdir(exist_ok=True)
    plan = dict(profiles=names, designs=designs, rounds=args.rounds, timeout=args.timeout, jobs=args.jobs)
    plan_path = phase/'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, 'use a new phase/root for a different plan'
    else:
        plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    def run(name, design):
        wd = phase/f'{name}.{design}'
        result = wd/'row.tsv'
        if result.exists():
            return result.read_text()
        profile = manifest['profiles'][name]
        assert profile['EPMAPPING_WINDOW_SYN'] == '0'
        environment = {k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_', 'EPM_'))}
        environment.update(EPM_BIN=str(frozen/'epmapping_check'), RAYON_NUM_THREADS='1',
                           EPM_TIMEOUT=str(args.timeout), EPM_RETRIES='1', EPM_CAFFEINATE='0')
        extras = [f'{k}={v}' for k,v in profile.items()] + [f'EPM_GENLIB={frozen / "library.genlib"}']
        command = ['bash', '-c', 'source "$1"; shift; run_engine "$@"', '_',
                   str(frozen/'common.sh'), str(wd), design, str(args.rounds), 'direct', *extras]
        # common.sh derives PROJECT_ROOT from its location; use the original helper and verify its hash.
        command[4] = str(PROJECT/'scripts/experiments/common.sh')
        assert sha(Path(command[4])) == manifest['frozen_sha256']['common.sh']
        completed = subprocess.run(command, env=environment, cwd=PROJECT, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr)
        row = completed.stdout.strip().splitlines()[-1]+'\n'
        assert len(row.strip().split('\t')) == 14, completed.stdout
        result.write_text(row)
        print(f'{args.phase} {name}.{design}: '+row.strip(), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        tasks = [pool.submit(run,n,d) for n in names for d in designs]
        for task in as_completed(tasks):
            task.result()
    header = 'stage design recipe rounds base_a base_d out_a out_d dA% dD% rnd cec secs rc'.split()
    with (phase/'summary.tsv').open('w') as stream:
        stream.write('\t'.join(header)+'\n')
        for n in names:
            for d in designs:
                stream.write((phase/f'{n}.{d}'/'row.tsv').read_text())

if __name__ == '__main__':
    main()
