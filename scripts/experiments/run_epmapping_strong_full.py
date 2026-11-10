#!/usr/bin/env python3
"""Normal six-round cap on the SAME strong, <=6-input shared mapping.

Retains the original one-round control. Reuses the frozen binary/inputs and
ABC observations from that campaign, never reruns or overwrites its searches.
"""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import traceback
from pathlib import Path

from run_epmapping_supplement import engine, sha, save


def initialize(root, reference):
    original = json.loads((reference / 'manifest.json').read_text())
    for name, digest in original['input_sha256'].items():
        assert sha(reference / 'inputs' / name) == digest, name
    path = root / 'manifest.json'
    if path.exists():
        m = json.loads(path.read_text())
        assert m['reference_manifest_sha256'] == sha(reference / 'manifest.json')
        for name, digest in m['input_sha256'].items():
            assert sha(root / 'inputs' / name) == digest, name
        return m
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(reference / 'inputs', root / 'inputs')
    m = dict(original, schema=2, common_rounds=6, expected_outputs=32,
             scope='normal six-round cap; same strong 40-entry shared inputs; retains one-round control',
             reference=str(reference), reference_manifest_sha256=sha(reference / 'manifest.json'))
    m['goal'] = dict(m['goal'], EPMAPPING_SUPER=str(root / 'inputs/database.super'))
    sources = root / 'inputs/supplement_controls'
    sources.mkdir()
    for name in ['run_epmapping_strong_full.py', 'run_epmapping_supplement.py']:
        shutil.copy2(Path(__file__).parent / name, sources / name)
    m['input_sha256'] = {str(p.relative_to(root / 'inputs')): sha(p) for p in (root / 'inputs').rglob('*') if p.is_file()}
    save(path, m)
    return m


def pair(root, reference, m, design):
    cache = root / 'cache/common/ours' / design
    for mode in ['cold', 'warm']:
        r = engine(root, m, 'common', 'ours', design, mode, cache, rounds=6)
        path = reference / 'common/ours' / (mode + '.' + design) / 'epmapping_baseline.v'
        if path.exists():
            assert r['initial_sha256'] == sha(path), ('changed common input', path)
    cold = root / 'common/ours' / ('cold.' + design) / 'epmapping_baseline.v'
    warm = root / 'common/ours' / ('warm.' + design) / 'epmapping_baseline.v'
    assert sha(cold) == sha(warm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=2)
    args = ap.parse_args()
    root, reference = args.root.resolve(), args.reference.resolve()
    m = initialize(root, reference)
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(pair, root, reference, m, d): d for d in m['designs']}
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                errors.append((futures[f], repr(e)))
                traceback.print_exc()
    save(root / 'execution_status.json', dict(jobs=len(m['designs']), errors=errors, complete=not errors))
    assert not errors, errors


if __name__ == '__main__':
    main()
