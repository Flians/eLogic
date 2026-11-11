#!/usr/bin/env python3
"""Canonical eMapper main: the single, globally selected main-off configuration.

No preprocessing, external seed or per-design policy selection is performed.
The historical configurable runner remains scripts/emapper/run.py.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def command(aig, golden, output):
    profile = json.loads((ROOT/'emapper/profiles/main-off.json').read_text())
    assert profile['name'] == 'main-off' and profile['mapping_seconds'] == 600
    assert profile['objective'] == 'delay-area' and not profile['external_seed']
    library = ROOT/profile['library']
    assert hashlib.sha256(library.read_bytes()).hexdigest() == profile['library_sha256']
    return ['taskset', '-c', str(profile['cpu']), sys.executable,
            str(ROOT/'scripts/emapper/run.py'), str(aig.resolve()),
            '--golden', str(golden.resolve()), '--output', str(output.resolve()),
            *profile['fixed_arguments']]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('aig', type=pathlib.Path, help='prepared named AIG; no resyn2 is rerun')
    ap.add_argument('--golden', type=pathlib.Path, help='original named AIG for independent CEC')
    ap.add_argument('--output', type=pathlib.Path, required=True, help='new output directory')
    ap.add_argument('--print-command', action='store_true', help='show exact argv without mapping')
    a = ap.parse_args()
    argv = command(a.aig, a.golden or a.aig, a.output)
    if a.print_command:
        print(json.dumps(argv, indent=2));return
    env = {k:v for k,v in os.environ.items() if not k.startswith('EGM_')}
    case = a.output.resolve()
    env['EMAPPER_AFFINITY_RECEIPT'] = str(case/'native-affinity.json')
    env['EMAPPER_AFFINITY_NONCE'] = hashlib.sha256(str(case).encode()).hexdigest()
    raise SystemExit(subprocess.call(argv, cwd=ROOT, env=env))


if __name__ == '__main__':
    main()
