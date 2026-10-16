#!/usr/bin/env python3
"""Verify and summarize paired whole-network lexicographic objective runs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

def key(mode, area, delay):
    a, d = math.floor(area * 1e4 + .5), math.floor(delay * 1e3 + .5)
    return (d, a) if mode == 'delay_area' else (a, d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / 'manifest.json').read_text())
    assert hashlib.sha256((root / manifest['archived_engine']).read_bytes()).hexdigest() == manifest['engine_sha256']
    with (root / 'objectives/summary.tsv').open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    index = {}
    failures = []
    for row in rows:
        mode, design = row['stage'], row['design']
        assert mode in ('delay_area', 'area_delay')
        assert (mode, design) not in index, (mode, design)
        folder = root / 'objectives' / f'{mode}.{design}'
        if row['rc'] != '0' or row['cec'] != '1':
            failures.append({'mode': mode, 'design': design, 'rc': row['rc'], 'cec': row['cec']})
            index[(mode, design)] = None
            continue
        values = [float(row[name]) for name in ('base_a', 'base_d', 'out_a', 'out_d')]
        assert all(math.isfinite(x) for x in values)
        log = (folder / 'log').read_text()
        assert re.search(r'^\[i\] cec=1$', log, re.M)
        assert f'objective_order={mode}, global_objective_guard=1, preserve_depth=0, preserve_delay=0,' in log
        assert 'window_synth windows=0 applied=0' in log
        final = re.findall(r'epmapping area ([0-9.eE+-]+) -> ([0-9.eE+-]+) delay ([0-9.eE+-]+) -> ([0-9.eE+-]+)', log)[-1]
        assert values == [float(final[i]) for i in (0, 2, 1, 3)]
        assert key(mode, values[2], values[3]) <= key(mode, values[0], values[1]), (mode, design)
        row['area_change'] = 100*(values[2]/values[0]-1)
        row['delay_change'] = 100*(values[3]/values[1]-1)
        row['rollbacks'] = int(re.search(r'objective_guard rollbacks=(\d+)', log)[1])
        index[(mode, design)] = row
    specs=(root/'.specs_objectives').read_text().splitlines()
    expected={(Path(line.split()[0]).name.rsplit('.',1)[0],line.split()[1]) for line in specs}
    assert set(index)==expected, 'objective batch is incomplete'
    designs = sorted({r['design'] for r in rows})
    complete = [d for d in designs if all(index.get((m, d)) for m in ('delay_area', 'area_delay'))]
    lines = ['# Paired lexicographic objectives', '',
             'Both modes use the same serialized area-oriented direct mapping and 40-cell ASAP7 library. '
             'Each has one round by default. Whole-window synthesis is off. Secondary regression is '
             'permitted on primary improvement; the measured whole-network pair is guarded per round.', '',
             '| Design | Initial A | Initial D | (delay, area) A | D | (area, delay) A | D |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for d in complete:
        da, ad = index[('delay_area', d)], index[('area_delay', d)]
        assert da['base_a'] == ad['base_a'] and da['base_d'] == ad['base_d']
        baseline = [root/'objectives'/f'{m}.{d}'/'epmapping_baseline.v' for m in ('delay_area', 'area_delay')]
        assert baseline[0].read_bytes() == baseline[1].read_bytes(), f'different initial netlists: {d}'
        lines.append(f"| {d} | {da['base_a']} | {da['base_d']} | {da['out_a']} | {da['out_d']} | {ad['out_a']} | {ad['out_d']} |")
    means = {}
    lines += ['', f'Complete CEC-verified pairs: {len(complete)}/{len(designs)}.', '',
              '| Objective | Mean area change (%) | Mean delay change (%) | Rolled-back rounds |',
              '|---|---:|---:|---:|']
    for mode in ('delay_area', 'area_delay'):
        records = [index[(mode, d)] for d in complete]
        if not records:
            continue
        means[mode] = {'area_change': statistics.mean(r['area_change'] for r in records),
                       'delay_change': statistics.mean(r['delay_change'] for r in records),
                       'rollbacks': sum(r['rollbacks'] for r in records)}
        m = means[mode]
        lines.append(f"| {mode} | {m['area_change']:+.2f} | {m['delay_change']:+.2f} | {m['rollbacks']} |")
    lines += ['', 'Means use the identical complete design subset and average per-design percentages. '
              'Area is cell area; delay is worst output arrival. Runtime is observational. '
              'Lexicographic acceptance does not imply globally optimal search.', '',
              'Failed/timeout configurations: ' + json.dumps(failures), '']
    (root / 'objectives.md').write_text('\n'.join(lines))
    artifacts = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (root / 'objectives').rglob('*') if p.is_file()}
    result = {'complete_designs': complete, 'means': means, 'failures': failures, 'sha256': artifacts}
    (root / 'verified_objectives.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'complete_pairs': len(complete), 'means': means, 'failures': failures}, indent=2))

if __name__ == '__main__':
    main()
