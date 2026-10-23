#!/usr/bin/env python3
"""Export the horizontally grouped mfs3 comparison from the verified campaign."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from statistics import mean


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def change(out, initial):
    return 100 * (float(out) / float(initial) - 1)


def formatted(value):
    return '0.00' if round(value, 2) == 0 else f'{value:+.2f}'


def export(root):
    verified = json.loads((root / 'verified_metrics.json').read_text())
    names = ['tab_mfs3/summary.tsv', 'tab_mfs3/abc.tsv',
             'tab_mfs3/abc_baseline_metrics.tsv']
    for name in names:
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == verified['raw_sha256'][name], f'Unverified source: {name}'
    rows = read_rows(root / names[0])
    ours = {r['design']: r for r in rows if r['stage'] == 'ours'}
    abc = {r['design']: r for r in read_rows(root / names[1])}
    starts = {r['design']: r for r in read_rows(root / names[2])}
    assert len(ours) == 16 and ours.keys() == abc.keys() == starts.keys()
    body, paired_abc, paired_own, all_own = [], [], [], []
    paired_times, all_own_times = [], []
    violations = 0
    for design, row in ours.items():
        assert row['cec'] == '1' and row['rc'] == '0'
        own = [change(row['out_a'], row['base_a']), change(row['out_d'], row['base_d'])]
        all_own.append(own)
        log_name = f'tab_mfs3/ours.{design}/log'
        log = root / log_name
        assert hashlib.sha256(log.read_bytes()).hexdigest() == verified['raw_sha256'][log_name]
        times = re.findall(r'epmapping timing .*?optimizer=([0-9.eE+-]+)', log.read_text())
        assert len(times) == 1, f'Missing optimizer runtime: {design}'
        own_time = float(times[0])
        assert 0 <= own_time <= 1000.01
        all_own_times.append(own_time)
        values = ['TO', 'TO']
        result, start = abc[design], starts[design]
        if result['rc'] == '0':
            assert result['cec'] == '1'
            raw = [change(result['area'], start['abc_area']),
                   change(result['delay'], start['abc_delay'])]
            paired_abc.append(raw)
            paired_own.append(own)
            paired_times.append([float(result['secs']), own_time])
            violation = float(result['delay']) > float(start['abc_delay']) + 1e-3
            violations += violation
            values = [formatted(raw[0]), formatted(raw[1]) + (r'$^{\dagger}$' if violation else '')]
        else:
            assert result['rc'] in ('124', '142'), f'Unexpected failure: {design}'
        escaped = design.replace('_', r'\_')
        runtime = '1000 (TO)' if result['rc'] != '0' else f"{float(result['secs']):.1f}"
        body.append(' & '.join([escaped, f"{float(row['base_a']):.2f}",
                                f"{float(row['base_d']):.2f}", *values,
                                '1000' if result['rc'] != '0' else runtime,
                                *map(formatted, own), f'{own_time:.1f}']) + r' \\')
    metrics = verified['metrics']
    assert len(paired_abc) == metrics['abc_completed'] == 13
    assert violations == metrics['abc_delay_violations'] == 7
    for group, keys in [(paired_abc, ['abc_raw_area', 'abc_raw_delay']),
                        (paired_own, ['paired_ours_area', 'paired_ours_delay']),
                        (all_own, ['mfs_ours_area', 'mfs_ours_delay'])]:
        for i, key in enumerate(keys):
            assert abs(mean(r[i] for r in group) - metrics[key]) < 1e-9, key
    body += [r'\midrule']
    paired_values = []
    for group, i in [(paired_abc, 0), (paired_own, 1)]:
        paired_values += [*[f'{1 + mean(r[j] for r in group) / 100:.4f}' for j in range(2)],
                          f'{mean(r[i] for r in paired_times):.1f}']
    body.append(' & '.join(['Ave. Ratio', '1.0000', '1.0000', *paired_values]) + r' \\')
    digest = hashlib.sha256((root / 'verified_metrics.json').read_bytes()).hexdigest()
    return '\n'.join([
        '% GENERATED from CEC-verified raw logs; do not edit numbers.',
        '% verified_metrics_sha256: ' + digest,
        r'\newcommand{\MfsComparisonTable}{%',
        r'\begin{table}[t]', r'\centering\footnotesize',
        r'\setlength{\tabcolsep}{2pt}',
        r"\caption{One-pass comparison on identical serialized inputs and the same six-input library. $A_0$ ($\mu m^2$), $D_0$ (ps): common initial mapping, measured before ABC import. Changes (\%) use each tool's native start; ABC normalizes output buffers on import (\Cref{sec:mfs3}). $t$ (s) excludes CEC: ABC times its command, ours times the optimizer. TO: 1000\,s cap; $\dagger$: delay violation. Ave. Ratio gives mean output/input area and delay ratios on the common 13 completed designs (initial = 1); its runtime entries are mean seconds.}\label{tab:mfs3}",
        r'\begin{tabular}{@{}lrrrrrrrr@{}}', r'\toprule',
        r'\multirow{2}{*}{Circuit} & \multicolumn{2}{c}{Initial} & \multicolumn{3}{c}{ABC mfs3} & \multicolumn{3}{c}{Ours} \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}',
        r' & $A_0$ & $D_0$ & $\delta A$ & $\delta D$ & $t$ & $\delta A$ & $\delta D$ & $t$ \\',
        r'\midrule', *body, r'\bottomrule', r'\end{tabular}', r'\end{table}', '}']) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(export(args.root))
