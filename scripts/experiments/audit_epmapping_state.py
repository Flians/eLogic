#!/usr/bin/env python3
"""Read-only audit of the frozen post-mapping campaign, sources and paper numbers.

This reuses existing CEC evidence and independently remeasures serialized PPA.
It does not rerun optimization or CEC, regenerate tables, or alter frozen records.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

from report_three_parts import audit

PROJECT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review(root, project):
    delivery = json.loads((root / 'delivery_manifest.json').read_text())
    for name, expected in delivery['result_sha256'].items():
        assert sha(root / name) == expected, f'changed delivery record: {name}'
    paper = project / 'epmapping_paper/DATE27-ePMapping-sub'
    assert sha(paper / 'DATE27-ePMapping.pdf') == delivery['pdf_sha256']
    profiles = json.loads((root / 'verified_profiles.json').read_text())
    generated = (paper / 'doc/generated_results.tex').read_text()
    records = {}
    for mode in ['cold', 'warm']:
        folder = root / mode
        manifest, data, abc = audit(folder)
        certificate = json.loads((folder / 'verified_metrics.json').read_text())
        assert sha(folder / 'verified_metrics.json') == profiles['cohort_certificate_sha256'][mode]
        assert certificate['metrics'] == profiles['cohort_metrics'][mode]
        for name, expected in certificate['raw_sha256'].items():
            assert sha(folder / name) == expected, f'changed certified record: {mode}/{name}'
        method = [(stage, row) for stage, rows in data.items() for row in rows
                  if row['stage'] != 'base']
        main = [row for row in data['tab_main'] if row['stage'] == 'ae']
        area = statistics.mean(100 * (float(x['out_a']) / float(x['base_a']) - 1) for x in main)
        delay = statistics.mean(100 * (float(x['out_d']) / float(x['base_d']) - 1) for x in main)
        assert abs(area - certificate['metrics']['main_own_area']) < 1e-9
        assert abs(delay - certificate['metrics']['main_own_delay']) < 1e-9
        if mode == 'cold':
            for name, value in [('OwnArea', area), ('OwnDelay', delay)]:
                pattern = r'\\newcommand\{\\' + name + r'\}\{([^}]+)\}'
                assert re.search(pattern, generated)[1] == f'{value:+.2f}'
        primary = [r for _, r in method if r['stage'] in ('ae', 'ae.r1', 'ours', 'full')]
        statuses = Counter(row['raw_status'] for _, row in method)
        records[mode] = {
            'method_attempts': len(method), 'statuses': dict(statuses),
            'primary_attempts': len(primary),
            'primary_completed': sum(x['raw_status'] == 'complete' for x in primary),
            'main_area_percent': area, 'main_delay_percent': delay,
            'main_area_wins': sum(float(x['out_a']) < float(x['base_a']) for x in main),
            'abc_completed': sum(x['rc'] == 0 for x in abc),
            'abc_timeouts': sum(x['rc'] == 124 for x in abc),
        }
        print(f'{mode}: {len(method)} method outputs verified', flush=True)
    frozen = root / 'cold/inputs'
    sources = {}
    for path in sorted((project / 'epmapping').glob('*.hpp')) + [project / 'epmapping/main.cpp']:
        sources[str(path.relative_to(project))] = {
            'sha256': sha(path), 'matches_paper_source': path.read_bytes() == (frozen / path.name).read_bytes()}
    assert all(x['matches_paper_source'] for x in sources.values()), 'algorithm source drift from the paper'
    return {
        'campaign': root.name, 'optimization_rerun': False, 'cec_rerun': False,
        'verification': 'frozen input/source hashes, existing CEC proofs, independent physical PPA, configuration and paper macros',
        'profiles': records, 'sources': sources,
        'paper_pdf_sha256': delivery['pdf_sha256'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT / 'results_v28_three_parts_query_budget')
    parser.add_argument('--out', type=Path, required=True, help='New audit result outside the frozen campaign')
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    assert not out.is_relative_to(root), 'never write new audit output into a frozen campaign'
    result = review(root, PROJECT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['profiles'], indent=2))


if __name__ == '__main__':
    main()
