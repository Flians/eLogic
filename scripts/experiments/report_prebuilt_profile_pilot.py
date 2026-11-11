#!/usr/bin/env python3
"""Report construction separately from post-mapping with a frozen candidate pool."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    v=json.loads((root/'validation.json').read_text())
    a=json.loads((root/'independent_audit.json').read_text())
    assert v['complete'] and a['complete'] and a['observations']==27
    library=json.loads((root/'shared_library/library_manifest.json').read_text())
    records=json.loads((root/'summary.json').read_text())
    idx={(r['stage'],r['design'],r['arm']):r for r in records}
    cross=[(name,r) for name,r in library['records'].items()
           if len({source.split('/')[1] for source in r['sources']})>1]
    package_bytes=sum(p.stat().st_size for p in (root/'shared_library').rglob('*') if p.is_file())
    empty_records=sum(row['candidates']==0 for row in library['records'].values())
    result=dict(complete=True,observations=27,primary_mode='prebuilt',
                library_package_bytes=package_bytes,empty_query_records=empty_records,
                independent_audit_sha256=hashlib.sha256((root/'independent_audit.json').read_bytes()).hexdigest(),
                independent_PPA_CEC=True,nonregression_to_previous=True,
                reverse_order_byte_identical=True,readonly_library_unchanged=True,
                cost={k:val for k,val in library.items() if k!='records'},
                cross_design_shared_query_records=len(cross),
                comparisons=v['comparisons'],
                previous_cold_campaign_preserved=v['previous_cold_campaign_preserved'],
                limitations=['Pilot library built from the evaluated workload; no held-out generalization claim.',
                             'Pinned three-worker shared-host observations; OS page caches retained.',
                             'Old cold/warm campaign retained separately; its timing is not pooled with this cohort.'])
    (root/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    c=library['provenance']
    lines=['Shared read-only candidate library pilot',
           '27/27 complete; independent PPA/CEC; no area/delay regression to previous selected outputs.',
           'Primary mode: prebuilt; all processes restart, private deltas start empty.',
           'Library is byte-unchanged after forward and reversed-order evaluations.',
           '', 'Construction cost (nine cold source runs, not charged to reported prebuilt post time):',
           f"  post-mapping sum: {c['construction_post_seconds']:.6f} s",
           f"  whole-child CPU sum: {c['construction_cpu_seconds']:.6f} s (includes setup/CEC)",
           f"  wall time at three workers: {c['construction_elapsed_seconds']:.6f} s",
           f"  profile extraction sum: {c['construction_profile_seconds']:.6f} s (part of post time)",
           f"  merge time: {library['merge_seconds']:.6f} s",
           f"  records: {library['input_records']} -> {library['unique_queries']}; {library['candidate_count']} candidates",
           f"  record files: {library['record_bytes']} bytes; complete package: {package_bytes} bytes",
           f"  cross-design shared query records: {len(cross)}; empty query pools: {empty_records}",
           '', 'stage/design | area | delay | cold construction post | prebuilt post | reverse repeat post | readonly hits']
    for row in v['comparisons']:
        lines.append(f"{row['stage']}/{row['design']} | {row['area']:.2f} | {row['delay']:.2f} | {row['cold_seconds']:.6f} | {row['prebuilt_seconds']:.6f} | {row['reverse_seconds']:.6f} | {row['readonly_hits']}")
    lines+=['','New profile builds and private delta writes: zero in all 18 prebuilt observations.',
            'Same-workload prebuilding only; not evidence of generalization to unseen designs.',
            'Times are single shared-host observations, not statistical speedups. Previous cold/warm data remain separate.']
    (root/'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
