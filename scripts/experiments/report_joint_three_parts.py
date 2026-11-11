#!/usr/bin/env python3
"""Compare audited joint-factor/lazy-variant observations without pooling cohorts."""
import argparse
import json
from pathlib import Path

from run_epmapping_supplement import save


def report(root, control):
    current = json.loads((root/'analysis.json').read_text())
    previous = json.loads((control/'analysis.json').read_text())
    assert current['complete'] and current['observed']==current['expected']==896
    idx = {tuple(r[k] for k in ['stage','arm','design','mode']):r
           for r in current['records']}
    old = {tuple(r[k] for k in ['stage','arm','design','mode']):r
           for r in previous['records']}
    comparisons = []
    for key, row in idx.items():
        if key not in old:
            continue
        before = old[key]
        assert row['initial_sha256']==before['initial_sha256'], key
        if row['arm']!='ours' and not(row['stage']=='ablation' and row['arm']=='full'):
            continue
        comparisons.append(dict(stage=row['stage'],arm=row['arm'],design=row['design'],
            mode=row['mode'],old_status=before['status'],new_status=row['status'],
            old_area=before['final']['area'],area=row['final']['area'],
            old_delay=before['final']['delay'],delay=row['final']['delay'],
            area_delta=row['final']['area']-before['final']['area'],
            delay_delta=row['final']['delay']-before['final']['delay'],
            old_seconds=before['optimizer_seconds'],seconds=row['optimizer_seconds']))
    assert len(comparisons)==160
    variants = []
    for design in current['records']:
        if design['stage']!='ablation' or design['arm'] not in ['noJOINT','noLAZYVARIANTS']:
            continue
        full=idx['ablation','full',design['design'],design['mode']]
        variants.append(dict(arm=design['arm'],design=design['design'],mode=design['mode'],
            full_status=full['status'],variant_status=design['status'],
            area_delta=design['final']['area']-full['final']['area'],
            delay_delta=design['final']['delay']-full['final']['delay'],
            full_seconds=full['optimizer_seconds'],variant_seconds=design['optimizer_seconds'],
            full_profile=full.get('profile_metrics'),variant_profile=design.get('profile_metrics'),
            full_resources=full.get('resource'),variant_resources=design.get('resource')))
    assert len(variants)==64
    transactions=[]
    for r in current['records']:
        cost=r.get('shared_sum_cost')
        if not cost:
            continue
        if cost['accepted']:
            assert cost['proposed_area']<cost['before_area']
            assert cost['proposed_delay']<=cost['before_delay']+.001
        transactions.append(dict(stage=r['stage'],arm=r['arm'],design=r['design'],mode=r['mode'],
                                 cost=cost,stats=r.get('shared_sum_metrics')))
    result=dict(complete=True,observed=current['observed'],original_inputs_identical=True,
        initial_area_and_delay_bounds_verified=True,host_exclusive=False,
        previous_comparison= comparisons,new_variant_comparisons=variants,
        full_configuration_complete=sum(r['new_status']=='complete' for r in comparisons),
        full_configuration_runs=160,
        full_area_worse=[r for r in comparisons if r['area_delta']>.0001],
        full_delay_worse=[r for r in comparisons if r['delay_delta']>.001],
        shared_sum_transactions=transactions,
        timing_scope='Single controlled-concurrency observations; post-only wall seconds. Resource CPU/RSS cover setup and CEC too. Ratios involving timeout runs do not measure equal work.')
    save(root/'joint_comparison.json',result)
    lines=['Joint factor / lazy library full campaign',
           '896 observations: 832 method, 64 ABC; all inputs and preprocessing unchanged.',
           f"Full-configuration completion: {result['full_configuration_complete']}/160",
           f"Compared with previous frozen normalized campaign: area worse {len(result['full_area_worse'])}; delay worse {len(result['full_delay_worse'])}",
           '', 'New full-library ablations (sum of variant-minus-full areas; same-mode total runtime ratio):']
    for arm in ['noJOINT','noLAZYVARIANTS']:
        for mode in ['cold','warm']:
            v=current['ablations'][arm+'/'+mode]
            lines.append(f"{arm}/{mode}: area delta={v['delta_area']:.6f}, time/full={v['time_ratio']:.6f}, TO={v['timeouts']}")
    lines+=['','Main changes compared with prior normalized observations:']
    for r in comparisons:
        if abs(r['area_delta'])>.0001 or abs(r['delay_delta'])>.001:
            lines.append(f"{r['stage']}/{r['arm']}/{r['mode']}.{r['design']}: delta area {r['area_delta']:+.6f}, delta delay {r['delay_delta']:+.6f}")
    lines+=['',result['timing_scope']]
    (root/'joint_report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--control',type=Path,default=Path('results_epmapping_prebuffer_v3_20261111'))
    a=p.parse_args();report(a.root.resolve(),a.control.resolve())
