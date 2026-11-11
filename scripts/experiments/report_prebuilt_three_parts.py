#!/usr/bin/env python3
"""Report primary prebuilt PPA separately from construction and cache controls."""
import argparse,gzip,hashlib,json,statistics,tarfile
from pathlib import Path
from exact_core_graph import exact_core_fingerprint


def archived_core_evidence(root,directory=None):
    """Read the frozen audit when superseded raw netlists were pruned earlier."""
    evidence=directory or root/'build_context/historical_buffer_core'
    manifest=json.loads((evidence/'manifest.json').read_text())
    digest=lambda data:hashlib.sha256(data).hexdigest()
    for name in ['primary_core_replay.json','analysis.json.gz','source_snapshot.tar.gz']:
        assert digest((evidence/name).read_bytes())==manifest['files'][name],name
    with tarfile.open(evidence/'source_snapshot.tar.gz','r:gz') as archive:
        archived_helper=archive.extractfile('auditors/exact_core_graph.py').read()
    assert digest(archived_helper)==digest((Path(__file__).parent/'exact_core_graph.py').read_bytes())
    proof=json.loads((evidence/'primary_core_replay.json').read_text())
    assert proof['complete'] and proof['pairs']==32
    fingerprints={(r['design'],r['mode']):r for r in proof['records']}
    frozen=json.loads(gzip.decompress((evidence/'analysis.json.gz').read_bytes()))
    oldidx={tuple(r[k] for k in ['stage','arm','design','mode']):r for r in frozen['records']}
    return fingerprints,oldidx,dict(source='frozen historical core audit; old raw netlists previously pruned',
        proof_sha256=manifest['files']['primary_core_replay.json'],
        frozen_analysis_sha256=manifest['files']['analysis.json.gz'],
        fingerprint_helper_sha256=digest(archived_helper))


def report(root,historical_evidence=None):
    x=json.loads((root/'analysis.json').read_text());assert x['complete'] and x['observed']==944
    idx={tuple(r[k] for k in ['stage','arm','design','mode']):r for r in x['records']}
    primary=[r for r in x['records'] if r['mode']=='prebuilt'];method=[r for r in primary if r['arm']!='abc']
    full=[r for r in method if r['arm'] in ['ours','full']]
    cold=[r for r in x['records'] if r['mode']=='cold'];warm=[r for r in x['records'] if r['mode']=='warm']
    full_flow=Path(__file__).resolve().parents[2]/'results_epmapping_three_parts_buffer_v1_20261111'
    older=json.loads((full_flow/'analysis.json').read_text())
    oldidx={tuple(r[k] for k in ['stage','arm','design','mode']):r for r in older['records']}
    archived_cores,frozen_old,core_provenance=archived_core_evidence(root,historical_evidence)
    flow_rows=[];core_rows=[]
    for r in full:
        old=oldidx[r['stage'],r['arm'],r['design'],'cold']
        frozen=frozen_old[r['stage'],r['arm'],r['design'],'cold']
        assert old['selected_sha256']==frozen['selected_sha256']
        for phase in ['initial','final']:
            for q in ['area','delay']:assert abs(old[phase][q]-frozen[phase][q])<.000001
        flow_rows.append(dict(stage=r['stage'],arm=r['arm'],design=r['design'],mode='prebuilt',
            old_initial=old['initial'],new_initial=r['initial'],old_final=old['final'],new_final=r['final'],
            area_delta=r['final']['area']-old['final']['area'],delay_delta=r['final']['delay']-old['final']['delay']))
        if r['stage']=='main':
            lib=root/'inputs/asap7.genlib'
            a=exact_core_fingerprint(root/r['selected_netlist'],lib)
            previous=archived_cores[r['design'],'cold']
            assert previous['current_netlist_sha256']==old['selected_sha256']
            b=previous['current']
            if (full_flow/old['selected_netlist']).exists():
                assert exact_core_fingerprint(full_flow/old['selected_netlist'],lib)==b
            core_rows.append(dict(design=r['design'],same_core=a==b,current_sha256=r['selected_sha256'],
                                  old_sha256=old['selected_sha256'],current_core=a,old_core=b))
    flow=dict(complete=True,protocol='current prebuilt, cleaned input denominator; earlier complete flow uses original buffered input',
              records=flow_rows,primary_core_replay=dict(pairs=16,identical=sum(r['same_core'] for r in core_rows),records=core_rows,
                  current='recomputed from current selected bytes',historical=core_provenance))
    (root/'full_flow_comparison.json').write_text(json.dumps(flow,indent=2)+'\n')
    banks=json.loads((root/'library_build_cost.json').read_text())
    control_cost=dict(cold_post_seconds=sum(r['optimizer_seconds'] for r in cold),
        warm_post_seconds=sum(r['optimizer_seconds'] for r in warm),
        cold_profile_seconds_observed=sum(r.get('profile_metrics',{}).get('profile_build_seconds',0) for r in cold),
        cold_profile_counter_runs=sum('profile_metrics' in r for r in cold),
        warm_profile_seconds_observed=sum(r.get('profile_metrics',{}).get('profile_build_seconds',0) for r in warm),
        warm_profile_counter_runs=sum('profile_metrics' in r for r in warm),
        cold_cpu_seconds=sum(r['resource']['cpu_seconds'] for r in cold),
        warm_cpu_seconds=sum(r['resource']['cpu_seconds'] for r in warm),
        merge_seconds=sum(b['merge_seconds'] for b in banks.values()),
        preparation=json.loads((root/'execution_status.json').read_text())['construction'],
        scope='Cold collection supplies banks; paired private warm controls are separate and never supply warm-only records. Whole-child CPU includes initialization/native CEC; post time excludes those phases.')
    receipt=json.loads((root/'cold_pool_receipts.json').read_text());assert receipt['complete']
    training={name:dict(runs=0,post_seconds=0.,cpu_seconds=0.) for name in banks}
    for row in cold:
        if row['stage'] in ['main','main_r1']:continue
        name=row['arm'] if row['stage']=='ablation' else 'full'
        training[name]['runs']+=1
        training[name]['post_seconds']+=row['optimizer_seconds']
        training[name]['cpu_seconds']+=row['resource']['cpu_seconds']
    assert training['full']['runs']==48 and all(v['runs']==16 for k,v in training.items() if k!='full')
    (root/'training_cost.json').write_text(json.dumps(dict(training=training,
        excluded_cold_controls=32,reason='full47 ablation bank trains only on its own sixteen cold cases; 40-entry source flows have a disjoint library namespace'),indent=2)+'\n')
    assert len(full)==80 and len(method)==416 and len(primary)==448
    out=dict(complete=True,observations=944,primary_mode='prebuilt',primary_method_runs=416,primary_abc_runs=32,
        full_completed=sum(r['status']=='complete' for r in full),full_runs=80,
        primary_method_timeouts=sum(r['status']=='timeout' for r in method),
        primary_abc_timeouts=sum(r['status']=='timeout' for r in primary if r['arm']=='abc'),
        cold_source_runs=len(cold),cold_source_timeouts=sum(r['status']=='timeout' for r in cold),
        private_warm_runs=len(warm),private_warm_timeouts=sum(r['status']=='timeout' for r in warm),
        construction_cost=control_cost,bank_costs=banks,effective_training_cost=training,
        warm_only_records_excluded=sum(r['warm_extra_records'] for r in receipt['records'].values()),
        all_libraries_unchanged=all(r['unchanged'] for r in x['library_checks']),
        full_previous_comparisons=x['full_previous_comparisons'],cache_comparisons=x['prebuilt_matches'],
        limitations=['Frozen banks collected from the evaluated workload; repeated-use evaluation, not held-out transfer.',
                    'Eight CPU-pinned workers on a shared host; no statistical speedup claim.',
                    'Timeouts remain incomplete and select legal checkpoints/inputs; all consumed budgets reported.'])
    (root/'prebuilt_report.json').write_text(json.dumps(out,indent=2)+'\n')
    lines=['Frozen prebuilt-bank full three-part study',f"944 observations; primary 416 method + 32 ABC; collection 416 cold + 80 private warm.",
           f"Full primary completion: {out['full_completed']}/80; method/ABC primary TOs: {out['primary_method_timeouts']}/{out['primary_abc_timeouts']}",
           f"Cold-source/private-warm TOs (separate controls): {out['cold_source_timeouts']}/{out['private_warm_timeouts']}",
           f"Cold collection post sum: {control_cost['cold_post_seconds']:.3f} s; private warm controls: {control_cost['warm_post_seconds']:.3f} s",
           f"Observed cold/warm profile extraction: {control_cost['cold_profile_seconds_observed']:.3f}/{control_cost['warm_profile_seconds_observed']:.3f} s; counters available for {control_cost['cold_profile_counter_runs']}/416 and {control_cost['warm_profile_counter_runs']}/80 runs (not a complete decomposition when TOs omit final counters)",
           f"Cold/warm whole-child CPU: {control_cost['cold_cpu_seconds']:.3f}/{control_cost['warm_cpu_seconds']:.3f} s",
           f"Preparation elapsed (cold plus private warm controls): {control_cost['preparation']['elapsed_seconds']:.3f} s; merge all banks: {control_cost['merge_seconds']:.3f} s",
           f"Warm-only records excluded from training: {out['warm_only_records_excluded']}",
           '','Primary main results (full 47-entry library, strong preprocessing):','design | area | delay | prebuilt seconds | status']
    designs=json.loads((root/'manifest.json').read_text())['designs']
    for design in designs:
        r=idx['main','ours',design,'prebuilt'];lines.append(f"{design} | {r['final']['area']:.2f} | {r['final']['delay']:.2f} | {r['optimizer_seconds']:.3f} | {r['status']}")
    lines+=['','Primary per-stage aggregates:',json.dumps(x['aggregates'],indent=2),
            '','All source runs, selected checkpoints, original inputs, private warm pools, cold-only snapshots and immutable banks are retained locally.']
    (root/'prebuilt_report.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['construction_cost','bank_costs','effective_training_cost','full_previous_comparisons','cache_comparisons']},indent=2))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--historical-core-evidence',type=Path,
                   help='Earlier buffer paper data directory; defaults to retained build_context evidence')
    a=p.parse_args();report(a.root.resolve(),a.historical_core_evidence.resolve() if a.historical_core_evidence else None)
