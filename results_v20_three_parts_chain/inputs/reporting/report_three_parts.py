#!/usr/bin/env python3
"""Audit the complete three-part campaign; export only verified current results."""
import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path
import export_paper
import report
from audit_mapped_ppa import audit as physical_audit
from run_three_parts import DESIGNS

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def pct(out,base): return 100*(float(out)/float(base)-1)
def mean(xs): return statistics.mean(xs)
def esc(s): return s.replace('_',r'\_')
def write_tsv(path,rows,keys):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys,delimiter='\t'); w.writeheader(); w.writerows(rows)

def audit(root):
    manifest=json.loads((root/'manifest.json').read_text())
    ABLATION=manifest['ablation_designs']; ARMS=manifest['arms']
    staged=root/'ablation_execution_manifest.json'
    if staged.exists():
        recorded=json.loads(staged.read_text())
        for key in ['goal','arms','designs','ablation_designs','engine_sha256','post_mapping_timeout']:
            assert recorded[key]==manifest[key], f'staged ablation configuration differs: {key}'
        for rel,h in recorded['frozen_sha256'].items(): assert sha(root/rel)==h, f'staged frozen source differs: {rel}'
    assert manifest['whole_window_synthesis'] is False
    assert manifest['goal']['EPMAPPING_OBJECTIVE']=='area_delay_bound'
    assert manifest['post_mapping_timeout']==1000 and manifest['abc_timeout']==1000
    for rel,h in manifest['frozen_sha256'].items(): assert sha(root/rel)==h, rel
    data={stage:export_paper.rows(root,stage) for stage in ['tab_main','tab_mfs3','ablation']}
    expected={'tab_main':{(a,d) for a in ['ae.r1','ae'] for d in DESIGNS},
              'tab_mfs3':{(a,d) for a in ['base','ours'] for d in DESIGNS},
              'ablation':{(a,d) for a in ARMS for d in ABLATION}}
    for stage,rr in data.items():
        assert {(r['stage'],r['design']) for r in rr}==expected[stage], f'incomplete {stage}'
        for r in rr:
            wd=root/stage/f'{r["stage"]}.{r["design"]}'; text=(wd/'log').read_text()
            if r['stage']=='base':
                assert r['rc']=='0' and '[i] baseline_cec=1' in text
                continue
            assert 'area_under_delay=1,' in text and 'global_objective_guard=1,' in text
            assert '[i] post_mapping_timeout=1000 seconds' in text
            assert report.read_counters(str(wd/'log'))['time_optimizer']<=1000.01, f'post-mapping budget exceeded: {wd}'
            assert 'max_area_growth=0,' in text and 'max_delay_growth=0,' in text
            assert 'path_objective=0,' in text and 'preserve_depth=0,' in text
            cap=1 if stage=='tab_mfs3' or r['stage']=='ae.r1' else 6
            assert int(r['rounds'])==cap and f'max_rounds={cap},' in text
            assert float(r['out_d'])<=float(r['base_d'])+.001, f'delay violation {wd}'
            assert float(r['out_a'])<=float(r['base_a'])+.001, f'area violation {wd}'
            if manifest.get('original_root'):
                lib=root/'inputs'/('asap7_6in.genlib' if stage=='tab_mfs3' else 'asap7.genlib')
                physical=[]
                for filename,ak,dk in [('epmapping_baseline.v','base_a','base_d'),('epmapping_mapped.v','out_a','out_d')]:
                    measured=physical_audit(wd/filename,lib); physical.append(measured)
                    assert abs(measured['area']-float(r[ak]))<.011 and abs(measured['delay']-float(r[dk]))<.011, f'independent physical audit: {wd}/{filename}'
            if manifest.get('original_root'):
                assert physical[1]['serialized_area']<=physical[0]['serialized_area']+.001 and physical[1]['serialized_delay']<=physical[0]['serialized_delay']+.001, f'serialized PPA regression: {wd}'
                original=Path(manifest['original_root']); old_stage='tab_mfs3' if stage=='tab_mfs3' else 'tab_main'
                old_arm='ours' if stage=='tab_mfs3' else 'ae'
                cohort='direct' if stage=='tab_mfs3' else 'high_effort'
                original_hash=manifest.get('original_mapped_sha256',{}).get(f'{cohort}/{r["design"]}')
                if original_hash: assert sha(wd/'epmapping_baseline.v')==original_hash, f'original input hash changed: {wd}'
                oldfile=original/old_stage/f'{old_arm}.{r["design"]}'/'epmapping_baseline.v'
                if oldfile.exists(): assert (wd/'epmapping_baseline.v').read_bytes()==oldfile.read_bytes(), f'changed original input: {wd}'
                else: assert original_hash, f'original reference unavailable: {wd}'
                chain=0 if r['stage']=='noCHAIN' else 1; share=0 if r['stage']=='noSHARE' else 1
                assert f', chain_factor={chain},' in text and f', chain_share={share},' in text
                assert ', chain_block=2,' in text and ', feasible_covers=1,' in text
                for key in ['dependency_cuts','global_divisors','observability_care']:
                    assert f', {key}=0,' in text, f'unplanned experimental operator: {wd}/{key}'
            flags=(wd/'experiment_env.txt').read_text().splitlines()
            assert [s for s in flags if s.startswith('EPMAPPING_OBJECTIVE=')][-1]=='EPMAPPING_OBJECTIVE=area_delay_bound'
            if stage=='tab_main' or stage=='ablation':
                assert 'ABC baseline: high_effort' in text
                assert 'fraig; resyn2rs; fraig; compress2rs' in text
                assert 'ands (no reduction); fixpoint after' in text, f'preprocessing did not reach its stopping criterion: {wd}'
                assert 'stopped at the iteration cap' not in text, f'preprocessing safety cap: {wd}'
            else: assert 'ABC baseline: direct' in text
    main={(r['stage'],r['design']):r for r in data['tab_main']}
    abl={(r['stage'],r['design']):r for r in data['ablation']}
    for d in DESIGNS:
        assert (root/'tab_main'/f'ae.{d}'/'epmapping_baseline.v').read_bytes()==(root/'tab_main'/f'ae.r1.{d}'/'epmapping_baseline.v').read_bytes()
    for d in ABLATION:
        for arm in ARMS:
            assert (root/'ablation'/f'{arm}.{d}'/'epmapping_baseline.v').read_bytes()==(root/'tab_main'/f'ae.{d}'/'epmapping_baseline.v').read_bytes(), f'changed ablation start {arm}.{d}'
        for key in ['out_a','out_d']:
            assert abl['full',d][key]==main['ae',d][key], f'full ablation does not reproduce main QoR: {d}/{key}'
    abc=json.loads((root/'tab_mfs3/abc_results.json').read_text())
    assert len(abc)==16 and {r['design'] for r in abc}==set(DESIGNS)
    for r in abc:
        d=r['design']; wd=root/'tab_mfs3'/f'abc.{d}'
        assert json.loads((wd/'result.json').read_text())==r
        if manifest.get('abc_source'):
            provenance=json.loads((wd/'reuse.json').read_text())
            assert provenance['source_archive']==manifest['abc_source']
            for name,h in provenance['copied_sha256'].items(): assert sha(wd/name)==h, f'changed reused ABC evidence: {d}/{name}'
        b=root/'tab_mfs3'/f'base.{d}'/'epmapping_mapped.v'
        assert sha(b)==r['baseline_sha256']
        assert b.read_bytes()==(root/'tab_mfs3'/f'ours.{d}'/'epmapping_baseline.v').read_bytes()
        for k in ['area','delay']:
            assert float(re.findall(r'\b'+k+r'\s*=\s*([0-9.]+)',(wd/'baseline.log').read_text())[-1])==r['baseline'][k]
        if r['rc']==0:
            assert r['cec']==1 and 'Networks are equivalent' in (wd/'cec.log').read_text(), f'ABC CEC {d}'
            for k in ['area','delay']:
                assert float(re.findall(r'\b'+k+r'\s*=\s*([0-9.]+)',(wd/'log').read_text())[-1])==r[k]
        else: assert r['rc']==124, f'ABC error, not timeout: {d}'
    return manifest,data,abc

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path); a=ap.parse_args(); root=a.root.resolve()
    manifest,data,abc=audit(root)
    ABLATION=manifest["ablation_designs"]; ARMS=manifest["arms"]
    idx={stage:{(r['stage'],r['design']):r for r in rr} for stage,rr in data.items()}
    own=lambda rr,k:mean(pct(r[f'out_{k}'],r[f'base_{k}']) for r in rr)
    mainrows=[idx['tab_main']['ae',d] for d in DESIGNS]
    one=[idx['tab_main']['ae.r1',d] for d in DESIGNS]
    ours={d:idx['tab_mfs3']['ours',d] for d in DESIGNS}
    complete=[r for r in abc if r['rc']==0 and r['cec']==1]
    feasible=[r for r in complete if r['delay']<=r['baseline']['delay']+.001 and r['area']<=r['baseline']['area']+.001]
    paper=report.read_paper_table(str(root/'inputs/compare_with_aspdac25.sh'))
    ref={d:(paper[d][0]*(1+paper[d][2]/100),paper[d][3]*(1+paper[d][5]/100)) for d in DESIGNS}
    times=[report.read_counters(str(root/'ablation'/f'full.{d}'/'log')) for d in ABLATION]
    keys=['secs_stage1','secs_stage2','secs_synth','secs_egg','secs_care','secs_window','secs_chain']
    totals={k:sum(t[k] for t in times) for k in keys}
    assert totals['secs_window']==0
    metrics=dict(whole_window_synthesis=False,engine_cec_outputs=sum(len(v) for v in data.values())-16,baseline_cec_outputs=16,
                 main_own_area=own(mainrows,'a'),main_own_delay=own(mainrows,'d'),
                 main_one_area=own(one,'a'),main_one_delay=own(one,'d'),
                 main_other_area=own([r for r in mainrows if r['design']!='arbiter'],'a'),
                 main_other_delay=own([r for r in mainrows if r['design']!='arbiter'],'d'),
                 main_area_wins=sum(float(r['out_a'])<float(r['base_a']) for r in mainrows),
                 reference_area=mean(pct(r['out_a'],ref[r['design']][0]) for r in mainrows),
                 reference_delay=mean(pct(r['out_d'],ref[r['design']][1]) for r in mainrows),
                 reference_wins=sum(float(r['out_a'])<ref[r['design']][0] for r in mainrows),
                 mfs_ours_area=own(list(ours.values()),'a'),mfs_ours_delay=own(list(ours.values()),'d'),
                 abc_completed=len(complete),abc_timeouts=16-len(complete),abc_feasible=len(feasible),
                 abc_delay_violations=sum(r['delay']>r['baseline']['delay']+.001 for r in complete),
                 cut_oracle_times=totals)
    for k,metric in [('a','area'),('d','delay')]:
        metrics['abc_raw_'+metric]=mean(pct(r[metric],r['baseline'][metric]) for r in complete)
        metrics['abc_guarded_'+metric]=mean(pct(r[metric],r['baseline'][metric]) if r in feasible else 0 for r in complete)
        metrics['paired_ours_'+metric]=own([ours[r['design']] for r in complete],k)
    metrics['paired_area_wins']=sum(float(ours[r['design']]['out_a'])<r['area'] for r in complete)
    metrics['guarded_paired_area_wins']=sum(float(ours[r['design']]['out_a'])<(r['area'] if r in feasible else r['baseline']['area']) for r in complete)
    selection=[]
    for r in complete:
        d=r['design']; keep=r in feasible
        source=root/'tab_mfs3'/(f'abc.{d}/mapped.v' if keep else f'base.{d}/epmapping_mapped.v')
        target=root/'tab_mfs3'/f'abc.{d}'/'guarded.v'
        target.write_bytes(source.read_bytes())
        selection.append(dict(design=d,retained_raw_output=keep,area=r['area'] if keep else r['baseline']['area'],
                              delay=r['delay'] if keep else r['baseline']['delay'],selected_netlist=str(target.relative_to(root)),
                              selected_sha256=sha(target),source_netlist=str(source.relative_to(root))))
    (root/'tab_mfs3/guarded_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    delta={}; ablbody=[]
    fulltime=sum(report.read_counters(str(root/'ablation'/f'full.{d}'/'log'))['time_optimizer'] for d in ABLATION)
    for arm in ARMS:
        rr=[idx['ablation'][arm,d] for d in ABLATION]
        deltas=[float(r['out_a'])-float(idx['ablation']['full',r['design']]['out_a']) for r in rr]
        opttime=sum(report.read_counters(str(root/'ablation'/f'{arm}.{d}'/'log'))['time_optimizer'] for d in ABLATION)
        delta[arm]=dict(area=sum(deltas),worse=sum(v>.001 for v in deltas),better=sum(v<-.001 for v in deltas),
                       own_area=own(rr,'a'),own_delay=own(rr,'d'),optimizer_seconds=opttime,time_ratio=opttime/fulltime)
        v=delta[arm]; ablbody.append([esc(arm),f'{v["own_area"]:+.2f}',f'{v["own_delay"]:+.2f}',f'{v["area"]:+.2f}',v['worse'],v['better'],f'{v["time_ratio"]:.2f}'])
    metrics['ablation']=delta
    metrics['chain_activity']={d:{k:report.read_counters(str(root/'tab_main'/f'ae.{d}'/'log'))[k] for k in ['chain_wins','secs_chain']} for d in DESIGNS}
    artifacts={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.relative_to(root).parts and 'paper_snapshot' not in p.relative_to(root).parts and
               (p.name in ['log','baseline.log','cec.log','summary.tsv','row.tsv','result.json','reuse.json','abc_results.json','guarded_selection.json','experiment_env.txt','manifest.json','execution_plan.json','ablation_execution_manifest.json','ablation_initial_manifest.json'] or p.suffix=='.v' or 'inputs' in p.relative_to(root).parts)}
    # Adapter for the existing publication plotting code; native starts remain explicit.
    write_tsv(root/'tab_mfs3/abc.tsv',[dict(design=r['design'],area=r.get('area','-'),delay=r.get('delay','-'),secs=r['secs'],rc=r['rc'],cec=r['cec']) for r in abc],['design','area','delay','secs','rc','cec'])
    write_tsv(root/'tab_mfs3/abc_baseline_metrics.tsv',[dict(design=r['design'],abc_area=r['baseline']['area'],abc_delay=r['baseline']['delay']) for r in abc],['design','abc_area','abc_delay'])
    for name in ['tab_mfs3/abc.tsv','tab_mfs3/abc_baseline_metrics.tsv']:
        artifacts[name]=sha(root/name)
    (root/'verified_metrics.json').write_text(json.dumps(dict(metrics=metrics,raw_sha256=artifacts),indent=2)+'\n')
    macros={
        'MainWins':metrics['reference_wins'],'MainAreaAll':f'{metrics["reference_area"]:+.2f}', 'MainDelayAll':f'{metrics["reference_delay"]:+.2f}',
        'OwnArea':f'{metrics["main_own_area"]:+.2f}','OwnDelay':f'{metrics["main_own_delay"]:+.2f}', 'OwnAreaWins':metrics['main_area_wins'],
        'OwnOneArea':f'{metrics["main_one_area"]:+.2f}','OwnOneDelay':f'{metrics["main_one_delay"]:+.2f}',
        'ABCCompleted':len(complete),'ABCTimeouts':16-len(complete),'ABCFeasible':len(feasible),'ABCViolations':metrics['abc_delay_violations'],
        'ABCArea':f'{metrics["abc_raw_area"]:+.2f}','ABCDelay':f'{metrics["abc_raw_delay"]:+.2f}',
        'ABCGuardArea':f'{metrics["abc_guarded_area"]:+.2f}','ABCGuardDelay':f'{metrics["abc_guarded_delay"]:+.2f}',
        'MfsOwnArea':f'{metrics["mfs_ours_area"]:+.2f}','MfsOwnDelay':f'{metrics["mfs_ours_delay"]:+.2f}',
        'PairedOwnArea':f'{metrics["paired_ours_area"]:+.2f}','PairedOwnDelay':f'{metrics["paired_ours_delay"]:+.2f}',
        'ChainAreaWins':sum(v['chain_wins']>0 for v in metrics['chain_activity'].values()),
        'OwnOtherArea':f'{metrics["main_other_area"]:+.2f}', 'OwnOtherDelay':f'{metrics["main_other_delay"]:+.2f}',
        'TimeSynShare':f'{100*totals["secs_synth"]/sum(totals.values()):.1f}', 'TimeEggShare':f'{100*totals["secs_egg"]/sum(totals.values()):.1f}',
        'DeltaSyn':f'{delta["noSYN"]["area"]:+.2f}','DeltaEgg':f'{delta["noEGG"]["area"]:+.2f}',
        'DeltaDB':f'{delta["noDB"]["area"]:+.2f}','DeltaCare':f'{delta["noCARE"]["area"]:+.2f}',
        'DeltaChain':f'{delta.get("noCHAIN",{}).get("area",0):+.2f}',
        'DeltaShare':f'{delta.get("noSHARE",{}).get("area",0):+.2f}'}
    tablebody=[]
    for r in mainrows:
        d=r['design']; t=report.read_counters(str(root/'tab_main'/f'ae.{d}'/'log'))['time_optimizer']
        tablebody.append([esc(d),f'{ref[d][0]:.2f}',f'{ref[d][1]:.2f}',*[f'{float(r[k]):.2f}' for k in ['base_a','base_d','out_a','out_d']],f'{t:.1f}'])
    ratios=[mean(float(r[key])/ref[r['design']][col] for r in mainrows)
            for key,col in [('base_a',0),('base_d',1),('out_a',0),('out_d',1)]]
    tablebody.append(['Ave. Ratio','1.0000','1.0000',*[f'{v:.4f}' for v in ratios],'--'])
    tables={
        'MainResultTable':export_paper.latex_table('Fully preprocessed inputs and six-round-cap post-mapping results. The ASPDAC\'25 columns reproduce reported final results; mapped starts differ. Ave. Ratio averages per-design ratios to the corresponding ASPDAC area or delay over all 16 designs. $t_{opt}$ excludes preprocessing, mapping and CEC. Area: $\\mu m^2$; delay: ps.', 'tab:main',['Circuit','ASPDAC area','ASPDAC delay','$A_e$','$D_e$','Ours area','Ours delay','$t_{opt}$ (s)'],tablebody,'lrrrrrrr'),
        'AblationResultTable':export_paper.latex_table(f'Ablations on {len(ABLATION)} identical $A_e$ starts, with the same delay bound and six-round cap. $\\Delta A$ sums signed differences from full; W/B count worse/better areas. Time is total optimizer time divided by full.', 'tab:ablation',['Arm','$\\delta A$ (\\%)','$\\delta D$ (\\%)','$\\Delta A$','W','B','Time/full'],ablbody,'lrrrrrr')}
    text='% GENERATED from verified three-part campaign; do not edit numbers.\n'
    text+='\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k,v in macros.items())+'\n'
    for k,v in tables.items(): text+=f'\\newcommand{{\\{k}}}{{%\n{v}\n}}\n'
    (root/'generated_results.tex').write_text(text)
    if a.out: a.out.write_text(text)
    lines=['# Three-part constrained-area experiments','', 'Every ePMapping output satisfies its own initial area/delay bounds and passes CEC.',
           'ABC raw area-mode outputs are preserved. The guarded comparison reverts an infeasible completed output to its native initial mapping; it is an external output-selection policy, not an internal mfs3 timing constraint. Timeouts are excluded from paired means.', '',json.dumps(metrics,indent=2),'', '## Per-design fully preprocessed results','',report.md_table(['Design','A0','D0','A','D','dA %','dD %'],[[r['design'],r['base_a'],r['base_d'],r['out_a'],r['out_d'],f'{pct(r["out_a"],r["base_a"]):+.2f}',f'{pct(r["out_d"],r["base_d"]):+.2f}'] for r in mainrows])]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
