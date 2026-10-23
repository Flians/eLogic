#!/usr/bin/env python3
"""Aggregate the new version only; retain all old baseline failures and sources."""
import argparse,json,copy,hashlib,subprocess,sys,collections,fcntl
from pathlib import Path
R=Path(__file__).resolve().parents[3]
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('campaign',type=Path);a.add_argument('--repeats',nargs='*',type=Path,default=[]);a.add_argument('--output',type=Path,required=True);a=a.parse_args();a.output.mkdir(exist_ok=False)
 cp=a.campaign.resolve();full=load(cp.parent/'binding_validation.json');assert full['complete']
 allv=[full]+[load(p.parent/'binding_validation.json')for p in a.repeats];assert all(v['complete'] for v in allv)
 for p in [cp,*a.repeats]:
  c=load(p);assert len(c['jobs'])==c['expected_jobs'] and c['status']=='attempted'
  with (p.parent/'runner.lock').open('a')as h:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for f,h in c['fingerprint'].items():assert sha(f)==h,f
  for j in c['jobs']:
   for k in ['report','audit','admission']:
    assert sha(j[k])==j[k+'_sha256']
 old=load(R/'results_v2/single_output_components_evidence_20261017/integrity.json')
 for p,h in old['baseline_small_binary_check'].items():assert sha(p)==h,p
 required=['abc_map_area_resyn2','abc_map_delay_resyn2','mockturtle_emap_area_resyn2','mockturtle_emap_delay_resyn2','eqmap_lut_amap','esyn_frozen_model_amap_bounded_dag','emorphic_final_sa']
 G=a.output/'baseline_scope';G.mkdir();gates=[]
 for k in [3,4]:
  base=R/f'results_v2/guarded_joint_evidence_20261016/baseline_scope/baselines_{k}in.json';b=load(base)
  for profile in ['reuse']:
   d=copy.deepcopy(b);d['results']=[]
   for x in full['rows']:
    if x['k']!=k or x['profile']!=profile:continue
    report=load(x['report']);r=copy.deepcopy(report['results'][0]);assert report['library_sha256']==b['library_sha256'] and report['metric_model']==b['metric_model']
    assert all(r['input_sha256']==e['input_sha256'] for e in b['results'] if e['bench']==r['bench'])
    r['method']='egraph_frontier_V45_'+profile;d['results'].append(r)
   assert len(d['results'])==20;p=G/f'candidate_{k}in_{profile}.json';p.write_text(json.dumps(d,indent=2)+'\n')
   for objective in ['delay-area','area-delay','pareto']:
    q=G/f'{k}in_{profile}_{objective}.json';argv=[sys.executable,str(R/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(p),'--baselines',str(base),'--required-baselines',*required,'--objective',objective,'--output',str(q)]
    r=subprocess.run(argv,capture_output=True,text=True);q.with_suffix('.log').write_text(r.stdout+r.stderr);assert q.exists()
    gates.append(dict(k=k,profile=profile,objective=objective,passed=load(q)['all_passed'],argv=argv,baseline_sha256=sha(base)))
 q=dict(version='V45 necessary-generation-reference reuse0/1, one frozen binary',full_attempts=full['attempts'],full_CEC=full['cec'],full_summary=full['summary'],full_coverage=full['coverage'],repetition_attempts=sum(v['attempts']for v in allv[1:]),repetition_CEC=sum(v['cec']for v in allv[1:]),repetitions=[v['summary']for v in allv[1:]],gates=gates,baseline_algorithms_changed=False,baseline_small_binary_check=old['baseline_small_binary_check'],prior_full_13403_file_baseline_source_audit_referenced_not_rerun=True,complete_visited_seed_protection=True,historical_full_budget_final_cover_protection=False,fixed_delay_shared_area_recovery_extension=True,unconnected=['MtM'],AD_gates_posthoc_not_AD_optimization=True,no_automatic_PPA_superiority_claim=True,source_plus_cuts_full_suite_validated=False)

 q['version']='V45 deadline-aware exchange setup, initial/unchanged-pool witness reuse, counts-only optional pool diagnostics'
 q['promoted_default']=False
 q['independent_ablation_of_all_three_innovations']=False
 q['older_same_binary_flag_study']='results_v2/streamed_provenance_20261019/evidence/qualification.json; initial transfer and unchanged-pool reuse are bundled; all36 repetitions including10 failures retained'
 q['repetition_status_records']=[dict(k=x['k'],bench=x['bench'],profile=x['profile'],status=x['status'],area=x['area'],delay=x['delay'],error=x['error'],native_failure=x['native_failure'],report=x['report'])for v in allv[1:]for x in v['rows']]
 old_full=load(R/'results_v2/binding_noop_recovery_20261019/evidence/qualification.json');q['reference_B_source']='f84b79fb25391e63557c97f636e567df506a2c6468ade8b276a77ed451ebce1a; frozen functional profile, cross-version contextual comparison, no winner mixing'
 context=[]
 old_rows=load('/tmp/emapper-binding-noop-full-20261019/binding_validation.json')['rows']
 from check_single_output_components import outcome
 for x in full['rows']:
  b=next(z for z in old_rows if(z['k'],z['bench'],z['profile'])==(x['k'],x['bench'],'functional_cuts'));p=dict(k=x['k'],bench=x['bench'],reference=b,candidate=x,valid=x['valid']and b['valid'])
  if p['valid']:p.update(area_outcome=outcome(x['area'],b['area']),delay_outcome=outcome(x['delay'],b['delay']),area_delta=x['area']-b['area'],delay_delta=x['delay']-b['delay'])
  context.append(p)
 q['functional_reference_context']={m:dict(collections.Counter(x[m+'_outcome']for x in context if x['valid']))for m in ['area','delay']}
 (a.output/'functional_reference_context.json').write_text(json.dumps(context,indent=2)+'\n')
 (a.output/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))
if __name__=='__main__':main()
