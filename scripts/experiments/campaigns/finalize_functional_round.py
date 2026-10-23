#!/usr/bin/env python3
"""Post-run aggregation; original-baseline data preserved and all failures included."""
from pathlib import Path
import json,copy,hashlib,subprocess,sys,collections,fcntl
R=Path(__file__).resolve().parents[3];O=R/'results_v2/functional_cuts_evidence_20261019'
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert load(O/'finite_batches_finished.json')['all_finite_batches_ended']
 full=load(R/'results_v2/functional_cuts_full_20261019/functional_validation.json');rep=[load(R/f'results_v2/functional_cuts_repeat_r{i}_20261019/functional_validation.json') for i in range(3)];source=load(O/'source_validation.json');records={}
 for name in ['functional_cuts_full_20261019']+[f'functional_cuts_repeat_r{i}_20261019' for i in range(3)]+['source_closure_pilot_20261019']:
  cp=R/'results_v2'/name/'campaign.json';c=load(cp);assert c['status']=='attempted' and len(c['jobs'])==c['expected_jobs']
  for p,h in c['fingerprint'].items():assert sha(p)==h,p
  with (cp.parent/'runner.lock').open('r') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for j in c['jobs']:records[j['report']]=load(j['report'])['results'][0]
 old=load(R/'results_v2/single_output_components_evidence_20261017/integrity.json')
 for p,h in old['baseline_small_binary_check'].items():assert sha(p)==h,p
 assert sha(R/'target/release/bench_rs_egraph_map')==old['main_sha256']
 G=O/'baseline_scope';G.mkdir(exist_ok=False);gates=[];required=['abc_map_area_resyn2','abc_map_delay_resyn2','mockturtle_emap_area_resyn2','mockturtle_emap_delay_resyn2','eqmap_lut_amap','esyn_frozen_model_amap_bounded_dag','emorphic_final_sa'];sqrt=[]
 for k in [3,4]:
  base=R/f'results_v2/guarded_joint_evidence_20261016/baseline_scope/baselines_{k}in.json';b=load(base);d=copy.deepcopy(b);d['results']=[]
  for row in full['rows']:
   if row['k']==k and row['profile']=='functional_cuts':
    r=copy.deepcopy(load(row['report'])['results'][0]);r['method']='egraph_functional_cuts_V31';d['results'].append(r)
  assert len(d['results'])==20;p=G/f'candidate_{k}in.json';p.write_text(json.dumps(d,indent=2)+'\n')
  for objective in ['delay-area','area-delay','pareto']:
   q=G/f'{k}in_{objective}.json';argv=[sys.executable,str(R/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(p),'--baselines',str(base),'--required-baselines',*required,'--objective',objective,'--output',str(q)];r=subprocess.run(argv,capture_output=True,text=True);q.with_suffix('.log').write_text(r.stdout+r.stderr);assert q.exists();gates.append(dict(k=k,objective=objective,passed=load(q)['all_passed'],argv=argv,baseline_sha256=sha(base)))
  a=next(r for r in d['results'] if r['bench']=='sqrt');e=next(r for r in b['results'] if r['bench']=='sqrt' and r['method']=='mockturtle_emap_delay_resyn2');sqrt.append(dict(k=k,candidate=a['measured'],emap_delay=e['measured'],area_relative=a['measured']['area']/e['measured']['area']-1,delay_relative=a['measured']['delay']/e['measured']['delay']-1,scope='fixed author emap delay mode, not per-design baseline-mode selection'))
 counts=collections.Counter()
 for v in rep:
  for metric,outcomes in v['summary']['functional_cuts'].items():
   for outcome,n in outcomes.items():counts[metric+'_'+outcome]+=n
 q=dict(unique_physical_attempts=len(records),unique_CEC=sum(r['status']=='ok' for r in records.values()),full_attempts=full['attempts'],full_CEC=full['cec'],full_summary=full['summary'],full_coverage=full['coverage'],repetition_attempts=sum(v['attempts'] for v in rep),repetition_CEC=sum(v['cec'] for v in rep),repetition_summary=dict(counts),repetitions=[v['summary'] for v in rep],source_attempts=source['attempts'],source_CEC=source['CEC'],source_failures=source['failures'],sqrt_vs_fixed_emap_delay=sqrt,gates=gates,main_promoted=False,main_sha256=old['main_sha256'],baseline_algorithms_changed=False,baseline_small_binary_check=old['baseline_small_binary_check'],prior_full_13403_file_baseline_audit_referenced_not_rerun=True,unconnected=['MtM'],AD_posthoc_only=True,actual_mapping_versions_not_mixed=True,source_V32_separate_not_combined_with_V31=True,source_counts_not_lower_bounds_or_actual_complete_imports=True,retained_pool_cap_not_raw_generation_RAM_cap=True,post_admission_recovery_2s_not_whole_mapping=True)
 (O/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q[k] for k in ['unique_physical_attempts','unique_CEC','full_summary','repetition_summary','source_CEC','main_promoted']},indent=2))
if __name__=='__main__':main()
