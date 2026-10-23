#!/usr/bin/env python3
"""Independent finite-pilot evidence; never fill missing new-version designs with old winners."""
import collections,copy,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];O=R/'results_v2/phase_source_evidence_20261019'
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def outcome(a,b):
 t=1e-6*max(1,abs(a),abs(b));return 'better' if a<b-t else 'worse' if a>b+t else 'equal'
def main():
 O.mkdir(exist_ok=False);records={};campaigns={};source_rows=[]
 for label in ['ranked_source','terminal_source','phase_screen']:
  cp=R/f'results_v2/{label}_pilot_20261019/campaign.json';c=load(cp);assert c['status']=='attempted' and len(c['jobs'])==c['expected_jobs'];campaigns[label]=c
  with (cp.parent/'runner.lock').open('r') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for p,h in c['fingerprint'].items():assert sha(p)==h,p
  for j in c['jobs']:
   for key in ['report','audit','pool','candidates','trace','source_proof','joint_source']:
    if key in j:assert sha(j[key])==j[key+'_sha256']
   r=load(j['report'])['results'][0];records[j['report']]=r
   if label!='phase_screen':
    d=load(j['joint_source']) if 'joint_source'in j else None
    source_rows.append(dict(study=label,k=j['k'],bench=j['bench'],profile=j['profile'],status=r['status'],source=d,report=j['report']))
 (O/'source_plans.json').write_text(json.dumps(source_rows,indent=2)+'\n')
 phase=load(R/'results_v2/phase_screen_pilot_20261019/phase_screen_validation.json');pairs=[]
 for k in [3,4]:
  for bench in campaigns['phase_screen']['designs']:
   rr={j['profile']:load(j['report'])['results'][0] for j in campaigns['phase_screen']['jobs'] if (j['k'],j['bench'])==(k,bench)}
   for ref in ['legacy_control','functional_cuts']:
    a,b=rr['phase_screen'],rr[ref];row=dict(k=k,bench=bench,reference=ref,valid=a['status']==b['status']=='ok')
    if row['valid']:row.update(candidate=a['measured'],control=b['measured'],**{m:outcome(a['measured'][m],b['measured'][m]) for m in ['area','delay']})
    pairs.append(row)
 gates=[];required=['abc_map_area_resyn2','abc_map_delay_resyn2','mockturtle_emap_area_resyn2','mockturtle_emap_delay_resyn2','eqmap_lut_amap','esyn_frozen_model_amap_bounded_dag','emorphic_final_sa']
 for label,profile in [('phase_screen','phase_screen'),('terminal_source','terminal_groups')]:
  for k in [3,4]:
   base=R/f'results_v2/guarded_joint_evidence_20261016/baseline_scope/baselines_{k}in.json';b=load(base);d=copy.deepcopy(b);d['results']=[]
   found={j['bench']:load(j['report'])['results'][0] for j in campaigns[label]['jobs'] if j['k']==k and j['profile']==profile}
   for bench in sorted({x['bench'] for x in b['results']}):
    if bench in found:r=copy.deepcopy(found[bench])
    else:r=dict(bench=bench,status='missing',input_sha256=next(x['input_sha256'] for x in b['results'] if x['bench']==bench),error='New-version fixed pilot only; not qualified for full20/repetition/promotion; no historical candidate filled in.')
    r['method']='egraph_'+label;d['results'].append(r)
   assert len(d['results'])==20;p=O/f'{label}_{k}in.json';p.write_text(json.dumps(d,indent=2)+'\n')
   for objective in ['delay-area','area-delay','pareto']:
    q=O/f'{label}_{k}in_{objective}.json';argv=[sys.executable,str(R/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(p),'--baselines',str(base),'--required-baselines',*required,'--objective',objective,'--output',str(q)]
    r=subprocess.run(argv,capture_output=True,text=True);q.with_suffix('.log').write_text(r.stdout+r.stderr);assert q.exists();gates.append(dict(study=label,k=k,objective=objective,passed=load(q)['all_passed'],argv=argv,baseline_sha256=sha(base)))
 old=load(R/'results_v2/single_output_components_evidence_20261017/integrity.json');assert sha(R/'target/release/bench_rs_egraph_map')==old['main_sha256']
 for p,h in old['baseline_small_binary_check'].items():assert sha(p)==h,p
 locks=[]
 for p in set((R/'results_v2').glob('**/runner.lock')):
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 q=dict(attempts=len(records),CEC=sum(r['status']=='ok' for r in records.values()),failures=sum(r['status']!='ok' for r in records.values()),phase_pairs=pairs,gates=gates,main_promoted=False,main_sha256=old['main_sha256'],baseline_algorithms_changed=False,baseline_small_binary_check=old['baseline_small_binary_check'],prior_full_13403_file_baseline_audit_referenced_not_rerun=True,baseline_original_counts=dict(groups=240,slots=520,CEC=372,reported_failures=124,unmeasured=24),idle_runner_locks=len(locks),unconnected=['MtM'],full20_new_versions_complete=False,new_versions_repeated=False,AD_posthoc_only=True,old_V31_full80_and36_targeted_repeats_not_overwritten=True,source_versions_not_combined_with_phase_screen=True,complete_cross_pool_old_cover_protection_implemented=False,source_count_certificate_not_import_PPA_certificate=True,raw_generation_RAM_not_retained_cap=True,post_admission_recovery_2s_not_whole_mapping=True)
 (O/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');print({k:q[k] for k in ['attempts','CEC','failures','main_promoted','idle_runner_locks']})
if __name__=='__main__':main()
