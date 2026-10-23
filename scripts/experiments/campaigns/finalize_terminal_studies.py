#!/usr/bin/env python3
"""Audit complete uniform terminal study, independent failed cache pilot and repeats."""
import json,subprocess,sys,hashlib,copy,collections,fcntl
from pathlib import Path
R=Path(__file__).resolve().parents[3];B=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def main():
 names=['terminal_area_full_20261018','scored_area_pilot_20261018']+[f'terminal_area_int2float_r{i}_20261018' for i in range(3)];O=R/'results_v2/terminal_area_evidence_20261018';O.mkdir(exist_ok=False);versions={};records={};small=load(R/'results_v2/single_output_components_evidence_20261017/integrity.json')
 for name in names:
  cp=R/'results_v2'/name/'campaign.json';c=load(cp);assert c['status']=='attempted' and len(c['jobs'])==c['expected_jobs']
  checker='check_scored_area.py' if name.startswith('scored') else 'check_terminal_area.py'
  with (O/(name+'.log')).open('w') as f:subprocess.run([sys.executable,str(B/checker),str(cp)],stdout=f,stderr=subprocess.STDOUT,check=True)
  v=load(cp.parent/('archive_validation.json' if name.startswith('scored') else 'terminal_validation.json'));versions[name]=v;(O/(name+'.json')).write_text(json.dumps(v,indent=2)+'\n')
  with (cp.parent/'runner.lock').open('r') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for j in c['jobs']:records[j['report']]=load(j['report'])['results'][0]
 for p,h in small['baseline_small_binary_check'].items():assert sha(p)==h,p
 assert sha(R/'target/release/bench_rs_egraph_map')==small['main_sha256']
 full=versions[names[0]];required=['abc_map_area_resyn2','abc_map_delay_resyn2','mockturtle_emap_area_resyn2','mockturtle_emap_delay_resyn2','eqmap_lut_amap','esyn_frozen_model_amap_bounded_dag','emorphic_final_sa'];G=O/'baseline_scope';G.mkdir();gates=[]
 for k in [3,4]:
  base=R/f'results_v2/guarded_joint_evidence_20261016/baseline_scope/baselines_{k}in.json';d=copy.deepcopy(load(base));d['results']=[]
  for row in full['rows']:
   if row['k']==k and row['profile']=='terminal_area':
    r=load(row['report'])['results'][0];r['method']='egraph_terminal_area_V29';d['results'].append(r)
  assert len(d['results'])==20;candidate=G/f'candidate_{k}in.json';candidate.write_text(json.dumps(d,indent=2)+'\n')
  for obj in ['delay-area','area-delay','pareto']:
   out=G/f'{k}in_{obj}.json';argv=[sys.executable,str(R/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(candidate),'--baselines',str(base),'--required-baselines',*required,'--objective',obj,'--output',str(out)];p=subprocess.run(argv,capture_output=True,text=True);out.with_suffix('.log').write_text(p.stdout+p.stderr);gates.append(dict(k=k,objective=obj,passed=load(out)['all_passed'],argv=argv,baseline_sha256=sha(base)))
 rep=collections.Counter();cert=[]
 for name,v in versions.items():
  if '_r' in name and 'int2float' in name:
   for m,h in v['summary']['terminal_area'].items():
    for key,n in h.items():rep[m+'_'+key]+=n
   cert.extend(v['terminal_proofs'])
 q=dict(unique_physical_attempts=len(records),unique_CEC=sum(r['status']=='ok' for r in records.values()),full_summary=full['summary'],full_CEC=full['cec'],full_attempts=full['attempts'],full_stage_attempts=full['terminal_attempted_cases'],full_stage_accepts=full['terminal_accepted_cases'],repeat_summary=dict(rep),repeat_certificates=cert,cache_summary=versions[names[1]]['summary'],cache_stage_accepts=versions[names[1]]['archive_accepted_cases'],main_promoted=False,main_sha256=small['main_sha256'],baseline_algorithms_changed=False,baseline_small_binary_check=small['baseline_small_binary_check'],prior_full_13403_file_baseline_audit_referenced_not_rerun=True,gates=gates,unconnected=['MtM'],AD_posthoc_only=True,per_run_protection_not_separate_wall_sensitive_control_guarantee=True)
 (O/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))
if __name__=='__main__':main()
