#!/usr/bin/env python3
"""Read-only independent CEC/command/budget and component-factor audit."""
import argparse,collections,hashlib,json
from pathlib import Path
from audit_generation_commands import audit
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def outcome(x,y):
 t=1e-6*max(1,abs(x),abs(y));return 'better' if x<y-t else 'worse' if x>y+t else 'equal'
def main():
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);a=p.parse_args();c=json.loads(a.campaign.read_text())
 for path,h in c['fingerprint'].items():assert sha(path)==h,path
 rows=[];proofs=[]
 for j in c['jobs']:
  for k in ['report','audit','pool','candidates','trace','pool_frontier','pool_guard','helper_allocation']:
   if k in j:assert sha(j[k])==j[k+'_sha256']
  proofs.append(audit(j,c));r=json.loads(Path(j['report']).read_text())['results'][0];s=r['settings'];profile=j['profile'];valid=r['status']=='ok'
  expected={'EGM_SINGLE_STRUCTURE_CUTS':int(profile=='single_structure'),'EGM_NO_INPUT_PHASE_MATCHES':int(profile=='no_input_phase'),'EGM_CANONICAL_PIN_MATCHING':int(profile=='canonical_pins'),'EGM_NO_ACTUAL_AREA_TAIL':int(profile=='no_actual_area'),'EGM_DC_SUPPORT':0 if profile=='no_sdc' else 4,'EGM_PHASE_RECOVERY':0 if profile=='no_phase_recovery' else 1,'EGM_TAIL_SHARED_AREA':c['factors'][profile][1],'EGM_REFRESH_AREA_TIMING':0,'EGM_CUT_PIN_FLOW':1,'EGM_MULTIOUTPUT':0,'EGM_NODES':20000,'EGM_CUTS':16,'EGM_INSTANCES':100000,'EGM_ITER':0 if profile=='no_saturation' else 16,'EGM_CUT_ROUNDS':32}
  expected.update({k:0 if profile=='no_timing_area_module' else 1 for k in ['EGM_REQUIRED_AREA_DAG','EGM_DEFER_AREA_DAG','EGM_SLACK_FLOW']})
  assert all(s[k]==str(v) for k,v in expected.items()),(j['name'],expected)
  assert s['EGM_OBJECTIVE']=='delay-area' and float(s['EGM_REFINE_TIME'])==2 and int(s['EGM_REFINE_TRIALS'])==2000
  if valid:
   n=r['native'];m=r['measured'];assert n['saturation_stop'] in (['Some(IterationLimit(0))'] if profile=='no_saturation' else ['Some(Saturated)','Some(IterationLimit(16))']),('saturation budget proof requires completed node-limit checks',j['name'],n['saturation_stop']);assert m['cec']==1 and m['multioutput_instances']==0
   assert n['imported_enodes']<=20000 and n['enodes']<=20000 and n['max_cuts_per_class']<=16 and n['instances']<=100000 and n['heuristic_secs']<=2
   assert abs(n['area']-m['area'])<1e-6 and abs(n['delay']-m['delay'])<1e-6
   t=json.loads(Path(j['trace']).read_text());tot=[e['seconds'] for e in t['events'] if e['stage']=='global_total'];assert len(tot)==1 and tot[0]<=2
   if profile=='no_phase_recovery':assert n['phase_trials']==0
   if profile in ['no_actual_area','no_timing_area_module']:assert n['slack_local_scored']==0
   if profile=='no_sdc':assert n['dc_with_unreachable']==0 and n['dc_bindings']==0
   for e in t['events']:
    if e['stage']=='tail_shared_reference':assert e['data']['after_original_flow'] and e['data']['policy_changed']==e['data']['saved_predecessor']
  native_failure={}
  if not valid:
   log=Path(s['EGM_DIAG_PATH']).parent/'map.log'
   if log.exists():
    reason=log.read_text()[-1800:];stage='import_budget' if 'cannot import AIG within node allowance' in reason else 'physical_cover_missing' if 'undriven signal' in reason or 'finite physical' in reason else 'other_native_failure'
    native_failure=dict(stage=stage,log=str(log),sha256=sha(log),message=reason)
  rows.append(dict(k=j['k'],bench=j['bench'],profile=profile,valid=valid,status=r['status'],area=r.get('measured',{}).get('area'),delay=r.get('measured',{}).get('delay'),error=r.get('error'),native_failure=native_failure,report=j['report'],native=r.get('native')))
 pairs=[]
 ref='protected_area' if c['components'] else 'legacy_control'
 for r in rows:
  if r['profile']==ref:continue
  b=next((x for x in rows if (x['k'],x['bench'],x['profile'])==(r['k'],r['bench'],ref)),None)
  if b is None:continue
  z=dict(k=r['k'],bench=r['bench'],profile=r['profile'],reference=b,candidate=r,valid=b['valid'] and r['valid'])
  if z['valid']:z.update(area_delta=r['area']-b['area'],delay_delta=r['delay']-b['delay'],area_outcome=outcome(r['area'],b['area']),delay_outcome=outcome(r['delay'],b['delay']))
  pairs.append(z)
 summary={f:{metric:dict(collections.Counter(x[metric] for x in pairs if x['profile']==f and x['valid'])) for metric in ['area_outcome','delay_outcome']} for f in c['factors'] if f!=ref}
 coverage={f:dict(attempts=sum(r['profile']==f for r in rows),CEC=sum(r['profile']==f and r['valid'] for r in rows),failures=dict(collections.Counter(r['native_failure'].get('stage','unclassified') for r in rows if r['profile']==f and not r['valid']))) for f in c['factors']}
 pair_coverage={f:dict(total=sum(p['profile']==f for p in pairs),both_legal=sum(p['profile']==f and p['valid'] for p in pairs),reference_failure=sum(p['profile']==f and not p['reference']['valid'] for p in pairs),candidate_only_failure=sum(p['profile']==f and p['reference']['valid'] and not p['candidate']['valid'] for p in pairs)) for f in c['factors'] if f!=ref}
 v=dict(complete=len(rows)==c['expected_jobs'],attempts=len(rows),expected=c['expected_jobs'],cec=sum(r['valid'] for r in rows),rows=rows,pairs=pairs,summary=summary,reference=ref,coverage=coverage,pair_coverage=pair_coverage,independent_commands=proofs,source_and_budget_audits_pass=True,main_promoted=False,baseline_algorithms_changed=False)
 (a.campaign.parent/'validation.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v[k] for k in ['complete','attempts','cec','summary']},indent=2))
 for r in pairs:
  if r['valid'] and (r['area_outcome']=='worse' or r['delay_outcome']=='worse'):print('adverse',r['k'],r['bench'],r['profile'],r['area_delta'],r['delay_delta'])
if __name__=='__main__':main()
