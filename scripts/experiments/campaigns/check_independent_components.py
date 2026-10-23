#!/usr/bin/env python3
"""Audit component experiments without executing or repairing any mapping."""
import argparse,collections,hashlib,json,math
from pathlib import Path
from audit_generation_commands import audit
from run_independent_components import PROFILES
ROOT=Path(__file__).resolve().parents[3]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def outcome(x,y):
 t=1e-6*max(1.,abs(x),abs(y));return 'lower' if x<y-t else 'higher' if x>y+t else 'equal'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('campaign',type=Path);a=ap.parse_args();c=json.loads(a.campaign.read_text());assert all(sha(p)==h for p,h in c['fingerprint'].items());rows=[];proofs=[];settings={}
 for j in c['jobs']:
  for key in ['report','audit','admission','pool','candidates','trace']:
   if key in j:assert sha(j[key])==j[key+'_sha256']
  if 'report'not in j:
   rows.append(dict(k=j['k'],bench=j['bench'],profile=j['profile'],valid=False,status='unmeasured_failure',error='no report; original argv/log retained',report=None));continue
  d=json.loads(Path(j['report']).read_text())
  for manifest in ['binary_sha256','source_sha256']:
   for path,digest in d[manifest].items():
    if path in c['fingerprint']:assert digest==c['fingerprint'][path],(j['name'],'executed report source/binary drift',path)
  assert len(d['results'])==1;r=d['results'][0];s=r['settings'];valid=r['status']=='ok';profile=j['profile'];proof=audit(j,c);proofs.append(proof)
  expected=dict(EGM_ABLATE_FUNCTIONAL_CUTS=int(profile=='no_functional_cuts'),EGM_SINGLE_STRUCTURE_CUTS=int(profile=='single_witness_cuts'),EGM_NO_INPUT_PHASE_MATCHES=int(profile=='no_input_phase'),EGM_CANONICAL_PIN_MATCHING=int(profile=='canonical_pins'),EGM_DC_SUPPORT=0 if profile=='no_sdc' else 4,EGM_ABLATE_SHARED_AREA=int(profile=='no_shared_area'),EGM_ABLATE_INCUMBENT_GUARD=int(profile=='no_incumbent_guard'))
  assert all(s[k]==str(v)for k,v in expected.items()),(j['name'],expected)
  fixed=dict(EGM_STRICT_BUDGETS=1,EGM_MULTIOUTPUT=0,EGM_NODES=20000,EGM_CUTS=16,EGM_INSTANCES=100000,EGM_ITER=16,EGM_CUT_ROUNDS=32,EGM_REFINE_ROUNDS=4,EGM_REFINE_TRIALS=2000,EGM_PROTECTED_EXCHANGE=1,EGM_RETAIN_GENERATION_FRONTIER=1,EGM_EXCHANGE_FUNCTIONAL_SEED=1,EGM_FIXED_AREA_BOUND=1,EGM_FUNCTIONAL_CUTS=1,EGM_PHASE_RECOVERY=1,EGM_NO_ACTUAL_AREA_TAIL=0,EGM_POOL_FRONTIER_AUDIT=0,EGM_POOL_DIAG_DETAIL=0)
  assert all(s[k]==str(v)for k,v in fixed.items()),(j['name'],'fixed settings');assert float(s['EGM_REFINE_TIME'])==2 and s['EGM_OBJECTIVE']=='delay-area'
  normalized={k:v for k,v in s.items() if k not in ['EGM_DIAG_PATH','EGM_POOL_DIAG_PATH','EGM_RECOVERY_TRACE',*expected]};key=j['k'],j['bench'];assert key not in settings or settings[key]==normalized,(j['name'],'unexpected multi-factor setting drift');settings[key]=normalized
  ledger=json.loads(Path(j['admission']).read_text()) if 'admission'in j else None
  if ledger:
   assert ledger['node_allowance']==20000 and not ledger['refund_on_rebuild'];assert ledger['resident_enodes']<=ledger['cumulative_admitted_enodes']<=20000
   if ledger['import_complete']:assert ledger['strict'] and ledger['all_strict_rewrites_preflighted']
   else:assert not valid and ledger['rejected_before_insertion'] and s['EGM_STRICT_BUDGETS']=='1'
  mapping=sum(v['mapping_wall_secs']for v in json.loads(Path(j['audit']).read_text())['states'].values());failure=None;selected=0;mechanisms=collections.Counter();seed_reference=None;outside_seed_envelope=None
  if valid:
   assert ledger and ledger['import_complete'] and ledger['all_strict_rewrites_preflighted'];n=r['native'];m=r['measured'];assert m['cec']==1 and m['multioutput_instances']==0
   assert n['cumulative_admitted_enodes']==ledger['cumulative_admitted_enodes']<=20000 and n['cumulative_imported_enodes']<=n['cumulative_admitted_enodes'];assert n['enodes']<=n['cumulative_admitted_enodes'];assert n['max_cuts_per_class']<=16 and n['instances']<=100000 and n['heuristic_secs']<=2 and n['cut_rounds_started']<=32 and n['slack_rounds_started']<=4 and n['slack_local_trials']<=8000
   assert abs(n['area']-m['area'])<1e-6 and abs(n['delay']-m['delay'])<1e-6;assert n['saturation_stop'] in ['Some(Saturated)','Some(IterationLimit(16))']
   events=json.loads(Path(j['trace']).read_text())['events'];total=[e['seconds']for e in events if e['stage']=='global_total'];assert len(total)==1 and total[0]<=2
   ab=[e['data']for e in events if e['stage']=='component_ablation'];assert len(ab)==1 and ab[0]['same_global_deadline'] and ab[0]['structural_legality_and_proofs_retained'];assert ab[0]['functional_cuts_disabled']==(profile=='no_functional_cuts') and ab[0]['shared_area_disabled']==(profile=='no_shared_area') and ab[0]['incumbent_metric_guard_disabled']==(profile=='no_incumbent_guard')
   if profile=='no_shared_area':assert n['slack_local_scored']==0 and n['slack_attempts']==0 and not any(e['stage']=='fixed_bound_area_flow'for e in events)
   if profile=='no_sdc':assert n['dc_with_unreachable']==n['dc_bindings']==0
   if profile=='no_functional_cuts':assert n['functional_cut_trials']==n['functional_cut_admitted']==n['exchange_trials']==0
   seed_reference=next(e['data']for e in events if e['stage']=='protected_reference');outside_seed_envelope=n['area']>seed_reference['area'] or n['delay']>seed_reference['delay']
   if profile!='no_incumbent_guard':assert not outside_seed_envelope
   canonical_pins={cell['name']:cell['vars']for cell in json.loads((Path(j['report']).parent/'library.json').read_text())};diag=json.loads(Path(j['candidates']).read_text());bank={(x['class'],tuple(x['leaves']),x['function'])for x in diag['cuts']};aliases=diag['canonical_signals']
   for instance in diag['selected_instances']:
    for output in instance['outputs']:
     for target in output['selected_targets']:
      sources=[]
      for source in output['binding_sources']:
       if source['target']!=target:continue
       assert aliases[source['raw_target']]==target
       if source['kind']=='exact_phase_complement':assert source['signed_physical_inputs']==[source['raw_target']^1] and output['function']==1
       else:
        mask=(1<<(1<<len(source['boundary'])))-1;raw=source['phased_function']^(mask if source['phase'] else 0)
        if (source['class'],tuple(source['boundary']),raw)not in bank:continue
        assert source['care']&~mask==0
       sources.append(source)
      assert sources,(j['name'],'selected binding lost original source/care/phase/pin proof',target);selected+=1
      if all(p['kind']=='fixed_boundary_sdc'for p in sources):mechanisms['selected_sdc_only_bindings']+=1
      if all(p['kind']!='exact_phase_complement'and any(v%2 for v in p['signed_physical_inputs'])for p in sources):mechanisms['selected_complemented_input_only_bindings']+=1
      if all(p['named_pins']!=canonical_pins[instance['cell']]for p in sources):mechanisms['selected_noncanonical_pin_only_bindings']+=1
   if profile=='no_sdc':assert mechanisms['selected_sdc_only_bindings']==0
   if profile=='no_input_phase':assert mechanisms['selected_complemented_input_only_bindings']==0
   if profile=='canonical_pins':assert mechanisms['selected_noncanonical_pin_only_bindings']==0
   assert selected>0
  else:
   folder=Path(j['report']).parent/(r['bench']+'.'+r['method']);log=folder/'map.log';message=log.read_text()[-2200:] if log.exists() else '';error=r.get('error','')
   stage='cumulative_node_budget' if ledger and not ledger['import_complete'] else 'native_guard' if error=='timeout after 60s' else 'recovery_budget' if 'recovery' in message and 'budget:' in message else 'physical_cover_missing' if 'undriven signal' in message or 'finite physical' in message else 'verification_or_other_failure'
   failure=dict(stage=stage,error=error,map_log=str(log),message=message)
  rows.append(dict(k=j['k'],bench=j['bench'],profile=profile,valid=valid,status=r['status'],area=r.get('measured',{}).get('area'),delay=r.get('measured',{}).get('delay'),mapping_wall_upper_secs=mapping,selected_proven_bindings=selected,selected_mechanisms=dict(mechanisms),seed_reference=seed_reference,outside_seed_envelope=outside_seed_envelope,failure=failure,report=j['report'],report_sha256=j['report_sha256'],audit=j['audit'],native=r.get('native')))
 pairs=[]
 for row in rows:
  if row['profile']=='full':continue
  ref=next((r for r in rows if (r['k'],r['bench'],r['profile'])==(row['k'],row['bench'],'full')),None)
  if ref is None:continue
  z=dict(k=row['k'],bench=row['bench'],profile=row['profile'],valid=row['valid']and ref['valid'],candidate=row,reference=ref)
  if z['valid']:z.update(area_outcome=outcome(row['area'],ref['area']),delay_outcome=outcome(row['delay'],ref['delay']),area_delta=row['area']-ref['area'],delay_delta=row['delay']-ref['delay'])
  pairs.append(z)
 coverage={p:dict(attempts=sum(r['profile']==p for r in rows),CEC=sum(r['profile']==p and r['valid']for r in rows),failure_stages=dict(collections.Counter((r.get('failure')or{}).get('stage','unmeasured')for r in rows if r['profile']==p and not r['valid'])))for p in c['profiles']}
 summary={p:{m:dict(collections.Counter(z[m+'_outcome']for z in pairs if z['profile']==p and z['valid']))for m in ['area','delay']}for p in c['profiles']if p!='full'}
 validation=dict(complete=len(rows)==c['expected_jobs'] and c.get('status')=='attempted',attempts=len(rows),expected=c['expected_jobs'],cec=sum(r['valid']for r in rows),coverage=coverage,summary=summary,rows=rows,pairs=pairs,commands=proofs,baseline_algorithms_changed=False,source_budget_and_single_factor_audit_pass=True,historical_full_budget_incumbent_guaranteed=False)
 (a.campaign.parent/'validation.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps({k:validation[k]for k in ['complete','attempts','cec','coverage','summary']},indent=2))
if __name__=='__main__':main()
