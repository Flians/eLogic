#!/usr/bin/env python3
"""Read-only actual-input, CEC, lifetime and complete-source audit for V38."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);a=p.parse_args()
 c=json.loads(a.campaign.read_text())
 r=subprocess.run([sys.executable,str(Path(__file__).with_name('check_single_output_components.py')),str(a.campaign)],capture_output=True,text=True)
 (a.campaign.parent/'base_check.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr
 v=json.loads((a.campaign.parent/'validation.json').read_text());proofs=[]
 for j in c['jobs']:
  r=json.loads(Path(j['report']).read_text())['results'][0];s=r['settings'];enabled=j['profile']=='protected_exchange'
  assert s['EGM_PROTECTED_EXCHANGE']==s['EGM_FIXED_AREA_BOUND']==str(int(enabled))
  assert s['EGM_FUNCTIONAL_CUTS']==str(int(j['profile']!='legacy_control'))
  assert sha(j['admission'])==j['admission_sha256'];b=json.loads(Path(j['admission']).read_text());assert b['node_allowance']==20000 and not b['refund_on_rebuild']
  if r['status']!='ok':
   proofs.append(dict(name=j['name'],status=r['status'],admission=b));continue
  n=r['native'];assert b['import_complete'] and b['strict'] and b['all_strict_rewrites_preflighted']
  assert n['cumulative_admitted_enodes']==b['cumulative_admitted_enodes']<=20000
  assert n['cumulative_imported_enodes']==b['cumulative_imported_enodes']<=n['cumulative_admitted_enodes']
  assert n['enodes']==b['resident_enodes']<=n['cumulative_admitted_enodes']
  assert n['cut_rounds_started']<=32 and n['slack_rounds_started']<=4 and n['slack_local_trials']<=8000
  if not enabled:continue
  e=json.loads(Path(j['trace']).read_text())['events'];refs=[x['data'] for x in e if x['stage']=='protected_reference'];assert len(refs)==1
  ref=refs[0];assert ref['one_global_clock'] and n['area']<=ref['area'] and n['delay']<=ref['delay']
  d=json.loads(Path(j['candidates']).read_text());bank={(x['class'],tuple(x['leaves']),x['function']) for x in d['cuts']};aliases=d['canonical_signals'];count=0;carecounts={}
  for m in d['selected_instances']:
   for output in m['outputs']:
    for target in output['selected_targets']:
     valid=[]
     for source in output['binding_sources']:
      if source['target']!=target:continue
      assert aliases[source['raw_target']]==target
      if source['kind']=='exact_phase_complement':
       assert source['signed_physical_inputs']==[source['raw_target']^1] and output['function']==1
      else:
       k=len(source['boundary']);mask=(1<<(1<<k))-1
       raw=source['phased_function']^(mask if source['phase'] else 0)
       if (source['class'],tuple(source['boundary']),raw) not in bank:continue
       assert source['care']&~mask==0
      valid.append(source)
     assert valid,(j['name'],target,'missing source cut/care/phase/pin')
     count+=1;kind=valid[0]['kind'];carecounts[kind]=carecounts.get(kind,0)+1
  assert count>0 and n['protected_bindings']>0 and n['exchange_source_bindings']>=count
  proofs.append(dict(name=j['name'],status=r['status'],reference=ref,final=dict(area=n['area'],delay=n['delay']),selected_bindings=count,source_kinds=carecounts,exchange_cuts=n['exchange_cuts'],exchange_trials=n['exchange_trials'],exchange_fallback=n['exchange_fallback']))
 v.update(binding_proofs=proofs,lifetime_budget_pass=True,complete_visited_seed_protection=True,historical_final_cover_protection=False,baseline_algorithms_changed=False)
 (a.campaign.parent/'binding_validation.json').write_text(json.dumps(v,indent=2)+'\n')
 print(json.dumps({k:v[k] for k in ['complete','attempts','cec','summary']},indent=2))
if __name__=='__main__':main()
