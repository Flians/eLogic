#!/usr/bin/env python3
"""Read-only V17 source-proof factor audit; no false coverage from source counts."""
import argparse,hashlib,json
from pathlib import Path
from audit_live_area_commands import audit
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();c=json.loads(a.campaign.read_text());assert c['status']=='attempted' and len(c['jobs'])==c['expected_jobs']==16
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rows=[]
for file,h in c['fingerprint'].items():assert sha(file)==h,file
for j in c['jobs']:
 for key in ['report','audit','pool','candidates','trace','source_proof']:
  if key in j:assert sha(j[key])==j[key+'_sha256'],(j['name'],key)
 r=json.loads(Path(j['report']).read_text())['results'][0];s=r['settings'];assert s['EGM_PROOF_SOURCE']=='1';assert s['EGM_SOURCE_PROPAGATION']=='1';assert s['EGM_SOURCE_GROUPS']==str(int(j['profile']=='protected_area'));assert s['EGM_ZERO_COST_SOURCE']==s['EGM_JOINT_SOURCE']=='1'
 proof=json.loads(Path(j['source_proof']).read_text()) if 'source_proof' in j else None
 assert proof and proof['proof_decisions']<=4095 and proof['proof_calls']<=20000 and proof['after_source_and']<=proof['before_source_and'];assert proof['structural_propagation'] is True
 base=Path(j['report']).parent/(r['bench']+'.'+r['method']);side={}
 for name in ['joint-source','import']:
  path=base/('pool-diagnostics.'+name+'.json')
  if path.exists():side[name]=dict(path=str(path),sha256=sha(path),record=json.loads(path.read_text()))
 if r['status']=='ok':
  n=r['native'];m=r['measured'];t=json.loads(Path(j['trace']).read_text());total=[e['seconds'] for e in t['events'] if e['stage']=='global_total'];assert len(total)==1 and total[0]<=2
  assert j['exit_code']==0 and m['cec']==1 and n['enodes']<=20000 and n['imported_enodes']<=20000 and n['max_cuts_per_class']<=16 and n['instances']<=100000 and n['heuristic_secs']<=2 and r['mapping_wall_secs']<=300
  assert abs(n['area']-m['area'])<1e-6 and abs(n['delay']-m['delay'])<1e-6
 else:
  assert not r.get('partial_measured') and not r.get('partial_status')
  if 'import' in side and not side['import']['record'].get('import_complete',True):assert side['import']['record']['cumulative_import_admitted_enodes']<=20000 and side['import']['record']['node_allowance']==20000
 if 'joint-source' in side:
  g=side['joint-source']['record'];assert g['groups_enabled']==(j['profile']=='protected_area') and g['max_root_transactions']<=16 and g['pair_trial_cap_per_root']==16
 rows.append(dict(name=j['name'],k=j['k'],bench=j['bench'],profile=j['profile'],row=r,proof=proof,sidecars=side,command_audit=audit(j,c)))
result=dict(attempts=len(rows),CEC=sum(r['row']['status']=='ok' for r in rows),failures=sum(r['row']['status']!='ok' for r in rows),rows=rows,main_promoted=False,baseline_algorithms_changed=False,scope='V25 bounded existing-choice groups, parent V22; not mixed with V24/V19 cut-protection engine; all actual failed cases retained. Source reductions/proof aliases are not imported covers or mapped PPA. Search-window checkpoints are not preemptive whole-stage caps; all proof and graph rewiring charged to original mapper guard. Temporary BDD decisions are never egraph cut/cell candidates.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print({k:result[k] for k in ['attempts','CEC','failures']})
