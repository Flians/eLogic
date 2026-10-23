#!/usr/bin/env python3
"""Read-only audit for the V13 sqrt allocation study; failures remain failures."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=json.loads(a.campaign.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not a.output.exists();assert c['status']=='attempted' and len(c['jobs'])==c['expected_jobs']==6
errors=[];rows=[]
for file,h in c['fingerprint'].items():
 if sha(Path(file))!=h:errors.append(['source',file])
for j in c['jobs']:
 for key in ['report','audit','pool','candidates','trace','pool_frontier','pool_guard','helper_allocation','diagnostic_profile']:
  if key in j and sha(Path(j[key]))!=j[key+'_sha256']:errors.append(['artifact',j['name'],key])
 r=json.loads(Path(j['report']).read_text())['results'][0];settings=r['settings'];assert settings['EGM_ONLINE_SHARED_BOUND']==str(int(j['profile']=='protected_area'))
 for flag in ['IMPORT_HELPER_FILL','INTRODUCE_PHASE_INVERTERS','SHARED_AREA_BOUND','FUSED_COVER_SCORING','PRESERVE_SHARED_TRAJECTORY','ONLINE_AREA','ACTIVATE_DEFERRED','TIMING_WITNESS_POOL']:assert settings['EGM_'+flag]=='1'
 folder=Path(j['report']).parent/(r['bench']+'.'+r['method']);sidecars={}
 for name in ['import','canonical-import','helper-allocation']:
  path=folder/('pool-diagnostics.'+name+'.json')
  if path.exists():sidecars[name]=dict(path=str(path),sha256=sha(path),record=json.loads(path.read_text()))
 row=dict(name=j['name'],k=j['k'],bench=j['bench'],profile=j['profile'],status=r['status'],sidecars=sidecars,error=r.get('error'))
 if r['status']=='ok':
  n=r['native'];m=r['measured'];trace=json.loads(Path(j['trace']).read_text());total=[e['seconds'] for e in trace['events'] if e['stage']=='global_total'];assert len(total)==1
  assert j['exit_code']==0 and m['cec']==1 and n['enodes']<=20000 and n['imported_enodes']<=20000 and n['max_cuts_per_class']<=16 and n['instances']<=100000 and n['heuristic_secs']<=2 and total[0]<=2 and r['mapping_wall_secs']<=300
  assert abs(n['area']-m['area'])<1e-6 and abs(n['delay']-m['delay'])<1e-6
  row.update(measured=m,native=n,recovery_total=total[0],bound_events=[e for e in trace['events'] if e['stage']=='shared_area_bound'])
 else:
  assert not r.get('partial_measured') and not r.get('partial_status')
  imp=sidecars.get('import',{}).get('record',{})
  if not imp.get('import_complete',True):assert imp['cumulative_import_admitted_enodes']<=20000 and imp['node_allowance']==20000
 rows.append(row)
assert not errors,errors
pairs=[]
for k in [4,3]:
 for bench in c['designs']:
  rs={r['profile']:r for r in rows if r['k']==k and r['bench']==bench};x=rs['legacy_control'];y=rs['protected_area'];v=dict(k=k,bench=bench,control=x['status'],candidate=y['status'])
  if x['status']==y['status']=='ok':v.update(control_area=x['measured']['area'],candidate_area=y['measured']['area'],control_delay=x['measured']['delay'],candidate_delay=y['measured']['delay'])
  pairs.append(v)
result=dict(fresh_attempts=len(rows),direct_cec=sum(r['status']=='ok' for r in rows),rows=rows,pairs=pairs,errors=errors,budget_verified=True,promoted=False,baseline_algorithms_changed=False,scope='V13 fixed-delay online sharing-bound pilot only. Same source/pool generation; exact actual physical cover scoring with fresh refs/users/active after online commits. Not full20/repeated qualification or baseline victory; original algorithms unchanged. Failed or timed-out attempts remain failed.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['fresh_attempts','direct_cec','errors','promoted']}))
