#!/usr/bin/env python3
"""Read-only audit of fixed-budget zero-cost shared-source selection."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();c=json.loads(a.campaign.read_text());assert c['status']=='attempted' and len(c['jobs'])==c['expected_jobs']==12;errs=[];rows=[]
for file,h in c['fingerprint'].items():
 if sha(Path(file))!=h:errs.append(['source',file])
for j in c['jobs']:
 for key in ['report','audit','pool','candidates','trace','pool_frontier','pool_guard']:
  if key in j and sha(Path(j[key]))!=j[key+'_sha256']:errs.append(['artifact',j['name'],key])
 r=json.loads(Path(j['report']).read_text())['results'][0];s=r['settings'];assert s['EGM_ZERO_COST_SOURCE']==str(int(j['profile']=='protected_area')) and s['EGM_JOINT_SOURCE']==s['EGM_SHARED_MAJORITY']=='1';f=Path(j['report']).parent/(r['bench']+'.'+r['method']);side={}
 for suffix in ['joint-source','import']:
  path=f/('pool-diagnostics.'+suffix+'.json')
  if path.exists():side[suffix]=dict(path=str(path),sha256=sha(path),record=json.loads(path.read_text()),first_captured_by_readonly_audit=True)
 q=side['joint-source']['record'];assert q['before']>=q['after_single']>=q['after_positive']>=q['after_pairs'] and q['pair_trial_cap_per_root']==16
 row=dict(name=j['name'],k=j['k'],bench=j['bench'],profile=j['profile'],row=r,sidecars=side)
 if r['status']=='ok':
  n=r['native'];m=r['measured'];t=json.loads(Path(j['trace']).read_text());total=[e['seconds'] for e in t['events'] if e['stage']=='global_total'];assert len(total)==1 and total[0]<=2
  assert m['cec']==1 and n['enodes']<=20000 and n['imported_enodes']<=20000 and n['max_cuts_per_class']<=16 and n['instances']<=100000 and n['heuristic_secs']<=2 and r['mapping_wall_secs']<=300
  assert abs(n['area']-m['area'])<1e-6 and abs(n['delay']-m['delay'])<1e-6
 else:
  assert not r.get('partial_status') and not r.get('partial_measured')
  if 'import' in side and not side['import']['record'].get('import_complete',True):assert side['import']['record']['cumulative_import_admitted_enodes']<=20000 and side['import']['record']['node_allowance']==20000
 rows.append(row)
assert not errs,errs
a.output.write_text(json.dumps(dict(fresh_attempts=len(rows),direct_cec=sum(r['row']['status']=='ok' for r in rows),rows=rows,errors=errs,promoted=False,baseline_algorithms_changed=False,scope='Independent V14source-only factor, no sourcecounts-as-actual-cover or lower-bound claims; original global budgets unchanged. No per-design version mixing.'),indent=2)+'\n');print('attempts',len(rows),'CEC',sum(r['row']['status']=='ok' for r in rows),'errors',errs)
