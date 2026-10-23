#!/usr/bin/env python3
"""Audit all observations; never select the best randomized/deadline run."""
import argparse,hashlib,json
from pathlib import Path
from report_delay_frontier import pool_hash
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaigns',nargs='+',type=Path,required=True);ap.add_argument('--full',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=False)
 observations=[];errors=[];groups={};native=None
 full=json.loads(a.full.read_text());sources=[(a.full,full,True)]+[(p,json.loads(p.read_text()),False) for p in a.campaigns]
 designs={b for _,c,f in sources if not f for b in c['designs']}
 for path,c,isfull in sources:
  frozen={p:h for p,h in c['fingerprint'].items() if Path(p).name not in {'run_epoch_incumbent.py','run_epoch_incumbent_repeats.py'}}
  if native is None:native=frozen
  assert frozen==native,'Actual frozen dependencies differ'
  if not isfull:assert len(c['jobs'])==c['expected_jobs'],'Incomplete repeat batch'
  for j in c['jobs']:
   if j['bench'] not in designs or j['profile'] not in ('legacy_control','protected_area'):continue
   if isfull and j.get('adopted_from') and j.get('new_mapper_run') is False:continue
   for field in ('report','audit','pool','candidates','trace'):
    if field in j:assert sha(Path(j[field]))==j[field+'_sha256'],(j['name'],field)
   r=json.load(open(j['report']))['results'][0] if 'report' in j else {'status':'unmeasured_failure','error':'Driver exited before report; original argv/log/audit retained','exit_code':j['exit_code']};rec={'campaign':str(path),'job':j['name'],'k':j['k'],'bench':j['bench'],'profile':j['profile'],'repetition':'full_r0' if isfull else j['repetition'],'row':r,'report':j.get('report'),'report_sha256':j.get('report_sha256'),'argv':j['argv']}
   if 'candidates' in j:rec['pool_fingerprint']=pool_hash(Path(j['candidates']))
   if r['status']=='ok':
    n=r['native'];assert r['measured']['cec']==1
    if n['enodes']>20000 or n['max_cuts_per_class']>16 or n['instances']>100000 or n['heuristic_secs']>2:errors.append(j['name'])
   observations.append(rec);groups.setdefault((str(path),j.get('repetition','full_r0'),j['k'],j['bench']),{})[j['profile']]=rec
 pairs=[];counts={'delay':[0,0,0],'area':[0,0,0]};fp=[]
 for key,g in groups.items():
  assert set(g)=={'legacy_control','protected_area'},key
  x,y=g['legacy_control'],g['protected_area'];pair={'key':key,'control':x,'candidate':y};pairs.append(pair)
  if x['row']['status']==y['row']['status']=='ok':
   assert x['row']['input_sha256']==y['row']['input_sha256']
   assert x['row']['preprocessing']['sha256']==y['row']['preprocessing']['sha256']
   if x.get('pool_fingerprint')!=y.get('pool_fingerprint'):fp.append(key)
   for m in counts:
    u=x['row']['measured'][m];v=y['row']['measured'][m];t=1e-6*max(1,abs(u),abs(v));counts[m][0 if v<u-t else 2 if v>u+t else 1]+=1
 ranges=[]
 for k,b in sorted({(o['k'],o['bench']) for o in observations}):
  item={'k':k,'bench':b}
  for profile in ('legacy_control','protected_area'):
   rs=[o for o in observations if o['k']==k and o['bench']==b and o['profile']==profile];ok=[r for r in rs if r['row']['status']=='ok'];item[profile]={'observations':len(rs),'CEC':len(ok),'area_range':[min(r['row']['measured']['area'] for r in ok),max(r['row']['measured']['area'] for r in ok)] if ok else None,'delay_range':[min(r['row']['measured']['delay'] for r in ok),max(r['row']['measured']['delay'] for r in ok)] if ok else None}
  ranges.append(item)
 summary={'observations':len(observations),'fresh_repeat_observations':sum(not o['repetition']=='full_r0' for o in observations),'CEC':sum(o['row']['status']=='ok' for o in observations),'paired_observations':len(pairs),'better_equal_worse':counts,'pool_fingerprint_mismatches':fp,'budget_errors':errors,'ranges':ranges,'baseline_algorithms_changed':False,'production_promoted':False,'scope':'Targeted repetitions and full-suite first observations; all observations retained; no best-repeat selection; two repeated batches alternate profile order.'}
 (a.output/'observations.json').write_text(json.dumps(observations,indent=2));(a.output/'paired.json').write_text(json.dumps(pairs,indent=2));(a.output/'validation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
