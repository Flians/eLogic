#!/usr/bin/env python3
"""Compare two explicit fixed recovery profiles with physical CEC/budget checks."""
import argparse,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def pool_hash(p):
 with p.open() as f:
  carry=''
  while True:
   data=f.read(65536)
   if not data:raise ValueError('missing pool fingerprint')
   carry+=data;m=re.search(r'"retained_pool_fingerprint"\s*:\s*"([0-9a-f]+)"',carry)
   if m:return m[1]
   carry=carry[-256:]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--control',default='trace_control');ap.add_argument('--candidate',default='fast_frontier');a=ap.parse_args();out=a.output;out.mkdir(exist_ok=False);c=json.loads(a.campaign.read_text());groups={};budget_errors=[];rows=[];hash_errors=[]
 for j in c['jobs']:
  for field in ['report','audit','pool','candidates','trace','diagnostic_profile']:
   if field in j and sha(Path(j[field]))!=j[field+'_sha256']:hash_errors.append(j['name']+':'+field)
  r=json.load(open(j['report']))['results'][0];record={'job':j['name'],'k':j['k'],'bench':j['bench'],'profile':j['profile'],'row':r,'report':j['report'],'report_sha256':j['report_sha256'],'new_mapper_run':j.get('new_mapper_run',not bool(j.get('adopted_from'))),'adopted_from':j.get('adopted_from')}
  if 'trace' in j:record['trace']=json.load(open(j['trace']))
  if 'diagnostic_profile' in j:record['diagnostic_profile']=json.load(open(j['diagnostic_profile']))
  if 'candidates' in j:record['pool_fingerprint']=pool_hash(Path(j['candidates']))
  if r['status']=='ok':
   n=r['native'];m=r['measured'];assert m['cec']==1
   if n['enodes']>20000 or n['max_cuts_per_class']>16 or n['instances']>100000 or n['heuristic_secs']>2:budget_errors.append(j['name'])
  rows.append(record);groups.setdefault((j['k'],j['bench']),{})[j['profile']]=record
 pairs=[];counts={'delay':[0,0,0],'area':[0,0,0]};fp=[]
 for (k,b),g in groups.items():
  x,y=g[a.control],g[a.candidate];pair={'k':k,'bench':b,'control':x,'candidate':y};pairs.append(pair)
  if x['row']['status']==y['row']['status']=='ok':
   assert x['row']['input_sha256']==y['row']['input_sha256']
   assert all(x['row']['preprocessing'].get(key)==y['row']['preprocessing'].get(key) for key in ['sha256','recipe','reused_from','source_report_sha256'])
   assert x['row']['preprocessing']['cec']==y['row']['preprocessing']['cec']==1
   if x.get('pool_fingerprint')!=y.get('pool_fingerprint'):fp.append([k,b])
   for key in counts:
    u,v=x['row']['measured'][key],y['row']['measured'][key];t=1e-6*max(1,abs(u),abs(v));counts[key][0 if v<u-t else 2 if v>u+t else 1]+=1
 summary={'case_slots':len(rows),'expected_slots':c['expected_jobs'],'cec':sum(r['row']['status']=='ok' for r in rows),'failure_slots':sum(r['row']['status']!='ok' for r in rows),'same_source_pilot_reused':sum(bool(r['adopted_from']) and r['row']['status']=='ok' for r in rows),'historical_pre_recovery_rejections_reused':sum(bool(r['adopted_from']) and r['row']['status']!='ok' for r in rows),'control':a.control,'candidate':a.candidate,'eligible_pairs':sum(counts['delay']),'better_equal_worse':counts,'pool_fingerprint_mismatch':fp,'budget_errors':budget_errors,'artifact_hash_errors':hash_errors,'baseline_algorithms_changed':False,'production_promoted':False,'objective':'delay-area only','source_manifest':str(a.campaign),'source_manifest_sha256':sha(a.campaign),'scope':'Full declared scope, failures retained; historical import rejections are not new runs, no per-design source/profile winners'}
 (out/'validation.json').write_text(json.dumps(summary,indent=2));(out/'paired.json').write_text(json.dumps(pairs,indent=2));(out/'all_rows.json').write_text(json.dumps(rows,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
