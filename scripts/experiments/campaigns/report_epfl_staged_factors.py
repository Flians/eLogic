#!/usr/bin/env python3
"""Read-only complete-scope factorial contrasts; never select per-design winners."""
import argparse,csv,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 c=json.loads(a.campaign.read_text());out=a.output;out.mkdir(exist_ok=False)
 assert len(c['jobs'])==c['expected_jobs']==160
 hashes={str(a.campaign.resolve()):sha(a.campaign)};rows={};success=0;failures=[]
 for p,h in c['source_sha256'].items():assert sha(p)==h,(p,'source drift')
 for j in c['jobs']:
  for key in ['report','audit']:
   assert key in j
   assert sha(j[key])==j[key+'_sha256'];hashes[j[key]]=j[key+'_sha256']
  d=json.loads(Path(j['report']).read_text());assert len(d['results'])==1
  r=d['results'][0];r['library_sha256']=d['library_sha256'];key=(j['k'],j['bench'],j['profile']);assert key not in rows;rows[key]=r
  if r['status']=='ok':
   success+=1;n=r['native'];assert r['measured']['cec']==1
   assert n['enodes']<=20000 and n['max_cuts_per_class']<=16 and n['instances']<=100000
   assert n['heuristic_secs']<=2 and n['refinement_trials']<=2000
   assert r['measured']['delay']<=n['staged_delay_bound']+1e-6
  else:failures.append({'k':j['k'],'bench':j['bench'],'profile':j['profile'],'error':r.get('error')})
 contrasts=[('event_no_critical','control','region_phase'),('event_with_critical','critical_only','both'),('critical_no_event','control','critical_only'),('critical_with_event','region_phase','both'),('combined','control','both')]
 paired=[];counts={}
 for label,base,candidate in contrasts:
  count={key:0 for key in ['valid_pairs','both_failed','one_failed','delay_better','delay_equal','delay_worse','area_better','area_equal','area_worse']}
  for (k,bench,profile),b in rows.items():
   if profile!=base:continue
   r=rows[k,bench,candidate];assert b['input_sha256']==r['input_sha256'];assert b['preprocessing']['sha256']==r['preprocessing']['sha256'];assert b['library_sha256']==r['library_sha256']
   row={'contrast':label,'k':k,'bench':bench,'base_profile':base,'candidate_profile':candidate,'base_status':b['status'],'candidate_status':r['status']}
   if b['status']==r['status']=='ok':
    count['valid_pairs']+=1
    for metric in ['area','delay']:
     x=b['measured'][metric];y=r['measured'][metric];delta=y-x
     row.update({metric+'_base':x,metric+'_candidate':y,metric+'_delta':delta})
     count[metric+('_better' if delta < -1e-6 else '_worse' if delta > 1e-6 else '_equal')]+=1
   else:count['both_failed' if b['status']!='ok' and r['status']!='ok' else 'one_failed']+=1
   paired.append(row)
  counts[label]=count
 fields=list(dict.fromkeys(key for row in paired for key in row))
 with (out/'paired.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(paired)
 v={'jobs':len(c['jobs']),'expected_jobs':160,'success':success,'failed':len(failures),'contrasts':counts,'hashes':hashes,'scope':'same binary, all20,2 libraries,delay-area,4 fixed staged profiles; no per-design selection','baseline_algorithms_changed':False,'main_changed':False}
 (out/'validation.json').write_text(json.dumps(v,indent=2));(out/'full_scope.json').write_text(json.dumps([{'k':k,'bench':b,'profile':p,'result':r} for (k,b,p),r in rows.items()],indent=2));(out/'failures.json').write_text(json.dumps(failures,indent=2))
 lines=['# Staged recovery factorial experiment','','160 attempted jobs, 4 fixed profiles, same isolated binary and immutable budgets. All failures retained.','']
 for label,count in counts.items():lines.append(f"- {label}: {count}")
 lines+=['','Changes affect propagation/proposal ordering under the same global2s recovery and2000-total-trial cap. These single deadline-sensitive runs do not establish optimality or statistical stability; no profile is adopted by default.']
 (out/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(v['contrasts']))
if __name__=='__main__':main()
