#!/usr/bin/env python3
"""Audit both stage profiles, with failed and repeated cases retained."""
import argparse,csv,json
from pathlib import Path
from run_epfl_baselines import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();c=json.loads(a.campaign.read_text());assert len(c['jobs'])==c['expected_jobs'];out=a.output;out.mkdir(exist_ok=False);records={};counts={k:0 for k in ['success','failure','valid_pairs','both_failed','new_success','new_failure','delay_better','delay_equal','delay_worse','area_better','area_equal','area_worse']};scheduling=[]
 for p,h in c['fingerprint'].items():assert sha(Path(p))==h
 for j in c['jobs']:
  for k in ['report','audit']:assert sha(Path(j[k]))==j[k+'_sha256']
  d=json.loads(Path(j['report']).read_text());r=d['results'][0];records[j['k'],j['bench'],j['repetition'],j['variant']]=(r,d['library_sha256']);counts['success' if r['status']=='ok' else 'failure']+=1
  if r['status']=='ok':
   n=r['native'];assert r['measured']['cec']==1 and n['enodes']<=20000 and n['instances']<=100000 and n['max_cuts_per_class']<=16 and n['heuristic_secs']<=2
   scheduling.append({'job':j['name'],'native':{k:n.get(k) for k in ['heuristic_secs','refinement_trials','slack_attempts','slack_accepted','phase_secs','seed_delay']}})
 pairs=[]
 for (k,b,rep,v),(y,l) in records.items():
  if v!=1:continue
  x,ll=records[k,b,rep,0];assert l==ll and x['input_sha256']==y['input_sha256'] and x['preprocessing']['sha256']==y['preprocessing']['sha256'];r={'k':k,'bench':b,'repetition':rep,'control_status':x['status'],'candidate_status':y['status']}
  if x['status']==y['status']=='ok':
   counts['valid_pairs']+=1
   for m in ['area','delay']:
    delta=y['measured'][m]-x['measured'][m];r.update({m+'_control':x['measured'][m],m+'_candidate':y['measured'][m],m+'_delta':delta});counts[m+('_better' if delta<-1e-6 else '_worse' if delta>1e-6 else '_equal')]+=1
  elif x['status']!='ok' and y['status']=='ok':counts['new_success']+=1;r.update(candidate_area=y['measured']['area'],candidate_delay=y['measured']['delay'])
  elif x['status']=='ok':counts['new_failure']+=1
  else:counts['both_failed']+=1
  pairs.append(r)
 fields=list(dict.fromkeys(k for r in pairs for k in r))
 with (out/'paired.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(pairs)
 v={'mode':c['mode'],'jobs':len(c['jobs']),'expected_jobs':c['expected_jobs'],'counts':counts,'pairs':pairs,'scheduling':scheduling,'campaign':str(a.campaign.resolve()),'campaign_sha256':sha(a.campaign),'all_trials_retained':True,'main_modified':False,'baseline_modified':False,'scope':'full EPFL20 two libraries delay-area' if c['mode']!='reserve' else 'sqrt two libraries two repetitions; coverage experiment, no delay delta for failed controls'}
 (out/'validation.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(counts))
if __name__=='__main__':main()
