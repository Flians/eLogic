#!/usr/bin/env python3
"""Complete fixed-profile/repetition evidence, including every failure."""
import argparse,csv,json
from pathlib import Path
from run_epfl_baselines import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();c=json.loads(a.campaign.read_text());out=a.output;out.mkdir(exist_ok=False)
 assert len(c['jobs'])==c['expected_jobs'];records={};hashes={str(a.campaign.resolve()):sha(a.campaign)}
 for path,h in c['source_sha256'].items():assert sha(Path(path))==h
 for j in c['jobs']:
  for key in ['report','audit']:assert sha(Path(j[key]))==j[key+'_sha256'];hashes[j[key]]=j[key+'_sha256']
  d=json.loads(Path(j['report']).read_text());assert len(d['results'])==1;r=d['results'][0];r['library_sha256']=d['library_sha256'];r['observation_report']=j['report'];r['execution']=j.get('execution','new execution')
  if r['status']=='ok':
   n=r['native'];assert r['measured']['cec']==1 and n['enodes']<=20000 and n['max_cuts_per_class']<=16 and n['instances']<=100000 and n['heuristic_secs']<=2
   if r['settings']['EGM_STAGED_RECOVERY']=='1':assert n['refinement_trials']<=2000 and r['measured']['delay']<=n['staged_delay_bound']+1e-6
  records[j['k'],j['bench'],j['repetition'],j['profile']]=r
 contrasts=[('event','staged_sweep')] if c['study']=='repeat' else [('event','coupled_phase0'),('event','coupled_quarter'),('coupled_queue','coupled_quarter')]
 paired=[];counts={}
 for candidate,base in contrasts:
  label=candidate+'_vs_'+base;count={name:0 for name in ['valid_pairs','both_failed','one_failed','delay_better','delay_equal','delay_worse','area_better','area_equal','area_worse']}
  for (k,bench,rep,p),r in records.items():
   if p!=candidate:continue
   b=records[k,bench,rep,base];assert r['input_sha256']==b['input_sha256'] and r['preprocessing']['sha256']==b['preprocessing']['sha256'] and r['library_sha256']==b['library_sha256']
   row={'contrast':label,'k':k,'bench':bench,'repetition':rep,'candidate_status':r['status'],'base_status':b['status']}
   if r['status']==b['status']=='ok':
    count['valid_pairs']+=1
    for metric in ['area','delay']:
     x=b['measured'][metric];y=r['measured'][metric];delta=y-x;row.update({metric+'_base':x,metric+'_candidate':y,metric+'_delta':delta});count[metric+('_better' if delta < -1e-6 else '_worse' if delta > 1e-6 else '_equal')]+=1
    row.update(base_trials=b['native']['refinement_trials'],candidate_trials=r['native']['refinement_trials'])
   else:count['both_failed' if r['status']!='ok' and b['status']!='ok' else 'one_failed']+=1
   paired.append(row)
  counts[label]=count
 fields=list(dict.fromkeys(key for r in paired for key in r))
 with (out/'paired.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(paired)
 v={'jobs':len(c['jobs']),'expected_jobs':c['expected_jobs'],'contrasts':counts,'success':sum(r['status']=='ok' for r in records.values()),'failure':sum(r['status']!='ok' for r in records.values()),'adopted_pre_import_failures':sum(j.get('execution')=='adopted_pre_stage_failure' for j in c['jobs']),'hashes':hashes,'scope':c['scope'],'limitations':['same configured2000 trial parameter; staged total cap versus legacy per-round coupled cap, actual trials retained','no per-design profile or repetition winner selection','targeted repetitions do not replace full20 or establish global optimality']}
 (out/'validation.json').write_text(json.dumps(v,indent=2));(out/'full_scope.json').write_text(json.dumps([{'k':k,'bench':b,'repetition':rep,'profile':p,'result':r} for (k,b,rep,p),r in records.items()],indent=2));(out/'report.md').write_text('# Event-flow validation\n\n'+json.dumps(counts,indent=2)+'\n\nEvery profile/run retained; no selection of best results.\n');print(json.dumps(counts))
if __name__=='__main__':main()
