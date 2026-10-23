#!/usr/bin/env python3
"""All targeted alternating-order repeats plus full-study r0; no best-run selection."""
import hashlib,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cmp(x,y):
 t=1e-6*max(abs(x),abs(y),1);return -1 if x<y-t else 1 if x>y+t else 0
full=ROOT/'results_v2/canonical_wire_checkpoint_20261014/campaign.json';rep=ROOT/'results_v2/canonical_wire_repeats_20261014/campaign.json'
f=json.loads(full.read_text());c=json.loads(rep.read_text());assert len(f['jobs'])==80 and len(c['jobs'])==20
for camp in [f,c]:
 for q,h in camp['fingerprint'].items():assert sha(Path(q))==h
rows={};sources={str(full):sha(full),str(rep):sha(rep)}
for camp,r0 in [(f,True),(c,False)]:
 for j in camp['jobs']:
  if r0 and (j['k'],j['bench']) not in [(4,'ctrl'),(4,'priority'),(4,'bar'),(4,'dec'),(3,'bar')]:continue
  for key in ['report','audit','pool','candidates']:
   if key in j:assert sha(Path(j[key]))==j[key+'_sha256'];sources[j[key]]=j[key+'_sha256']
  d=json.loads(Path(j['report']).read_text());r=d['results'][0];n=r['native'];a=json.loads(Path(j['audit']).read_text());clock=max((s['mapping_wall_secs'] for s in a['states'].values()),default=0)
  assert r['status']=='ok' and r['measured']['cec']==1 and n['enodes']<=20000 and n['max_cuts_per_class']<=16 and n['instances']<=100000 and n['heuristic_secs']<=2 and clock<=300
  rows[j['k'],j['bench'],0 if r0 else j['repetition'],j['profile']]={'measured':r['measured'],'input':r['input_sha256'],'prepared':r['preprocessing']['sha256'],'library':d['library_sha256'],'report':j['report']}
pairs=[];count=collections.Counter()
for k,b in [(4,'ctrl'),(4,'priority'),(4,'bar'),(4,'dec'),(3,'bar')]:
 for rep in [0,1,2]:
  x=rows[k,b,rep,'wires_area'];y=rows[k,b,rep,'control'];assert (x['input'],x['prepared'],x['library'])==(y['input'],y['prepared'],y['library'])
  m=x['measured'];n=y['measured'];dd=cmp(m['delay'],n['delay']);aa=cmp(m['area'],n['area']);lex=dd if dd else aa
  for label,v in [('delay',dd),('area',aa),('delay_area',lex)]:count[label+('_better' if v<0 else '_worse' if v>0 else '_equal')]+=1
  pairs.append({'k':k,'bench':b,'repetition':rep,'control':n,'trial':m,'delay_difference':m['delay']-n['delay'],'area_difference':m['area']-n['area'],'lex_comparison':lex,'control_report':y['report'],'trial_report':x['report']})
out=ROOT/'results_v2/canonical_wire_repeats_full_20261014';out.mkdir(exist_ok=True);v={'covers':30,'pairs':15,'counts':dict(count),'all_runs':pairs,'sources_sha256':sources,'actual_budgets_verified':True,'named_port_cec_passed':30,'not_full_suite_repeats':True,'no_best_run_selection':True};(out/'validation.json').write_text(json.dumps(v,indent=2)+'\n');print(dict(count))
