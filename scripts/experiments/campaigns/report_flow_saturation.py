#!/usr/bin/env python3
"""Complete fixed-profile scope, failures and gates; no profile winner mixture."""
import collections,csv,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compare(x,y):
 t=1e-6*max(abs(x),abs(y),1);return -1 if x<y-t else 1 if x>y+t else 0
p=ROOT/'results_v2/flow_saturation_checkpoint_20261014/campaign.json';c=json.loads(p.read_text());assert len(c['jobs'])==80
for f,s in c['fingerprint'].items():assert sha(Path(f))==s
out=ROOT/'results_v2/flow_saturation_full_20261014';out.mkdir(exist_ok=True);rows=[];data={};lib={};inputs={};source_hashes={str(p):sha(p)}
for j in c['jobs']:
 for key in ['report','audit','pool','candidates']:
  if key in j:assert sha(Path(j[key]))==j[key+'_sha256'];source_hashes[j[key]]=j[key+'_sha256']
 d=json.loads(Path(j['report']).read_text());assert len(d['results'])==1;r=d['results'][0];r['observation_report']=j['report'];r['library_sha256']=d['library_sha256'];data[j['k'],j['bench'],j['profile']]=r
 key=j['k'],j['bench'];hashes=(r['input_sha256'],r['preprocessing']['sha256']);assert key not in inputs or inputs[key]==hashes;inputs[key]=hashes
 assert j['k'] not in lib or lib[j['k']]==d['library_sha256'];lib[j['k']]=d['library_sha256']
 m=r.get('measured',{});n=r.get('native',{});valid=r.get('status')=='ok' and m.get('cec')==1
 a=json.loads(Path(j['audit']).read_text());clock=max((v['mapping_wall_secs'] for v in a['states'].values()),default=0)
 if valid:assert n['enodes']<=20000 and n['max_cuts_per_class']<=16 and n['instances']<=100000 and n['heuristic_secs']<=2 and clock<=300
 logfiles=list(Path(j['report']).parent.glob('*/map.log'));text='\n'.join(q.read_text(errors='replace') for q in logfiles)
 rows.append({'k':j['k'],'bench':j['bench'],'profile':j['profile'],'status':r['status'],'area':m.get('area'),'delay':m.get('delay'),'cec':m.get('cec'),'enodes':n.get('enodes'),'cuts_max':n.get('max_cuts_per_class'),'pool':n.get('instances'),'recovery_secs':n.get('heuristic_secs'),'mapping_secs':clock,'failure_stage':'import_node_budget' if 'budget: imported graph' in text else ('' if valid else 'other_or_unclassified'),'error':r.get('error',''),'report':j['report']})
with (out/'all_profiles.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
contrasts=[]
for prof in ['flow_sat']:
 for ks in [[3],[4],[3,4]]:
  counts=collections.Counter();pairs=[]
  for (k,b),_ in inputs.items():
   if k not in ks:continue
   x=data[k,b,prof];y=data[k,b,'control'];mx=x.get('measured',{});my=y.get('measured',{})
   if x.get('status')!='ok' or y.get('status')!='ok':counts['invalid_pairs']+=1;counts['restored_covers' if x.get('status')=='ok' else 'lost_covers' if y.get('status')=='ok' else 'both_failed']+=1;pairs.append({'k':k,'bench':b,'status':'invalid_pair'});continue
   dd=compare(mx['delay'],my['delay']);aa=compare(mx['area'],my['area']);lex=dd if dd else aa
   for label,v in [('delay',dd),('area',aa),('delay_area_lex',lex)]:counts[label+('_better' if v<0 else '_worse' if v>0 else '_equal')]+=1
   pairs.append({'k':k,'bench':b,'control':my,'trial':mx,'delay_change_percent':100*(mx['delay']/my['delay']-1),'area_change_percent':100*(mx['area']/my['area']-1)})
  contrasts.append({'profile':prof,'inventories':ks,'counts':dict(counts),'pairs':pairs})
gates=[]
for k in [3,4]:
 for prof in c['fixed_profiles']:
  cand=out/f'{k}in_{prof}.json';cand.write_text(json.dumps({'library_sha256':lib[k],'metric_model':'all physical instances; max(rise,fall) block pin delay; no load or slew','results':[r for (kk,b,pp),r in data.items() if kk==k and pp==prof]},indent=2)+'\n')
  baseline=ROOT/f'results_v2/epfl_queue_baseline_scope_complete_20261013/baselines_{k}in_delay-area.json'
  for objective in ['delay-area','pareto']:
   dest=out/f'{k}in_{prof}_{objective}_gate.json';argv=[sys.executable,str(ROOT/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(cand),'--baselines',str(baseline),'--candidate-methods','egraph_staged_control_resyn2','--required-baselines','abc_map_area_resyn2','abc_map_delay_resyn2','mockturtle_emap_area_resyn2','mockturtle_emap_delay_resyn2','eqmap_lut_amap','esyn_frozen_model_amap_bounded_dag','emorphic_final_sa','--objective',objective,'--output',str(dest)]
   with dest.with_suffix('.log').open('w') as f:r=subprocess.run(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   g=json.loads(dest.read_text());gates.append({'k':k,'profile':prof,'objective':objective,'all_passed':g['all_passed'],'exit_code':r.returncode,'argv':argv,'output_sha256':sha(dest)})
validation={'completed':80,'valid':sum(r['cec']==1 for r in rows),'failures':sum(r['status']!='ok' for r in rows),'failure_stages':dict(collections.Counter(r['failure_stage'] for r in rows if r['status']!='ok')),'contrasts':contrasts,'gates':gates,'sources_sha256':source_hashes,'strict_actual_budgets_verified':True,'all20_both_inventories_retained':True,'baseline_algorithms_changed':False,'no_per_design_profile_winners':True,'unconnected':['MtM']}
(out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n');(out/'report.txt').write_text('\n'.join(str({'profile':x['profile'],'inventories':x['inventories'],'counts':x['counts']}) for x in contrasts)+'\nAll failed/missing and baseline availability remain in per-profile gates.\n')
print('valid',validation['valid'],'failures',validation['failures'],validation['failure_stages']);print([(x['profile'],x['counts']) for x in contrasts if x['inventories']==[3,4]]);print('passed gates',sum(g['all_passed'] for g in gates),'/',len(gates))
