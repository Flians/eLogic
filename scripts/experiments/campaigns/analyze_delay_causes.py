#!/usr/bin/env python3
"""Read-only reconstruction from frozen CEC-passed netlists; not a causal experiment."""
from collections import Counter,defaultdict,deque
from pathlib import Path
import hashlib,json,struct,math
ROOT=Path(__file__).resolve().parents[3]
f32=lambda x:struct.unpack('f',struct.pack('f',x))[0]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def reconstruct(path,library,measured):
 pins=defaultdict(list)
 for c in json.loads(library.read_text()):pins[c['name']].append(c)
 m=json.loads(path.read_text())['modules']['mapped'];a={'0':0.,'1':0.};previous={};drivers={};area=0.;depth={'0':0,'1':0}
 for name,p in m['ports'].items():
  if p['direction']=='input':
   for b in p['bits']:a[b]=0.;depth[b]=0;previous[b]={'input_port':name}
 for name,c in m['cells'].items():
  models=pins[c['type']];assert models;area+=f32(models[0]['area'])
  for model in models:
   b=c['connections'][model['out_pin']][0];assert b not in drivers;drivers[b]=(name,c,model)
 unknown={};users=defaultdict(list);ready=deque();order=[]
 for b,(_,c,p) in drivers.items():
  deps={c['connections'][pin][0] for pin in p['vars']};unknown[b]=len(deps-set(a))
  if not unknown[b]:ready.append(b)
  for x in deps-set(a):users[x].append(b)
 while ready:
  b=ready.popleft();name,c,p=drivers[b];deps=[(pin,c['connections'][pin][0]) for pin in p['vars']]
  val,pin,x=max(((a[x]+f32(p['pin_delay'].get(pin,0.)),pin,x) for pin,x in deps),default=(0.,None,None))
  a[b]=val;depth[b]=1+max((depth[x] for _,x in deps),default=0);previous[b]={'cell':name,'type':c['type'],'output_bit':b,'input_bit':x,'input_pin':pin,'arc_delay':f32(p['pin_delay'].get(pin,0.)),'arrival':val};order.append(b)
  for y in users[b]:
   unknown[y]-=1
   if not unknown[y]:ready.append(y)
 assert len(order)==len(drivers),'cycle or undriven net'
 outputs=[(a[b],n,b) for n,p in m['ports'].items() if p['direction']=='output' for b in p['bits']];delay,endpoint,b=max(outputs)
 assert abs(area-measured['area'])<1e-5,(area,measured['area']);assert abs(delay-measured['delay'])<1e-5,(delay,measured['delay'])
 pathcells=[]
 while previous.get(b,{}).get('cell'):
  step=previous[b];pathcells.append(step);b=step['input_bit']
 pathcells.reverse();req=defaultdict(lambda:math.inf)
 for _,_,b in outputs:req[b]=delay
 for b in reversed(order):
  _,c,p=drivers[b]
  for pin in p['vars']:
   x=c['connections'][pin][0];req[x]=min(req[x],req[b]-f32(p['pin_delay'].get(pin,0.)))
 critical=[b for b in order if req[b]-a[b]<=1e-5];near=[b for b in order if req[b]-a[b]<=.02*delay+1e-5]
 return {'area':area,'delay':delay,'endpoint':endpoint,'critical_path_cells':len(pathcells),'maximum_logic_depth':max((depth[b] for _,_,b in outputs),default=0),'mean_critical_arc_delay':delay/len(pathcells) if pathcells else 0.,'critical_path_types':dict(Counter(c['type'] for c in pathcells)),'critical_path_pin_arcs':dict(Counter(c['type']+':'+str(c['input_pin']) for c in pathcells)),'all_cell_types':dict(Counter(c['type'] for c in m['cells'].values())),'zero_slack_outputs':len(critical),'two_percent_slack_outputs':len(near),'path':pathcells,'sources':{str(p):sha(p) for p in [path,library]}}
def main():
 out=ROOT/'results_v2/delay_cause_analysis_20261015';out.mkdir(exist_ok=True);campaign=json.loads((ROOT/'results_v2/compound_cap_epfl_full_20261014/campaign.json').read_text());bc=json.loads((ROOT/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/campaign.json').read_text());regional=json.loads((ROOT/'results_v2/actual_area_recovery_full_20261014/report.json').read_text());records=[];failures=[];seeds=[]
 for row in regional['results']:
  k,bench=row['k'],row['bench']
  if not(row['status']=='diagnostic_cec_ok' and row['formal_same_import_budget_eligible']):continue
  j=next(j for j in campaign['jobs'] if j['k']==k and j['bench']==bench and j['profile']=='control');report=Path(j['report']);r=json.loads(report.read_text());mr=r['results'][0];folder=report.parent
  jobs=next(j for j in bc['jobs'] if j['name']==f'{k}in_core_{bench}');br=Path(jobs['report']);base=json.loads(br.read_text())
  specimens=[('production',folder/f'{bench}.{mr["method"]}/mapped.json',folder/'library.json',mr['measured']),('regional_area',ROOT/f'results_v2/actual_area_recovery_full_20261014/{k}in_{bench}_r0/mapped.json',folder/'library.json',row['measured'])]
  for method in ['abc_map_delay_resyn2','mockturtle_emap_delay_resyn2']:
   b=next(x for x in base['results'] if x['method']==method)
   if b['status']=='ok':
    assert b['input_sha256']==row['input_sha256'] and b['preprocessing']['sha256']==row['prepared_sha256'];assert base['library_sha256']==row['library_sha256'];specimens.append((method,br.parent/f'{bench}.{method}/mapped.json',br.parent/'library.json',b['measured']))
  n=mr['native'];seeds.append({'k':k,'bench':bench,'initial_delay':n.get('initial_delay'),'seed_delay':n.get('seed_delay'),'final_delay':n['delay'],'refinement_trials':n.get('refinement_trials'),'refinement_accepted':n.get('refinement_accepted'),'phase_trials':n.get('phase_trials'),'phase_accepted':n.get('phase_accepted'),'slack_attempts':n.get('slack_attempts'),'heuristic_secs':n.get('heuristic_secs'),'settings':mr['settings'],'regional_initial_delay':row['budget']['fixed_delay_bound'],'regional_final_delay':row['measured']['delay'],'regional_area_trials':row['budget']['area_trials'],'regional_area_accepts':row['budget']['area_accepts']})
  for label,net,lib,measured in specimens:
   try:
    assert measured['cec']==1;rec=reconstruct(net,lib,measured);rec.update(k=k,bench=bench,profile=label,input_sha256=row['input_sha256'],prepared_sha256=row['prepared_sha256'],library_sha256=row['library_sha256']);records.append(rec)
   except Exception as e:failures.append({'k':k,'bench':bench,'profile':label,'error':repr(e)})
  print(k,bench,flush=True)
 data={'scope':'all 30 prior-import-eligible cases, frozen first full run; selected cover descriptions are not candidate absence or causal proof','records':records,'errors':failures,'baseline_algorithms_changed':False,'new_mapping_runs':0,'script_sha256':sha(Path(__file__))};(out/'paths.json').write_text(json.dumps(data,indent=2));(out/'seed_stage_stats.json').write_text(json.dumps(seeds,indent=2));print(len(records),'reproduced metrics;',len(failures),'errors')
if __name__=='__main__':main()
