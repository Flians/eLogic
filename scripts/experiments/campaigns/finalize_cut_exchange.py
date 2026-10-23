#!/usr/bin/env python3
"""Full uniform V23 evidence, targeted reverse-order repeat, independent V22 source study."""
import argparse,collections,copy,hashlib,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3]
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def outcome(a,b):
 t=1e-6*max(1.,abs(a),abs(b));return 'better' if a<b-t else 'worse' if a>b+t else 'equal'
def summary(pairs):
 return {m:dict(collections.Counter(outcome(p['candidate'][m],p['control'][m]) for p in pairs if p['candidate']['valid'] and p['control']['valid'])) for m in ['delay','area']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();O=a.output;O.mkdir(exist_ok=False)
 studies={};vs={}
 for key,folder in [('scalar_pilot','cut_exchange_pilot_20261017'),('full','phase_cut_exchange_full_20261017'),('repeat','phase_cut_exchange_repeat_20261017')]:
  cp=R/'results_v2'/folder/'campaign.json';cmd=[sys.executable,str(Path(__file__).with_name('check_cut_exchange.py')),str(cp)];run=subprocess.run(cmd,capture_output=True,text=True);(O/(key+'.audit.log')).write_text(run.stdout+run.stderr);assert run.returncode==0,run.stderr
  v=load(cp.parent/'validation.json');assert v['complete'];vs[key]=v;save(O/(key+'_validation.json'),v);studies[key]=dict(campaign=str(cp),sha256=sha(cp),attempts=v['attempts'],CEC=v['cec'],summary=summary(v['pairs']))
 source=R/'results_v2/source_propagation_pilot_20261017/validation.json';sv=load(source);assert sv['attempts']==16;save(O/'source_validation.json',sv)
 records=[]
 for r in vs['full']['rows']:records.append(dict(k=r['k'],bench=r['bench'],profile=r['profile'],row=load(r['report'])['results'][0]))
 save(O/'all_rows.json',records)
 main=load(R/'results_v2/scored_main_full_report_20261015/all_rows.json');old={(r['k'],r['bench']):r['row'] for r in main if r['profile']=='protected_area'};mp=[]
 for r in vs['full']['rows']:
  if r['profile']!='protected_area':continue
  b=old[(r['k'],r['bench'])];mp.append(dict(k=r['k'],bench=r['bench'],control=dict(valid=b['status']=='ok',area=b.get('measured',{}).get('area'),delay=b.get('measured',{}).get('delay')),candidate=r))
 save(O/'main_pairs.json',mp)
 sameDreg=[dict(k=p['k'],bench=p['bench'],area_delta=p['candidate']['area']-p['control']['area']) for p in mp if p['candidate']['valid'] and p['control']['valid'] and outcome(p['candidate']['delay'],p['control']['delay'])=='equal' and outcome(p['candidate']['area'],p['control']['area'])=='worse']
 G=O/'baseline_scope';G.mkdir();gates=[];required=['abc_map_area_resyn2','abc_map_delay_resyn2','mockturtle_emap_area_resyn2','mockturtle_emap_delay_resyn2','eqmap_lut_amap','esyn_frozen_model_amap_bounded_dag','emorphic_final_sa']
 for k in [3,4]:
  base=R/f'results_v2/pin_cut_evidence_20261016/baseline_scope/baselines_{k}in.json'
  # Existing evidence references baselines in the parent evidence directory.
  if not base.exists():base=R/f'results_v2/guarded_joint_evidence_20261016/baseline_scope/baselines_{k}in.json'
  d=copy.deepcopy(load(base));d['results']=[dict(copy.deepcopy(r['row']),method='egraph_phase_cut_exchange_proto') for r in records if r['k']==k and r['profile']=='protected_area'];assert len(d['results'])==20;d['objective']='delay-area';d['area_delay_check_is_posthoc']=True;candidate=G/f'candidate_{k}in.json';save(candidate,d)
  for objective in ['delay-area','area-delay','pareto']:
   dest=G/f'{k}in_{objective}.json';argv=[sys.executable,str(R/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(candidate),'--baselines',str(base),'--required-baselines',*required,'--objective',objective,'--output',str(dest)];p=subprocess.run(argv,capture_output=True,text=True);dest.with_suffix('.log').write_text(p.stdout+p.stderr);gates.append(dict(k=k,objective=objective,passed=load(dest)['all_passed'],argv=argv,exit_code=p.returncode,baseline_sha256=sha(base)))
 q=dict(studies=studies,source=dict(validation=str(source),sha256=sha(source),attempts=sv['attempts'],CEC=sv['CEC'],failures=sv['failures']),main_common_legal=summary(mp),same_delay_main_area_regressions=sameDreg,gates=gates,main_promoted=False,baseline_algorithms_changed=False,unconnected=['MtM'],physical_attempts=sum(s['attempts'] for s in studies.values())+sv['attempts'],physical_CEC=sum(s['CEC'] for s in studies.values())+sv['CEC'],scope='V23 full80 includes12 identical-FP pilot observations counted once. Reverse targeted repeat is not full-suite replication. V21 and V22 independent sources, never combined by design or configuration. Both local cut guards have sin delay counterexamples, hence no promotion. Gates area-delay are posthoc, not AD optimization. Source-only CEC is distinct from physical mapping CEC.')
 save(O/'qualification.json',q)
 (O/'report.md').write_text('# Cut exchange and exact source propagation\n\n'+json.dumps(q,indent=2)+'\n\nSame once-resyn2 EPFL20 inputs and restricted ASAP7 inventories; public unique physical instance area/unloaded dependent-pin delay/original named-portCEC. Actual cumulative20keNodes/cuts16/retained100k/SAT16/inclusive recovery2s/native60/mapping300 unchanged. No baseline algorithm changes, no per-design configuration winners. Transient ranking/proof state is not a RAM-cap or free hidden function claim. Original baseline240groups/520slots/372CEC/124reportedfailure/24unmeasured retained. Earlier13403-file baseline audit referenced, not freshly repeated. Main8f6 and reference tables unchanged. Cut-key membership does not prove full physical old-cover availability. Next: concrete full-cover cut/instance pinning and incremental live exchanges under the same global deadline, not local-label dominance.\n')
 print(json.dumps(q),flush=True)
if __name__=='__main__':main()
