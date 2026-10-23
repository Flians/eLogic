#!/usr/bin/env python3
"""Reconcile nonoverlapping observations and original frozen-baseline gates; no promotion."""
import argparse,json,hashlib,subprocess,sys,copy
from pathlib import Path
R=Path(__file__).resolve().parents[3];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();O=a.output;O.mkdir(exist_ok=False);load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();save=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n')
full=load(R/'results_v2/helper_phase_full_report_20261016/paired.json');rep=load(R/'results_v2/helper_phase_repeats_report_20261016/paired.json');unique=full+[p for p in rep if p['observation']!='full_r0'];assert len(unique)==76
eligible=[p for p in unique if all(p[k]['row']['status']=='ok' for k in ['control','candidate'])];counts={}
for metric in ['delay','area']:
 d=[p['candidate']['row']['measured'][metric]-p['control']['row']['measured'][metric] for p in eligible];counts[metric]=[sum(x<-1e-6 for x in d),sum(abs(x)<=1e-6 for x in d),sum(x>1e-6 for x in d)]
# Baseline sources were fully reaudited in this round, unchanged since that audit.
old=R/'results_v2/guarded_joint_evidence_20261016';audit=load(old/'baseline_integrity.json');assert audit['checked_files']==13403 and not audit['drift'];save(O/'baseline_audit_reference.json',dict(report=str(old/'baseline_integrity.json'),sha256=sha(old/'baseline_integrity.json'),scope='Reuse same-round13403dependency audit, not a second fresh audit; no baseline native/model/helper edits since audit.'))
rows=load(R/'results_v2/helper_phase_full_report_20261016/all_rows.json');gates=[];G=O/'baseline_scope';G.mkdir()
primary=['abc_map_area_resyn2','abc_map_delay_resyn2','mockturtle_emap_area_resyn2','mockturtle_emap_delay_resyn2','eqmap_lut_amap','esyn_frozen_model_amap_bounded_dag','emorphic_final_sa']
for k in [3,4]:
 b=old/f'baseline_scope/baselines_{k}in.json';base=load(b);cp=G/f'candidate_{k}in.json';d=copy.deepcopy(base);d['results']=[]
 for item in rows:
  if item['k']==k and item['profile']=='protected_area':
   r=copy.deepcopy(item['row']);r['method']='egraph_helper_phase_proto';d['results'].append(r)
 assert len(d['results'])==20;d['objective']='delay-area';d['area_delay_check_is_posthoc']=True;save(cp,d)
 for objective in ['delay-area','area-delay','pareto']:
  out=G/f'{k}in_{objective}.json';argv=[sys.executable,str(R/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(cp),'--baselines',str(b),'--required-baselines',*primary,'--candidate-methods','egraph_helper_phase_proto','--objective',objective,'--output',str(out)];r=subprocess.run(argv,capture_output=True,text=True);out.with_suffix('.log').write_text(r.stdout+r.stderr);gates.append(dict(argv=argv,exit_code=r.returncode,all_passed=load(out)['all_passed'],baseline_source=str(b),baseline_sha256=sha(b)))
save(G/'validation.json',dict(gates=gates,baseline_algorithms_changed=False,original_status_slots=520,unconnected=['MtM']))
canonical=load(R/'results_v2/scored_main_full_report_20261015/all_rows.json');canon={(r['k'],r['bench']):r['row']['measured'] for r in canonical if r['profile']=='protected_area' and r['row']['status']=='ok'};adverse=[]
for r in rows:
 if r['profile']=='protected_area' and r['row']['status']=='ok' and (r['k'],r['bench']) in canon:
  m=r['row']['measured'];oldm=canon[(r['k'],r['bench'])];dd=m['delay']-oldm['delay'];da=m['area']-oldm['area']
  if dd>1e-6 or da>1e-6:adverse.append(dict(k=r['k'],bench=r['bench'],delay_delta=dd,area_delta=da,tied_delay_area_regression=abs(dd)<=1e-6 and da>1e-6))
save(O/'qualification.json',dict(unique_fresh_native_attempts=152,unique_direct_cec=sum(p[k]["row"]["status"]=="ok" for p in unique for k in ["control","candidate"]),unique_eligible_pairs=len(eligible),unique_better_equal_worse=counts,canonical_main_adverse_metrics=adverse,baseline_gate_results=[g['all_passed'] for g in gates],production_qualified=False,production_promoted=False,reason='Area costs/remaining import failures retained; no universal preservation proof for previous-final-shared-cover across changed pools. Main source/defaults/tables unchanged; delay-area lex advantage is not Pareto dominance.'))
print('unique',counts,'gates',[g['all_passed'] for g in gates],'canonical adverse',len(adverse))
