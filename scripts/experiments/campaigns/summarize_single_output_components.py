#!/usr/bin/env python3
"""Summarize fixed-factor experiments without selecting per-design winners."""
import argparse,collections,copy,csv,hashlib,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3]
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def outcome(a,b):
 t=1e-6*max(1,abs(a),abs(b));return 'better' if a<b-t else 'worse' if a>b+t else 'equal'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--studies',nargs='+',type=Path,required=True);a=ap.parse_args();O=a.output;O.mkdir(exist_ok=False);studies={}
 for cp in a.studies:
  run=subprocess.run([sys.executable,str(Path(__file__).with_name('check_single_output_components.py')),str(cp)],capture_output=True,text=True);(O/(cp.parent.name+'.log')).write_text(run.stdout+run.stderr);assert run.returncode==0,run.stderr
  v=load(cp.parent/'validation.json');studies[cp.parent.name]=v;save(O/(cp.parent.name+'.json'),v)
 fields=['study','k','bench','profile','valid','reference_area','area','area_delta','reference_delay','delay','delay_delta','area_outcome','delay_outcome']
 with (O/'pairs.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for name,v in studies.items():
   for p in v['pairs']:
    d={k:p.get(k) for k in ['k','bench','profile','valid','area_delta','delay_delta','area_outcome','delay_outcome']};d.update(study=name,reference_area=p['reference'].get('area'),area=p['candidate'].get('area'),reference_delay=p['reference'].get('delay'),delay=p['candidate'].get('delay'));w.writerow(d)
 q={name:{k:v[k] for k in ['complete','attempts','expected','cec','summary','reference','coverage','pair_coverage']} for name,v in studies.items()};q.update(main_promoted=False,baseline_algorithms_changed=False,unconnected=['MtM'],area_delay_objective_not_optimized=True)
 # Gate the protected profile of a complete pairwise full20 study only.
 full=[v for v in studies.values() if v['expected']==320 and v['reference']=='protected_area' and v['complete']] or [v for v in studies.values() if v['expected']==80 and v['reference']=='legacy_control' and v['complete']]
 gates=[]
 if full:
  v=full[0];G=O/'baseline_scope';G.mkdir();required=['abc_map_area_resyn2','abc_map_delay_resyn2','mockturtle_emap_area_resyn2','mockturtle_emap_delay_resyn2','eqmap_lut_amap','esyn_frozen_model_amap_bounded_dag','emorphic_final_sa']
  for k in [3,4]:
   base=R/f'results_v2/guarded_joint_evidence_20261016/baseline_scope/baselines_{k}in.json';d=copy.deepcopy(load(base));d['results']=[]
   for row in v['rows']:
    if row['k']==k and row['profile']=='protected_area':
     r=load(row['report'])['results'][0];r['method']='egraph_single_output_components_proto';d['results'].append(r)
   assert len(d['results'])==20;candidate=G/f'candidate_{k}in.json';save(candidate,d)
   for obj in ['delay-area','area-delay','pareto']:
    dest=G/f'{k}in_{obj}.json';argv=[sys.executable,str(R/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(candidate),'--baselines',str(base),'--required-baselines',*required,'--objective',obj,'--output',str(dest)];run=subprocess.run(argv,capture_output=True,text=True);dest.with_suffix('.log').write_text(run.stdout+run.stderr);gates.append(dict(k=k,objective=obj,passed=load(dest)['all_passed'],argv=argv,exit_code=run.returncode,baseline_sha256=sha(base)))
 q['gate_reference']='Uniform full-method component reference (tail0) if full320 available; rejected tail1 guard otherwise. No per-design profile selection.'
 activations={}
 for name,v in studies.items():
  refs=[r for r in v['rows'] if r['profile']==v['reference'] and r['valid']]
  activations[name]={k:sum((r.get('native') or {}).get(k,0)>0 for r in refs) for k in ['phase_trials','phase_accepted','dc_with_unreachable','dc_bindings','slack_attempts','slack_accepted','slack_local_scored','slack_local_accepted','online_area_updates']}
 q['reference_mechanism_active_case_counts']=activations
 q['activation_caveat']='Generated bindings and accepted/intermediate updates are not proof that those bindings or covers were selected in the final returned implementation.'
 q['metadata_erratum']='V28 component controller frozen descriptive scope inherited a V27 label; authoritative V28 source/binary fingerprint and actual argv are unchanged, external scope_correction.json records this. All component runs tail0.'
 q['gates']=gates
 if full:
  baseline=load(R/'results_v2/epfl_original_baselines_complete_v2_20261013/all_baselines.json');bs=[]
  for r in full[0]['rows']:
   if r['profile']!='protected_area' or not r['valid']:continue
   for b in baseline:
    if (b['k'],b['bench'],b['status'])==(r['k'],r['bench'],'ok') and b['method'] in ['abc_map_delay_resyn2','mockturtle_emap_delay_resyn2']:
     bs.append(dict(k=r['k'],bench=r['bench'],method=b['method'],area_outcome=outcome(r['area'],b['area']),delay_outcome=outcome(r['delay'],b['delay']),candidate=r,baseline=b))
  historical=load(R/'results_v2/single_output_area_full_20261017/validation.json')['rows'];hp=[]
  for r in full[0]['rows']:
   if r['profile']!='protected_area' or not r['valid']:continue
   for old in historical:
    if old['k']==r['k'] and old['bench']==r['bench'] and old['valid']:
     hp.append(dict(k=r['k'],bench=r['bench'],reference_profile=old['profile'],candidate=r,reference=old,area_outcome=outcome(r['area'],old['area']),delay_outcome=outcome(r['delay'],old['delay'])))
  save(O/'historical_pairs.json',hp)
  q['historical_reference_caveat']='Different frozen binary/run; historical comparison is not a paired independent factor or a guarantee. All same-delay losses retained.'
  q['historical_reference']={m:{metric:dict(collections.Counter(b[metric] for b in hp if b['reference_profile']==m)) for metric in ['area_outcome','delay_outcome']} for m in ['legacy_control','protected_area']}
  save(O/'baseline_pairs.json',bs)
  q['baseline_common_legal']={m:{metric:dict(collections.Counter(b[metric] for b in bs if b['method']==m)) for metric in ['area_outcome','delay_outcome']} for m in sorted({b['method'] for b in bs})}
 save(O/'qualification.json',q)
 (O/'report.md').write_text('# Single-output independent component studies\n\nUniform configurations, same once-resyn2 EPFL input and original single-output libraries; cumulative20k nodes/cuts16/retainedpool100k/SAT16/cutrounds32/inclusive recovery2s/native60/mapping300 unchanged. Original baselines unchanged. No per-design profile or best-repeat selection. A tail-pass ablation does not ablate all sharing-aware seeds/local recovery. Witness-optional-cut restriction keeps identical saturation and mandatory source fallbacks but changes optional candidate coverage/ranking; not a no-egraph/no-saturation baseline. The2s timer is post-admission frozen-pool recovery; existing pre-admission finite/frontier ranking uses original5s subguards within native60/mapping300 and is not included in that2s timer. Do not call it whole-mapper/all-cover-work2s. Input-phase matching and atomic phase recovery are distinct factors. Phase/pin restriction probes remove legal matches as well as optimization choices; they are not function-preserving one-legal-assignment timing-only controls. NP matching itself is established, not a claimed new primitive. Module removal can change time available to subsequent passes and the recovery trajectory; counts are whole-pipeline effects, not isolated intrinsic module quality or statistical significance. SDC/phase/pin factors also affect admission under the same cap. AD gates are posthoc. Failed import prefixes remain failures.\n\n```json\n'+json.dumps(q,indent=2)+'\n```\n')
 print(json.dumps(q,indent=2))
if __name__=='__main__':main()
