#!/usr/bin/env python3
"""Audit V16 uniform all-EPFL factor and targeted alternating-order repetitions."""
import argparse,hashlib,json,copy,subprocess,sys
from pathlib import Path
from audit_live_area_commands import audit
R=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();O=a.output;O.mkdir(exist_ok=False)
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def counts(values):return dict(better=sum(v<-1e-6 for v in values),equal=sum(abs(v)<=1e-6 for v in values),worse=sum(v>1e-6 for v in values))
data={};proofs={};studies={}
for name,folder,checker in [('full','delta_area_full_20261016','check_delta_area_full.py'),('repeats','delta_area_repeats_20261016','check_delta_area_repeats.py')]:
 cp=R/'results_v2'/folder/'campaign.json';c=load(cp);out=O/(name+'_validation.json');argv=[sys.executable,str(Path(__file__).with_name(checker)),'--campaign',str(cp),'--output',str(out)];run=subprocess.run(argv,capture_output=True,text=True);out.with_suffix('.log').write_text(run.stdout+run.stderr);assert run.returncode==0,(run.stdout,run.stderr)
 rows=[]
 for j in c['jobs']:
  r=load(j['report'])['results'][0];proofs[name+'_'+j['name']]=audit(j,c)
  trace=load(j['trace']) if 'trace' in j else None
  rows.append(dict(name=j['name'],k=j['k'],bench=j['bench'],profile=j['profile'],row=r,pool_sha256=j.get('pool_sha256'),delta_events=[] if trace is None else [e for e in trace['events'] if e['stage']=='delta_area_filter']))
 data[name]=rows;save(O/(name+'_rows.json'),rows);studies[name]=dict(campaign=str(cp),campaign_sha256=sha(cp),attempts=len(rows),CEC=sum(r['row']['status']=='ok' for r in rows),failures=sum(r['row']['status']!='ok' for r in rows))
pairs=load(O/'full_validation.json')['pairs']
for x in pairs:
 if 'control_area' in x:x.update(area_delta=x['candidate_area']-x['control_area'],delay_delta=x['candidate_delay']-x['control_delay'])
save(O/'full_pairs.json',pairs)
repeatpairs=load(O/'repeats_validation.json')['pairs']
for x in repeatpairs:
 if 'control_area' in x:x.update(area_delta=x['candidate_area']-x['control_area'],delay_delta=x['candidate_delay']-x['control_delay'])
save(O/'repeat_pairs.json',repeatpairs)
G=O/'baseline_scope';G.mkdir();required=['abc_map_area_resyn2','abc_map_delay_resyn2','mockturtle_emap_area_resyn2','mockturtle_emap_delay_resyn2','eqmap_lut_amap','esyn_frozen_model_amap_bounded_dag','emorphic_final_sa'];gates=[]
for k in [3,4]:
 base=R/f'results_v2/guarded_joint_evidence_20261016/baseline_scope/baselines_{k}in.json';d=copy.deepcopy(load(base));d['results']=[dict(copy.deepcopy(i['row']),method='egraph_delta_area_proto') for i in data['full'] if i['k']==k and i['profile']=='protected_area'];assert len(d['results'])==20;d['objective']='delay-area';d['area_delay_check_is_posthoc']=True;candidate=G/f'candidate_{k}in.json';save(candidate,d)
 for obj in ['delay-area','area-delay','pareto']:
  dest=G/f'{k}in_{obj}.json';argv=[sys.executable,str(R/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(candidate),'--baselines',str(base),'--required-baselines',*required,'--objective',obj,'--output',str(dest)];run=subprocess.run(argv,capture_output=True,text=True);dest.with_suffix('.log').write_text(run.stdout+run.stderr);gates.append(dict(k=k,objective=obj,all_passed=load(dest)['all_passed'],argv=argv,exit_code=run.returncode,baseline_sha256=sha(base)))
old=load(R/'results_v2/scored_main_full_report_20261015/all_rows.json');ref={(x['k'],x['bench']):x['row'] for x in old if x['profile']=='protected_area'};mainpairs=[]
for r in data['full']:
 if r['profile']!='protected_area':continue
 m=ref[(r['k'],r['bench'])];n=r['row'];v=dict(k=r['k'],bench=r['bench'],main_status=m['status'],candidate_status=n['status'])
 if m['status']==n['status']=='ok':v.update(area_delta=n['measured']['area']-m['measured']['area'],delay_delta=n['measured']['delay']-m['measured']['delay'])
 mainpairs.append(v)
save(O/'main_pairs.json',mainpairs);save(O/'independent_commands.json',proofs)
reference=R/'results_v2/guarded_joint_evidence_20261016/baseline_integrity.json';check=load(reference);assert check['checked_files']==13403 and not check['drift'];save(O/'baseline_audit_reference.json',dict(path=str(reference),sha256=sha(reference),scope='Prior zero-drift audit reference only, not a fresh baseline audit. Original baseline algorithms/params unchanged.'))
save(G/'validation.json',dict(gates=gates,unconnected=['MtM'],original_status_slots=520,baseline_algorithms_changed=False))
def summary(ps):return {metric:counts([x[metric+'_delta'] for x in ps if metric+'_delta' in x]) for metric in ['area','delay']}
result=dict(studies=studies,full=summary(pairs),targeted_repeats=summary(repeatpairs),main_common_legal=summary(mainpairs),baseline_gates=[g['all_passed'] for g in gates],promoted=False,reason='Full physical timing acceptance unchanged. Local filter correctness and targeted repetitions do not establish overall baseline dominance or solve failed imports. Source proof study is a separate version, not merged.')
save(O/'qualification.json',result)
(O/'report.md').write_text('# V16 exact physical-area rejection filter\n\n'+json.dumps(result,indent=2)+'\n\nEvery failed attempt retained. Same prepared inputs, libraries, actual 20k egraph nodes, cuts16, pool100k, SAT16 and inclusive2s recovery; native60 and mapping300 guards unchanged. CEC and public evaluator excluded from mapping clock. Six gates include full20 missing/failures; area-delay is posthoc. Baseline algorithms untouched; main not promoted. Repeated sqrt observations are targeted, not a full-suite replication. Proof-source V17 is independent.\n')
print(json.dumps(result))
