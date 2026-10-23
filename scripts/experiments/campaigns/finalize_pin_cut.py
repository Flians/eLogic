#!/usr/bin/env python3
"""Uniform full-scope and targeted-repeat pin-aware cut ranking evidence; no promotion."""
import argparse,copy,hashlib,json,subprocess,sys,collections
from pathlib import Path
R=Path(__file__).resolve().parents[3];load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def outcome(x,y):
 tol=1e-6*max(1,abs(x),abs(y));return 'better' if x<y-tol else 'worse' if x>y+tol else 'equal'
def summary(ps):
 return {metric:dict(collections.Counter(outcome(p['candidate'][metric],p['control'][metric]) for p in ps if p['candidate']['valid'] and p['control']['valid'])) for metric in ['delay','area']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();O=a.output;O.mkdir(exist_ok=False);studies={};rows={};pairs={}
 for label,folder in [('full','pin_cut_full_20261016'),('repeat1','pin_cut_repeat1_20261016'),('repeat2','pin_cut_repeat2_20261016')]:
  cp=R/'results_v2'/folder/'campaign.json';c=load(cp);cmd=[sys.executable,str(Path(__file__).with_name('check_pin_cut.py')),str(cp)];run=subprocess.run(cmd,capture_output=True,text=True);(O/(label+'.log')).write_text(run.stdout+run.stderr);assert run.returncode==0,run.stderr;v=load(cp.parent/'validation.json');assert v['complete'];save(O/(label+'_validation.json'),v);rows[label]=v['rows'];pairs[label]=v['pairs'];studies[label]=dict(campaign=str(cp),sha256=sha(cp),attempts=v['attempts'],CEC=v['cec'])
 G=O/'baseline_scope';G.mkdir();required=['abc_map_area_resyn2','abc_map_delay_resyn2','mockturtle_emap_area_resyn2','mockturtle_emap_delay_resyn2','eqmap_lut_amap','esyn_frozen_model_amap_bounded_dag','emorphic_final_sa'];gates=[];data=[]
 for row in rows['full']:
  native=load(row['report'])['results'][0];data.append(dict(k=row['k'],bench=row['bench'],profile=row['profile'],row=native))
 save(O/'all_rows.json',data)
 for k in [3,4]:
  base=R/f'results_v2/guarded_joint_evidence_20261016/baseline_scope/baselines_{k}in.json';d=copy.deepcopy(load(base));d['results']=[dict(copy.deepcopy(i['row']),method='egraph_pin_cut_proto') for i in data if i['k']==k and i['profile']=='protected_area'];assert len(d['results'])==20;d['objective']='delay-area';d['area_delay_check_is_posthoc']=True;candidate=G/f'candidate_{k}in.json';save(candidate,d)
  for obj in ['delay-area','area-delay','pareto']:
   dest=G/f'{k}in_{obj}.json';argv=[sys.executable,str(R/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(candidate),'--baselines',str(base),'--required-baselines',*required,'--objective',obj,'--output',str(dest)];run=subprocess.run(argv,capture_output=True,text=True);dest.with_suffix('.log').write_text(run.stdout+run.stderr);gates.append(dict(k=k,objective=obj,passed=load(dest)['all_passed'],argv=argv,exit_code=run.returncode,baseline_sha256=sha(base)))
 main=load(R/'results_v2/scored_main_full_report_20261015/all_rows.json');ref={(r['k'],r['bench']):r['row'] for r in main if r['profile']=='protected_area'};mainpairs=[]
 for r in rows['full']:
  if r['profile']!='protected_area':continue
  old=ref[(r['k'],r['bench'])];control=dict(valid=old['status']=='ok',area=old.get('measured',{}).get('area'),delay=old.get('measured',{}).get('delay'));mainpairs.append(dict(k=r['k'],bench=r['bench'],control=control,candidate=r))
 baseline=load(R/'results_v2/epfl_original_baselines_complete_v2_20261013/all_baselines.json');baselinepairs=[]
 for r in rows['full']:
  if r['profile']!='protected_area' or not r['valid']:continue
  for b in baseline:
   if b['k']==r['k'] and b['bench']==r['bench'] and b['status']=='ok' and b['method'] in ['abc_map_delay_resyn2','mockturtle_emap_delay_resyn2']:
    baselinepairs.append(dict(k=r['k'],bench=r['bench'],method=b['method'],area_outcome=outcome(r['area'],b['area']),delay_outcome=outcome(r['delay'],b['delay']),candidate=r,baseline=b))
 save(O/'baseline_pairs.json',baselinepairs);save(O/'main_pairs.json',mainpairs)
 repeats=pairs['repeat1']+pairs['repeat2'];repeatdetail=[]
 for x in repeats:
  q=dict(x);q.update(delay_outcome=outcome(x['candidate']['delay'],x['control']['delay']) if x['candidate']['valid'] and x['control']['valid'] else 'failed');repeatdetail.append(q)
 save(O/'repeat_pairs.json',repeatdetail)
 record=dict(studies=studies,full=summary(pairs['full']),targeted_repeats=summary(repeats),main_common_legal=summary(mainpairs),gates=gates,promoted=False,unconnected=['MtM'],baseline_algorithms_changed=False,scope='Fixed per-library all20 DA single factor; repeats targeted bar/sin/log2, not whole-suite replication. All failures retained. Area-delay gates are posthoc, not AD-optimized mapper. No best-run or per-design configuration selection.')
 save(O/'qualification.json',record)
 (O/'report.md').write_text('# Pin-aware cut retention\n\n'+json.dumps(record,indent=2)+'\n\nSame original once-resyn2 EPFL subjects, restricted ASAP7 inventories, public physical area/unloaded pin delay/original named-port CEC. Actual cumulative egraph nodes20000, retained cuts16/class, retained eligible physical instances100000, SAT16, inclusive recovery2s/native60/mapping300 unchanged. Generated candidates and transient ranking edges are not a RAM cap. Offline baseline correspondence never enters mapper. Independent per-run provenance and budget checks in validation files. Original baseline520slots/372CEC/124reportedfailure/24unmeasured unchanged; prior13403file audit referenced, not rerun. Main/defaults/reference tables not promoted.\n')
 print(json.dumps(record),flush=True)
if __name__=='__main__':main()
