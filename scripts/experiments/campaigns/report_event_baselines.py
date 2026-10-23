#!/usr/bin/env python3
"""Full-scope EPFL baseline gates, including failures and LUT staging time bounds."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from run_epfl_baselines import ROOT, EPFL, sha
PRIMARY=['abc_map_area_resyn2','abc_map_delay_resyn2','mockturtle_emap_area_resyn2','mockturtle_emap_delay_resyn2','eqmap_lut_amap','esyn_frozen_model_amap_bounded_dag','emorphic_final_sa']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate-profile',choices=['event','coupled_queue','coupled_quarter','coupled_phase0'],required=True);ap.add_argument('--candidate-campaign',type=Path,required=True);ap.add_argument('--baseline-campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(exist_ok=False);candidate=json.loads(a.candidate_campaign.read_text());baseline=json.loads(a.baseline_campaign.read_text());subjects={r['bench']:r for r in json.loads(Path(candidate['subjects']).read_text())['results']}
    baserows=[];candrows=[];hashes={str(p.resolve()):sha(p) for p in [a.candidate_campaign,a.baseline_campaign]};time_bounds=[]
    for c,target in [(candidate,candrows),(baseline,baserows)]:
      for j in c['jobs']:
        for key in ['report','audit']:
          if key in j:
            assert sha(Path(j[key]))==j[key+'_sha256'];hashes[j[key]]=j[key+'_sha256']
        if target is candrows and j['profile']!=a.candidate_profile:continue
        k=int(j.get('k',int(j['name'][0])));goal=j.get('objective')
        if not goal:
          goal=next((x for x in ['area-delay','delay-area'] if x in j['name']),None)
        if 'report' not in j:
          method='eqmap_lut_amap' if '_eqmap_' in j['name'] else 'esyn_frozen_model_amap_bounded_dag' if '_esyn_' in j['name'] else 'emorphic_final_sa'
          target.append((k,goal,{'bench':j['bench'],'method':method,'input_sha256':subjects[j['bench']]['input_sha256'],'status':'failed','error':j.get('error','no report'),'unmeasured':True,'source_job':j['name']}));continue
        d=json.loads(Path(j['report']).read_text())
        for original in d['results']:
          r=copy.deepcopy(original);
          if target is candrows and j.get('execution')=='adopted_pre_stage_failure':
            r['source_observation_method']=r['method'];r['method']='egraph_staged_resyn2' if a.candidate_profile=='event' else 'egraph_staged_control_resyn2';r['method_descriptor_note']='failed import precedes recovery; intended profile descriptor for gate, original failed report and source method retained'
          assert r['input_sha256']==subjects[r['bench']]['input_sha256'];r['observation_report']=j['report'];r['library_sha256']=d['library_sha256']
          if r['status']=='ok':assert r['measured']['cec']==1
          if r['method'] in ['eqmap_lut_amap','lut_stage_amap'] and 'audit' in j:
            audit=json.loads(Path(j['audit']).read_text());enclosing=[]
            for s in audit['states'].values():
              for cmd in s['commands']:
                if Path(cmd['argv'][0]).name=='yosys' and 'abc -lut' in ' '.join(cmd['argv']):enclosing.append(cmd['wall_secs'])
            direct=audit['states'].get(r['bench']+'/'+r['method'],{}).get('mapping_wall_secs',0.)
            upper=direct+sum(enclosing);r['mapping_budget_upper_bound_secs']=upper;r['mapping_time_exact']=False
            r['mapping_time_note']='nested initial Yosys ABC LUT stage not timestamped separately; enclosing Yosys elapsed supplies conservative upper bound, not charged conversion time'
            if r['status']=='ok' and upper>300:r['status']='failed';r['error']='mapping-only300s not proven by conservative bound';r['partial_measured']=r.pop('measured')
            time_bounds.append({'job':j['name'],'method':r['method'],'direct_mapping_secs':direct,'enclosing_lut_stage_secs':sum(enclosing),'mapping_upper_bound_secs':upper,'under300':upper<=300})
          target.append((k,goal,r))
    gates=[]
    for k in [3,4]:
      for objective in ['area-delay','delay-area']:
        cr=[r for lib,goal,r in candrows if lib==k and goal==objective]
        if not cr:continue
        libhash=cr[0]['library_sha256'];model='all physical instances; max(rise,fall) block pin delay; no load or slew'
        missing=lambda bench,method:{'bench':bench,'method':method,'input_sha256':subjects[bench]['input_sha256'],'status':'missing','error':'not attempted'}
        for bench in EPFL:
          if not any(r['bench']==bench for r in cr):cr.append(missing(bench,cr[0]['method']))
        br=[r for lib,goal,r in baserows if lib==k and (goal is None or goal==objective)]
        for r in br:
          if 'library_sha256' in r:assert r['library_sha256']==libhash
        for bench in EPFL:
          for method in PRIMARY:
            if not any(r['bench']==bench and r['method']==method for r in br):br.append(missing(bench,method))
        cp=out/f'candidate_{k}in_{objective}.json';bp=out/f'baselines_{k}in_{objective}.json'
        cp.write_text(json.dumps({'library_sha256':libhash,'metric_model':model,'results':cr},indent=2));bp.write_text(json.dumps({'library_sha256':libhash,'metric_model':model,'results':br},indent=2))
        for check in [objective,'pareto']:
          gp=out/f'{k}in_{objective}_{check}_gate.json';argv=[sys.executable,str(ROOT/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(cp),'--baselines',str(bp),'--required-baselines',*PRIMARY,'--candidate-methods',cr[0]['method'],'--objective',check,'--output',str(gp)]
          p=subprocess.run(argv,capture_output=True,text=True);gp.with_suffix('.log').write_text(p.stdout+p.stderr);result=json.loads(gp.read_text());gates.append({'argv':argv,'exit_code':p.returncode,'all_passed':result['all_passed']})
    summary={'candidate_profile':a.candidate_profile,'candidate_jobs':len(candidate['jobs']),'baseline_jobs':len(baseline['jobs']),'baseline_expected':240,'gates':gates,'hashes':hashes,'eqmap_staging_mapping_time_bounds':time_bounds,'clock_limitation':'initial nested Yosys ABC LUT time has only a parent-process upper bound; exact EqMap mapping runtime is not reported','unconnected':['MtM'],'baseline_algorithms_changed':False}
    (out/'validation.json').write_text(json.dumps(summary,indent=2));print('baseline jobs',summary['baseline_jobs'],'gates',[g['all_passed'] for g in gates])

if __name__=='__main__':main()
