#!/usr/bin/env python3
"""Frozen, global-budget native region/phase mapper and same-binary controls."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from run_epfl_baselines import ROOT, EPFL, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mapper-root', type=Path, required=True)
    ap.add_argument('--subjects', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--timing-profile', action='store_true')
    ap.add_argument('--objectives', nargs='+', choices=['area-delay','delay-area'], default=['area-delay','delay-area'])
    ap.add_argument('--max-jobs', type=int, default=2)
    a = ap.parse_args()
    if a.max_jobs < 1: ap.error('max-jobs must be positive')
    out, mapper, subjects = a.output.resolve(), a.mapper_root.resolve(), a.subjects.resolve()
    out.mkdir(parents=True, exist_ok=True)
    handles=[]
    for p in [out/'runner.lock', ROOT/'results_v2/epfl_mapping_policy_checkpoint_20261013/runner.lock',
              ROOT/'results_v2/epfl_strict_main_checkpoint_20261013/runner.lock',
              ROOT/'results_v2/epfl_baseline_checkpoint_20261013/runner.lock']:
        f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f)
    prep=json.loads(subjects.read_text())
    if [r['bench'] for r in prep['results']] != list(EPFL) or not all(r['preprocessing']['cec']==1 for r in prep['results']):
        raise ValueError('requires all20 common once-resyn2 subjects')
    files=[Path(__file__).resolve(),Path(__file__).with_name('bench_egraph_regions.py'),
           Path(__file__).with_name('run_with_mapping_budget.py'),Path(__file__).with_name('nested_mapping_clock.py'),
           subjects,mapper/'target/release/bench_rs_egraph_map',mapper/'src/bin/bench_rs_egraph_map.rs',
           ROOT/'target/release/bench_rs_netlist_eval',ROOT/'tools/mockturtle/experiments/cell_libraries/asap7.genlib',
           Path('/usr/bin/abc'),Path('/usr/local/bin/yosys'),*sorted((mapper/'emapper/src').rglob('*.rs')),
           *sorted((ROOT/'scripts/experiments').glob('*.py'))]
    frozen={str(p):sha(p) for p in files};sp=out/'campaign.json'
    state=json.loads(sp.read_text()) if sp.exists() else {'expected_jobs':2*len(a.objectives)*20*2,'objectives':a.objectives,'jobs':[],
        'source_sha256':frozen,'subjects':str(subjects),'mapper_root':str(mapper),'mapping_seconds':300,
        'profiles':{'region_phase':{'region_classes':256,'phase_recovery':1,'library_timing':int(a.timing_profile),'timing_cap':int(a.timing_profile)},'control':{'region_classes':0,'phase_recovery':0,'library_timing':0,'timing_cap':0}},
        'scope':'one global egraph; regional physical boundary ownership; no per-region node/iteration/pool/time reset; no baseline changes'}
    if state['source_sha256']!=frozen: raise ValueError('source/binary changed: new explicit matrix required')
    for j in state['jobs']:
        for key in ['report','audit']:
            if key in j and sha(Path(j[key]))!=j[key+'_sha256']:raise ValueError('completed observation changed')
    def save():
        tmp=sp.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(sp)
    count=0
    for k in [3,4]:
      for objective in state['objectives']:
       for bench in EPFL:
        for profile,config in state['profiles'].items():
          name=f'{k}in_{objective}_{bench}_{profile}'
          if any(j['name']==name for j in state['jobs']):continue
          folder=out/name;audit=out/(name+'.audit.json');log=out/(name+'.log')
          if any(p.exists() for p in [folder,audit,log]):raise ValueError('preserve and audit interrupted artifacts: '+name)
          cmd=[sys.executable,str(Path(__file__).with_name('run_with_mapping_budget.py')),'--mapping-seconds','300',
               '--driver',str(Path(__file__).with_name('bench_egraph_regions.py')),'--audit',str(audit),'--',
               '--mapper-root',str(mapper),'--region-classes',str(config['region_classes']),
               '--phase-recovery',str(config['phase_recovery']),'--library-timing',str(config['library_timing']),
               '--timing-cap',str(config['timing_cap']),'--output',str(folder),'--k',str(k),
               '--objective',objective,'--preprocess','resyn2','--prepared-report',str(subjects),
               '--methods','egraph_bounded','--suite','epfl','--benches',bench]
          state['current_job']={'name':name,'argv':cmd};save()
          with log.open('w') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
          j={'name':name,'bench':bench,'k':k,'objective':objective,'profile':profile,'argv':cmd,'exit_code':r.returncode}
          for key,p in [('report',folder/'report.json'),('audit',audit)]:
              if p.exists():j.update({key:str(p),key+'_sha256':sha(p)})
          if 'report' in j:
              rows=json.loads(Path(j['report']).read_text())['results'];j['ok']=sum(x['status']=='ok' for x in rows);j['failed']=sum(x['status']!='ok' for x in rows)
          else:j.update(ok=0,failed=1,error='no report; log retained')
          state['jobs'].append(j);state.pop('current_job',None);save();print(name,j['ok'],j['failed'],flush=True)
          count+=1
          if count>=a.max_jobs:return
    state['status']='all_attempted';save()

if __name__=='__main__':main()
