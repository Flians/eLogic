#!/usr/bin/env python3
"""Frozen, checkpointed one-component removal study; stock baselines untouched."""
import argparse,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
from run_epfl_baselines import EPFL
ROOT=Path(__file__).resolve().parents[3]
PROFILES={
 'full':{},
 'no_functional_cuts':{'--ablate-functional-cuts':'1'},
 'single_witness_cuts':{'--single-structure-cuts':'1'},
 'no_input_phase':{'--no-input-phase-matches':'1'},
 'canonical_pins':{'--canonical-pin-matching':'1'},
 'no_sdc':{'--dc-support':'0'},
 'no_shared_area':{'--ablate-shared-area':'1'},
 'no_incumbent_guard':{'--ablate-incumbent-guard':'1'},
}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(c,path):
 q=path.with_suffix('.tmp');q.write_text(json.dumps(c,indent=2)+'\n');q.replace(path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-jobs',type=int,default=16);ap.add_argument('--designs',nargs='+',default=EPFL);ap.add_argument('--profiles',nargs='+',choices=list(PROFILES),default=list(PROFILES));ap.add_argument('--repetition',type=int,default=-1);a=ap.parse_args();out=a.output.resolve();out.mkdir(exist_ok=True);cp=out/'campaign.json'
 template_path=Path('/tmp/emapper-sparse-cover-full-20261019/campaign.json');template=json.loads(template_path.read_text());driver=Path(__file__).with_name('bench_independent_components.py');wrapper=driver.with_name('run_with_mapping_budget.py')
 sources=set(map(Path,template['fingerprint']));sources.update([driver,Path(__file__).resolve(),template_path]);sources.update(Path(p)for p in json.loads((ROOT/'results_v2/single_output_components_evidence_20261017/integrity.json').read_text())['baseline_small_binary_check']);fp={str(p):sha(p) for p in sorted(sources)}
 assert set(a.designs)<=set(EPFL) and len(set(a.designs))==len(a.designs)
 c=json.loads(cp.read_text()) if cp.exists() else dict(fingerprint=fp,expected_jobs=len(a.designs)*2*len(a.profiles),designs=a.designs,profiles=a.profiles,repetition=a.repetition,objective='delay-area',component_overrides={p:PROFILES[p]for p in a.profiles},jobs=[],budgets=dict(mapping=300,native=60,recovery=2,nodes=20000,cuts=16,pool=100000,SAT_iterations=16,cut_rounds=32,area_rounds=4,area_trials_per_round=2000),scope='One frozen executable, uniform V46 algorithm/configuration plus default-off ablation instrumentation. Full profile chosen before new experiments, not by per-design outcomes. No fresh frontend or additional budget; failures preserved.')
 assert c['fingerprint']==fp and c['designs']==a.designs and c['profiles']==a.profiles and c['repetition']==a.repetition,'frozen experiment drift'
 locks=[]
 for p in [out/'runner.lock',ROOT/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock',ROOT/'results_v2/small_cut_quota_cases_20261014/runner.lock',ROOT/'results_v2/independent_components_20261019/global.lock']:
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 for j in c['jobs']:
  for key in ['report','audit','admission','pool','candidates','trace']:
   if key in j:assert sha(j[key])==j[key+'_sha256']
 done={j['name'] for j in c['jobs']};n=0;save(c,cp)
 libs=[4,3] if a.repetition%2 else [3,4]
 if a.repetition<0:libs=[4,3]
 designs=a.designs if a.repetition%2 else list(reversed(a.designs))
 if a.repetition<0:designs=a.designs
 for k in libs:
  for bi,bench in enumerate(designs):
   order=a.profiles[(bi+max(0,a.repetition))%len(a.profiles):]+a.profiles[:(bi+max(0,a.repetition))%len(a.profiles)]
   if a.repetition%2==0 and a.repetition>=0:order=order[::-1]
   for profile in order:
    name=f'{k}in_{bench}_{profile}'
    if name in done:continue
    if n>=a.max_jobs:c['status']='checkpointed';save(c,cp);return
    assert all(sha(p)==h for p,h in fp.items()),'source drift before mapping'
    folder=out/name;audit=out/(name+'.audit.json');assert not folder.exists() and not audit.exists(),'unrecorded partial requires independent audit, never overwritten'
    old=next(j for j in template['jobs'] if j['k']==k and j['bench']==bench);argv=list(old['argv']);argv[0]=sys.executable;argv[argv.index('--driver')+1]=str(driver);argv[argv.index('--audit')+1]=str(audit);argv[argv.index('--output')+1]=str(folder)
    extra={'--ablate-functional-cuts':'0','--ablate-shared-area':'0','--ablate-incumbent-guard':'0','--single-structure-cuts':'0','--no-input-phase-matches':'0','--canonical-pin-matching':'0','--no-actual-area-tail':'0','--dc-support':'4'};extra.update(PROFILES[profile])
    for flag,value in extra.items():
     if flag in argv:argv[argv.index(flag)+1]=value
     else:argv.extend([flag,value])
    c['in_progress']=dict(name=name,argv=argv,folder=str(folder));save(c,cp)
    with (out/(name+'.log')).open('w') as log:r=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    j=dict(name=name,k=k,bench=bench,profile=profile,argv=argv,exit_code=r.returncode)
    for key,p in [('report',folder/'report.json'),('audit',audit)]:
     if p.exists():j[key]=str(p);j[key+'_sha256']=sha(p)
    if 'report'in j:
     for row in json.loads(Path(j['report']).read_text())['results']:
      for key,env in [('pool','EGM_POOL_DIAG_PATH'),('candidates','EGM_DIAG_PATH'),('trace','EGM_RECOVERY_TRACE')]:
       p=Path(row.get('settings',{}).get(env,'/nonexistent'))
       if p.is_file():j[key]=str(p);j[key+'_sha256']=sha(p)
       if key=='pool' and p.with_suffix('.admission.json').is_file():
        q=p.with_suffix('.admission.json');j['admission']=str(q);j['admission_sha256']=sha(q)
    c['jobs'].append(j);c.pop('in_progress',None);save(c,cp);n+=1;print(name,r.returncode,flush=True)
 c['status']='attempted';save(c,cp)
if __name__=='__main__':main()
