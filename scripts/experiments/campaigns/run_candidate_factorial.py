#!/usr/bin/env python3
"""Checkpointed fixed candidate-retention factorial; author baselines untouched."""
import argparse,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
from run_epfl_baselines import EPFL
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-jobs',type=int,default=10);a=ap.parse_args();out=a.output.resolve();out.mkdir(exist_ok=True)
 proto=Path('/tmp/emapper-small-cut-quota-20261014');driver=Path(__file__).with_name('bench_egraph_small_cut.py');wrap=driver.with_name('run_with_mapping_budget.py');subjects=ROOT/'results_v2/epfl_resyn2_subjects_20261012/report.json'
 sources=[proto/'target/release/bench_rs_egraph_map',ROOT/'target/release/bench_rs_netlist_eval',subjects,driver,wrap,Path(__file__).resolve(),driver.with_name('nested_mapping_clock.py'),*sorted((proto/'emapper/src').glob('*.rs')),proto/'src/bin/bench_rs_egraph_map.rs',ROOT/'scripts/experiments/port_identifiers.py',ROOT/'scripts/experiments/shared_subject.py',ROOT/'scripts/experiments/benchmark_suites.py',ROOT/'scripts/experiments/restrict_genlib.py',ROOT/'tools/mockturtle/experiments/cell_libraries/asap7.genlib']
 fp={str(p):sha(p) for p in sources};cp=out/'campaign.json'
 c=json.loads(cp.read_text()) if cp.exists() else {'fingerprint':fp,'expected_jobs':160,'objective':'delay-area','fixed_profiles':{'control':[0,0],'timing_cap':[1,0],'small_quota':[0,8],'combined':[1,8]},'jobs':[],'budgets':{'mapping':300,'recovery':2,'nodes':20000,'cuts':16,'pool':100000,'SAT_iterations':16},'scope':'Two inventories/all20 designs/four fixed profiles; no per-design winning profile selection. Diagnostic instrumentation on all profiles.'}
 assert c['fingerprint']==fp,'frozen source drift'
 locks=[]
 for p in [out/'runner.lock',ROOT/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock',ROOT/'results_v2/small_cut_quota_cases_20261014/runner.lock']:
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 for j in c['jobs']:
  for key in ['report','audit','pool','candidates']:
   if key in j:assert sha(Path(j[key]))==j[key+'_sha256']
 if not c['jobs']:
  prior=ROOT/'results_v2/small_cut_quota_cases_20261014/campaign.json';d=json.loads(prior.read_text())
  # Same binary, Rust sources, driver and clock. The only new file is the orchestrator.
  for p,s in d['fingerprint'].items():
   if p==str(ROOT/'scripts/experiments/campaigns/run_small_cut_quota.py'):continue
   assert fp.get(p)==s,(p,'adoption dependency mismatch')
  for j in d['jobs']:
   q=8 if j['name'].endswith('q8') else 0;profile='small_quota' if q else 'control';new=dict(j);new.update(name=f"4in_adder_{profile}",profile=profile,timing_cap=0,small_quota=q,adopted_from=str(prior),adopted_campaign_sha256=sha(prior),adoption_note='identical executed source/binary/driver/wrapper/input; orchestration only')
   for key in ['report','audit','pool','candidates']:
    if key in new:assert sha(Path(new[key]))==new[key+'_sha256']
   c['jobs'].append(new)
  cp.write_text(json.dumps(c,indent=2)+'\n')
 done={j['name'] for j in c['jobs']};count=0
 for k in [4,3]:
  for bench in ['adder','max','sin',*[b for b in EPFL if b not in ['adder','max','sin']]]:
   for profile,(cap,quota) in c['fixed_profiles'].items():
    name=f'{k}in_{bench}_{profile}'
    if name in done:continue
    if count>=a.max_jobs:c['status']='checkpointed';cp.write_text(json.dumps(c,indent=2)+'\n');return
    assert all(sha(Path(p))==s for p,s in fp.items());folder=out/name;audit=out/(name+'.audit.json');assert not folder.exists(),'Unrecorded partial requires independent audit; not overwritten'
    args=['--mapper-root',str(proto),'--k',str(k),'--objective','delay-area','--methods','egraph_bounded','--coupled-event-flow','1','--phase-recovery','1','--phase-time-fraction','0.25','--region-classes','256','--library-timing','0','--timing-cap',str(cap),'--small-cut-quota',str(quota),'--output',str(folder),'--prepared-report',str(subjects),'--suite','epfl','--benches',bench]
    argv=[sys.executable,str(wrap),'--mapping-seconds','300','--driver',str(driver),'--audit',str(audit),'--',*args]
    with (out/(name+'.log')).open('w') as log:r=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    j={'name':name,'k':k,'bench':bench,'profile':profile,'timing_cap':cap,'small_quota':quota,'argv':argv,'exit_code':r.returncode}
    for key,p in [('report',folder/'report.json'),('audit',audit)]:
     if p.exists():j[key]=str(p);j[key+'_sha256']=sha(p)
    if 'report' in j:
     for row in json.loads(Path(j['report']).read_text())['results']:
      for key,env in [('pool','EGM_POOL_DIAG_PATH'),('candidates','EGM_DIAG_PATH')]:
       p=Path(row.get('settings',{}).get(env,'/nonexistent'))
       if p.is_file():j[key]=str(p);j[key+'_sha256']=sha(p)
    c['jobs'].append(j);cp.write_text(json.dumps(c,indent=2)+'\n');count+=1;print(name,r.returncode,flush=True)
 c['status']='attempted';cp.write_text(json.dumps(c,indent=2)+'\n')
if __name__=='__main__':main()
