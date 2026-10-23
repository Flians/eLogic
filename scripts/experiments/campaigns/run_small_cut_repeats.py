#!/usr/bin/env python3
"""Finite diagnostic-only pool study, no baseline algorithm or budget changes."""
import argparse,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False)
 proto=Path('/tmp/emapper-small-cut-quota-20261014');driver=Path(__file__).with_name('bench_egraph_small_cut.py');wrap=driver.with_name('run_with_mapping_budget.py');subjects=ROOT/'results_v2/epfl_resyn2_subjects_20261012/report.json'
 paths=[proto/'target/release/bench_rs_egraph_map',ROOT/'target/release/bench_rs_netlist_eval',subjects,driver,wrap,Path(__file__).resolve(),*sorted((proto/'emapper/src').glob('*.rs')),proto/'src/bin/bench_rs_egraph_map.rs',ROOT/'scripts/experiments/port_identifiers.py']
 fingerprints={str(p):sha(p) for p in paths};campaign={'diagnostic_only':True,'interpretation_repeats':'Two additional alternating-order repetitions, plus separately retained r0 from complete factorial; no best-run selection','interpretation':'Local estimates under common area witness, not optimal timing bounds; instrumented observations not substituted into canonical comparisons.','fingerprint':fingerprints,'jobs':[],'expected_jobs':8}
 locks=[]
 for p in [out/'runner.lock',ROOT/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock',ROOT/'results_v2/epfl_event_current_flow_v2_checkpoint_20261013/runner.lock']:
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 for case in [(b,rep,q) for b in ['adder','max'] for rep in [1,2] for q in ([8,0] if rep==1 else [0,8])]:
  bench,rep,quota=case
  k=4
  assert all(sha(Path(p))==s for p,s in fingerprints.items());name=f'{k}in_{bench}_r{rep}_q{quota}';folder=out/name;audit=out/(name+'.audit.json')
  args=['--small-cut-quota',str(quota),'--mapper-root',str(proto),'--k',str(k),'--objective','delay-area','--methods','egraph_bounded','--coupled-event-flow','1','--phase-recovery','1','--phase-time-fraction','0.25','--region-classes','256','--library-timing','0','--timing-cap','0','--output',str(folder),'--prepared-report',str(subjects),'--suite','epfl','--benches',bench]
  argv=[sys.executable,str(wrap),'--mapping-seconds','300','--driver',str(driver),'--audit',str(audit),'--',*args]
  with (out/(name+'.log')).open('w') as log:r=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  job={'name':name,'k':k,'bench':bench,'repetition':rep,'small_quota':quota,'argv':argv,'exit_code':r.returncode}
  for key,p in [('report',folder/'report.json'),('audit',audit),('pool',folder/f'{bench}.egraph_bounded_resyn2/pool-diagnostics.json'),('candidates',folder/f'{bench}.egraph_bounded_resyn2/candidate-diagnostics.json')]:
   if p.exists():job[key]=str(p);job[key+'_sha256']=sha(p)
  # Obtain diagnostic paths from the recorded mapper settings, not an assumed label.
  if 'report' in job:
   report=json.loads(Path(job['report']).read_text())
   for row in report['results']:
    for key,env in [('pool','EGM_POOL_DIAG_PATH'),('candidates','EGM_DIAG_PATH')]:
     p=Path(row.get('settings',{}).get(env,'/nonexistent'))
     if p.is_file():job[key]=str(p);job[key+'_sha256']=sha(p)
  campaign['jobs'].append(job);(out/'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n');print(name,r.returncode,flush=True)
 campaign['status']='attempted';(out/'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n')
if __name__=='__main__':main()
