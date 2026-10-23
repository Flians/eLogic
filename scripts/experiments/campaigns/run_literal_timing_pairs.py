#!/usr/bin/env python3
"""Fixed two-profile literal-import timing ablation, all trials retained."""
import argparse,json,subprocess,sys,fcntl,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False)
 root=Path('/tmp/emapper-literal-import-20261013');subjects=ROOT/'results_v2/epfl_resyn2_subjects_20261012/report.json';driver=Path(__file__).with_name('bench_egraph_literal_probe.py');wrap=Path(__file__).with_name('run_with_mapping_budget.py')
 files=[root/'target/release/bench_rs_egraph_map',ROOT/'target/release/bench_rs_netlist_eval',ROOT/'scripts/experiments/restrict_genlib.py',ROOT/'scripts/experiments/shared_subject.py',ROOT/'scripts/experiments/port_identifiers.py',ROOT/'tools/mockturtle/experiments/cell_libraries/asap7.genlib',subjects,driver,wrap,*sorted((root/'emapper/src').glob('*.rs')),root/'src/bin/bench_rs_egraph_map.rs',Path(__file__).resolve()];f={str(p):sha(p) for p in files};state={'subjects':str(subjects),'expected_jobs':16,'profiles':{'control':{'library_timing':0,'timing_cap':0},'timing':{'library_timing':1,'timing_cap':1}},'global_recovery_secs':2,'node_limit':20000,'retained_pool':100000,'cuts':16,'iterations':16,'mapping_seconds':300,'fingerprint':f,'jobs':[]}
 locks=[]
 for p in [out/'runner.lock',ROOT/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock']:
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 for k in (3,4):
  for n,bench in enumerate(['arbiter','voter','square','sqrt']):
   for profile in (['control','timing'] if n%2==0 else ['timing','control']):
    assert all(sha(Path(p))==h for p,h in f.items());name=f'{k}in_{bench}_{profile}';folder=out/name;audit=out/(name+'.audit.json');t=int(profile=='timing')
    args=[sys.executable,str(wrap),'--mapping-seconds','300','--driver',str(driver),'--audit',str(audit),'--','--mapper-root',str(root),'--k',str(k),'--objective','delay-area','--methods','egraph_bounded','--literal-import','1','--coupled-event-flow','1','--phase-recovery','1','--phase-time-fraction','0.25','--region-classes','256','--library-timing',str(t),'--timing-cap',str(t),'--output',str(folder),'--prepared-report',str(subjects),'--suite','epfl','--benches',bench]
    with (out/(name+'.log')).open('w') as log:r=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    j={'name':name,'k':k,'bench':bench,'profile':profile,'argv':args,'exit_code':r.returncode}
    for key,p in [('report',folder/'report.json'),('audit',audit)]:
     if p.exists():j[key]=str(p);j[key+'_sha256']=sha(p)
    state['jobs'].append(j);(out/'campaign.json').write_text(json.dumps(state,indent=2)+'\n');print(name,r.returncode,flush=True)
 state['status']='attempted';(out/'campaign.json').write_text(json.dumps(state,indent=2)+'\n')
if __name__=='__main__':main()
