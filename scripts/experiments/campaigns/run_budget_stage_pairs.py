#!/usr/bin/env python3
"""Separate early-area and return-reserve factors, fixed budgets and all observations."""
import argparse,fcntl,json,subprocess,sys
from pathlib import Path
from run_epfl_baselines import ROOT,EPFL,sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['early','reserve'],required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False);driver=Path(__file__).with_name('bench_egraph_budget_stages.py');wrap=Path(__file__).with_name('run_with_mapping_budget.py');subjects=ROOT/'results_v2/epfl_resyn2_subjects_20261012/report.json';root=Path('/tmp/emapper-early-required-20261013' if a.mode=='early' else '/tmp/emapper-return-reserve-20261013')
 files=[root/'target/release/bench_rs_egraph_map',ROOT/'target/release/bench_rs_netlist_eval',subjects,driver,wrap,Path(__file__).resolve(),*sorted((root/'emapper/src').glob('*.rs')),root/'src/bin/bench_rs_egraph_map.rs'];f={str(p):sha(p) for p in files};c={'mode':a.mode,'subjects':str(subjects),'fingerprint':f,'expected_jobs':80 if a.mode=='early' else 8,'budgets':{'mapping':300,'recovery':2,'nodes':20000,'cuts':16,'pool':100000,'iterations':16},'jobs':[]};locks=[]
 for p in [out/'runner.lock',ROOT/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock']:
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 for k in [3,4]:
  for n,b in enumerate(EPFL if a.mode=='early' else ['sqrt']):
   for rep in range(1 if a.mode=='early' else 2):
    for variant in ([0,1] if (n+rep)%2==0 else [1,0]):
     assert all(sha(Path(p))==h for p,h in f.items());name=f'{k}in_{b}_r{rep}_v{variant}';folder=out/name;audit=out/(name+'.audit.json');options=['--mapper-root',str(root),'--k',str(k),'--objective','delay-area','--methods','egraph_bounded','--coupled-event-flow','1','--phase-recovery','1','--phase-time-fraction','0.25','--region-classes','256','--early-slack-flow',str(variant if a.mode=='early' else 0),'--return-reserve',str(0.15 if a.mode=='reserve' and variant else 0.05),'--literal-import',str(int(a.mode=='reserve')),'--library-timing',str(int(a.mode=='reserve')),'--timing-cap',str(int(a.mode=='reserve')),'--output',str(folder),'--prepared-report',str(subjects),'--suite','epfl','--benches',b]
     argv=[sys.executable,str(wrap),'--mapping-seconds','300','--driver',str(driver),'--audit',str(audit),'--',*options]
     with (out/(name+'.log')).open('w') as log:r=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
     j={'name':name,'k':k,'bench':b,'repetition':rep,'variant':variant,'argv':argv,'exit_code':r.returncode}
     for key,p in [('report',folder/'report.json'),('audit',audit)]:
      if p.exists():j[key]=str(p);j[key+'_sha256']=sha(p)
     c['jobs'].append(j);(out/'campaign.json').write_text(json.dumps(c,indent=2)+'\n');print(name,r.returncode,flush=True)
 c['status']='attempted';(out/'campaign.json').write_text(json.dumps(c,indent=2)+'\n')
if __name__=='__main__':main()
