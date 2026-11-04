#!/usr/bin/env python3
"""Finite same-binary native-fast / critical-quotient representative controls.
No baseline execution, no winner selection, no input preoptimization.
"""
import argparse,csv,fcntl,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'scripts/emapper'))
from provenance import sha,git_revision,hashes
from run_fast import source
PROTOCOL=R/'results_v2/emapper_seeded_epfl_20261101/protocol.json'
def read(p):return json.loads(Path(p).read_text())
def dump(p,j):
 p=Path(p);q=p.with_suffix('.pending');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
def identity():
 return dict(current_source=source(),current_binaries=hashes([R/'target/release/emapper_fast',R/'target/release/emapper_measure']),build_receipt_sha256=sha(R/'target/release/emapper-fast-build.json'),driver_sha256=sha(Path(__file__)),profile_sha256=sha(R/'emapper/profiles/native_fast.json'),baseline_sha256=sha(R/'results_v2/full_baselines_20261025/baselines.json'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-jobs',type=int,default=12);a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);locks=[]
 for p in (R/'results_v2/independent_components_20261019/global.lock',out/'runner.lock'):
  f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 if (out/'campaign.json').exists():c=read(out/'campaign.json');assert c['frozen']==identity()
 else:
  inputs=read(PROTOCOL)['inputs'];names=['ctrl','bar','i2c','sin','sqrt','hyp'];plan=[]
  for i,n in enumerate(names):
   for arm in (['off','on']if i%2==0 else['on','off']):plan.append(dict(key=n+'_'+arm,design=n,critical_quotients=arm))
  c=dict(schema=1,source_commit=git_revision(),frozen=identity(),inputs={n:inputs[n]for n in names},plan=plan,jobs={},expected=len(plan),scope='six representative inputs x2 same native binary; full library loaded, wide matching not implemented; no external mapper seed;16-cut frontier and2area passes; independent finalCEC; not fullEPFL; no baseline run',baseline_runs=0)
 count=0
 for p in c['plan']:
  if p['key']in c['jobs']:continue
  if count>=a.max_jobs:break
  assert c['frozen']==identity();case=out/'cases'/p['key'];assert not case.exists(),'retain unrecorded partial';inp=c['inputs'][p['design']]
  for k in ('prepared','golden'):assert sha(inp[k])==inp[k+'_sha256']
  argv=[sys.executable,str(R/'scripts/emapper/run_fast.py'),inp['prepared'],'--golden',inp['golden'],'--output',str(case),'--cut-frontier','16','--critical-quotients',p['critical_quotients'],'--area-passes','2'];dump(out/'campaign.json',c);dump(out/'state.json',dict(status='running',active=p['key'],completed=len(c['jobs']),expected=c['expected']));print(json.dumps(dict(event='start',key=p['key'])),flush=True)
  with (out/(p['key']+'.log')).open('x')as f:result=subprocess.run(argv,cwd=R,stdout=f,stderr=subprocess.STDOUT,pass_fds=tuple(x.fileno()for x in locks))
  report=case/'report.json';j=dict(**p,argv=argv,exit=result.returncode,status='unreported_exit')
  if report.exists():r=read(report);j.update(report=str(report),report_sha256=sha(report),status=r['status'],A=(r.get('measured')or{}).get('area'),D=(r.get('measured')or{}).get('delay'),T=r.get('mapping_wall_seconds'),CEC=r.get('CEC'),error=r.get('error'),native_stats=(r.get('native')or{}).get('stats'))
  assert c['frozen']==identity();c['jobs'][p['key']]=j;dump(out/'campaign.json',c);count+=1;print(json.dumps(dict(event='end',key=p['key'],status=j['status'],A=j.get('A'),D=j.get('D'),T=j.get('T'))),flush=True)
 dump(out/'state.json',dict(status='complete'if len(c['jobs'])==c['expected']else'batch_complete',completed=len(c['jobs']),expected=c['expected']))
 rows=[dict(p,**{k:v for k,v in c['jobs'].get(p['key'],{}).items()if k not in p})for p in c['plan']];keys=list(dict.fromkeys(k for row in rows for k in row))
 with(out/'current.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
