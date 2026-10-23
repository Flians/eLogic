"""New capacity protocol; preserve the old fixed20k studies byte-for-byte."""
from pathlib import Path
import argparse,json,hashlib,fcntl,subprocess,sys
R=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--max-jobs',type=int,default=6);v=a.parse_args();out=v.output.resolve();out.mkdir(exist_ok=True);cp=out/'campaign.json';template=json.loads(Path('/tmp/emapper-independent-full-20261019/campaign.json').read_text());fp=dict(template['fingerprint']);assert all(sha(p)==h for p,h in fp.items())
 for p in [Path(__file__).resolve(),D/'bench_input_capacity.py',D/'input_capacity_policy.py']:fp[str(p)]=sha(p)
 c=json.loads(cp.read_text())if cp.exists()else dict(fingerprint=fp,expected_jobs=6,policy='input-upper-plus20000; pool100000 unchanged; native60/recovery2/mapping300 unchanged; not old20k budget',jobs=[])
 assert c['fingerprint']==fp;locks=[]
 for p in [out/'runner.lock',R/'results_v2/independent_components_20261019/global.lock',R/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock',R/'results_v2/small_cut_quota_cases_20261014/runner.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 for j in c['jobs']:
  for key in ['report','audit','admission']:
   if key in j:assert sha(j[key])==j[key+'_sha256']
 def save():
  t=cp.with_suffix('.pending');t.write_text(json.dumps(c,indent=2));t.replace(cp)
 done={j['name']for j in c['jobs']};n=0
 for b in ['div','mem_ctrl','hyp']:
  for k in [3,4]:
   name=f'{k}in_{b}';
   if name in done:continue
   if n>=v.max_jobs:save();return
   assert all(sha(p)==h for p,h in fp.items());folder=out/name;audit=out/(name+'.audit.json');assert not folder.exists()and not audit.exists(),'audit unrecorded partial before resume';j=next(j for j in template['jobs']if j['k']==k and j['bench']==b and j['profile']=='full');argv=list(j['argv']);argv[0]=sys.executable
   for flag,value in [('--driver',D/'bench_input_capacity.py'),('--audit',audit),('--output',folder)]:argv[argv.index(flag)+1]=str(value)
   argv+=['--node-extra','20000','--pool-extra','0'];c['in_progress']=dict(name=name,argv=argv);save()
   with(out/(name+'.log')).open('w')as log:r=subprocess.run(argv,cwd=R,stdout=log,stderr=subprocess.STDOUT)
   z=dict(name=name,bench=b,k=k,profile='full',argv=argv,exit_code=r.returncode)
   for key,p in [('report',folder/'report.json'),('audit',audit),('admission',folder/(b+'.egraph_staged_control_resyn2/pool-diagnostics.admission.json'))]:
    if p.exists():z[key]=str(p);z[key+'_sha256']=sha(p)
   c['jobs'].append(z);c.pop('in_progress');save();n+=1;print(name,r.returncode,flush=True)
 c['status']='attempted';save()
if __name__=='__main__':main()
