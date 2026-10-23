"""One uniform source-first configuration, frozen sources, no baseline changes."""
import argparse,json,hashlib,fcntl,subprocess,sys
from pathlib import Path
from run_epfl_baselines import EPFL
R=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-jobs',type=int,default=40);ap.add_argument('--constructor-only',type=int,choices=[0,1],default=0);ap.add_argument('--designs',nargs='+',default=['ctrl','div','mem_ctrl','hyp']+[x for x in EPFL if x not in ['ctrl','div','mem_ctrl','hyp']]);a=ap.parse_args();assert len(set(a.designs))==len(a.designs)and set(a.designs)<=set(EPFL);out=a.output.resolve();out.mkdir(exist_ok=True);cp=out/'campaign.json';template=json.loads(Path('/tmp/emapper-independent-full-20261019/campaign.json').read_text());fp=dict(template['fingerprint']);assert all(sha(p)==h for p,h in fp.items())
 for p in [Path(__file__).resolve(),*[D/x for x in ['bench_source_first.py','input_capacity_policy.py','egraphmap_complete.py','source_cover.py','run_complete_mapping_budget.py']],R/'scripts/experiments/prototypes/region_stitch.py',R/'scripts/experiments/analyze_epfl_structural_budget.py']:fp[str(p)]=sha(p)
 c=json.loads(cp.read_text())if cp.exists()else dict(fingerprint=fp,expected_jobs=2*len(a.designs),designs=a.designs,constructor_only=a.constructor_only,policy='source-first-input-capacity-v3; original bounded optimizer, retry only import refusal using global remaining node allowance; single shared60s optimization window; source construction and optimization count toward300s; explicit fallback',jobs=[])
 assert c['fingerprint']==fp and c['designs']==a.designs and c['constructor_only']==a.constructor_only,'frozen source/config drift';locks=[]
 for p in [out/'runner.lock',R/'results_v2/independent_components_20261019/global.lock',R/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock',R/'results_v2/small_cut_quota_cases_20261014/runner.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 for j in c['jobs']:
  for p,h in j['sha256'].items():assert sha(p)==h
 def save():
  p=cp.with_suffix('.pending');p.write_text(json.dumps(c,indent=2));p.replace(cp)
 done={j['name']for j in c['jobs']};n=0;save()
 for b in a.designs:
  for k in [3,4]:
   name=f'{k}in_{b}'
   if name in done:continue
   if n>=a.max_jobs:save();return
   assert all(sha(p)==h for p,h in fp.items());folder=out/name;audit=out/(name+'.audit.json');assert not folder.exists()and not audit.exists(),'unrecorded partial requires separate audit';j=next(j for j in template['jobs']if j['k']==k and j['bench']==b and j['profile']=='full');argv=list(j['argv']);argv[0]=sys.executable;argv[1]=str(D/'run_complete_mapping_budget.py')
   for flag,value in [('--driver',D/'bench_source_first.py'),('--audit',audit),('--output',folder)]:argv[argv.index(flag)+1]=str(value)
   argv+=['--node-extra','20000','--pool-extra','0','--constructor-only',str(a.constructor_only)];c['in_progress']=dict(name=name,argv=argv);save()
   with(out/(name+'.log')).open('w')as log:r=subprocess.run(argv,cwd=R,stdout=log,stderr=subprocess.STDOUT)
   z=dict(name=name,bench=b,k=k,profile='full',argv=argv,exit_code=r.returncode,sha256={str(p):sha(p)for p in folder.rglob('*')if p.is_file()and p.name in ['report.json','raw.complete.json','raw.source.v','raw.v','mapped.v','mapped.json','evaluate.log','attempt0.admission.json','attempt1.admission.json','attempt0.stdout','attempt0.stderr','attempt1.stdout','attempt1.stderr']});z['sha256'][str(audit)]=sha(audit);z['report']=str(folder/'report.json');z['audit']=str(audit);c['jobs'].append(z);c.pop('in_progress');save();n+=1;print(name,r.returncode,flush=True)
 c['status']='attempted';save()
if __name__=='__main__':main()
