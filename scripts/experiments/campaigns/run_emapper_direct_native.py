#!/usr/bin/env python3
"""Finite four-representative direct-native study; no source/seed construction."""
import fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'scripts/emapper'))
from provenance import source_files,hashes,sha,git_revision
from physical import measure_cover
from verify import verify
PROTOCOL=R/'results_v2/emapper_seeded_epfl_20261101/protocol.json'
NAMES=['ctrl','sin','div','hyp']
COMMON=['--ablation','source-editable','--initial-cover','direct','--recovery-inputs','overlay','--recovery-noop-prefilter','on','--recovery-schedule','legacy','--area-reject-cache','off','--sdc-window-inputs','0','--observe','--cut-join','legacy','--cut-truth','legacy','--batch-frontier-cache','on','--frontier-preparation','off','--recovery-cache','legacy','--flow-witness','legacy']
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 q=p.with_suffix('.pending');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def fingerprint():
 files=source_files()+list((R/'scripts/emapper').glob('*.py'))+[Path(__file__),PROTOCOL,R/'target/release/emapper-build.json',R/'target/release/emapper_kernel',R/'target/release/emapper_measure',R/'emapper/profiles/delivery.json',R/'scripts/experiments/port_identifiers.py']+list((R/'emapper/libraries/asap7').iterdir())
 return hashes(sorted(set(p for p in files if p.is_file())))
def main():
 import argparse
 ap=argparse.ArgumentParser(__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-jobs',type=int,default=1);a=ap.parse_args();assert a.max_jobs>0
 out=a.output.resolve();out.mkdir(exist_ok=True,parents=True);locks=[]
 for p in [R/'results_v2/independent_components_20261019/global.lock',out/'runner.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 if (out/'campaign.json').exists():c=read(out/'campaign.json')
 else:
  assert read(R/'target/release/emapper-build.json')==dict(source=hashes(source_files()),binaries=hashes([R/'target/release/emapper_kernel',R/'target/release/emapper_measure']))
  assert read(R/'emapper/profiles/delivery.json')['mapping_seconds']==600
  assert read(R/'results_v2/emapper_direct_native_20261107/public/validation.json')['CEC_checks']==10
  original=read(PROTOCOL)['inputs'];inputs={n:{k:v for k,v in original[n].items()if k in ['prepared','prepared_sha256','golden','golden_sha256']}for n in NAMES}
  c=dict(source_commit=git_revision(),frozen=fingerprint(),inputs=inputs,planned=NAMES,jobs={},expected=4,common=COMMON,mapping_seconds=600,CEC_in_clock=False,tools={n:dict(path=shutil.which(n),sha256=sha(shutil.which(n)))for n in ['abc','yosys']},baseline_runs=0,GitHub_push=False,scope='direct native existing complete pipeline; no source netlist/bundle/project/warm-initial-drain. Four representatives only, not fullEPFL or matched oldbinary causal experiment.')
  for n,h in c['frozen'].items():
   q=out/'frozen'/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/n,q);assert sha(q)==h
  dump(out/'campaign.json',c)
 assert c['frozen']==fingerprint(),'frozen dependency drift'
 for j in c['jobs'].values():
  for n,h in j['artifacts'].items():assert sha(Path(j['case'])/n)==h
 count=0
 for name in c['planned']:
  if name in c['jobs']:continue
  if count>=a.max_jobs:break
  assert c['frozen']==fingerprint()
  inp=c['inputs'][name]
  for n in ['prepared','golden']:assert sha(inp[n])==inp[n+'_sha256']
  case=out/'cases'/name;assert not case.exists(),'audit partial; never overwrite'
  argv=[sys.executable,str(R/'scripts/emapper/run.py'),inp['prepared'],'--golden',inp['golden'],'--output',str(case),*COMMON];dump(out/(name+'-attempt.json'),dict(argv=argv,input=inp,source_commit=c['source_commit'],frozen=c['frozen']));dump(out/'state.json',dict(status='running',active=name,completed=len(c['jobs']),expected=4));print('start',name,flush=True)
  with (out/(name+'.log')).open('x')as f:p=subprocess.run(argv,cwd=R,stdout=f,stderr=subprocess.STDOUT,pass_fds=tuple(f.fileno()for f in locks))
  j=dict(case=str(case),runner_exit=p.returncode,status='unmeasured_failure',argv=argv)
  rp=case/'report.json'
  try:
   r=read(rp);j.update(report=str(rp),report_sha256=sha(rp),status=r['status']);assert r['cec']==1 and r['evaluation']['exact_agreement'] and r['status']=='ok'
   assert r['direct_native'] and r['native']['direct_native'] and r['native']['source']is None
   assert not any((case/n).exists()for n in ['raw.source.v','source.bundle.json','external.bundle.json','incumbents/source-seed.json','incumbents/native-initial-delay.json'])
   assert not r.get('external_seed')and not r.get('fresh_seed_generation')
   first=r['first_native_cover'];measured=measure_cover(R/'emapper/libraries/asap7/asap7_full.genlib',Path(first['netlist']),case/'first-cover-check',R/'target/release/emapper_measure',c['tools']['yosys']['path']);os.environ['EGM_YOSYS']=c['tools']['yosys']['path'];os.environ['BENCH_RS_EMAP_ABC']=c['tools']['abc']['path'];measured['CEC']=int(verify(R/'emapper/libraries/asap7/cells.v',Path(inp['golden']),Path(measured['netjson']),case/'first-cover-check/cec'));measured['exact_AD']=(measured['measured']['area'],measured['measured']['delay'])==(first['area'],first['delay']);j['first_check']=measured;assert measured['CEC']==1 and measured['exact_AD'];j.update(first_confirmed_seconds=first['mapping_elapsed_seconds'],first_AD=[first['area'],first['delay']],final=r['measured'],T=r['mapping_wall_secs'],net=r['search_vs_first_native'],search_finished=r['search_finished'],search_deadline=r['search_timeout'])
  except Exception as e:j.update(status='invalid',error=repr(e))
  assert c['frozen']==fingerprint();j['artifacts']={str(p.relative_to(case)):sha(p)for p in sorted(case.rglob('*'))if p.is_file()};j['attempt_sha256']=sha(out/(name+'-attempt.json'));j['log_sha256']=sha(out/(name+'.log'));c['jobs'][name]=j;dump(out/'campaign.json',c);count+=1;print('end',name,j['status'],flush=True)
  if j['status']!='ok':dump(out/'state.json',dict(status='failed_stopped_for_audit',active=None,completed=len(c['jobs']),error=j.get('error')));return 1
 dump(out/'state.json',dict(status='complete_requires_review'if len(c['jobs'])==4 else'batch_complete',active=None,completed=len(c['jobs']),expected=4,default_promoted=False))
 return 0
if __name__=='__main__':raise SystemExit(main())
