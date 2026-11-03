#!/usr/bin/env python3
"""Same-native cold fixed-pool recovery repeats; not full mapping PPA.
No candidate generation or Flow. Initial and final covers have independent CEC.
"""
import argparse,fcntl,json,os,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'scripts/emapper'))
from provenance import source_files,hashes,sha,git_revision
from archive import seal,verify_files
from physical import measure_cover
from verify import verify
import shutil
COMMON=['--ablation','source-editable','--recovery-inputs','overlay','--recovery-noop-prefilter','on','--recovery-cache','legacy','--recovery-schedule','legacy','--area-reject-cache','off','--observe']
SOURCE=R/'results_v2/emapper_window_p0_light_20261103/campaign.json'
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def fingerprint():return hashes(source_files()+list((R/'scripts/emapper').glob('*.py'))+[Path(__file__),R/'target/release/emapper-build.json',R/'target/release/emapper_kernel',R/'target/release/emapper_measure',R/'emapper/profiles/delivery.json',*filter(Path.is_file,(R/'emapper/libraries/asap7').glob('*'))])
def initialize():
 source=read(SOURCE);assert len(source['jobs'])==4
 cases={}
 for key,j in source['jobs'].items():
  assert not j['diagnostic']['errors'] and j['initial']['CEC']==1 and sha(j['report'])==j['report_sha256'];r=read(j['report']);assert r['cec']==1
  snapshot=Path(j['case'])/'recovery-snapshot.json';assert sha(snapshot)==j['artifacts']['recovery-snapshot.json']
  name,arm=key.rsplit('_',1);cases[key]=dict(snapshot=str(snapshot),snapshot_sha256=sha(snapshot),expected=j['diagnostic']['snapshot_summary'],input=source['inputs'][name],window=source['arms'][arm],origin_report=j['report'],origin_report_sha256=j['report_sha256'])
 order=list(cases);planned=[dict(key=f'r{rep}_{key}',source_key=key)for rep in range(2)for key in(order if rep==0 else list(reversed(order)))]
 return dict(schema=1,scope='diagnostic cold recovery of frozen candidate/proof pools; generation/Flow disabled; same600s+externalCEC; initial R0->delivered R1, NOT seed->mapped PPA or a new full mapper gain',source_commit=git_revision(),source_campaign=str(SOURCE),source_campaign_sha256=sha(SOURCE),frozen=fingerprint(),cases=cases,planned=planned,jobs={},expected=8,repetitions=2,baseline_runs=0,GitHub_push=False)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-jobs',type=int,default=2);ap.add_argument('--verify-artifacts-only',action='store_true');a=ap.parse_args();assert a.max_jobs>0
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);locks=[]
 for p in [R/'results_v2/independent_components_20261019/global.lock',out/'runner.lock']:
  p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 c=read(out/'campaign.json')if(out/'campaign.json').exists()else initialize();assert c['frozen']==fingerprint();assert sha(SOURCE)==c['source_campaign_sha256'];dump(out/'campaign.json',c)
 for j in c['jobs'].values():
  assert sha(j['report'])==j['report_sha256']
  if a.verify_artifacts_only:verify_files(Path(j['case']),j['artifacts'],j.get('capsule'))
 if a.verify_artifacts_only:dump(out/'artifact-validation.json',dict(campaign_sha256=sha(out/'campaign.json'),cases=len(c['jobs']),files=sum(len(j['artifacts'])for j in c['jobs'].values()),all_hashes_match=True));return
 count=0
 for plan in c['planned']:
  key=plan['key'];source=c['cases'][plan['source_key']]
  if key in c['jobs']:continue
  if count>=a.max_jobs:dump(out/'state.json',dict(status='batch_complete',completed=len(c['jobs']),expected=8));return
  assert c['frozen']==fingerprint();assert sha(source['snapshot'])==source['snapshot_sha256']
  inp=source['input']
  for k in ['prepared','golden','seed']:assert sha(inp[k])==inp[k+'_sha256']
  case=out/'cases'/key;assert not case.exists(),'unrecorded partial must be audited and retained'
  argv=[sys.executable,str(R/'scripts/emapper/run.py'),inp['prepared'],'--golden',inp['golden'],'--output',str(case),*COMMON,'--sdc-window-inputs',str(source['window']),'--replay-snapshot',source['snapshot']]
  dump(out/(key+'-attempt.json'),dict(argv=argv,frozen=c['frozen'],source=source));dump(out/'state.json',dict(status='running',active=key,completed=len(c['jobs']),expected=8));print(json.dumps(dict(event='start',key=key)),flush=True)
  with(out/(key+'.log')).open('x')as log:p=subprocess.run(argv,cwd=R,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(f.fileno()for f in locks))
  report=case/'report.json';r=read(report);assert r['status']!='running';j=dict(case=str(case),report=str(report),report_sha256=sha(report),runner_exit=p.returncode,status=r['status'],errors=[])
  try:
   assert r['diagnostic_scope'].startswith('fixed candidate/proof snapshot') and r['snapshot_sha256']==source['snapshot_sha256']
   m=read(case/'incumbents/00000001/manifest.json');expected=source['expected']['initial_AD'];assert [m['area'],m['delay']]==expected
   initial=measure_cover(R/'emapper/libraries/asap7/asap7_full.genlib',case/'incumbents/00000001/mapped.v',case/'replay-initial',R/'target/release/emapper_measure',shutil.which('yosys'))
   os.environ['EGM_YOSYS']=shutil.which('yosys');os.environ['BENCH_RS_EMAP_ABC']=shutil.which('abc');initial['CEC']=int(verify(R/'emapper/libraries/asap7/cells.v',Path(inp['golden']),Path(initial['netjson']),case/'replay-initial/cec'));j['initial']=initial
   assert initial['CEC']==1 and [initial['measured']['area'],initial['measured']['delay']]==expected
   final=r['measured'];assert r['cec']==1 and r['evaluation']['exact_agreement'];assert(final['delay'],final['area'])<=(expected[1],expected[0])
   j.update(valid_replay=True,replay_search_gain='improved'if(final['delay'],final['area'])<(expected[1],expected[0])else'equal',initial_AD=expected,final_AD=[final['area'],final['delay']])
   rp=case/'raw.replay.json'
   if rp.exists():j['native_replay']=read(rp)
  except Exception as error:j.update(valid_replay=False);j['errors'].append(type(error).__name__+': '+str(error))
  assert c['frozen']==fingerprint();j['artifacts']={str(p.relative_to(case)):sha(p)for p in sorted(case.rglob('*'))if p.is_file()};j['capsule']=seal(case,out/'capsules');verify_files(case,j['artifacts'],j['capsule']);c['jobs'][key]=j;dump(out/'campaign.json',c);count+=1;print(json.dumps(dict(event='end',key=key,valid=j.get('valid_replay'),gain=j.get('replay_search_gain'),AD=j.get('final_AD'),errors=j['errors'])),flush=True)
 dump(out/'state.json',dict(status='complete',completed=len(c['jobs']),expected=8))
if __name__=='__main__':main()
