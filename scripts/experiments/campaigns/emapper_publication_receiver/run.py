"""Atomic commit receipt receiver ablation; serial real native mapping, no baseline reruns."""
import argparse,fcntl,hashlib,json,os,pathlib,shutil,socket,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[4];O=R/'results_v2/emapper_publication_receiver_20261111/benchmark'
sys.path.insert(0,str(R/'scripts/emapper'));from provenance import source_files,sha,hashes
from observations import read as read_observations
SUB=R/'results_v2/epfl_resyn2_subjects_20261012/report.json'
BASE=R/'results_v2/full_baselines_20261025/baselines.json'
PREVIOUS=R/'results_v2/emapper_timing_scratch_20261111/study/campaign.json'
def read(p):return json.loads(pathlib.Path(p).read_text())
def save(p,x):
 t=p.with_suffix('.pending');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def fingerprint():
 files=source_files()+list((R/'scripts/emapper').glob('*.py'))+list((R/'emapper/libraries/asap7').iterdir())+[R/'emapper/profiles/delivery.json',R/'target/release/emapper_kernel',R/'target/release/emapper_measure',R/'target/release/emapper-build.json',pathlib.Path(__file__),pathlib.Path(__file__).with_name("observations.py"),SUB,BASE,PREVIOUS,R/'scripts/experiments/port_identifiers.py']
 return {str(p.relative_to(R)):sha(p)for p in sorted(set(files))if p.is_file()}
def subjects():
 j=read(SUB);assert len(j['results'])==20 and j['preprocess']=='resyn2';out={}
 for r in j['results']:
  n=r['bench'];prepared=SUB.parent/(n+'.resyn2.aig');golden=R/'tools/mockturtle/experiments/benchmarks'/(n+'.aig')
  assert r['status']=='ok'and r['preprocessing']['cec']==1
  assert sha(prepared)==r['preprocessing']['sha256']and sha(golden)==r['input_sha256']
  out[n]=dict(prepared=str(prepared),prepared_sha256=sha(prepared),golden=str(golden),golden_sha256=sha(golden),preparation='reuse once-resyn2, no preprocessing rerun')
 assert len(out)==20;return out
def state(c,active=None):save(O/'state.json',dict(active=active,completed=len(c['jobs']),expected=8,baseline_runs=0,default_promoted=False,GitHub_push=False))
def execute(c,job,locks):
 inp=c['subjects'][job['design']];case=O/'cases'/job['key'];assert not case.exists(),'unrecorded partial requires audit, never overwrite'
 assert shutil.disk_usage(R).free>8*(1<<30),'pre-dispatch storage guard, not a mapper limit'
 argv=['taskset','-c','6',sys.executable,str(R/'scripts/emapper/run.py'),inp['prepared'],'--golden',inp['golden'],'--output',str(case),*c['common'],'--trial-safe-area','off','--trial-memo','off','--trial-timing-dependencies','on','--trial-fixed-area','on','--trial-release-reuse','off','--trial-delta-cost','off','--trial-timing-scratch','on','--selected-witness-only','on','--evidence-transfer','move','--publication-receiver',job['receiver'],'--publication-timing']
 assert '--replay-snapshot'not in argv and '--seed-verilog'not in argv and '--fresh-emap-seed'not in argv
 nonce=hashlib.sha256(str(case).encode()).hexdigest();env=dict(os.environ,EMAPPER_AFFINITY_RECEIPT=str(case/'native-affinity.json'),EMAPPER_AFFINITY_NONCE=nonce)
 save(O/(job['key']+'-attempt.json'),dict(argv=argv,input=inp,nonce=nonce,scope=c['policy']))
 with(O/(job['key']+'.log')).open('x')as out:p=subprocess.run(argv,cwd=R,env=env,stdout=out,stderr=subprocess.STDOUT,pass_fds=tuple(f.fileno()for f in locks))
 row=dict(**job,case=str(case),exit=p.returncode,status='unmeasured_failure')
 try:
  r=read(case/'report.json');row.update(report_status=r.get('status'),error=r.get('error'),T=r.get('mapping_wall_secs'),search_timeout=r.get('search_timeout'),CEC=r.get('cec'),measured=r.get('measured'),first_native_cover=r.get('first_native_cover'))
  if p.returncode==0 and r.get('status')=='ok':
   assert r['cec']==1 and r['evaluation']['exact_agreement']and r['selected_evidence']['valid']and r['selected_evidence']['netlist_binding_agreement']
   cfg=r['configuration'];assert cfg['mapping_seconds']==600 and cfg['objective']=='delay-area'and cfg['initial_cover_policy']=='direct'and cfg['trial_timing_scratch']=='on'and cfg['selected_witness_only']=='on'and cfg['evidence_transfer']=='move'and cfg['publication_receiver']==job['receiver']and cfg['publication_timing']and cfg['trial_delta_cost']=='off'and cfg['trial_timing_dependencies']=='on'and cfg['trial_fixed_area']=='on'and cfg['trial_release_reuse']=='off'
   assert r['direct_native']and 'snapshot_sha256'not in r and not r.get('external_seed')and not r.get('seed_generation_in_mapping_clock')
   affinity=read(case/'native-affinity.json');assert affinity['nonce']==nonce and affinity['cpus_allowed_list']=='6'
   first=r['first_native_cover'];ad0=(first['area'],first['delay']);ad1=(r['measured']['area'],r['measured']['delay'])
   row.update(status='ok',A=ad1[0],D=ad1[1],initial_AD=list(ad0),search_net_gain='better'if(ad1[1],ad1[0])<(ad0[1],ad0[0])else('equal'if ad1==ad0 else'worse'),selected_evidence=r['selected_evidence'],search_finished=r['search_finished'],output_sha256=r['output_sha256'],configuration=cfg)
  else:row['status']='mapping_failure'
  last=None;progress=None
  events,tail=read_observations(case/'observation.jsonl')if(case/'observation.jsonl').exists()else([],None)
  row['observation_tail_fragment']=tail
  for x in events:
   last=x
   if x.get('details',{}).get('cover_materialization'):row['evidence_handoff_prefix']=x['details']['cover_materialization']
   if x.get('details',{}).get('trial_diagnostics'):progress=x
  if last:row['exclusive_seconds']=last.get('exclusive_seconds')
  if progress:row.update(prefix_trial_diagnostics=progress['details']['trial_diagnostics'],prefix_elapsed_seconds=progress.get('elapsed_seconds'),counter_scope='observation prefix, not final totals')
 except Exception as exc:row.update(status='harness_invalid',audit_error=repr(exc))
 row['artifacts']={str(p.relative_to(case)):sha(p)for p in sorted(case.rglob('*'))if p.is_file()};return row
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--max-jobs',type=int,default=2);a=ap.parse_args();assert a.max_jobs>0
 O.mkdir(parents=True,exist_ok=True);locks=[]
 for p in [R/'results_v2/independent_components_20261019/global.lock',O/'runner.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 if(O/'campaign.json').exists():c=read(O/'campaign.json')
 else:
  receipt=read(R/'target/release/emapper-build.json');assert receipt==dict(source=hashes(source_files()),binaries=hashes([R/'target/release/emapper_kernel',R/'target/release/emapper_measure']))
  sub=subjects();base=read(BASE);assert base['library_sha256']==sha(R/'emapper/libraries/asap7/asap7_full.genlib')
  for r in base['results']:
   if r['bench']in sub:assert r['input_sha256']==sub[r['bench']]['golden_sha256']
  common=read(PREVIOUS)['common'];assert '--replay-snapshot'not in common and '--seed-verilog'not in common and '--fresh-emap-seed'not in common
  order=['dec','sin'];variants=[('control','scan'),('events','events')]
  plan=[dict(key=f'{n}_r{rep}_{v}',design=n,rep=rep,variant=v,receiver=s,stage='representative')for rep in range(2)for i,n in enumerate(order)for v,s in (variants if (i+rep)%2==0 else list(reversed(variants)))]
  c=dict(implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),executed_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),frozen=fingerprint(),subjects=sub,common=common,plan=plan,jobs={},expected=8,host=socket.gethostname(),affinity='6',baseline_runs=0,default_promoted=False,GitHub_push=False,policy='full native mapping, no restored candidate pool or external seed; same once-resyn2 and ASAP7 single-output library; all native generation/first cover/search/proof/publication within600s, pure conversion and CEC/measurement excluded; same binary, timing scratch ON both; single selected witness ON both; evidence transfer move ON both; shared absolute deadline alignment and commit/receipt CLOCK_MONOTONIC trace ON both; only polling scan vs atomic commit event receiver factor; two alternating repetitions dec/sin; serial CPU6; no baseline reruns, no per-design winner selection')
  save(O/'campaign.json',c);state(c)
 assert c['frozen']==fingerprint(),'source, binary or harness drift; new version required'
 for r in c['jobs'].values():
  assert r['status']!='harness_invalid','audit invalid case before resume'
  for n,h in r['artifacts'].items():assert sha(pathlib.Path(r['case'])/n)==h
 done=0
 for job in c['plan']:
  if job['key']in c['jobs']:continue
  if done>=a.max_jobs:break
  assert c['frozen']==fingerprint();inp=c['subjects'][job['design']]
  for k in ['prepared','golden']:assert sha(inp[k])==inp[k+'_sha256']
  state(c,job['key']);print('start',job['key'],flush=True);r=execute(c,job,locks)
  assert c['frozen']==fingerprint();c['jobs'][job['key']]=r;save(O/'campaign.json',c);done+=1
  print('end',job['key'],r['status'],r.get('T'),r.get('A'),r.get('D'),r.get('audit_error',r.get('error')),flush=True)
  if r['status']=='harness_invalid':state(c);return 1
 state(c);return 0
if __name__=='__main__':raise SystemExit(main())
