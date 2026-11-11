"""DAC convergence: one source/binary, representative gate before uniform EPFL.

No baseline dispatch. Completed/failed attempts are immutable; partials need an
explicit audit. Samples are outside case artifacts. No detached snapshot producer.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,pathlib,shutil,socket,subprocess,sys,threading
R=pathlib.Path(__file__).resolve().parents[3]
DEFAULT=R/'results_v2/emapper_dac_convergence_20261111/benchmark'
sys.path.insert(0,str(R/'scripts/emapper'));from provenance import sha,hashes,source_files
sys.path.insert(0,str(R/'scripts/experiments/campaigns/emapper_physical_stamp'));from observations import read as observations
from timing import case as timing_case
sys.path.insert(0,str(R/'scripts/experiments/diagnostics/emapper_stamp_cost'));from resource_pair import sample
PRIOR=R/'results_v2/emapper_physical_stamp_20261111/benchmark/campaign.json'
SUB=R/'results_v2/epfl_resyn2_subjects_20261012/report.json'
BASE=R/'results_v2/full_baselines_20261025/baselines.json'
def read(p):return json.loads(pathlib.Path(p).read_text())
def save(p,v):
 t=p.with_suffix('.pending');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def fingerprint():
 files=source_files()+list((R/'scripts/emapper').glob('*.py'))+list((R/'emapper/libraries/asap7').iterdir())+[R/'emapper/profiles/delivery.json',R/'target/release/emapper_kernel',R/'target/release/emapper_measure',R/'target/release/emapper-build.json',pathlib.Path(__file__),SUB,BASE,PRIOR,R/'scripts/experiments/port_identifiers.py']
 files += [R/'scripts/experiments/campaigns/emapper_physical_stamp'/n for n in ['observations.py','timing.py']]
 files += [R/'scripts/experiments/diagnostics/emapper_stamp_cost/resource_pair.py']
 return hashes(sorted(set(p for p in files if p.is_file())))
def subjects():
 j=read(SUB);assert j['preprocess']=='resyn2'and len(j['results'])==20
 out={}
 for r in j['results']:
  n=r['bench'];a=SUB.parent/(n+'.resyn2.aig');g=R/'tools/mockturtle/experiments/benchmarks'/(n+'.aig')
  assert r['status']=='ok'and r['preprocessing']['cec']==1 and sha(a)==r['preprocessing']['sha256']and sha(g)==r['input_sha256']
  out[n]=dict(prepared=str(a),prepared_sha256=sha(a),golden=str(g),golden_sha256=sha(g))
 return out
def make_campaign():
 assert read(R/'target/release/emapper-build.json')==dict(source=hashes(source_files()),binaries=hashes([R/'target/release/emapper_kernel',R/'target/release/emapper_measure']))
 sub=subjects();base=read(BASE);assert base['library_sha256']==sha(R/'emapper/libraries/asap7/asap7_full.genlib')
 for r in base['results']:
  if r['bench']in sub:assert r['input_sha256']==sub[r['bench']]['golden_sha256']
 plans={}
 plans['representatives']=[dict(key=f'{n}_r0_{v}',design=n,rep=0,variant=v,phase='representatives')for i,n in enumerate(['dec','sin','sqrt','div'])for v in (['off','on']if i%2==0 else ['on','off'])]
 plans['repeats']=[dict(key=f'{n}_r{rep}_{v}',design=n,rep=rep,variant=v,phase='repeats')for rep in [1,2]for i,n in enumerate(['dec','sin'])for v in (['off','on']if(i+rep)%2==0 else ['on','off'])]
 return dict(implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=R).strip(),frozen=fingerprint(),subjects=sub,common=read(PRIOR)['common'],plans=plans,jobs={},host=socket.gethostname(),affinity='native+frontend CPU6, passive sampler CPU2',baseline_runs=0,default_promoted=False,GitHub_push=False,
  policy='600s total native mapping/proof/publication; once-resyn2, full ASAP7 single-output, no external seed, CEC/conversion/public measurement outside clock; off=fragmented trace+copied evidence arrays, on=single-line trace+owned arrays; physical stamp construct BOTH; all prior candidate/scoring/recovery settings held',
  selection_policy='after representatives and 3 paired dec/sin repetitions: require valid CEC/source/clock evidence, no representative DA regression and no repeated sin DA regression; dec median paired runtime reduction >=5%, at least 2/3 faster pairs, and lower total time across all 3 pairs; otherwise retain off globally; never select by design',
  remaining_stages='freeze one global off/on; use same-source representative r0 references in full20 coverage, run remaining designs; independently audit/wire three method-module ablations before dispatch; then paper tables/conclusions/DAC6+1 QA. No new optimization components.')
def execute(c,j,o,locks):
 inp=c['subjects'][j['design']];case=o/'cases'/j['key'];assert not case.exists(),'unrecorded partial: audit before recovery, never overwrite'
 assert shutil.disk_usage(R).free>16*(1<<30),'pre-dispatch storage guard, not candidate or memory allowance'
 modes=('line','move')if j['variant']=='on'else('fragmented','copy')
 argv=['taskset','-c','6',sys.executable,str(R/'scripts/emapper/run.py'),inp['prepared'],'--golden',inp['golden'],'--output',str(case),*c['common'],'--trial-safe-area','off','--trial-memo','off','--trial-timing-dependencies','on','--trial-fixed-area','on','--trial-release-reuse','off','--trial-delta-cost','off','--trial-timing-scratch','on','--selected-witness-only','on','--evidence-transfer','move','--publication-receiver','events','--publication-timing','--physical-stamp','construct','--trace-write',modes[0],'--evidence-arrays',modes[1]]
 assert not any(a in argv for a in ['--seed-verilog','--fresh-emap-seed','--replay-snapshot'])
 nonce=hashlib.sha256(str(case).encode()).hexdigest();env={k:v for k,v in os.environ.items()if not k.startswith('EGM_')};env.update(EMAPPER_AFFINITY_RECEIPT=str(case/'native-affinity.json'),EMAPPER_AFFINITY_NONCE=nonce)
 save(o/(j['key']+'-attempt.json'),dict(argv=argv,input=inp,nonce=nonce,policy=c['policy']))
 stop=threading.Event();samples=o/(j['key']+'-samples.jsonl');t=threading.Thread(target=sample,args=(case,stop,samples),daemon=True);t.start()
 try:
  with(o/(j['key']+'.log')).open('x')as log:p=subprocess.run(argv,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(f.fileno()for f in locks))
 finally:stop.set();t.join()
 row=dict(**j,case=str(case),exit=p.returncode,status='unmeasured_failure',sample_file=str(samples),sample_sha256=sha(samples))
 if(case/'report.json').exists():
  report=read(case/'report.json');row.update(report_status=report.get('status'),error=report.get('error'),T=report.get('mapping_wall_secs'),CEC=report.get('cec'),search_timeout=report.get('search_timeout'),search_finished=report.get('search_finished'),first_native_cover=report.get('first_native_cover'))
  if p.returncode==0 and report.get('status')=='ok':
   cfg=report['configuration'];assert cfg['mapping_seconds']==600 and cfg['objective']=='delay-area'and cfg['initial_cover_policy']=='direct'and cfg['trace_write']==modes[0]and cfg['evidence_arrays']==modes[1]and cfg['physical_stamp']=='construct'
   assert report['direct_native']and not report.get('external_seed')and not report.get('seed_generation_in_mapping_clock')
   assert report['cec']==1 and report['evaluation']['exact_agreement']and report['selected_evidence']['valid']and report['selected_evidence']['netlist_binding_agreement']
   aff=read(case/'native-affinity.json');assert aff['nonce']==nonce and aff['cpus_allowed_list']=='6'
   clock=timing_case(case);save(o/(j['key']+'-clock.json'),clock);assert clock['qualified_complete_monotonic_chain']and not clock['late_receipt_sequences']
   frames,tail=observations(case/'observation.jsonl');last=next((f for f in reversed(frames)if f.get('details',{}).get('selected_evidence',{}).get('cost')is not None),None)
   row.update(status='ok',A=report['measured']['area'],D=report['measured']['delay'],configuration=cfg,output_sha256=report['output_sha256'],observation_tail=tail,qualified_clock_chain=True)
   if last:row['observed_prefix']=dict(elapsed=last['elapsed_seconds'],exclusive_seconds=last['exclusive_seconds'],details=last['details'])
   # Initial cover is a native intermediate, not an external seed. Independently
   # check it outside the mapping clock before reporting a search net gain.
   from physical import measure_cover
   from verify import verify
   from selected_evidence import replay
   first=report['first_native_cover'];lib=R/'emapper/libraries/asap7/asap7_full.genlib';tools=report['tools'];check=case/'first-cover-validation'
   initial=measure_cover(lib,pathlib.Path(first['netlist']),check,R/'target/release/emapper_measure',tools['yosys']['path'])
   saved={k:os.environ.get(k)for k in ['EGM_YOSYS','BENCH_RS_EMAP_ABC']}
   try:
    os.environ.update(EGM_YOSYS=tools['yosys']['path'],BENCH_RS_EMAP_ABC=tools['abc']['path'])
    first_cec=verify(lib.parent/'cells.v',pathlib.Path(inp['golden']),pathlib.Path(initial['netjson']),check/'cec')
   finally:
    for k,v in saved.items():
     if v is None:os.environ.pop(k,None)
     else:os.environ[k]=v
   proof=replay(pathlib.Path(first['manifest']),case/'selected-provenance/objects',pathlib.Path(initial['netjson']))
   initial_ad=[initial['measured']['area'],initial['measured']['delay']]
   assert first_cec and proof['valid']and proof['netlist_binding_agreement']and initial_ad==[first['area'],first['delay']]
   assert(row['D'],row['A'])<=(initial_ad[1],initial_ad[0])
   row.update(initial_AD=initial_ad,first_independent_CEC=True,first_selected_evidence=proof,first_cover_time=first['mapping_elapsed_seconds'],search_net_gain='better'if(row['D'],row['A'])<(initial_ad[1],initial_ad[0])else'equal')
  else:row['status']='mapping_failure'
 row['artifacts']={str(p.relative_to(case)):sha(p)for p in sorted(case.rglob('*'))if p.is_file()};return row
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,default=DEFAULT);ap.add_argument('--phase',choices=['representatives','repeats'],default='representatives');ap.add_argument('--max-jobs',type=int,default=2);a=ap.parse_args();assert a.max_jobs>0
 o=a.output.resolve();o.mkdir(parents=True,exist_ok=True);locks=[]
 for p in [R/'results_v2/independent_components_20261019/global.lock',R/'results_v2/emapper_physical_stamp_20261111/benchmark/runner.lock',o/'runner.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 if(o/'campaign.json').exists():c=read(o/'campaign.json')
 else:
  c=make_campaign();save(o/'campaign.json',c)
  for n in c['frozen']:
   dst=o/'frozen'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/n,dst)
 assert c['frozen']==fingerprint(),'frozen source/binary/harness drift; do not mix versions'
 for r in c['jobs'].values():
  assert r['status']!='harness_invalid'
  assert all(sha(pathlib.Path(r['case'])/n)==h for n,h in r['artifacts'].items())and sha(pathlib.Path(r['sample_file']))==r['sample_sha256']
 if a.phase=='repeats':assert all(j['key']in c['jobs']for j in c['plans']['representatives']),'finish representative scope before repeats'
 done=0
 for j in c['plans'][a.phase]:
  if j['key']in c['jobs']:continue
  if done>=a.max_jobs:break
  assert c['frozen']==fingerprint();save(o/'state.json',dict(active=j['key'],completed=len(c['jobs']),phase=a.phase,baseline_runs=0,default_promoted=False));print('start',j['key'],flush=True)
  try:r=execute(c,j,o,locks)
  except Exception as e:
   save(o/(j['key']+'-audit-failure.json'),dict(error=repr(e),partial_preserved=True));raise
  completed=o/(j['key']+'-completed.json');assert not completed.exists();save(completed,r);assert c['frozen']==fingerprint();c['jobs'][j['key']]=r;save(o/'campaign.json',c);done+=1
  print('end',j['key'],r['status'],r.get('T'),r.get('A'),r.get('D'),flush=True)
 save(o/'state.json',dict(active=None,completed=len(c['jobs']),phase=a.phase,planned_representatives=8,planned_repeats=8,baseline_runs=0,default_promoted=False))
if __name__=='__main__':main()
