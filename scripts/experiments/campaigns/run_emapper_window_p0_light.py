#!/usr/bin/env python3
"""P0 light-cost sin/max control, corrected TT4 runtime; no full lifecycle audit.

One unchanged binary; all other parameters/inputs/library/600s identical. No
baseline runs. Finite batches and full artifact rechecks; unknown partials stop.
"""
import argparse,copy,csv,fcntl,json,os,shutil,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'scripts/emapper'))
from provenance import source_files,hashes,sha,git_revision
from physical import measure_cover
from verify import verify
from archive import seal,verify_files
ORDER=['sin','max']
COMMON=['--seed-native','editable','--ablation','source-editable','--recovery-inputs','overlay','--recovery-noop-prefilter','on','--recovery-cache','legacy','--recovery-schedule','legacy','--area-reject-cache','off','--observe','--snapshot-after','540']
ARMS={'legacy':0,'window8':8}
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def fingerprint():
 paths=source_files()+list((R/'scripts/emapper').glob('*.py'))+[Path(__file__),R/'target/release/emapper-build.json',R/'target/release/emapper_kernel',R/'target/release/emapper_measure',R/'emapper/profiles/delivery.json']+list((R/'emapper/libraries/asap7').glob('*'))
 return hashes([p for p in paths if p.is_file()])
def initialize():
 protocol=R/'results_v2/emapper_seeded_epfl_20261101/protocol.json';all_inputs=read(protocol)['inputs'];inputs={name:all_inputs[name]for name in ORDER}
 for inp in inputs.values():
  for key in ['prepared','golden','seed']:assert sha(inp[key])==inp[key+'_sha256']
 return dict(schema=1,source_commit=git_revision(),frozen=fingerprint(),designs=ORDER,arms=ARMS,jobs={},expected=4,repetitions=1,inputs=inputs,source_protocol=str(protocol),source_protocol_sha256=sha(protocol),common_cli=COMMON,mapping_seconds=600,CEC_in_clock=False,AIG_rounds=5,alpha=10,wide_rounds=2,beta=2,baseline_runs=0,GitHub_push=False,scope='P0 diagnostic sin/max only; aggregate observation I/O; full candidate/cut audit disabled and snapshot at existing recovery boundary after540s online are charged to600s; no extra time; cold pool replay is separate and not full mapping PPA; historical seeds reused, no baseline runs',tools={k:dict(path=shutil.which(k),sha256=sha(shutil.which(k)))for k in ['yosys','abc']})
def validate(j,deep=False):
 assert sha(j['report'])==j['report_sha256']
 if deep:verify_files(Path(j['case']),j['artifacts'],j.get('capsule'))
 return read(j['report'])
def summarize(out,c):
 rows=[];pairs=[]
 for name in c['designs']:
  values={}
  for arm in c['arms']:
   key=name+'_'+arm
   if key not in c['jobs']:rows.append(dict(design=name,arm=arm,status='missing'));continue
   j=c['jobs'][key];r=validate(j);m=r.get('measured',{});row=dict(design=name,arm=arm,status=j['status'],CEC=r.get('cec'),initial_CEC=j.get('initial',{}).get('CEC'),A=m.get('area'),D=m.get('delay'),T=r.get('mapping_wall_secs'),deadline=r.get('search_timeout'),search_finished=r.get('search_finished'),search_vs_import=r.get('native_search_net_gain',{}).get('outcome'),error=j.get('error',r.get('error')));rows.append(row)
   if j['status']in ('ok','source_only') and row['CEC']==1 and row['initial_CEC']==1:values[arm]=row
  if set(values)==set(ARMS):
   a=values['legacy'];b=values['window8'];left=(b['D'],b['A']);right=(a['D'],a['A']);pairs.append(dict(design=name,DA='win' if left<right else 'loss'if left>right else'tie',A_change=b['A']-a['A'],D_change=b['D']-a['D'],legacy_A=a['A'],window_A=b['A'],legacy_D=a['D'],window_D=b['D'],legacy_T=a['T'],window_T=b['T']))
 for filename,items in [('current.csv',rows),('paired.csv',pairs)]:
  keys=list(dict.fromkeys(k for row in items for k in row))
  with(out/filename).open('w')as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(items)
 dump(out/'summary.json',dict(completed=len(c['jobs']),expected=c['expected'],complete=len(c['jobs'])==c['expected'],output_CEC=sum(r.get('CEC')==1 for r in rows),initial_CEC=sum(r.get('initial_CEC')==1 for r in rows),valid_pairs=len(pairs),DA={k:sum(x['DA']==k for x in pairs)for k in ['win','tie','loss']},missing=sum(r['status']=='missing'for r in rows),invalid=sum(r['status']not in ['ok','source_only','missing']for r in rows),baseline_runs=0,GitHub_push=False))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-jobs',type=int,default=2);ap.add_argument('--verify-artifacts-only',action='store_true');a=ap.parse_args();assert a.max_jobs>0
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);locks=[]
 for p in [R/'results_v2/independent_components_20261019/global.lock',out/'runner.lock']:
  p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 c=read(out/'campaign.json')if(out/'campaign.json').exists()else initialize();assert c['frozen']==fingerprint(),'source/binary/harness drift';dump(out/'campaign.json',c)
 for j in c['jobs'].values():validate(j,a.verify_artifacts_only)
 if a.verify_artifacts_only:
  dump(out/'artifact-validation.json',dict(campaign_sha256=sha(out/'campaign.json'),cases=len(c['jobs']),all_hashes_match=True,files=sum(len(j['artifacts'])for j in c['jobs'].values())));summarize(out,c);return
 count=0
 for ix,name in enumerate(c['designs']):
  for arm in list(ARMS)if ix%2==0 else list(reversed(ARMS)):
   key=name+'_'+arm
   if key in c['jobs']:continue
   if count>=a.max_jobs:summarize(out,c);dump(out/'state.json',dict(status='batch_complete',completed=len(c['jobs']),expected=c['expected']));return
   assert c['frozen']==fingerprint()
   for t in c['tools'].values():assert sha(t['path'])==t['sha256']
   inp=c['inputs'][name]
   for k in ['prepared','golden','seed']:assert sha(inp[k])==inp[k+'_sha256']
   case=out/'cases'/key;assert not case.exists(),'unrecorded partial: audit and retain before recovery'
   argv=[sys.executable,str(R/'scripts/emapper/run.py'),inp['prepared'],'--golden',inp['golden'],'--seed-verilog',inp['seed'],'--output',str(case),*COMMON,'--sdc-window-inputs',str(ARMS[arm])]
   dump(out/(key+'-attempt.json'),dict(argv=argv,frozen=c['frozen'],input=inp));dump(out/'state.json',dict(status='running',active=key,completed=len(c['jobs']),expected=c['expected']));print(json.dumps(dict(event='start',key=key)),flush=True)
   with(out/(key+'.log')).open('x')as log:p=subprocess.run(argv,cwd=R,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(f.fileno()for f in locks))
   report=case/'report.json';r=read(report);assert r['status']!='running','retain interrupted partial report'
   j=dict(case=str(case),report=str(report),report_sha256=sha(report),runner_exit=p.returncode,status=r['status'])
   if not r.get('external_native_import',{}).get('applied'):j.update(status='native_import_failed',error='outer fallback is not native seed import')
   if r.get('external_native_import',{}).get('applied'):
    try:
     m=read(case/'incumbents/00000001/manifest.json');assert m['kind']=='complete-source-cover'
     initial=measure_cover(R/'emapper/libraries/asap7/asap7_full.genlib',case/'incumbents/00000001/mapped.v',case/'native-initial',R/'target/release/emapper_measure',shutil.which('yosys'))
     os.environ['EGM_YOSYS']=shutil.which('yosys');os.environ['BENCH_RS_EMAP_ABC']=shutil.which('abc')
     initial['CEC']=int(verify(R/'emapper/libraries/asap7/cells.v',Path(inp['golden']),Path(initial['netjson']),case/'native-initial/cec'));initial['exact_AD_agreement']=all(initial['measured'][k]==m[k]for k in ['area','delay']);j['initial']=initial
     if initial['CEC']!=1 or not initial['exact_AD_agreement']:j['status']='initial_invalid'
    except Exception as e:j.update(status='initial_failed',error=str(e))
   diagnostic=dict(scope='P0 only: cold fixed-pool replay evidence; all observation overhead inside600s',errors=[])
   for artifact_name in ['observation.jsonl','recovery-snapshot.json']:
    path=case/artifact_name
    if not path.exists():diagnostic['errors'].append('missing '+artifact_name);continue
    diagnostic[artifact_name]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
   if (case/'recovery-snapshot.json').exists():
    try:
     snapshot=read(case/'recovery-snapshot.json');assert snapshot['schema']==1 and snapshot['chosen'] and snapshot['context']
     assert snapshot['environment']['EGM_DIAG_NATIVE_SHA256']==c['frozen']['target/release/emapper_kernel']
     diagnostic['snapshot_summary']=dict(pool=len(snapshot['pool']),cuts=sum(len(b)for b in snapshot['bank']),initial_AD=snapshot['initial_ad'],context=snapshot['context'])
    except Exception as error:diagnostic['errors'].append(type(error).__name__+': '+str(error))
   j['diagnostic']=diagnostic
   assert c['frozen']==fingerprint(),'source drift during timed job'
   j['artifacts']={str(p.relative_to(case)):sha(p)for p in sorted(case.rglob('*'))if p.is_file()};j['capsule']=seal(case,out/'capsules');verify_files(case,j['artifacts'],j['capsule']);c['jobs'][key]=j;dump(out/'campaign.json',c);summarize(out,c);count+=1
   print(json.dumps(dict(event='end',key=key,status=j['status'],CEC=r.get('cec'),AD=r.get('measured'))),flush=True)
 summarize(out,c);dump(out/'state.json',dict(status='complete',completed=len(c['jobs']),expected=c['expected']))
if __name__=='__main__':main()
