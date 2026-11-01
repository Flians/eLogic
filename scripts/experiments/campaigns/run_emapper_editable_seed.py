#!/usr/bin/env python3
"""Fixed EPFL20 same-binary outer/editable seed comparison; no baseline runs.

Imported-cover CEC is separate from delivered-cover CEC. All definitions/proofs,
logical admission, matching and recovery share the unmodified600-second clock.
A finite batch completes in this invocation; partial cases stop rather than overwrite.
"""
import argparse,csv,fcntl,json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'scripts/emapper'))
from provenance import source_files,hashes,sha,git_revision
from physical import measure_cover
from verify import verify
import os,shutil
ORDER=['ctrl','priority','bar','sqrt','sin','voter','i2c','adder','max','arbiter','cavlc','dec','int2float','router','multiplier','square','log2','mem_ctrl','div','hyp']
COMMON=['--ablation','source-editable','--recovery-inputs','overlay','--recovery-noop-prefilter','on','--recovery-cache','legacy','--recovery-schedule','legacy','--area-reject-cache','off']
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def fingerprint():
 paths=source_files()+list((R/'scripts/emapper').glob('*.py'))+[Path(__file__),R/'target/release/emapper-build.json',R/'target/release/emapper_kernel',R/'target/release/emapper_measure',R/'emapper/profiles/delivery.json']+list((R/'emapper/libraries/asap7').glob('*'))
 return hashes([p for p in paths if p.is_file()])
def initialize(out,stage):
 old=R/'results_v2/emapper_seeded_epfl_20261101';protocol=read(old/'protocol.json');assert set(protocol['inputs'])==set(ORDER)
 for inp in protocol['inputs'].values():
  for side in ['prepared','golden','seed']:assert sha(inp[side])==inp[side+'_sha256']
 designs=ORDER if stage=='full'else ['priority','sqrt','sin','voter','bar','i2c']
 repeats=1 if stage=='full'else 3
 return dict(schema=1,stage=stage,repetitions=repeats,designs=designs,source_commit=git_revision(),frozen=fingerprint(),inputs=protocol['inputs'],source_protocol=str(old/'protocol.json'),source_protocol_sha256=sha(old/'protocol.json'),common_cli=COMMON,order=ORDER,arms=['outer','editable'],expected=len(designs)*2*repeats,jobs={},tools={k:dict(path=shutil.which(k),sha256=sha(shutil.which(k)))for k in ['yosys','abc']},baseline_runs=0,mapping_seconds=600,CEC_in_clock=False,seed_generation='historical supplied emap-delay; never rerun; T is refinement only',policy='one shared original-input-derived cumulative node allowance; no refund or renewed stages',AIG_rounds=5,alpha=10,wide_rounds=2,beta=2,GitHub_push=False)
def validate(j):
 assert sha(j['report'])==j['report_sha256']
 for p,h in j['artifacts'].items():assert sha(p)==h,p
 return read(j['report'])
def summary(out,c):
 rows=[]
 for repetition in range(c['repetitions']):
  for name in c['designs']:
   for arm in c['arms']:
    key=f'r{repetition}_{name}_{arm}'
    if key not in c['jobs']:rows.append(dict(repetition=repetition,design=name,arm=arm,status='missing'));continue
    j=c['jobs'][key];r=validate(j);v=r.get('measured',{});seed=r.get('external_seed',{}).get('measured',{});initial=r.get('editable_source',{}).get('native_AD',[None,None]);net=r.get('native_search_net_gain',{})
    rows.append(dict(repetition=repetition,design=name,arm=arm,status=j['status'],CEC=r.get('cec'),seed_A=seed.get('area'),seed_D=seed.get('delay'),imported_A=initial[0]if arm=='editable'else None,imported_D=initial[1]if arm=='editable'else None,A=v.get('area'),D=v.get('delay'),T=r.get('mapping_wall_secs'),import_applied=r.get('external_native_import',{}).get('applied')if arm=='editable'else None,initial_native_CEC=j.get('native_initial',{}).get('CEC'),search_vs_seed=r.get('search_vs_seed',{}).get('outcome'),search_vs_import=net.get('outcome'),deadline=r.get('search_timeout'),error=r.get('error',j.get('error'))))
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with(out/'current.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 dump(out/'summary.json',dict(completed=len(c['jobs']),expected=c['expected'],output_CEC=sum(x.get('CEC')==1 for x in rows),external_imports=sum(x.get('import_applied')is True for x in rows),native_initial_CEC=sum(x.get('initial_native_CEC')==1 for x in rows),results=rows,complete=len(c['jobs'])==c['expected'],paper_updated=False,baseline_runs=0,GitHub_push=False))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-jobs',type=int,default=2);ap.add_argument('--stage',choices=['full','repeats'],default='full');a=ap.parse_args();assert a.max_jobs>0
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);locks=[]
 for path in [R/'results_v2/independent_components_20261019/global.lock',out/'runner.lock']:
  path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 c=read(out/'campaign.json')if(out/'campaign.json').exists()else initialize(out,a.stage);dump(out/'campaign.json',c)
 assert c['stage']==a.stage
 assert c['frozen']==fingerprint(),'frozen source/binary/configuration drift'
 for j in c['jobs'].values():validate(j)
 count=0
 for repetition in range(c['repetitions']):
  for ix,name in enumerate(c['designs']):
   for arm in c['arms']if (ix+repetition)%2==0 else reversed(c['arms']):
    key=f'r{repetition}_{name}_{arm}'
    if key in c['jobs']:continue
    if count>=a.max_jobs:
     summary(out,c);dump(out/'state.json',dict(status='batch_complete',completed=len(c['jobs']),expected=c['expected']));return
    assert fingerprint()==c['frozen']
    for tool in c['tools'].values():assert sha(tool['path'])==tool['sha256']
    inp=c['inputs'][name]
    for side in ['prepared','golden','seed']:assert sha(inp[side])==inp[side+'_sha256']
    case=out/'cases'/key;assert not case.exists(),'unrecorded partial: retain and audit before recovery'
    argv=[sys.executable,str(R/'scripts/emapper/run.py'),inp['prepared'],'--golden',inp['golden'],'--seed-verilog',inp['seed'],'--seed-native',arm,'--output',str(case),*COMMON]
    dump(out/(key+'-attempt.json'),dict(argv=argv,source_commit=c['source_commit'],frozen=c['frozen'],input=inp))
    dump(out/'state.json',dict(status='running',active=key,completed=len(c['jobs']),expected=c['expected']));print(json.dumps(dict(event='start',key=key)),flush=True)
    with(out/(key+'.log')).open('x')as log:p=subprocess.run(argv,cwd=R,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(f.fileno()for f in locks))
    report=case/'report.json';r=read(report);assert r['status']!='running','interrupted partial report, do not adopt as complete'
    j=dict(report=str(report),report_sha256=sha(report),runner_exit=p.returncode,status=r['status'])
    if arm=='editable'and not r.get('external_native_import',{}).get('applied'):j.update(status='native_import_failed',error='legal outer fallback is not an editable seed import')
    # Independent CEC and common physical evaluation of the native initial seed,
    # not just the outer delivered seed. No matching or remapping is invoked.
    if arm=='editable'and r.get('external_native_import',{}).get('applied'):
     try:
      manifest=read(case/'incumbents/00000001/manifest.json');assert manifest['kind']=='complete-source-cover'
      initial=measure_cover(R/'emapper/libraries/asap7/asap7_full.genlib',case/'incumbents/00000001/mapped.v',case/'native-initial',R/'target/release/emapper_measure',shutil.which('yosys'))
      os.environ['EGM_YOSYS']=shutil.which('yosys');os.environ['BENCH_RS_EMAP_ABC']=shutil.which('abc')
      initial['CEC']=int(verify(R/'emapper/libraries/asap7/cells.v',Path(inp['golden']),Path(initial['netjson']),case/'native-initial/cec'))
      initial['exact_AD_agreement']=all(initial['measured'][k]==manifest[k]for k in ['area','delay']);j['native_initial']=initial
      if initial['CEC']!=1 or not initial['exact_AD_agreement']:j['status']='native_initial_invalid'
     except Exception as exc:j.update(status='native_initial_failed',error=str(exc))
    assert c['frozen']==fingerprint(),'source changed during timed case'
    j['artifacts']={str(p):sha(p)for p in sorted(case.rglob('*'))if p.is_file()};c['jobs'][key]=j;dump(out/'campaign.json',c);summary(out,c)
    print(json.dumps(dict(event='end',key=key,status=j['status'],CEC=r.get('cec'),imported=r.get('external_native_import'),AD=r.get('measured'))),flush=True);count+=1
 dump(out/'state.json',dict(status='complete',completed=len(c['jobs']),expected=c['expected']))
if __name__=='__main__':main()
