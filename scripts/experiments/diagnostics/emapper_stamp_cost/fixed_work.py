"""One frozen native snapshot, identical cold recovery; no full-mapping PPA claim."""
import argparse, fcntl, hashlib, json, os, pathlib, subprocess, sys, threading, time
from resource_pair import R, sample
OLD=R/'results_v2/emapper_physical_stamp_20261111/benchmark'
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--snapshot',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);args=ap.parse_args();O=args.output.resolve();snap=args.snapshot.resolve();producer=snap.parent
 O.mkdir(parents=True,exist_ok=True);locks=[]
 for path in [R/'results_v2/independent_components_20261019/global.lock',OLD/'runner.lock',O/'runner.lock']:
  f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 # An already atomic snapshot is read-only input. This driver NEVER dispatches
 # a producer, so no detached frontend can outlive a diagnostic stop.
 old=json.loads((OLD/'dec_r1_control-attempt.json').read_text())
 assert not (O/'snapshot-identity.json').exists(),'existing cohort: audit, never overwrite'
 frozen=json.loads(snap.read_text());save(O/'snapshot-identity.json',dict(sha256=sha(snap),pool=len(frozen['pool']),initial_AD=frozen['initial_ad'],context=frozen['context']))
 rows=[]
 for variant in ['construct','borrow']:
  case=O/variant;case.mkdir();env=dict(os.environ)
  # Snapshot paths are redirected, never write into producer or original cohort.
  for k,v in frozen['environment'].items():env[k]=v.replace(str(producer),str(case))
  for k in list(env):
   if k in ['EGM_RECOVERY_SNAPSHOT','EGM_RECOVERY_SNAPSHOT_AFTER','EGM_SOURCE_BUNDLE','EGM_EXTERNAL_BUNDLE','EGM_PUBLICATION_EVENT_FD']:env.pop(k)
  env.update(EGM_TOTAL_TIME_ONLY='1',EGM_TOTAL_REMAINING_SECS='600',EGM_MAPPING_DEADLINE_MONOTONIC=str(time.monotonic()+600),EGM_SELECTED_PHYSICAL_BORROW='1' if variant=='borrow' else '0',EMAPPER_AFFINITY_RECEIPT=str(case/'native-affinity.json'),EMAPPER_AFFINITY_NONCE='fixed-work-'+variant)
  argv=['taskset','-c','6',str(R/'target/release/emapper_kernel'),'--replay-recovery',str(snap),str(case/'mapped.v'),'legacy']
  save(case/'command.json',dict(argv=argv,environment={k:v for k,v in env.items() if k.startswith('EGM_')},snapshot_sha256=sha(snap),kernel_sha256=sha(R/'target/release/emapper_kernel')))
  stop=threading.Event();t=threading.Thread(target=sample,args=(case,stop,O/(variant+'-samples.jsonl')),daemon=True);t.start();start=time.monotonic()
  try:
   with (case/'run.log').open('x') as log:p=subprocess.run(argv,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(.001,float(env['EGM_MAPPING_DEADLINE_MONOTONIC'])-time.monotonic()),pass_fds=tuple(f.fileno() for f in locks))
   result=dict(variant=variant,exit=p.returncode,wall=time.monotonic()-start)
  except subprocess.TimeoutExpired:result=dict(variant=variant,timeout=True,wall=time.monotonic()-start)
  finally:stop.set();t.join()
  if (case/'mapped.replay.json').exists():result['replay']=json.loads((case/'mapped.replay.json').read_text());result['netlist_sha256']=sha(case/'mapped.v')
  if result.get('exit')==0:
   # Original independent named-port verifier; all conversion/CEC outside replay clock.
   sys.path.insert(0,str(R/'scripts/emapper'));from verify import verify
   library=R/'emapper/libraries/asap7/asap7_full.genlib'
   tools=json.loads((OLD/'cases/dec_r1_control/report.json').read_text())['tools']
   yosys,abc=tools['yosys']['path'],tools['abc']['path']
   cmd=[yosys,'-Q','-T','-p',f'read_verilog -lib "{library.parent / "cells.v"}"; read_verilog "{case / "mapped.v"}"; hierarchy -check -auto-top; rename -top mapped; write_json "{case / "mapped.json"}"']
   normalized=subprocess.run(cmd,capture_output=True,text=True);(case/'normalize.log').write_text(normalized.stdout+normalized.stderr);normalized.check_returncode()
   measured=subprocess.run([str(R/'target/release/emapper_measure'),str(library),str(case/'mapped.json')],capture_output=True,text=True);measured.check_returncode();result['public_AD']=json.loads(measured.stdout)
   os.environ['EGM_YOSYS'],os.environ['BENCH_RS_EMAP_ABC']=yosys,abc
   result['CEC']=verify(library.parent/'cells.v',pathlib.Path(old['input']['golden']),case/'mapped.json',case/'cec')
  rows.append(result);save(O/'progress.json',rows);print(variant,result.get('exit'),result['wall'],result.get('CEC'),flush=True)
 save(O/'result.json',dict(rows=rows,scope='same frozen pool/bank/chosen/proof snapshot; cold-start legacy recovery, generation and flow disabled; compare only completed equal-work replays; CEC must be separately checked'))
if __name__=='__main__':main()
