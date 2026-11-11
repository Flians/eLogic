"""Instruction-count diagnostic of existing sin code, NOT native PPA/runtime.

Uses installed Callgrind directly (no snap settings/capabilities changed). Periodic
dumps preserve partial profiles when the unchanged 600-second deadline is reached.
"""
import fcntl, hashlib, json, os, pathlib, subprocess, time
from resource_pair import R
O=R/'results_v2/emapper_stamp_diagnosis_20261111/sin-callgrind'
V='/snap/valgrind/current/usr/bin/valgrind';VL='/snap/valgrind/current/usr/libexec/valgrind'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=True);locks=[]
 for p in [R/'results_v2/independent_components_20261019/global.lock',R/'results_v2/emapper_physical_stamp_20261111/benchmark/runner.lock',O/'runner.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 assert not (O/'command.json').exists(),'preserve existing diagnostic'
 old=R/'results_v2/emapper_physical_stamp_20261111/benchmark/cases/sin_r1_borrow'
 report=json.loads((old/'report.json').read_text());env=dict(os.environ,VALGRIND_LIB=VL)
 for k,v in report['environment'].items():env[k]=v.replace(str(old),str(O))
 env.update(EGM_TOTAL_REMAINING_SECS='600',EGM_MAPPING_DEADLINE_MONOTONIC=str(time.monotonic()+600),EGM_INCUMBENT_DIR=str(O/'incumbents'),EGM_STAGE_TRACE=str(O/'online.stages.jsonl'),EGM_POOL_DIAG_PATH=str(O/'online.pool.json'),EGM_NATIVE_PUBLICATION_TIMING=str(O/'native.publication-timing.jsonl'),EMAPPER_AFFINITY_RECEIPT=str(O/'native-affinity.json'),EMAPPER_AFFINITY_NONCE='sin-callgrind-diagnostic')
 argv=['taskset','-c','6',V,'--tool=callgrind','--collect-systime=nsec','--dump-every-bb=100000000','--callgrind-out-file='+str(O/'callgrind.%p'),str(R/'target/release/emapper_kernel'),str(R/'emapper/libraries/asap7/asap7_full.genlib'),str(R/'results_v2/epfl_resyn2_subjects_20261012/sin.resyn2.aig'),str(O/'worker.v')]
 (O/'command.json').write_text(json.dumps(dict(argv=argv,environment={k:v for k,v in env.items()if k.startswith('EGM_')or k=='VALGRIND_LIB'},kernel_sha256=sha(R/'target/release/emapper_kernel'),source_report_sha256=sha(old/'report.json'),scope='Callgrind simulated instruction counts, not native wall cost, PPA or complete-search claim; no frontend receiver in this diagnostic; 600 seconds total, no CEC'),indent=2)+'\n')
 start=time.monotonic()
 with (O/'run.log').open('x') as log:
  try:p=subprocess.run(argv,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(.001,float(env['EGM_MAPPING_DEADLINE_MONOTONIC'])-time.monotonic()),pass_fds=tuple(f.fileno()for f in locks));result=dict(exit=p.returncode)
  except subprocess.TimeoutExpired:result=dict(external_deadline=True)
 result.update(wall=time.monotonic()-start,profiles=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)for p in sorted(O.glob('callgrind.*'))if p.is_file()])
 (O/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
