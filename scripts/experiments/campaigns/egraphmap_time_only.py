#!/usr/bin/env python3
"""Time-only execution: source construction and native mapping share600 seconds.
No node/pool/iteration caps or independent native/stage clocks are imposed. A complete
source incumbent survives any failed attempt; external original CEC is required.
"""
import json,sys,os,time,math,shutil,subprocess,resource,hashlib
from pathlib import Path
from source_cover import build_source_cover

def better_da(candidate,incumbent):
 t=1e-6*max(1.,abs(candidate['delay']),abs(incumbent['delay']))
 if candidate['delay']<incumbent['delay']-t:return True
 if abs(candidate['delay']-incumbent['delay'])>t:return False
 return candidate['area']<incumbent['area']-1e-6*max(1.,abs(candidate['area']),abs(incumbent['area']))

def main():
 start=time.monotonic();deadline=float(os.environ.get("EGM_MAPPING_DEADLINE_MONOTONIC",start+600));constructor_only=os.environ.get("EGM_CONSTRUCTOR_ONLY","0")=="1"
 if len(sys.argv)!=4:raise ValueError('expected genlib prepared-AIG output-Verilog')
 library,subject,out=map(Path,sys.argv[1:]);child=Path(os.environ['EGM_NATIVE_OPTIMIZER']);models=library.parent/'library.json';meta=out.with_suffix('.complete.json');source=out.with_suffix('.source.v');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 text,seed=build_source_cover(subject.read_bytes(),models,deadline,phase_aware=True);source.write_text(text)
 bank_limit=global_pool=global_nodes=(1<<64)-1;shutil.copyfile(source,out)
 info=dict(policy='time-only600-v1',native_search_strategy='one global600s clock, no configured node/pool/cut/iteration/helper ceiling; existing matching algorithms unchanged',full_candidate_debug_report_disabled=True,search_ratio=None,primitive_input_node_bound=int(os.environ['EGM_INPUT_NODE_BOUND']),proportional_search_allowance=None,mapping_limit_secs=600,optimizer_shared_window_secs=None,constructor_only=constructor_only,optimizer_recovery_secs=None,global_cumulative_node_allowance=None,optimizer_pool_limit=None,global_retained_candidate_limit=None,source=seed,source_verilog=str(source),source_sha256=sha(source),subject_sha256=sha(subject),library_sha256=sha(library),library_models_sha256=sha(models),native_optimizer=str(child),native_optimizer_sha256=sha(child),frontend_sha256=sha(__file__),optimizer_succeeded=False,selected='source_witness',unconditional_CEC_required=True,old_fixed20k_protocol=False,time_only=True,count_limits=None,per_stage_time_limits=None,optimizer_address_space_limit_bytes=None,attempts=[],no_refund_on_retry=True,source_construction_is_inside600s=True)
 chosen=source;metrics=dict(area=seed['area'],delay=seed['delay'],cec=-1,status='source-witness-fallback',proven_optimal=False,egraph_cover_selected=False);opt_end=time.monotonic() if constructor_only else deadline;accounted=0
 meta.write_text(json.dumps(dict(info,selected_verilog_sha256=sha(out),mapping_complete=False),indent=2))
 for index in range(1):
  limit=global_nodes-accounted
  remaining=opt_end-time.monotonic()
  if remaining<=0 or limit<=0:break
  prefix=out.parent/f'attempt{index}';worker=prefix.with_suffix('.v');pool=prefix.with_suffix('.pool.json');diag=prefix.with_suffix('.candidates.json');trace=prefix.with_suffix('.trace.json');ledger=pool.with_suffix('.admission.json')
  env=dict(os.environ,EGM_NOCEC='1',EGM_NODES=str(limit),EGM_POOL_DIAG_PATH=str(pool),EGM_RECOVERY_TRACE=str(trace),EGM_STAGE_TRACE=str(prefix.with_suffix('.stages.jsonl')));env.pop('EGM_DIAG_PATH',None);env.update(EGM_TOTAL_TIME_ONLY='1',EGM_TOTAL_REMAINING_SECS=str(remaining));argv=[str(child),str(library),str(subject),str(worker)]
  info['active_attempt']=dict(argv=argv,remaining_total_seconds=remaining,stdout_path=str(prefix.with_suffix('.stdout')),stderr_path=str(prefix.with_suffix('.stderr')),independent_native_guard=None)
  meta.write_text(json.dumps(dict(info,selected_verilog_sha256=sha(out),mapping_complete=False),indent=2))
  t=time.monotonic();timeout=False
  with prefix.with_suffix('.stdout').open('w')as stdout_file,prefix.with_suffix('.stderr').open('w')as stderr_file:
   p=subprocess.Popen(argv,env=env,stdout=stdout_file,stderr=stderr_file,text=True)
   try:p.wait(timeout=max(0,deadline-time.monotonic()))
   except subprocess.TimeoutExpired:timeout=True;p.kill();p.wait()
  stdout=prefix.with_suffix('.stdout').read_text();stderr=prefix.with_suffix('.stderr').read_text();a=json.loads(ledger.read_text())if ledger.exists()else None
  info.pop('active_attempt',None)
  # A killed attempt without a final ledger is conservatively charged its full
  # reservation. The next attempt never receives a fresh independent budget.
  charge=a['cumulative_admitted_enodes']if a else limit;assert 0<=charge<=limit;accounted+=charge;assert accounted<=global_nodes
  records=[json.loads(l)for l in stdout.splitlines()if l.startswith('{"area":')];ok=p.returncode==0 and len(records)==1 and worker.is_file() and all(math.isfinite(records[0][k])for k in ['area','delay'])
  event=dict(index=index,argv=argv,requested_timeout_secs=remaining,node_allowance=None,charged_admission_upper=charge if a else None,charge_is_measured=a is not None,cumulative_accounted_upper=accounted if a else None,admission=a,admission_path=str(ledger),pool_path=str(pool),candidate_path=str(diag),trace_path=str(trace),stdout_path=str(prefix.with_suffix('.stdout')),stderr_path=str(prefix.with_suffix('.stderr')),exit_code=p.returncode,timeout=timeout,wall_secs=time.monotonic()-t,complete_returned_cover=ok)
  info['attempts'].append(event)
  if ok:
   z=records[0];assert z['cumulative_admitted_enodes']<=limit and z['instances']<=bank_limit;info['optimizer_succeeded']=True;info['optimizer_native']=z
   if better_da(z,metrics):chosen=worker;metrics=dict(area=z['area'],delay=z['delay'],cec=-1,status='checked-native-egraph-cover',proven_optimal=False,egraph_cover_selected=True);info['selected']='egraph_cover';info['selected_attempt']=index
   break
  # Retry only a proved incomplete import, not arbitrary timeout/recovery error.
  if not(a and not a['import_complete'] and a['rejected_before_insertion']):break
 info.update(cumulative_accounted_enode_upper=accounted if all(x['charge_is_measured'] for x in info['attempts']) else None,optimizer_address_space_limit_bytes=None,source_plus_optimizer_retained_upper=None,actual_source_instances=seed['instances'],parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,optimizer_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,sum_of_process_peak_rss_is_conservative=True)
 shutil.copyfile(chosen,out);info['selected_verilog_sha256']=sha(out);info['total_mapping_frontend_secs']=time.monotonic()-start;info['mapping_complete']=True;info['mapping_timed_out']=any(a['timeout'] for a in info['attempts'])
 if time.monotonic()>deadline:raise TimeoutError('complete frontend exceeded600s; no success reported')
 meta.write_text(json.dumps(info,indent=2));metrics.update(source_first=True,optimization_succeeded=info['optimizer_succeeded'],selected_mode=info['selected'],source_instances=seed['instances'],global_retained_candidate_limit=None,cumulative_accounted_enode_upper=accounted,total_mapping_frontend_secs=info['total_mapping_frontend_secs'])
 print(json.dumps(metrics,separators=(',',':')),flush=True)
if __name__=='__main__':main()
