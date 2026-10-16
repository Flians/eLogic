#!/usr/bin/env python3
"""Run paired cold/warm campaigns without changing the original synthesis inputs."""
import argparse, datetime, hashlib, json, subprocess, sys
from pathlib import Path
from run_three_parts import ARMS, DESIGNS, GOAL
PROJECT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--original-root',type=Path,default=PROJECT/'results_v20_three_parts_chain');ap.add_argument('--jobs',type=int,default=8);a=ap.parse_args()
 assert a.jobs>0
 p=a.root.resolve();p.mkdir(parents=True,exist_ok=True)
 plan=dict(recorded_at=datetime.datetime.now().astimezone().isoformat(),method='live inverse-query acceleration, decoded/cached cut truth tables, two support-ranked pivots, complete-coverage rechecks and shared mapped covers',cohorts=['cold','warm'],circuits=len(DESIGNS),parts=['mfs3','main','ablation'],main_caps=[1,6],mfs3_cap=1,ablation_cap=6,ablation_arms=list(ARMS),ablation_flags=ARMS,method_goal_flags=GOAL,post_timeout_seconds=1000,abc_timeout_seconds=1000,outer_process_guard_seconds=3600,workers_per_cohort=a.jobs,engine_workers=1,maximum_native_workers=2*a.jobs,independent_abc_workers=a.jobs,cold_policy='empty private e-graph pool per stage/arm/design',warm_policy='reuse only paired cold pool after its result/provenance; all native caches restart',baseline_policy='fresh 1000-second ABC batch shared by byte-identical warm cohort',baseline_scope='same <=6-input sublibrary and initial netlist for mfs3; original full library for main and ablation',original_archive=str(a.original_root.resolve()),expected_engine_runs=2*(64+len(DESIGNS)*len(ARMS)),expected_method_runs=2*(48+len(DESIGNS)*len(ARMS)),expected_fresh_abc_runs=16,timeout_selection='checkpoint',checkpoint_policy='independent CEC and serialized PPA guard of best saved output; TO remains incomplete, not completed',native_cache_entries=262144,capacity_ablation_entries=65536,engine_sha256=sha(PROJECT/'build/epmapping/epmapping_check'))
 file=p/'execution_plan.json'
 if file.exists():
  old=json.loads(file.read_text());assert all(old[k]==v for k,v in plan.items() if k!='recorded_at'),'Plan changed: use a fresh root'
 else:file.write_text(json.dumps(plan,indent=2)+'\n')
 procs=[]
 for mode in ['cold','warm']:
  cmd=[sys.executable,str(PROJECT/'scripts/experiments/run_three_parts.py'),'--root',str(p/mode),'--original-root',str(a.original_root.resolve()),'--jobs',str(a.jobs),'--timeout','3600','--abc-timeout','1000','--timeout-selection','checkpoint','--profile-cache-mode',mode,'--cache-root',str(p/'cache')]
  if mode=='warm':cmd+=['--abc-source',str(p/'cold')]
  log=(p/f'{mode}_execution.log').open('a');procs.append((mode,subprocess.Popen(cmd,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT),log))
 results={}
 for mode,q,log in procs:results[mode]=q.wait();log.close();print(mode,results[mode],flush=True)
 (p/'execution_status.json').write_text(json.dumps(results,indent=2)+'\n')
 assert all(v==0 for v in results.values()),results
if __name__=='__main__':main()
