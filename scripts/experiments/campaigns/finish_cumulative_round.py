#!/usr/bin/env python3
"""Finite sequential checks/repeats after the full campaign has ended."""
import json,sys,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[3];B=Path(__file__).parent;O=R/'results_v2/cumulative_admission_20261019'
def run(name,args):
 print('start',name,flush=True)
 with (O/(name+'.log')).open('w')as f:r=subprocess.run([sys.executable,*map(str,args)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,(name,r.returncode);print('end',name,flush=True)
def main():
 full=R/'results_v2/cumulative_admission_full_20261019/campaign.json';c=json.loads(full.read_text());assert c['status']=='attempted' and len(c['jobs'])==80
 run('full_check',[B/'check_cumulative_admission.py',full]);reps=[]
 for i in range(3):
  name=f'cumulative_admission_repeat_r{i}_20261019';target=Path('/tmp/emapper-cumulative-admission-20261019')/name;target.mkdir(exist_ok=False);p=R/'results_v2'/name;p.symlink_to(target,target_is_directory=True)
  args=[B/'run_cumulative_admission.py','--output',p,'--designs','sqrt','sin','adder','--max-jobs','12']+(['--reverse-order']if i%2 else [])
  (target/'scheduled_command.json').write_text(json.dumps([sys.executable,*map(str,args)],indent=2)+'\n')
  run(f'repeats_{i}',args);run(f'repeats_check_{i}',[B/'check_cumulative_admission.py',p/'campaign.json']);reps.append(p/'campaign.json')
 run('qualification',[B/'finalize_cumulative_admission.py',full,'--repeats',*reps,'--output',O/'evidence'])
 run('incumbent_risk',[B/'audit_functional_incumbent.py',full,'--output',O/'incumbent_risk.json'])
 (O/'finite_finished.json').write_text(json.dumps(dict(full=80,repeats=36,all_finite_mapping_batches_ended=True),indent=2)+'\n')
 print('all finite checks/repeats finished',flush=True)
if __name__=='__main__':main()
