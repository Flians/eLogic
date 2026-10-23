#!/usr/bin/env python3
"""Finite sequential qualification; no native baseline algorithm changes."""
import json,sys,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[3];B=Path(__file__).parent;O=R/'results_v2/binding_noop_recovery_20261019'
def run(name,args):
 print('start',name,flush=True)
 with (O/(name+'.log')).open('w') as f:r=subprocess.run([sys.executable,*map(str,args)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,(name,r.returncode);print('end',name,flush=True)
def main():
 full=Path('/tmp/emapper-binding-noop-full-20261019/campaign.json');c=json.loads(full.read_text());assert c['status']=='attempted' and len(c['jobs'])==120
 run('full_check',[B/'check_binding_exchange.py',full]);run('full_stage_check',[B/'audit_binding_stages.py',full]);v=json.loads((full.parent/'binding_validation.json').read_text())
 # Unexpected stage/CEC failures are evidence; stop before blindly repeating them.
 unexpected=[x for x in v['rows'] if not x['valid'] and x['native_failure'].get('stage')!='import_budget']
 if unexpected:
  (O/'unexpected_failures.json').write_text(json.dumps(unexpected,indent=2)+'\n')
  run('qualification',[B/'finalize_binding_recovery.py',full,'--output',O/'evidence'])
  print('unexpected failures preserved; no unchanged repeat',flush=True);return
 reps=[]
 for i in range(3):
  target=Path(f'/tmp/emapper-binding-noop-repeat-r{i}-20261019')
  args=[B/'run_binding_noop_recovery.py','--output',target,'--designs','sqrt','sin','adder','--max-jobs','18']+(['--reverse-order']if i%2 else [])
  if target.exists():
   cp=target/'campaign.json';assert cp.exists(),'partial without checkpoint requires independent audit'
  run(f'repeats_{i}',args);run(f'repeats_check_{i}',[B/'check_binding_exchange.py',target/'campaign.json']);run(f'repeats_stage_check_{i}',[B/'audit_binding_stages.py',target/'campaign.json']);reps.append(target/'campaign.json')
 run('qualification',[B/'finalize_binding_recovery.py',full,'--repeats',*reps,'--output',O/'evidence'])
 (O/'finite_finished.json').write_text(json.dumps(dict(full=120,repeats=54,all_finite_mapping_batches_ended=True),indent=2)+'\n');print('finite qualification finished',flush=True)
if __name__=='__main__':main()
