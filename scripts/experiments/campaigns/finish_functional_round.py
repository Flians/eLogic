#!/usr/bin/env python3
"""Finite sequential qualification after the already-running full matrix ends."""
from pathlib import Path
import subprocess,sys,json
R=Path(__file__).resolve().parents[3];B=Path(__file__).parent
O=R/'results_v2/functional_cuts_evidence_20261019'
def run(name,args):
 print('start',name,flush=True)
 with (O/(name+'.log')).open('w') as f:p=subprocess.run([sys.executable,*map(str,args)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,(name,p.returncode)
 print('end',name,flush=True)
def main():
 O.mkdir(exist_ok=False)
 run('full_check',[B/'check_functional_cuts.py',R/'results_v2/functional_cuts_full_20261019/campaign.json'])
 run('repetitions',[B/'run_functional_repeats.py'])
 for rep in range(3):run(f'repeat_check_{rep}',[B/'check_functional_cuts.py',R/f'results_v2/functional_cuts_repeat_r{rep}_20261019/campaign.json'])
 t=Path('/tmp/emapper-artifacts-20261019/source_closure_pilot_20261019');t.mkdir(exist_ok=False);p=R/'results_v2/source_closure_pilot_20261019';p.symlink_to(t)
 run('source_pilot',[B/'run_source_closure.py','--output',p,'--max-jobs','16'])
 run('source_check',[B/'check_source_closure.py','--campaign',p/'campaign.json','--output',O/'source_validation.json'])
 run('selected_cover_analysis',[B/'analyze_functional_cuts.py',R/'results_v2/functional_cuts_full_20261019/campaign.json','--output',O/'selected_cover_analysis.json'])
 (O/'finite_batches_finished.json').write_text(json.dumps({'full':80,'targeted_repetitions':36,'source':16,'all_finite_batches_ended':True,'main_promoted':False},indent=2)+'\n')
 print('all finite batches ended',flush=True)
if __name__=='__main__':main()
