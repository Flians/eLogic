#!/usr/bin/env python3
"""Fixed finite serial repetition plan; no best-run or per-design selection."""
from pathlib import Path
import subprocess,sys,json
R=Path(__file__).resolve().parents[3];B=R/'scripts/experiments/campaigns';T=Path('/tmp/emapper-artifacts-20261017');plan=[]
for order in ['reverse','forward']:
 rev=['--reverse-order'] if order=='reverse' else []
 plan.append((f'single_output_components_bar_sin_{order}_20261017','run_single_output_component_module.py',['--components','--designs','bar','sin',*rev],32))
 plan.append((f'single_output_components_sqrt_{order}_20261017','run_single_output_component_repeat.py',['--components','--designs','sqrt','--profiles','protected_area','single_structure','no_sdc','no_actual_area','no_phase_recovery','no_timing_area_module',*rev],12))
 plan.append((f'single_output_guard_repeats_{order}_20261017','run_single_output_components.py',['--designs','sqrt','int2float','priority','arbiter',*rev],16))
for name,controller,args,count in plan:
 target=T/name;logical=R/'results_v2'/name
 assert not logical.exists() and not target.exists(),name
 target.mkdir();logical.symlink_to(target,target_is_directory=True)
 argv=[sys.executable,str(B/controller),'--output',str(logical),'--max-jobs',str(count),*args]
 (target/'scheduled_command.json').write_text(json.dumps(argv,indent=2)+'\n')
 print('START',name,flush=True)
 with (target/'runner.log').open('w') as f:p=subprocess.run(argv,cwd=R,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,(name,p.returncode)
 c=json.loads((target/'campaign.json').read_text());assert c['status']=='attempted' and len(c['jobs'])==count
 print('COMPLETE',name,count,flush=True)
