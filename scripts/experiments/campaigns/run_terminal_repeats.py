#!/usr/bin/env python3
"""Three predeclared alternating-order int2float repetitions, no best-run choice."""
from pathlib import Path
import subprocess,sys,json
R=Path(__file__).resolve().parents[3];B=Path(__file__).with_name('run_terminal_area.py')
for rep in range(3):
 n=f'terminal_area_int2float_r{rep}_20261018';t=Path('/tmp/emapper-artifacts-20261018')/n;t.mkdir(exist_ok=False);p=R/'results_v2'/n;p.symlink_to(t,target_is_directory=True)
 argv=[sys.executable,str(B),'--output',str(p),'--designs','int2float','--max-jobs','4']+(['--reverse-order'] if rep%2 else [])
 (t/'scheduled_command.json').write_text(json.dumps(argv,indent=2)+'\n')
 with (t/'runner.log').open('w') as f:r=subprocess.run(argv,cwd=R,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,(n,r.returncode)
 c=json.loads((t/'campaign.json').read_text());assert len(c['jobs'])==4 and c['status']=='attempted';print(n,'complete4',flush=True)
