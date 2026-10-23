#!/usr/bin/env python3
"""Predeclared sqrt/bar/sin both libraries, three alternating-order pairs, no best-run selection."""
from pathlib import Path
import subprocess,sys,json
R=Path(__file__).resolve().parents[3];B=Path(__file__).with_name('run_functional_cuts.py')
for rep in range(3):
 n=f'functional_cuts_repeat_r{rep}_20261019';t=Path('/tmp/emapper-artifacts-20261019')/n;t.mkdir(exist_ok=False);p=R/'results_v2'/n;p.symlink_to(t,target_is_directory=True)
 argv=[sys.executable,str(B),'--output',str(p),'--designs','sqrt','bar','sin','--max-jobs','12']+(['--reverse-order'] if rep%2 else [])
 (t/'scheduled_command.json').write_text(json.dumps(argv,indent=2)+'\n')
 with (t/'runner.log').open('w') as f:r=subprocess.run(argv,cwd=R,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,(n,r.returncode)
 c=json.loads((t/'campaign.json').read_text());assert len(c['jobs'])==12 and c['status']=='attempted';print(n,'complete12',flush=True)
