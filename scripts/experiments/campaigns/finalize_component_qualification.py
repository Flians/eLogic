#!/usr/bin/env python3
"""Run independent post-campaign checks, never mapping, on fixed scopes."""
import subprocess,sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];B=Path(__file__).parent
names=['single_output_components_guard_full_20261017','single_output_components_full_20261017','single_output_no_saturation_full_20261017']
for order in ['reverse','forward']:
 names.extend([f'single_output_components_bar_sin_{order}_20261017',f'single_output_components_sqrt_{order}_20261017',f'single_output_guard_repeats_{order}_20261017'])
paths=[R/'results_v2'/n/'campaign.json' for n in names]
for p in paths:
 c=json.loads(p.read_text());assert c['status']=='attempted' and len(c['jobs'])==c['expected_jobs'],p
out=R/'results_v2/single_output_components_evidence_20261017'
subprocess.run([sys.executable,str(B/'summarize_single_output_components.py'),'--output',str(out),'--studies',*map(str,paths)],check=True)
with (out/'integrity.log').open('w') as f:subprocess.run([sys.executable,str(B/'audit_single_output_components.py'),'--studies',*map(str,paths),'--output',str(out/'integrity.json')],stdout=f,stderr=subprocess.STDOUT,check=True)
print('complete independent checks',out,flush=True)
