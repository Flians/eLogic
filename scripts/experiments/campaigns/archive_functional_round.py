#!/usr/bin/env python3
"""Archive every successful/failed raw observation with tar-member SHA verification."""
from pathlib import Path
import subprocess,sys,json
R=Path(__file__).resolve().parents[3];B=Path(__file__).parent;O=R/'results_v2/functional_cuts_evidence_20261019'
names=['functional_cuts_pilot_20261019','functional_cuts_full_20261019']+[f'functional_cuts_repeat_r{i}_20261019' for i in range(3)]+['source_closure_pilot_20261019']
for n in names:
 with (O/(n+'.archive.log')).open('w') as f:subprocess.run([sys.executable,str(B/'archive_component_artifacts.py'),str(R/'results_v2'/n),'--label',n],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(n,'member SHA verified',flush=True)
data=[json.loads((R/'results_v2/archives'/f'{n}.verified.json').read_text()) for n in names]
assert all(d['stream_member_hash_verification'] and '/tmp/emapper-artifacts-20261019' in d['restore'] for d in data)
(O/'archives.json').write_text(json.dumps(data,indent=2)+'\n')
print('durable archives',len(data),'compressed bytes',sum(d['compressed_bytes'] for d in data),flush=True)
