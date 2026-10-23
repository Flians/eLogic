#!/usr/bin/env python3
"""Explicit same-binary, same-settings adoption of a complete fixed pilot into full scope."""
import argparse,hashlib,json
from pathlib import Path
from run_epfl_baselines import EPFL
from audit_live_area_commands import audit
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=json.loads(a.source.read_text());assert len(c['jobs'])==c['expected_jobs'];assert not a.output.exists()
for f,h in c['fingerprint'].items():assert sha(f)==h
proof=[]
for j in c['jobs']:
 for key in ['report','audit','pool','candidates','trace','pool_frontier','pool_guard','helper_allocation']:
  if key in j:assert sha(j[key])==j[key+'_sha256']
 proof.append(audit(j,c))
a.output.mkdir();adoption={'source_campaign':str(a.source.resolve()),'source_sha256':sha(a.source),'same_source_binary_input_library_parameters':True,'all_jobs_including_failures_adopted':True,'jobs':[j['name'] for j in c['jobs']],'independent_commands':proof,'adopter_sha256':sha(__file__),'limitation':'Reuse of identical observations, not additional runs or repeated validation.'}
(a.output/'adoption.json').write_text(json.dumps(adoption,indent=2)+'\n');c.update(designs=EPFL,expected_jobs=80,status='checkpointed',adoption=str((a.output/'adoption.json').resolve()));(a.output/'campaign.json').write_text(json.dumps(c,indent=2)+'\n')
