#!/usr/bin/env python3
"""Adopt only identical V27 pilot observations into the full V27 pair study."""
import argparse,copy,hashlib,json
from pathlib import Path
from run_epfl_baselines import EPFL
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--donor',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();d=json.loads(a.donor.read_text());assert d['status']=='attempted' and len(d['jobs'])==d['expected_jobs'];assert not a.output.exists() and not d['components']
 for p,h in d['fingerprint'].items():assert sha(p)==h,p
 for j in d['jobs']:
  for k in ['report','audit','pool','candidates','trace','pool_frontier','pool_guard','helper_allocation']:
   if k in j:assert sha(j[k])==j[k+'_sha256']
 c=copy.deepcopy(d);c.update(expected_jobs=80,designs=EPFL,status='checkpointed',reverse_order=False,adoption={'donor':str(a.donor.resolve()),'sha256':sha(a.donor),'count':len(c['jobs']),'same_fingerprint':True,'no_duplicate_observation_count':True,'basis':'Exact same native/source/driver/input/library/evaluator dependencies and effective native settings. Same own-mapper profiles; only suite bookkeeping expanded. V28 binary differs and cannot adopt these mapped observations.'})
 for j in c['jobs']:j.update(adopted_from=str(a.donor.resolve()),adopted_campaign_sha256=sha(a.donor),adoption_basis=c['adoption']['basis'])
 a.output.mkdir();(a.output/'campaign.json').write_text(json.dumps(c,indent=2)+'\n');print('adopted',len(c['jobs']),'/',c['expected_jobs'])
if __name__=='__main__':main()
