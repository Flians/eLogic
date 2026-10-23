#!/usr/bin/env python3
"""Explicitly adopt all subset observations including failures, exact dependencies only."""
import argparse,json,hashlib,copy
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--donor',required=True,type=Path);ap.add_argument('--target',required=True,type=Path);a=ap.parse_args();d=json.loads(a.donor.read_text());c=json.loads(a.target.read_text());assert d['status']=='attempted' and len(d['jobs'])==d['expected_jobs'] and not c['jobs'] and set(d['designs'])<=set(c['designs'])
 for k in ['fingerprint','factors','fixed_profiles','budgets','scope','objective','component_overrides','components']:assert d[k]==c[k],k
 for p,h in d['fingerprint'].items():assert sha(p)==h,p
 for j in d['jobs']:
  for k,h in j.items():
   if k.endswith('_sha256') and k[:-7] in j:assert sha(j[k[:-7]])==h,(j['name'],k)
  q=copy.deepcopy(j);q.update(adopted_from=str(a.donor.resolve()),adopted_campaign_sha256=sha(a.donor),adoption_basis='All subset records including failures, identical full actual dependency fingerprint/profile/budgets. No per-design selection or cross-binary adoption. Actual original argv/report/audit retained; count once.');c['jobs'].append(q)
 c['adoption']={'donor':str(a.donor.resolve()),'sha256':sha(a.donor),'count':len(d['jobs']),'all_subset_records_including_failures':True,'same_dependency_fingerprint':True}
 a.target.write_text(json.dumps(c,indent=2)+'\n');print('explicit identical-dependency subset adoption',len(c['jobs']))
if __name__=='__main__':main()
