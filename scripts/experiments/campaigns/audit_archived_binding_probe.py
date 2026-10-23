#!/usr/bin/env python3
"""Audit a negative checkpoint against its exact frozen source/binary snapshot."""
import argparse,hashlib,json,sys
from pathlib import Path
from audit_live_area_commands import audit, R
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=json.loads(a.campaign.read_text());resolved={};evidence=[]
 original=Path.read_bytes
 for path,h in c['fingerprint'].items():
  actual=Path(path)
  if hashlib.sha256(original(actual)).hexdigest()==h:continue
  archived=a.snapshot/actual.relative_to(R)
  assert hashlib.sha256(original(archived)).hexdigest()==h,('snapshot drift',path)
  resolved[actual]=archived
 # Only hash/read those declared source/binary dependencies from the exact
 # snapshot; no result, input, library or command is modified or substituted.
 def frozen_read(p):return original(resolved.get(p,p))
 Path.read_bytes=frozen_read
 try:
  for j in c['jobs']:
   for key in ['report','audit','trace','admission','candidates','pool']:
    if key in j:assert sha(j[key])==j[key+'_sha256']
   proof=audit(j,c);r=json.loads(Path(j['report']).read_text())['results'][0]
   t=json.loads(Path(j['trace']).read_text()) if 'trace' in j else {}
   b=json.loads(Path(j['admission']).read_text()) if 'admission' in j else {}
   assert b.get('node_allowance')==20000 and b['cumulative_admitted_enodes']<=20000 and not b['refund_on_rebuild']
   evidence.append(dict(name=j['name'],status=r['status'],error=r.get('error'),measured=r.get('measured'),commands=proof,trace=t,admission=b))
 finally:Path.read_bytes=original
 a.output.write_text(json.dumps(dict(attempts=len(evidence),CEC=sum(x['status']=='ok' for x in evidence),full_epfl=False,qualified=False,source_resolution={str(p):str(q) for p,q in resolved.items()},scope='Frozen actual invocation source/binary, not current-main hash; no result adoption or replacement',evidence=evidence),indent=2)+'\n')
if __name__=='__main__':main()
