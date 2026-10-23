#!/usr/bin/env python3
"""Verify lifetime (not resident) admissions on every completed campaign case."""
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);a=p.parse_args()
 cmd=[sys.executable,str(Path(__file__).with_name('check_functional_cuts.py')),str(a.campaign)]
 r=subprocess.run(cmd,capture_output=True,text=True)
 (a.campaign.parent/'functional_check.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr
 c=json.loads(a.campaign.read_text());records=[]
 for j in c['jobs']:
  assert 'admission' in j,('missing lifetime certificate',j['name'])
  assert sha(j['admission'])==j['admission_sha256']
  b=json.loads(Path(j['admission']).read_text());r=json.loads(Path(j['report']).read_text())['results'][0]
  assert b['node_allowance']==20000 and not b['refund_on_rebuild']
  if r['status']=='ok':
   n=r['native'];assert b['import_complete'] and b['strict'] and b['all_strict_rewrites_preflighted']
   assert n['imported_enodes']==b['resident_imported_enodes']<=n['cumulative_imported_enodes']==b['cumulative_imported_enodes']
   assert n['enodes']==b['resident_enodes']<=n['cumulative_admitted_enodes']==b['cumulative_admitted_enodes']<=20000
   assert n['cumulative_imported_enodes']<=n['cumulative_admitted_enodes']
  else:
   assert not b['import_complete'] and b['rejected_before_insertion']
   assert b['resident_enodes']<=b['cumulative_admitted_enodes']==20000
  records.append(dict(name=j['name'],status=r['status'],certificate=b))
 v=json.loads((a.campaign.parent/'functional_validation.json').read_text());v['lifetime_admission_audits']=records;v['lifetime_budget_pass']=True
 v['scope']='One lifetime ledger covers every strict mapper import and RHS insertion. Cross-pool old-cover protection is still absent. Baseline algorithms unchanged; no automatic PPA improvement claim.'
 (a.campaign.parent/'cumulative_validation.json').write_text(json.dumps(v,indent=2)+'\n')
 print(json.dumps({k:v[k] for k in ['complete','attempts','cec','summary','lifetime_budget_pass']},indent=2))
if __name__=='__main__':main()
