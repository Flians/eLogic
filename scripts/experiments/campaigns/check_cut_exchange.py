#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
from audit_live_area_commands import audit
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);a=p.parse_args();c=json.loads(a.campaign.read_text());rows=[];proofs=[]
 for path,h in c['fingerprint'].items():assert sha(path)==h,path
 for j in c['jobs']:
  for key in ['report','audit','pool','candidates','trace','pool_frontier','pool_guard','helper_allocation']:
   if key in j:assert sha(j[key])==j[key+'_sha256']
  proofs.append(audit(j,c));r=json.loads(Path(j['report']).read_text())['results'][0];valid=r['status']=='ok';s=r['settings'];assert s['EGM_CUT_PIN_FLOW']=='1';assert s['EGM_CUT_EXCHANGE_GUARD']==str(int(j['profile']=='protected_area'));assert s['EGM_DELTA_AREA_FILTER']=='0';assert s['EGM_OBJECTIVE']=='delay-area'
  if valid:
   n=r['native'];assert n['imported_enodes']<=20000;assert r['mapping_wall_secs']<=300;trace=json.loads(Path(j['trace']).read_text());totals=[e['seconds'] for e in trace['events'] if e['stage']=='global_total'];assert len(totals)==1 and totals[0]<=2;assert r['measured']['cec']==1;assert n['enodes']<=20000 and n['max_cuts_per_class']<=16 and n['instances']<=100000 and n['heuristic_secs']<=2
   assert abs(n['area']-r['measured']['area'])<1e-6 and abs(n['delay']-r['measured']['delay'])<1e-6
  rows.append(dict(name=j['name'],k=j['k'],bench=j['bench'],profile=j['profile'],status=r['status'],valid=valid,area=r.get('measured',{}).get('area'),delay=r.get('measured',{}).get('delay'),error=r.get('error'),native=r.get('native'),report=j['report']))
 pairs=[]
 for r in rows:
  if r['profile']!='protected_area':continue
  b=next((b for b in rows if (b['k'],b['bench'],b['profile'])==(r['k'],r['bench'],'legacy_control')),None)
  if b is None:continue
  pair=dict(k=r['k'],bench=r['bench'],control=b,candidate=r)
  if b['valid'] and r['valid']:
   dt=r['delay']-b['delay'];at=r['area']-b['area'];tol=1e-6*max(1,abs(r['delay']),abs(b['delay']));pair.update(delay_delta=dt,area_delta=at,delay_outcome='better' if dt < -tol else 'worse' if dt>tol else 'equal')
  pairs.append(pair)
 out=a.campaign.parent/'validation.json';out.write_text(json.dumps(dict(attempts=len(rows),expected=c['expected_jobs'],complete=len(rows)==c['expected_jobs'],cec=sum(r['valid'] for r in rows),rows=rows,pairs=pairs,independent_commands=proofs,source_and_budget_audits_pass=True,baseline_algorithms_changed=False,main_promotion=False),indent=2)+'\n')
 for p in pairs:print(p['k'],p['bench'],p.get('delay_outcome','failed'),p.get('delay_delta'),p.get('area_delta'),flush=True)
 print('CEC',sum(r['valid'] for r in rows),'/',len(rows))
if __name__=='__main__':main()
