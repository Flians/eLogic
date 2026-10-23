#!/usr/bin/env python3
"""Reuse only identical full-method native observations; retain actual old driver/hash."""
import argparse,copy,hashlib,json
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--donor',type=Path,required=True);ap.add_argument('--target',type=Path,required=True);a=ap.parse_args();d=json.loads(a.donor.read_text());c=json.loads(a.target.read_text());assert d['status']=='attempted' and len(d['jobs'])==320 and not c['jobs'] and c['expected_jobs']==80
 old=Path(__file__).with_name('bench_single_output_components.py').resolve();new=old.with_name('bench_single_output_saturation.py');controller=new.with_name('run_single_output_saturation.py')
 normalized=new.read_text().replace("    ap.add_argument('--sat-iterations',type=int,choices=[0,16],default=16)\n",'').replace('EGM_ITER=str(a.sat_iterations)','EGM_ITER="16"');assert normalized==old.read_text(),'Driver change must be only optional parser plus its effective iteration setting'
 assert {p:h for p,h in c['fingerprint'].items() if p not in [str(new),str(controller)]}==d['fingerprint'],'All actual old dependencies retained unchanged in new fingerprint'
 for fp in [c['fingerprint'],d['fingerprint']]:
  for p,h in fp.items():assert sha(p)==h,p
 for j in d['jobs']:
  if j['profile']!='protected_area':continue
  for key in ['report','audit','pool','candidates','trace','pool_frontier','pool_guard','helper_allocation']:
   if key in j:assert sha(j[key])==j[key+'_sha256']
  r=json.loads(Path(j['report']).read_text())['results'][0];assert r['settings']['EGM_ITER']=='16' and r['settings']['EGM_TAIL_SHARED_AREA']=='0'
  q=copy.deepcopy(j);q.update(adopted_from=str(a.donor.resolve()),adopted_campaign_sha256=sha(a.donor),effective_SAT_iterations=16,adoption_basis='Identical V28 native binary/source/input/library/evaluator and every effective native setting; new driver adds only optional SAT iteration parser with default16. Old actual argv/driver/commands retained, old dependencies remain fingerprinted. All40 references including failures adopted, no per-design winner, no extra physical attempt count.')
  c['jobs'].append(q)
 assert len(c['jobs'])==40
 c['adoption']={'donor':str(a.donor.resolve()),'sha256':sha(a.donor),'count':40,'driver_text_equivalence_modulo_unused_default16_parser':True,'all_original_actual_dependencies_preserved':True,'new_only_dependencies':{str(p):sha(p) for p in [new,controller]},'no_duplicate_observation_count':True}
 a.target.write_text(json.dumps(c,indent=2)+'\n');print('explicit same-native reference adoption',len(c['jobs']),'/',c['expected_jobs'])
if __name__=='__main__':main()
