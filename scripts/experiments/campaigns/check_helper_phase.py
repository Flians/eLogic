#!/usr/bin/env python3
"""Read-only actual-budget and original-helper-subset audit of fixed helper/phase studies."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
c=json.loads(a.campaign.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();errors=[];cases=[]
for path,digest in c['fingerprint'].items():
 if sha(Path(path))!=digest:errors.append(['source',path])
for j in c['jobs']:
 for key in ['report','audit','pool','candidates','trace','pool_frontier','pool_guard','helper_allocation','diagnostic_profile']:
  if key in j and sha(Path(j[key]))!=j[key+'_sha256']:errors.append(['artifact',j['name'],key])
 report=json.loads(Path(j['report']).read_text());r=report['results'][0];n=r.get('native',{});helper_path=Path(j['helper_allocation']) if 'helper_allocation' in j else Path(j['report']).parent/(r['bench']+'.'+r['method'])/'pool-diagnostics.helper-allocation.json';h=json.loads(helper_path.read_text());assert set(h['old_helper_sources'])<=set(h['actual_helper_sources'])
 assert h['global_node_allowance']==20000 and h['initial_node_ceiling']==18000
 tr=json.loads(Path(j['trace']).read_text());events=tr['events'];gt=[e['seconds'] for e in events if e['stage']=='global_total'];assert len(gt)==1
 assert r['status']=='ok' and r['measured']['cec']==1 and j['exit_code']==0
 assert n['enodes']<=20000 and n['imported_enodes']<=20000 and n['max_cuts_per_class']<=16 and n['instances']<=100000 and n['heuristic_secs']<=2 and gt[0]<=2 and r['mapping_wall_secs']<=300
 assert abs(n['area']-r['measured']['area'])<1e-6 and abs(n['delay']-r['measured']['delay'])<1e-6
 assert r['preprocessing']['source_report_sha256']==c['fingerprint'][str(Path('results_v2/epfl_resyn2_subjects_20261012/report.json').resolve())]
 cases.append(dict(name=j['name'],area=r['measured']['area'],delay=r['measured']['delay'],actual_import=n['imported_enodes'],actual_postsat=n['enodes'],actual_helpers=len(h['actual_helper_sources']),old_helpers=len(h['old_helper_sources']),retained=n['instances'],generated=n['generated_instances'],recovery=gt[0],phase_trials=n['phase_trials'],phase_accepted=n['phase_accepted'],mapping=r['mapping_wall_secs'],helper_metadata_sha256=sha(helper_path),helper_metadata_first_captured_by_audit='helper_allocation' not in j))
assert not errors,errors
result=dict(cases=cases,actual_budget_and_hash_errors=errors,main_promoted=False,baseline_algorithms_changed=False,scope='Fixed target pilot only; not full EPFL/repeated qualification. Actual helpers explicitly larger than oldmandatorysubset within original global capacity. Retainedpool not transientRAMcap.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
