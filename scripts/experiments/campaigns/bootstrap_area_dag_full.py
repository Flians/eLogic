#!/usr/bin/env python3
"""Same-version pilot reuse and explicit unchanged-import-failure proof for full scope."""
import copy,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=ROOT/'results_v2/delay_area_dag_full_20261015';cp=out/'campaign.json';c=json.load(open(cp));assert not c['jobs'];proto=Path('/tmp/emapper-delay-area-dag-20261015');proof={}
for name in ['egraph_regions.rs','egraph_sat.rs','aig_graph.rs','techlib.rs']:
 a=ROOT/'emapper/src'/name;b=proto/'emapper/src'/name;assert a.read_bytes()==b.read_bytes();proof[str(a)]=sha(a)
def prefix(t):return t[t.index('pub fn map(aig:'):t.index('    // A fixed representation policy,',t.index('pub fn map(aig:'))]
a=(ROOT/'emapper/src/egraph_map.rs').read_text();b=(proto/'emapper/src/egraph_map.rs').read_text();assert prefix(a)==prefix(b);proof['unchanged_import_guard_prefix_sha256']=hashlib.sha256(prefix(a).encode()).hexdigest()
a=ROOT/'src/bin/bench_rs_egraph_map.rs';b=proto/'src/bin/bench_rs_egraph_map.rs';assert a.read_bytes()==b.read_bytes();proof[str(a)]=sha(a)
pilot=ROOT/'results_v2/delay_area_dag_pilot_20261015/campaign.json';p=json.load(open(pilot));assert p['status']=='attempted' and len(p['jobs'])==16
# Only the unexecuted batch orchestrator differs; driver/native/actual dependencies identical.
allowed={str(ROOT/'scripts/experiments/campaigns/run_delay_area_dag.py'),str(ROOT/'scripts/experiments/campaigns/run_delay_area_dag_full.py')};assert {k:v for k,v in p['fingerprint'].items() if k not in allowed}=={k:v for k,v in c['fingerprint'].items() if k not in allowed}
for j in p['jobs']:
 for key in ['report','audit','pool','candidates','trace']:
  if key in j:assert sha(Path(j[key]))==j[key+'_sha256']
 j=copy.deepcopy(j);j.update(adopted_from=str(pilot),adoption_reason='same native/harness/settings/inputs/libraries; only batch orchestrator differs; not a new repeat');c['jobs'].append(j)
oldpath=ROOT/'results_v2/compound_cap_epfl_full_20261014/campaign.json';old=json.load(open(oldpath));adopt=[]
for k in [3,4]:
 for bench in ['div','hyp','log2','multiplier','mem_ctrl']:
  j=next(j for j in old['jobs'] if j['k']==k and j['bench']==bench and j['profile']=='control');rp=Path(j['report']);assert sha(rp)==j['report_sha256'];r=json.load(open(rp));row=r['results'][0];assert row['status']=='failed' and not row.get('measured') and not row.get('native');log=rp.parent/f'{bench}.{row["method"]}/map.log';m=re.fullmatch(r'Error: "budget: imported graph (\d+) exceeds node allowance 20000"\s*',log.read_text());assert m and int(m[1])>20000
  assert sha(ROOT/f'results_v2/epfl_resyn2_subjects_20261012/{bench}.resyn2.aig')==row['preprocessing']['sha256'];assert sha(ROOT/f'tools/mockturtle/experiments/benchmarks/{bench}.aig')==row['input_sha256']
  expected={'EGM_K':str(k),'EGM_NODES':'20000','EGM_ITER':'16','EGM_LITERAL_IMPORT':'0','EGM_LITERAL_AUTO':'1','EGM_REGION_CLASSES':'256','EGM_STRICT_BUDGETS':'1','EGM_CANONICAL_PHASE':'1','EGM_CANONICAL_WIRES':'1','EGM_BOOLEAN':'0','EGM_LIBRARY_TIMING':'0'};assert all(row['settings'][x]==v for x,v in expected.items());assert sha(rp.parent/r['library'])==r['library_sha256']
  for profile in c['fixed_profiles']:
   name=f'{k}in_{bench}_{profile}';folder=out/name;folder.mkdir();new=copy.deepcopy(r);new['library']=str(rp.parent/r['library']);new['adoption']={'prior_report':str(rp),'prior_report_sha256':sha(rp),'prior_log':str(log),'prior_log_sha256':sha(log),'prior_imported_nodes':int(m[1]),'new_mapper_run':False,'new_runtime_measurement':False,'reason':'unchanged import prefix/modules/input/library/parameters; changed recovery never executes','requested_unexecuted_recovery_flags':{'EGM_FAST_DELAY_SEED':int(profile!='legacy_control'),'EGM_REQUIRED_AREA_DAG':int(profile=='required_area')}};np=folder/'report.json';np.write_text(json.dumps(new,indent=2));c['jobs'].append({'name':name,'k':k,'bench':bench,'profile':profile,'report':str(np),'report_sha256':sha(np),'exit_code':1,'adopted_from':str(oldpath),'original_argv':j['argv'],'new_mapper_run':False});adopt.append(new['adoption'])
c['status']='checkpointed';cp.write_text(json.dumps(c,indent=2));(out/'adoption.json').write_text(json.dumps({'proof':proof,'pilot_manifest_sha256':sha(pilot),'old_manifest_sha256':sha(oldpath),'pilot_jobs_reused':16,'historical_failure_slots_reused':30,'allowed_unexecuted_orchestrator_difference':sorted(allowed),'failures':adopt},indent=2));print('46 explicit reused slots, 74 new jobs remain')
