#!/usr/bin/env python3
"""Explicitly reuse same-version pilot jobs and unchanged pre-recovery import failures."""
import copy,hashlib,json,re
from pathlib import Path
from run_epfl_baselines import EPFL
ROOT=Path(__file__).resolve().parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=ROOT/'results_v2/delay_frontier_full_20261015';out.mkdir(exist_ok=False)
pilot=ROOT/'results_v2/delay_frontier_sqrt_pilot_20261015/campaign.json';c=json.loads(pilot.read_text());assert c['status']=='attempted' and len(c['jobs'])==4
proto=Path('/tmp/emapper-delay-frontier-20261015');proof={}
for name in ['egraph_regions.rs','egraph_sat.rs','aig_graph.rs','techlib.rs']:
 a=ROOT/'emapper/src'/name;b=proto/'emapper/src'/name;assert a.read_bytes()==b.read_bytes();proof[str(a)]=sha(a)
a=(ROOT/'emapper/src/egraph_map.rs').read_text();b=(proto/'emapper/src/egraph_map.rs').read_text()
def prefix(t):return t[t.index('pub fn map(aig:'):t.index('    // A fixed representation policy,',t.index('pub fn map(aig:'))]
assert prefix(a)==prefix(b);proof['import_startup_validation_and_node_guard_sha256']=hashlib.sha256(prefix(a).encode()).hexdigest()
a=ROOT/'src/bin/bench_rs_egraph_map.rs';b=proto/'src/bin/bench_rs_egraph_map.rs';assert a.read_bytes()==b.read_bytes();proof[str(a)]=sha(a)
oldpath=ROOT/'results_v2/compound_cap_epfl_full_20261014/campaign.json';old=json.loads(oldpath.read_text());c['expected_jobs']=80;c['designs']=list(EPFL);c['status']='checkpointed'
proof.update(pilot_manifest_sha256=sha(pilot),old_manifest_sha256=sha(oldpath),new_flags_read_only_in_recovery=True,reused_failure_has_no_new_runtime_measurement=True)
for j in c['jobs']:
 for key in ['report','audit','pool','candidates','trace']:
  if key in j:assert sha(Path(j[key]))==j[key+'_sha256']
 j['adopted_from']=str(pilot);j['adoption_reason']='identical frozen mapper/harness/flags/input/library; expand explicit scope; not a new repeat'
rows=[]
for k in [3,4]:
 for bench in ['div','hyp','log2','multiplier','mem_ctrl']:
  oldjob=next(j for j in old['jobs'] if j['k']==k and j['bench']==bench and j['profile']=='control');rp=Path(oldjob['report']);assert sha(rp)==oldjob['report_sha256'];r=json.loads(rp.read_text());row=r['results'][0];assert row['status']=='failed' and not row.get('measured') and not row.get('native')
  log=rp.parent/f'{bench}.{row["method"]}/map.log';text=log.read_text();match=re.fullmatch(r'Error: "budget: imported graph (\d+) exceeds node allowance 20000"\s*',text);assert match and int(match[1])>20000
  subject=ROOT/f'results_v2/epfl_resyn2_subjects_20261012/{bench}.resyn2.aig';golden=ROOT/f'tools/mockturtle/experiments/benchmarks/{bench}.aig';assert sha(subject)==row['preprocessing']['sha256'] and sha(golden)==row['input_sha256']
  # Compare every input/import-relevant EGM setting with the new fixed profile.
  expected={'EGM_K':str(k),'EGM_NODES':'20000','EGM_ITER':'16','EGM_LITERAL_IMPORT':'0','EGM_LITERAL_AUTO':'1','EGM_REGION_CLASSES':'256','EGM_STRICT_BUDGETS':'1','EGM_CANONICAL_PHASE':'1','EGM_CANONICAL_WIRES':'1','EGM_BOOLEAN':'0','EGM_LIBRARY_TIMING':'0'}
  assert all(row['settings'][key]==value for key,value in expected.items())
  assert sha(rp.parent/r['library'])==r['library_sha256']
  for profile in c['fixed_profiles']:
   name=f'{k}in_{bench}_{profile}';folder=out/name;folder.mkdir();report=copy.deepcopy(r);report['adoption']={'prior_report':str(rp),'prior_report_sha256':sha(rp),'prior_log':str(log),'prior_log_sha256':sha(log),'reason':'exact unchanged import code/input/library/config; recovery flags never executed','new_mapper_run':False,'new_runtime_measurement':False,'prior_imported_nodes':int(match[1])};report['results'][0]['method']='egraph_staged_control_resyn2';report['results'][0]['settings'].update(EGM_FAST_DELAY_SEED='1' if profile=='fast_frontier' else '0',EGM_CRITICAL_RECOVERY='0',EGM_BOUNDED_PROPOSALS='0',EGM_SEED_WINDOW_FRACTION='1.0');np=folder/'report.json';np.write_text(json.dumps(report,indent=2));j={'name':name,'k':k,'bench':bench,'profile':profile,'report':str(np),'report_sha256':sha(np),'exit_code':1,'adopted_from':str(oldpath),'adoption_reason':'pre-recovery import-node rejection; recovery source/flags cannot affect this stage','original_argv':oldjob['argv'],'new_mapper_run':False};c['jobs'].append(j);rows.append(report['adoption'])
(out/'campaign.json').write_text(json.dumps(c,indent=2));(out/'adoption.json').write_text(json.dumps({'proof':proof,'pilot_jobs_reused':4,'failure_slots_reused':20,'failure_sources':rows,'baseline_algorithms_changed':False,'limitations':'Retains failures only. No claim that new recovery ran or solved these designs; original logs/commands/runtime remain historical.'},indent=2));print('4 same-source pilot jobs and 20 unchanged import failures explicitly reused; 56 new jobs remain')
