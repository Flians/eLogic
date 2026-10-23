"""Audit complete and failed source plans, actual admission budgets, and paired PPA."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);a=p.parse_args()
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=load(a.report/'all_rows.json');pairs=load(a.report/'paired.json');s=load(a.report/'validation.json');errors=[];ledgers=[];coverage=[];gains=[]
for item in rows:
 row=item['row'];active=item['profile']=='protected_area';assert row['settings']['EGM_SHARED_MAJORITY']==str(int(active))
 plan=Path(row['settings']['EGM_POOL_DIAG_PATH']).with_suffix('.import-plan.json');dpath=plan.with_name('pool-diagnostics.import.json')
 if not plan.is_file():errors.append(dict(report=item['report'],reason='missing import plan'));continue
 if not load(plan)['xor_import']:continue
 if not dpath.is_file():errors.append(dict(report=item['report'],reason='missing import ledger'));continue
 d=load(dpath);ledgers.append(dict(k=item['k'],bench=item['bench'],profile=item['profile'],path=str(dpath),sha256=sha(dpath),data=d))
 assert d['schema_version']==4
 if active:
  if not d['import_complete']:
   assert row['status']!='ok' and d['cumulative_import_admitted_enodes']<=20000
   folder=Path(item['report']).parent/(row['bench']+'.'+row['method']);assert 'budget: cumulative import would exceed node allowance' in (folder/'map.log').read_text()
   continue
  assert d['source_planning_has_no_egraph_retirement']
  assert d['pre_rebuild_import_enodes']<=20000
  assert d['source_fringe_cap']<=16 and d['source_truth_queries_per_node_actual_max']<=16
  assert d['source_planning_choices_per_node']<=16 and d['actual_helpers']<=2048
  if d['compact_primitives']:
   assert d['raw_source_upper']>20000 and d['source_plan_after']<=d['source_plan_before']
  else:assert d['raw_source_upper']<=20000
 if row['status']=='ok':
  n=row['native'];assert row['measured']['cec']==1
  assert d['import_complete'] and d['actual_imported_enodes']==n['imported_enodes']<=n['enodes']<=20000
  assert n['max_cuts_per_class']<=16 and n['instances']<=100000 and n['heuristic_secs']<=2 and row['mapping_wall_secs']<=300
for pair in pairs:
 old,new=pair['control']['row'],pair['candidate']['row'];key=[pair['k'],pair['bench'],pair.get('observation'),pair.get('repetition')]
 if old['status']=='ok' and new['status']!='ok':errors.append(dict(case=key,reason='lost legal cover'));continue
 if old['status']!='ok' and new['status']=='ok':coverage.append(key)
 if old['status']==new['status']=='ok':
  for m in ['delay','area']:
   x,y=old['measured'][m],new['measured'][m];t=1e-6*max(1,abs(x),abs(y))
   if y>x+t:errors.append(dict(case=key,metric=m,old=x,new=y))
   if y<x-t:gains.append(dict(case=key,metric=m,old=x,new=y))
assert not s['budget_errors'] and not s['artifact_hash_errors']
source=load(Path(s['source_manifest']));full=len(source['designs'])==20 and len(rows)==80
result=dict(full_twenty_design_comparison=full,old_legal_regressions=errors,coverage_gains=coverage,metric_gains=gains,ledgers=ledgers,
 paired_nonregression_pass=not errors,observed_improvement=bool(coverage or gains),qualifies_for_promotion=full and not errors and bool(coverage or gains),
 all_twenty_mapped=full and sum(r['profile']=='protected_area' and r['row']['status']=='ok' for r in rows)==40,
 node_unit_caveat='Richer exact XOR/TT3 actual enodes, not unchanged expanded-AND work units; all admitted functions and helpers count; no egraph state retirement.',baseline_algorithms_changed=False)
(a.report/'source_plan_gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='ledgers'},indent=2))
