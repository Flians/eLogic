"""Audit explicit changed representation; do not confuse it with unchanged pools."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--threshold-percent',type=int,default=80);a=p.parse_args()
rows=json.loads((a.report/'all_rows.json').read_text());pairs=json.loads((a.report/'paired.json').read_text());summary=json.loads((a.report/'validation.json').read_text());errors=[];gains=[];changed=[];coverage=[];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for pair in pairs:
 x,y=pair['control'],pair['candidate'];old,new=x['row'],y['row'];key=[pair['k'],pair['bench'],pair.get('observation'),pair.get('repetition')]
 assert old['input_sha256']==new['input_sha256'] and old['preprocessing']['sha256']==new['preprocessing']['sha256'] and x['library_sha256']==y['library_sha256']
 plan_path=Path(new['settings']['EGM_POOL_DIAG_PATH']).with_suffix('.import-plan.json');
 if not plan_path.exists():errors.append(dict(case=key,reason='missing import-plan audit'));continue
 plan=json.loads(plan_path.read_text());assert plan.get('threshold_percent',80)==a.threshold_percent;assert plan['xor_import']==(plan['compact_upper']>20000*a.threshold_percent//100)
 if old['status']=='ok' and new['status']!='ok':errors.append(dict(case=key,reason='lost legal cover'));continue
 if old['status']!='ok' and new['status']=='ok':
  folder=Path(x['report']).parent/(old['bench']+'.'+old['method']);text=(folder/'map.log').read_text();match=re.search(r'imported graph (\d+) exceeds node allowance (\d+)',text);assert match and int(match[1])>int(match[2])==20000
  coverage.append(key)
 if new['status']=='ok':
  n=new['native'];assert new['measured']['cec']==1 and n['imported_enodes']<=n['enodes']<=20000 and n['max_cuts_per_class']<=16 and n['instances']<=100000 and n['heuristic_secs']<=2 and new['mapping_wall_secs']<=300
  if plan['xor_import']:
   path=plan_path.with_name('pool-diagnostics.import.json');d=json.loads(path.read_text());assert d['actual_imported_enodes']==n['imported_enodes'];assert d['retained_xor_operators']+d.get('retained_tt3_operators',0)<=d['original_source_ands'];assert d.get('actual_helpers',0)<=d.get('helper_quota',2048);assert not d.get('actual_helpers',0) or d['actual_imported_enodes']<=18000;changed.append(dict(case=key,plan=str(plan_path),plan_sha256=sha(plan_path),ledger=str(path),ledger_sha256=sha(path),ledger_data=d))
 if old['status']==new['status']=='ok':
  if not plan['xor_import'] and x.get('pool_fingerprint')!=y.get('pool_fingerprint'):errors.append(dict(case=key,reason='inactive representation changed pool'))
  for metric in ['area','delay']:
   u,v=old['measured'][metric],new['measured'][metric];tol=1e-6*max(1,abs(u),abs(v))
   if v>u+tol:errors.append(dict(case=key,metric=metric,control=u,candidate=v))
   if pair['bench']=='sqrt' and metric=='delay' and v<u-tol:gains.append(key)
assert not summary['budget_errors'] and not summary['artifact_hash_errors']
source=json.loads(Path(summary['source_manifest']).read_text())
full_suite=len(set(source['designs']))==20 and {tuple((q['k'],q['bench'])) for q in pairs}=={(k,b) for k in [3,4] for b in source['designs']}
result=dict(threshold_percent=a.threshold_percent,full_suite=full_suite,declared_designs=source['designs'],old_legal_regressions=errors,coverage_gains=coverage,sqrt_delay_gains=gains,changed_representation=changed,unchanged_pool_mismatch_is_error=True,changed_pool_is_intentional=True,baseline_algorithms_changed=False,pass_gate=not errors and {x[0] for x in gains}=={3,4} and {tuple(x[:2]) for x in coverage}>={(3,'log2'),(4,'log2'),(3,'multiplier'),(4,'multiplier')})
(a.report/'import_gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='changed_representation'},indent=2));raise SystemExit(0 if result['pass_gate'] else 1)
