"""Read-only joint-pool/recovery admission and canonical-main qualification checks."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);a=p.parse_args()
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=load(a.report/'all_rows.json');pairs=load(a.report/'paired.json');v=load(a.report/'validation.json');campaign=load(v['source_manifest']);jobs={j['report']:j for j in campaign['jobs']};errors=[];audits=[];failures=[]
if campaign.get('reference_full'):
 assert sha(campaign['reference_full'])==campaign['reference_full_sha256']
 jobs.update({j['report']:j for j in load(campaign['reference_full'])['jobs']})
for x in rows:
 r=x['row'];active=x['profile']=='protected_area';e=r['settings']
 for name in ['EGM_TIMING_WITNESS_POOL','EGM_ACTIVATE_DEFERRED','EGM_ONLINE_AREA']:assert e[name]==str(int(active))
 assert e['EGM_JOINT_CONES']=='0' and e['EGM_POOL_FRONTIER_AUDIT']=='1'
 folder=Path(x['report']).parent/(r['bench']+'.'+r['method']);ledger=Path(e['EGM_POOL_DIAG_PATH']).with_suffix('.import.json')
 if r['status']!='ok':
  f=dict(k=x['k'],bench=x['bench'],profile=x['profile'],report=x['report'],error=r.get('error'),map_log=(folder/'map.log').read_text())
  if ledger.is_file():
   d=load(ledger);f['import_ledger']=d
   if not d.get('import_complete',True):assert d['cumulative_import_admitted_enodes']<=d['node_allowance']==20000
  failures.append(f);continue
 n=r['native'];assert r['measured']['cec']==1 and n['imported_enodes']<=n['enodes']<=20000 and n['max_cuts_per_class']<=16 and n['instances']<=100000 and n['heuristic_secs']<=2 and r['mapping_wall_secs']<=300
 for metric in ['area','delay']:
  assert abs(n[metric]-r['measured'][metric])<=1e-6*max(1,abs(n[metric]),abs(r['measured'][metric])), 'native/public physical model disagreement'
 if ledger.is_file():
  d=load(ledger);assert d['import_complete'] and d['pre_rebuild_import_enodes']<=20000
 else:
  plan=load(Path(e['EGM_POOL_DIAG_PATH']).with_suffix('.import-plan.json'))
  assert not plan['xor_import'] and plan['compact_upper']<=18000 and plan['node_limit']==20000
 sidecar=Path(e['EGM_POOL_DIAG_PATH']).with_suffix('.frontier.json');q=load(sidecar)
 job=jobs.get(x['report'])
 if job:assert job['pool_frontier']==str(sidecar) and sha(sidecar)==job['pool_frontier_sha256']
 assert q['timing_witness_pool']==active and q['union_witness_instances']<=n['instances']
 for k in ['before','after']:assert q[k]['pool_delay_certificate_pass'] and q[k]['violations']==0
 assert q['after']['instances']==n['instances'] and q['before']['instances']>=q['after']['instances']
 if active:assert abs(q['after']['root_label_bound']-q['before']['root_label_bound'])<1e-5
 assert r['measured']['delay']+1e-5>=q['after']['root_label_bound']
 audits.append(dict(k=x['k'],bench=x['bench'],profile=x['profile'],observation=x.get('observation'),repetition=x.get('repetition'),source=str(sidecar),sha256=sha(sidecar),local_inequality_tolerance=1e-6,conservative_root_tolerance=2*n['eclasses']*1e-6,data=q))
for pair in pairs:
 x,y=pair['control']['row'],pair['candidate']['row']
 if x['status']=='ok' and y['status']!='ok':errors.append(dict(k=pair['k'],bench=pair['bench'],reason='lost legal control cover'))
 if x['status']==y['status']=='ok':
  for m in ['area','delay']:
   old,new=x['measured'][m],y['measured'][m]
   if new>old+1e-6*max(1,abs(old),abs(new)):errors.append(dict(k=pair['k'],bench=pair['bench'],metric=m,control=old,candidate=new,observation=pair.get('observation'),repetition=pair.get('repetition')))
main=load(Path(__file__).resolve().parents[3]/'results_v2/scored_main_full_report_20261015/all_rows.json');main={(x['k'],x['bench']):x for x in main if x['profile']=='protected_area'};main_errors=[];gains=[]
for x in rows:
 if x['profile']!='protected_area':continue
 old=main[x['k'],x['bench']]['row'];new=x['row'];key=dict(k=x['k'],bench=x['bench'],observation=x.get('observation'),repetition=x.get('repetition'))
 if old['status']=='ok' and new['status']!='ok':main_errors.append(dict(**key,reason='lost canonical-main cover'))
 if old['status']==new['status']=='ok':
  for m in ['area','delay']:
   oldval,newval=old['measured'][m],new['measured'][m];tol=1e-6*max(1,abs(oldval),abs(newval))
   d=dict(**key,metric=m,main=oldval,candidate=newval)
   if newval>oldval+tol:main_errors.append(d)
   if newval<oldval-tol:gains.append(d)
assert not v['budget_errors'] and not v['artifact_hash_errors']
result=dict(same_binary_regressions=errors,canonical_main_observation_regressions=main_errors,canonical_main_gains=gains,successful_admission_audits=audits,failures=failures,full20=len(campaign['designs'])==20 and campaign['expected_jobs']==80,qualifies_for_promotion=False,promotion_note='This checker does not integrate or certify repeated/full-main validation; an explicit qualification report is required. All old adverse observations remain.',baseline_algorithms_changed=False,pool_bound_scope='Only the supplied physical pool under unloaded dependent-pin delays, local inequality tolerance1e-6; conservative root error <=2*eclasses*1e-6. No exact-global/library/area/lex proof. Pre-cap retention ranking is generation work charged to original native/mapping guards; not a second recovery window.')
(a.report/'joint_budget_gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:val for k,val in result.items() if k not in ['successful_admission_audits','failures']},indent=2))
