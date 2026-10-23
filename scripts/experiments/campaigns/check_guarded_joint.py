"""Read-only global-budget, immutable-source and historical-main comparison audit."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--report',required=True,type=Path);a=p.parse_args();R=Path(__file__).resolve().parents[3]
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
v=load(a.report/'validation.json');c=load(v['source_manifest']);rows=load(a.report/'all_rows.json');pairs=load(a.report/'paired.json');jobs={j['report']:j for j in c['jobs']};errors=[];adopted=[];certs=[]
if c.get('reference_full'):
 assert sha(c['reference_full'])==c['reference_full_sha256'];jobs.update({j['report']:j for j in load(c['reference_full'])['jobs']})
# Adopted status slots share one original report; look up by name/profile as well.
alljobs=list(c['jobs'])
if c.get('reference_full'):alljobs+=load(c['reference_full'])['jobs']
for x in rows:
 r=x['row'];j=next(j for j in alljobs if j['name']==x['job']);active=x['profile']=='protected_area'
 for key in ['report','audit','pool','candidates','trace','pool_frontier','pool_guard','canonical_import','diagnostic_profile']:
  if key in j:assert sha(j[key])==j[key+'_sha256'],key
 if j.get('adopted_from'):
  assert r['status']=='failed' and not r.get('measured') and not r.get('partial_measured') and not r.get('partial_status') and j['new_mapper_run'] is False
  assert sha(j['adopted_from'])==j['source_campaign_sha256'];assert sha(j['adoption_proof'])==j['adoption_proof_sha256'];proof=load(j['adoption_proof']);assert proof['source_campaign_sha256']==j['source_campaign_sha256'];assert j['source_job'] in proof['source_failure_jobs']
  d=load(j['canonical_import']);assert d['canonical_source_inputs'] is False and not d['import_complete'] and d['actual_nodes']==d['node_allowance']==20000
  adopted.append(dict(job=j['name'],source_job=j['source_job'],report=j['report'],proof=j['adoption_proof'],no_fresh_mapper_or_runtime=True));continue
 e=r['settings']
 for flag in ['EGM_TIMING_WITNESS_POOL','EGM_ONLINE_AREA','EGM_ACTIVATE_DEFERRED','EGM_PRESERVE_SHARED_TRAJECTORY','EGM_FUSED_COVER_SCORING','EGM_SHARED_AREA_BOUND']:assert e[flag]==str(int(active)),flag
 assert e['EGM_CANONICAL_SOURCE_INPUTS']=='0' and e['EGM_LIBRARY_TIMING']=='0' and e['EGM_JOINT_CONES']=='0'
 if r['status']!='ok':errors.append(dict(job=j['name'],phase='failed actual attempt',error=r.get('error')));continue
 n=r['native'];m=r['measured'];assert m['cec']==1 and n['imported_enodes']<=n['enodes']<=20000 and n['max_cuts_per_class']<=16 and n['instances']<=100000 and n['heuristic_secs']<=2 and r['mapping_wall_secs']<=300
 for metric in ['area','delay']:assert abs(n[metric]-m[metric])<=1e-6*max(1,abs(n[metric]),abs(m[metric]))
 trace=load(j['trace']);total=[ev['seconds'] for ev in trace['events'] if ev['stage']=='global_total'];assert len(total)==1 and total[0]<=2
 folder=Path(j['pool']);ledger=folder.with_suffix('.import.json')
 if ledger.is_file():d=load(ledger);assert d['import_complete'] and d['pre_rebuild_import_enodes']<=20000
 else:d=load(folder.with_suffix('.import-plan.json'));assert not d['xor_import'] and d['compact_upper']<=18000
 q=load(j['pool_frontier']);assert q['timing_witness_pool']==active and q['union_witness_instances']<=n['instances'];assert q['after']['instances']==n['instances']
 for side in ['before','after']:assert q[side]['pool_delay_certificate_pass'] and q[side]['violations']==0
 if active:assert abs(q['after']['root_label_bound']-q['before']['root_label_bound'])<1e-5
 assert m['delay']+1e-5>=q['after']['root_label_bound']
 if 'pool_guard' in j:
  pg=load(j['pool_guard']);assert active and pg['guard'];assert pg['adopt_fast']==(pg['full_root_delay']+1e-6<pg['old_root_delay'])
 certs.append(dict(job=j['name'],k=x['k'],bench=x['bench'],profile=x['profile'],root_bound=q['after']['root_label_bound'],measured_delay=m['delay'],conservative_accumulated_margin=2*n['eclasses']*1e-6,frontier_sha256=j['pool_frontier_sha256']))
def compare(old,new,key):
 if old['status']=='ok' and new['status']!='ok':return dict(**key,lost_legal_cover=True)
 if old['status']!=new['status'] or old['status']!='ok':return None
 om,nm=old['measured'],new['measured'];diff={m:nm[m]-om[m] for m in ['area','delay']};worse=[m for m,d in diff.items() if d>1e-6*max(1,abs(nm[m]),abs(om[m]))]
 absolute_worse=[m for m,d in diff.items() if d>1e-6];lex_worse=diff['delay']>1e-6 or abs(diff['delay'])<=1e-6 and diff['area']>1e-6
 if worse or absolute_worse:return dict(**key,difference=diff,reporter_tolerance_worse=worse,absolute_worse=absolute_worse,delay_area_lex_worse=lex_worse)
control=[]
for pair in pairs:
 x=compare(pair['control']['row'],pair['candidate']['row'],dict(k=pair['k'],bench=pair['bench'],observation=pair.get('observation'),repetition=pair.get('repetition')))
 if x:control.append(x)
main={(x['k'],x['bench']):x['row'] for x in load(R/'results_v2/scored_main_full_report_20261015/all_rows.json') if x['profile']=='protected_area'};historical=[]
for row in rows:
 if row['profile']=='protected_area':
  x=compare(main[row['k'],row['bench']],row['row'],dict(k=row['k'],bench=row['bench'],observation=row.get('observation'),repetition=row.get('repetition')))
  if x:historical.append(x)
assert not v['budget_errors'] and not v['artifact_hash_errors']
result=dict(successful_budget_checks=len(certs),actual_failed_attempts=errors,adopted_preimport_failure_slots=adopted,same_binary_adverse_observations=control,canonical_main_adverse_observations=historical,pool_delay_brackets=certs,production_promoted=False,qualifies_for_promotion=False,scope='Adverse D/A/lex observations retained separately. Supplied frozen physical pool only, local1e-6 inequality tolerance and accumulated error2*eclasses*1e-6; no area/global/cut optimality claim. Explicit failed pre-import adoption is not fresh execution or successful/PPA mixing. A separate complete qualification and actual-main validation are required.')
(a.report/'budget_and_incumbent_gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['pool_delay_brackets','adopted_preimport_failure_slots']},indent=2))
