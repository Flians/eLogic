"""Audit identical-pool DA recovery; preserve failures and raw regressions."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);a=p.parse_args()
pairs=json.loads((a.report/'paired.json').read_text());v=json.loads((a.report/'validation.json').read_text());errors=[];used=[];missing=[]
def cmp(x,y):
 t=1e-6*max(1,abs(x),abs(y));return -1 if x<y-t else 1 if x>y+t else 0
for pair in pairs:
 x,y=pair['control'],pair['candidate'];old,new=x['row'],y['row'];key=[pair['k'],pair['bench'],pair.get('observation'),pair.get('repetition')]
 assert old['input_sha256']==new['input_sha256'] and old['preprocessing']['sha256']==new['preprocessing']['sha256'] and x['library_sha256']==y['library_sha256']
 assert old['settings']['EGM_SCORED_INCUMBENT']=='0' and new['settings']['EGM_SCORED_INCUMBENT']=='1'
 assert old['settings']['EGM_XOR_IMPORT']==new['settings']['EGM_XOR_IMPORT']=='1'
 if old['status']=='ok' and new['status']!='ok':errors.append(dict(case=key,reason='lost legal cover'))
 for row in [old,new]:
  if row['status']=='ok':
   n=row['native'];assert row['measured']['cec']==1 and n['enodes']<=20000 and n['max_cuts_per_class']<=16 and n['instances']<=100000 and n['heuristic_secs']<=2 and row['mapping_wall_secs']<=300
 if old['status']==new['status']=='ok':
  if x.get('pool_fingerprint')!=y.get('pool_fingerprint'):errors.append(dict(case=key,reason='different physical pool'))
  for metric in ['delay','area']:
   if cmp(new['measured'][metric],old['measured'][metric])>0:errors.append(dict(case=key,metric=metric,control=old['measured'][metric],candidate=new['measured'][metric]))
  folder=Path(y['report']).parent/f"{new['bench']}.{new['method']}";trace=folder/'recovery-trace.json'
  events=[e['data'] for e in json.loads(trace.read_text())['events'] if e['stage']=='scored_incumbent']
  for e in events:
   assert e['additional_evaluations']==0
   if e['used']:
    assert e['updates']>0
    f=new['measured'];assert (cmp(f['delay'],e['delay']) or cmp(f['area'],e['area']))<=0
    used.append(dict(case=key,event=e,final=new['measured']))
 else:missing.append(dict(case=key,control_status=old['status'],candidate_status=new['status']))
assert not v['budget_errors'] and not v['artifact_hash_errors']
source=json.loads(Path(v['source_manifest']).read_text());full_suite=len(set(source['designs']))==20 and len({(q['k'],q['bench']) for q in pairs})==40
r=dict(full_suite_attempted=full_suite,declared_designs=source['designs'],common_valid_pairs=v['eligible_pairs'],raw_regressions=errors,nonlegal_pairs=missing,scored_incumbent_used=used,no_extra_evaluation_claim='Reuse already validated scoring cover; no new metrics call in retention code. Test seam compares scoring counts; source patch and limits retained.',paired_nonregression_pass=not errors,declared_scope_all_mapped=not missing,all_twenty_mapped=full_suite and not missing,baseline_algorithms_changed=False,production_promoted=False)
(a.report/'scored_gate.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['scored_incumbent_used','nonlegal_pairs']},indent=2));raise SystemExit(0 if not errors else 1)
