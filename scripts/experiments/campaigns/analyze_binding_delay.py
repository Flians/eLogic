#!/usr/bin/env python3
"""Separate immutable-pool delay limits from extraction and candidate gaps."""
import argparse,collections,hashlib,json
from pathlib import Path
from check_single_output_components import outcome
R=Path(__file__).resolve().parents[3]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=json.loads(a.campaign.read_text());records=[];bases={k:json.loads((R/f'results_v2/guarded_joint_evidence_20261016/baseline_scope/baselines_{k}in.json').read_text())for k in [3,4]}
 for j in c['jobs']:
  r=json.loads(Path(j['report']).read_text())['results'][0];z=dict(name=j['name'],k=j['k'],bench=j['bench'],profile=j['profile'],status=r['status'])
  if r['status']!='ok':records.append(z);continue
  assert sha(j['pool_frontier'])==j['pool_frontier_sha256'];f=json.loads(Path(j['pool_frontier']).read_text());before=f['before'];after=f['after'];assert before['pool_delay_certificate_pass'] and after['pool_delay_certificate_pass']
  selected=r['measured']['delay'];events=json.loads(Path(j['trace']).read_text())['events'];exchange=[e['data']for e in events if e['stage']=='protected_exchange'];changed=j['profile']=='protected_exchange' and any(x.get('committed')and not x.get('no_op')for x in exchange)
  z.update(raw_bound=before['root_label_bound'],retained_bound=after['root_label_bound'],selected_delay=selected,immutable_initial_pool=not changed,retention_outcome=outcome(after['root_label_bound'],before['root_label_bound']),initial_pool_source=j['pool_frontier'])
  if changed:z['new_binding_coverage_below_original_raw_bound']=outcome(selected,before['root_label_bound'])=='better'
  else:z['selected_attains_retained_bound']=outcome(selected,after['root_label_bound'])=='equal'
  z['baselines']={}
  for name in ['abc_map_delay_resyn2','mockturtle_emap_delay_resyn2']:
   b=next(x for x in bases[j['k']]['results']if x['bench']==j['bench']and x['method']==name)
   if b['status']!='ok':z['baselines'][name]={'status':b['status']};continue
   assert b['input_sha256']==r['input_sha256'];d=b['measured']['delay'];kind='no_selected_delay_disadvantage'
   if outcome(selected,d)=='worse':
    if outcome(before['root_label_bound'],d)=='worse':kind='raw_candidate_coverage_gap'
    elif outcome(after['root_label_bound'],d)=='worse':kind='retained_pool_gap'
    else:kind='extraction_or_declared_constraint_gap'
   if changed:kind='changed_pool_no_immutable_certificate_classification'
   z['baselines'][name]={'delay':d,'kind':kind}
  records.append(z)
 summary={f'K{k}_{profile}_{baseline}':dict(collections.Counter(x['baselines'][baseline].get('kind','baseline_failure')for x in records if x['k']==k and x['profile']==profile and 'baselines'in x))for k in [3,4]for profile in c['factors']for baseline in ['abc_map_delay_resyn2','mockturtle_emap_delay_resyn2']}
 a.output.write_text(json.dumps(dict(records=records,summary=summary,scope='Certificates prove unloaded dependent-pin D optimality ONLY for each immutable supplied pool; no area/lex/global-library optimum. Changed-pool output is never evaluated as if old certificate applied. Diagnostic computations are NOT fed back into budgeted mapping.'),indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
