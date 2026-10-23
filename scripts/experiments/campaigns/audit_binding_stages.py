#!/usr/bin/env python3
"""Supplementary audit of actual stage limits and eligible whole-cover rescue."""
import argparse,json,hashlib
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);a=p.parse_args();c=json.loads(a.campaign.read_text());records=[]
 for j in c['jobs']:
  if j['profile']!='protected_exchange':continue
  r=json.loads(Path(j['report']).read_text())['results'][0];b=json.loads(Path(j['admission']).read_text())
  assert b['resident_enodes']<=b['cumulative_admitted_enodes']<=20000 and not b['refund_on_rebuild']
  if r['status']!='ok':records.append(dict(name=j['name'],status=r['status'],admission=b));continue
  n=r['native'];events=json.loads(Path(j['trace']).read_text())['events'];ref=next(e['data'] for e in events if e['stage']=='protected_reference')
  assert n['exchange_cuts']<=64 and n['slack_local_trials']<=8000 and n['slack_rounds_started']<=4
  assert n['slack_local_trials']+n['online_area_updates']<=8000
  assert n['area']<=ref['area'] and n['delay']<=ref['delay']
  flows=[];saved=[];exchanges=[]
  for e in events:
   d=e['data']
   if e['stage']=='protected_exchange':
    if d.get('no_op'):assert not d['committed'] and d['raw_additional']==0
    else:assert d['offered_additional']<=2048 and d['new_cut_batch_limit']==64 and d['match_prefix_limit']==2048
    exchanges.append(d)
   if e['stage']=='fixed_bound_area_flow':
    assert d['fixed_delay_limit']<=ref['delay'] and d['protected_area_limit']==ref['area'] and d['shared_global_deadline'];flows.append(d)
   if e['stage']=='feasible_visited_incumbent':
    assert d['additional_evaluations']==0 and d['whole_cover_only'] and d['area_limit']==ref['area']
    assert d['area']<=ref['area'] and d['delay']<=d['fixed_delay_limit']<=ref['delay'];saved.append(d)
  if flows and saved:assert all(x['fixed_delay_limit']==flows[-1]['fixed_delay_limit'] for x in saved)
  records.append(dict(name=j['name'],status=r['status'],protected_reference=ref,area_flows=flows,visited_cover_archives=saved,exchanges=exchanges,actual_trials=n['slack_local_trials'],online_updates=n['online_area_updates'],native_inclusive_recovery=n['heuristic_secs']))
 data=dict(records=records,observed_stage_and_lifetime_limits_pass=True,local_stage_fractions_not_preemptive_hard_caps=True,historical_full_budget_final_cover_guarantee=False,checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (a.campaign.parent/'binding_stage_validation.json').write_text(json.dumps(data,indent=2)+'\n');print('stage audit passed',len(records))
if __name__=='__main__':main()
