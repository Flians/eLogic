#!/usr/bin/env python3
import argparse,collections,csv,json
from pathlib import Path
from check_single_output_components import outcome
def main():
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',nargs='*',type=Path,default=[]);a=p.parse_args();v=json.loads((a.campaign.parent/'binding_validation.json').read_text());a.output.mkdir(exist_ok=True)
 pairs=[];rows=v['rows']
 for x in rows:
  if x['profile']!='protected_exchange':continue
  for profile in ['legacy_control','functional_cuts']:
   b=next(y for y in rows if (y['k'],y['bench'],y['profile'])==(x['k'],x['bench'],profile));valid=x['valid'] and b['valid'];z=dict(k=x['k'],bench=x['bench'],reference=profile,valid=valid,reference_status=b['status'],candidate_status=x['status'])
   if valid:z.update(reference_area=b['area'],candidate_area=x['area'],reference_delay=b['delay'],candidate_delay=x['delay'],area_delta=x['area']-b['area'],delay_delta=x['delay']-b['delay'],area_outcome=outcome(x['area'],b['area']),delay_outcome=outcome(x['delay'],b['delay']))
   pairs.append(z)
 fields=list(dict.fromkeys(k for x in pairs for k in x));
 with (a.output/'paired.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(pairs)
 summary={f'K{k}_{ref}':{metric:dict(collections.Counter(x[metric]for x in pairs if x['k']==k and x['reference']==ref and x['valid'])) for metric in ['area_outcome','delay_outcome']}for k in [3,4]for ref in ['legacy_control','functional_cuts']}
 adverse=[x for x in pairs if x['valid'] and ('worse'in[x['area_outcome'],x['delay_outcome']])]
 refs={};
 for r in a.repeats:
  d=json.loads((r.parent/'binding_validation.json').read_text());refs[str(r)]={'attempts':d['attempts'],'CEC':d['cec'],'rows':d['rows'],'summary':d['summary']}
 q=dict(full_epfl_complete=v['complete'] and len(rows)==120,full_attempts=len(rows),full_CEC=v['cec'],summary=summary,adverse=adverse,full_shared_seed_protection=True,historical_full_budget_final_cover_protection=False,promotion_pass=not adverse and v['cec']==102,main_method_promoted=False,baseline_algorithms_changed=False,repeats=refs)
 (a.output/'summary.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q[k] for k in ['full_epfl_complete','full_attempts','full_CEC','summary','promotion_pass']},indent=2))
if __name__=='__main__':main()
