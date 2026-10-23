#!/usr/bin/env python3
"""Re-analyse all historical EPFL cases without relabeling them as merged-main runs."""
import json,math,hashlib,collections
from pathlib import Path
R=Path(__file__).resolve().parents[3];O=R/'results_v2/main_integration_20261019'
load=lambda p:json.loads(Path(p).read_text())
def relation(a,b):
 t=1e-6*max(1,abs(a),abs(b));return 'better' if a<b-t else 'worse' if a>b+t else 'equal'
def main():
 cp=R/'results_v2/functional_cuts_full_20261019/campaign.json';c=load(cp);rows=[];summary=[]
 for k in [3,4]:
  b=load(R/f'results_v2/guarded_joint_evidence_20261016/baseline_scope/baselines_{k}in.json');stats={x:collections.Counter() for x in ['area','delay']};ratios={x:[] for x in stats}
  for bench in c['designs']:
   j=next(j for j in c['jobs'] if (j['k'],j['bench'],j['profile'])==(k,bench,'functional_cuts'));assert hashlib.sha256(Path(j['report']).read_bytes()).hexdigest()==j['report_sha256'];report=load(j['report']);assert report['library_sha256']==b['library_sha256'] and report['metric_model']==b['metric_model'];r=report['results'][0];e=next(x for x in b['results'] if (x['bench'],x['method'])==(bench,'mockturtle_emap_delay_resyn2'));assert r['input_sha256']==e['input_sha256'];row=dict(k=k,bench=bench,candidate_status=r['status'],emap_status=e['status'],candidate_report=j['report'],version='Historical isolated V31, NOT merged-main V36')
   if r['status']=='ok' and e['status']=='ok':
    row.update(candidate=r['measured'],emap=e['measured'])
    for metric in stats:
     row[metric+'_outcome']=relation(r['measured'][metric],e['measured'][metric]);stats[metric][row[metric+'_outcome']]+=1;ratios[metric].append(r['measured'][metric]/e['measured'][metric])
   rows.append(row)
  summary.append(dict(k=k,legal_pairs=len(ratios['delay']),total_designs=20,counts={k:dict(v)for k,v in stats.items()},conditional_geomean_ratio={m:math.exp(sum(map(math.log,v))/len(v)) for m,v in ratios.items()},qualification='Geomeans over matched legal pairs only; failures remain in rows, never count these aggregates as full20 success. Fixed emap delay profile, not per-design mode picking.'))
 smoke=load(R/'results_v2/main_integration_smoke_20261019/functional_validation.json');integration=load(O/'integration.json')
 q=dict(historical_comparison=rows,historical_summary=summary,new_main_smoke=dict(attempts=smoke['attempts'],CEC=smoke['cec'],summary=smoke['summary'],pair_coverage=smoke['pair_coverage']),old_main_sha256=integration['old_main_sha256'],new_main_sha256=integration['new_main_sha256'],main_source_merged=True,new_main_full20_complete=False,new_main_repeated=False,source_plus_recovery_full_suite_validated=False,hard_phase_screen_enabled_by_default=False,budget_or_baseline_algorithm_changed=False)
 (O/'analysis_data.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(summary,indent=2));print(q['new_main_smoke'])
if __name__=='__main__':main()
