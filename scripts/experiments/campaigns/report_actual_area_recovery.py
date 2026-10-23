#!/usr/bin/env python3
"""Read-only full-scope comparison. Resident-only covers never become legacy wins."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def summarize(rows):
 out={}
 for field in ['delay','area']:
  counts=[0,0,0]
  for a,b in rows:
   x=a['measured'][field];y=b['measured'][field];tol=1e-6*max(1,abs(x),abs(y));counts[0 if y<x-tol else 2 if y>x+tol else 1]+=1
  out[field+'_better_equal_worse']=counts
 return out
p=ROOT/'results_v2/actual_area_recovery_full_20261014/report.json';x=read(p)
out=ROOT/'results_v2/actual_area_recovery_report_20261014';out.mkdir(exist_ok=False)
reference=ROOT/'results_v2/retirement_cuts_v2_epfl_full_20261014/report.json';old=read(reference)
lookup={(r['k'],r['bench']):r for r in old['results']};pairs=[];audit_errors=[]
for r in x['results']:
 if r['status']=='diagnostic_cec_ok':
  b=r['budget'];m=r['measured']
  if m['cec']!=1 or m['delay']>b['fixed_delay_bound']+1e-5 or b['global_recovery_seconds']>2 or b['global_pool_prefix_plus_plans_and_proposals_peak']>100000 or b['global_resident_charged_state_peak']>20000 or b['cuts_per_original_class']>16:audit_errors.append([r['k'],r['bench']])
 a=lookup.get((r['k'],r['bench']))
 if a and a['status']==r['status']=='diagnostic_cec_ok' and a['formal_same_import_budget_eligible'] and r['formal_same_import_budget_eligible']:
  assert all(a[k]==r[k] for k in ['input_sha256','prepared_sha256','library_sha256']);pairs.append((a,r))
main_path=ROOT/'results_v2/retirement_cuts_v2_epfl_report_20261014/paired_main.json'
main_lookup={(r['k'],r['bench']):r for r in read(main_path)};main_pairs=[];main_rows=[]
for r in x['results']:
 a=main_lookup[(r['k'],r['bench'])];main_rows.append({'k':r['k'],'bench':r['bench'],'control_status':a['control_status'],'trial_status':r['status'],'control':a['control'],'trial':r.get('measured'),'legacy_eligible':r['formal_same_import_budget_eligible']})
 if a['control_status']=='ok' and r['status']=='diagnostic_cec_ok' and r['formal_same_import_budget_eligible']:main_pairs.append(({'measured':a['control']},r))
(out/'paired_main.json').write_text(json.dumps(main_rows,indent=2))
validation={'attempted':len(x['results']),'expected':40,'cec_passes':sum(r['status']=='diagnostic_cec_ok' for r in x['results']),'legacy_eligible_cec':sum(r['status']=='diagnostic_cec_ok' and r['formal_same_import_budget_eligible'] for r in x['results']),'failures':[r for r in x['results'] if r['status']!='diagnostic_cec_ok'],'actual_bound_budget_errors':audit_errors,'paired_v2_count':len(pairs),'paired_v2':summarize(pairs),'paired_main_count':len(main_pairs),'paired_main':summarize(main_pairs),'production_promoted':False,'baseline_algorithms_changed':False,'incumbent_scope':'regional initial cover only, not production incumbent','source_reports':{str(p):sha(p),str(reference):sha(reference)}}
(out/'paired_v2.json').write_text(json.dumps(pairs,indent=2));(out/'validation.json').write_text(json.dumps(validation,indent=2));print(json.dumps({k:v for k,v in validation.items() if k not in ['failures','source_reports']},indent=2))
