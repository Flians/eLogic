#!/usr/bin/env python3
"""Account isolated retired-state observations without promoting them into the main mapper."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
a.output.mkdir(parents=True,exist_ok=False);c=json.loads((a.campaign/'report.json').read_text());base=ROOT/'results_v2/epfl_original_baselines_complete_v2_20261013/all_baselines.json';baselines=json.loads(base.read_text())
rows=[]
for r in c['results']:
    if r['status']=='diagnostic_cec_ok':
        b=r['budget'];assert b['resident_peak_enodes']<=20000 and b['global_pool_committed']<=100000 and b['global_recovery_seconds']<=2 and b['cuts_per_original_class']<=16 and b['saturation_iterations']<=16
        assert r['mapping_wall_seconds']<=300 and r['measured']['cec']==1
    refs=[b for b in baselines if b['k']==r['k'] and b['bench']==r['bench'] and b['method']=='abc_map_delay_resyn2']
    assert len(refs)==1;ref=refs[0];assert ref['library_sha256']==r['library_sha256']
    row={'k':r['k'],'bench':r['bench'],'repetition':r['repetition'],'status':r['status'],'resident_diagnostic_cec':r['status']=='diagnostic_cec_ok','old_import_scope_eligible':r['formal_same_import_budget_eligible'],'abc_map_delay_status':ref['status'],'native_budget':r.get('budget'), 'error':r.get('error')}
    if r['status']=='diagnostic_cec_ok' and ref['status']=='ok':
        row.update(candidate_area=r['measured']['area'],candidate_delay=r['measured']['delay'],abc_area=ref['area'],abc_delay=ref['delay'],delay_percent_vs_abc=100*(r['measured']['delay']/ref['delay']-1),area_percent_vs_abc=100*(r['measured']['area']/ref['area']-1))
    rows.append(row)
summary={'observations':len(rows),'status_counts':dict(Counter(r['status'] for r in rows)),'old_import_scope_eligible_cec':sum(r['old_import_scope_eligible'] and r['resident_diagnostic_cec'] for r in rows),'resident_only_cec_not_legacy_budget_success':sum(not r['old_import_scope_eligible'] and r['resident_diagnostic_cec'] for r in rows),'production_promoted':False,'scope':'Fixed primitive retired-state diagnostic; original baseline algorithms unchanged. Node residency is separately reported from cumulative work; cannot relabel a resident-only cover as a prior import-budget win. No best-run mixing.','baseline_report_sha256':sha(base),'campaign_report_sha256':sha(a.campaign/'report.json'),'rows':rows}
(a.output/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print({k:v for k,v in summary.items() if k!='rows'})
