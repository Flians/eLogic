#!/usr/bin/env python3
"""Inspect original SA's recorded author-selected incumbent; never rerank candidates."""
import json
from pathlib import Path
from analyze_delay_causes import ROOT,reconstruct,sha
out=ROOT/'results_v2/delay_cause_analysis_20261015';existing=json.load(open(out/'paths.json'))['records'];idx={(r['k'],r['bench']):r for r in existing if r['profile']=='production'}
rows=json.load(open(ROOT/'results_v2/epfl_original_baselines_complete_v2_20261013/all_baselines.json'));records=[];errors=[]
for row in rows:
 if row['method']!='emorphic_final_sa' or row['status']!='ok' or (row['k'],row['bench']) not in idx:continue
 try:
  report=Path(row['report']);x=json.load(open(report));r=next(r for r in x['results'] if r['bench']==row['bench']);a=idx[row['k'],row['bench']]
  assert r['status']=='ok' and r['measured']['cec']==1
  assert r['input_sha256']==a['input_sha256'] and r['preprocessing']['sha256']==a['prepared_sha256'] and x['library_sha256']==a['library_sha256']
  path=report.parent/r['bench']/'trace'/r['selected_candidate']/'mapped.json'
  z=reconstruct(path,report.parent/'library.json',r['measured']);z.update(k=row['k'],bench=row['bench'],profile='emorphic_final_sa',selected_native_candidate=r['selected_candidate'],selection_policy='original native winner; no public metric reranking',report=str(report),report_sha256=sha(report),input_sha256=a['input_sha256'],prepared_sha256=a['prepared_sha256'],library_sha256=a['library_sha256']);records.append(z)
 except Exception as e:errors.append({'k':row['k'],'bench':row['bench'],'error':repr(e)})
(out/'author_selected_sa_paths.json').write_text(json.dumps({'records':records,'errors':errors,'baseline_algorithms_changed':False,'new_mapping_runs':0,'script_sha256':sha(Path(__file__))},indent=2));print('SA metrics reproduced',len(records),'errors',errors)
for r in records:
 if r['bench'] in ['priority','bar','max','ctrl']:print(r['k'],r['bench'],r['delay'],r['critical_path_cells'],r['critical_path_types'])
