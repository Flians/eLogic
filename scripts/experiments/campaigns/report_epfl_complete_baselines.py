#!/usr/bin/env python3
"""Audit every original baseline checkpoint and publish all statuses, not winners."""
import argparse,json,csv,hashlib
from pathlib import Path
from run_epfl_baselines import EPFL,sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();c=json.loads(a.campaign.read_text());assert len(c['jobs'])==240 and not c.get('current_job');out=a.output;out.mkdir(exist_ok=False)
 manifest=json.loads(Path(c['subjects']).read_text());subjects={r['bench']:r for r in manifest['results']};rows=[];hashes={str(a.campaign):sha(a.campaign),str(a.candidate):sha(a.candidate)};coverage={};counts={'successful_rows':0,'failed_rows':0,'unmeasured_jobs':0}
 for j in c['jobs']:
  k=int(j['name'][0]);coverage.setdefault((k,j['bench']),[]).append(j['name']);goal=next((v for v in ['area-delay','delay-area'] if v in j['name']),None)
  for key in ['report','audit']:
   if j.get(key):assert sha(Path(j[key]))==j[key+'_sha256'];hashes[j[key]]=j[key+'_sha256']
  if not j.get('report'):
   counts['unmeasured_jobs']+=1
   missing_methods=['abc_amap','lut_stage_amap','eqmap_lut_amap'] if '_eqmap_' in j['name'] else ['emorphic_final_sa'] if '_emorphic_' in j['name'] else ['esyn_frozen_model_amap_bounded_dag']
   for method in missing_methods: rows.append({'k':k,'bench':j['bench'],'objective':goal,'method':method,'status':'unmeasured','error':j.get('error','no report'),'job':j['name'],'report':None,'job_exit_code':j['exit_code']})
   continue
  d=json.loads(Path(j['report']).read_text())
  for r in d['results']:
   assert r['input_sha256']==subjects[r['bench']]['input_sha256'];assert r['preprocessing']['sha256']==subjects[r['bench']]['preprocessing']['sha256']
   if r['status']=='ok':assert r['measured']['cec']==1
   counts['successful_rows' if r['status']=='ok' else 'failed_rows']+=1
   rows.append({'k':k,'bench':r['bench'],'objective':goal,'method':r['method'],'status':r['status'],'area':(r.get('measured') or {}).get('area'),'delay':(r.get('measured') or {}).get('delay'),'error':r.get('error'),'job':j['name'],'report':j['report'],'library_sha256':d['library_sha256']})
 assert len(coverage)==40 and all(len(v)==6 for v in coverage.values())
 counts['unmeasured_summary_rows']=sum(r['status']=='unmeasured' for r in rows)
 assert len(rows)==520, len(rows)
 cand=json.loads(a.candidate.read_text());candidates={}
 for j in cand['jobs']:
  if j['profile']!='coupled_queue':continue
  assert sha(Path(j['report']))==j['report_sha256'];d=json.loads(Path(j['report']).read_text());r=d['results'][0]
  if r['status']=='ok':assert r['measured']['cec']==1 and r['native']['enodes']<=20000 and r['native']['instances']<=100000 and r['native']['max_cuts_per_class']<=16 and r['native']['heuristic_secs']<=2
  candidates[j['k'],j['bench']]=(r,d['library_sha256'])
 assert len(candidates)==40
 fields=list(dict.fromkeys(key for r in rows for key in r));
 with (out/'all_baselines.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 (out/'all_baselines.json').write_text(json.dumps(rows,indent=2)+'\n')
 methods=['abc_map_delay_resyn2','mockturtle_emap_delay_resyn2','eqmap_lut_amap','esyn_frozen_model_amap_bounded_dag','emorphic_final_sa'];tables=[]
 for k in [3,4]:
  for metric in ['area','delay']:
   lines=[r'\begin{table*}[t]\centering\small',r'\setlength{\tabcolsep}{6pt}',r'\caption{EPFL '+str(k)+'-input library '+metric+r' for fixed delay-oriented profiles. F denotes failed or unmeasured output, N the queue study node-budget rejection. All numeric entries pass independent named-port CEC.}',r'\label{tab:epfl'+str(k)+metric+r'}',r'\begin{tabular}{lrrrrrr}\toprule',r'Design & ABC map & emap & EqMap & E-Syn & E-morphic & Queue study \\ \midrule']
   for bench in EPFL:
    r,lib=candidates[k,bench];values=[]
    for method in methods:
     matches=[x for x in rows if x['k']==k and x['bench']==bench and x['method']==method and x['objective'] in [None,'delay-area']];assert len(matches)==1,(k,bench,method,matches);x=matches[0]
     if x.get('library_sha256'):assert x['library_sha256']==lib
     values.append(f"{x[metric]:.2f}" if x['status']=='ok' else 'F')
    values.append(f"{r['measured'][metric]:.2f}" if r['status']=='ok' else 'N')
    lines.append(bench.replace('_',r'\_')+' & '+' & '.join(values)+r' \\')
   lines.extend([r'\bottomrule\end{tabular}\end{table*}']);tables.append('\n'.join(lines))
 (out/'tables.tex').write_text('\n\n'.join(tables)+'\n');v={'jobs':240,'unique_design_library_pairs':40,'configs_per_pair':6,'counts':counts,'hashes':hashes,'baseline_algorithms_changed':False,'all_attempted_not_all_successful':True,'queue_study_source':'isolated immutable same-binary comparison; main opt-in package not relabelled full main-run','unconnected':['MtM']};(out/'validation.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(counts))
if __name__=='__main__':main()
