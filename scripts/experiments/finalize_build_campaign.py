#!/usr/bin/env python3
"""Audit paired construction-reuse experiments and build an unapproved PDF for visual QA."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();p=a.root.resolve();scripts=ROOT/'scripts/experiments';paper=ROOT/'epmapping_paper/DATE27-ePMapping-sub';fig=paper/'figs/results'
 assert all(v==0 for v in json.loads((p/'execution_status.json').read_text()).values())
 def run(name,cmd,cwd=ROOT):
  with (p/name).open('w') as log:subprocess.run(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
 for mode in ['cold','warm']:run(f'{mode}_selection_audit.log',[sys.executable,str(scripts/'timeout_selection.py'),'--root',str(p/mode)])
 run('paired_summary_audit.log',[sys.executable,str(scripts/'report_three_parts_profiles.py'),'--root',str(p),'--out',str(paper/'doc/generated_results.tex')])
 for script,folder in [('plot_paper.py',p/'cold'),('plot_cofactor_ablation.py',p),('plot_profile_cache.py',p),('plot_build_ablation.py',p)]:
  run(script+'.log',[sys.executable,str(scripts/script),'--root',str(folder),'--out',str(fig)])
 run('summary_audit.log',[sys.executable,str(scripts/'summarize_cofactor_campaign.py'),'--root',str(p)])
 v=json.loads((p/'verified_profiles.json').read_text());plan=json.loads((p/'execution_plan.json').read_text())
 assert len(v['cache_runs'])==plan['expected_method_runs']
 totals={m:dict(completed=v['cohort_metrics'][m]['engine_cec_outputs'],timeouts=v['cohort_metrics'][m]['method_timeouts'],selected=v['cohort_metrics'][m]['selected_cec_outputs']) for m in ['cold','warm']}
 (p/'execution_complete.json').write_text(json.dumps(dict(complete=True,engine_runs=plan['expected_engine_runs'],method_runs=len(v['cache_runs']),fresh_abc_runs=16,cohorts=totals,verified_profiles_sha256=sha(p/'verified_profiles.json')),indent=2)+'\n')
 tex='DATE27-ePMapping'
 commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error',tex+'.tex'],['bibtex',tex],['pdflatex','-interaction=nonstopmode','-halt-on-error',tex+'.tex'],['pdflatex','-interaction=nonstopmode','-halt-on-error',tex+'.tex']]
 with (p/'paper_build.log').open('w') as log:
  for cmd in commands:subprocess.run(cmd,cwd=paper,stdout=log,stderr=subprocess.STDOUT,check=True)
 pdf=paper/(tex+'.pdf');pages=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True).split('\f');pages=pages[:-1] if not pages[-1].strip() else pages;refpages=[i+1 for i,page in enumerate(pages) if 'REFERENCES' in page or 'R EFERENCES' in page]
 log=(paper/(tex+'.log')).read_text();q=dict(pdf=str(pdf),sha256=sha(pdf),pages=len(pages),body_pages=(refpages[0]-1 if refpages else None),reference_pages=len(pages)-refpages[0]+1 if refpages else None,visual_QA_pending=True,visually_checked_pages=[],overfull_boxes='Overfull \\hbox' in log or 'Overfull \\vbox' in log,unresolved_references=any(x in log for x in ['undefined references','Citation `','Reference `']))
 (p/'build_ready.json').write_text(json.dumps(q,indent=2)+'\n')
 print(json.dumps(q))
if __name__=='__main__':main()
