"""Rebuild the final prose, record layout checks, and render every page for inspection."""
import hashlib,json,subprocess
from pathlib import Path
project=Path(__file__).resolve().parents[2];research=Path(__file__).resolve().parent
paper=project/'epmapping_paper/DATE27-ePMapping-sub';campaign=project/'results_v28_three_parts_query_budget'
with (research/'final_manuscript_build.log').open('w') as log:
 for _ in range(2):subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','DATE27-ePMapping.tex'],cwd=paper,stdout=log,stderr=subprocess.STDOUT,check=True)
pdf=paper/'DATE27-ePMapping.pdf';pages=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True).split('\f');pages=[p for p in pages if p.strip()]
refs=[i+1 for i,p in enumerate(pages) if 'REFERENCES' in p or 'R EFERENCES' in p]
log=(paper/'DATE27-ePMapping.log').read_text()
ready=dict(pdf=str(pdf),sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),pages=len(pages),body_pages=refs[0]-1,reference_pages=len(pages)-refs[0]+1,visual_QA_pending=True,visually_checked_pages=[],overfull_boxes='Overfull \\hbox' in log or 'Overfull \\vbox' in log,unresolved_references=any(s in log for s in ['undefined references','Citation `','Reference `']))
(campaign/'build_ready.json').write_text(json.dumps(ready,indent=2)+'\n');print(json.dumps(ready),flush=True)
assert ready['pages']==7 and ready['body_pages']==6 and ready['reference_pages']==1
assert not ready['overfull_boxes'] and not ready['unresolved_references']
render=Path('/tmp/epm-v28-final');render.mkdir(exist_ok=True)
subprocess.run(['pdftoppm','-scale-to','1600','-png',str(pdf),str(render/'page')],check=True)
