"""Layout-only draft: existing numbers and explicit placeholders, never delivered."""
from pathlib import Path
import shutil,subprocess
root=Path(__file__).resolve().parents[2];paper=root/'epmapping_paper/DATE27-ePMapping-sub';out=Path('/tmp/epm-v28-layout');out.mkdir(exist_ok=True)
draft=out/'paper'
shutil.copytree(paper,draft,dirs_exist_ok=True,ignore=shutil.ignore_patterns('data','*.pdf','*.png','*.log','*.aux'))
shutil.copytree(paper/'figs',draft/'figs',dirs_exist_ok=True)
generated=draft/'doc/generated_results.tex'
text=generated.read_text();start=text.index('\\newcommand{\\AblationResultTable}')
text=text[:start]+text[start:].replace('\\bottomrule',''.join(arm+' & -- & -- & -- & -- & -- & -- '+r'\\'+'\n' for arm in ['noQUERY','noTRUTH','allPIVOTS'])+'\\bottomrule',1)
generated.write_text(text)
if (paper/'DATE27-ePMapping.bbl').exists():shutil.copy2(paper/'DATE27-ePMapping.bbl',out/'layout_v28_draft.bbl')
names=[a+b for a in ['Query','Truth','Pivot'] for b in ['ColdRatio','WarmRatio','AreaCold','AreaWarm']]
names+=['FullCompletionSentence']
source=''.join('\\def\\'+n+'{--}' for n in names)+'\\input{DATE27-ePMapping.tex}'
with (out/'build.log').open('w') as log:
 for _ in range(2):subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-jobname=layout_v28_draft','-output-directory='+str(out),source],cwd=draft,stdout=log,stderr=subprocess.STDOUT,check=True)
pages=subprocess.check_output(['pdftotext',str(out/'layout_v28_draft.pdf'),'-'],text=True).split('\f');print('Draft pages',len([p for p in pages if p.strip()]),'references',[i+1 for i,p in enumerate(pages) if 'REFERENCES' in p or 'R EFERENCES' in p])
