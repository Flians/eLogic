#!/usr/bin/env python3
"""Per-design cold/warm runtime effects of independently audited controls."""
import argparse,hashlib,json,math,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/epmapping-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from run_three_parts import DESIGNS
from timeout_selection import read_rows

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();root=a.root;out=a.out;out.mkdir(parents=True,exist_ok=True)
 v=json.loads((root/'verified_profiles.json').read_text())
 for m,h in v['cohort_certificate_sha256'].items():assert sha(root/m/'verified_metrics.json')==h
 arms=['noQUERY','noTRUTH','allPIVOTS'] if 'noQUERY' in v['cohort_metrics']['cold']['ablation'] else ['cache64K','noAIGREUSE','noLIBSHARE'];labels=['No query accel.','No TT reuse','All pivots'] if arms[0]=='noQUERY' else ['64K cache','No AIG reuse','No lib sharing']
 times={(r['mode'],r['arm'],r['design']):r for r in v['cache_runs'] if r['stage']=='ablation'}
 matrices={};timeouts={};ratios={}
 for mode in ['cold','warm']:
  ratios[mode]=[[times[mode,arm,d]['optimizer_seconds']/times[mode,'full',d]['optimizer_seconds'] for arm in arms] for d in DESIGNS]
  matrices[mode]=np.log2(np.array(ratios[mode]))
  timeouts[mode]=[[any(times[mode,x,d]['status']=='timeout' for x in [arm,'full']) for arm in arms] for d in DESIGNS]
 plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42,'mathtext.fontset':'stix'})
 bound=max(.5,*[abs(x) for mat in matrices.values() for x in mat.ravel()])
 fig,axes=plt.subplots(1,3,figsize=(7.15,3.0),gridspec_kw={'width_ratios':[1.1,1.2,1.0]})
 y=np.arange(3);aggregate={}
 for mode,offset,color in [('cold',-.16,'#24628d'),('warm',.16,'#bd6b2c')]:
  aggregate[mode]=[v['cohort_metrics'][mode]['ablation'][arm]['time_ratio'] for arm in arms]
  axes[0].barh(y+offset,aggregate[mode],height=.28,color=color,label=mode.capitalize())
  for yy,value in zip(y+offset,aggregate[mode]):axes[0].annotate(f'{value:.2f}',(value,yy),xytext=(2,0),textcoords='offset points',ha='left',va='center',fontsize=6.5)
 axes[0].set(yticks=y,yticklabels=labels,xlabel='Time/full (all runs)',title='(a) Total runtime',xlim=(0,max(*aggregate['cold'],*aggregate['warm'])*1.3))
 axes[0].axvline(1,color='#777777',ls='--',lw=.7);axes[0].invert_yaxis();axes[0].grid(axis='x',color='#dddddd',lw=.5)
 axes[0].legend(frameon=False,fontsize=6.5,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.22))
 for i,(ax,mode) in enumerate(zip(axes[1:],['cold','warm']),1):
  im=ax.imshow(matrices[mode],cmap='RdBu_r',norm=TwoSlopeNorm(0,-bound,bound),aspect='auto')
  ax.set(xticks=range(3),xticklabels=labels,yticks=range(16),yticklabels=DESIGNS if i==1 else [],title=f'({chr(97+i)}) '+mode.capitalize())
  ax.tick_params(axis='y',labelsize=6.7);plt.setp(ax.get_xticklabels(),rotation=55,ha='right',fontsize=6.7)
  for y in range(16):
   for x in range(3):
    r=ratios[mode][y][x];label=f'{r:.2f}' if r<10 else f'{r:.1f}'
    if timeouts[mode][y][x]:label+='†'
    ax.text(x,y,label,ha='center',va='center',fontsize=6.5,color='white' if abs(matrices[mode][y,x])>.6*bound else '#222222')
  for spine in ax.spines.values():spine.set_visible(False)
 fig.colorbar(im,ax=axes[2],fraction=.06,pad=.04).set_label(r'$\log_2(t_{arm}/t_{full})$',fontsize=7)
 fig.tight_layout(w_pad=.8)
 for ext in ['pdf','png']:fig.savefig(out/f'build_ablation.{ext}',dpi=200,bbox_inches='tight',pad_inches=.045)
 plt.close(fig)
 side=out/'plot_data.json';store=json.loads(side.read_text()) if side.exists() else {}
 store['build_ablation']=dict(arms=arms,designs=DESIGNS,runtime_ratio=ratios,aggregate_runtime_ratio=aggregate,timeout_comparisons=timeouts,note='Ratios use optimizer elapsed budget; dagger denotes at least one incomplete run. A ratio of one for two timeouts is not evidence of equal completed work.',verified_profiles_sha256=sha(root/'verified_profiles.json'),pdf_sha256=sha(out/'build_ablation.pdf'))
 for provenance in ['_provenance','_profile_provenance']:
  if provenance in store:store[provenance]['vector_pdf_sha256']={p.name:sha(p) for p in sorted(out.glob('*.pdf'))}
 side.write_text(json.dumps(store,indent=2)+'\n')
 print('Wrote paired construction-control runtime heatmaps')
if __name__=='__main__':main()
