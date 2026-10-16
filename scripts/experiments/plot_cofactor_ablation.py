#!/usr/bin/env python3
"""Paired cold/warm operator effects and selected-outcome round curves."""
import argparse,csv,hashlib,json,os,statistics
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
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();root=a.root;out=a.out
    v=json.loads((root/'verified_profiles.json').read_text())
    for mode,h in v['cohort_certificate_sha256'].items():assert sha(root/mode/'verified_metrics.json')==h
    arms=['noSHANNON','noDAVIO','noREUSE','noPRICE'];labels=['Shannon','Davio','INV reuse','Pricing']
    if not all(x in v['cohort_metrics']['cold']['ablation'] for x in arms):return
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'mathtext.fontset':'stix','axes.spines.top':False,'axes.spines.right':False})
    data={m:{(r['stage'],r['design']):r for r in read_rows(root/m,'ablation')} for m in ['cold','warm']}
    matrices={m:np.array([[100*(float(data[m][arm,d]['out_a'])/float(data[m]['full',d]['out_a'])-1) for arm in arms] for d in DESIGNS]) for m in data}
    interrupted={m:[[any(data[m][x,d]['raw_status']=='timeout' for x in [arm,'full']) for arm in arms] for d in DESIGNS] for m in data}
    totals={m:[round(sum(float(data[m][arm,d]['out_a'])-float(data[m]['full',d]['out_a']) for d in DESIGNS),6) for arm in arms] for m in data}
    fig,axes=plt.subplots(1,3,figsize=(7.15,3.0),gridspec_kw={'width_ratios':[1.1,1.2,1.0]})
    palette=['#24628d','#bd6b2c'];y=np.arange(4)
    for mode,offset,color in zip(['cold','warm'],[-.16,.16],palette):
        vals=totals[mode];axes[0].barh(y+offset,vals,height=.28,color=color,label=mode.capitalize())
        for yy,x in zip(y+offset,vals):axes[0].annotate(f'{x:+.2f}',(x,yy),xytext=(2 if x>=0 else -2,0),textcoords='offset points',ha='left' if x>=0 else 'right',va='center',fontsize=6.5)
    low=min([0]+totals['cold']+totals['warm']);high=max([0]+totals['cold']+totals['warm']);span=max(high-low,1.)
    axes[0].set(yticks=y,yticklabels=labels,xlim=(low-.27*span,high+.35*span),xlabel=r'Area difference ($\mu m^2$)',title='(a) Summed effect');axes[0].invert_yaxis();axes[0].grid(axis='x',linewidth=.5,color='#dddddd');axes[0].legend(frameon=False,fontsize=6.5,loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2)
    bound=max(.001,*[abs(x) for mat in matrices.values() for x in mat.ravel()])
    for i,mode in enumerate(['cold','warm'],1):
        ax=axes[i];mat=matrices[mode];im=ax.imshow(mat,cmap='RdBu_r',norm=TwoSlopeNorm(0,-bound,bound),aspect='auto')
        ax.set(xticks=range(4),xticklabels=labels,yticks=range(16),yticklabels=DESIGNS if i==1 else [],title=f'({"b" if i==1 else "c"}) {mode.capitalize()} (%)')
        plt.setp(ax.get_xticklabels(),rotation=55,ha='right',fontsize=6.5)
        for yy in range(16):
            for xx in range(4):
                z=mat[yy,xx];text='' if abs(z)<.005 else (f'{z:+.0f}' if abs(z)>=100 else f'{z:+.2f}')
                if interrupted[mode][yy][xx]:text+='†'
                ax.text(xx,yy,text,ha='center',va='center',fontsize=6.5,color='white' if abs(z)>bound*.6 else '#222222')
    fig.colorbar(im,ax=axes[2],fraction=.06,pad=.04).ax.tick_params(labelsize=6)
    fig.tight_layout(w_pad=.65)
    def save(fig,name):fig.savefig(out/f'{name}.pdf',bbox_inches='tight',pad_inches=.045);fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight',pad_inches=.045);plt.close(fig)
    save(fig,'ablation_effects')
    profile={m:{(r['stage'],r['design']):r for r in read_rows(root/m,'tab_main')} for m in data};progress={}
    for mode in data:
        progress[mode]={k:[0.]+[-statistics.mean(100*(float(profile[mode][arm,d]['out_'+k])/float(profile[mode][arm,d]['base_'+k])-1) for d in DESIGNS) for arm in ['ae.r1','ae']] for k in ['a','d']}
    equal=progress['cold']==progress['warm'];fig,ax=plt.subplots(figsize=(3.4,2.25))
    for k,name,color,marker in [('a','Area','#24628d','o'),('d','Delay','#bd6b2c','s')]:
        for mode in (['cold'] if equal else ['cold','warm']):
            vals=progress[mode][k];ax.plot([0,1,6],vals,color=color,marker=marker,markersize=4,linewidth=1.3,linestyle='-' if mode=='cold' else '--',markerfacecolor=color if mode=='cold' else 'white',label=name if equal else name+' '+('C' if mode=='cold' else 'W'))
        if equal:ax.annotate(f'{vals[-1]:.2f}%',(6,vals[-1]),xytext=(-4,7),textcoords='offset points',ha='right',fontsize=7,color=color)
    ax.set(xlabel='Initial mapping / configured round cap',ylabel='Mean reduction (%)\nfrom own initial mapping',xticks=[0,1,6],xticklabels=['Initial','1','6'],xlim=(-.15,6.25));ax.grid(axis='y',color='#dddddd',linewidth=.5);ax.legend(frameon=False,fontsize=6.5,ncol=1 if equal else 2,loc='lower right');fig.tight_layout();save(fig,'round_progress')
    side=out/'plot_data.json';stored=json.loads(side.read_text());stored['ablation_effects']=dict(designs=DESIGNS,arms=arms,cohort_summed_area_delta=totals,cohort_per_design_percent={m:x.tolist() for m,x in matrices.items()},cohort_timeout_comparisons=interrupted,timeout_policy='independently verified saved output on optimizer deadline under the campaign selection policy; dagger marks either compared run interrupted');stored['round_progress']=dict(cohort_percent_reduction=progress,identical=equal);stored['_cofactor_provenance']=dict(verified_profiles_sha256=sha(root/'verified_profiles.json'));side.write_text(json.dumps(stored,indent=2)+'\n')
    print('Wrote paired operator heatmaps and selected-outcome progression')
if __name__=='__main__':main()
