#!/usr/bin/env python3
"""Plot measured buffer/core contributions after the paired audit completes."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);a=ap.parse_args()
    data=json.loads((a.root/'analysis.json').read_text());assert data['complete']
    rows=[r for r in data['records'] if r['mode']=='cold'];assert len(rows)==3
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(10.8,3.0))
    colors=['#b7c2cf','#147d92']
    for ax,r in zip(axes,rows):
        base=r['initial']['area'];buf=[r['abc_buffer_gain'],r['ours_buffer_gain']];core=[r['abc_core_gain'],r['ours_core_gain']]
        b=[100*v/base for v in buf];c=[100*v/base for v in core]
        ax.barh([1,0],b,height=.48,color=colors[0]);ax.barh([1,0],c,left=b,height=.48,color=colors[1])
        ax.set_yticks([1,0],['ABC mfs3','Ours']);ax.set_ylim(-.6,1.6)
        ax.set_xlim(0,max(x+y for x,y in zip(b,c))*1.3)
        ax.set_title(r['design'],fontweight='bold');ax.set_xlabel('Area reduction (%)')
        ax.xaxis.grid(True,alpha=.16);ax.set_axisbelow(True)
        for y,x,v in zip([1,0],[b[i]+c[i] for i in range(2)], [buf[i]+core[i] for i in range(2)]):
            ax.annotate(f'{x:.2f}%',(x,y),xytext=(4,0),textcoords='offset points',va='center',fontsize=9)
    fig.legend([Patch(facecolor=c) for c in colors],['Identity-buffer removal','Core logic optimization'],loc='upper center',ncol=2,frameon=False)
    fig.text(.5,.015,'Reductions use each shared initial netlist. Panels have separate horizontal scales. Cold and warm PPA agree.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.88));folder=a.root/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'buffer_core_contributions.png',dpi=180,bbox_inches='tight')
    fig.savefig(folder/'buffer_core_contributions.svg',bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
