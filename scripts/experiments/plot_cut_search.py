#!/usr/bin/env python3
"""Plot audited fixed-profile results; never select a profile per benchmark."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument('--root',type=Path,required=True)
ap.add_argument('--profiles',default='control,gain,required')
ap.add_argument('--compare-rounds',action='store_true')
args=ap.parse_args()
data=json.loads((args.root/'verified_search.json').read_text())
profiles=args.profiles.split(',')
names={'control':'Strict control','gain':'Net area gain','required':'Area-first with timing slack','path':'Critical timing / noncritical area'}
colors={'control':'#777777','gain':'#0072B2','required':'#D55E00','path':'#0072B2'}
rows={(r['profile'],r['design']):r for r in data['records'] if r['phase'] in ['tuning','validation','all']}
if args.compare_rounds:
    assert json.loads((args.root/'all/plan.json').read_text())['rounds']==1
    assert json.loads((args.root/'converged/plan.json').read_text())['rounds']==6
    profiles=['bounded_one','bounded_six']
    names.update(bounded_one='Area under delay bound: one round',bounded_six='Area under delay bound: six-round cap')
    colors.update(bounded_one='#777777',bounded_six='#0072B2')
    rows={('bounded_one' if r['phase']=='all' else 'bounded_six',r['design']):r
          for r in data['records'] if r['profile']=='bounded' and r['phase'] in ['all','converged']}
designs=sorted({d for p,d in rows})
assert len(designs)==16 and all((p,d) in rows for p in profiles for d in designs)
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(2,1,figsize=(10,5.3),sharex=True)
for ax,metric,out,base in zip(axes,['Area reduction (%)','Delay reduction (%)'],['out_a','out_d'],['base_a','base_d']):
    for p in profiles:
        values=[100*(1-rows[p,d][out]/rows[p,d][base]) for d in designs]
        ax.plot(range(16),values,color=colors[p],marker='o',markersize=3.5,linewidth=1.15,label=names[p])
    ax.axhline(0,color='black',linewidth=.6)
    ax.set_ylabel(metric)
    ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    if out=='out_a' and min(100*(1-rows[p,d][out]/rows[p,d][base]) for p in profiles for d in designs)<-20:
        inset=ax.inset_axes([.025,.17,.51,.46])
        for p in profiles:
            values=[100*(1-rows[p,d][out]/rows[p,d][base]) for d in designs]
            inset.plot(range(16),values,color=colors[p],marker='o',markersize=2,linewidth=.8)
        inset.axhline(0,color='black',linewidth=.5)
        inset.set_ylim(-6,6)
        inset.set_xticks([])
        inset.set_yticks([-5,0,5])
        inset.tick_params(labelsize=7)
        inset.grid(axis='y',alpha=.2)
        inset.set_title('Detail: -6% to +6%; sqrt lies outside this range',fontsize=7)
        for p in profiles:
            for i,d in enumerate(designs):
                value=100*(1-rows[p,d][out]/rows[p,d][base])
                if value<-20:
                    ax.annotate(f'{d}: area +{-value:.2f}%',(i,value),xytext=(12,9),
                                textcoords='offset points',fontsize=8,color=colors[p])
axes[0].legend(ncol=3,loc='upper left',frameon=False)
axes[1].set_xticks(range(16),designs,rotation=40,ha='right',fontsize=8)
fig.suptitle('Cut rewriting only: identical initial mappings, all 16 CEC-verified designs'+
             ('; one round versus six-round cap' if args.compare_rounds else '; one round'),fontsize=10)
fig.tight_layout()
out=args.root/'figures'
out.mkdir(exist_ok=True)
fig.savefig(out/'paired_search.pdf',bbox_inches='tight')
fig.savefig(out/'paired_search.png',dpi=180,bbox_inches='tight')
(out/'plot_data.json').write_text(json.dumps({'profiles':profiles,'designs':designs,
    'rows':[rows[p,d] for p in profiles for d in designs]},indent=2)+'\n')
print(out/'paired_search.pdf')
