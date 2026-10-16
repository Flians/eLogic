#!/usr/bin/env python3
"""Publication plots with independent cold/warm time denominators and provenance."""
import argparse,csv,hashlib,json,os,statistics
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/epmapping-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_three_parts import DESIGNS

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();root=a.root.resolve();out=a.out;out.mkdir(parents=True,exist_ok=True)
    v=json.loads((root/'verified_profiles.json').read_text())
    for m,h in v['cohort_certificate_sha256'].items():assert sha(root/m/'verified_metrics.json')==h
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'mathtext.fontset':'stix','axes.spines.top':False,'axes.spines.right':False})
    emitted={}
    def save(fig,name,data):
        fig.savefig(out/f'{name}.pdf',bbox_inches='tight',pad_inches=.045);fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight',pad_inches=.045);plt.close(fig);emitted[name]=data
    keys=['secs_stage1','secs_stage2','secs_synth','secs_egg','secs_care','secs_chain'];labels=['L1','L2','L3','L4','Care','Chain'];colors=['#24628d','#70a6be','#bd6b2c','#8c6da5','#86917a','#508b88']
    fig,axes=plt.subplots(1,2,figsize=(3.55,2.4));shares={}
    for ax,m in zip(axes,['cold','warm']):
        time=v['cohort_metrics'][m]['cut_oracle_times'];values=[time[k] for k in keys];assert time['secs_window']==0
        wedges,_=ax.pie(values,startangle=90,counterclock=False,colors=colors,wedgeprops=dict(width=.4,edgecolor='white',linewidth=.6))
        ax.text(0,0,f'{sum(values):.0f} s',ha='center',va='center',fontsize=8);ax.set_title(m.capitalize(),fontsize=8);ax.set_aspect('equal');shares[m]=dict(zip(keys,values))
    fig.legend(wedges,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.02));fig.tight_layout(rect=(0,.19,1,1))
    save(fig,'oracle_time_share',{'cohort_stage_seconds':shares,'completed_full_runs':{m:v['cohort_metrics'][m]['phase_timed_full_runs'] for m in ['cold','warm']},'denominator':'sum of listed phases from completed full-ablation runs only; interrupted phase counters are unavailable; excludes CEC and unmeasured phases'})
    runs=v['cache_runs'];details={}
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.75),sharex=True)
    for ax,stage,arm,title in zip(axes,['tab_mfs3','tab_main'],['ours','ae'],['Common-input one-pass','After AIG preprocessing, cap 6']):
        data={m:{r['design']:r['optimizer_seconds'] for r in runs if r['mode']==m and r['stage']==stage and r['arm']==arm} for m in ['cold','warm']};details[stage]=data
        for m,color,marker in [('cold','#24628d','o'),('warm','#bd6b2c','s')]:
            ax.plot(range(len(DESIGNS)),[data[m][d] for d in DESIGNS],color=color,marker=marker,markersize=3,linewidth=1.1,label=m.capitalize())
            timed_out={r['design'] for r in runs if r['mode']==m and r['stage']==stage and r['arm']==arm and r.get('status')=='timeout'}
            if timed_out:ax.scatter([DESIGNS.index(d) for d in timed_out],[1000]*len(timed_out),marker='x',color=color,s=28,zorder=5)
        ax.set(title=title,ylabel='Optimizer time (s)',xticks=range(len(DESIGNS)),xticklabels=DESIGNS);ax.tick_params(axis='x',rotation=65,labelsize=6);ax.grid(axis='y',color='#dddddd',linewidth=.5);ax.legend(frameon=False)
    fig.tight_layout();save(fig,'cache_runtime',{'designs':DESIGNS,'seconds':details,'timeouts':[{k:r[k] for k in ['mode','stage','arm','design']} for r in runs if r.get('status')=='timeout' and (r['stage'],r['arm']) in [('tab_mfs3','ours'),('tab_main','ae')]],'note':'Crosses at 1000 s denote optimizer timeouts, not completion times. Lines join distinct circuits to aid reading; no continuous workload interpolation is implied.'})
    # Preserve numerical sidecars for the pre-existing QoR plots, replacing only the timing plot.
    old=out/'plot_data.json';merged=json.loads(old.read_text()) if old.exists() else {};merged.update(emitted)
    if '_provenance' in merged:merged['_provenance']['vector_pdf_sha256']={p.name:sha(p) for p in sorted(out.glob('*.pdf'))}
    merged['_profile_provenance']=dict(verified_profiles_sha256=sha(root/'verified_profiles.json'),cohort_certificate_sha256=v['cohort_certificate_sha256'],vector_pdf_sha256={p.name:sha(p) for p in sorted(out.glob('*.pdf'))})
    old.write_text(json.dumps(merged,indent=2)+'\n')
    print('Wrote cold/warm time-share and runtime plots')
if __name__=='__main__':main()
