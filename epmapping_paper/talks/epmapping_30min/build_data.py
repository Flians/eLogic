#!/usr/bin/env python3
"""Export editable LaTeX tables from the admitted experiment records."""
import csv,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUN=ROOT/'results_v28_three_parts_query_budget'
v=json.loads((RUN/'verified_profiles.json').read_text())
a=json.loads((RUN/'query_budget_analysis.json').read_text())
rows=list(csv.DictReader((RUN/'paired_results.tsv').open(),delimiter='\t'))
c=v['cohort_metrics']['cold'];w=v['cohort_metrics']['warm']
def esc(x):return x.replace('_',r'\_')
def macro(k,val):return chr(92)+'newcommand{'+chr(92)+k+'}{'+str(val)+'}\n'
s='% Generated from verified_profiles.json and paired_results.tsv.\n'
for key,val in dict(MainArea=f"{-c['main_own_area']:.2f}",MainDelay=f"{-c['main_own_delay']:.2f}",OneArea=f"{-c['main_one_area']:.2f}",OneDelay=f"{-c['main_one_delay']:.2f}",OtherArea=f"{-c['main_other_area']:.2f}",OtherDelay=f"{-c['main_other_delay']:.2f}",ColdMax=f"{a['primary_completion']['cold']['max_seconds']:.1f}",WarmMax=f"{a['primary_completion']['warm']['max_seconds']:.1f}",RawRatio=f"{a['paired']['cold']['mean_area_ratio_to_raw_mfs3']:.4f}",OursCold=f"{a['paired']['cold']['ours_mean_seconds']:.1f}",OursWarm=f"{a['paired']['warm']['ours_mean_seconds']:.1f}",AbcMean=f"{a['paired']['cold']['mfs3_mean_seconds']:.1f}").items():s+=macro(key,val)
# Figures are native PGFPlots coordinates and remain editable in the source.
s+=macro('AreaCoordinates',f"(0,0) (1,{-c['main_one_area']:.6f}) (6,{-c['main_own_area']:.6f})")
s+=macro('DelayCoordinates',f"(0,0) (1,{-c['main_one_delay']:.6f}) (6,{-c['main_own_delay']:.6f})")
def tablemacro(name,lines):
 global s
 s+=chr(92)+'newcommand{'+chr(92)+name+'}{%\n'+'\n'.join(lines)+'\n}\n'
for idx,part in enumerate([['aes_core','arbiter','div','ethernet','iwls05_i2c','mem_ctrl','RISC','sasc'],['simple_spi','spi','sqrt','systemcaes','systemcdes','tv80','usb_funct','usb_phy']],1):
 ll=[]
 for d in part:
  r=next(x for x in rows if x['mode']=='cold' and x['part']=='tab_main' and x['arm']=='ae' and x['design']==d)
  cold=next(x for x in v['cache_runs'] if x['mode']=='cold' and x['stage']=='tab_main' and x['arm']=='ae' and x['design']==d)
  hot=next(x for x in v['cache_runs'] if x['mode']=='warm' and x['stage']=='tab_main' and x['arm']=='ae' and x['design']==d)
  ll.append(f"{esc(d)} & {float(r['initial_area']):.2f} & {float(r['area']):.2f} & {float(r['initial_delay']):.2f} & {float(r['delay']):.2f} & {cold['optimizer_seconds']:.1f}/{hot['optimizer_seconds']:.1f}"+r' \\')
 tablemacro('MainRows'+('A' if idx==1 else 'B'),ll)
abc={r['design']:r for r in json.loads((RUN/'cold/tab_mfs3/abc_results.json').read_text())}
designs=list(abc)
for i,ds in enumerate([designs[:8],designs[8:]],1):
 ll=[]
 for d in ds:
  b=abc[d];o=next(x for x in rows if x['mode']=='cold' and x['part']=='tab_mfs3' and x['design']==d)
  vals=['TO','TO','1000'] if b['rc'] else [f"{b['area']/b['baseline']['area']:.4f}",f"{b['delay']/b['baseline']['delay']:.4f}"+(r'$^\dagger$' if b['delay']>b['baseline']['delay']+.001 else ''),f"{b['secs']:.1f}"]
  times=[next(x['optimizer_seconds'] for x in v['cache_runs'] if x['mode']==m and x['stage']=='tab_mfs3' and x['design']==d) for m in ['cold','warm']]
  ll.append(' & '.join([esc(d),*vals,f"{float(o['area'])/float(o['initial_area']):.4f}",f"{float(o['delay'])/float(o['initial_delay']):.4f}",f'{times[0]:.1f}/{times[1]:.1f}'])+r' \\')
 tablemacro('AbcRows'+('A' if i==1 else 'B'),ll)
arms=list(c['ablation'])
for i,aa in enumerate([arms[:10],arms[10:]],1):
 ll=[]
 for arm in aa:
  x,y=c['ablation'][arm],w['ablation'][arm]
  tos=[sum(r['status']=='timeout' for r in v['cache_runs'] if r['mode']==m and r['stage']=='ablation' and r['arm']==arm) for m in ['cold','warm']]
  ll.append(f"{esc(arm)} & {x['area']:+.2f}/{y['area']:+.2f} & {x['time_ratio']:.2f}/{y['time_ratio']:.2f} & {tos[0]}/{tos[1]}"+r' \\')
 tablemacro('AblationRows'+('A' if i==1 else 'B'),ll)
for name,aa in [('CoreRows',['noSYN','noEGG','noCARE','noCHAIN','noPHASE','noSHANNON','noDAVIO']),('ReuseRows',['noQUERY','noTRUTH','allPIVOTS'])]:
 ll=[]
 for arm in aa:
  x,y=c['ablation'][arm],w['ablation'][arm]
  tos=[sum(r['status']=='timeout' for r in v['cache_runs'] if r['mode']==m and r['stage']=='ablation' and r['arm']==arm) for m in ['cold','warm']]
  ll.append(f"{arm} & {x['area']:+.2f}/{y['area']:+.2f} & {x['time_ratio']:.2f}/{y['time_ratio']:.2f} & {tos[0]}/{tos[1]}"+r' \\')
 tablemacro(name,ll)
(HERE/'assets/data.tex').write_text(s)
proof=dict(campaign=RUN.name,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'verified_profiles.json',RUN/'paired_results.tsv',RUN/'query_budget_analysis.json']},constraints=json.loads((RUN/'delivery_manifest.json').read_text())['constraints'])
(HERE/'assets/provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
print('Exported verified data for the lecture.')
