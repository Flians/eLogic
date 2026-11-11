#!/usr/bin/env python3
"""Compare protocols without mixing initial denominators or claiming equal work."""
import argparse,json,statistics,re
from pathlib import Path

def compare(root,control):
 new=json.loads((root/'analysis.json').read_text());old=json.loads((control/'analysis.json').read_text())
 oi={tuple(r[k]for k in ['stage','arm','design','mode']):r for r in old['records']};rows=[]
 for n in new['records']:
  key=tuple(n[k]for k in ['stage','arm','design','mode'])
  if n['arm']!='ours' and not(n['stage']=='ablation' and n['arm']=='full'):continue
  o=oi[key]
  core=lambda r:r['final']['area']-r['final_buffer_area']
  def counters(base):
   log=base/n['stage']/n['arm']/f"{n['mode']}.{n['design']}"/'log'
   text=log.read_text();egg=re.findall(r'egg_calls=(\d+)',text);profile=re.findall(r'egraph_profile (.*)',text)
   return dict(egg_calls=int(egg[-1])if egg else None,profile={k:int(v)for k,v in re.findall(r'(\w+)=(\d+)',profile[-1])}if profile else None)
  rows.append(dict(stage=n['stage'],arm=n['arm'],design=n['design'],mode=n['mode'],old_status=o['status'],new_status=n['status'],old_initial=o['initial'],new_initial=n['initial'],old_final=o['final'],new_final=n['final'],old_core_area=core(o),new_core_area=core(n),core_area_change=core(n)-core(o),total_area_change=n['final']['area']-o['final']['area'],delay_change=n['final']['delay']-o['final']['delay'],old_optimizer_seconds=o['optimizer_seconds'],new_optimizer_seconds=n['optimizer_seconds'],old_counters=counters(control),new_counters=counters(root)))
 summary={}
 for stage in ['main','main_r1','mfs3','strong_common','ablation']:
  for mode in ['cold','warm']:
   group=[r for r in rows if r['stage']==stage and r['mode']==mode]
   summary[stage+'/'+mode]=dict(designs=len(group),core_area_better=sum(r['core_area_change']<-.0001 for r in group),core_area_worse=sum(r['core_area_change']>.0001 for r in group),core_area_equal=sum(abs(r['core_area_change'])<=.0001 for r in group),delay_better=sum(r['delay_change']<-.001 for r in group),delay_worse=sum(r['delay_change']>.001 for r in group),new_post_area_percent=statistics.mean(100*(r['new_final']['area']/r['new_initial']['area']-1)for r in group),old_full_flow_area_percent=statistics.mean(100*(r['old_final']['area']/r['old_initial']['area']-1)for r in group))
 result=dict(complete=True,protocol='separate initial denominators; same original AIG and preprocessing; buffer normalization precedes current search',summary=summary,records=rows)
 (root/'full_flow_comparison.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--control',type=Path,required=True);a=ap.parse_args();x=compare(a.root.resolve(),a.control.resolve());print(json.dumps(x['summary'],indent=2))
