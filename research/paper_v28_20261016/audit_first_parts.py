"""Audit the completed first two parts while independent ablations continue."""
import hashlib,json,statistics,sys
from pathlib import Path
project=Path(__file__).resolve().parents[2];sys.path.insert(0,str(project/'scripts/experiments'))
from audit_mapped_ppa import audit
from report import read_counters
root=project/'results_v28_three_parts_query_budget';ref=project/'results_v20_three_parts_chain';records=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for mode in ['cold','warm']:
 for stage in ['tab_mfs3','tab_main']:
  for f in sorted((root/mode/stage).glob('*/row.tsv')):
   row=f.read_text().split();arm,d=row[:2]
   if arm=='base':continue
   wd=f.parent;assert row[-1]=='0' and '[i] cec=1' in (wd/'log').read_text()
   lib=root/mode/'inputs'/('asap7_6in.genlib' if stage=='tab_mfs3' else 'asap7.genlib')
   before=audit(wd/'epmapping_baseline.v',lib);after=audit(wd/'epmapping_mapped.v',lib)
   assert all(after[k]<=before[k]+.001 for k in ['area','delay','serialized_area','serialized_delay'])
   old=ref/('tab_mfs3/ours.'+d if stage=='tab_mfs3' else 'tab_main/ae.'+d)/'epmapping_baseline.v'
   assert sha(old)==sha(wd/'epmapping_baseline.v')
   seconds=read_counters(str(wd/'log'))['time_optimizer'];assert seconds<=1000.01
   records.append(dict(mode=mode,stage=stage,arm=arm,design=d,initial=before,final=after,optimizer_seconds=seconds,log_sha256=sha(wd/'log'),initial_sha256=sha(old),final_sha256=sha(wd/'epmapping_mapped.v')))
assert len(records)==96
summaries={}
for mode in ['cold','warm']:
 for stage,arm in [('tab_mfs3','ours'),('tab_main','ae.r1'),('tab_main','ae')]:
  rr=[r for r in records if r['mode']==mode and r['stage']==stage and r['arm']==arm];assert len(rr)==16
  summaries[f'{mode}/{stage}/{arm}']=dict(completed=16,timeouts=0,mean_area_change_percent=100*(statistics.mean(r['final']['area']/r['initial']['area'] for r in rr)-1),mean_delay_change_percent=100*(statistics.mean(r['final']['delay']/r['initial']['delay'] for r in rr)-1),mean_seconds=statistics.mean(r['optimizer_seconds'] for r in rr),max_seconds=max(r['optimizer_seconds'] for r in rr))
out=dict(verified=96,summaries=summaries,records=records)
Path(__file__).with_name('first_two_parts_verified.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summaries,indent=2))
