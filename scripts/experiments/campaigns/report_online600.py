"""Read-only evidence for versioned online mapper cohorts; missing designs remain missing."""
import argparse,hashlib,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path):
 s=json.loads(path.read_text())
 for f,h in s['fingerprint'].items():assert sha(f)==h,('frozen dependency drift',f)
 rows={}
 for j in s['jobs']:
  for f,h in j['sha256'].items():assert sha(f)==h,('artifact drift',f)
  r=json.loads(Path(j['report']).read_text());rows[j['bench']]=r
  if r.get('status')=='ok':
   assert r['measured']['cec']==1 and r['native']['optimizer_succeeded']
   x=r['native']['incumbent'];assert x['observed_monotonic']<x['deadline_monotonic']==r['mapping']['deadline_monotonic']
 return s,rows
def main():
 a=argparse.ArgumentParser();a.add_argument('--study',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=a.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 states={};rows={}
 for kind in ['epfl_v1','epfl_v2_control','epfl_v2','epfl_v2_repeat1']:
  states[kind],rows[kind]=load(a.study/kind/'campaign.json')
 profile=states['epfl_v2']['configuration'];control=states['epfl_v2_control']['configuration'];assert {k:v for k,v in profile.items()if k!='area_recovery'}=={k:v for k,v in control.items()if k!='area_recovery'}
 old=json.loads((ROOT/'results_v2/time_only_600_20261022/epfl_checkpoint/campaign.json').read_text());prior={j['bench']:json.loads(Path(j['report']).read_text())for j in old['jobs']}
 pairs=[]
 for bench in profile['designs']:
  p={'bench':bench}
  for name,r in rows.items():
   if bench not in r:p[name]='missing';continue
   r=r[bench];p[name]=dict(status=r['status'],A=r.get('measured',{}).get('area'),D=r.get('measured',{}).get('delay'),T=r['mapping']['wall_secs'],cec=r.get('measured',{}).get('cec'),search_timeout=r['mapping']['timeout'])
  if isinstance(p['epfl_v2'],dict)and isinstance(p['epfl_v2_control'],dict):
   c=p['epfl_v2_control'];v=p['epfl_v2'];p.update(area_change=v['A']-c['A'],delay_change=v['D']-c['D'])
   oldrow=prior[bench];p['old_complete_flow']=oldrow.get('measured');p['area_vs_old']=v['A']-oldrow['measured']['area'];p['delay_vs_old']=v['D']-oldrow['measured']['delay']
   repeat=p['epfl_v2_repeat1'];p['repeat_identical_AD']=isinstance(repeat,dict)and(v['A'],v['D'])==(repeat['A'],repeat['D'])
  pairs.append(p)
 result=dict(stage='isolated v2 selected-input exact shared-area prototype, not main adoption',rows=pairs,completed={k:len(v)for k,v in rows.items()},expected_full20=len(profile['designs']),main_merged=False,baseline_algorithms_changed=False,full_EPFL_and_full_repeated_validation_complete=False,limitations=['All matches still physically materialized per delta; late materialization not implemented','Area sweep only after cut closure, only already selected input cones','Same-binary first2 ablation + one candidate repeat is not full-suite repetition','Old full-flow area regressions remain; no adoption'],source_manifests={k:str(a.study/k/'campaign.json')for k in states})
 (a.output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
 text=['# Online mapper v2: limited-scope verification','','600s total mapping search; independent CEC excluded; original baselines unchanged. Full20 coverage pending.','','|Design|Control A|Area recovery A|D (both)|T control / recovery|Old-flow A|Repeat AD|','|---|---:|---:|---:|---:|---:|---|']
 for p in pairs:
  if not isinstance(p['epfl_v2'],dict):continue
  c=p['epfl_v2_control'];v=p['epfl_v2'];o=p['old_complete_flow'];text.append(f"|{p['bench']}|{c['A']:.2f}|{v['A']:.2f}|{v['D']:.2f}|{c['T']:.2f} / {v['T']:.2f}|{o['area']:.2f}|{p['repeat_identical_AD']}|")
 text+=['','All shown outputs pass independent named-port CEC. V2 still loses area against old complete recovery. This does not establish baseline dominance or justify main adoption. Late materialization and full shared-cover recovery remain incomplete.']
 (a.output/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(result['completed']))
if __name__=='__main__':main()
