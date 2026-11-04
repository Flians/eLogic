#!/usr/bin/env python3
"""Explicit-profile native-fast experiments; no baseline execution or seed shortcut."""
import argparse,csv,fcntl,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'scripts/emapper'))
from provenance import sha,hashes,git_revision
from run_fast import source
PROTOCOL=R/'results_v2/emapper_seeded_epfl_20261101/protocol.json'
def read(p):return json.loads(Path(p).read_text())
def dump(p,j):
 p=Path(p);q=p.with_suffix('.pending');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
def fingerprint(profile):return dict(source=source(),binaries=hashes([R/'target/release/emapper_fast',R/'target/release/emapper_measure']),receipt_sha256=sha(R/'target/release/emapper-fast-build.json'),driver_sha256=sha(Path(__file__)),profile_sha256=sha(profile),protocol_sha256=sha(PROTOCOL))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--designs',default='ctrl,bar,i2c');ap.add_argument('--max-jobs',type=int,default=1000);a=ap.parse_args();profile=a.profile.resolve();config=read(profile);out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);locks=[]
 for p in (R/'results_v2/independent_components_20261019/global.lock',out/'runner.lock'):
  f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 if (out/'campaign.json').exists():c=read(out/'campaign.json');assert c['frozen']==fingerprint(profile)
 else:
  subjects=read(PROTOCOL)['inputs'];names=list(subjects)if a.designs=='all'else a.designs.split(',');assert len(names)==len(set(names))and set(names)<=set(subjects);plan=[]
  for i,n in enumerate(names):
   arms=list(config['arms']);offset=i%len(arms);arms=arms[offset:]+arms[:offset]
   for arm in arms:plan.append(dict(key=n+'_'+arm,design=n,arm=arm))
  c=dict(schema=1,source_commit=git_revision(),frozen=fingerprint(profile),profile=config,profile_path=str(profile),inputs={n:{k:v for k,v in subjects[n].items()if k in ('prepared','prepared_sha256','golden','golden_sha256')}for n in names},designs=names,planned=plan,expected=len(plan),jobs={},baseline_runs=0,seed_cost='no external seed; native own seed included in mapping clock',scope='explicit uniform same-native-binary controls; all failures retained; wide matching not yet implemented; no fullEPFL claim unless all20specified')
 count=0
 for p in c['planned']:
  key=p['key']
  if key in c['jobs']:continue
  if count>=a.max_jobs:break
  assert c['frozen']==fingerprint(profile);inp=c['inputs'][p['design']]
  for k in ('prepared','golden'):assert sha(inp[k])==inp[k+'_sha256']
  case=out/'cases'/key;assert not case.exists(),'audit unrecorded partial, never overwrite';arm=config['arms'][p['arm']];common=config['common'];argv=[sys.executable,str(R/'scripts/emapper/run_fast.py'),inp['prepared'],'--golden',inp['golden'],'--output',str(case),'--cut-frontier',str(common['cut_frontier']),'--area-passes',str(common['area_passes_per_epoch']),'--critical-quotients',arm['critical_quotients'],'--critical-cuts',arm['critical_cuts'],'--mapped-cut-rank',arm.get('mapped_cut_rank','off'),'--shared-area',arm.get('shared_area','off')];dump(out/'campaign.json',c);dump(out/'state.json',dict(status='running',active=key,completed=len(c['jobs']),expected=c['expected']));print(json.dumps(dict(event='start',key=key)),flush=True)
  with(out/(key+'.log')).open('x')as f:r=subprocess.run(argv,cwd=R,stdout=f,stderr=subprocess.STDOUT,pass_fds=tuple(x.fileno()for x in locks))
  j=dict(**p,argv=argv,exit=r.returncode,status='unreported_exit');report=case/'report.json'
  if report.exists():v=read(report);j.update(report=str(report),report_sha256=sha(report),status=v['status'],A=(v.get('measured')or{}).get('area'),D=(v.get('measured')or{}).get('delay'),T=v.get('mapping_wall_seconds'),CEC=v.get('CEC'),error=v.get('error'),native_stats=(v.get('native')or{}).get('stats'))
  assert c['frozen']==fingerprint(profile);c['jobs'][key]=j;dump(out/'campaign.json',c);count+=1;print(json.dumps(dict(event='end',**{k:j.get(k)for k in ('key','status','A','D','T','CEC','error')})),flush=True)
 dump(out/'state.json',dict(status='complete'if len(c['jobs'])==c['expected']else'batch_complete',completed=len(c['jobs']),expected=c['expected']))
 rows=[dict(p,**{k:v for k,v in c['jobs'].get(p['key'],{}).items()if k not in p})for p in c['planned']];keys=list(dict.fromkeys(k for r in rows for k in r))
 with(out/'current.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
