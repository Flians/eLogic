#!/usr/bin/env python3
"""Original baseline continuation after an unused candidate CLI/preset change."""
import argparse,copy,fcntl,json,sys
from pathlib import Path
import run_epfl_mapping_campaign as policy
from run_epfl_baselines import sha

UNUSED={'plot_paper.py','run_three_parts.py','report_three_parts.py'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--subjects',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--adopt-campaign',type=Path);ap.add_argument('--max-jobs',type=int,default=24);a=ap.parse_args()
 original_jobs,original_f=policy.jobs,policy.fingerprint;policy.jobs=lambda:(j for j in original_jobs() if '_egraph_' not in j[0])
 def fingerprint(subjects):
  f=original_f(subjects);f['files']={p:h for p,h in f['files'].items() if p not in {str(policy.ROOT/'scripts/experiments'/n) for n in UNUSED}}
  for name in ['run_epfl_original_baselines.py','run_epfl_original_dependencies.py','run_epfl_original_cecfile.py',Path(__file__).name]:
   p=Path(__file__).with_name(name).resolve();f['files'][str(p)]=sha(p)
  return f
 policy.fingerprint=fingerprint;out=a.output.resolve();out.mkdir(exist_ok=True);sp=out/'campaign.json'
 if a.adopt_campaign and not sp.exists():
  oldpath=a.adopt_campaign.resolve();lock=(oldpath.parent/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  old=json.loads(oldpath.read_text());assert not old.get('current_job') and old['mapping_seconds']==300 and not old['baseline_algorithms_changed']
  f=fingerprint(a.subjects.resolve());assert old['fingerprint']['native']==f['native']
  allowed={str(policy.ROOT/'target/release/bench_rs_egraph_map'),str(policy.ROOT/'src/bin/bench_rs_egraph_map.rs')};same={};changed={}
  before=policy.ROOT/'results_v2/queue_profile_main_integration_20261013'
  assert old['fingerprint']['files'][str(policy.ROOT/'target/release/bench_rs_egraph_map')]==sha(before/'bench_rs_egraph_map.before')
  assert old['fingerprint']['files'][str(policy.ROOT/'src/bin/bench_rs_egraph_map.rs')]==sha(before/'cli.before.rs')
  for p,h in old['fingerprint']['files'].items():
   current=sha(Path(p))
   if h!=current:
    assert p in allowed,('executed dependency changed',p);changed[p]={'old':h,'new':current,'reason':'unused candidate mapper executable/CLI; no baseline job invokes this program'}
   else:same[p]=h
  for j in old['jobs']:
   for key in ['report','audit']:
    if key in j:assert sha(Path(j[key]))==j[key+'_sha256']
   if 'audit' in j:
    audit=json.loads(Path(j['audit']).read_text())
    if 'states' not in audit:
     assert audit.get('no_mapping_commands_proven') and audit.get('mapping_wall_secs')==0 and audit.get('status')=='failed'
     assert sha(Path(audit['source']))==audit['source_sha256']
    for state in audit.get('states',{}).values():
     for cmd in state['commands']:assert Path(cmd['argv'][0]).name!='bench_rs_egraph_map'
  for p in [Path(policy.__file__).resolve(),Path(__file__).with_name('run_with_mapping_budget.py'),Path(__file__).with_name('nested_mapping_clock.py'),*sorted((policy.ROOT/'emapper/src').rglob('*.rs')),policy.ROOT/'src/bin/bench_rs_egraph_map.rs']:f['files'][str(p)]=sha(p)
  proof={'prior_campaign':str(oldpath),'prior_campaign_sha256':sha(oldpath),'identical_executed_dependencies':same,'unused_candidate_changes':changed,'native_fingerprint_equal':True,'statuses_and_observations_preserved':True,'baseline_algorithms_changed':False,'clock_or_verifier_changed':False}
  pp=out/'adoption.json';pp.write_text(json.dumps(proof,indent=2));new=copy.deepcopy(old);new.update(fingerprint=f,adoption_report=str(pp),adoption_sha256=sha(pp))
  for j in new['jobs']:j['adopted_unused_candidate_campaign']=str(oldpath)
  sp.write_text(json.dumps(new,indent=2));lock.close()
 sys.argv=[policy.__file__,'--subjects',str(a.subjects.resolve()),'--output',str(out),'--max-jobs',str(a.max_jobs)];policy.main()
if __name__=='__main__':main()
