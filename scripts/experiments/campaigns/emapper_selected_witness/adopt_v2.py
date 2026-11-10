"""Explicit telemetry-reader repair; preserve original invalid checkpoint verbatim."""
import copy,fcntl,importlib.util,json,pathlib,sys
D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(D))
import run_v2 as m
from observations import read

def main():
 locks=[]
 for p in [m.R/'results_v2/independent_components_20261019/global.lock',m.R/'results_v2/emapper_selected_witness_20261111/benchmark/runner.lock',m.O/'runner.lock']:
  p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 oldpath=m.R/'results_v2/emapper_selected_witness_20261111/benchmark/campaign.json';old=m.read(oldpath);assert m.read(oldpath.parent/'state.json')['active']is None
 assert not(m.O/'campaign.json').exists();before=old['frozen'];after=m.fingerprint();old_driver=str((D/'run.py').relative_to(m.R));new_driver=str((D/'run_v2.py').relative_to(m.R));helper=str((D/'observations.py').relative_to(m.R))
 assert set(before)-set(after)=={old_driver}and set(after)-set(before)=={new_driver,helper}
 assert all(before[k]==after[k]for k in set(before)&set(after));assert m.sha(D/'run.py')==before[old_driver]
 changes=[];c=copy.deepcopy(old)
 for key,r in c['jobs'].items():
  for n,h in r['artifacts'].items():assert m.sha(pathlib.Path(r['case'])/n)==h
  report=m.read(pathlib.Path(r['case'])/'report.json');assert report['status']=='ok'and report['cec']==1 and report['evaluation']['exact_agreement']and report['selected_evidence']['valid']and report['selected_evidence']['netlist_binding_agreement']
  assert r['exit']==0 and(r['A'],r['D'])==(report['measured']['area'],report['measured']['delay'])
  events,tail=read(pathlib.Path(r['case'])/'observation.jsonl')
  if r['status']=='harness_invalid':
   assert tail is not None and r['audit_error'].startswith('JSONDecodeError(');r['original_harness_status']=r['status'];r['original_harness_error']=r.pop('audit_error');r['status']='ok'
  r['observation_tail_fragment']=tail
  if events:r['exclusive_seconds']=events[-1].get('exclusive_seconds')
  for x in reversed(events):
   if x.get('details',{}).get('trial_diagnostics'):
    r.update(prefix_trial_diagnostics=x['details']['trial_diagnostics'],prefix_elapsed_seconds=x.get('elapsed_seconds'),counter_scope='complete observation prefix; final fragment retained separately, never invented totals');break
  changes.append(dict(key=key,original_status=old['jobs'][key]['status'],adopted_status=r['status'],tail=tail,case=r['case']))
 c['frozen']=after;c['adoption']=dict(source=str(oldpath),source_sha256=m.sha(oldpath),changes=changes,scope='same native results, all artifact hashes checked; only telemetry reader changed; no rerun or native algorithm change',old_driver_sha256=before[old_driver],new_driver_sha256=after[new_driver],helper_sha256=after[helper]);m.save(m.O/'adoption.json',c['adoption']);m.save(m.O/'original-campaign.json',old);m.save(m.O/'campaign.json',c);m.state(c)
 print(json.dumps(c['adoption'],indent=2))
if __name__=='__main__':main()
