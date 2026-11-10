"""Resume only a daemon-interrupted, unrecorded attempt into a new directory."""
import copy,fcntl,json,pathlib,sys,time
import run_v2 as m
from observations import read

def main():
 locks=[]
 for p in [m.R/'results_v2/independent_components_20261019/global.lock',m.O/'runner.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 c=m.read(m.O/'campaign.json');assert c['frozen']==m.fingerprint()
 key='sin_r1_witness';assert key not in c['jobs'];assert m.read(m.O/'state.json')['active']==key
 old=m.O/'cases'/key;report=m.read(old/'report.json');assert report['status']=='running'
 assert time.time()-(old/'observation.jsonl').stat().st_mtime>60,'possible surviving worker; do not overlap'
 for r in c['jobs'].values():
  assert r['status']=='ok'
  for n,h in r['artifacts'].items():assert m.sha(pathlib.Path(r['case'])/n)==h
 events,tail=read(old/'observation.jsonl');ad=next(x for x in reversed(events)if x.get('details',{}).get('AD'))
 audit=dict(original_case=str(old),original_key=key,reason='daemon restart, original exec session unavailable, global and runner locks idle, no native process found; log stale over60s',status='interrupted_infrastructure',last_complete_observation_elapsed=ad['elapsed_seconds'],last_observed_AD=ad['details']['AD'],CEC=None,not_a_full_budget_result=True,tail=tail,artifacts={str(p.relative_to(old)):m.sha(p)for p in sorted(old.rglob('*'))if p.is_file()},retry_key=key+'_restart1',same_native_binary_and_configuration=True,new_independent_attempt_budget_seconds=600,old_attempt_retained=True,not_a_search_continuation=True)
 m.save(m.O/'interrupted-state.json',m.read(m.O/'state.json'));m.save(m.O/'interrupted-campaign.json',c);m.save(m.O/'interruption-audit.json',audit)
 c['interruptions']=[dict(audit_file=str(m.O/'interruption-audit.json'),sha256=m.sha(m.O/'interruption-audit.json'),key=key,status='interrupted_infrastructure')];m.save(m.O/'campaign.json',c)
 job=copy.deepcopy(next(x for x in c['plan']if x['key']==key));job['key']=audit['retry_key'];m.state(c,job['key']);print('start independent retry',job['key'],flush=True)
 r=m.execute(c,job,locks);assert c['frozen']==m.fingerprint();r.update(attempt_key=job['key'],key=key,interrupted_attempt=str(old),interruption_audit=str(m.O/'interruption-audit.json'));c['jobs'][key]=r;m.save(m.O/'campaign.json',c);m.state(c);print('end retry',r['status'],r.get('T'),r.get('A'),r.get('D'),r.get('audit_error'),flush=True)
 return 0 if r['status']=='ok'else 1
if __name__=='__main__':raise SystemExit(main())
