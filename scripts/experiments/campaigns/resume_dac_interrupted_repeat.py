"""Audit an incomplete attempt; restart in a NEW directory without selecting a winner."""
import fcntl,pathlib
import run_dac_convergence as core
def main():
 o=core.DEFAULT;locks=[]
 for p in [core.R/'results_v2/independent_components_20261019/global.lock',o/'runner.lock',o.parent/'continuation.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 c=core.read(o/'campaign.json');assert c['frozen']==core.fingerprint()
 logical='dec_r2_on';actual=logical+'_retry1';original=o/'cases'/logical
 assert logical not in c['jobs'],'already registered: never rerun a completed configuration'
 assert core.read(original/'report.json')['status']=='running'
 assert not(o/'cases'/actual).exists(),'retry partial must be separately audited'
 frames,tail=core.observations(original/'observation.jsonl');last=frames[-1]if frames else{}
 audit=dict(original_case=str(original),logical_key=logical,retry_key=actual,reason='supervisor session absent and all locks idle; separate host process inventory found no owned mapping processes; interruption cause not established',status='interrupted_incomplete',last_observed_seconds=last.get('elapsed_seconds'),last_observed_AD=last.get('details',{}).get('AD'),observation_tail=tail,full_mapping_T=None,CEC=None,eligible_for_PPA=False,original_report_unchanged=True,scope='job restart, not search continuation, no incumbent/pool imported, no best-of-attempt selection',artifacts={str(p.relative_to(original)):core.sha(p)for p in sorted(original.rglob('*'))if p.is_file()})
 path=o.parent/'interruption-dec-r2-on.json';assert not path.exists();core.save(path,audit)
 j=next(j for j in c['plans']['repeats']if j['key']==logical);j=dict(j,key=actual)
 core.save(o/'state.json',dict(active=actual,logical_key=logical,completed=len(c['jobs']),phase='repeats',baseline_runs=0,default_promoted=False));print('start',actual,flush=True)
 row=core.execute(c,j,o,locks)
 row.update(key=logical,attempt_key=actual,interrupted_attempt=str(original),interruption_audit=str(path))
 receipt=o/(actual+'-completed.json');assert not receipt.exists();core.save(receipt,row)
 assert c['frozen']==core.fingerprint();c['jobs'][logical]=row;c.setdefault('interruptions',[]).append(dict(audit=str(path),sha256=core.sha(path),retry_key=actual,logical_key=logical));core.save(o/'campaign.json',c)
 core.save(o/'state.json',dict(active=None,completed=len(c['jobs']),phase='repeats',planned_representatives=8,planned_repeats=8,baseline_runs=0,default_promoted=False));print('end',actual,row['status'],row.get('T'),row.get('A'),row.get('D'),flush=True)
if __name__=='__main__':main()
