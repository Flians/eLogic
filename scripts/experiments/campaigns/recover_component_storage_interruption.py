#!/usr/bin/env python3
"""One explicitly audited infrastructure retry; original partial artifacts never overwritten."""
import copy,fcntl,hashlib,json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[3];O=(R/'results_v2/single_output_components_full_20261017').resolve();cp=O/'campaign.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 tmp=p.with_name(p.name+'.recovery.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
locks=[]
for p in [O/'runner.lock',R/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock',R/'results_v2/small_cut_quota_cases_20261014/runner.lock']:
 f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
c=json.loads(cp.read_text());assert len(c['jobs'])==220
for p,h in c['fingerprint'].items():assert sha(p)==h,p
name='3in_sin_no_sdc';partial=O/name;assert partial.exists() and not (partial/'report.json').exists()
assert (O/(name+'.audit.json')).stat().st_size==0
archive=O/'interruption_archive/enospc1';archive.mkdir(parents=True,exist_ok=False)
files={str(p.relative_to(O)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in partial.rglob('*') if p.is_file()}
for p in [O/(name+'.audit.json'),O/(name+'.log'),O/'campaign.json.tmp']:
 if p.exists():
  files[str(p.relative_to(O))]={'sha256':sha(p),'bytes':p.stat().st_size}
  if p.name=='campaign.json.tmp':(archive/'checkpoint-write.tmp').write_bytes(p.read_bytes())
retry_name=name+'.retry_enospc1';folder=O/retry_name;audit=O/(retry_name+'.audit.json');log=O/(retry_name+'.log');assert not folder.exists() and not audit.exists()
ref=next(j for j in c['jobs'] if j['name']=='3in_sin_protected_area');argv=copy.deepcopy(ref['argv']);argv[argv.index('--audit')+1]=str(audit);argv[argv.index('--output')+1]=str(folder);argv+=['--dc-support','0']
ledger=dict(kind='infrastructure_ENOSPC',case=name,original_committed_jobs=220,original_checkpoint_sha256=sha(cp),original_relocation_ledger=str(O/'relocation_ledger.json'),original_partial_files=files,original_native_completed=True,original_independent_CEC='unmeasured',original_final_report_missing=True,original_mapping_clock_audit_missing=True,original_not_qualified_as_success=True,retry=retry_name,retry_argv=argv,retry_reason='Storage failure occurred after native/normalization but before independent evaluator/report/audit could be persisted. Relocation is SHA-verified. One predetermined retry after environment repair; no old partial result is supplied as a hint and no best-run selection.',baseline_algorithms_changed=False,mapper_parameters_or_budgets_changed=False,helper_sha256=sha(Path(__file__).resolve()),time_unix=time.time())
ledger_path=archive/'ledger.json';save(ledger_path,ledger)
with log.open('w') as f:run=subprocess.run(argv,cwd=R,stdout=f,stderr=subprocess.STDOUT)
for p,h in c['fingerprint'].items():assert sha(p)==h,p
artifact_keys=['report','audit','pool','candidates','trace','pool_frontier','pool_guard','helper_allocation','diagnostic_profile']
j={k:copy.deepcopy(v) for k,v in ref.items() if k not in ['argv','exit_code'] and k not in artifact_keys and not any(k==a+'_sha256' for a in artifact_keys)}
j.update(name=name,profile='no_sdc',argv=argv,exit_code=run.returncode,component_overrides=['--dc-support','0'],infrastructure_recovery=str(ledger_path),infrastructure_recovery_sha256=sha(ledger_path),actual_output_directory=str(folder))
for key,p in [('report',folder/'report.json'),('audit',audit)]:
 if p.exists():j[key]=str(p);j[key+'_sha256']=sha(p)
assert 'report' in j and 'audit' in j
for row in json.loads(Path(j['report']).read_text())['results']:
 for key,env in [('pool','EGM_POOL_DIAG_PATH'),('candidates','EGM_DIAG_PATH'),('trace','EGM_RECOVERY_TRACE')]:
  p=Path(row.get('settings',{}).get(env,'/nonexistent'))
  if p.is_file():j[key]=str(p);j[key+'_sha256']=sha(p)
  for extra,suffix in ([('pool_frontier','.frontier.json'),('pool_guard','.pool-guard.json'),('helper_allocation','.helper-allocation.json')] if key=='pool' else [('diagnostic_profile','.profile.json')] if key=='candidates' else []):
   q=p.with_suffix(suffix)
   if q.is_file():j[extra]=str(q);j[extra+'_sha256']=sha(q)
c['jobs'].append(j);c.setdefault('infrastructure_interruptions',[]).append({'ledger':str(ledger_path),'sha256':sha(ledger_path),'extra_native_attempts':1,'extra_independent_CEC':'unmeasured','qualified_retry':name,'storage_relocation':str(O/'relocation_ledger.json')});c['status']='checkpointed_after_storage_recovery';save(cp,c)
print('Preserved original partial; registered predetermined retry',run.returncode,'jobs',len(c['jobs']),flush=True)
