#!/usr/bin/env python3
"""Post-run integrity/causal-scope checks. Never feeds diagnostics to the mapper."""
import argparse,collections,fcntl,hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[3]
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def field(raw,key,following):
 a=raw.index(('"'+key+'":').encode())+len(key)+3;b=raw.index((',"'+following+'":').encode(),a)
 assert raw[a:a+1]==b'[' and raw[b-1:b]==b']';return hashlib.sha256(raw[a:b]).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--studies',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();records={};diags=[];fail=[];libs={};infra={}
 for cp in a.studies:
  c=load(cp);assert c['status']=='attempted' and len(c['jobs'])==c['expected_jobs']
  for p,h in c['fingerprint'].items():assert sha(p)==h,p
  for entry in c.get('infrastructure_interruptions',[]):
   lp=Path(entry['ledger']);assert sha(lp)==entry['sha256'];ledger=load(lp);root=cp.parent.resolve()
   for rel,meta in ledger['original_partial_files'].items():
    f=lp.parent/'checkpoint-write.tmp' if rel=='campaign.json.tmp' else root/rel
    assert sha(f)==meta['sha256'] and f.stat().st_size==meta['bytes'],str(f)
   infra[str(lp)]=dict(ledger=str(lp),sha256=sha(lp),extra_native_attempts=entry['extra_native_attempts'],independent_CEC='unmeasured',original_not_counted_as_success=True,qualified_retry=entry['qualified_retry'])

  for j in c['jobs']:
   r=load(j['report'])['results'][0];records[j['report']]=r
   if j['k'] not in libs:
    d=load(j['report']);p=Path(d['library']);p=p if p.is_absolute() else Path(j['report']).parent/p
    names=collections.Counter(line.split()[1] for line in p.read_text().splitlines() if line.startswith('GATE '));assert names and max(names.values())==1
    libs[j['k']]=dict(path=str(p),sha256=sha(p),physical_cell_names=len(names),maximum_outputs_per_cell_name=1)
   if r['status']!='ok':
    p=Path(r['settings']['EGM_POOL_DIAG_PATH']).with_suffix('.import.json');z=dict(study=cp.parent.name,name=j['name'],error=r.get('error'))
    if p.exists():
     d=load(p)
     if d['import_complete']:
      assert d['schema_version']==2 and d['actual_imported_enodes']==d['pre_rebuild_import_enodes'] and d['actual_imported_enodes']<=20000
      z.update(import_path=str(p),sha256=sha(p),import_complete=True,actual_imported_enodes=d['actual_imported_enodes'],allowance=int(r['settings']['EGM_NODES']),failure_after_import=True)
     else:
      assert d['cumulative_import_admitted_enodes']<=20000 and d['node_allowance']==20000
      z.update(import_path=str(p),sha256=sha(p),import_complete=False,actual_prefix=d['cumulative_import_admitted_enodes'],allowance=d['node_allowance'])
    fail.append(z);continue
   raw=Path(j['candidates']).read_bytes();pool=re.search(rb'"retained_pool_fingerprint":"([^"]+)"',raw).group(1).decode()
   diags.append(dict(study=cp.parent.name,k=j['k'],bench=j['bench'],profile=j['profile'],cut_sha256=field(raw,'cuts','delay'),graph_sha256=field(raw,'egraph_nodes','free'),canonical_sha256=field(raw,'canonical_signals','canonical_wire_only'),roots_sha256=field(raw,'roots','selected_instances'),recorded_pool_fingerprint=pool))
 for study,k,b in {(d['study'],d['k'],d['bench']) for d in diags}:
  ps=[d for d in diags if (d['study'],d['k'],d['bench'])==(study,k,b) and d['profile']!='no_saturation']
  assert all(all(p[f]==ps[0][f] for f in ['graph_sha256','canonical_sha256','roots_sha256']) for p in ps),(study,k,b,'graph')
  normal=[p for p in ps if p['profile']!='single_structure'];assert all(p['cut_sha256']==normal[0]['cut_sha256'] for p in normal),(study,k,b,'cuts')
  recovery=[p for p in ps if p['profile'] in ['legacy_control','protected_area','no_actual_area','no_phase_recovery','no_timing_area_module']]
  assert all(p['recorded_pool_fingerprint']==recovery[0]['recorded_pool_fingerprint'] for p in recovery),(study,k,b,'pool')
 old=load(R/'results_v2/cover_transaction_evidence_20261017/qualification.json');baseline=old['baseline_small_binary_check']
 for p,h in baseline.items():assert sha(p)==h,p
 assert sha(R/'target/release/bench_rs_egraph_map')==old['main_sha256']
 for folder in ['single_output_ablation_implementation_20261017','single_output_clean_components_implementation_20261017']:
  archived=load(R/'results_v2'/folder/'source_manifest.json')
  for p,h in archived.items():assert sha(p)==h
 locks=[];busy=[]
 for p in sorted(set((R/'results_v2').rglob('runner.lock')) | {cp.parent/'runner.lock' for cp in a.studies}):
  with p.open('r') as f:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(str(p))
   except BlockingIOError:busy.append(str(p))
 assert not busy,busy
 out=dict(unique_physical_attempts=len(records)+sum(e['extra_native_attempts'] for e in infra.values()),qualified_recorded_attempts=len(records),infrastructure_extra_attempts=list(infra.values()),unique_CEC=sum(r['status']=='ok' for r in records.values()),failures=fail,diagnostics=diags,single_output_libraries=libs,graph_canonical_roots_equal=True,cut_banks_equal_except_single_structure=True,recovery_profile_recorded_pool_fingerprints_equal=True,pool_caveat='Recorded64bit fingerprint, not independently exported cryptographic entire-pool proof. Within each recovery the pool is immutable. Candidate availability differs intentionally for cut/phase/pin/SDC factors.',main_sha256=old['main_sha256'],baseline_small_binary_check=baseline,prior_full_13403_file_baseline_audit_reused_not_rerun=True,main_promoted=False,runner_locks_idle=len(locks),busy=busy)
 a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:out[k] for k in ['unique_physical_attempts','unique_CEC','runner_locks_idle','main_promoted']})
if __name__=='__main__':main()
