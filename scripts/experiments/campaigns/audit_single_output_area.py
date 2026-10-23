#!/usr/bin/env python3
"""Post-run source/budget/pool diagnostics; never supplies mapper advice."""
import hashlib,json,re,fcntl,collections
from pathlib import Path
R=Path(__file__).resolve().parents[3]
O=R/'results_v2/single_output_area_evidence_20261017'
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def field(raw,key,following):
 a=raw.index(('"'+key+'":').encode())+len(key)+3
 b=raw.index((',"'+following+'":').encode(),a)
 assert raw[a:a+1]==b'[' and raw[b-1:b]==b']',(key,following)
 return hashlib.sha256(raw[a:b]).hexdigest()
records={};pool=[];fail=[]
for folder in ['single_output_area_full_20261017','single_output_area_pilot_20261017','single_output_area_repeat1_20261017','single_output_area_repeat2_20261017']:
 c=load(R/'results_v2'/folder/'campaign.json');assert c['status']=='attempted' and len(c['jobs'])==c['expected_jobs']
 for p,h in c['fingerprint'].items():assert sha(p)==h,p
 for j in c['jobs']:
  key=j['report'];r=load(key)['results'][0];records[key]=r
  if r['status']!='ok':
   p=Path(r['settings']['EGM_POOL_DIAG_PATH']).with_suffix('.import.json');d=load(p)
   assert not d['import_complete'] and d['cumulative_import_admitted_enodes']<=20000 and d['node_allowance']==20000
   fail.append(dict(folder=folder,name=j['name'],import_path=str(p),sha256=sha(p),actual_cumulative_prefix=d['cumulative_import_admitted_enodes']))
  if folder!='single_output_area_full_20261017' or r['status']!='ok':continue
  raw=Path(j['candidates']).read_bytes()
  p=re.search(rb'"retained_pool_fingerprint":"([^"]+)"',raw).group(1).decode()
  pool.append(dict(k=j['k'],bench=j['bench'],profile=j['profile'],cut_sha256=field(raw,'cuts','delay'),graph_sha256=field(raw,'egraph_nodes','free'),canonical_sha256=field(raw,'canonical_signals','canonical_wire_only'),roots_sha256=field(raw,'roots','selected_instances'),recorded_pool_fingerprint=p,recorded_pool_fingerprint_is_64bit_not_cryptographic=True))
for k,b in {(r['k'],r['bench']) for r in pool}:
 p=[r for r in pool if (r['k'],r['bench'])==(k,b)];assert len(p)==2
 assert all(p[0][f]==p[1][f] for f in ['cut_sha256','graph_sha256','canonical_sha256','roots_sha256','recorded_pool_fingerprint']),(k,b)
assert len(records)==128 and sum(r['status']=='ok' for r in records.values())==116 and len(fail)==12
libs={}
c=load(R/'results_v2/single_output_area_full_20261017/campaign.json')
for j in c['jobs']:
 if j['k'] in libs:continue
 d=load(j['report']);p=Path(d['library']);p=p if p.is_absolute() else Path(j['report']).parent/p
 names=collections.Counter(line.split()[1] for line in p.read_text().splitlines() if line.startswith('GATE '));assert names and max(names.values())==1
 libs[j['k']]=dict(path=str(p),sha256=sha(p),physical_cell_names=len(names),maximum_output_definitions_per_cell_name=max(names.values()))
archive=load(R/'results_v2/single_output_area_implementation_20261017/source_sha256.json');proto=Path('/tmp/emapper-single-output-area-20261017')
for p,h in archive.items():assert sha(proto/('target/release/bench_rs_egraph_map' if p=='binary' else p))==h,p
old=load(R/'results_v2/cover_transaction_evidence_20261017/qualification.json');baseline={}
for p,h in old['baseline_small_binary_check'].items():assert sha(p)==h;baseline[p]=h
assert sha(R/'target/release/bench_rs_egraph_map')==old['main_sha256']
idle=[];busy=[]
for p in (R/'results_v2').rglob('runner.lock'):
 with p.open('r') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN);idle.append(str(p))
  except BlockingIOError:busy.append(str(p))
assert not busy,busy
result=dict(unique_physical_attempts=len(records),unique_physical_CEC=sum(r['status']=='ok' for r in records.values()),actual_node_failures=fail,full_legal_pair_pool_diagnostics=pool,single_output_library_audit=libs,full_cut_graph_canonical_root_fields_sha_equal=True,recorded_pool_fingerprints_equal=True,pool_caveat='Stored64bit pool fingerprints plus deterministic unchanged generation/source; not independently exported cryptographic complete-pool hashes. Within one recovery pool is immutable.',source_archive_verified=True,main_sha256=old['main_sha256'],main_source_defaults_binary_unchanged=True,baseline_algorithms_changed=False,baseline_small_binary_check=baseline,prior_full_13403_file_baseline_audit_reused_not_rerun=True,runner_locks_idle=len(idle),busy=busy,main_promoted=False)
(O/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:result[k] for k in ['unique_physical_attempts','unique_physical_CEC','runner_locks_idle','main_promoted']})
