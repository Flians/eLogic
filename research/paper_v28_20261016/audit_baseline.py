"""Validate the newly executed ABC batch and exact common-input provenance."""
from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parents[2]/'results_v28_three_parts_query_budget'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
records=[]
assert (p/'cold/inputs/asap7_6in.genlib').read_bytes()==(p/'warm/inputs/asap7_6in.genlib').read_bytes()
for f in sorted((p/'cold/tab_mfs3').glob('abc.*/result.json')):
 r=json.loads(f.read_text());d=r['design'];wd=f.parent
 assert r['rc'] in [0,124] and r['secs']<=1000.01
 for mode in ['cold','warm']:
  start=p/mode/'tab_mfs3'/f'ours.{d}'/'epmapping_baseline.v'
  assert sha(start)==r['baseline_sha256']
  assert start.read_bytes()==(p/mode/'tab_mfs3'/f'base.{d}'/'epmapping_mapped.v').read_bytes()
 if r['rc']==0:assert r['cec']==1 and 'Networks are equivalent' in (wd/'cec.log').read_text()
 else:assert r['cec']==0
 records.append(dict(design=d,returncode=r['rc'],seconds=r['secs'],cec=r['cec'],initial_sha256=r['baseline_sha256'],result_sha256=sha(f),raw_log_sha256=sha(wd/'log')))
assert len(records)==16
out=dict(fresh_abc_runs=16,completed=sum(r['returncode']==0 for r in records),timeouts=sum(r['returncode']==124 for r in records),post_mapping_timeout=1000,records=records)
Path(__file__).with_name('baseline_verified.json').write_text(json.dumps(out,indent=2)+'\n');print('Fresh ABC verified:',out['completed'],'CEC completions,',out['timeouts'],'timeouts; all starts byte-identical')
