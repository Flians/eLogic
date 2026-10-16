"""Preserve interrupted attempts and restore genuine paired cold/warm cache starts."""
from pathlib import Path
import datetime,hashlib,json,shutil
root=Path(__file__).resolve().parents[2]/'results_v28_three_parts_query_budget'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
archive=root/'interrupted_attempts'/stamp
records=[]
for mode in ['cold','warm']:
 for stage in ['tab_mfs3','tab_main','ablation']:
  for wd in sorted((root/mode/stage).glob('*')):
   if not wd.is_dir():continue
   if wd.name.startswith('abc.'):
    if (wd/'result.json').exists():continue
    dest=archive/mode/stage/wd.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(wd),str(dest));records.append(dict(mode=mode,stage=stage,arm=wd.name,reason='external interruption; no ABC result'));continue
   row=wd/'row.tsv'
   if row.exists():
    rc=row.read_text().strip().split('\t')[-1]
    assert rc in ['0','142'],(row,'Diagnose a recorded failure; never relabel it as an interruption',rc)
    continue
   cache=root/'cache'/stage/wd.name
   dest=archive/mode/stage/wd.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(wd),str(dest))
   saved=archive/'cache'/stage/wd.name
   if cache.exists():saved.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(cache),str(saved))
   restored={}
   if mode=='warm':
    proof=root/'cold'/stage/wd.name/'cache_provenance.json'
    if proof.exists():
     trace=json.loads(proof.read_text())
     for name,digest in trace['cache_record_sha256'].items():
      source=saved/name;assert source.is_file() and sha(source)==digest,(source,'cold pool unavailable')
      cache.mkdir(parents=True,exist_ok=True);shutil.copy2(source,cache/name);restored[name]=digest
   records.append(dict(mode=mode,stage=stage,arm=wd.name,reason='external interruption; no admitted final row',cold_records_restored=restored))
if records:
 archive.mkdir(parents=True,exist_ok=True);(archive/'resume_manifest.json').write_text(json.dumps(dict(reason='tool session interrupted externally; attempts are not native timeouts or completed runs',attempts=records),indent=2)+'\n')
print(json.dumps(dict(archived=len(records),archive=str(archive))))
