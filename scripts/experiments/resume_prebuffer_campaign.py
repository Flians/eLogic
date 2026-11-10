#!/usr/bin/env python3
"""Preserve recorded observations; archive only daemon-interrupted attempts.

A warm retry retains precisely its completed cold pool. New warm files are
identified after the cold result was committed; the retained key count must
match the cold receipt. No completed/timeout result is replaced.
"""
import argparse,hashlib,json,shutil,datetime
from pathlib import Path
from run_prebuffer_three_parts import ARMS,DESIGNS

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def prepare(root):
 stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S');archive=root/('interrupted_'+stamp);archive.mkdir(exist_ok=False)
 completed={str(p.relative_to(root)):sha(p) for p in root.glob('*/*/*/result.json') if p.parts[-4]in['main','main_r1','mfs3','strong_common','ablation']}
 specs=[(s,'ours',d)for s in ['main','mfs3','strong_common','main_r1']for d in DESIGNS]+[('ablation',a,d)for d in DESIGNS for a in ARMS]
 actions=[]
 for stage,arm,design in specs:
  for mode in ['cold','warm']:
   wd=root/stage/arm/f'{mode}.{design}';pool=root/'cache'/stage/arm/design
   if (wd/'result.json').exists():continue
   if not wd.exists():continue
   entry=dict(case=str(wd.relative_to(root)),reason='daemon_restart',completed_observation=False)
   if mode=='cold':
    if pool.exists():
     target=archive/'cache'/stage/arm/design;target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(pool),target)
     entry['cold_partial_pool_archived']=str(target.relative_to(root))
   else:
    cold=wd.parent/f'cold.{design}'/'result.json';assert cold.exists(),cold
    record=json.loads(cold.read_text());cut=cold.stat().st_mtime_ns
    keep=[p for p in pool.rglob('*.json')if p.stat().st_mtime_ns<=cut];extra=[p for p in pool.rglob('*.json')if p.stat().st_mtime_ns>cut]
    assert len(keep)==record['cache_files'],('cannot reconstruct exact cold key set',wd,len(keep),record['cache_files'])
    entry.update(cold_result_sha256=sha(cold),cold_pool_retained_files=len(keep),retained_cold_pool_sha256={str(p.relative_to(pool)):sha(p)for p in keep},warm_files_removed=len(extra))
    for p in extra:
     target=archive/'warm_extra_cache'/stage/arm/design/p.relative_to(pool);target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(p),target)
   target=archive/stage/arm/wd.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(wd),target);entry['archived_attempt']=str(target.relative_to(root));actions.append(entry)
 assert all(sha(root/p)==h for p,h in completed.items())
 result=dict(reason='daemon restart interrupted active children; recorded completed/timeouts retained byte-for-byte',archive=str(archive.relative_to(root)),completed_results=completed,interrupted_attempts=actions,frozen_engine_sha256=sha(root/'inputs/engine'),optimizer_configuration_changed=False)
 (archive/'resume_manifest.json').write_text(json.dumps(result,indent=2)+'\n');(root/'latest_resume.json').write_text(json.dumps(dict(archive=result['archive'],completed_records_retained=len(completed),interrupted_retries=len(actions)),indent=2)+'\n')
 print(json.dumps(dict(completed_records_retained=len(completed),interrupted_retries=len(actions),archive=result['archive']),indent=2))
 return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();prepare(a.root.resolve())
