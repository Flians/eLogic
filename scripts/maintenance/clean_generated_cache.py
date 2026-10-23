"""Remove only inventoried, regenerable build/bytecode/TeX intermediates.
Final executables, dynamic libraries, source snapshots and manifest-pinned files survive.
"""
import argparse, json, os, re, shutil, tarfile, fcntl
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SKIP={'.git','.codex','.agents','.aws'}
CACHE={'deps','incremental','.fingerprint','CMakeFiles','__pycache__'}
MARKERS=('/deps/','/incremental/','/.fingerprint/','/CMakeFiles/','/__pycache__/')
AUX={'.aux','.log','.out','.fls','.fdb_latexmk','.synctex.gz','.toc','.nav','.snm','.vrb'}

def files_under(folder):
 for base,ds,fs in os.walk(folder,followlinks=False):
  ds[:]=[d for d in ds if d not in SKIP and not Path(base,d).is_symlink()]
  for name in fs:
   p=Path(base,name)
   if not p.is_symlink():yield p

def stat(p):
 s=p.stat();return dict(size=s.st_size,allocated=s.st_blocks*512,inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns,mode=s.st_mode&0o777)

def plan(output):
 protected=set();manifests=[];cache_dirs=[];meta=[]
 for base,ds,fs in os.walk(ROOT,followlinks=False):
  ds[:]=[d for d in ds if d not in SKIP and not Path(base,d).is_symlink()]
  p=Path(base)
  # Do not modify frozen third-party source snapshots, even when they accidentally include builds.
  if any(x in p.parts for x in ['frozen_source','stock']):ds[:]=[];continue
  for d in list(ds):
   q=p/d
   if d=='__pycache__' or d=='CMakeFiles' or (d in CACHE and p.name in ['release','debug']):
    cache_dirs.append(q);ds.remove(d)
  meta.extend(p/f for f in fs if f.endswith('.json') and not f.startswith('.') and any(k in f.lower()for k in ['campaign','freeze','manifest','fingerprint','source_sha','sources_sha','source_hash','snapshot','integration','provenance','delivery','build','compile','validation','checkpoint','plan','test','probe','record']))
 def strings(v):
  if isinstance(v,str):yield v
  elif isinstance(v,dict):
   for k,x in v.items():yield k;yield from strings(x)
  elif isinstance(v,list):
   for x in v:yield from strings(x)
 print('cache directories',len(cache_dirs),'metadata references',len(meta),flush=True)
 for ordinal,p in enumerate(meta):
  if ordinal%250==0:
   output.with_suffix('.progress.json').write_text(json.dumps({'metadata_checked':ordinal,'metadata_total':len(meta),'protected_cache_files':len(protected)}))
  if p.stat().st_size>8*1024**2:continue
  # Only metadata that can pin one of the cache categories can protect those categories.
  with p.open('rb') as f:
   if not any(any(m.encode() in chunk for m in MARKERS) for chunk in iter(lambda:f.read(262144),b'')):continue
  try:d=json.loads(p.read_text())
  except (ValueError,UnicodeError):continue
  manifests.append(str(p.relative_to(ROOT)))
  for text in strings(d):
   if not any(m in '/'+text for m in MARKERS):continue
   try:
    q=Path(text)
    bases=[q] if q.is_absolute() else [ROOT/q,p.parent/q,p.parent/'frozen_source'/q,p.parent/'source'/q]
    for b in bases:
     if b.is_file():protected.add(str(b.resolve()))
   except (ValueError,OSError):pass
 rows=[];kept=[]
 for folder in cache_dirs:
  for p in files_under(folder):
   if p.stat().st_mode&0o111 or str(p.resolve()) in protected or re.search(r'\.(so(?:\..*)?|dll|dylib|pyd)$',p.name):
    kept.append(str(p.relative_to(ROOT)));continue
   rows.append(dict(path=str(p.relative_to(ROOT)),kind='generated_build_or_bytecode',**stat(p)))
 for p in files_under(ROOT/'epmapping_paper'):
  if any(p.name.endswith(s) and (p.parent/(p.name[:-len(s)]+'.tex')).is_file() for s in AUX) and '.git' not in p.parts:
   rows.append(dict(path=str(p.relative_to(ROOT)),kind='latex_auxiliary_archive_before_removal',**stat(p)))
 # No ancestor-directory deletion: final native binaries outside the listed files remain.
 state=dict(root=str(ROOT),files=rows,preserved=kept,manifest_reference_sources=manifests,protected_cache_paths=sorted(protected),total_allocated_before_hardlink_adjustment=sum(x['allocated']for x in rows),git_not_touched=True,source_and_results_not_removed=True)
 output.write_text(json.dumps(state,indent=2)+'\n');print('planned',len(rows),'files,',round(state['total_allocated_before_hardlink_adjustment']/1024**3,3),'GiB allocated; pinned/dynamic preserved',len(kept))

def apply(plan_path,output):
 state=json.loads(plan_path.read_text());assert state['root']==str(ROOT);locks=[]
 for name in ['results_v2/independent_components_20261019/global.lock','results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock']:
  f=(ROOT/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 active=ROOT/'results_v2/search_resources_20261021/small_key_epfl/campaign.json'
 if active.exists():
  c=json.loads(active.read_text());assert len(c['jobs'])==c['expected_jobs'] and not c.get('current_job'),'finite mapper batch still active'
 before=shutil.disk_usage(ROOT).free
 latex=[x for x in state['files']if x['kind'].startswith('latex')]
 if latex:
  archive=output.parent/'latex_auxiliary.tar.gz';assert not archive.exists()
  with tarfile.open(archive,'w:gz')as t:
   for x in latex:t.add(ROOT/x['path'],arcname=x['path'],recursive=False)
 removed=[];skipped=[];reclaimable=0
 for x in state['files']:
  p=ROOT/x['path'];assert ROOT in p.parents and not any(k in p.parts for k in SKIP)
  if not p.is_file() or p.is_symlink() or stat(p)!= {k:x[k]for k in stat(p)}:skipped.append(x['path']);continue
  if p.stat().st_nlink==1:reclaimable+=p.stat().st_blocks*512
  p.unlink();removed.append(x)
 # Remove only now-empty generated directories.
 dirs={q for x in removed for q in (ROOT/x['path']).parents if ROOT in q.parents and any(k in q.parts for k in CACHE)}
 for p in sorted(dirs,key=lambda p:len(p.parts),reverse=True):
  try:p.rmdir()
  except OSError:pass
 output.write_text(json.dumps(dict(removed=removed,allocated_bytes_unlinked_last_link=reclaimable,skipped_changed_or_missing=skipped,free_before=before,free_after=shutil.disk_usage(ROOT).free,git_not_touched=True,essential_data_not_deleted=True),indent=2)+'\n')
 print('removed',len(removed),'files; changed/missing skipped',len(skipped),'free-space increase GiB',round((shutil.disk_usage(ROOT).free-before)/1024**3,3))

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=['plan','apply']);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path);a=ap.parse_args()
 if a.mode=='plan':plan(a.plan)
 else:assert a.output is not None;apply(a.plan,a.output)
