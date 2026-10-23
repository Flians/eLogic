"""Content-deduplicate immutable experiment data; never pick by experiment quality.
All paths and SHA256 contents survive. Shared historical aliases become read-only.
"""
import argparse, collections, fcntl, hashlib, json, os, shutil, stat, uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
EXCLUDE={'.git','.codex','.agents','.aws','src','source','frozen_source','stock','tools','target','fixtures','epfl_resyn2_subjects_20261012','search_resources_20261021'}
EXT={'.v','.blif','.aig','.aag','.dot','.genlib','.lib'}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def info(p):
 s=p.stat();return dict(size=s.st_size,inode=s.st_ino,device=s.st_dev,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,uid=s.st_uid,gid=s.st_gid,allocated=s.st_blocks*512)
def lock():
 streams=[]
 for name in ['results_v2/independent_components_20261019/global.lock','results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock']:
  f=(ROOT/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);streams.append(f)
 return streams

def plan(output):
 locks=lock();sizes=collections.defaultdict(list);count=0
 for root in ROOT.iterdir():
  if not root.is_dir() or root.is_symlink() or not(root.name=='results' or root.name.startswith('results_v')):continue
  for base,ds,fs in os.walk(root,followlinks=False):
   ds[:]=[d for d in ds if d not in EXCLUDE and not d.startswith(('source_','frozen_')) and not Path(base,d).is_symlink()]
   for name in fs:
    p=Path(base,name)
    if p.suffix.lower()not in EXT or p.is_symlink():continue
    s=p.stat()
    # Never change metadata of a pre-existing hardlink that may reach outside this inventory.
    if s.st_size<65536 or s.st_nlink!=1 or s.st_uid!=os.getuid():continue
    count+=1;sizes[(s.st_dev,s.st_size,s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode))].append(p)
 samples=collections.defaultdict(list)
 for key,files in sizes.items():
  if len(files)<2:continue
  for p in files:
   with p.open('rb')as f:
    first=f.read(8192);f.seek(max(0,p.stat().st_size-8192));last=f.read(8192)
   samples[(key,hashlib.sha256(first+last).hexdigest())].append(p)
 groups=[];hashed=0
 for files in samples.values():
  if len(files)<2:continue
  full=collections.defaultdict(list)
  for p in files:full[sha(p)].append(p);hashed+=1
  for digest,files in full.items():
   if len(files)<2:continue
   rows=[dict(path=str(p.relative_to(ROOT)),**info(p))for p in sorted(files,key=lambda p:(len(str(p)),str(p)))]
   groups.append(dict(sha256=digest,files=rows))
 state=dict(root=str(ROOT),candidate_files=count,full_hashed_files=hashed,groups=groups,duplicate_paths=sum(len(g['files'])-1 for g in groups),estimated_redundant_allocated_bytes=sum(f['allocated']for g in groups for f in g['files'][1:]),excluded=sorted(EXCLUDE),preserve_paths_and_sha256=True,shared_aliases_read_only=True)
 output.write_text(json.dumps(state,indent=2)+'\n');print('groups',len(groups),'duplicate paths',state['duplicate_paths'],'redundant GiB',round(state['estimated_redundant_allocated_bytes']/1024**3,3),flush=True)

def apply(plan_path,output):
 assert not output.exists();locks=lock();state=json.loads(plan_path.read_text());assert state['root']==str(ROOT)
 before=shutil.disk_usage(ROOT).free;linked=0;reclaimed=0;skipped=[];canonical=[];log=output.with_suffix('.actions.jsonl')
 assert not log.exists()
 with log.open('w')as stream:
  for group in state['groups']:
   members=group['files'];paths=[ROOT/x['path']for x in members]
   assert all(ROOT in p.parents and p.parts[len(ROOT.parts)].startswith('results')and not any(n in p.parts for n in EXCLUDE)for p in paths)
   if any(not p.is_file()or p.is_symlink()or info(p)!={k:v for k,v in x.items()if k!='path'} for p,x in zip(paths,members)):
    # Compare only saved stat fields; path is not itself a stat field.
    changed=[str(p)for p,x in zip(paths,members)if not p.is_file()or p.is_symlink()or info(p)!={k:v for k,v in x.items()if k!='path'}]
    if changed:skipped.extend(changed);continue
   anchor=paths[0];assert sha(anchor)==group['sha256']
   anchor.chmod(members[0]['mode']&~0o222);canonical.append(dict(path=members[0]['path'],old_mode=members[0]['mode']))
   for p,x in zip(paths[1:],members[1:]):
    temporary=p.with_name('.cleanup-link-'+uuid.uuid4().hex)
    try:os.link(anchor,temporary,follow_symlinks=False);os.replace(temporary,p)
    finally:
     if temporary.exists():temporary.unlink()
    assert p.stat().st_ino==anchor.stat().st_ino
    reclaimed+=x['allocated']
    stream.write(json.dumps(dict(path=x['path'],canonical=members[0]['path'],sha256=group['sha256'],old_stat=x))+'\n');linked+=1
   stream.flush()
 result=dict(plan=str(plan_path),linked_duplicate_paths=linked,allocated_bytes_replaced=reclaimed,canonical_read_only=canonical,skipped_changed=skipped,free_before=before,free_after=shutil.disk_usage(ROOT).free,content_deleted=False,source_algorithms_changed=False,git_not_touched=True,actions=str(log))
 output.write_text(json.dumps(result,indent=2)+'\n');print('linked',linked,'free-space increase GiB',round((result['free_after']-before)/1024**3,3))

def verify(plan_path,output):
 state=json.loads(plan_path.read_text());checked=0
 for g in state['groups']:
  anchor=ROOT/g['files'][0]['path'];digest=sha(anchor);assert digest==g['sha256']
  for x in g['files']:
   p=ROOT/x['path'];assert p.is_file()and p.stat().st_size==x['size']
   if p.stat().st_ino!=anchor.stat().st_ino:assert sha(p)==g['sha256']
   checked+=1
 output.write_text(json.dumps(dict(verified_paths=checked,groups=len(state['groups']),all_original_contents_and_paths_preserved=True),indent=2)+'\n');print('verified',checked,'original data paths')

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=['plan','apply','verify']);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path);a=ap.parse_args()
 if a.mode=='plan':plan(a.plan)
 elif a.mode=='apply':assert a.output;apply(a.plan,a.output)
 else:assert a.output;verify(a.plan,a.output)
