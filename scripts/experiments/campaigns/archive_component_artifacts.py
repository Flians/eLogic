#!/usr/bin/env python3
"""Durable verified workspace archive for temporary-runtime experiment artifacts."""
import argparse,fcntl,hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
def digest(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def sha(p):
 with p.open('rb') as f:return digest(f)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);ap.add_argument('--label',required=True);a=ap.parse_args();folder=a.folder.resolve();c=json.loads((folder/'campaign.json').read_text());assert c.get('status')=='attempted' and len(c['jobs'])==c['expected_jobs']
 locks=[]
 for p in [folder/'runner.lock',R/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock',R/'results_v2/small_cut_quota_cases_20261014/runner.lock']:
  f=p.open('r');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 dest=R/'results_v2/archives';dest.mkdir(exist_ok=True);archive=dest/(a.label+'.tar.zst');manifest=dest/(a.label+'.manifest.json');verified=dest/(a.label+'.verified.json');assert not any(p.exists() for p in [archive,manifest,verified]);files={}
 for p in folder.rglob('*'):
  rel=str(Path(folder.name)/p.relative_to(folder))
  if p.is_symlink():files[rel]={'kind':'symlink','target':os.readlink(p)}
  elif p.is_file():files[rel]={'kind':'file','sha256':sha(p),'bytes':p.stat().st_size}
 manifest.write_text(json.dumps(dict(folder=str(folder),files=files,campaign_sha256=sha(folder/'campaign.json'),time_unix=time.time()),indent=2)+'\n')
 tar=subprocess.Popen(['tar','-C',str(folder.parent),'-cf','-',folder.name],stdout=subprocess.PIPE)
 z=subprocess.run(['zstd','-T1','-3','-q','-o',str(archive)],stdin=tar.stdout);tar.stdout.close();assert z.returncode==0 and tar.wait()==0
 dec=subprocess.Popen(['zstd','-q','-d','-c',str(archive)],stdout=subprocess.PIPE);seen=set()
 with tarfile.open(fileobj=dec.stdout,mode='r|') as t:
  for m in t:
   if m.isdir():continue
   assert m.name in files,m.name;f=files[m.name];seen.add(m.name)
   if m.isfile():assert f['kind']=='file' and m.size==f['bytes'] and digest(t.extractfile(m))==f['sha256'],m.name
   else:assert m.issym() and f['kind']=='symlink' and m.linkname==f['target'],m.name
 dec.stdout.read();dec.stdout.close()
 assert dec.wait()==0 and seen==set(files)
 data=dict(archive=str(archive),archive_sha256=sha(archive),manifest=str(manifest),manifest_sha256=sha(manifest),files=len(files),original_bytes=sum(f.get('bytes',0) for f in files.values()),compressed_bytes=archive.stat().st_size,stream_member_hash_verification=True,includes_interruption_artifacts=True,raw_runtime_storage=str(folder),logical_workspace_path=str(a.folder),no_data_or_failures_deleted=True,restore=f'zstd -dc ARCHIVE | tar -xf - -C {folder.parent} ; retain logical workspace symlink')
 verified.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data),flush=True)
if __name__=='__main__':main()
