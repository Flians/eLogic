"""Durable, content-verified capsules for completed mapper snapshot trees."""
from pathlib import Path
import hashlib,json,subprocess,tarfile,shutil

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while data:=f.read(8*1024*1024):h.update(data)
 return h.hexdigest()

def dump(p,j):
 temp=p.with_suffix('.pending');temp.write_text(json.dumps(j,indent=2)+'\n');temp.replace(p)

def verify_archive(archive,manifest):
 seen=set();process=subprocess.Popen(['zstd','-q','-dc',str(archive)],stdout=subprocess.PIPE)
 try:
  with tarfile.open(fileobj=process.stdout,mode='r|') as tf:
   for entry in tf:
    if entry.isdir():continue
    assert entry.isfile(),entry.name
    name=entry.name.removeprefix('./');assert name in manifest,name
    h=hashlib.sha256();f=tf.extractfile(entry)
    while data:=f.read(8*1024*1024):h.update(data)
    assert h.hexdigest()==manifest[name],name;seen.add(name)
  assert process.wait()==0
 finally:
  if process.poll() is None:process.kill();process.wait()
 assert seen==set(manifest),(len(seen),len(manifest))

def seal(folder,archives):
 archives.mkdir(exist_ok=True,parents=True);receipt=archives/(folder.name+'.json');archive=archives/(folder.name+'.tar.zst')
 if receipt.exists():
  audit=json.loads(receipt.read_text());assert audit['original_directory']==str(folder)and sha(archive)==audit['archive_sha256'];verify_archive(archive,audit['all_files_sha256'])
  keep=Path(audit['selected_snapshot_kept'])if audit['selected_snapshot_kept']else None
  tree=folder/'incumbents'
  if tree.exists():
   for child in tree.iterdir():
    if child!=keep and child.is_dir()and child.name.isdigit():
     assert not child.is_symlink();shutil.rmtree(child)
  return dict(receipt=str(receipt),receipt_sha256=sha(receipt),archive=str(archive),archive_sha256=audit['archive_sha256'])
 if archive.exists():
  number=0
  while archive.with_name(archive.name+f'.interrupted-{number}').exists():number+=1
  archive.rename(archive.with_name(archive.name+f'.interrupted-{number}'))
 manifest={str(p.relative_to(folder)):sha(p)for p in folder.rglob('*')if p.is_file()}
 command=['tar','--hard-dereference','-I','zstd -T2 -3','-cf',str(archive),'-C',str(folder),'.']
 subprocess.run(command,check=True);verify_archive(archive,manifest)
 keep=None;p=folder/'raw.incumbent.receipt.json'
 if p.exists():
  j=json.loads(p.read_text());selected=Path(j['manifest']);assert selected.is_relative_to(folder/'incumbents');keep=selected.parent
 audit=dict(schema=1,archive=str(archive),archive_sha256=sha(archive),original_directory=str(folder),all_files_sha256=manifest,full_stream_replay_verified=True,command=command,selected_snapshot_kept=str(keep)if keep else None,remove_only_redundant_snapshots_after_verification=True)
 dump(receipt,audit)
 # Complete archive exists on the same persistent workspace filesystem before removal.
 tree=folder/'incumbents'
 if tree.exists():
  for child in tree.iterdir():
   if child!=keep and child.is_dir()and child.name.isdigit():
    assert not child.is_symlink();shutil.rmtree(child)
 return dict(receipt=str(receipt),receipt_sha256=sha(receipt),archive=str(archive),archive_sha256=audit['archive_sha256'])

def verify_files(folder,artifacts,capsule=None):
 saved={}
 if capsule:
  p=Path(capsule['receipt']);assert sha(p)==capsule['receipt_sha256'];audit=json.loads(p.read_text());assert sha(Path(capsule['archive']))==capsule['archive_sha256']==audit['archive_sha256'];assert audit['full_stream_replay_verified'];saved=audit['all_files_sha256']
 for rel,h in artifacts.items():
  p=folder/rel
  if p.is_file():assert sha(p)==h,rel
  else:assert capsule and saved.get(rel)==h,rel
