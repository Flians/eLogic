"""Archive an idle checkpoint prefix plus adopted external artifacts, never imply completion."""
import argparse,fcntl,hashlib,json,subprocess,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]

def digest(stream):
 h=hashlib.sha256()
 for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()

def sha(path):
 with Path(path).open("rb") as stream:return digest(stream)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--checkpoint",type=Path,required=True);ap.add_argument("--label",required=True);a=ap.parse_args()
 assert a.label and all(c.isalnum() or c in "-_" for c in a.label)
 source=a.checkpoint.resolve();locks=[]
 for path in [ROOT/"results_v2/independent_components_20261019/global.lock",source.parent/"runner.lock"]:
  stream=path.open("a");fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(stream)
 before=source.read_bytes();campaign=json.loads(before)
 assert not campaign.get("current_job") and not campaign.get("in_progress"),"active/partial job must be audited first"
 assert len({job["name"] for job in campaign["jobs"]})==len(campaign["jobs"])
 expected={}
 for job in campaign["jobs"]:
  for path,wanted in job["sha256"].items():
   assert path not in expected or expected[path]==wanted
   expected[path]=wanted
 files={}
 for path in sorted(set([source,*[p for p in source.parent.rglob("*") if p.is_file()],*[Path(p) for p in expected]])):
  assert path.is_absolute() and path.is_file()
  value=sha(path)
  if str(path) in expected:assert value==expected[str(path)],path
  files[str(path).lstrip("/")]=dict(path=str(path),sha256=value,bytes=path.stat().st_size)
 assert source.read_bytes()==before
 out=ROOT/"results_v2/archives";out.mkdir(exist_ok=True)
 archive=out/(a.label+".tar.zst");manifest=out/(a.label+".manifest.json");verified=out/(a.label+".verified.json")
 assert not any(p.exists() for p in [archive,manifest,verified])
 meta=dict(checkpoint=str(source),checkpoint_sha256=hashlib.sha256(before).hexdigest(),observed_jobs=len(campaign["jobs"]),expected_jobs=campaign["expected_jobs"],complete_attempt_inventory=len(campaign["jobs"])==campaign["expected_jobs"],qualification_claim=False,includes_adopted_external_job_artifacts=True,files=files,time_unix=time.time())
 manifest.write_text(json.dumps(meta,indent=2)+"\n")
 z=subprocess.Popen(["zstd","-T1","-3","-q","-o",str(archive)],stdin=subprocess.PIPE)
 with tarfile.open(fileobj=z.stdin,mode="w|",dereference=True) as writer:
  for member,item in files.items():writer.add(item["path"],arcname=member,recursive=False)
 z.stdin.close();assert z.wait()==0
 decoder=subprocess.Popen(["zstd","-q","-dc",str(archive)],stdout=subprocess.PIPE);seen=set()
 with tarfile.open(fileobj=decoder.stdout,mode="r|") as reader:
  for member in reader:
   assert member.isfile() and member.name in files and member.name not in seen
   record=files[member.name];assert member.size==record["bytes"]
   assert digest(reader.extractfile(member))==record["sha256"],member.name
   seen.add(member.name)
 decoder.stdout.read();decoder.stdout.close();assert decoder.wait()==0 and seen==set(files)
 assert source.read_bytes()==before,"checkpoint changed while archive lock held"
 result=dict(archive=str(archive),archive_sha256=sha(archive),manifest=str(manifest),manifest_sha256=sha(manifest),checkpoint_sha256=meta["checkpoint_sha256"],observed_jobs=meta["observed_jobs"],expected_jobs=meta["expected_jobs"],complete_attempt_inventory=meta["complete_attempt_inventory"],qualification_claim=False,verified_files=len(files),original_bytes=sum(r["bytes"] for r in files.values()),compressed_bytes=archive.stat().st_size,all_members_hash_verified=True,raw_files_deleted=False)
 verified.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result),flush=True)

if __name__=="__main__":main()
