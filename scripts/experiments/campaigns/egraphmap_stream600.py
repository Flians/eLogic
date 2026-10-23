#!/usr/bin/env python3
"""Anytime eGraphMap frontend: only complete, pre-deadline receipts survive search kill."""
import hashlib,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
from source_cover import build_source_cover
from egraphmap_time_only import better_da

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fnv(data):
    v=0xcbf29ce484222325
    for b in data:v=((v^b)*0x100000001b3)&((1<<64)-1)
    return f'{v:016x}'
def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.pending');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def scan(root,receipt_path,deadline,provenance,source,best=None):
    """Parent-observed complete transactions, not an arbitrary killed worker file."""
    for folder in sorted(root.glob('[0-9]'*8)) if root.exists() else []:
        if best and int(folder.name)<=best['sequence']:continue
        try:
            manifest=folder/'manifest.json';m=json.loads(manifest.read_text());netlist=folder/'mapped.v';data=netlist.read_bytes()
            if m['kind']!='complete-generated-cover' or m['sequence']!=int(folder.name):continue
            if m['bytes']!=len(data)or m['fnv64']!=fnv(data):continue
            if not all(isinstance(m[k],(float,int))and math.isfinite(m[k])and m[k]>=0 for k in ['area','delay']):continue
            reference=best if best else source
            if not better_da(m,reference):continue
            record=dict(m,netlist=str(netlist.resolve()),netlist_sha256=hashlib.sha256(data).hexdigest(),manifest=str(manifest.resolve()),manifest_sha256=sha(manifest),provenance=provenance)
            observed=time.monotonic()
            if observed>=deadline:break
            record.update(observed_monotonic=observed,deadline_monotonic=deadline,CEC='not yet checked')
            atomic_json(receipt_path,record);best=record
        except (OSError,ValueError,KeyError,TypeError):continue
    return best

def verify_receipt(path,library,subject,native):
    r=json.loads(Path(path).read_text());p=r['provenance']
    assert p==dict(library_sha256=sha(library),prepared_sha256=sha(subject),native_sha256=sha(native))
    assert r['observed_monotonic']<r['deadline_monotonic']
    assert sha(r['netlist'])==r['netlist_sha256'] and sha(r['manifest'])==r['manifest_sha256']
    m=json.loads(Path(r['manifest']).read_text());assert m['kind']=='complete-generated-cover'
    assert m['fnv64']==fnv(Path(r['netlist']).read_bytes())
    assert all(r[k]==m[k]for k in ['area','delay','sequence','instances','matched_cuts'])
    return r

def main():
    library,subject,out=map(Path,sys.argv[1:]);start=time.monotonic();deadline=float(os.environ.get('EGM_MAPPING_DEADLINE_MONOTONIC',start+600))
    native=Path(os.environ['EGM_NATIVE_OPTIMIZER']);models=library.parent/'library.json';meta=out.with_suffix('.complete.json')
    text,seed=build_source_cover(subject.read_bytes(),models,deadline,phase_aware=True);source=out.with_suffix('.source.v');source.write_text(text);shutil.copyfile(source,out)
    receipt=out.with_suffix('.incumbent.receipt.json');root=out.parent/'incumbents';root.mkdir(exist_ok=False)
    provenance=dict(library_sha256=sha(library),prepared_sha256=sha(subject),native_sha256=sha(native))
    info=dict(policy='stream600-v1',source=seed,source_sha256=sha(source),library_sha256=sha(library),prepared_sha256=sha(subject),native_optimizer=str(native),native_optimizer_sha256=sha(native),optimizer_succeeded=False,selected='source_witness',CEC_required=True,main_mapper_merged=False,count_limits=None,phase_limits=None)
    atomic_json(meta,info)
    worker=out.parent/'online-worker.v';env=dict(os.environ,EGM_ONLINE_MAPPING='1',EGM_INCUMBENT_DIR=str(root.resolve()),EGM_TOTAL_TIME_ONLY='1',EGM_TOTAL_REMAINING_SECS=str(max(.000001,deadline-time.monotonic())),EGM_NOCEC='1',EGM_STAGE_TRACE=str(out.parent/'online.stages.jsonl'),EGM_POOL_DIAG_PATH=str(out.parent/'online.pool.json'))
    env.pop('EGM_DIAG_PATH',None);best=None;timed_out=False
    with(out.parent/'native.stdout').open('w')as stdout,(out.parent/'native.stderr').open('w')as stderr:
        proc=subprocess.Popen([str(native),str(library),str(subject),str(worker)],env=env,stdout=stdout,stderr=stderr)
        while True:
            best=scan(root,receipt,deadline,provenance,seed,best)
            if proc.poll()is not None:
                best=scan(root,receipt,deadline,provenance,seed,best)
                break
            remaining=deadline-time.monotonic()
            if remaining<=0:timed_out=True;proc.kill();proc.wait();break
            time.sleep(min(.05,remaining))
    if best:
        verified=verify_receipt(receipt,library,subject,native);shutil.copyfile(verified['netlist'],out)
        info.update(optimizer_succeeded=True,selected='validated_online_incumbent',incumbent=verified)
    info.update(search_finished=proc.returncode==0,search_deadline_hit=timed_out,native_exit_code=proc.returncode,total_mapping_frontend_secs=time.monotonic()-start,selected_verilog_sha256=sha(out))
    atomic_json(meta,info)
    print(json.dumps(dict(area=best['area']if best else seed['area'],delay=best['delay']if best else seed['delay'],optimizer_succeeded=bool(best),search_finished=proc.returncode==0)))
if __name__=='__main__':main()
