#!/usr/bin/env python3
"""Anytime eGraphMap frontend: only complete, pre-deadline receipts survive search kill."""
import hashlib,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
from source_cover import build_source_cover
OBJECTIVE_POLICY='finite-physical-delay-area-lex-v1'
def better_da(candidate,incumbent):
    values=[candidate[k]for k in ['area','delay']]+[incumbent[k]for k in ['area','delay']]
    return all(isinstance(v,(int,float))and math.isfinite(v)and v>=0 for v in values)and(candidate['delay'],candidate['area'])<(incumbent['delay'],incumbent['area'])

class ScanState:
    def __init__(self):
        self.processed=set()
        self.counts=dict(manifests_seen=0,metadata_rejected=0,nonimproving=0,netlists_checked=0,netlist_bytes_checked=0,integrity_rejected=0,receipts_committed=0)
    def report(self):return dict(publication_policy='latest-committed-first-v1',objective_policy=OBJECTIVE_POLICY,examined_sequences=len(self.processed),counters=self.counts,internal_CEC_implied=False)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fnv(data):
    v=0xcbf29ce484222325
    for b in data:v=((v^b)*0x100000001b3)&((1<<64)-1)
    return f'{v:016x}'
def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.pending');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def scan(root,receipt_path,deadline,provenance,source,best=None,state=None):
    """Committed transactions are examined once; rejected metadata authorizes no output."""
    state=state if state is not None else ScanState()
    for folder in sorted(root.glob('[0-9]'*8),reverse=True) if root.exists()else[]:
        sequence=int(folder.name)
        if sequence in state.processed:continue
        if time.monotonic()>=deadline:break
        try:
            manifest=folder/'manifest.json';m=json.loads(manifest.read_text());state.counts['manifests_seen']+=1
            if m.get('kind')not in ('complete-generated-cover','complete-source-cover')or m.get('sequence')!=sequence or not all(isinstance(m.get(k),(int,float))and math.isfinite(m[k])and m[k]>=0 for k in ['area','delay']):
                state.processed.add(sequence);state.counts['metadata_rejected']+=1;continue
            reference=best if best else source
            if not better_da(m,reference):
                state.processed.add(sequence);state.counts['nonimproving']+=1;continue
            netlist=folder/'mapped.v';data=netlist.read_bytes();state.counts['netlists_checked']+=1;state.counts['netlist_bytes_checked']+=len(data)
            if m['bytes']!=len(data)or m['fnv64']!=fnv(data):
                state.processed.add(sequence);state.counts['integrity_rejected']+=1;continue
            record=dict(m,netlist=str(netlist.resolve()),netlist_sha256=hashlib.sha256(data).hexdigest(),manifest=str(manifest.resolve()),manifest_sha256=sha(manifest),provenance=provenance)
            observed=time.monotonic()
            if observed>=deadline:break
            record.update(publication_policy='latest-committed-first-v1',observed_monotonic=observed,deadline_monotonic=deadline,CEC='not yet checked',objective_policy=OBJECTIVE_POLICY)
            atomic_json(receipt_path,record);best=record;state.processed.add(sequence);state.counts['receipts_committed']+=1
        except (OSError,ValueError,KeyError,TypeError):continue
    return best

def verify_receipt(path,library,subject,native):
    r=json.loads(Path(path).read_text());p=r['provenance']
    assert p==dict(library_sha256=sha(library),prepared_sha256=sha(subject),native_sha256=sha(native))
    assert r['observed_monotonic']<r['deadline_monotonic']
    assert sha(r['netlist'])==r['netlist_sha256'] and sha(r['manifest'])==r['manifest_sha256']
    m=json.loads(Path(r['manifest']).read_text());assert m['kind']in ('complete-generated-cover','complete-source-cover')
    assert m['fnv64']==fnv(Path(r['netlist']).read_bytes())
    assert all(r[k]==m[k]for k in ['area','delay','sequence','instances','matched_cuts'])
    return r

def main():
    library,subject,out=map(Path,sys.argv[1:]);start=time.monotonic();deadline=float(os.environ.get('EGM_MAPPING_DEADLINE_MONOTONIC',start+600))
    native=Path(os.environ['EGM_NATIVE_OPTIMIZER']);models=library.parent/'library.json';meta=out.with_suffix('.complete.json')
    editable=os.environ.get('EGM_EDITABLE_SOURCE')=='1';bundle={}if editable else None
    text,seed=build_source_cover(subject.read_bytes(),models,deadline,phase_aware=True,bundle=bundle);source=out.with_suffix('.source.v');source.write_text(text);shutil.copyfile(source,out)
    if editable:atomic_json(out.parent/'source.bundle.json',bundle)
    receipt=out.with_suffix('.incumbent.receipt.json');root=out.parent/'incumbents';root.mkdir(exist_ok=False)
    provenance=dict(library_sha256=sha(library),prepared_sha256=sha(subject),native_sha256=sha(native))
    info=dict(policy='stream600-v1',source=seed,source_sha256=sha(source),library_sha256=sha(library),prepared_sha256=sha(subject),native_optimizer=str(native),native_optimizer_sha256=sha(native),optimizer_succeeded=False,selected='source_witness',CEC_required=True,main_mapper_merged=True,delivery_profile='single-output-da-streamed-library',count_limits=None,phase_limits=None)
    atomic_json(meta,info)
    worker=out.parent/'online-worker.v';env=dict(os.environ,EGM_ONLINE_MAPPING='1',EGM_INCUMBENT_DIR=str(root.resolve()),EGM_TOTAL_TIME_ONLY='1',EGM_TOTAL_REMAINING_SECS=str(max(.000001,deadline-time.monotonic())),EGM_NOCEC='1',EGM_STAGE_TRACE=str(out.parent/'online.stages.jsonl'),EGM_POOL_DIAG_PATH=str(out.parent/'online.pool.json'))
    if editable:env['EGM_SOURCE_BUNDLE']=str((out.parent/'source.bundle.json').resolve())
    env.pop('EGM_DIAG_PATH',None);best=None;timed_out=False;scan_state=ScanState();last_audit=0.
    with(out.parent/'native.stdout').open('w')as stdout,(out.parent/'native.stderr').open('w')as stderr:
        proc=subprocess.Popen([str(native),str(library),str(subject),str(worker)],env=env,stdout=stdout,stderr=stderr)
        while True:
            best=scan(root,receipt,deadline,provenance,seed,best,scan_state)
            if time.monotonic()-last_audit>=1.:
                atomic_json(out.parent/'publication.json',scan_state.report());last_audit=time.monotonic()
            if proc.poll()is not None:
                best=scan(root,receipt,deadline,provenance,seed,best,scan_state)
                break
            remaining=deadline-time.monotonic()
            if remaining<=0:timed_out=True;proc.kill();proc.wait();break
            time.sleep(min(.05,remaining))
    atomic_json(out.parent/'publication.json',scan_state.report())
    if best:
        verified=verify_receipt(receipt,library,subject,native);shutil.copyfile(verified['netlist'],out)
        info.update(optimizer_succeeded=verified['kind']=='complete-generated-cover',selected='validated_online_incumbent'if verified['kind']=='complete-generated-cover'else'native_source_seed',incumbent=verified)
    info.update(search_finished=proc.returncode==0,search_deadline_hit=timed_out,native_exit_code=proc.returncode,total_mapping_frontend_secs=time.monotonic()-start,selected_verilog_sha256=sha(out))
    atomic_json(meta,info)
    print(json.dumps(dict(area=best['area']if best else seed['area'],delay=best['delay']if best else seed['delay'],optimizer_succeeded=info['optimizer_succeeded'],search_finished=proc.returncode==0)))
if __name__=='__main__':main()
