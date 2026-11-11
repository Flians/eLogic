#!/usr/bin/env python3
"""Anytime eGraphMap frontend: only complete, pre-deadline receipts survive search kill."""
import hashlib,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
from source_cover import build_source_cover
from physical import supplied_seed
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
def valid_integrity(manifest,data):
    """An explicit protocol selects its authoritative full-byte digest; never downgrade."""
    if manifest.get('bytes')!=len(data):return False
    algorithm=manifest.get('integrity_algorithm','fnv64')
    if algorithm=='sha256':return manifest.get('sha256')==hashlib.sha256(data).hexdigest()
    if algorithm=='fnv64':return manifest.get('fnv64')==fnv(data)
    return False
def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.pending');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def scan(root,receipt_path,deadline,provenance,source,best=None,state=None,folders=None,trace=None):
    """Committed transactions are examined once; rejected metadata authorizes no output."""
    state=state if state is not None else ScanState()
    for folder in folders if folders is not None else (sorted(root.glob('[0-9]'*8),reverse=True) if root.exists()else[]):
        sequence=int(folder.name)
        if sequence in state.processed:continue
        if time.monotonic()>=deadline:break
        if trace:trace.emit('discovered',sequence=sequence)
        try:
            manifest=folder/'manifest.json';m=json.loads(manifest.read_text());state.counts['manifests_seen']+=1
            if m.get('kind')not in ('complete-generated-cover','complete-source-cover')or m.get('sequence')!=sequence or not all(isinstance(m.get(k),(int,float))and math.isfinite(m[k])and m[k]>=0 for k in ['area','delay']):
                state.processed.add(sequence);state.counts['metadata_rejected']+=1;continue
            reference=best if best else source
            if reference is not None and not better_da(m,reference):
                state.processed.add(sequence);state.counts['nonimproving']+=1;continue
            netlist=folder/'mapped.v';data=netlist.read_bytes();state.counts['netlists_checked']+=1;state.counts['netlist_bytes_checked']+=len(data)
            if not valid_integrity(m,data):
                state.processed.add(sequence);state.counts['integrity_rejected']+=1;continue
            record=dict(m,netlist=str(netlist.resolve()),netlist_sha256=hashlib.sha256(data).hexdigest(),manifest=str(manifest.resolve()),manifest_sha256=sha(manifest),provenance=provenance)
            observed=time.monotonic()
            if trace:trace.emit('verified',monotonic=observed,sequence=sequence,eligible=observed<deadline)
            if observed>=deadline:break
            record.update(publication_policy='latest-committed-first-v1',observed_monotonic=observed,deadline_monotonic=deadline,CEC='not yet checked',objective_policy=OBJECTIVE_POLICY)
            if trace:trace.emit('receipt_write_begin',sequence=sequence)
            atomic_json(receipt_path,record)
            if trace:
                written=time.monotonic();trace.emit('receipt_written',monotonic=written,sequence=sequence,receipt_sha256=sha(receipt_path),write_completed_before_deadline=written<deadline)
            best=record;state.processed.add(sequence);state.counts['receipts_committed']+=1
        except (OSError,ValueError,KeyError,TypeError):continue
    return best

def verify_receipt(path,library,subject,native):
    r=json.loads(Path(path).read_text());p=r['provenance']
    assert p==dict(library_sha256=sha(library),prepared_sha256=sha(subject),native_sha256=sha(native))
    assert r['observed_monotonic']<r['deadline_monotonic']
    assert sha(r['netlist'])==r['netlist_sha256'] and sha(r['manifest'])==r['manifest_sha256']
    m=json.loads(Path(r['manifest']).read_text());assert m['kind']in ('complete-generated-cover','complete-source-cover')
    assert valid_integrity(m,Path(r['netlist']).read_bytes())
    assert all(r[k]==m[k]for k in ['area','delay','sequence','instances','matched_cuts'])
    return r

def main():
    library,subject,out=map(Path,sys.argv[1:]);start=time.monotonic();deadline=float(os.environ.get('EGM_MAPPING_DEADLINE_MONOTONIC',start+600))
    native=Path(os.environ['EGM_NATIVE_OPTIMIZER']);models=library.parent/'library.json';meta=out.with_suffix('.complete.json')
    direct=os.environ.get('EGM_DIRECT_NATIVE')=='1'
    if direct and any(os.environ.get(k) for k in ('EGM_SEED_VERILOG','EGM_SOURCE_BUNDLE','EGM_EXTERNAL_BUNDLE')):
        raise ValueError('direct native mapping rejects external seed/source bundle')
    editable=not direct and os.environ.get('EGM_EDITABLE_SOURCE')=='1'
    bundle={}if editable else None
    seed=None;constructive_seed=None;external=None;source=None
    if not direct:
        text,seed=build_source_cover(subject.read_bytes(),models,deadline,phase_aware=True,bundle=bundle);constructive_seed=dict(seed)
        source=out.with_suffix('.source.v')
        if os.environ.get('EGM_SEED_VERILOG'):
            _external=Path(os.environ['EGM_SEED_VERILOG'])
            measure=Path(os.environ.get('EGM_MEASURE_BINARY',str(native.with_name('emapper_measure'))))
            text,external=supplied_seed(library,_external,subject,out.parent/'external-seed-measurement',measure,
                                       os.environ.get('EGM_YOSYS','yosys'),os.environ.get('EGM_SEED_MANIFEST'))
            seed=dict(external['measured'],kind='external-complete-cover',external_seed=str(_external.resolve()),
                      external_seed_sha256=external['source_sha256'],canonical_measurement=external,egraph_search=False)
        source.write_text(text);shutil.copyfile(source,out)
        if editable:atomic_json(out.parent/'source.bundle.json',bundle)
    receipt=out.with_suffix('.incumbent.receipt.json');root=out.parent/'incumbents';root.mkdir(exist_ok=False)
    provenance=dict(library_sha256=sha(library),prepared_sha256=sha(subject),native_sha256=sha(native))
    info=dict(policy='direct-native600-v1'if direct else'stream600-v1',source=seed,source_sha256=sha(source)if source else None,library_sha256=sha(library),prepared_sha256=sha(subject),native_optimizer=str(native),native_optimizer_sha256=sha(native),optimizer_succeeded=False,selected='no_complete_cover'if direct else'source_witness',CEC_required=True,main_mapper_merged=True,delivery_profile='single-output-da-streamed-library',count_limits=None,phase_limits=None,direct_native=direct)
    if external:
        info.update(constructive_source=constructive_seed,external_seed=external,
                    source_bundle_role='native constructive cover; external seed is an outer delivery comparator',
                    experiment_scope='externally seeded refinement; supplied seed generation is not measured')
    atomic_json(meta,info)
    _grace=min(max(0.,float(os.environ.get('EGM_FINISH_GRACE_SECS','0'))),max(0.,deadline-time.monotonic()-1.))
    worker=out.parent/'online-worker.v';env=dict(os.environ,EGM_ONLINE_MAPPING='1',EGM_INCUMBENT_DIR=str(root.resolve()),EGM_TOTAL_TIME_ONLY='1',EGM_TOTAL_REMAINING_SECS=str(max(.000001,deadline-time.monotonic()-_grace)),EGM_FINISH_GRACE_SECS=str(_grace),EGM_NOCEC='1',EGM_STAGE_TRACE=str(out.parent/'online.stages.jsonl'),EGM_POOL_DIAG_PATH=str(out.parent/'online.pool.json'))
    if editable:env['EGM_SOURCE_BUNDLE']=str((out.parent/'source.bundle.json').resolve())
    if env.get('EGM_EXTERNAL_EDITABLE')=='1':
        from editable_seed import build_editable_seed
        try:
            if not external or not editable:raise ValueError('editable external seed missing checked physical input')
            foreign=build_editable_seed(subject.read_bytes(),models,external['netjson'],external['measured'],deadline)
            bundle_path=out.parent/'external.bundle.json';atomic_json(bundle_path,foreign)
            env['EGM_EXTERNAL_BUNDLE']=str(bundle_path.resolve())
            info.update(source_bundle_role='external exact physical seed projected into editable native cover',external_native_import_requested=True)
            atomic_json(meta,info)
        except Exception as exc:
            info.update(external_native_import_requested=True,external_native_import_failed=str(exc),search_finished=False,search_deadline_hit=time.monotonic()>=deadline,selected_verilog_sha256=sha(out))
            atomic_json(meta,info);return
    # Definition construction is mapping work: never renew its elapsed time.
    if time.monotonic()>=deadline:
        info.update(search_finished=False,search_deadline_hit=True,external_native_import_failed='mapping deadline before native launch',selected_verilog_sha256=sha(out)if out.exists()else None);atomic_json(meta,info);return
    env['EGM_TOTAL_REMAINING_SECS']=str(max(.000001,deadline-time.monotonic()-_grace))
    env.pop('EGM_DIAG_PATH',None);best=None;timed_out=False;scan_state=ScanState();last_audit=0.
    from publication_receiver import Receiver,TimingTrace
    receiver=Receiver(root,env.get('EGM_PUBLICATION_RECEIVER','scan'))
    trace=TimingTrace(out.parent/'frontend.publication-timing.jsonl',deadline)if env.get('EGM_PUBLICATION_TIMING')=='1'else None
    if trace:env['EGM_NATIVE_PUBLICATION_TIMING']=str(out.parent/'native.publication-timing.jsonl')
    def receive(best):
        folders=receiver.folders()
        best=scan(root,receipt,deadline,provenance,seed,best,scan_state,folders=folders,trace=trace)
        receiver.retry(folders,scan_state.processed)
        return best
    def audit():return dict(scan_state.report(),receiver=receiver.report())
    first_recorded=False
    with(out.parent/'native.stdout').open('w')as stdout,(out.parent/'native.stderr').open('w')as stderr:
        native_argv = [str(native), '--replay-recovery', env['EGM_REPLAY_SNAPSHOT'], str(worker), 'fair' if env.get('EGM_RECOVERY_FAIR') == '1' else 'legacy'] if env.get('EGM_REPLAY_SNAPSHOT') else [str(native),str(library),str(subject),str(worker)]
        proc=subprocess.Popen(native_argv,env=env,stdout=stdout,stderr=stderr)
        while True:
            best=receive(best)
            if direct and best and not first_recorded:
                first=dict(best,mapping_elapsed_seconds=best['observed_monotonic']-float(os.environ.get('EGM_MAPPING_STARTED_MONOTONIC',start)),constructor='existing native primitive/cut match and incremental delay Flow',external_mapper_called=False,source_bundle_used=False,timing_scope='first externally verified complete native cover, including supervisor lag')
                atomic_json(out.parent/'first-native-cover.json',first);first_recorded=True
            if time.monotonic()-last_audit>=1.:
                atomic_json(out.parent/'publication.json',audit());last_audit=time.monotonic()
            if proc.poll()is not None:
                best=receive(best)
                break
            remaining=deadline-time.monotonic()
            if remaining<=0:timed_out=True;proc.kill();proc.wait();break
            receiver.wait(min(.05,remaining))
    atomic_json(out.parent/'publication.json',audit())
    receiver.close()
    if direct and best and not first_recorded:
        first=dict(best,mapping_elapsed_seconds=best['observed_monotonic']-float(os.environ.get('EGM_MAPPING_STARTED_MONOTONIC',start)),constructor='existing native primitive/cut match and incremental delay Flow',external_mapper_called=False,source_bundle_used=False,timing_scope='first externally verified complete native cover, including supervisor lag')
        atomic_json(out.parent/'first-native-cover.json',first)
    if best:
        verified=verify_receipt(receipt,library,subject,native);shutil.copyfile(verified['netlist'],out)
        info.update(optimizer_succeeded=verified['kind']=='complete-generated-cover',selected='validated_online_incumbent'if verified['kind']=='complete-generated-cover'else'native_source_seed',incumbent=verified)
    info.update(search_finished=proc.returncode==0 and (best is not None or not direct),search_deadline_hit=timed_out,native_exit_code=proc.returncode,total_mapping_frontend_secs=time.monotonic()-start,selected_verilog_sha256=sha(out)if out.exists()else None)
    atomic_json(meta,info)
    print(json.dumps(dict(area=best['area']if best else seed['area']if seed else None,delay=best['delay']if best else seed['delay']if seed else None,optimizer_succeeded=info['optimizer_succeeded'],search_finished=info['search_finished'])))
if __name__=='__main__':main()
