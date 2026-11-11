#!/usr/bin/env python3
"""Paired three-part rerun from shared pre-normalized mapped inputs."""
import argparse,concurrent.futures,json,os,re,shutil,sys,traceback,time
from pathlib import Path
import run_epmapping_supplement as common
from run_three_parts import DESIGNS,ARMS
from run_buffer_core_validation import buffer_area
from analyze_epmapping_supplement import restore_abc_output_aliases
from audit_buffer_core import core_fingerprint
from audit_mapped_ppa import audit
from recover_checkpoint import recover
from report import read_counters
from mapped_shared_blif import convert

PROJECT=common.PROJECT
PREVIOUS=PROJECT/'results_v28_three_parts_query_budget'
FLOW=PROJECT/'results_epmapping_three_parts_buffer_v1_20261111'
ARMS=dict(ARMS,noBUFFER={'EPMAPPING_BUFFER_CLEANUP':'0'})

def proof(root,wd,netlist,design,lib,name):
    log=wd/(name+'.log')
    reader=f'read_blif -c {netlist}; short_names' if netlist.suffix=='.blif' else f'read -m {netlist}'
    rc,_=common.call([str(root/'inputs/abc'),'-c',f'read_genlib {lib}; {reader}; cec -n {root/"inputs/benchmarks"/(design+".aig")}'],dict(os.environ),log,3600)
    assert rc==0 and 'Networks are equivalent' in log.read_text(),('CEC',log,rc)
    return common.sha(log)

def native(root,m,stage,arm,design,mode,cache,flags):
    wd=root/stage/arm/f'{mode}.{design}';result=wd/'result.json'
    if result.exists():return json.loads(result.read_text())
    wd.mkdir(parents=True,exist_ok=True);f=root/'inputs'
    recipe='direct' if stage=='mfs3' else 'high_effort'
    rounds=1 if stage in ['mfs3','main_r1'] else 6
    restricted=stage in ['mfs3','strong_common'];lib=f/('asap7_6in.genlib' if restricted else 'asap7.genlib')
    bench=f/'benchmarks'/f'{design}.aig'
    options=dict(m['goal']);options.update(flags)
    shared=root/'normalized'/('direct6' if stage=='mfs3' else 'strong6' if restricted else 'strong47')/(design+'.blif')
    options.update(EPMAPPING_BASELINE=recipe,EPMAPPING_ROUNDS=str(rounds),MAP_EGG_PROFILE_CACHE_DIR=str(cache),EPMAPPING_PROFILE_RETAIN='vector',EPMAPPING_MAPPED_INPUT=str(shared),EPMAPPING_PRE_BUFFER_CLEANUP='1')
    if mode in ['cold','warm']:
        options['MAP_EGG_PROFILE_READONLY_DIR']=''  # Explicitly disable native auto-discovery.
    if mode in ['cold','prebuilt']:assert not cache.exists(),('nonempty fresh private pool',cache)
    else:assert cache.is_dir(),('missing warm pool',cache)
    cache.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','MAP_EGG_','EPM_'))}
    env.update(options,ABC=str(f/'abc'),RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1')
    command=[str(f/'engine'),str(lib),str(bench),str(wd)]
    common.save(wd/'command.json',dict(command=command,env=options,engine_sha256=m['engine_sha256'],input_sha256=common.sha(bench),library_sha256=common.sha(lib)))
    rc,wall=common.call(command,env,wd/'log',3600);text=(wd/'log').read_text()
    assert 'imported mapped input:' in text and 'pre_buffer_cleanup' in text,wd
    assert 'post_mapping_timeout=1000' in text and ('window_synth windows=0 applied=0' in text or rc==-14),wd
    initial=audit(wd/'epmapping_baseline.v',lib)
    prior=shared
    assert common.sha(prior.with_suffix('.v'))==common.sha(wd/'epmapping_baseline.v'),('changed shared normalized mapping',wd)
    r=dict(stage=stage,arm=arm,design=design,mode=mode,rc=rc,wall_seconds=wall,status='error',initial=initial,initial_sha256=common.sha(prior),initial_verilog_sha256=common.sha(prior.with_suffix('.v')),round_cap=rounds,recipe=recipe,library_sha256=common.sha(lib))
    if rc==0:
        assert '[i] cec=1' in text,wd
        selected=wd/'epmapping_mapped.v';secs=read_counters(str(wd/'log'))['time_optimizer']
        # Internal CEC serializes the same finalized graph; bind its exact bytes.
        cec_files=list(wd.glob('cec_*.v'));assert len(cec_files)==1 and common.sha(cec_files[0])==common.sha(selected),('CEC representation mismatch',wd)
        r.update(status='complete',cec=1,optimizer_seconds=secs)
    elif rc==-14:
        recovery=recover(wd,lib,bench,f/'abc',rc);selected=wd/recovery['checkpoint']['netlist']
        r.update(status='timeout',cec=1,optimizer_seconds=1000,selection=recovery['selection'])
    else:
        common.save(result,r);raise RuntimeError((wd,rc))
    final=audit(selected,lib)
    assert final['area']<=initial['area']+.0001 and final['delay']<=initial['delay']+.001,(wd,initial,final)
    assert final['serialized_area']<=initial['serialized_area']+.0001 and final['serialized_delay']<=initial['serialized_delay']+.001,(wd,'serialized regression')
    assert r['optimizer_seconds']<=1000.1,(wd,r['optimizer_seconds'])
    initial_buffers,initial_counts=buffer_area(wd/'epmapping_baseline.v',lib);final_buffers,final_counts=buffer_area(selected,lib)
    cleanup=re.search(r'buffer_cleanup area_removed=([0-9.e+-]+) delay_removed=([0-9.e+-]+) seconds=([0-9.e+-]+)',text)
    r.update(final=final,selected_netlist=str(selected.relative_to(root)),selected_sha256=common.sha(selected),initial_buffer_area=initial_buffers,final_buffer_area=final_buffers,initial_buffer_counts=initial_counts,final_buffer_counts=final_counts,buffer_gain=initial_buffers-final_buffers,core_gain=(initial['area']-initial_buffers)-(final['area']-final_buffers),cleanup_seconds=float(cleanup[3]) if cleanup else None,profile_stats=re.findall(r'egraph_profile (.*)',text),cache_files=sum(1 for _ in cache.rglob('*.json')))
    common.save(result,r)
    print(json.dumps({k:r[k] for k in ['stage','arm','design','mode','status','optimizer_seconds','buffer_gain','core_gain']}),flush=True)
    return r

def audit_abc_initial(baseline,lib,wd):
    # Independent float32 propagation over ABC's exported, alias-restored start.
    cohort=baseline.parent.name;design=baseline.stem
    initial=baseline.parents[2]/'normalization_abc_blif'/cohort/design/'import_aliases.v'
    return common.audit_abc(initial,lib,float32=True)

def abc(root,m,stage,design,mode,baseline):
    wd=root/stage/'abc'/f'{mode}.{design}';result=wd/'result.json'
    if result.exists():return json.loads(result.read_text())
    wd.mkdir(parents=True,exist_ok=True);f=root/'inputs';lib=f/'asap7_6in.genlib';env=dict(os.environ,RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1')
    cmd=lambda s:[str(f/'abc'),'-c',s]
    initial=audit(baseline.with_suffix('.v'),lib)
    script=f'read_genlib {lib}; read_blif -c {baseline}; short_names; print_stats; mfs3 -a; print_stats; write_verilog {wd/"mapped.v"}'
    common.save(wd/'command.json',dict(script=script,input_sha256=common.sha(baseline),library_sha256=common.sha(lib),timeout=1000))
    rc,secs=common.call(cmd(script),env,wd/'log',1000)
    r=dict(stage=stage,arm='abc',design=design,mode=mode,rc=rc,status='timeout' if rc==124 else 'complete',optimizer_seconds=min(secs,1000.) if rc==124 else secs,initial=initial,initial_sha256=common.sha(baseline),initial_verilog_sha256=common.sha(baseline.with_suffix('.v')),cec=1)
    if rc==0:
        rawproof=proof(root,wd,wd/'mapped.v',design,lib,'raw_cec')
        text,replaced=restore_abc_output_aliases((wd/'mapped.v').read_text(),lib)
        normalized=wd/'mapped_native_aliases.v';normalized.write_text(text)
        measured=common.audit_abc(normalized,lib);single=common.audit_abc(normalized,lib,float32=True)
        log=(wd/'log').read_text();reported={k:float(re.findall(r'\b'+k+r'\s*=\s*([0-9.]+)',log)[-1]) for k in ['area','delay']}
        native_initial={k:float(re.findall(r'\b'+k+r'\s*=\s*([0-9.]+)',log)[0]) for k in ['area','delay']}
        assert abs(native_initial['area']-initial['area'])<=.005001 and abs(native_initial['delay']-audit_abc_initial(baseline,lib,wd)['delay'])<=.005001,('ABC imported a different initial mapping',wd,native_initial,initial)
        r['native_initial']=native_initial
        assert abs(measured['area']-reported['area'])<=.005001 and abs(single['delay']-reported['delay'])<=.005001,('ABC native audit',wd,measured,single,reported)
        cecproof=proof(root,wd,normalized,design,lib,'selected_native_cec')
        legal=measured['area']<=initial['area']+.0001 and measured['delay']<=initial['delay']+.001
        r.update(raw=measured,native_reported=reported,native_float32=single,aliases=replaced,raw_cec_sha256=rawproof,native_cec_sha256=cecproof,guard_selected='optimized' if legal else 'initial',delay_violation=measured['delay']>initial['delay']+.001)
        selected=normalized if legal else baseline.with_suffix('.v');final=measured if legal else initial
    else:
        assert rc==124,(wd,rc)
        selected=baseline.with_suffix('.v');final=initial;r.update(guard_selected='initial')
    if selected==baseline.with_suffix('.v'):r['initial_cec_sha256']=proof(root,wd,baseline,design,lib,'selected_initial_cec')
    b0,c0=buffer_area(baseline.with_suffix('.v'),lib);bf,cf=buffer_area(selected,lib)
    r.update(final=final,selected_netlist=str(selected.relative_to(root)),selected_sha256=common.sha(selected),initial_buffer_area=b0,final_buffer_area=bf,buffer_gain=b0-bf,core_gain=(initial['area']-b0)-(final['area']-bf))
    common.save(result,r);print(json.dumps({k:r[k] for k in ['stage','arm','design','mode','status','optimizer_seconds','guard_selected']}),flush=True);return r

def initialize(root,binary,jobs):
    m=common.initialize(root,binary)
    if 'three_part_plan' in m:
        assert m['three_part_plan']['jobs']==jobs
        return m
    m['goal'].update(EPMAPPING_BUFFER_CLEANUP='1',EPMAPPING_PRE_BUFFER_CLEANUP='1')
    # Freeze the new runner, auditors and all native implementation headers.
    sources=root/'inputs/sources'
    for name in ['run_buffer_three_parts.py','run_prebuffer_three_parts.py','mapped_shared_blif.py','verify_prebuffer_imports.py','analyze_prebuffer_three_parts.py','run_three_parts.py','run_buffer_core_validation.py','audit_buffer_core.py','analyze_epmapping_supplement.py']:
        shutil.copy2(PROJECT/'scripts/experiments'/name,sources/name)
    for p in (PROJECT/'tools/patches/mockturtle/overlay').rglob('*'):
        if p.is_file():
            dest=sources/'mockturtle_overlay'/p.relative_to(PROJECT/'tools/patches/mockturtle/overlay');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for name in ['Cargo.toml','Cargo.lock','build.rs','CMakeLists.txt']:
        source=PROJECT/'map_egg'/name
        if source.exists():shutil.copy2(source,sources/('map_egg_'+name))
    m.update(three_part_plan=dict(jobs=jobs,designs=DESIGNS,arms=ARMS,stages=['mfs3','main_r1','main','ablation','strong_common'],expected_method_runs=768,expected_abc_runs=64,restricted_stages=['mfs3','strong_common'],full_library_stages=['main_r1','main','ablation'],abc_modes='fresh independently timed processes, no e-graph cache'),input_sha256={str(p.relative_to(root/'inputs')):common.sha(p) for p in (root/'inputs').rglob('*') if p.is_file()})
    common.save(root/'manifest.json',m);return m

def prepare_inputs(root,m,jobs):
    if (root/'normalized/manifest.json').exists():
        saved=json.loads((root/'normalized/manifest.json').read_text())
        for row in saved['records']:assert common.sha(root/row['netlist'])==row['normalized_sha256']
        return saved
    f=root/'inputs';records=[]
    def one(spec):
        cohort,design=spec;restricted=cohort!='strong47'
        stage='mfs3' if cohort=='direct6' else 'strong_common' if restricted else 'main'
        source=FLOW/stage/'ours'/f'cold.{design}/epmapping_baseline.v'
        reference=json.loads((FLOW/stage/'ours'/f'cold.{design}/result.json').read_text())
        assert common.sha(source)==reference['initial_sha256']
        wd=root/'normalization'/cohort/design;wd.mkdir(parents=True,exist_ok=True)
        lib=f/('asap7_6in.genlib' if restricted else 'asap7.genlib');bench=f/'benchmarks'/f'{design}.aig'
        env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','MAP_EGG_','EPM_'))}
        env.update(ABC=str(f/'abc'),EPMAPPING_MAPPED_INPUT=str(source),EPMAPPING_PRE_BUFFER_CLEANUP='1',EPMAPPING_SKIP='1',EPMAPPING_SUPER='',RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1')
        command=[str(f/'engine'),str(lib),str(bench),str(wd)]
        common.save(wd/'command.json',dict(command=command,env={k:v for k,v in env.items() if k.startswith(('EPMAPPING_','MAP_EGG_'))}))
        rc,secs=common.call(command,env,wd/'log',3600);assert rc==0,(wd,rc)
        net=root/'normalized'/cohort/(design+'.v');net.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(wd/'epmapping_mapped.v',net)
        shared=net.with_suffix('.blif');convert(net,lib,shared)
        env['EPMAPPING_MAPPED_INPUT']=str(shared)
        reimport=wd/'reimport';rr,_=common.call([str(f/'engine'),str(lib),str(bench),str(reimport)],env,wd/'reimport.log',3600);assert rr==0
        assert common.sha(reimport/'epmapping_mapped.v')==common.sha(net),('shared BLIF import changed normalized mapping',design)
        initial=audit(net,lib);b,_=buffer_area(net,lib);assert abs(b)<.0001 and initial['area']<=reference['initial']['area']+.0001 and initial['delay']<=reference['initial']['delay']+.001
        cec=proof(root,wd,net,design,lib,'normalized_cec')
        row=dict(cohort=cohort,design=design,source=str(source),source_sha256=common.sha(source),netlist=str(net.relative_to(root)),normalized_sha256=common.sha(net),shared_blif=str(shared.relative_to(root)),shared_blif_sha256=common.sha(shared),native_import_identical=True,original=reference['initial'],normalized=initial,buffer_area=0,cec=1,cec_sha256=cec,normalization_process_seconds=secs)
        print(json.dumps(dict(normalized=cohort,design=design,area=initial['area'],delay=initial['delay'])),flush=True)
        return row
    specs=[(cohort,d) for cohort in ['direct6','strong6','strong47'] for d in DESIGNS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:records=list(pool.map(one,specs))
    out=dict(records=records,normalization='identity BUF bypass only; original AIG and preprocessing unchanged; final fixed delay bound is normalized delay',budget='common input preparation outside both 1000s post-mapping limits',source_campaign=FLOW.name)
    common.save(root/'normalized/manifest.json',out);return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--jobs',type=int,default=16);ap.add_argument('--binary',type=Path,default=PROJECT/'build/epmapping/epmapping_check');a=ap.parse_args()
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True);m=initialize(root,a.binary,a.jobs)
    prepare_inputs(root,m,min(a.jobs,8))
    if not (root/'abc_import_proofs.json').exists():
        from verify_prebuffer_imports import verify
        verify(root)
    specs=[(stage,'ours',d,{}) for stage in ['main','mfs3','strong_common','main_r1'] for d in DESIGNS]
    specs.extend(('ablation',arm,d,flags) for d in DESIGNS for arm,flags in ARMS.items())
    def pair(spec):
        stage,arm,design,flags=spec;cache=root/'cache'/stage/arm/design
        for mode in ['cold','warm']:
            native(root,m,stage,arm,design,mode,cache,flags)
            if stage in ['mfs3','strong_common']:abc(root,m,stage,design,mode,root/'normalized'/('direct6' if stage=='mfs3' else 'strong6')/(design+'.blif'))
    errors=[];complete=0;start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures={pool.submit(pair,s):s[:3] for s in specs}
        for future in concurrent.futures.as_completed(futures):
            try:future.result();complete+=1
            except Exception as error:errors.append((futures[future],repr(error)));traceback.print_exc()
            common.save(root/'execution_status.json',dict(complete=complete==len(specs) and not errors,pairs_finished=complete,pairs_expected=len(specs),errors=errors,elapsed_seconds=time.monotonic()-start))
    assert not errors,errors
if __name__=='__main__':main()
