#!/usr/bin/env python3
"""Frozen three-part rerun, paired caches, identity-buffer ablation and core accounting."""
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

PROJECT=common.PROJECT
PREVIOUS=PROJECT/'results_v28_three_parts_query_budget'
ARMS=dict(ARMS,noBUFFER={'EPMAPPING_BUFFER_CLEANUP':'0'})

def proof(root,wd,netlist,design,lib,name):
    log=wd/(name+'.log')
    rc,_=common.call([str(root/'inputs/abc'),'-c',f'read_genlib {lib}; read -m {netlist}; cec -n {root/"inputs/benchmarks"/(design+".aig")}'],dict(os.environ),log,3600)
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
    options.update(EPMAPPING_BASELINE=recipe,EPMAPPING_ROUNDS=str(rounds),MAP_EGG_PROFILE_CACHE_DIR=str(cache),EPMAPPING_PROFILE_RETAIN='vector')
    if mode=='cold':assert not cache.exists(),('nonempty cold pool',cache)
    else:assert cache.is_dir(),('missing warm pool',cache)
    cache.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','MAP_EGG_','EPM_'))}
    env.update(options,ABC=str(f/'abc'),RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1')
    command=[str(f/'engine'),str(lib),str(bench),str(wd)]
    common.save(wd/'command.json',dict(command=command,env=options,engine_sha256=m['engine_sha256'],input_sha256=common.sha(bench),library_sha256=common.sha(lib)))
    rc,wall=common.call(command,env,wd/'log',3600);text=(wd/'log').read_text()
    assert f'ABC baseline: {recipe}' in text,wd
    if recipe=='high_effort':assert 'ands (no reduction); fixpoint after' in text and 'stopped at the iteration cap' not in text,wd
    assert 'post_mapping_timeout=1000' in text and ('window_synth windows=0 applied=0' in text or rc==-14),wd
    initial=audit(wd/'epmapping_baseline.v',lib)
    if stage=='mfs3':prior=PREVIOUS/'cold/tab_mfs3'/f'ours.{design}/epmapping_baseline.v'
    elif stage=='strong_common':prior=PROJECT/'results_epmapping_supplement_v1_20261110/common/ours'/f'cold.{design}/epmapping_baseline.v'
    else:prior=PREVIOUS/'cold/tab_main'/f'ae.{design}/epmapping_baseline.v'
    assert common.sha(prior)==common.sha(wd/'epmapping_baseline.v'),('changed original initial mapping',wd)
    r=dict(stage=stage,arm=arm,design=design,mode=mode,rc=rc,wall_seconds=wall,status='error',initial=initial,initial_sha256=common.sha(prior),round_cap=rounds,recipe=recipe,library_sha256=common.sha(lib))
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

def abc(root,m,stage,design,mode,baseline):
    wd=root/stage/'abc'/f'{mode}.{design}';result=wd/'result.json'
    if result.exists():return json.loads(result.read_text())
    wd.mkdir(parents=True,exist_ok=True);f=root/'inputs';lib=f/'asap7_6in.genlib';env=dict(os.environ,RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1')
    cmd=lambda s:[str(f/'abc'),'-c',s]
    initial=audit(baseline,lib)
    script=f'read_genlib {lib}; read -m {baseline}; mfs3 -a; print_stats; write_verilog {wd/"mapped.v"}'
    common.save(wd/'command.json',dict(script=script,input_sha256=common.sha(baseline),library_sha256=common.sha(lib),timeout=1000))
    rc,secs=common.call(cmd(script),env,wd/'log',1000)
    r=dict(stage=stage,arm='abc',design=design,mode=mode,rc=rc,status='timeout' if rc==124 else 'complete',optimizer_seconds=min(secs,1000.) if rc==124 else secs,initial=initial,initial_sha256=common.sha(baseline),cec=1)
    if rc==0:
        rawproof=proof(root,wd,wd/'mapped.v',design,lib,'raw_cec')
        text,replaced=restore_abc_output_aliases((wd/'mapped.v').read_text(),lib)
        normalized=wd/'mapped_native_aliases.v';normalized.write_text(text)
        measured=common.audit_abc(normalized,lib);single=common.audit_abc(normalized,lib,float32=True)
        log=(wd/'log').read_text();reported={k:float(re.findall(r'\b'+k+r'\s*=\s*([0-9.]+)',log)[-1]) for k in ['area','delay']}
        assert abs(measured['area']-reported['area'])<=.005001 and abs(single['delay']-reported['delay'])<=.005001,('ABC native audit',wd,measured,single,reported)
        cecproof=proof(root,wd,normalized,design,lib,'selected_native_cec')
        legal=measured['area']<=initial['area']+.0001 and measured['delay']<=initial['delay']+.001
        r.update(raw=measured,native_reported=reported,native_float32=single,aliases=replaced,raw_cec_sha256=rawproof,native_cec_sha256=cecproof,guard_selected='optimized' if legal else 'initial',delay_violation=measured['delay']>initial['delay']+.001)
        selected=normalized if legal else baseline;final=measured if legal else initial
    else:
        assert rc==124,(wd,rc)
        selected=baseline;final=initial;r.update(guard_selected='initial')
    if selected==baseline:r['initial_cec_sha256']=proof(root,wd,baseline,design,lib,'selected_initial_cec')
    b0,c0=buffer_area(baseline,lib);bf,cf=buffer_area(selected,lib)
    r.update(final=final,selected_netlist=str(selected.relative_to(root)),selected_sha256=common.sha(selected),initial_buffer_area=b0,final_buffer_area=bf,buffer_gain=b0-bf,core_gain=(initial['area']-b0)-(final['area']-bf))
    common.save(result,r);print(json.dumps({k:r[k] for k in ['stage','arm','design','mode','status','optimizer_seconds','guard_selected']}),flush=True);return r

def initialize(root,binary,jobs):
    m=common.initialize(root,binary)
    if 'three_part_plan' in m:
        assert m['three_part_plan']['jobs']==jobs
        return m
    m['goal']['EPMAPPING_BUFFER_CLEANUP']='1'
    # Freeze the new runner, auditors and all native implementation headers.
    sources=root/'inputs/sources'
    for name in ['run_buffer_three_parts.py','run_three_parts.py','run_buffer_core_validation.py','audit_buffer_core.py','analyze_epmapping_supplement.py']:
        shutil.copy2(PROJECT/'scripts/experiments'/name,sources/name)
    for p in (PROJECT/'tools/patches/mockturtle/overlay').rglob('*'):
        if p.is_file():
            dest=sources/'mockturtle_overlay'/p.relative_to(PROJECT/'tools/patches/mockturtle/overlay');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for name in ['Cargo.toml','Cargo.lock','build.rs','CMakeLists.txt']:
        source=PROJECT/'map_egg'/name
        if source.exists():shutil.copy2(source,sources/('map_egg_'+name))
    m.update(three_part_plan=dict(jobs=jobs,designs=DESIGNS,arms=ARMS,stages=['mfs3','main_r1','main','ablation','strong_common'],expected_method_runs=768,expected_abc_runs=64,restricted_stages=['mfs3','strong_common'],full_library_stages=['main_r1','main','ablation'],abc_modes='fresh independently timed processes, no e-graph cache'),input_sha256={str(p.relative_to(root/'inputs')):common.sha(p) for p in (root/'inputs').rglob('*') if p.is_file()})
    common.save(root/'manifest.json',m);return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--jobs',type=int,default=16);ap.add_argument('--binary',type=Path,default=PROJECT/'build/epmapping/epmapping_check');a=ap.parse_args()
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True);m=initialize(root,a.binary,a.jobs)
    specs=[(stage,'ours',d,{}) for stage in ['main','mfs3','strong_common','main_r1'] for d in DESIGNS]
    specs.extend(('ablation',arm,d,flags) for d in DESIGNS for arm,flags in ARMS.items())
    def pair(spec):
        stage,arm,design,flags=spec;cache=root/'cache'/stage/arm/design
        for mode in ['cold','warm']:
            native(root,m,stage,arm,design,mode,cache,flags)
            if stage in ['mfs3','strong_common']:abc(root,m,stage,design,mode,root/stage/arm/f'{mode}.{design}/epmapping_baseline.v')
    errors=[];complete=0;start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures={pool.submit(pair,s):s[:3] for s in specs}
        for future in concurrent.futures.as_completed(futures):
            try:future.result();complete+=1
            except Exception as error:errors.append((futures[future],repr(error)));traceback.print_exc()
            common.save(root/'execution_status.json',dict(complete=complete==len(specs) and not errors,pairs_finished=complete,pairs_expected=len(specs),errors=errors,elapsed_seconds=time.monotonic()-start))
    assert not errors,errors
if __name__=='__main__':main()
