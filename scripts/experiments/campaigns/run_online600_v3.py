"""Explicit online600-v3 prototype cohort; committed pre-deadline covers only, independent CEC outside clock."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from benchmark_suites import EPFL

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mapping_process(argv,env,seconds=600):
    start=time.monotonic();env=dict(env,EGM_MAPPING_DEADLINE_MONOTONIC=str(start+seconds))
    p=subprocess.Popen(argv,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    timed_out=False
    try:out,err=p.communicate(timeout=max(0,start+seconds-time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(p.pid,signal.SIGKILL);out,err=p.communicate()
    return dict(exit_code=p.returncode,timeout=timed_out,wall_secs=time.monotonic()-start,started_monotonic=start,deadline_monotonic=start+seconds,stdout=out,stderr=err)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--research-root',type=Path,default=ROOT/'results_v2/online_mapping_20261022/source_v3')
    ap.add_argument('--designs',nargs='+',default=EPFL)
    ap.add_argument('--profile',type=int,choices=[0,1,3],default=1)
    ap.add_argument('--area-recovery',type=int,choices=[0,1,2],default=2)
    ap.add_argument('--max-jobs',type=int,default=1)
    a=ap.parse_args();assert set(a.designs)<=set(EPFL) and len(a.designs)==len(set(a.designs)) and a.max_jobs>0
    output=a.output.resolve();output.mkdir(exist_ok=True,parents=True)
    locks=[]
    for f in [output/'runner.lock',ROOT/'results_v2/independent_components_20261019/global.lock',ROOT/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock']:
        h=f.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
    policy_path=ROOT/'emapper/EXPERIMENT_POLICY.json';policy=json.loads(policy_path.read_text())
    assert policy['mapping_seconds_per_design_method']==600 and all(policy[k] is None for k in ['node_limit','retained_candidate_limit','cut_count_limit','iteration_limit','native_subguard_seconds','stage_time_limits','address_space_limit_bytes'])
    source=a.research_root.resolve();native=source/'target/release/bench_rs_egraph_wide'
    prior=ROOT/'results_v2/critical_compact_20261022/full_epfl/campaign.json'
    study=json.loads(prior.read_text());subjects=ROOT/'results_v2/epfl_resyn2_subjects_20261012/report.json'
    files=[policy_path,Path(__file__),HERE/'egraphmap_online600.py',HERE/'egraphmap_time_only.py',HERE/'source_cover.py',HERE/'fast_original_cec.py',native,source/'Cargo.toml',source/'Cargo.lock',source/'main.rs',prior,subjects,*list((source/'emapper/src').rglob('*.rs'))]
    files += [source/'emapper/Cargo.toml', ROOT/'results_v2/single_output_wide_20261021/source_kernels/target/release/bench_rs_wide_measure']
    for bench in a.designs:
        job=next(j for j in study['jobs'] if j['bench']==bench and j['profile']==1)
        report=Path(job['report']);row=json.loads(report.read_text())['results'][0]
        lib,prepared=map(Path,row['scalability']['attempts'][0]['argv'][1:3])
        files += [report,lib,prepared,lib.parent/'library.json',lib.parent/'cells.v',ROOT/f'tools/mockturtle/experiments/benchmarks/{bench}.aig']
    configuration=dict(policy='online600-v3',prototype_stage='semi-naive cut pairs, incremental matching, early legal delay incumbent; optional fixed-bound shared-area recovery with unselected-input witness; complete incumbent protected; deferred materialization not yet enabled',area_recovery=a.area_recovery,complete_predeadline_incumbent_may_survive_search_timeout=True,designs=list(a.designs),profile=a.profile,mapping_seconds=600,
        node_limit=None,retained_candidate_limit=None,cut_count_limit=None,iteration_limit=None,
        native_subguard=None,stage_time_limits=None,address_space_limit=None,CEC_in_mapping_clock=False,
        source_construction_in_mapping_clock=True,baseline_algorithms_changed=False,default_main_mapper_merged=False)
    frozen={str(f.resolve()):sha(f)for f in files}
    cp=output/'campaign.json';state=json.loads(cp.read_text())if cp.exists()else dict(configuration=configuration,fingerprint=frozen,expected_jobs=len(a.designs),jobs=[],status='checkpointed')
    assert state['configuration']==configuration and state['fingerprint']==frozen,'frozen policy/source drift'
    def save():
        tmp=cp.with_suffix('.pending');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(cp)
    for j in state['jobs']:
        assert all(sha(f)==h for f,h in j['sha256'].items())
    done={j['bench']for j in state['jobs']};count=0
    for bench in a.designs:
        if bench in done:continue
        old_job=next(j for j in study['jobs']if j['bench']==bench and j['profile']==1)
        old_path=Path(old_job['report']);assert sha(old_path)==old_job['sha256'][str(old_path)]
        old=json.loads(old_path.read_text())['results'][0];attempt=old['scalability']['attempts'][0]
        lib,prepared=map(Path,attempt['argv'][1:3]);original=ROOT/f'tools/mockturtle/experiments/benchmarks/{bench}.aig'
        assert sha(original)==old['input_sha256'] and sha(prepared)==old['prepared_subject_sha256']
        folder=output/bench;folder.mkdir(exist_ok=False);raw=folder/'raw.v'
        env={k:v for k,v in os.environ.items()if not k.startswith(('EGM_','ESYN_'))}
        env.update(old['settings']);env.update(study['configuration']['fixed_native_configuration'])
        maximum=str(2**64-1)
        env.update(EGM_ONLINE_AREA_RECOVERY=str(a.area_recovery),EGM_NATIVE_OPTIMIZER=str(native),EGM_ONLINE_MAPPING='1',EGM_TOTAL_TIME_ONLY='1',EGM_PENDING_FRONTIER=str(a.profile&1),EGM_FOLDED_PIN_MATCH=str((a.profile>>1)&1),EGM_CUT_STAGE_SECS='0',EGM_NODES=maximum,EGM_INSTANCES=maximum,EGM_CUTS=maximum,EGM_CUT_ROUNDS=maximum,EGM_ITER=maximum,EGM_REFINE_ROUNDS=maximum,EGM_REFINE_TRIALS=maximum,EGM_REFINE_TIME='600',EGM_RETURN_RESERVE='0')
        env.pop('EGM_DIAG_PATH',None)
        argv=[sys.executable,str(HERE/'egraphmap_online600.py'),str(lib),str(prepared),str(raw)]
        command=dict(argv=argv,environment={k:v for k,v in env.items()if k.startswith('EGM_')},input_sha256=sha(original),prepared_sha256=sha(prepared),library_sha256=sha(lib),library_models_sha256=sha(lib.parent/'cells.v'),old_report=str(old_path),old_report_sha256=sha(old_path))
        (folder/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        state['current_job']=bench;state['status']='running';save()
        result=mapping_process(argv,env)
        (folder/'mapping.stdout').write_text(result.pop('stdout'));(folder/'mapping.stderr').write_text(result.pop('stderr'))
        row=dict(bench=bench,policy=configuration,mapping=result,input_sha256=sha(original),prepared_sha256=sha(prepared),status='failed')
        meta=raw.with_suffix('.complete.json')
        if meta.exists():row['native']=json.loads(meta.read_text())
        receipt=raw.with_suffix('.incumbent.receipt.json')
        if receipt.exists():
            from egraphmap_online600 import verify_receipt
            committed=verify_receipt(receipt,lib,prepared,native)
            assert Path(committed['netlist']).resolve().is_relative_to((folder/'incumbents').resolve())
            assert committed['deadline_monotonic']==result['deadline_monotonic']
            assert committed['observed_monotonic']<result['deadline_monotonic']
            # Pure post-run promotion of a PRE-deadline parent-selected transaction.
            # No new mapping, candidate search or selection happens after deadline.
            import shutil
            shutil.copyfile(committed['netlist'],raw)
            row['native'].update(optimizer_succeeded=True,selected='validated_online_incumbent',incumbent=committed,search_deadline_hit=result['timeout'],search_finished=not result['timeout'] and result['exit_code']==0)
            row['incumbent_receipt']=str(receipt)
        if raw.exists():
            netjson=folder/'mapped.json';normalized=folder/'mapped.v'
            models=lib.parent/'cells.v'
            script=f'read_verilog -lib "{models}"; read_verilog "{raw}"; hierarchy -check -auto-top; rename -top mapped; write_verilog -noattr -noexpr "{normalized}"; write_json "{netjson}"'
            normalization=subprocess.run(['yosys','-Q','-T','-p',script],capture_output=True,text=True)
            (folder/'normalize.log').write_text(normalization.stdout+normalization.stderr)
            if normalization.returncode==0:
                from fast_original_cec import verify
                evaluator=ROOT/'results_v2/single_output_wide_20261021/source_kernels/target/release/bench_rs_wide_measure'
                measure=subprocess.run([str(evaluator),str(lib),str(netjson)],capture_output=True,text=True)
                (folder/'measure.log').write_text(measure.stdout+measure.stderr)
                if measure.returncode==0:
                    row['measured']=json.loads(measure.stdout);t=time.monotonic()
                    row['measured']['cec']=int(verify(models,original,netjson,folder/'functional-cec'))
                    row['status']=('ok'if row['measured']['cec']==1 and row.get('native',{}).get('optimizer_succeeded',False)else'mapping_timeout'if result['timeout']else'source_only'if row['measured']['cec']==1 else'failed')
        (folder/'report.json').write_text(json.dumps(row,indent=2)+'\n')
        artifacts=list(folder.rglob('*'));entry=dict(bench=bench,report=str(folder/'report.json'),status=row['status'],sha256={str(f):sha(f)for f in artifacts if f.is_file()})
        state['jobs'].append(entry);state.pop('current_job');state['status']='attempted'if len(state['jobs'])==state['expected_jobs']else'checkpointed';save()
        print(bench,row['status'],row.get('native',{}).get('optimizer_succeeded'),flush=True)
        count+=1
        if count>=a.max_jobs:break
if __name__=='__main__':main()
