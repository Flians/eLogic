"""After audited full20: capture full admitted pools, compare fixed-origin trials.

Test-only diagnostic wiring on the one local main; no production optimization.
Each capture and each differential diagnostic has its own disclosed 600s clock.
Their results are NOT added together or used as a 1200s main mapper result.
"""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

R=pathlib.Path(__file__).resolve().parents[3]
S=R/'results_v2/emapper_dac_convergence_20261111'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)


def inventory(path):
    f=read(path);assert f['schema']==1 and f['selected_context'] is not None
    n=len(f['free']);assert len(f['chosen'])==len(f['constants'])==n
    assert len(f['bank'])*2==n
    sources=targets=unproved_targets=0
    for i,m in enumerate(f['pool']):
        assert len(m['inputs'])==len(m['input_pins'])
        assert all(isinstance(s,int) and 0<=s<n for s in m['inputs'])
        for o in m['outputs']:
            assert len(o['delays'])==len(m['inputs'])
            assert all(0<=s<n for s in o['targets'])
            targets+=len(o['targets']);sources+=len(o['sources'])
            unproved_targets+=sum(not any(p['target']==s for p in o['sources']) for s in o['targets'])
    for b in [*f['chosen'],*(f['witness'] or [])]:
        if b is not None:assert 0<=b['instance']<len(f['pool']) and 0<=b['pin']<len(f['pool'][b['instance']]['outputs'])
    return dict(snapshot_sha256=sha(path),pool_instances=len(f['pool']),targets=targets,
                recorded_binding_sources=sources,targets_without_explicit_source=unproved_targets,
                unproved_target_note='inventory only; generator usability requires Rust certificate, not just source presence',
                cut_count=sum(len(v)for v in f['bank']),initial_AD=f['initial_ad'],context=f['context'],
                pool_sha256=hashlib.sha256(json.dumps(f['pool'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                chosen_sha256=hashlib.sha256(json.dumps(f['chosen'],separators=(',',':')).encode()).hexdigest(),
                complete_admitted_pool_at_capture=True,all_future_candidates_enumerated=False,
                producer_cursor_serialized=False,pending_work_note='source/logic and cut bank captured; pending production cursors are not serialised; absence means absent at this epoch only')


def summarize_events(path):
    result=dict(complete_rows=0,physical_unavailable=0,source_blocked_improvements=0,
                verified_missed_instances=[],false_metric_acceptances=0,truncated_tail_bytes=0)
    if not path.exists():return result
    with path.open('rb')as stream:
        for line in stream:
            try:row=json.loads(line)
            except (ValueError,UnicodeDecodeError):
                assert not line.endswith(b'\n'),'corrupt complete diagnostic record'
                result['truncated_tail_bytes']=len(line);break
            result['complete_rows']+=1
            if 'physical_unavailable'in row:result['physical_unavailable']+=1;continue
            if row.get('oracle_area_improves_under_initial_D')and row.get('source_error'):result['source_blocked_improvements']+=1
            if row.get('verified_missed_proposal'):result['verified_missed_instances'].append(row['instance'])
            if row.get('production_accept')and not row.get('oracle_area_improves_under_initial_D'):result['false_metric_acceptances']+=1
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=pathlib.Path,required=True)
    ap.add_argument('--max-wait-seconds',type=float,default=28800);a=ap.parse_args()
    assert not a.output.exists();a.output.mkdir(parents=True)
    state=dict(stage='waiting_for_full_audit_and_quality_report',cases={},mapping_seconds=600,
               baseline_runs=0,GitHub_push=False,production_algorithm_changed=False)
    def checkpoint():save(a.output/'state.json',state)
    checkpoint();locks=[]
    with(S/'fixed-pool-diagnosis.lock').open('a')as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB);until=time.monotonic()+a.max_wait_seconds
        while True:
            audit=S/'full-epfl-final-audit/validation.json';analysis=S/'native-seed-quality-gap/validation.json'
            if audit.exists()and analysis.exists():
                assert read(audit)['completed']==20 and not read(audit)['invalid_artifacts']
                assert read(analysis)['current_full_artifact_audit_passed']
                try:
                    for path in [R/'results_v2/independent_components_20261019/global.lock',S/'full-epfl/runner.lock']:
                        f=path.open('a');locks.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    for f in locks:f.close()
                    locks.clear()
            if time.monotonic()>=until:state['stage']='monitor_expired_no_changes';checkpoint();return
            time.sleep(15)
        try:
            sys.path.insert(0,str(R/'scripts/experiments/campaigns'));import run_dac_convergence as core
            c=read(S/'full-epfl/campaign.json');assert core.fingerprint()==c['frozen'],'source drift; inspect before wiring'
            target=R/'emapper/src/egraph_recovery_snapshot.rs';added=R/'emapper/src/egraph_fixed_pool_area_diagnostic.rs'
            template=R/'scripts/experiments/diagnostics/fixed_pool_area.rs.in';assert not added.exists()
            old=target.read_bytes();(a.output/'snapshot-module-before.rs').write_bytes(old)
            added.write_bytes(template.read_bytes());target.write_bytes(old+b'\n#[cfg(test)]\n#[path="egraph_fixed_pool_area_diagnostic.rs"]\nmod fixed_pool_area_diagnostic;\n')
            state.update(stage='test_only_wiring_and_build',source_before_sha256=hashlib.sha256(old).hexdigest(),template_sha256=sha(template));checkpoint()
            env={k:v for k,v in os.environ.items()if not k.startswith('EGM_')}
            fixture=a.output/'fixture';fixture.mkdir()
            for name in ['asap7_full.genlib','library.json','cells.v']:shutil.copy2(R/'emapper/libraries/asap7'/name,fixture/name)
            env['EGM_TEST_DATA_ROOT']=str(fixture.resolve())
            commands=[('release-build',['bash','scripts/emapper/build.sh','--offline']),
                      ('rust-regressions',['cargo','test','--locked','--offline','-p','emapper','egraph','-j','2','--','--test-threads=1']),
                      ('test-executable',['cargo','test','--locked','--offline','-p','emapper','--lib','--no-run','--message-format=json','-j','2'])]
            for label,cmd in commands:
                with(a.output/(label+'.log')).open('x')as log:p=subprocess.run(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
                assert p.returncode==0,label+' failed; preserve source/log and inspect'
            artifacts=[]
            for line in(a.output/'test-executable.log').read_text().splitlines():
                try:item=json.loads(line)
                except ValueError:continue
                if item.get('reason')=='compiler-artifact'and item.get('executable'):artifacts.append(item['executable'])
            assert len(artifacts)==1;test_binary=pathlib.Path(artifacts[0])
            state['test_binary_sha256']=sha(test_binary);state['native_sha256']=sha(R/'target/release/emapper_kernel')
            git=['git','-c','core.hooksPath=/dev/null','-c','gc.auto=0','-c','maintenance.auto=false','-c','filter.lfs.process=','-c','filter.lfs.required=false']
            owned=[str(target.relative_to(R)),str(added.relative_to(R))]
            subprocess.run(git+['add','--',*owned],cwd=R,check=True)
            subprocess.run(git+['commit','--only','-m','Add test-only fixed-origin area recovery differential','--',*owned],cwd=R,check=True)
            state['local_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();checkpoint()
            sys.path.insert(0,str(R/'scripts/emapper'))
            from physical import measure_cover
            from verify import verify
            profile=read(R/'emapper/profiles/main-off.json');library=R/profile['library']
            for name in ['adder','priority']:
                case=a.output/name;case.mkdir();capture=case/'capture';subject=read(S/'benchmark/campaign.json')['subjects'][name]
                row=dict(stage='capturing_admitted_pool',status='running');state['cases'][name]=row;checkpoint()
                try:
                    # Capture instrumentation is charged INSIDE the original
                    # mapping clock. 540s is an observation trigger, not a cap.
                    argv=['taskset','-c','6',sys.executable,str(R/'scripts/emapper/run.py'),subject['prepared'],'--golden',subject['golden'],'--output',str(capture),*profile['fixed_arguments'],'--snapshot-after','540','--recovery-audit']
                    save(case/'capture-command.json',dict(argv=argv,snapshot_trigger_seconds=540,stage_deadline=False,additional_mapping_budget=False))
                    capture_env=dict(env);nonce=hashlib.sha256(str(capture).encode()).hexdigest()
                    capture_env.update(EMAPPER_AFFINITY_RECEIPT=str(capture/'native-affinity.json'),EMAPPER_AFFINITY_NONCE=nonce)
                    with(case/'capture.log').open('x')as log:proc=subprocess.run(argv,cwd=R,env=capture_env,stdout=log,stderr=subprocess.STDOUT)
                    assert proc.returncode==0;report=read(capture/'report.json');assert report['status']=='ok'and report['cec']==1
                    affinity=read(capture/'native-affinity.json');assert affinity['nonce']==nonce and affinity['cpus_allowed_list']=='6';row['capture_affinity']=affinity
                    snapshot=capture/'recovery-snapshot.json';summary=inventory(snapshot);save(case/'pool-inventory.json',summary)
                    frozen=read(snapshot);assert frozen['environment']['EGM_DIAG_NATIVE_SHA256']==state['native_sha256']
                    diag=case/'fixed-origin';diag.mkdir();denv={k:v for k,v in env.items()if not k.startswith('EGM_')};denv.update(frozen['environment'])
                    denv.update(EGM_SELECTED_PROVENANCE=str(diag/'selected-provenance'),EMAPPER_FIXED_POOL_SNAPSHOT=str(snapshot),EMAPPER_FIXED_POOL_OUTPUT=str(diag))
                    # No frontend/objective override: this is an explicit test
                    # diagnostic comparing BOTH evaluators on the SAME origin.
                    cmd=['taskset','-c','6',str(test_binary),'fixed_pool_area_screens_against_complete_oracle','--ignored','--nocapture','--test-threads=1']
                    row['stage']='fixed_origin_differential';checkpoint();t=time.monotonic()
                    with(case/'fixed-origin.log').open('x')as log:
                        proc=subprocess.Popen(cmd,cwd=R,env=denv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        for _ in range(50):
                            if proc.poll() is not None or sorted(os.sched_getaffinity(proc.pid))==[6]:break
                            time.sleep(.01)
                        if proc.poll() is None:
                            actual=sorted(os.sched_getaffinity(proc.pid));assert actual==[6]
                            save(case/'diagnostic-affinity.json',dict(pid=proc.pid,cpus=actual,argv=cmd))
                        try:code=proc.wait(timeout=max(0,600-(time.monotonic()-t)));timed_out=False
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait();timed_out=True
                    row.update(diagnostic_exit=code,diagnostic_wall=time.monotonic()-t,diagnostic_timeout=timed_out)
                    outputs=[('initial',diag/'initial.v')]
                    outputs += [(d.name,d/'mapped.v')for d in sorted(diag.glob('best-*'))if d.is_dir()and not d.name.endswith('.pending')]
                    checks=[];tools=report['tools'];saved={k:os.environ.get(k)for k in ['EGM_YOSYS','BENCH_RS_EMAP_ABC']}
                    try:
                        os.environ.update(EGM_YOSYS=tools['yosys']['path'],BENCH_RS_EMAP_ABC=tools['abc']['path'])
                        for label,netlist in outputs:
                            measured=measure_cover(library,netlist,case/('validation-'+label),R/'target/release/emapper_measure',tools['yosys']['path'])
                            ok=verify(library.parent/'cells.v',pathlib.Path(subject['golden']),pathlib.Path(measured['netjson']),case/('cec-'+label))
                            expected=summary['initial_AD']if label=='initial'else read(netlist.parent/'metadata.json')['AD']
                            assert [measured['measured']['area'],measured['measured']['delay']]==expected
                            assert ok;checks.append(dict(label=label,AD=expected,CEC=1,netlist_sha256=sha(netlist)))
                    finally:
                        for k,v in saved.items():
                            if v is None:os.environ.pop(k,None)
                            else:os.environ[k]=v
                    assert checks and checks[0]['label']=='initial'
                    row.update(status='partial_diagnostic'if timed_out else'ok',stage='complete',capture_CEC=1,outputs=checks,comparison=read(diag/'comparison.json')if(diag/'comparison.json').exists()else None,
                               completed_event_summary=summarize_events(diag/'proposals.jsonl'),
                               snapshot_sha256=sha(snapshot),sources_recorded=True,joint_proposals_exhausted=False,full_mapper_PPA_claim=False)
                    if not timed_out:assert code==0
                except Exception as error:row.update(status='failed_preserved',stage='failed',error=repr(error))
                checkpoint()
            state['stage']='finite_representative_diagnostics_finished';checkpoint()
        except Exception as error:state.update(stage='failed_preserved',error=repr(error));checkpoint();raise


if __name__=='__main__':main()
