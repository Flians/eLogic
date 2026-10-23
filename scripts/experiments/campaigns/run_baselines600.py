"""Original baseline algorithms, explicitly audited reuse, new600 mapping clock."""
import argparse,csv,fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
STUDY=ROOT/'results_v2/time_only_600_20261022'
CATALOGUE=ROOT/'results_v2/single_output_wide_20261021/primary_baseline120_checkpoint.json'
TIMES=CATALOGUE.parent/'full_library_baseline_A_D_T/baseline_A_D_T.csv'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(files):
    for p,h in files.items():assert sha(p)==h,'dependency/artifact drift: '+p
def command_for(job,folder,audit):
    argv=list(job['argv']);argv[0]=sys.executable;argv[1]=str(HERE/'run_baseline600_budget.py')
    argv[argv.index('--mapping-seconds')+1]='600'
    argv[argv.index('--audit')+1]=str(audit)
    argv[argv.index('--output')+1]=str(folder)
    # Driver watchdogs are overridden by the timing-only harness: mapping uses
    # the cumulative clock; conversion/CEC/nonmapping native work are excluded.
    if '--timeout'in argv:argv[argv.index('--timeout')+1]='600'
    else:argv+=['--timeout','600']
    return argv
def audit_old():
    prior=json.loads(CATALOGUE.read_text());verify(prior['fingerprint']['files'])
    dependencies=json.loads((STUDY/'baseline_dependency_audit.json').read_text())
    assert not dependencies['bad'];verify(dependencies['required_actual_dependencies'])
    system=json.loads((STUDY/'baseline_system_tools.json').read_text())
    for group in ['old_reported_tools_verified','newly_pinned_system_dependencies']:
        verify(system[group]);dependencies['required_actual_dependencies'].update(system[group])
    native=json.loads((STUDY/'baseline_native_audit.json').read_text());assert not native['bad']
    for root,info in native['native'].items():
        commit=subprocess.check_output(['git','-C',root,'rev-parse','HEAD'],text=True).strip()
        assert commit==info['commit']
        verify({str(Path(root)/p):h for p,h in info['tracked_sha256'].items()})
    times=list(csv.DictReader(TIMES.open()))
    accepted=[];pending=[]
    for job in prior['jobs']:
        verify(job['sha256'])
        report=json.loads(Path(job['report']).read_text())if job.get('report')else{}
        rows=report.get('results',[]);time_rows=[r for r in times if r['source_job']==job['name']]
        complete=bool(rows)and all(r.get('status')=='ok'and r.get('measured',{}).get('cec')==1 for r in rows)
        if complete:
            assert len(rows)==len(time_rows)
            for r in time_rows:
                assert r['status']=='ok'and r['qualification']=='equivalent'
                assert r['T']and 0<=float(r['T'])<300
                assert sha(r['audit'])==r['audit_sha256']
            accepted.append(dict(job,observation_kind='adopted_completed_under_nonbinding_stricter_guard',
                adopted_from=str(CATALOGUE),timing_rows=time_rows,
                adoption_reason='Same original algorithm/input/library/native/model/evaluator; all method outputs completed with CEC=1 and observed mapping time or conservative upper bound below300s. No timeout incumbent or per-design winner selection. Original policy and failures remain in source catalogue.'))
        else:pending.append(job['name'])
    return prior,accepted,pending,dependencies
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-jobs',type=int,default=1);a=ap.parse_args();assert a.max_jobs>=0
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    locks=[]
    paths=[output/'runner.lock']
    if a.max_jobs:paths += [ROOT/'results_v2/independent_components_20261019/global.lock',ROOT/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock']
    for path in paths:
        f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    prior,adopted,pending,deps=audit_old()
    files=[Path(__file__),HERE/'run_baseline600_budget.py',HERE/'meter_yosys_abc600.py',HERE/'nested_mapping_clock.py',
        CATALOGUE,TIMES,STUDY/'baseline_dependency_audit.json',STUDY/'baseline_native_audit.json',STUDY/'baseline_system_tools.json',ROOT/'emapper/EXPERIMENT_POLICY.json']
    frozen={str(p.resolve()):sha(p)for p in files};frozen.update(deps['required_actual_dependencies'])
    config=dict(scope='EPFL20 full47 single-output original baseline algorithms',mapping_seconds=600,
        baseline_algorithms_changed=False,native_parameters_changed=False,driver_process_watchdog=None,
        CEC_conversion_metric_reporting_excluded=True,EqMap_new_nested_ABC_exact=True,
        adopted_old_T_kind_preserved=True,selection='all fully completed groups adopted; all incomplete groups retested, original observations retained')
    cp=output/'campaign.json'
    if cp.exists():state=json.loads(cp.read_text());assert state['configuration']==config and state['fingerprint']==frozen,'frozen study drift'
    else:
        state=dict(configuration=config,fingerprint=frozen,expected_jobs=120,jobs=adopted,status='checkpointed',
            rerun_plan=pending,all_original_observations=str(CATALOGUE),old_failures_retained=True)
    def save():
        tmp=cp.with_suffix('.pending');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(cp)
    for j in state['jobs']:verify(j['sha256'])
    save()
    if a.max_jobs==0:print('adopted',len(adopted),'pending',len(pending));return
    done={j['name']for j in state['jobs']};count=0
    for old in prior['jobs']:
        if old['name']in done:continue
        verify(frozen);name=old['name'];folder=output/name;audit=output/(name+'.audit.json');log=output/(name+'.log')
        assert not any(p.exists()for p in [folder,audit,log]),'unrecorded partial; audit before recovery: '+name
        argv=command_for(old,folder,audit);state.update(status='running',current_job=dict(name=name,argv=argv));save()
        env={k:v for k,v in os.environ.items()if not k.startswith(('EGM_','ESYN_','MAPPING_CLOCK_','METER_'))}
        with log.open('w')as stream:r=subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        clock_path=audit.with_suffix('.clock.json');capture_dir=clock_path.parent/(clock_path.stem+'.yosys-abc')
        artifacts=[p for p in folder.rglob('*')if p.is_file()]+[p for p in [audit,clock_path,log]if p.is_file()]
        if capture_dir.exists():artifacts += [p for p in capture_dir.rglob('*')if p.is_file()]
        entry=dict(name=name,bench=old['bench'],source_cohort=old['source_cohort'],argv=argv,exit_code=r.returncode,
            observation_kind='new600_actual_mapping_clock',original_job=old,ABC_capture_dir=str(capture_dir)if capture_dir.exists()else None,sha256={str(p):sha(p)for p in artifacts})
        report=folder/'report.json'
        if report.exists():
            rows=json.loads(report.read_text())['results'];entry.update(report=str(report),ok=sum(x.get('status')=='ok'and x.get('measured',{}).get('cec')==1 for x in rows),failed=sum(x.get('status')!='ok'or x.get('measured',{}).get('cec')!=1 for x in rows))
        else:entry.update(ok=0,failed=1,unmeasured=True)
        state['jobs'].append(entry);state.pop('current_job');state['status']='attempted'if len(state['jobs'])==120 else'checkpointed';save()
        print(name,entry['ok'],entry['failed'],flush=True);count+=1
        if count>=a.max_jobs:break
if __name__=='__main__':main()
