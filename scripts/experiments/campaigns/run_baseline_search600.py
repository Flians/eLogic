"""Search-inclusive600 cohort; original baseline algorithms/native parameters unchanged."""
import argparse,copy,fcntl,json,os,subprocess,sys
from pathlib import Path
import run_baselines600 as old
ROOT,HERE,STUDY=old.ROOT,old.HERE,old.STUDY
sha,verify=old.sha,old.verify
PREVIOUS=STUDY/'original_baselines_epfl/campaign.json'
PROOF=STUDY/'hyp_identical_delay_proof/proof_reuse.json'

def audited_adoptions():
    prior,_,_,deps=old.audit_old()
    current=json.loads(PREVIOUS.read_text());verify(current['fingerprint'])
    proof=json.loads(PROOF.read_text());verify(proof['sha256'])
    adopted=[]
    for job in current['jobs']:
        verify(job['sha256'])
        if not job.get('report'):continue
        report=json.loads(Path(job['report']).read_text());rows=report.get('results',[])
        if not rows:continue
        times={r['method']:r for r in job.get('timing_rows',[])}
        audit={}
        if not times:
            ap=Path(job['argv'][job['argv'].index('--audit')+1]);audit=json.loads(ap.read_text())
        complete=True;newtimes=[]
        for row in rows:
            method=row['method'];positive=row.get('status')=='ok' and row.get('measured',{}).get('cec')==1
            if job['name']==proof['source_job'] and method==proof['method']:
                positive=proof['mapping_completed_before_verification'] and proof['measured']['cec']==1
                assert proof['current_report']==str(Path(job['report']).resolve())
            if not positive:complete=False;break
            previous=times.get(method)
            if previous:
                base=float(previous['T']);base_kind=previous['T_kind']
            else:
                state=audit.get('clock',{}).get(job['bench']+'/'+method,{})
                assert state and not any(state.get(k)for k in ['active','incomplete','exhausted'])
                base=state['used'];base_kind='mapping_only_active_interval_wall'
            # Conservative: SA includes its feedback ABC already, so adding the
            # old mapping time double counts it only in this adoption upper bound.
            # The fresh clock uses interval union, never this sum.
            extra=0.
            if method=='emorphic_final_sa':
                assert all(k in row for k in ['rewrite_secs','sa_secs'])
                extra=row['rewrite_secs']+row['sa_secs']
            elif method.startswith('esyn_'):
                assert 'native_wall_secs' in row;extra=row['native_wall_secs']
            # EqMap native search was already charged by its direct mapper clock.
            total=base+extra
            if not 0<=total<600:complete=False;break
            newtimes.append(dict(method=method,T=total,T_kind='search_inclusive_conservative_upper_bound' if extra else base_kind,
                                 prior_T=base,prior_T_kind=base_kind,added_search_wall_secs=extra,
                                 original_policy_retained=True))
        if complete:
            item=copy.deepcopy(job);item.update(observation_kind='adopted_search600_with_explicit_nonbinding_upper_bound',
                timing_rows=newtimes,adopted_from=str(PREVIOUS),prior_observation_kind=job['observation_kind'],
                adoption_reason='Exact input/library/author dependencies and positiveCEC; stored native search plus previous mapping upper bound below600. No new run or changed algorithm. Original records retained.')
            adopted.append(item)
    return prior,adopted,deps

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-jobs',type=int,default=1);ap.add_argument('--benches',nargs='*');a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);locks=[]
    for p in [out/'runner.lock',STUDY/'original_baselines_epfl/runner.lock',ROOT/'results_v2/independent_components_20261019/global.lock',ROOT/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock']:
        f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    prior,adopted,deps=audited_adoptions()
    paths=[Path(__file__),HERE/'run_baseline_search600_budget.py',HERE/'run_baselines600.py',HERE/'run_baseline600_budget.py',HERE/'nested_mapping_clock.py',HERE/'meter_yosys_abc600.py',PREVIOUS,PROOF,STUDY/'search_scope_transition/sealed.json']
    frozen={str(p.resolve()):sha(p)for p in paths};frozen.update(deps['required_actual_dependencies'])
    configuration=dict(policy='search-inclusive600-v2',mapping_seconds=600,search_included=True,baseline_algorithms_changed=False,native_parameters_changed=False,
        excludes=['once resyn2','pure format conversion','independentCEC','metric reporting'],nested_feedback='union with parent search interval, no double charge')
    cp=out/'campaign.json'
    if cp.exists():
        state=json.loads(cp.read_text());assert state['fingerprint']==frozen and state['configuration']==configuration,'frozen cohort drift'
    else:state=dict(configuration=configuration,fingerprint=frozen,expected_jobs=120,jobs=adopted,status='checkpointed',old_cohort=str(PREVIOUS),all_failures_retained=True)
    def save():
        temp=cp.with_suffix('.pending');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(cp)
    for j in state['jobs']:verify(j['sha256'])
    save();print('qualified adopted',len(adopted),'remaining',120-len(state['jobs']),flush=True)
    if a.max_jobs==0:return
    done={j['name']for j in state['jobs']};count=0
    for j in prior['jobs']:
        if j['name']in done or (a.benches and j['bench']not in a.benches):continue
        verify(frozen);name=j['name'];folder=out/name;audit=out/(name+'.audit.json');log=out/(name+'.log')
        assert not any(p.exists()for p in [folder,audit,log]),'partial output requires audit: '+name
        argv=old.command_for(j,folder,audit);argv[1]=str(HERE/'run_baseline_search600_budget.py')
        state.update(status='running',current_job=dict(name=name,argv=argv));save()
        env={k:v for k,v in os.environ.items()if not k.startswith(('EGM_','ESYN_','MAPPING_CLOCK_','METER_'))}
        with log.open('w')as f:r=subprocess.run(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        clock=audit.with_suffix('.clock.json');captures=clock.parent/(clock.stem+'.yosys-abc')
        paths=[p for p in folder.rglob('*')if p.is_file()]+[p for p in [audit,clock,log]if p.is_file()]
        if captures.exists():paths += [p for p in captures.rglob('*')if p.is_file()]
        entry=dict(name=name,bench=j['bench'],source_cohort=j['source_cohort'],argv=argv,exit_code=r.returncode,
                   observation_kind='new600_search_inclusive_union_clock',original_job=j,sha256={str(p):sha(p)for p in paths})
        report=folder/'report.json'
        if report.exists():
            rows=json.loads(report.read_text())['results'];entry.update(report=str(report),ok=sum(x.get('status')=='ok'and x.get('measured',{}).get('cec')==1 for x in rows),failed=sum(x.get('status')!='ok'or x.get('measured',{}).get('cec')!=1 for x in rows))
        else:entry.update(ok=0,failed=1,unmeasured=True)
        state['jobs'].append(entry);state.pop('current_job');state['status']='attempted'if len(state['jobs'])==120 else'checkpointed';save()
        print(name,entry['ok'],entry['failed'],flush=True);count+=1
        if count>=a.max_jobs:break
if __name__=='__main__':main()
