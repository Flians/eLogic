#!/usr/bin/env python3
"""Frozen, global-budget native region/phase mapper and same-binary controls."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from run_epfl_baselines import ROOT, EPFL, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mapper-root', type=Path, required=True)
    ap.add_argument('--subjects', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--study', choices=['repeat','current-flow'], required=True)
    ap.add_argument('--import-failure-source', type=Path)
    ap.add_argument('--cec-transport-proof', type=Path)
    ap.add_argument('--objectives', nargs='+', choices=['delay-area'], default=['delay-area'])
    ap.add_argument('--max-jobs', type=int, default=2)
    a = ap.parse_args()
    if a.max_jobs < 1: ap.error('max-jobs must be positive')
    out, mapper, subjects = a.output.resolve(), a.mapper_root.resolve(), a.subjects.resolve()
    out.mkdir(parents=True, exist_ok=True)
    handles=[]
    for p in [out/'runner.lock', ROOT/'results_v2/epfl_original_baselines_checkpoint_20261013/runner.lock', ROOT/'results_v2/epfl_original_baselines_dependency_checkpoint_20261013/runner.lock', ROOT/'results_v2/epfl_original_baselines_cecfile_checkpoint_20261013/runner.lock', ROOT/'results_v2/epfl_mapping_policy_checkpoint_20261013/runner.lock',
              ROOT/'results_v2/epfl_strict_main_checkpoint_20261013/runner.lock',
              ROOT/'results_v2/epfl_baseline_checkpoint_20261013/runner.lock']:
        f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f)
    prep=json.loads(subjects.read_text())
    if [r['bench'] for r in prep['results']] != list(EPFL) or not all(r['preprocessing']['cec']==1 for r in prep['results']):
        raise ValueError('requires all20 common once-resyn2 subjects')
    files=[Path(__file__).resolve(),Path(__file__).with_name('bench_egraph_event_validation.py'),
           Path(__file__).with_name('run_with_mapping_budget.py'),Path(__file__).with_name('nested_mapping_clock.py'),
           subjects,mapper/'target/release/bench_rs_egraph_map',mapper/'src/bin/bench_rs_egraph_map.rs',
           ROOT/'target/release/bench_rs_netlist_eval',ROOT/'tools/mockturtle/experiments/cell_libraries/asap7.genlib',
           Path('/usr/bin/abc'),Path('/usr/local/bin/yosys'),*sorted((mapper/'emapper/src').rglob('*.rs')),
           *sorted((ROOT/'scripts/experiments').glob('*.py'))]
    frozen={str(p):sha(p) for p in files};sp=out/'campaign.json'
    state=json.loads(sp.read_text()) if sp.exists() else {'expected_jobs':24 if a.study=='repeat' else 160,'objectives':a.objectives,'jobs':[],
        'source_sha256':frozen,'subjects':str(subjects),'mapper_root':str(mapper),'mapping_seconds':300,
        
        'study':a.study,
        'profiles': {name:{'region_classes':256,'phase_recovery':phase,'staged_recovery':stage,'phase_time_fraction':fraction,'library_timing':0,'timing_cap':0,'event_flow':event,'critical_order':0,'coupled_event_flow':coupled}
            for name,phase,stage,fraction,event,coupled in (
                [('staged_sweep',0,1,1.,0,0),('event',0,1,1.,1,0)] if a.study=='repeat' else
                [('coupled_phase0',0,0,1.,0,0),('coupled_quarter',1,0,.25,0,0),('coupled_queue',1,0,.25,0,1),('event',0,1,1.,1,0)])},
        'scope':'Same-binary fixed event flow verification: repeated adder/max vs staged sweep, or full20 vs coupled phase0 and phase-quarter. Same libraries/input, global2s recovery, nodes20k/cuts16/pool100k/SAT16. Configured local trials2000; staged interprets total cap, coupled retains original per-round policy, actual counts retained. No baselines altered.'}

    if state['source_sha256']!=frozen: raise ValueError('source/binary changed: new explicit matrix required')
    for j in state['jobs']:
        for key in ['report','audit']:
            if key in j and sha(Path(j[key]))!=j[key+'_sha256']:raise ValueError('completed observation changed')
    def save():
        tmp=sp.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(sp)
    adoption=None
    if a.import_failure_source:
        if a.study!='current-flow':raise ValueError('only pre-import full-scope failures may be adopted')
        adoption=json.loads(a.import_failure_source.read_text())
        old=adoption['source_sha256']
        old_only=set(old)-set(frozen);new_only=set(frozen)-set(old)
        previous_helper=Path(__file__).with_name('bench_egraph_staged_factors.py')
        current_helper=Path(__file__).with_name('bench_egraph_event_validation.py')
        assert old_only=={str(Path(__file__).with_name('run_epfl_staged_factors.py')),str(previous_helper)}
        assert new_only=={str(Path(__file__).resolve()),str(current_helper)}
        expected=previous_helper.read_text().replace('    ap.add_argument("--staged-event-flow",','    ap.add_argument("--coupled-event-flow", type=int, choices=[0,1], default=0)\n    ap.add_argument("--staged-event-flow",').replace('env.update(EGM_REGION_CLASSES=str(a.region_classes),','env.update(EGM_FLOW_WORKLIST=str(a.coupled_event_flow), EGM_REGION_CLASSES=str(a.region_classes),')
        assert current_helper.read_text()==expected
        changed={p for p in set(old)&set(frozen) if old[p]!=frozen[p]}
        if changed:
            assert a.cec_transport_proof
            transport=json.loads(a.cec_transport_proof.read_text())
            evaluator=str(ROOT/'target/release/bench_rs_netlist_eval')
            assert changed=={evaluator}
            assert old[evaluator]==transport['old_evaluator_sha256'] and frozen[evaluator]==transport['new_evaluator_sha256']
            assert transport['all_other_production_code_byte_identical'] and not transport['public_metrics_code_changed']
        else:transport=None
        code=(mapper/'emapper/src/egraph_map.rs').read_text()
        block=code[code.index('let (graph, roots) = if ps.region_classes'):code.index('let imported_enodes =')]
        excluded=['phase_recovery','staged_recovery','staged_event_flow','staged_critical_order','phase_time_fraction','flow_worklist']
        assert not any('ps.'+field in block for field in excluded)
        proof={'source_campaign':str(a.import_failure_source.resolve()),'sha256':sha(a.import_failure_source),
            'shared_dependency_sha256':{p:frozen[p] for p in set(old)&set(frozen)},
            'unchanged_import_block':block,'import_block_sha256':__import__('hashlib').sha256(block.encode()).hexdigest(),
            'changed_flags_only_after_import_or_valid_parameter_checks':excluded,'CEC_transport_proof':str(a.cec_transport_proof) if transport else None,'new_helper_is_previous_plus_explicit_coupled_flow_flag':True,
            'note':'All profiles retain region256/strict1/node20000 and valid stage/phase combinations. Failed imported graph is deterministic before these flags can affect recovery. Original argv/audit/report/errors retained; no new execution/runtime claim.'}
        pp=out/'import_failure_adoption.json'
        if pp.exists():assert json.loads(pp.read_text())==proof
        else:pp.write_text(json.dumps(proof,indent=2))
    count=0
    for k in [3,4]:
      for objective in state['objectives']:
       tasks=[(bench,rep) for bench in (['adder','max'] if a.study=='repeat' else EPFL) for rep in range(3 if a.study=='repeat' else 1)]
       for bench,rep in tasks:
        profiles=list(state['profiles'].items())
        if rep%2:profiles.reverse()
        for profile,config in profiles:
          name=f'{k}in_{objective}_{bench}_r{rep}_{profile}'
          if any(j['name']==name for j in state['jobs']):continue
          folder=out/name;audit=out/(name+'.audit.json');log=out/(name+'.log')
          if any(p.exists() for p in [folder,audit,log]):raise ValueError('preserve and audit interrupted artifacts: '+name)
          cmd=[sys.executable,str(Path(__file__).with_name('run_with_mapping_budget.py')),'--mapping-seconds','300',
               '--driver',str(Path(__file__).with_name('bench_egraph_event_validation.py')),'--audit',str(audit),'--',
               '--mapper-root',str(mapper),'--region-classes',str(config['region_classes']),
               '--coupled-event-flow',str(config['coupled_event_flow']),'--staged-event-flow',str(config['event_flow']),'--staged-critical-order',str(config['critical_order']),
               '--phase-recovery',str(config['phase_recovery']),'--staged-recovery',str(config['staged_recovery']),'--phase-time-fraction',str(config['phase_time_fraction']),'--library-timing',str(config['library_timing']),
               '--timing-cap',str(config['timing_cap']),'--output',str(folder),'--k',str(k),
               '--objective',objective,'--preprocess','resyn2','--prepared-report',str(subjects),
               '--methods','egraph_bounded','--suite','epfl','--benches',bench]
          source=None
          if adoption:
            source=next((j for j in adoption['jobs'] if j['k']==k and j['bench']==bench and j['profile']=='region_phase' and j.get('failed')==1),None)
          if source:
            for key in ['report','audit']:assert sha(source[key])==source[key+'_sha256']
            d=json.loads(Path(source['report']).read_text());assert len(d['results'])==1
            row=d['results'][0];assert row['status']!='ok'
            mp=Path(source['report']).parent/(bench+'.'+row['method'])/'map.log'
            import re
            assert re.fullmatch(r'Error: "budget: imported graph [0-9]+ exceeds node allowance 20000"',mp.read_text().strip())
            j={key:source[key] for key in ['report','report_sha256','audit','audit_sha256','exit_code','ok','failed']}
            j.update(name=name,bench=bench,k=k,objective=objective,profile=profile,repetition=rep,
                requested_argv_not_executed=cmd,execution='adopted_pre_stage_failure',source_job=source['name'],
                original_argv=source['argv'],original_map_log=str(mp),original_map_log_sha256=sha(mp),proof=str(out/'import_failure_adoption.json'))
            state['jobs'].append(j);save();print(name,'adopted import failure',flush=True)
            count+=1
            if count>=a.max_jobs:return
            continue
          state['current_job']={'name':name,'argv':cmd};save()
          with log.open('w') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
          j={'name':name,'bench':bench,'k':k,'objective':objective,'profile':profile,'repetition':rep,'argv':cmd,'exit_code':r.returncode}
          for key,p in [('report',folder/'report.json'),('audit',audit)]:
              if p.exists():j.update({key:str(p),key+'_sha256':sha(p)})
          if 'report' in j:
              rows=json.loads(Path(j['report']).read_text())['results'];j['ok']=sum(x['status']=='ok' for x in rows);j['failed']=sum(x['status']!='ok' for x in rows)
          else:j.update(ok=0,failed=1,error='no report; log retained')
          state['jobs'].append(j);state.pop('current_job',None);save();print(name,j['ok'],j['failed'],flush=True)
          count+=1
          if count>=a.max_jobs:return
    state['status']='all_attempted';save()

if __name__=='__main__':main()
