#!/usr/bin/env python3
"""Checkpointed fixed candidate-retention factorial; author baselines untouched."""
import argparse,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
from run_epfl_baselines import EPFL
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(c,cp):
 tmp=cp.with_suffix(".json.tmp");tmp.write_text(json.dumps(c,indent=2)+"\n");tmp.replace(cp)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-jobs',type=int,default=10);ap.add_argument('--designs',nargs='+');ap.add_argument('--reference-study',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();out.mkdir(exist_ok=True)
 proto=Path('/tmp/emapper-live-shared-area-20261016');driver=Path(__file__).with_name('bench_live_shared_area.py');wrap=driver.with_name('run_with_mapping_budget.py');subjects=ROOT/'results_v2/epfl_resyn2_subjects_20261012/report.json'
 sources=[ROOT/'target/release/bench_rs_egraph_map',*sorted((ROOT/'emapper/src').glob('*.rs')),ROOT/'src/bin/bench_rs_egraph_map.rs',proto/'target/release/bench_rs_egraph_map',ROOT/'target/release/bench_rs_netlist_eval',subjects,driver,wrap,Path(__file__).resolve(),driver.with_name('nested_mapping_clock.py'),*sorted((proto/'emapper/src').glob('*.rs')),proto/'src/bin/bench_rs_egraph_map.rs',ROOT/'scripts/experiments/port_identifiers.py',ROOT/'scripts/experiments/shared_subject.py',ROOT/'scripts/experiments/benchmark_suites.py',ROOT/'scripts/experiments/restrict_genlib.py',ROOT/'tools/mockturtle/experiments/cell_libraries/asap7.genlib']
 sources += [ROOT/'Cargo.toml',ROOT/'Cargo.lock',ROOT/'emapper/Cargo.toml',ROOT/'emapper/tests/fixtures/diagnostic_names.genlib',ROOT/'emapper/tests/fixtures/diagnostic_names_golden.json']
 fp={str(p):sha(p) for p in sources};cp=out/'campaign.json';designs=a.designs or ['sqrt'];assert len(set(designs))==len(designs) and all(b in EPFL for b in designs)
 c=json.loads(cp.read_text()) if cp.exists() else {'fingerprint':fp,'expected_jobs':len(designs)*12,'designs':designs,'repetitions':2,'objective':'delay-area','fixed_profiles':{'legacy_control':[1,8,0,1,1,1,1,1,0,0.0,0.05,0,1,0,0,1.0,1,1],'allocation_only':[1,8,0,1,1,1,1,1,0,0.6,0.05,0,1,0,0,1.0,1,1],'protected_area':[1,8,0,1,1,1,1,1,0,0.6,0.05,0,1,0,0,1.0,1,1]},'jobs':[],'budgets':{'mapping':300,'recovery':2,'nodes':20000,'cuts':16,'pool':100000,'SAT_iterations':16},'scope':'Two fresh alternating-profile-order repetitions of all3fixed V13sqrt allocation profiles: fraction0/bound0, fraction0.6/bound0, fraction0.6/bound1. Same binary/library/once-resyn2 inputs/global2s20knodes/cut16/pool100k/SAT16/native60/mapping300.12newattempts, no adopted fullstudy observations or best-run selection. Original baseline algorithms unchanged. This is only sqrt repeat evidence, not full20 qualification.'}
 assert c['fingerprint']==fp,'frozen source drift'
 full=json.loads(a.reference_study.read_text());assert full['status']=='attempted' and len(full['jobs'])==full['expected_jobs']==6
 allowed={'run_live_area_allocation.py','run_live_area_allocation_repeats.py'};strip=lambda f:{p:h for p,h in f.items() if Path(p).name not in allowed};assert strip(full['fingerprint'])==strip(fp) and full['fixed_profiles']==c['fixed_profiles']
 c['reference_study']=str(a.reference_study.resolve());c['reference_study_sha256']=sha(a.reference_study)
 locks=[]
 for p in [out/'runner.lock',ROOT/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock',ROOT/'results_v2/small_cut_quota_cases_20261014/runner.lock']:
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 for j in c['jobs']:
  for key in ['report','audit','pool','candidates','trace','pool_frontier','pool_guard','helper_allocation']:
   if key in j:assert sha(Path(j[key]))==j[key+'_sha256']
 done={j['name'] for j in c['jobs']};count=0
 for repetition in range(2):
  for k in [4,3]:
   for bench in designs:
    for profile,(cap,quota,areaquota,mixed,canonical,refs,slack,wires,literal,area_fraction,reserve,fast,auto,critical,bounded,seed_fraction,frontier,area_dag) in (list(c['fixed_profiles'].items()) if repetition==0 else list(reversed(list(c['fixed_profiles'].items())))):
     name=f'{k}in_{bench}_{profile}_r{repetition}'
     if name in done:continue
     if count>=a.max_jobs:c['status']='checkpointed';save(c,cp);return
     assert all(sha(Path(p))==s for p,s in fp.items());folder=out/name;audit=out/(name+'.audit.json');assert not folder.exists(),'Unrecorded partial requires independent audit; not overwritten'
     args=['--online-shared-bound',str(int(profile=='protected_area')),'--canonical-source-inputs','0','--import-helper-fill','1','--introduce-phase-inverters','1','--shared-area-bound','1','--fused-cover-scoring','1','--preserve-shared-trajectory','1','--online-area','1','--activate-deferred','1','--timing-witness-pool','1','--joint-cones','0','--scored-incumbent','1','--cone-import','1','--mapper-root',str(proto),'--k',str(k),'--objective','delay-area','--methods','egraph_bounded','--coupled-event-flow','1','--phase-recovery','1','--phase-time-fraction','0.25','--region-classes','256','--library-timing','0','--timing-cap',str(cap),'--small-cut-quota',str(quota),'--area-cut-quota',str(areaquota),'--cap-diversity',str(mixed),'--canonical-phase',str(canonical),'--structural-refs',str(refs),'--slack-flow',str(slack),'--canonical-wires',str(wires),'--literal-import',str(literal),'--area-time-fraction',str(area_fraction),'--return-reserve',str(reserve),'--fast-area',str(fast),'--literal-auto',str(auto),'--critical-recovery',str(critical),'--bounded-proposals',str(bounded),'--seed-window-fraction',str(seed_fraction),'--fast-delay-seed',str(frontier),'--required-area-dag',str(area_dag),'--defer-area-dag',str(area_dag),'--preserve-legacy-seeds',str(area_dag),'--metric-epochs',str(area_dag),'--output',str(folder),'--prepared-report',str(subjects),'--suite','epfl','--benches',bench]
     argv=[sys.executable,str(wrap),'--mapping-seconds','300','--driver',str(driver),'--audit',str(audit),'--',*args]
     with (out/(name+'.log')).open('w') as log:r=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
     j={'name':name,'k':k,'bench':bench,'profile':profile,'timing_cap':cap,'small_quota':quota,'area_quota':areaquota,'cap_diversity':mixed,'canonical_phase':canonical,'structural_refs':refs,'slack_flow':slack,'canonical_wires':wires,'literal_import':literal,'area_fraction':area_fraction,'return_reserve':reserve,'fast_area':fast,'literal_auto':auto,'critical':critical,'bounded':bounded,'seed_fraction':seed_fraction,'fast_frontier':frontier,'required_area_dag':area_dag,'argv':argv,'exit_code':r.returncode}
     for key,p in [('report',folder/'report.json'),('audit',audit)]:
      if p.exists():j[key]=str(p);j[key+'_sha256']=sha(p)
     if 'report' in j:
      for row in json.loads(Path(j['report']).read_text())['results']:
       for key,env in [('pool','EGM_POOL_DIAG_PATH'),('candidates','EGM_DIAG_PATH'),('trace','EGM_RECOVERY_TRACE')]:
        p=Path(row.get('settings',{}).get(env,'/nonexistent'))
        if p.is_file():j[key]=str(p);j[key+'_sha256']=sha(p)
        if key=='pool' and p.with_suffix('.frontier.json').is_file():
         q=p.with_suffix('.frontier.json');j['pool_frontier']=str(q);j['pool_frontier_sha256']=sha(q)
        if key=='pool' and p.with_suffix('.helper-allocation.json').is_file():
         q=p.with_suffix('.helper-allocation.json');j['helper_allocation']=str(q);j['helper_allocation_sha256']=sha(q)
        if key=='pool' and p.with_suffix('.pool-guard.json').is_file():
         q=p.with_suffix('.pool-guard.json');j['pool_guard']=str(q);j['pool_guard_sha256']=sha(q)
        if key=='candidates' and p.with_suffix('.profile.json').is_file():
         q=p.with_suffix('.profile.json');j['diagnostic_profile']=str(q);j['diagnostic_profile_sha256']=sha(q)
     c['jobs'].append(j);save(c,cp);count+=1;print(name,r.returncode,flush=True)
 c['status']='attempted';save(c,cp)
if __name__=='__main__':main()
