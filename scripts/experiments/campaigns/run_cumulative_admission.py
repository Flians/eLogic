#!/usr/bin/env python3
"""Cumulative admission qualification; baseline algorithms untouched."""
import argparse,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
from run_epfl_baselines import EPFL
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(c,cp):
 tmp=cp.with_suffix(".json.tmp");tmp.write_text(json.dumps(c,indent=2)+"\n");tmp.replace(cp)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-jobs',type=int,default=10);ap.add_argument('--designs',nargs='+');ap.add_argument('--reverse-order',action='store_true');ap.add_argument('--factors',choices=['pair','all'],default='pair');ap.add_argument('--components',action='store_true');a=ap.parse_args();out=a.output.resolve();out.mkdir(exist_ok=True)
 proto=ROOT;driver=Path(__file__).with_name('bench_functional_cuts.py');wrap=driver.with_name('run_with_mapping_budget.py');subjects=ROOT/'results_v2/epfl_resyn2_subjects_20261012/report.json'
 sources=[ROOT/'target/release/bench_rs_egraph_map',*sorted((ROOT/'emapper/src').glob('*.rs')),ROOT/'src/bin/bench_rs_egraph_map.rs',proto/'target/release/bench_rs_egraph_map',ROOT/'target/release/bench_rs_netlist_eval',subjects,driver,wrap,Path(__file__).resolve(),driver.with_name('nested_mapping_clock.py'),*sorted((proto/'emapper/src').glob('*.rs')),proto/'src/bin/bench_rs_egraph_map.rs',ROOT/'scripts/experiments/port_identifiers.py',ROOT/'scripts/experiments/shared_subject.py',ROOT/'scripts/experiments/benchmark_suites.py',ROOT/'scripts/experiments/restrict_genlib.py',ROOT/'tools/mockturtle/experiments/cell_libraries/asap7.genlib']
 sources += [proto/'Cargo.toml',proto/'Cargo.lock',proto/'emapper/Cargo.toml',ROOT/'Cargo.toml',ROOT/'Cargo.lock',ROOT/'emapper/Cargo.toml',ROOT/'emapper/tests/fixtures/diagnostic_names.genlib',ROOT/'emapper/tests/fixtures/diagnostic_names_golden.json']
 fp={str(p):sha(p) for p in sources};cp=out/'campaign.json';designs=a.designs or EPFL;assert len(set(designs))==len(designs) and all(b in EPFL for b in designs)
 extra={}
 factor_map={'legacy_control':(0,0),'functional_cuts':(0,0)}
 assert not a.components, 'terminal study is a single terminal0/1 factor'
 c=json.loads(cp.read_text()) if cp.exists() else {'fingerprint':fp,'expected_jobs':len(designs)*2*len(factor_map),'designs':designs,'objective':'delay-area','factors':factor_map,'fixed_profiles':{p:[1,8,0,1,1,1,1,1,0,0.6,0.05,0,1,0,0,1.0,1,1] for p in factor_map},'jobs':[],'budgets':{'mapping':300,'recovery':2,'nodes':20000,'cuts':16,'pool':100000,'SAT_iterations':16},'scope':'V26 uniform tail-only single-output immutable-pool protected area recovery from V19. Both settings cut_pin_flow1/refresh_area_timing0; only tail_shared_area0/1 changes. The exploratory four-factor pilot including negative timing-refresh/combined cases is retained separately; no per-design choice. Save whole validated tail predecessor, activate better already-visited deferred cover, bound same tail flow fraction0.30, refresh actual arrivals/required labels after online commits. All original cumulative20knodes/cut16/retained100k/SAT16/inclusive2s/native60/mapping300 unchanged. No cut exchange or second pool/recovery. Does not prove preservation of another pipeline historical final cover. Original baseline algorithms untouched. All adverse cases retained; fixed config, no per-design winners.'}
 c['scope']='Cumulative admission V37. One lifetime ledger across imports and strict SAT, no refund on union/rebuild. Fixed functional cuts0/1, terminalarea1 both, hard phase screen0/source preprocessing0. Numeric budgets and baselines unchanged. Cross-pool complete old cover protection not implemented. Full20 only when all expected jobs complete; preserve failures. Not a PPA improvement claim.'
 c['component_overrides']=extra
 c['components']=a.components
 assert c['fingerprint']==fp,'frozen source drift'
 assert c['factors']=={k:list(v) for k,v in factor_map.items()} if cp.exists() else True
 if 'reverse_order' in c:assert c['reverse_order']==a.reverse_order
 else:c['reverse_order']=a.reverse_order
 profiles=list(c['fixed_profiles'].items());profiles=profiles[::-1] if a.reverse_order else profiles
 locks=[]
 for p in [out/'runner.lock',ROOT/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock',ROOT/'results_v2/small_cut_quota_cases_20261014/runner.lock']:
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 for j in c['jobs']:
  for key in ['report','audit','pool','candidates','trace','pool_frontier','pool_guard','helper_allocation','admission']:
   if key in j:assert sha(Path(j[key]))==j[key+'_sha256']
 done={j['name'] for j in c['jobs']};count=0
 for k in [4,3]:
  for bench in designs:
   for profile,(cap,quota,areaquota,mixed,canonical,refs,slack,wires,literal,area_fraction,reserve,fast,auto,critical,bounded,seed_fraction,frontier,area_dag) in profiles:
    name=f'{k}in_{bench}_{profile}'
    if name in done:continue
    if count>=a.max_jobs:c['status']='checkpointed';save(c,cp);return
    assert all(sha(Path(p))==s for p,s in fp.items());folder=out/name;audit=out/(name+'.audit.json');assert not folder.exists(),'Unrecorded partial requires independent audit; not overwritten'
    args=['--refresh-area-timing',str(factor_map[profile][0]),'--tail-shared-area',str(factor_map[profile][1]),'--cut-pin-flow','1','--delta-area-filter','0','--online-shared-bound','1','--canonical-source-inputs','0','--import-helper-fill','1','--introduce-phase-inverters','1','--shared-area-bound','1','--fused-cover-scoring','1','--preserve-shared-trajectory','1','--online-area','1','--activate-deferred','1','--timing-witness-pool','1','--joint-cones','0','--scored-incumbent','1','--cone-import','1','--mapper-root',str(proto),'--k',str(k),'--objective','delay-area','--methods','egraph_bounded','--coupled-event-flow','1','--phase-recovery','1','--phase-time-fraction','0.25','--region-classes','256','--library-timing','0','--timing-cap',str(cap),'--small-cut-quota',str(quota),'--area-cut-quota',str(areaquota),'--cap-diversity',str(mixed),'--canonical-phase',str(canonical),'--structural-refs',str(refs),'--slack-flow',str(slack),'--canonical-wires',str(wires),'--literal-import',str(literal),'--area-time-fraction',str(area_fraction),'--return-reserve',str(reserve),'--fast-area',str(fast),'--literal-auto',str(auto),'--critical-recovery',str(critical),'--bounded-proposals',str(bounded),'--seed-window-fraction',str(seed_fraction),'--fast-delay-seed',str(frontier),'--required-area-dag',str(area_dag),'--defer-area-dag',str(area_dag),'--preserve-legacy-seeds',str(area_dag),'--metric-epochs',str(area_dag),'--output',str(folder),'--prepared-report',str(subjects),'--suite','epfl','--benches',bench]
    args += ['--functional-cuts',str(int(profile=='functional_cuts')),'--terminal-area','1']
    args += extra.get(profile,[])
    if profile=='no_phase_recovery':args[args.index('--phase-recovery')+1]='0'
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
       if key=='pool' and p.with_suffix('.admission.json').is_file():
        q=p.with_suffix('.admission.json');j['admission']=str(q);j['admission_sha256']=sha(q)
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
