"""Close the diagnostic evidence chain, retaining the invalid interrupted pilot."""
import fcntl,hashlib,json,os,pathlib,subprocess,sys
from resource_pair import R
O=R/'results_v2/emapper_stamp_diagnosis_20261111'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 locks=[]
 for p in [R/'results_v2/independent_components_20261019/global.lock',R/'results_v2/emapper_physical_stamp_20261111/benchmark/runner.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 assert not (O/'validation.json').exists()
 sys.path.insert(0,str(R/'scripts/emapper'));from verify import verify
 from selected_evidence import replay
 from trajectory import covers
 live=read(O/'live-dec/summary.json')['rows'];frozen=read(O/'live-dec/metadata.json')['frozen']
 assert all(sha(R/p)==h for p,h in frozen.items()),'original source/binary/harness drift'
 for r in live:
  assert r['status']=='ok'and r['CEC']==1
  assert all(sha(pathlib.Path(r['case'])/p)==h for p,h in r['artifacts'].items())
 assert covers(pathlib.Path(live[0]['case']))==covers(pathlib.Path(live[1]['case']))
 fixed=read(O/'fixed-work-v2/result.json')['rows'];a,b=fixed
 for r in fixed:
  d=r['replay'];assert r['exit']==0 and r['CEC']and not d['deadline_reached']and d['legacy_stable']and not d['joint_pending']
  assert [r['public_AD']['area'],r['public_AD']['delay']]==d['final_AD']
 assert a['netlist_sha256']==b['netlist_sha256']and a['replay']['initial_AD']==b['replay']['initial_AD']and a['replay']['final_AD']==b['replay']['final_AD']
 da,db=[r['replay']['trial_diagnostics']for r in fixed]
 counter_diffs={k:[da[k],db[k]]for k in da if isinstance(da[k],(int,float))and not isinstance(da[k],bool)and not k.endswith('secs')and da[k]!=db[k]};assert not counter_diffs
 provenance=[]
 for mode in ['construct','borrow']:
  case=O/'fixed-work-v2'/mode;last=sorted((case/'incumbents').glob('[0-9]'*8))[-1]
  evidence=replay(last/'manifest.json',case/'selected-provenance/objects',case/'mapped.json');assert evidence['valid']and evidence['netlist_binding_agreement'];provenance.append(dict(mode=mode,evidence=evidence))
 initial=[O/'fixed-work-v2'/mode/'incumbents/00000001/mapped.v'for mode in ['construct','borrow']];assert sha(initial[0])==sha(initial[1])
 source=read(R/'results_v2/emapper_physical_stamp_20261111/benchmark/cases/dec_r1_control/report.json');tools=source['tools']
 os.environ['EGM_YOSYS'],os.environ['BENCH_RS_EMAP_ABC']=tools['yosys']['path'],tools['abc']['path']
 folder=O/'shared-initial';folder.mkdir();lib=R/'emapper/libraries/asap7/asap7_full.genlib'
 cmd=[tools['yosys']['path'],'-Q','-T','-p',f'read_verilog -lib "{lib.parent / "cells.v"}"; read_verilog "{initial[0]}"; hierarchy -check -auto-top; rename -top mapped; write_json "{folder / "mapped.json"}"']
 p=subprocess.run(cmd,capture_output=True,text=True);(folder/'normalize.log').write_text(p.stdout+p.stderr);p.check_returncode()
 golden=R/'tools/mockturtle/experiments/benchmarks/dec.aig';assert verify(lib.parent/'cells.v',golden,folder/'mapped.json',folder/'cec')
 profile=read(O/'sin-callgrind/final-analysis-v2.json');assert profile['totals']['Ir']>0
 arrays=read(O/'json-array-results.json');assert all(r['equal_JSON_value']and r['equal_serialized_bytes']and r['original_array_storage_retained']==[False,True]for r in arrays['rows'])
 result=dict(original_source_binary_harness_frozen=True,new_full_mapping_pair_CEC=2,new_full_pair_identical_committed_trajectories=True,fixed_replay_final_CEC=2,fixed_replay_selected_evidence=provenance,shared_byte_identical_initial_CEC=1,fixed_replay_trial_counter_differences=counter_diffs,profile_periodic_parts=len(profile['files']),micro_construction_fixtures=len(arrays['rows']),invalid_pilot='fixed-work/: detached producer frontend overlapped first replay; explicitly invalid, interrupted and retained; no timing claim; its earlier atomic snapshot was restored/validated/CEC checked in v2',baseline_runs=0,mapper_changes=False,default_promoted=False,GitHub_push=False,limits='600-second unchanged mapping deadline; profile is instrumented prefix, no PPA claim; original historical CPU/load attribution remains unidentifiable')
 (O/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
 # Flush our LAST stdout write before hashing artifacts, including a redirected
 # audit.log. No later print may mutate the just-recorded log hash.
 print(json.dumps(result,indent=2),flush=True)
 artifacts={str(p.relative_to(R)):sha(p)for p in sorted(O.rglob('*'))if p.is_file()and p.name not in ['runner.lock','verified-artifacts.json']}
 (O/'verified-artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')
if __name__=='__main__':main()
