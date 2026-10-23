"""Qualify an unchanged complete mapped output with its existing independent proof."""
import argparse,csv,hashlib,json,subprocess
from pathlib import Path
from run_time_only600 import ROOT,sha
def signature(path):
    top=json.loads(path.read_text())['modules']['mapped']
    semantic=dict(ports=top['ports'],cells={name:{k:v for k,v in cell.items()if k not in ['attributes','hide_name']}for name,cell in top['cells'].items()})
    return hashlib.sha256(json.dumps(semantic,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['mockturtle_emap_area_resyn2','mockturtle_emap_delay_resyn2'],required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    study=ROOT/'results_v2/time_only_600_20261022';oldstudy=ROOT/'results_v2/single_output_wide_20261021';catalog=json.loads((oldstudy/'primary_baseline120_checkpoint.json').read_text())
    job=next(j for j in catalog['jobs']if j['name']=='r0_6in_core_hyp');old=Path(job['report']).parent;new=study/'original_baselines_epfl/r0_6in_core_hyp'
    row=next(r for r in csv.DictReader((oldstudy/'full_library_baseline_A_D_T/baseline_A_D_T.csv').open())if r['bench']=='hyp'and r['method']==a.method)
    history=next(h for h in json.loads(row['verification_history'])if h['status']=='ok')
    for f,h in history['sha256'].items():assert sha(f)==h
    protocol_path=next(Path(f)for f in history['sha256']if Path(f).name=='protocol.json');protocol=json.loads(protocol_path.read_text())
    proof_path=protocol_path.with_name('verification.json');proof=json.loads(proof_path.read_text())
    assert proof['cec']==1 and proof['original_gate_bytes_unchanged']and proof['functional_models_from_verilog_not_mapper_truth_tables']and not proof['mapper_called']
    previous=old/('hyp.'+a.method);current=new/('hyp.'+a.method)
    verilog=current/'mapped.v';netjson=current/'mapped.json';library=new/'asap7_6in.genlib';models=new/'cells.v';golden=ROOT/'tools/mockturtle/experiments/benchmarks/hyp.aig'
    assert sha(verilog)==protocol['physical_verilog_sha256']==sha(previous/'mapped.v')
    assert sha(previous/'mapped.json')==protocol['physical_json_sha256']
    assert signature(netjson)==signature(previous/'mapped.json')
    assert sha(library)==protocol['library_sha256']and sha(golden)==protocol['original_sha256']
    assert sha(models)==sha(old/'cells.v')
    measure=oldstudy/'source_kernels/target/release/bench_rs_wide_measure'
    assert sha(measure)==protocol['measurement_binary_sha256']
    argv=[str(measure),str(library),str(netjson)];result=subprocess.run(argv,capture_output=True,text=True,check=True)
    (out/'measurement.log').write_text(result.stdout+result.stderr);metrics=json.loads(result.stdout);metrics['cec']=1
    clock_path=new.parent/'r0_6in_core_hyp.audit.clock.json';clock=json.loads(clock_path.read_text());state=clock['cases']['hyp/'+a.method]
    assert not state['active']and not state['exhausted']and not state['incomplete']and state['used']<600
    (out/'mapping_clock_snapshot.json').write_text(json.dumps(clock,indent=2)+'\n')
    files={str(p):sha(p)for p in [verilog,netjson,library,models,golden,previous/'mapped.v',previous/'mapped.json',old/'cells.v',measure,Path(__file__)]}
    files.update(history['sha256']);files[str(out/'measurement.log')]=sha(out/'measurement.log');files[str(out/'mapping_clock_snapshot.json')]=sha(out/'mapping_clock_snapshot.json')
    record=dict(bench='hyp',method=a.method,source_job='r0_6in_core_hyp',current_report=str(new/'report.json'),
        qualification='independent_named_port_CEC_reused_for_byte_identical_physical_netlist',fresh_CEC_run=False,
        original_mapping_algorithm_changed=False,mapping_rerun=False,physical_netlist_unchanged=True,
        physical_verilog_sha256=sha(verilog),physical_json_semantic_sha256=signature(netjson),library_sha256=sha(library),
        original_input_sha256=sha(golden),measured=metrics,mapping_wall_secs=state['used'],
        mapping_completed_before_verification=True,measurement_argv=argv,prior_proof_protocol=str(protocol_path),
        proof_hashes_verified=True,sha256=files,operator_verification_journal=str(out/'operator_verification.json'))
    (out/'proof_reuse.json').write_text(json.dumps(record,indent=2)+'\n');print(a.method,metrics,'mapping seconds',state['used'])
if __name__=='__main__':main()
