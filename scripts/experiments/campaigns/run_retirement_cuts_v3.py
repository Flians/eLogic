#!/usr/bin/env python3
"""Isolated cut retirement experiment; never relabel resident budget as old import budget."""
import argparse, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/experiments/prototypes'))
from region_stitch import read_binary_aig, symbols
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def export(subject):
    data=subject.read_bytes();i,roots,gates=read_binary_aig(data);ins,outs=symbols(data,i,roots,gates)
    live=set();todo=[r//2 for r in roots]
    while todo:
        v=todo.pop()
        if v in gates and v not in live:live.add(v);todo.extend(x//2 for x in gates[v])
    return {'inputs':ins,'outputs':list(zip(roots,outs)),'gates':[(v,*gates[v]) for v in sorted(live)]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--designs',nargs='+');ap.add_argument('--repetitions',type=int,default=1);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=ROOT/'research/state_retirement_cuts_v3_20261014/target/release/state_retirement_cuts_v3_probe';evaluator=ROOT/'target/release/bench_rs_netlist_eval'
    subjects=ROOT/'results_v2/epfl_resyn2_subjects_20261012/report.json';prep=json.loads(subjects.read_text())
    benches=a.designs or [r['bench'] for r in prep['results']]
    manifest={'scope':'regional-cuts actual-egg resident-state diagnostic, not old same-import-budget mapper or optimized regional saturation','baseline_algorithms_changed':False,'node_limit':20000,'retained_pool_limit':100000,'recovery_seconds':2,'mapping_external_seconds':300,'process_guard_seconds':60,'commands':[], 'results':[], 'hashes':{str(p):sha(p) for p in [binary,evaluator,subjects,Path(__file__),ROOT/'research/state_retirement_cuts_v3_20261014/src/main.rs']}}
    def run(cmd,log,mapping=False):
        begin=time.perf_counter();entry={'argv':[str(x) for x in cmd],'mapping':mapping};manifest['commands'].append(entry)
        with subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True) as p:
            try:stdout,stderr=p.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);stdout,stderr=p.communicate();entry['timeout']=True
            entry.update(seconds=time.perf_counter()-begin,exit=p.returncode);log.write_text(stdout+stderr)
            if p.returncode:raise RuntimeError(f'exit {p.returncode}: {log}')
        return stdout,entry['seconds']
    for repetition in range(a.repetitions):
      for k in [3,4]:
       reference=ROOT/f'results_v2/flow_saturation_main_checkpoint_20261014/{k}in_sqrt_flow_sat'
       lib=reference/f'asap7_{k}in.genlib';models=reference/'cells.v'
       for name in benches:
        folder=out/f'{k}in_{name}_r{repetition}';folder.mkdir();subject=ROOT/f'results_v2/epfl_resyn2_subjects_20261012/{name}.resyn2.aig';golden=ROOT/f'tools/mockturtle/experiments/benchmarks/{name}.aig'
        row={'bench':name,'k':k,'repetition':repetition,'input_sha256':sha(golden),'prepared_sha256':sha(subject),'library_sha256':sha(lib),'formal_same_import_budget_eligible':False}
        try:
            encoded=folder/'subject.json';encoded.write_text(json.dumps(export(subject)))
            raw=folder/'raw.v';audit=folder/'audit.json'
            text,row['mapping_wall_seconds']=run([str(binary),str(lib),str(encoded),str(raw),str(audit)],folder/'map.log',True)
            row['native']=json.loads(text);row['budget']=json.loads(audit.read_text())
            norm=folder/'mapped.v';netjson=folder/'mapped.json'
            script=f'read_verilog -lib "{models}"; read_verilog "{raw}"; hierarchy -check -auto-top; rename -top mapped; write_verilog -noattr -noexpr "{norm}"; write_json "{netjson}"'
            run(['yosys','-Q','-T','-p',script],folder/'normalize.log')
            text,row['evaluation_wall_seconds']=run([str(evaluator),str(lib),str(golden),str(norm),str(netjson)],folder/'evaluate.log')
            records=[json.loads(x) for x in text.splitlines() if x.startswith('{"area":')];assert len(records)==1;row['measured']=records[0]
            if row['measured']['cec']!=1:raise RuntimeError('named-port CEC failed')
            if abs(row['native']['area']-row['measured']['area'])>1e-5 or abs(row['native']['delay']-row['measured']['delay'])>1e-5:raise RuntimeError('common physical metrics disagree')
            row['status']='diagnostic_cec_ok';row['formal_same_import_budget_eligible']=row['budget']['old_cumulative_import_budget_compliant']
        except Exception as e:row.update(status='failed',error=str(e))
        row['artifact_sha256']={str(p):sha(p) for p in folder.iterdir() if p.is_file()};manifest['results'].append(row)
        (out/'report.json').write_text(json.dumps(manifest,indent=2)+'\n');print(k,name,repetition,row['status'],flush=True)
    print('finite experiment ended',flush=True)

if __name__=='__main__':main()
