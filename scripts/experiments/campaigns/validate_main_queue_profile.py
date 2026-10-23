#!/usr/bin/env python3
"""Main opt-in preset versus explicit equivalent settings; keep every observation."""
import argparse,json,os,subprocess,time
from pathlib import Path
from run_epfl_baselines import ROOT,sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False)
 c=json.loads(a.source_campaign.read_text());runs=[]
 mapper=ROOT/'target/release/bench_rs_egraph_map';evaluator=ROOT/'target/release/bench_rs_netlist_eval'
 for k,bench in [(3,'adder'),(4,'max')]:
  j=next(j for j in c['jobs'] if j['k']==k and j['bench']==bench and j['profile']=='coupled_queue');r=json.loads(Path(j['report']).read_text())['results'][0]
  folder=Path(j['report']).parent;genlib=next(folder.glob('asap7_*in.genlib'));prepared=folder/(bench+'.resyn2.aig');gold=ROOT/'tools/mockturtle/experiments/benchmarks'/(bench+'.aig')
  for mode in ['preset','explicit']:
   target=out/f'{k}in_{bench}_{mode}';target.mkdir();env={key:value for key,value in os.environ.items() if not key.startswith('EGM_')};env.update(r['settings']);env['EGM_YOSYS']='/usr/local/bin/yosys'
   if mode=='preset':
    for key in ['EGM_CUTS','EGM_DC_SUPPORT','EGM_TIME','EGM_OBJECTIVE','EGM_FLOW_WORKLIST','EGM_STRICT_BUDGETS','EGM_REGION_CLASSES','EGM_PHASE_RECOVERY','EGM_PHASE_TIME_FRACTION']:env.pop(key,None)
    env['EGM_RECOVERY_PROFILE']='queue-delay'
   argv=[str(mapper),str(genlib),str(prepared),str(target/'mapped.v')];t=time.monotonic();p=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=60);wall=time.monotonic()-t
   (target/'map.stdout.log').write_text(p.stdout);(target/'map.stderr.log').write_text(p.stderr);assert p.returncode==0,p.stderr
   native=next(json.loads(line) for line in p.stdout.splitlines() if line.startswith('{"area":'))
   nargv=['/usr/local/bin/yosys','-Q','-T','-p',f'read_verilog -lib "{folder}/cells.v"; read_verilog "{target}/mapped.v"; hierarchy -check -top mapped; write_json "{target}/mapped.json"']
   q=subprocess.run(nargv,capture_output=True,text=True,timeout=60);(target/'normalize.log').write_text(q.stdout+q.stderr);assert q.returncode==0
   eargv=[str(evaluator),str(genlib),str(gold),str(target/'mapped.v'),str(target/'mapped.json')];q=subprocess.run(eargv,env=env,capture_output=True,text=True,timeout=60);(target/'evaluate.log').write_text(q.stdout+q.stderr);assert q.returncode==0
   measured=next(json.loads(line) for line in q.stdout.splitlines() if line.startswith('{"area":'));assert measured['cec']==1
   assert native['enodes']<=20000 and native['max_cuts_per_class']<=16 and native['instances']<=100000 and native['heuristic_secs']<=2
   runs.append({'k':k,'bench':bench,'mode':mode,'argv':argv,'normalization_argv':nargv,'evaluation_argv':eargv,'settings':{key:value for key,value in env.items() if key.startswith('EGM_')},'native':native,'measured':measured,'mapping_wall_secs':wall,'input_sha256':sha(prepared),'library_sha256':sha(genlib)})
   print(k,bench,mode,measured,flush=True)
 (out/'validation.json').write_text(json.dumps({'runs':runs,'main_mapper_sha256':sha(mapper),'evaluator_sha256':sha(evaluator),'legacy_default_preserved':True,'baseline_algorithms_changed':False,'scope':'two targeted main preset/explicit acceptance pairs, not full-scope isolated results relabeled main'},indent=2))
if __name__=='__main__':main()
