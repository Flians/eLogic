#!/usr/bin/env python3
"""Synthetic boundary integration, not an EPFL performance observation."""
import os,json,hashlib,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[3];O=R/'results_v2/main_integration_20261019/joint_source_cuts';O.mkdir(exist_ok=False)
aig=O/'joint.aig';aig.write_bytes(b'aig 7 4 0 1 3\n14\n'+bytes([6,2,6,4,1,2])+b'i0 a[0]\ni1 b[0]\ni2 c[0]\ni3 unused[0]\no0 y[0]\nc\nexact !(a*b|a*c), four original PIs\n')
records=[]
for name,shared,limit,ok in [('no_source',0,6,False),('joint_enabled',1,6,True),('too_small',1,5,False)]:
 folder=O/name;folder.mkdir();settings={'EGM_K':'3','EGM_CUTS':'16','EGM_INSTANCES':'100000','EGM_ITER':'0','EGM_NODES':str(limit),'EGM_TIME':'0','EGM_OBJECTIVE':'delay-area','EGM_REFINE_TIME':'2','EGM_STRICT_BUDGETS':'1','EGM_MULTIOUTPUT':'0','EGM_CONE_IMPORT':'1','EGM_LITERAL_AUTO':'1','EGM_CANONICAL_PHASE':'1','EGM_CANONICAL_WIRES':'1','EGM_SLACK_FLOW':'1','EGM_TERMINAL_AREA':'1','EGM_FUNCTIONAL_CUTS':'1','EGM_FUNCTIONAL_PHASE_SCREEN':'0','EGM_SHARED_MAJORITY':str(shared),'EGM_PROOF_SOURCE':str(shared),'EGM_SOURCE_PROPAGATION':str(shared),'EGM_SOURCE_GROUPS':'1','EGM_JOINT_SOURCE':'1','EGM_ZERO_COST_SOURCE':'1','EGM_SOURCE_RANK_PRUNING':'1','EGM_SOURCE_TERMINAL_GROUPS':'1','EGM_POOL_DIAG_PATH':str(folder/'pool.json')}
 env={k:v for k,v in os.environ.items() if not k.startswith('EGM_')};env.update(settings);argv=[str(R/'target/release/bench_rs_egraph_map'),str(R/'tools/mockturtle/experiments/cell_libraries/asap7.genlib'),str(aig),str(folder/'mapped.v')];p=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=60);(folder/'stdout.log').write_text(p.stdout);(folder/'stderr.log').write_text(p.stderr)
 assert (p.returncode==0)==ok,(name,p.returncode,p.stderr[-1000:]);data=[json.loads(x)for x in p.stdout.splitlines() if x.startswith('{"area":')]
 if ok:
  assert len(data)==1 and data[0]['cec']==1 and data[0]['imported_enodes']<=limit and data[0]['enodes']<=limit and data[0]['heuristic_secs']<=2
  for symbol in ['a[0]','b[0]','c[0]','unused[0]','y[0]']:assert symbol in (folder/'mapped.v').read_text(),symbol
 else:assert 'node allowance'in p.stderr
 records.append(dict(case=name,argv=argv,settings=settings,exit_code=p.returncode,result=data))
(O/'validation.json').write_text(json.dumps(dict(cases=records,main_sha256=hashlib.sha256((R/'target/release/bench_rs_egraph_map').read_bytes()).hexdigest(),scope='Synthetic combined source/cuts/phase/terminal-area integration. Exact original ports and ABC CEC; tighter6/5-node boundary tests, not EPFL PPA or changed formal20k allowance.'),indent=2)+'\n');print([(r['case'],r['exit_code'])for r in records])
