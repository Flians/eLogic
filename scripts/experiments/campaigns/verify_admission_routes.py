#!/usr/bin/env python3
"""Exercise each CLI import route at small exact capacities, with original-port CEC."""
import json,os,subprocess,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];O=R/'results_v2/cumulative_admission_20261019/cli_routes'
def main():
 O.mkdir(exist_ok=False);records=[]
 source=R/'results_v2/main_integration_20261019/joint_source_cuts/joint.aig'
 routes={'ordinary':{},'demanded':{'EGM_REGION_CLASSES':'256'},'literal':{'EGM_LITERAL_IMPORT':'1'},'cone':{'EGM_CONE_IMPORT':'1','EGM_LITERAL_AUTO':'1'},'library_source':{'EGM_CONE_IMPORT':'1','EGM_LITERAL_AUTO':'1','EGM_SHARED_MAJORITY':'1','EGM_PROOF_SOURCE':'1','EGM_SOURCE_PROPAGATION':'1'}}
 for name,flags in routes.items():
  for cap in [0,5,6,16]:
   d=O/f'{name}_{cap}';d.mkdir();s={'EGM_K':'3','EGM_NODES':str(cap),'EGM_ITER':'16','EGM_CUTS':'16','EGM_INSTANCES':'100000','EGM_STRICT_BUDGETS':'1','EGM_TIME':'0','EGM_REFINE_TIME':'2','EGM_OBJECTIVE':'delay-area','EGM_MULTIOUTPUT':'0','EGM_REGION_CLASSES':'0','EGM_CONE_IMPORT':'0','EGM_LITERAL_AUTO':'0','EGM_PROOF_SOURCE':'0','EGM_CANONICAL_PHASE':'1','EGM_CANONICAL_WIRES':'1','EGM_POOL_DIAG_PATH':str(d/'pool.json'),**flags}
   e={k:v for k,v in os.environ.items() if not k.startswith('EGM_')};e.update(s)
   argv=[str(R/'target/release/bench_rs_egraph_map'),str(R/'tools/mockturtle/experiments/cell_libraries/asap7.genlib'),str(source),str(d/'mapped.v')]
   p=subprocess.run(argv,env=e,text=True,capture_output=True,timeout=60);(d/'stdout.log').write_text(p.stdout);(d/'stderr.log').write_text(p.stderr)
   audit=json.loads((d/'pool.admission.json').read_text());assert audit.get('cumulative_admitted_enodes')<=cap
   values=[json.loads(x)for x in p.stdout.splitlines() if x.startswith('{"area":')]
   if p.returncode==0:
    assert audit['import_complete'] and len(values)==1 and values[0]['cec']==1
    n=values[0];assert n['enodes']<=n['cumulative_admitted_enodes']<=cap and n['imported_enodes']<=n['cumulative_imported_enodes']<=cap
   else:
    assert not values and not audit['import_complete'] and audit['rejected_before_insertion']
    assert audit['cumulative_admitted_enodes']==cap and 'cannot import AIG within node allowance' in p.stderr
   records.append(dict(route=name,limit=cap,settings=s,argv=argv,exit_code=p.returncode,admission=audit,native=values))
 v=dict(records=records,CEC=sum(x['exit_code']==0 for x in records),expected_rejections=sum(x['exit_code']!=0 for x in records),binary_sha256=hashlib.sha256((R/'target/release/bench_rs_egraph_map').read_bytes()).hexdigest(),scope='Small API budget boundary correctness tests, not EPFL PPA. Original indexed and unused ports retained; source transformations are checked by independent ABC CEC, no tighter formal EPFL budget adopted.')
 (O/'validation.json').write_text(json.dumps(v,indent=2)+'\n');print({k:v[k]for k in ['CEC','expected_rejections']})
if __name__=='__main__':main()
