#!/usr/bin/env python3
"""Independent expression evaluator for the frozen sqrt cut phase replay tables."""
import importlib.util,itertools,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
path=Path(__file__).with_name('audit_sqrt_retained_phases.py')
spec=importlib.util.spec_from_file_location('audit',path);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
def evaluate(expr,values):
    k,v=next(iter(expr.items()))
    if k=='Var':return values[v]
    if k=='Not':return not evaluate(v,values)
    if k=='And':return all(evaluate(x,values) for x in v)
    if k=='Or':return any(evaluate(x,values) for x in v)
    raise ValueError(k)
checks=0
for k in [3,4]:
 lib=json.loads((ROOT/f'results_v2/flow_saturation_main_checkpoint_20261014/{k}in_sqrt_flow_sat/library.json').read_text())
 for cell in lib:
    if not any(cell['name'].startswith(f+'x') for f in ['AO31','AOI21','OAI21']):continue
    n=len(cell['vars'])
    for tt,rows in audit.variants(cell).items():
      for name,pins,neg in rows:
       for a in range(1<<n):
        values={pin:bool(((a>>i)&1)^((neg>>i)&1)) for i,pin in enumerate(pins)}
        assert evaluate(cell['expr'],values)==bool((tt>>a)&1);checks+=1
    # Both positive and negative target phases, without calling the table transformation.
    for a in range(1<<n):
        base=evaluate(cell['expr'],dict(zip(cell['vars'],[bool(a>>i&1) for i in range(n)])))
        assert base==bool(cell['tt'][0]>>a&1);assert (not base)==bool((cell['tt'][0]^((1<<(1<<n))-1))>>a&1)
report=ROOT/'results_v2/sqrt_retained_phase_audit_v2_20261014/report.json';d=json.loads(report.read_text())
assert all(c['exact_variants_retained']<=c['exact_cut_phase_variants_replayed'] for r in d['records'] for c in r['cells'])
(report.parent/'verification.json').write_text(json.dumps({'checks':checks,'independent_library_expression_evaluator':True,'both_target_phases_checked':True,'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limitation':'Does not prove every eligible pre-cap candidate survives cone pruning or that substitution is globally acyclic/timing beneficial.'},indent=2)+'\n')
print(checks,'independent exact expression checks passed')
