#!/usr/bin/env python3
"""Read-only exact small-PI-support alias audit; not a mapped budget/CEC proof."""
from pathlib import Path
import sys,json,hashlib,functools
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_epfl_structural_budget import read_binary_aig
ROOT=Path(__file__).resolve().parents[3]
@functools.lru_cache(maxsize=200000)
def stretch(support, truth, merged):
    positions=[merged.index(v) for v in support];value=0
    for assignment in range(1<<len(merged)):
        local=sum(((assignment>>p)&1)<<i for i,p in enumerate(positions))
        value|=((truth>>local)&1)<<assignment
    return value
rows=[]
for name in ['div','hyp','log2','multiplier','mem_ctrl']:
    p=ROOT/f'results_v2/epfl_resyn2_subjects_20261012/{name}.resyn2.aig'
    inputs,roots,gates=read_binary_aig(p.read_bytes());needed=set();stack=[v//2 for v in roots]
    while stack:
        v=stack.pop()
        if v in gates and v not in needed:needed.add(v);stack.extend(x//2 for x in gates[v])
    functions={0:((),0),**{v:((v,),2) for v in range(1,inputs+1)}}
    canonical={v:v for v in range(inputs+1)};seen={f:v for v,f in functions.items()};aliases=0;unsupported=0;small=0;keys=set()
    for v in sorted(needed):
        a,b=gates[v];parts=[functions.get(a//2),functions.get(b//2)]
        key=tuple(sorted((2*canonical[a//2]+(a&1),2*canonical[b//2]+(b&1))))
        keys.add(key);canonical[v]=v
        if None in parts:unsupported+=1;continue
        merged=tuple(sorted(set(parts[0][0])|set(parts[1][0])))
        if len(merged)>6:unsupported+=1;continue
        mask=(1<<(1<<len(merged)))-1;ts=[stretch(*f,merged) for f in parts]
        if a&1:ts[0]^=mask
        if b&1:ts[1]^=mask
        f=(merged,ts[0]&ts[1]);functions[v]=f;small+=1
        if f in seen:canonical[v]=canonical[seen[f]];aliases+=1
        else:seen[f]=v
    rows.append({'design':name,'prepared_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'original_live_ANDs':len(needed),'all_PIs_retained':inputs,'exact_PI_support_le6_nodes':small,'unhandled_larger_support_nodes':unsupported,'exact_function_aliases_found':aliases,'commutative_AND_keys_after_prior_aliases':len(keys),'estimated_nodes_excluding_extra_Not_aliases':len(keys)+inputs+1,'caution':'No native rewrite/import implementation, physical match, CEC or budget feasibility proof. Alternative original AND enodes are not omitted; only identical syntactic keys can share storage. Function alias count is not a savings count.'})
out=ROOT/'results_v2/flow_cost_implementation_20261014/small_support_alias_audit.json';out.write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'max_global_PI_support':6,'results':rows,'interpretation':'Exact whole-PI truth functions, not conditional/dont-care equivalences; unhandled cones remain unmerged. This is a limited audit, not evidence that semantic reduction in general is impossible.'},indent=2)+'\n');print([(r['design'],r['exact_function_aliases_found'],r['estimated_nodes_excluding_extra_Not_aliases']) for r in rows])
