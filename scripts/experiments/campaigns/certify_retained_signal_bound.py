#!/usr/bin/env python3
"""Exact distinguishing-input certificate for retaining original scalar signals.

Distinct sample signatures prove inequality, never equality. This bound applies
only when every original live scalar function stays represented, even allowing
complemented edges. It is NOT a lower bound on optimized netlist area or gates.
"""
from pathlib import Path
import json,hashlib,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_epfl_structural_budget import read_binary_aig
ROOT=Path(__file__).resolve().parents[3]
COUNT=2048;FULL=(1<<COUNT)-1
rows=[]
for name in ['div','hyp','log2','multiplier','mem_ctrl']:
    p=ROOT/f'results_v2/epfl_resyn2_subjects_20261012/{name}.resyn2.aig';data=p.read_bytes();digest=hashlib.sha256(data).hexdigest()
    inputs,roots,gates=read_binary_aig(data);live=set();stack=[r//2 for r in roots]
    while stack:
        v=stack.pop()
        if v in gates and v not in live:live.add(v);stack.extend(l//2 for l in gates[v])
    masks={};rates=[2,16,64,128,192,240,254]
    for v in range(1,inputs+1):
        bits=0
        for block in range(COUNT//32):
            random=hashlib.sha256((digest+f':{v}:{block}').encode()).digest()
            lane=(v-1)//max(1,inputs//4)
            for bit,x in enumerate(random):
                pattern=block*32+bit
                rate=128 if block<8 else rates[(block//8+lane)%len(rates)]
                value=x<rate
                if 256<=pattern<512:value=((v-1)%max(1,inputs//2))==((pattern-256)%max(1,inputs//2)) if v>inputs//2 else x<128
                if value:bits|=1<<pattern
        masks[v]=bits
    values={0:0,**masks};reps={};representatives=[]
    for v in [0,*range(1,inputs+1),*sorted(live)]:
        if v in gates:
            a,b=gates[v];values[v]=(values[a//2]^(FULL if a&1 else 0))&(values[b//2]^(FULL if b&1 else 0))
        key=min(values[v],values[v]^FULL)
        if key not in reps:
            reps[key]=v
            if len(representatives)<20001:representatives.append(v)
    # Separately check bit-vector simulation against scalar Boolean evaluation.
    for pattern in [0,1,31,63,127,255,256,399,511,777,1023,1535,2047]:
        scalar={0:False,**{v:bool((masks[v]>>pattern)&1) for v in masks}}
        for v in sorted(live):
            a,b=gates[v];scalar[v]=(scalar[a//2]^bool(a&1)) and (scalar[b//2]^bool(b&1))
            assert scalar[v]==bool((values[v]>>pattern)&1)
    row={'design':name,'source_sha256':digest,'patterns':COUNT,'distinct_signatures_up_to_complement':len(reps),'exceeds_20000_under_all_original_scalar_functions_retained':len(reps)>20000,'witness_representative_variables':representatives,'PI_masks_hex':{str(v):hex(m) for v,m in masks.items()},'scalar_cross_checks':13,'meaning':'Each pair of distinct canonical signatures is separated by at least one stored input assignment. Therefore original functions cannot all be merged into<=20k scalar representatives. Not a proof that mapping with retired/elided intermediate states or an optimized netlist is impossible.'}
    assert len({min(values[v],values[v]^FULL) for v in representatives})==len(representatives)
    rows.append(row)
out=ROOT/'results_v2/flow_saturation_implementation_20261014/retained_signal_certificate.json';out.write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'node_limit':20000,'scope':'Conditional lower bound on all-original-scalar-function retention, allowing phase edges; not PPA or general synthesis impossibility','results':rows},indent=2)+'\n');print([(r['design'],r['distinct_signatures_up_to_complement'],r['exceeds_20000_under_all_original_scalar_functions_retained']) for r in rows])
