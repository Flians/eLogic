#!/usr/bin/env python3
"""Independently decode AIGER and verify exact distinguishing input witnesses."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[3]
p=ROOT/'results_v2/flow_saturation_implementation_20261014/retained_signal_certificate.json';cert=json.loads(p.read_text());subjects=json.loads((ROOT/'results_v2/epfl_resyn2_subjects_20261012/report.json').read_text());verified=[]
for r in cert['results']:
    source=ROOT/f"results_v2/epfl_resyn2_subjects_20261012/{r['design']}.resyn2.aig";data=source.read_bytes();assert hashlib.sha256(data).hexdigest()==r['source_sha256']
    subject=next(x for x in subjects['results'] if x['bench']==r['design']);assert subject['preprocessing']['sha256']==r['source_sha256'] and subject['preprocessing']['cec']==1
    header,end=data.split(b'\n',1);kind,m,i,l,o,a=header.decode().split();m,i,l,o,a=map(int,[m,i,l,o,a]);assert kind=='aig' and l==0 and m==i+a
    outputs=[]
    for _ in range(o):line,end=end.split(b'\n',1);outputs.append(int(line))
    pos=0
    def take(position):
        result=0;shift=0
        while True:
            byte=end[position];position+=1;result|=(byte&127)<<shift
            if byte<128:return result,position
            shift+=7;assert shift<64
    gates={}
    for v in range(i+1,m+1):
        delta,pos=take(pos);x=2*v-delta;delta,pos=take(pos);y=x-delta;assert 0<=y<=x<2*v;gates[v]=(x,y)
    live=set();todo=[x//2 for x in outputs]
    while todo:
        v=todo.pop()
        if v in gates and v not in live:live.add(v);todo.extend(x//2 for x in gates[v])
    n=r['patterns'];mask=(1<<n)-1;values={0:0,**{int(k):int(v,16) for k,v in r['PI_masks_hex'].items()}};assert set(values)==set(range(i+1));assert all(v>=0 and v<=mask for v in values.values())
    for v in sorted(live):
        x,y=gates[v];left=values[x//2];right=values[y//2]
        if x&1:left=mask^left
        if y&1:right=mask^right
        values[v]=left&right
    witnesses=r['witness_representative_variables'];assert len(witnesses)==20001 and len(set(witnesses))==20001
    signs={min(values[v],mask^values[v]) for v in witnesses};assert len(signs)==20001
    all_signs={min(v,mask^v) for v in values.values()};assert len(all_signs)==r['distinct_signatures_up_to_complement']
    verified.append({'design':r['design'],'original_live_ANDs':len(live),'verified_function_pair_lower_bound':len(all_signs),'verified_20001_witnesses':True,'retaining_all_original_scalar_functions_exceeds_limit':True,'source_sha256':r['source_sha256']})
out=p.with_name('retained_signal_certificate_verified.json');out.write_text(json.dumps({'builder_certificate_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_decoder':True,'node_limit':20000,'scope':'Conditional bound on simultaneous retention of original scalar functions; NOT a minimum physical gate/area bound or impossibility of alternate mapping representations. Distinct signatures prove inequality; matching signatures do not establish equality.','verified':verified},indent=2)+'\n');print(verified)
