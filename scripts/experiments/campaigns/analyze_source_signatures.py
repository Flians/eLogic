#!/usr/bin/env python3
"""Simulation-only candidates for exact source aliases; no equivalence or import-savings proof."""
import argparse,hashlib,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_epfl_structural_budget import read_binary_aig
R=Path(__file__).resolve().parents[3]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();subjects=R/'results_v2/epfl_resyn2_subjects_20261012/report.json';report=json.loads(subjects.read_text());source={r['bench']:r for r in report['results']};rows=[]
 for bench in ['div','hyp','mem_ctrl']:
  f=subjects.parent/(bench+'.resyn2.aig');assert sha(f)==source[bench]['preprocessing']['sha256'];n,roots,gates=read_binary_aig(f.read_bytes());rng=random.Random(196016);random_bits=[rng.getrandbits(4096) for _ in range(n)];results=[]
  for samples in [64,1024,4096]:
   width=samples+2+2*n;mask=(1<<width)-1;values=[0]*(n+len(gates)+1)
   for i in range(n):
    struct=(1<<1)|(1<<(2+i))
    for j in range(n):
     if i!=j:struct|=1<<(2+n+j)
    values[i+1]=(random_bits[i]&((1<<samples)-1))|(struct<<samples)
   groups={}
   # Includes all source functions: a subset can have no MORE true alias savings.
   for v in range(n+1):
    sig=values[v];key=min(sig,mask^sig);groups.setdefault(key,[]).append(v)
   for v,(left,right) in gates.items():
    x=values[left//2]^(mask if left&1 else 0);y=values[right//2]^(mask if right&1 else 0);sig=x&y;values[v]=sig;key=min(sig,mask^sig);groups.setdefault(key,[]).append(v)
   aliases=sum(len(g)-1 for g in groups.values());ambiguous=[g for sig,g in groups.items() if sig!=0 and len(g)>1]
   results.append(dict(random_patterns=samples,structured_patterns=2+2*n,source_function_count=len(values),signed_signature_groups=len(groups),direct_class_merges_upper=aliases,constant_signature_group_size=len(groups.get(0,[])),nonconstant_collision_groups=len(ambiguous),nonconstant_direct_merges_upper=sum(len(g)-1 for g in ambiguous),example_variable_groups=sorted(ambiguous,key=lambda g:(-len(g),g[0]))[:8]))
  rows.append(dict(bench=bench,input_sha256=sha(f),inputs=n,original_ands=len(gates),results=results))
 a.output.write_text(json.dumps(dict(subject_report_sha256=sha(subjects),analyzer_sha256=sha(Path(__file__)),parser_sha256=sha(R/'scripts/experiments/analyze_epfl_structural_budget.py'),seed=196016,rows=rows,scope='Read-only simulation, not mapped candidate, equivalence proof, budget-feasible import or PPA. True equal/complement functions necessarily share signatures; collisions only bound DIRECT class coalescences from above, NOT total import savings: removing an alias may also reclaim its unique ancestor cone. No import lower bound or impossibility conclusion follows. No classes are merged or baseline/prepared inputs changed. All-zero/one signatures especially require proof. Variable IDs are original AIGER variables, NOT egraph IDs.'),indent=2)+'\n')
 print([(r['bench'],r['results'][-1]['direct_class_merges_upper'],r['results'][-1]['nonconstant_direct_merges_upper']) for r in rows])
if __name__=='__main__':main()
