#!/usr/bin/env python3
"""Bounded diagnostic proofs for simulation collisions. Never changes mapper inputs/classes."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_epfl_structural_budget import read_binary_aig
R=Path(__file__).resolve().parents[3]
def delta(n):
 b=bytearray()
 while n>=128:b.append((n&127)|128);n>>=7
 b.append(n);return b
def miter(n,gates,f,g,phase):
 last=max(n,f,g);nodes=[gates[v] for v in range(n+1,last+1)];x=f*2;y=g*2^phase
 nodes.extend([(x,y^1),(x^1,y),((last+1)*2^1,(last+2)*2^1)])
 maximum=last+3;out=bytearray(f'aig {maximum} {n} 0 1 {len(nodes)}\n{maximum*2+1}\n'.encode())
 for v,(a,b) in enumerate(nodes,n+1):
  a,b=max(a,b),min(a,b);assert 0<=b<=a<2*v;out.extend(delta(2*v-a));out.extend(delta(a-b))
 out.extend(b'c\nSource-function mismatch diagnostic; original prepared input unchanged.\n');return bytes(out)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();signature=R/'results_v2/zero_cost_source_implementation_20261016/signature_probe.json';data=json.loads(signature.read_text());results=[]
 for row in data['rows']:
  b=row['bench'];src=R/f'results_v2/epfl_resyn2_subjects_20261012/{b}.resyn2.aig';assert sha(src)==row['input_sha256'];n,_,gates=read_binary_aig(src.read_bytes());v=[False]*(n+len(gates)+1)
  for i,(x,y) in gates.items():v[i]=(v[x//2]^bool(x&1))&(v[y//2]^bool(y&1))
  groups=row['results'][-1]['example_variable_groups'];pairs=[(groups[0][0],groups[0][1]),(groups[0][0],groups[0][-1]),(groups[1][0],groups[1][-1])]
  for f,g in pairs:
   phase=int(v[f]^v[g]);stem=f'{b}_{f}_{g}';path=a.output/(stem+'.aig');path.write_bytes(miter(n,gates,f,g,phase));argv=['/usr/bin/abc','-c',f'read_aiger "{path.resolve()}"; strash; sat -C 10000 -I 1000000'];timeout=False
   try:r=subprocess.run(argv,capture_output=True,text=True,timeout=10);text=r.stdout+r.stderr;rc=r.returncode
   except subprocess.TimeoutExpired as e:
    timeout=True;rc=None;text=(e.stdout or b'').decode(errors='replace')+(e.stderr or b'').decode(errors='replace')+'\nDiagnostic timeout10seconds; no equality inferred.\n'
   log=a.output/(stem+'.log');log.write_text(text);status='proved_equal_or_complement' if 'UNSATISFIABLE' in text and not timeout else 'counterexample' if 'SATISFIABLE' in text and not timeout else 'unknown'
   record=dict(bench=b,variable_a=f,variable_b=g,proposed_complement=bool(phase),status=status,argv=argv,returncode=rc,timeout_seconds=10,conflict_limit=10000,inspection_limit=1000000,miter=str(path.resolve()),miter_sha256=sha(path),log=str(log.resolve()),log_sha256=sha(log),prepared_sha256=sha(src),abc_sha256=sha(Path('/usr/bin/abc')));results.append(record);print(b,f,g,status,flush=True)
  (a.output/'report.json').write_text(json.dumps(dict(results=results,signature_report_sha256=sha(signature),probe_sha256=sha(Path(__file__)),original_mapper_or_baselines_changed=False,classes_merged=0,scope='Read-only diagnostic SATmiter: each output is f XOR(g XOR proposedphase) over ALL original unconstrained PIs. UNSAT proves this pair only; SAT refutes this simulation collision. Does not prove whole groups, bounded import, mapping feasibility or PPA. No input-specific proof/advice is fed to any official mapper run; future implementation must charge all work inside original native/mapping budget.'),indent=2)+'\n')
if __name__=='__main__':main()
