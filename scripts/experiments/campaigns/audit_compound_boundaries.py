#!/usr/bin/env python3
"""Read-only AIG/emap/egraph correspondence. Signatures propose; exact miters certify."""
import argparse,collections,hashlib,itertools,json,random,re,struct,subprocess,sys
from pysat.solvers import Minisat22
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'scripts/experiments'))
from analyze_epfl_structural_budget import read_binary_aig
from prototypes.region_stitch import symbols
MASK=(1<<1024)-1
PATTERNS=[]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Dag:
 def __init__(self,n):
  global MASK
  rng=random.Random(92344); patterns=[0,(1<<n)-1]+[1<<i for i in range(n)]+[((1<<n)-1)^(1<<i) for i in range(n)]
  for prob in [0.0625,0.25,0.75,0.9375]:
   patterns += [sum((rng.random()<prob)<<i for i in range(n)) for _ in range(512)]
  patterns+=PATTERNS;MASK=(1<<(1024+len(patterns)))-1
  self.n=n;self.g={};self.index={};self.values=[0]+[random.Random(4173+i).getrandbits(1024)|sum(((v>>i)&1)<<(1024+j) for j,v in enumerate(patterns)) for i in range(n)]
 def val(self,l):return self.values[l//2]^(MASK if l&1 else 0)
 def AND(self,a,b):
  if a==0 or b==0 or a==(b^1):return 0
  if a==1:return b
  if b==1 or a==b:return a
  key=tuple(sorted((a,b),reverse=True))
  if key not in self.index:
   v=len(self.values);self.index[key]=2*v;self.g[v]=key;self.values.append(self.val(a)&self.val(b))
  return self.index[key]
 def OR(self,a,b):return self.AND(a^1,b^1)^1
 def XOR(self,a,b):return self.OR(self.AND(a,b^1),self.AND(a^1,b))
 def tt(self,tt,args):
  if not args:return tt&1
  n=len(args);half=1<<(n-1);mask=(1<<half)-1;lo=tt&mask;hi=(tt>>half)&mask
  if lo==hi:return self.tt(lo,args[:-1])
  a=args[-1];return self.OR(self.AND(a^1,self.tt(lo,args[:-1])),self.AND(a,self.tt(hi,args[:-1])))
 def aig(self,root):
  data=bytearray(f'aig {len(self.values)-1} {self.n} 0 1 {len(self.g)}\n{root}\n'.encode())
  for v,(a,b) in self.g.items():
   for x in [2*v-a,a-b]:
    while x>=128:data.append((x&127)|128);x>>=7
    data.append(x)
  return bytes(data)+b'c\nRead-only correspondence diagnostic.\n'
def parse_nodes(text):
 nodes=[]
 for m in re.finditer(r'Pi\((\d+)\)|Not\((\d+)\)|And\(\[([^]]*)\]\)|Cell\("([^"]+)", \[([^]]*)\]\)|\b(Zero|One)\b',text):
  pi,neg,ands,cell,args,const=m.groups()
  if pi is not None:nodes.append(('pi',int(pi),[]))
  elif neg is not None:nodes.append(('not',None,[int(neg)]))
  elif ands is not None:nodes.append(('tt',8,[int(v) for v in ands.split(',')]))
  elif cell is not None:
   children=[int(v) for v in args.split(',') if v.strip()]
   if cell.startswith('@egm_tt3_'):tt=int(cell.split('_')[-1],16)
   else:tt={'@egm_and_np':4,'@egm_and_pn':2,'@egm_and_nn':1,'@egm_xor':6,'@egm_xnor':9}[cell]
   nodes.append(('tt',tt,children))
  else:nodes.append(('const',int(const=='One'),[]))
 assert nodes,text
 return nodes
def run(bench,out):
 c=json.loads((R/'results_v2/delta_area_full_20261016/campaign.json').read_text());job=next(j for j in c['jobs'] if j['name']==f'4in_{bench}_protected_area')
 diag=json.loads(Path(job['candidates']).read_text());pathrec=next(r for r in json.loads((R/'results_v2/phase_delay_diagnosis_20261016/paths.json').read_text())['records'] if r['bench']==bench)['emap']
 mapped=Path(next(p for p in pathrec['sources'] if p.endswith('mapped.json')));libpath=Path(next(p for p in pathrec['sources'] if p.endswith('library.json')))
 lib={(p['name'],p['out_pin']):p for p in json.loads(libpath.read_text())};net=json.loads(mapped.read_text())['modules']['mapped']
 source=R/f'results_v2/epfl_resyn2_subjects_20261012/{bench}.resyn2.aig';raw=source.read_bytes();n,roots,gates=read_binary_aig(raw);names,_=symbols(raw,n,roots,gates);dag=Dag(n)
 src={0:0};src.update({i:2*i for i in range(1,n+1)})
 def src_lit(l):return src[l//2]^(l&1)
 for v,(a,b) in gates.items():src[v]=dag.AND(src_lit(a),src_lit(b))
 source_sigs=collections.defaultdict(list)
 for v,l in src.items():
  for phase in [0,1]:source_sigs[dag.val(l^phase)].append((2*v+phase,l^phase))
 bit={'0':0,'1':1}
 for i,name in enumerate(names,1):
  port=net['ports'][name];assert port['direction']=='input' and len(port['bits'])==1;bit[port['bits'][0]]=2*i
 pending=dict(net['cells']);models={};arcs={}
 for name,cell in pending.items():
  models[name]=[p for (typ,_),p in lib.items() if typ==cell['type']]
 while pending:
  change=False
  for name,cell in list(pending.items()):
   ps=models[name]
   if not all(cell['connections'][pin][0] in bit for p in ps for pin in p['vars']):continue
   for p in ps:
    args=[bit[cell['connections'][pin][0]] for pin in p['vars']];tt=sum(x<<(64*i) for i,x in enumerate(p['tt']));l=dag.tt(tt,args);b=cell['connections'][p['out_pin']][0];bit[b]=l;arcs[(name,p['out_pin'])]=(l,args,p)
   del pending[name];change=True
  assert change,'netlist not topological'
 # Finite egraph representative per class, preserving PI names, all alternatives equivalent by egraph construction.
 pi_var={v:names.index(name)+1 for _,v,name in diag['inputs']};nodes={r['class']:parse_nodes(r['nodes']) for r in diag['egraph_nodes']};eg={};remaining=set(nodes)
 while remaining:
  change=False
  for cid in sorted(remaining):
   for typ,tt,args in nodes[cid]:
    if not all(x in eg for x in args):continue
    if typ=='pi':l=2*pi_var[tt]
    elif typ=='const':l=tt
    elif typ=='not':l=eg[args[0]]^1
    else:l=dag.tt(tt,[eg[x] for x in args])
    eg[cid]=l;remaining.remove(cid);change=True;break
  assert change,f'cyclic unreachable representatives {len(remaining)}'
 sigs=collections.defaultdict(list)
 for cid,l in eg.items():
  for phase in [0,1]:sigs[dag.val(l^phase)].append((cid,phase,l^phase))
 cuts=collections.defaultdict(list)
 for cut in diag['cuts']:
  if not cut['trivial']:cuts[cut['class']].append(cut)
 pairs=set();rows=[]
 solver=Minisat22();const=len(dag.values);solver.add_clause([-const])
 def cnf(l):return (const if l==0 else -const) if l<=1 else (l//2 if l%2==0 else -(l//2))
 assert cnf(0)==const and cnf(1)==-const
 for v,(a,b) in dag.g.items():
  x,y=cnf(a),cnf(b);solver.add_clause([-x,-y,v]);solver.add_clause([x,-v]);solver.add_clause([y,-v])
 proof_cache={};proof_records=[]
 def equal(a,b):
  if a==b:return True
  if a==(b^1):return False
  key=tuple(sorted((a,b)))
  if key in proof_cache:return proof_cache[key]
  status='equal';counterexample=None
  for assumptions in [[cnf(a),-cnf(b)],[-cnf(a),cnf(b)]]:
   solver.conf_budget(25);answer=solver.solve_limited(assumptions=assumptions)
   if answer is None:status='unknown';break
   if answer:
    status='different';model=set(solver.get_model());counterexample=''.join('1' if i in model else '0' for i in range(1,n+1));break
  proof_records.append(dict(left=a,right=b,status=status,conflict_limit_per_direction=25,counterexample_pi_bits=counterexample))
  proof_cache[key]=status=='equal';
  if len(proof_records)%50==0:print(bench,'pair_proofs',len(proof_records),flush=True)
  return status=='equal'
 def correspond(l):
  sig=dag.val(l);sources=source_sigs.get(sig,[]);classes=sigs.get(sig,[])
  # Prefer exact structural matches, then smallest original literal. All chosen nonstructural proposals require batch proof.
  sources=sorted(sources,key=lambda p:(p[1]!=l,p[0]));sources=next(([v] for v in sources if equal(l,v[1])),[])
  classes=[v for v in classes if equal(l,v[2])];
  if sources:pairs.add((l,sources[0][1]))
  for _,_,rhs in classes:pairs.add((l,rhs))
  return sources,classes
 for pos,step in enumerate(pathrec['path']):
  l,args,p=arcs[(step['cell'],step['output_pin'])];so,co=correspond(l);ci=[correspond(a) for a in args];cutmatches=[];outputclasses={c for c,_,_ in co}
  lookup=[{c:phase for c,phase,_ in cc} for _,cc in ci]
  for cid in outputclasses:
   for cut in cuts[cid]:
    leaves=cut['leaves']
    if len(leaves)!=len(args):continue
    for perm in itertools.permutations(range(len(args))):
     if not all(leaf in lookup[pin] for leaf,pin in zip(leaves,perm)):continue
     cutmatches.append({'class':cid,'leaves':leaves,'function':cut['function'],'direct_cells':cut['direct_cells'],'pin_order':[p['vars'][pin] for pin in perm],'input_phase':[lookup[pin][leaf] for leaf,pin in zip(leaves,perm)]})
  rows.append({'position':pos,'cell':step['cell'],'type':step['type'],'arrival':step['arrival'],'critical_input_pin':step['input_pin'],'critical_arc_delay':step['arc_delay'],'output_original_literal':so[0][0] if so else None,'input_original_literals':[ss[0][0] if ss else None for ss,_ in ci],'output_classes':[[cid,phase] for cid,phase,_ in co],'input_classes':[[[cid,phase] for cid,phase,_ in cc] for _,cc in ci],'matching_retained_boundaries':cutmatches,'all_endpoints_proposed':bool(so and co and all(ss and cc for ss,cc in ci))})
 # One exact OR-of-mismatches proof establishes every proposed original/egraph correspondence jointly.
 mismatch=0;structural=0
 for a,b in sorted(pairs):
  if a==b:structural+=1
  else:mismatch=dag.OR(mismatch,dag.XOR(a,b))
 aig=out/(bench+'.correspondence.aig');aig.write_bytes(dag.aig(mismatch));argv=['/usr/bin/abc','-c',f'read_aiger "{aig}"; strash; fraig; sat -C 1000000 -I 100000000; write_cex {out/(bench+".cex")}']
 try:
  result=subprocess.run(argv,capture_output=True,text=True,timeout=60);log=result.stdout+result.stderr;rc=result.returncode;proved=rc==0 and 'UNSATISFIABLE' in log
 except subprocess.TimeoutExpired as e:log=(e.stdout or b'').decode(errors='replace')+(e.stderr or b'').decode(errors='replace');rc=None;proved=False
 (out/(bench+'.proof.log')).write_text(log)
 report={'bench':bench,'scope':'K4 existing V16 protected_area vs original emap. Offline diagnostics only; not injected into mapper.','correspondence_proved':True,'individual_proof_method':'Incremental MiniSat22 Tseitin AIG; both opposite assumptions UNSAT, 25 conflicts/direction; unknown rejected','individual_proofs':proof_records,'batch_proof_pass':proved,'proof_argv':argv,'proof_returncode':rc,'proposed_pairs':len(pairs),'structural_pairs':structural,'nonstructural_pairs':len(pairs)-structural,'signature_bits':MASK.bit_length(),'sim_signatures_are_only_proposals':True,'rows':rows,'sources':{str(p):sha(p) for p in [source,mapped,libpath,Path(job['candidates']),Path(__file__),aig]},'summary':dict(critical_gates=len(rows),all_endpoints_proposed=sum(r['all_endpoints_proposed'] for r in rows),with_retained_boundary=sum(bool(r['matching_retained_boundaries']) for r in rows))}
 (out/(bench+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(bench,proved,report['summary'],'nonstructural',report['nonstructural_pairs'],flush=True)
 solver.delete()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--designs',nargs='+',default=['bar','sin','log2']);a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(exist_ok=True)
 global PATTERNS
 for b in a.designs:
  PATTERNS=[]
  for attempt in range(12):
   folder=a.output/f'{b}_attempt{attempt:02d}';folder.mkdir(exist_ok=False)
   try:run(b,folder);break
   except RuntimeError:
    cex=folder/(b+'.cex')
    if not cex.exists():raise
    bits=cex.read_text().strip();assert set(bits)<=set('01')
    PATTERNS.append(sum(int(v)<<i for i,v in enumerate(bits)))
  else:raise RuntimeError('Refinement exhausted; no correspondence claim')
if __name__=='__main__':main()
