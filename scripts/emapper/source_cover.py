"""Constructive source-AIG library cover, used only as a checked incumbent.
No baseline tool is called. There is no egraph search in this constructor.
"""
import json,math,time,struct
from pathlib import Path
import sys
from aig_io import read_binary_aig, symbols, identifier

def build_source_cover(data,library_json,deadline,phase_aware=False):
 start=time.monotonic()
 if start>deadline:raise TimeoutError("source-cover construction exceeds mapping deadline")
 inputs,roots,gates=read_binary_aig(data);ins,outs=symbols(data,inputs,roots,gates)
 live=set();todo=list(roots)
 while todo:
  v=todo.pop()//2
  if v in gates and v not in live:live.add(v);todo.extend(gates[v])
 cells=json.loads(Path(library_json).read_text())
 f32=lambda x:struct.unpack('f',struct.pack('f',x))[0]
 for c in cells:
  c['area']=f32(c['area']);c['pin_delay']={p:f32(v)for p,v in c['pin_delay'].items()}
 names=[c['name']for c in cells]
 def choose(arity,truth):
  xs=[c for c in cells if len(c['vars'])==arity and c['tt'][0]==truth and names.count(c['name'])==1]
  if not xs:raise ValueError('source cover requires unique single-output AND2 and INV cells')
  return min(xs,key=lambda c:(max(c['pin_delay'].values(),default=0),c['area'],c['name']))
 and2=choose(2,8);inv=choose(1,1);nand2=choose(2,7)if phase_aware else None;prefix='_egm_source_'
 while any(n.startswith(prefix)for n in ins+outs):prefix+='x'
 demand={}
 for x in [*roots,*[x for v in live for x in gates[v]]]:
  if x>1:demand.setdefault(x//2,set()).add(x&1)
 neg={v for v,phases in demand.items()if 1 in phases}
 arrival={0:0.,1:0.};areas=[];count=0;inverters=0;used_cells=set();lines=['module mapped('+', '.join(['input '+identifier(n)for n in ins]+['output '+identifier(n)for n in outs])+');']
 def net(lit):
  if lit<2:return "1'b1"if lit else"1'b0"
  v=lit//2
  if v<=inputs and not lit&1:return identifier(ins[v-1])
  return prefix+('m'if lit&1 else'p')+str(v)
 def emit(cell,args,out,label):
  nonlocal count,inverters
  if time.monotonic()>deadline:raise TimeoutError('source-cover construction exceeds mapping deadline')
  count+=1;inverters+=int(cell['name']==inv['name']);used_cells.add(cell['name']);areas.append(cell['area']);lines.append('wire '+out+';');pins=[f'.{identifier(p)}({net(x)})'for p,x in zip(cell['vars'],args)];pins.append(f'.{identifier(cell["out_pin"])}({out})');lines.append(f'{identifier(cell["name"])} {prefix}{label}('+', '.join(pins)+');')
  arrival[(int(out[len(prefix)+1:])*2)+(out[len(prefix)]=='m')]=max((arrival[x]+cell['pin_delay'][p]for p,x in zip(cell['vars'],args)),default=0.)
 for v in range(1,inputs+1):
  arrival[2*v]=0.
  if v in neg:emit(inv,[2*v],net(2*v+1),'i'+str(v))
 for v in sorted(live):
  x,y=gates[v]
  if not phase_aware:
   args=min([(x,y),(y,x)],key=lambda a:(max(arrival[t]+and2['pin_delay'][p]for p,t in zip(and2['vars'],a)),a))
   emit(and2,args,net(2*v),'a'+str(v))
   if v in neg:emit(inv,[2*v],net(2*v+1),'i'+str(v))
  else:
   choices=[]
   for phase,cell in [(0,and2),(1,nand2)]:
    for args in [(x,y),(y,x)]:
     t=max(arrival[a]+cell['pin_delay'][p]for p,a in zip(cell['vars'],args));opposite=(1-phase)in demand[v];score=max(t+(inv['pin_delay'][inv['vars'][0]]if z!=phase else 0.)for z in demand[v]);choices.append(((score,cell['area']+inv['area']*opposite,cell['name'],args,phase),cell,args,phase,opposite))
   _,cell,args,phase,opposite=min(choices,key=lambda q:q[0]);emit(cell,args,net(2*v+phase),'a'+str(v))
   if opposite:emit(inv,[2*v+phase],net(2*v+1-phase),'i'+str(v))
 for name,root in zip(outs,roots):lines.append('assign '+identifier(name)+' = '+net(root)+';')
 lines.append('endmodule');assert count<=2*len(live)+inputs
 if not phase_aware:assert count==len(live)+len(neg)
 stats=dict(area=math.fsum(areas),delay=max((arrival[x]for x in roots),default=0.),instances=count,live_source_ands=len(live),shared_source_inverters=inverters,source_build_secs=time.monotonic()-start,input_names=ins,output_names=outs,source_cells=sorted(used_cells),phase_aware=phase_aware,proof='Exact AND2/NAND2 per live original AIG gate, shared explicit INV for complementary demanded phase, named ports preserved; independent CEC still required',egraph_search=False)
 if time.monotonic()>deadline:raise TimeoutError('source-cover construction exceeds mapping deadline')
 return '\n'.join(lines)+'\n',stats
