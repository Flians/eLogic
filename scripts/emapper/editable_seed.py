"""Deadline-charged exact definitions of the internal nets of a physical seed.

Each internal net gets an exact AND/NOT definition. Rust replays local truth
identities, charges auxiliary logical nodes to the existing admission ledger,
and accepts only a fresh prepared-input CEC receipt for unconditional PO aliases.
"""
import collections,json,time
from pathlib import Path
from aig_io import read_binary_aig,symbols


def expression(node,values,mask):
 if 'Var'in node:return values[node['Var']]
 if 'Const'in node:return mask if node['Const']else 0
 if 'Not'in node:return expression(node['Not'],values,mask)^mask
 for op in ['And','Or']:
  if op in node:
   a,b=node[op];a=expression(a,values,mask);b=expression(b,values,mask)
   return a&b if op=='And'else a|b
 raise ValueError('unsupported seed library expression')


def build_editable_seed(data,library_json,seed_netjson,seed_metrics,deadline):
 start=time.monotonic()
 if start>=deadline:raise TimeoutError('editable seed reached global deadline')
 count,roots,gates=read_binary_aig(data);ins,outs=symbols(data,count,roots,gates)
 original_max=max(gates,default=count)
 live=set();todo=[x//2 for x in roots]
 while todo:
  v=todo.pop()
  if v in gates and v not in live:live.add(v);todo.extend(x//2 for x in gates[v])
 gates={v:gates[v]for v in sorted(live)}
 next_var=original_max+1
 index={tuple(sorted(ab)):2*v for v,ab in gates.items()}
 def literal_expr(node,pins):
  nonlocal next_var
  if 'Var'in node:return pins[node['Var']]
  if 'Const'in node:return int(node['Const'])
  if 'Not'in node:return literal_expr(node['Not'],pins)^1
  op='And'if'And'in node else'Or';aa,bb=node[op];a=literal_expr(aa,pins);b=literal_expr(bb,pins)
  if op=='Or':a^=1;b^=1
  if a>b:a,b=b,a
  if a==0:out=0
  elif a==1:out=b
  elif a==b:out=a
  elif a==(b^1):out=0
  elif(a,b)in index:out=index[a,b]
  else:
   v=next_var;next_var+=1;gates[v]=(a,b);out=2*v;index[a,b]=out
  return out^(op=='Or')
 library=json.loads(Path(library_json).read_text());cells={c['name']:c for c in library}
 if len(cells)!=len(library):raise ValueError('editable seed requires unique single-output library')
 top=json.loads(Path(seed_netjson).read_text())['modules']['mapped'];ports=top['ports'];net={}
 if set(ports)!=set(ins+outs):raise ValueError('seed named ports differ from source')
 for i,name in enumerate(ins):
  p=ports[name]
  if p['direction']!='input'or len(p['bits'])!=1:raise ValueError('seed input is not named scalar')
  bit=p['bits'][0]
  if bit in net and net[bit]!=2*(i+1):raise ValueError('seed aliases independent primary inputs')
  net[bit]=2*(i+1)
 for name,lit in zip(outs,roots):
  p=ports[name]
  if p['direction']!='output'or len(p['bits'])!=1:raise ValueError('seed output is not named scalar')
 net.update({'0':0,'1':1})
 pending={};waiting=collections.defaultdict(list);ready=collections.deque();drivers=set(net)
 for name,m in sorted(top['cells'].items()):
  c=cells.get(m['type'])
  if c is None:raise ValueError('seed uses a cell outside the fixed single-output library')
  connections=m['connections'];keys=set(c['vars']+[c['out_pin']])
  if set(connections)!=keys or any(len(x)!=1 for x in connections.values()):raise ValueError('seed physical pin interface differs')
  bit=connections[c['out_pin']][0]
  if bit in drivers:raise ValueError('seed physical net has duplicate/free driver')
  drivers.add(bit);bits=[connections[p][0]for p in c['vars']];missing=set(x for x in bits if x not in net)
  pending[name]=[c,bits,bit,len(missing)]
  if not missing:ready.append(name)
  for x in missing:waiting[x].append(name)
 result=[];visits=0
 while ready:
  if time.monotonic()>=deadline:raise TimeoutError('editable seed reached global deadline')
  name=ready.popleft();c,bits,bit,_=pending.pop(name);inputs=[net[x]for x in bits]
  literal=literal_expr(c['expr'],dict(zip(c['vars'],inputs)))
  result.append(dict(cell=c['name'],inputs=inputs,output=literal,seed_instance=name));net[bit]=literal
  for user in waiting.pop(bit,[]):
   pending[user][3]-=1
   if pending[user][3]==0:ready.append(user)
 if pending:raise ValueError('seed physical graph is cyclic or has missing input drivers')
 seed_roots=[net[ports[name]['bits'][0]]for name in outs]
 return dict(schema=4,input_count=count,input_names=ins,output_names=outs,roots=roots,seed_roots=seed_roots,original_max=original_max,gates=[[v,*ab]for v,ab in gates.items()],instances=result,definitions={m['cell']:cells[m['cell']]['expr']for m in result},area=seed_metrics['area'],delay=seed_metrics['delay'],projection_seconds=time.monotonic()-start,proof_method='full formal pin truth followed by exact AIG substitution; fresh named prepared-input CEC authorizes only PO aliases',source_nodes_visited=visits,auxiliary_definitions=sum(v>original_max for v in gates),logical_admissions='Rust cumulative ledger; never free')
