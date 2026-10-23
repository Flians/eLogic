"""Reconstruct selected critical paths from emitted physical netlists; not candidate absence proof."""
from collections import Counter,defaultdict,deque
import json,struct
from pathlib import Path
f32=lambda x:struct.unpack('f',struct.pack('f',x))[0]
def analyze(folder,measured):
 paths=list(Path(folder).rglob('mapped.json'))
 if len(paths)!=1:return dict(available=False,reason='unique emitted physical JSON unavailable')
 cells=json.loads((Path(folder)/'library.json').read_text());models={}
 for c in cells:
  n=len(c['vars']);bits=c['tt'];truth=lambda a: bool(bits[a//64]&(1<<(a%64)))
  support=[p for p in range(n)if any(truth(a)!=truth(a|(1<<p))for a in range(1<<n)if not a&(1<<p))]
  models[c['name']]=(c,support)
 modules=json.loads(paths[0].read_text())['modules'];m=modules['mapped']if 'mapped'in modules else next(iter(modules.values()));arrival={'0':0.,'1':0.};prev={};drivers={};area=0.
 for name,p in m['ports'].items():
  if p['direction']=='input':
   for b in p['bits']:arrival[b]=0.
 for name,c in m['cells'].items():
  model,support=models[c['type']];area+=f32(model['area']);out=c['connections'][model['out_pin']][0];assert out not in drivers;drivers[out]=(name,c,model,support)
 users=defaultdict(list);wait={};ready=deque()
 for bit,(_,c,model,support)in drivers.items():
  deps={c['connections'][model['vars'][p]][0]for p in support}-arrival.keys();wait[bit]=len(deps)
  if not deps:ready.append(bit)
  for b in deps:users[b].append(bit)
 while ready:
  bit=ready.popleft();name,c,model,support=drivers[bit];options=[(arrival[c['connections'][model['vars'][p]][0]]+f32(model['pin_delay'].get(model['vars'][p],0.)),model['vars'][p],c['connections'][model['vars'][p]][0])for p in support];time,pin,src=max(options,default=(0.,'',None));arrival[bit]=time;prev[bit]=dict(cell=name,type=c['type'],input_pin=pin,input_bit=src,arrival=time)
  for b in users[bit]:
   wait[b]-=1
   if wait[b]==0:ready.append(b)
 assert all(b in arrival for b in drivers),'cycle or missing input'
 delay,port,bit=max((arrival[b],name,b)for name,p in m['ports'].items()if p['direction']=='output'for b in p['bits']);assert abs(area-measured['area'])<=1e-6*max(1,area);assert abs(delay-measured['delay'])<=1e-6*max(1,delay)
 chain=[]
 while bit in prev:
  v=prev[bit];chain.append(v);bit=v['input_bit']
 chain.reverse();return dict(available=True,public_metrics_reproduced=True,A=area,D=delay,output=port,path_cells=len(chain),path_cell_types=dict(Counter(x['type']for x in chain)),all_cell_types=dict(Counter(x['type']for x in m['cells'].values())),path=chain,source=str(paths[0]))
