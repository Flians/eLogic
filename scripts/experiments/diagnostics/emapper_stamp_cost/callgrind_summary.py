"""Aggregate non-overlapping periodic Callgrind dumps, preserving scope limitations."""
import argparse,collections,json,pathlib,re
def parse(path):
 names={};own=collections.Counter();edges=collections.Counter();calls=collections.Counter();events=[];fn=None;callee=None;pending=None;summary=None
 def name(value):
  m=re.match(r'\((\d+)\)(?: (.*))?$',value)
  if not m:return value
  key=int(m[1])
  if m[2] is not None:names[key]=m[2]
  return key
 for line in path.read_text().splitlines():
  if line.startswith('events: '):events=line[8:].split()
  elif line.startswith('summary: '):summary=[int(x) for x in line[9:].split()]
  elif line.startswith('fn='):fn=name(line[3:])
  elif line.startswith('cfn='):callee=name(line[4:])
  elif line.startswith('calls='):pending=int(line[6:].split()[0])
  elif line and (line[0].isdigit()or line[0]in '+-*')and fn is not None:
   a=line.split()
   if len(a)<2:continue
   values=[int(v)for v in a[1:]]+[0]*len(events)
   for i,k in enumerate(events):
    if pending is None:own[(fn,k)]+=values[i]
    else:edges[(fn,callee,k)]+=values[i]
   if pending is not None:calls[(fn,callee)]+=pending;pending=None
 def resolve(k):return names.get(k,str(k))
 assert summary is not None
 assert sum(v for(f,k),v in own.items()if k=='Ir')==summary[events.index('Ir')],str(path)+' inconsistent self instruction total'
 so,se,sc=collections.Counter(),collections.Counter(),collections.Counter()
 for(f,k),v in own.items():so[(resolve(f),k)]+=v
 for(f,c,k),v in edges.items():se[(resolve(f),resolve(c),k)]+=v
 for(f,c),v in calls.items():sc[(resolve(f),resolve(c))]+=v
 return so,se,sc,dict(zip(events,summary))
def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();assert not a.output.exists()
 own=collections.Counter();edges=collections.Counter();calls=collections.Counter();total=collections.Counter();files=[]
 # Only completed periodic parts; the unsuffixed file may be interrupted.
 for path in sorted(a.directory.glob('callgrind.*.*')):
  if not re.fullmatch(r'callgrind\.\d+\.\d+',path.name):continue
  x,y,z,t=parse(path);own.update(x);edges.update(y);calls.update(z);total.update(t);files.append(str(path))
 assert files
 names=set(f for f,k in own);child_cost=collections.Counter()
 for(c,child,k),v in edges.items():
  if k=='Ir':child_cost[c]+=v
 incl={f:own[(f,'Ir')]+child_cost[f]for f in names}
 focus=[]
 for f in names:
  if ('selected_evidence::capture' in f and ('LocalKey' in f or f.endswith('::capture')))or f.endswith(('online::publish_kind','online::publication_manifest','egraph_map::stage_trace'))or('>::write_verilog::<'in f and'iter::'not in f):
   children=[dict(function=child,instructions=v,calls=calls[(f,child)],percent_of_parent=100*v/incl[f]if incl[f]else 0,syscall_elapsed_ns=edges[(f,child,'sysTime')],syscall_cpu_ns=edges[(f,child,'sysCpuTime')],syscalls=edges[(f,child,'sysCount')])for(c,child,k),v in list(edges.items())if c==f and k=='Ir']
   focus.append(dict(function=f,inclusive_instructions=incl[f],self_instructions=own[(f,'Ir')],syscalls_in_direct_edges=sum(v for(c,child,k),v in edges.items()if c==f and k=='sysCount'),children=sorted(children,key=lambda r:r['instructions'],reverse=True)[:20],syscall_children=sorted(children,key=lambda r:r['syscall_elapsed_ns'],reverse=True)[:10]))
 result=dict(files=files,totals=dict(total),focus=sorted(focus,key=lambda r:r['inclusive_instructions'],reverse=True),top_self=[dict(function=f,instructions=own[(f,'Ir')])for f in sorted(names,key=lambda f:own[(f,'Ir')],reverse=True)[:30]],scope='non-overlapping completed periodic simulated-instruction parts, not native CPU/wall percentages or final search; inclusive parent/child costs must not be added; no PPA claim')
 a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
