"""Lossless single-output mapped Verilog -> BLIF, with ordered aliased COs."""
import re
from pathlib import Path

def convert(netlist,library,target):
 text=Path(netlist).read_text();functions={b.split()[0]:b.split()[2] for b in re.split(r'(?m)^GATE\s+',Path(library).read_text())[1:]}
 inputs=[n.strip()for g in re.findall(r'\binput\s+(.*?);',text,re.S)for n in g.split(',')]
 outputs=[n.strip()for g in re.findall(r'\boutput\s+(.*?);',text,re.S)for n in g.split(',')]
 aliases=dict(re.findall(r'\bassign\s+(\S+)\s*=\s*(\S+)\s*;',text))
 def resolve(x):
  seen=set()
  while x in aliases:
   assert x not in seen;seen.add(x);x=aliases[x]
  assert '~'not in x
  return x
 cells=[];used=set()
 for name,inst,ports in re.findall(r'^\s*(\w+)\s+(\w+)\s*\((.*?)\)\s*;',text,re.M|re.S):
  if name not in functions:continue
  pp=[(k,resolve(v.strip()))for k,v in re.findall(r'\.(\w+)\s*\(\s*([^)]*?)\s*\)',ports)]
  cells.append((name,pp));used.update(v for k,v in pp)
 oo=[resolve(x)for x in outputs];used.update(oo)
 constants={"1'b0":'__wire_const0',"1'b1":'__wire_const1'}
 rename=lambda x:constants.get(x,x)
 lines=['.model top','.inputs '+' '.join(inputs),'.outputs '+' '.join(rename(x)for x in oo)]
 for literal,wire in constants.items():
  if literal in used:lines.append('.gate _const'+('1'if literal=="1'b1"else'0')+'_ z='+wire)
 for name,pp in cells:lines.append('.gate '+name+' '+' '.join(k+'='+rename(v)for k,v in pp))
 lines.append('.end');Path(target).write_text('\n'.join(lines)+'\n')
 return dict(inputs=len(inputs),outputs=len(outputs),physical_cells=len(cells),aliased_output_drivers=len(oo)-len(set(oo)))
