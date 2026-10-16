#!/usr/bin/env python3
"""Independently sum mapped genlib area and propagate physical pin delays."""
from pathlib import Path
import re,json,sys

def audit(netlist,genlib):
 gates={}
 for block in re.split(r'(?m)^GATE\s+',Path(genlib).read_text())[1:]:
  first=block.splitlines()[0].split();name=first[0];area=float(first[1])
  pins={p: max(float(rise),float(fall)) for p,rise,fall in re.findall(r'PIN\s+(\S+)\s+\S+\s+\S+\s+\S+\s+(\S+)\s+\S+\s+(\S+)',block)}
  gates[name]=(area,pins,first[2])
 text=Path(netlist).read_text()
 inputs=re.search(r'\binput\s+(.*?);',text,re.S).group(1)
 outputs=re.search(r'\boutput\s+(.*?);',text,re.S).group(1)
 arrival={x.strip():0. for x in inputs.split(',')};arrival.update({"1'b0":0.,"1'b1":0.})
 po=[x.strip() for x in outputs.split(',')]
 area=0.;cells=0;types={};aliases_area=0.;aliases=0;native=dict(arrival)
 for name,pintext in re.findall(r'^\s+(\S+)\s+g\d+\( (.*?) \);',text,re.M):
  ports=dict(re.findall(r'\.(\S+) \((.*?)\)',pintext))
  if name in ('_const0_','_const1_'):
   assert set(ports)=={'z'},(name,ports)
   arrival[ports['z']]=0.;native[ports['z']]=0.;continue
  out=ports.pop('Y');cost,pd,function=gates[name]
  assert set(ports)==set(pd),(name,ports,pd)
  assert all(signal in arrival for signal in ports.values()),(name,ports)
  arrival[out]=max(arrival[signal]+pd[pin] for pin,signal in ports.items())
  # write_verilog inserts BUF cells between duplicate PO names. Native
  # cell_view metrics exclude these serialization aliases; retain both metrics.
  alias=len(ports)==1 and function=='Y=A;' and next(iter(ports.values())) in po and out in po
  native[out]=max(native[signal]+(0. if alias else pd[pin]) for pin,signal in ports.items())
  if alias:aliases_area+=cost;aliases+=1
  area+=cost;cells+=1;types[name]=types.get(name,0)+1
 po=[x.strip() for x in outputs.split(',')]
 assert cells and all(x in arrival for x in po)
 return dict(area=area-aliases_area,delay=max(native[x] for x in po),
             serialized_area=area,serialized_delay=max(arrival[x] for x in po),
             po_alias_buffers=aliases,cells=cells,outputs=len(po),cell_types=types)
if __name__=='__main__':print(json.dumps(audit(sys.argv[1],sys.argv[2]),indent=2))
