"""Read-only functional pin-family coalescence estimate; no exported netlist or PPA claim.

Same cell and complete input multiset, equal full Boolean function after wire
normalization; differing named-pin permutations are kept as separate members.
The quotient selects the fastest existing member under propagated arrivals.
This is not a claim that delay vectors are identical or universally dominated.
"""
import argparse,json
from collections import defaultdict,deque
from pathlib import Path
import audit_selected_physical_duplicates as exact


def inspect(net,models):
 library=defaultdict(list)
 for model in models:library[model["name"]].append(model)
 original={"0":0.,"1":0.};alias={"0":0,"1":1};timing={0:0.,1:0.};next_input=2
 for port in net["ports"].values():
  if port["direction"]=="input":
   for bit in port["bits"]:
    assert bit not in alias;alias[bit]=next_input;timing[next_input]=0.;next_input+=1;original[bit]=0.
 drivers,waiters={},defaultdict(list)
 for name,cell in net["cells"].items():
  assert len(library[cell["type"]])==1,"single-output scope"
  assert not cell.get("parameters",{}),"parameter-dependent truth is unsupported"
  model=library[cell["type"]][0];assert len(model["vars"])<=6
  inputs=model["vars"];assert set(inputs)=={p for p,d in cell["port_directions"].items() if d=="input"}
  out=model["out_pin"];assert cell["port_directions"][out]=="output"
  assert all(len(cell["connections"][p])==1 for p in [*inputs,out])
  bit=cell["connections"][out][0];assert bit not in drivers;drivers[bit]=(name,cell,model)
 missing,ready={},deque()
 for bit,(_,cell,model) in drivers.items():
  deps={cell["connections"][pin][0] for pin in model["vars"]}-original.keys();missing[bit]=len(deps)
  if not deps:ready.append(bit)
  for dep in deps:waiters[dep].append(bit)
 groups,ids=[],{};old_area=0.;visited=0
 while ready:
  bit=ready.popleft();name,cell,model=drivers[bit];pins=model["vars"]
  original[bit]=max((original[cell["connections"][pin][0]]+exact.f32(model["pin_delay"].get(pin,0.)) for pin in pins),default=0.)
  wires=[alias[cell["connections"][pin][0]] for pin in pins]
  variables=sorted(set(wires)-{0,1});positions={wire:i for i,wire in enumerate(variables)}
  truth=0
  for bits in range(1<<len(variables)):
   physical_bits=sum((wire if wire in [0,1] else (bits>>positions[wire])&1)<<i for i,wire in enumerate(wires))
   truth|=((model["tt"][0]>>physical_bits)&1)<<bits
  key=(cell["type"],model["out_pin"],tuple(sorted(wires)),truth)
  if key not in ids:
   ids[key]=len(groups);groups.append(dict(key=key,area=exact.f32(model["area"]),members=[]))
  index=ids[key];canonical=next_input+index;assert all(w<canonical for w in wires),"quotient must remain acyclic"
  groups[index]["members"].append(dict(cell=name,output=bit,inputs=wires,delays=[exact.f32(model["pin_delay"].get(pin,0.)) for pin in pins]))
  alias[bit]=canonical;old_area+=exact.f32(model["area"]);visited+=1
  for user in waiters[bit]:
   missing[user]-=1
   if not missing[user]:ready.append(user)
 assert visited==len(drivers),"cycle or undriven dependency"
 for index,group in enumerate(groups):
  score=lambda member:max((timing[wire]+delay for wire,delay in zip(member["inputs"],member["delays"])),default=0.)
  member=min(group["members"],key=lambda m:(score(m),m["cell"]))
  timing[next_input+index]=score(member);group["selected_member"]=member["cell"]
  assert all(timing[next_input+index]<=original[m["output"]]+1e-9 for m in group["members"])
 outputs=[bit for port in net["ports"].values() if port["direction"]=="output" for bit in port["bits"]]
 before=max(original[bit] for bit in outputs);after=max(timing[alias[bit]] for bit in outputs);assert after<=before+1e-9
 new_area=sum(group["area"] for group in groups)
 return dict(original_instances=visited,coalesced_instances=len(groups),duplicate_instances=visited-len(groups),original_area=old_area,coalesced_area_estimate=new_area,removable_duplicate_area=old_area-new_area,original_delay=before,coalesced_delay_estimate=after,duplicate_families=[dict(cell_type=g["key"][0],members=[m["cell"] for m in g["members"]],selected_member=g["selected_member"],area=g["area"]) for g in groups if len(g["members"])>1])


def analyze(path,method):
 old=exact.inspect
 try:
  exact.inspect=inspect
  return exact.analyze(path,method)
 finally:exact.inspect=old


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--report",type=Path,action="append",required=True);ap.add_argument("--method",action="append",required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();assert len(a.report)==len(a.method) and not a.output.exists()
 records=[analyze(p,m) for p,m in zip(a.report,a.method)]
 a.output.write_text(json.dumps(dict(diagnostic_only=True,mapper_calls=0,netlist_exports=0,source_sha256={str(Path(__file__)):exact.sha(Path(__file__)),str(Path(exact.__file__)):exact.sha(Path(exact.__file__))},limitation="Full Boolean-function and same physical input-multiset equality only, same cell. Timing chooses among already selected members, not all library pin assignments. No recovery integration, output CEC or new PPA result; any implementation must respect global recovery/source proof/binding budgets.",records=records),indent=2)+"\n")

if __name__=="__main__":main()
