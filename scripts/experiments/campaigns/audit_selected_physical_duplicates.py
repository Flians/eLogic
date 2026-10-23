"""Read-only exact named-pin structural duplicate audit; does not export optimized netlists."""
import argparse,json,hashlib,struct
from collections import defaultdict,deque
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(x):return struct.unpack("f",struct.pack("f",x))[0]

def inspect(net,models):
 library=defaultdict(list)
 for model in models:library[model["name"]].append(model)
 original,merged,alias={"0":0.,"1":0.},{"0":0.,"1":0.},{"0":"0","1":"1"}
 for port in net["ports"].values():
  if port["direction"]=="input":
   for bit in port["bits"]:original[bit]=merged[bit]=0.;alias[bit]=bit
 drivers,waiters={},defaultdict(list)
 for name,cell in net["cells"].items():
  assert len(library[cell["type"]])==1,"single-output only"
  model=library[cell["type"]][0];out=model["out_pin"]
  assert cell["port_directions"][out]=="output" and len(cell["connections"][out])==1
  bit=cell["connections"][out][0];assert bit not in drivers
  inputs=sorted(pin for pin,direction in cell["port_directions"].items() if direction=="input")
  assert set(inputs)==set(model["vars"]),"exact all-named-input identity required"
  assert all(len(cell["connections"][pin])==1 for pin in inputs)
  drivers[bit]=(name,cell,model,inputs)
 missing,ready={},deque()
 for bit,(_,cell,_,inputs) in drivers.items():
  deps={cell["connections"][pin][0] for pin in inputs}-original.keys();missing[bit]=len(deps)
  if not deps:ready.append(bit)
  for dep in deps:waiters[dep].append(bit)
 identities={};duplicates=[];old_area=new_area=0.;visited=0
 while ready:
  bit=ready.popleft();name,cell,model,inputs=drivers[bit]
  area=f32(model["area"]);old_area+=area
  original[bit]=max((original[cell["connections"][pin][0]]+f32(model["pin_delay"].get(pin,0.)) for pin in inputs),default=0.)
  key=(cell["type"],model["out_pin"],json.dumps(cell.get("parameters",{}),sort_keys=True),tuple((pin,alias[cell["connections"][pin][0]]) for pin in inputs))
  time=max((merged[alias[cell["connections"][pin][0]]]+f32(model["pin_delay"].get(pin,0.)) for pin in inputs),default=0.)
  if key in identities:
   canonical=identities[key];alias[bit]=canonical
   assert time==merged[canonical]
   duplicates.append(dict(cell=name,type=cell["type"],area=area,representative_output=canonical))
  else:
   identities[key]=bit;alias[bit]=bit;merged[bit]=time;new_area+=area
  assert merged[alias[bit]]==original[bit],"named-pin identity changed arrival"
  visited+=1
  for user in waiters[bit]:
   missing[user]-=1
   if missing[user]==0:ready.append(user)
 assert visited==len(drivers),"cycle or undriven dependency"
 outputs=[bit for port in net["ports"].values() if port["direction"]=="output" for bit in port["bits"]]
 old_d=max(original[b] for b in outputs);new_d=max(merged[alias[b]] for b in outputs)
 assert old_d==new_d
 return dict(original_instances=visited,coalesced_instances=len(identities),duplicate_instances=len(duplicates),original_area=old_area,coalesced_area_estimate=new_area,removable_duplicate_area=old_area-new_area,original_delay=old_d,coalesced_delay_estimate=new_d,duplicates=duplicates)

def analyze(report,method):
 d=json.loads(report.read_text());r=next(row for row in d["results"] if row["method"]==method);assert r["status"]=="ok" and r["measured"]["cec"]==1
 mapped=report.parent/(r["bench"]+"."+method)/"mapped.json";lib=report.parent/"library.json"
 result=inspect(json.loads(mapped.read_text())["modules"]["mapped"],json.loads(lib.read_text()))
 assert abs(result["original_area"]-r["measured"]["area"])<1e-6
 assert abs(result["original_delay"]-r["measured"]["delay"])<1e-6
 result.update(bench=r["bench"],method=method,sources={str(p):sha(p) for p in [report,mapped,lib]},input_sha256=r["input_sha256"],prepared_sha256=r["preprocessing"]["sha256"],library_sha256=d["library_sha256"])
 return result

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--report",action="append",type=Path,required=True);ap.add_argument("--method",action="append",required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();assert len(a.report)==len(a.method) and not a.output.exists()
 records=[analyze(p,m) for p,m in zip(a.report,a.method)]
 a.output.write_text(json.dumps(dict(diagnostic_only=True,mapper_calls=0,netlist_exports=0,source_sha256=sha(Path(__file__)),limitation="Exact same cell/parameters/named-pin wires on existing physical topology. Not implemented in recovery, no new CEC, no PPA improvement claimed; global recovery/retained budgets and proof-compatible export remain to validate.",records=records),indent=2)+"\n")
if __name__=="__main__":main()
