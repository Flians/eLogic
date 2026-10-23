#!/usr/bin/env python3
"""Original-AIG liveness only, not native region mapping or budget feasibility."""
from pathlib import Path
import json,hashlib,sys
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_epfl_structural_budget import read_binary_aig
ROOT=Path(__file__).resolve().parents[3]
rows=[]
for name in ['div','hyp','log2','multiplier','mem_ctrl']:
 p=ROOT/f'results_v2/epfl_resyn2_subjects_20261012/{name}.resyn2.aig';data=p.read_bytes();inputs,roots,gates=read_binary_aig(data);live=set();todo=[r//2 for r in roots]
 while todo:
  v=todo.pop()
  if v in gates and v not in live:live.add(v);todo.extend(x//2 for x in gates[v])
 uses=Counter(x//2 for v in live for x in gates[v]);uses.update(r//2 for r in roots)
 frontier=set();peak=0;samples=[]
 for index,v in enumerate(sorted(live)):
  for x in gates[v]:
   child=x//2;uses[child]-=1
   if uses[child]==0:frontier.discard(child)
  if uses[v]:frontier.add(v)
  peak=max(peak,len(frontier))
  if index%8000==0:samples.append({'processed_original_ANDs':index+1,'live_internal_frontier':len(frontier)})
 rows.append({'design':name,'prepared_sha256':hashlib.sha256(data).hexdigest(),'original_live_ANDs':len(live),'all_original_PIs':inputs,'topological_live_internal_frontier_peak':peak,'frontier_plus_all_PIs_and_zero':peak+inputs+1,'samples':samples})
o=ROOT/'results_v2/flow_saturation_main_integration_20261014/retired_state_frontier_plan.json';o.write_text(json.dumps({'analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Raw-AIG live frontier only. Counts do not include rewrite alternatives, region-local duplicated interfaces, actual physical pool/cover, timing or mandatory recovery work. Not a mapped CEC or compliance proof.','next_requirements':['Materialize/charge every visited original AND before certified retirement; log cumulative work separately from global resident peak.','Count all simultaneously resident graphs/cached symbolic state and every boundary in one ledger; no per-region counter resets.','One global retained physical pool100k; cuts16 across revisits; saturation16 shared, not per region.','One global recovery2s including mandatory cover setup/stitch/final validation; no timer suspension or local recovery allowance.','Explicitly distinguish retired-state representation from prior all-original-functions-resident scope; no old imported-total<=20k claim or hiding opaque original formulas.','Full named-port CEC/common library metrics and all original failures before any main promotion.'],'results':rows},indent=2)+'\n');print([(r['design'],r['frontier_plus_all_PIs_and_zero']) for r in rows])
