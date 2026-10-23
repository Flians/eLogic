#!/usr/bin/env python3
"""Read-only exact provenance risk for an entire old shared cover; no mapper advice."""
import argparse,collections,hashlib,json
from pathlib import Path

def key(c):return (c['class'],tuple(c['leaves']),c['function'])
def audit(old,new):
 aliases=old['canonical_signals'];assert aliases==new['canonical_signals'] and old['egraph_nodes']==new['egraph_nodes']
 indexed=collections.defaultdict(list)
 for cut in old['cuts']:
  if cut['trivial']:continue
  for phase in range(2):indexed[aliases[cut['class']*2+phase]].append((cut,phase))
 newkeys={key(c) for c in new['cuts']};critical={v['selected'][0] for v in old['critical_signals'] if v['selected'] is not None}
 records=[];protected=collections.defaultdict(set);unresolved=0
 for m in old['selected_instances']:
  for o in m['outputs']:
   for target in o['selected_targets']:
    matches=[]
    for c,phase in indexed[target]:
     leaves=c['leaves'];inputs=m['inputs']
     if len(leaves)!=len(inputs):continue
     positions=[next(((i,neg) for i,l in enumerate(leaves) for neg in range(2) if aliases[l*2+neg]==sig),None) for sig in inputs]
     if None in positions or len({i for i,n in positions})!=len(leaves):continue
     f=0
     for x in range(1<<len(leaves)):
      arg=sum(((((x>>i)&1)^neg)<<p) for p,(i,neg) in enumerate(positions));f|=((o['function']>>arg)&1)<<x
     expected=c['function']^(((1<<(1<<len(leaves)))-1) if phase else 0)
     if f==expected:matches.append(key(c))
    if not matches:unresolved+=1;continue
    for cls,leaves,f in matches:protected[cls].add((leaves,f))
    if not any(w in newkeys for w in matches):records.append(dict(instance=m['instance'],cell=m['cell'],target=target,on_old_critical_path=m['instance'] in critical,old_exact_witnesses=matches))
 return dict(old_area=old['area'],new_area=new['area'],old_delay=old['delay'],new_delay=new['delay'],old_selected_instances=len(old['selected_instances']),exact_bindings_with_all_old_witnesses_removed=records,old_critical_bindings_at_risk=sum(r['on_old_critical_path'] for r in records),bindings_without_exact_fullcare_witness=unresolved,max_union_of_old_selected_exact_witnesses_per_class=max(map(len,protected.values()),default=0))
def main():
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();assert not a.output.exists();c=json.loads(a.campaign.read_text());rows=[];sources={}
 for k in [3,4]:
  for b in ['sin','adder','sqrt','bar']:
   ds=[]
   for profile in ['legacy_control','functional_cuts']:
    j=next(j for j in c['jobs'] if (j['k'],j['bench'],j['profile'])==(k,b,profile));q=Path(j['candidates']);data=q.read_bytes();assert hashlib.sha256(data).hexdigest()==j['candidates_sha256'];sources[str(q)]=j['candidates_sha256'];ds.append(json.loads(data))
   rows.append(dict(k=k,bench=b,**audit(*ds)))
 a.output.write_text(json.dumps(dict(rows=rows,sources=sources,scope='Exact full-care old-selected binding provenance only, including full cover rather than just a path. Missing exact witnesses do NOT prove absence of a care-qualified SDC binding or a physical candidate in the omitted full pool. Union of all alternative witnesses is an upper bound, not a minimum set cover or selected mandatory cut count. Offline results are not supplied as free candidates/incumbents. No new mapping, CEC, runtime or QoR claim.'),indent=2)+'\n')
 print([(r['k'],r['bench'],len(r['exact_bindings_with_all_old_witnesses_removed']),r['old_critical_bindings_at_risk']) for r in rows])
if __name__=='__main__':main()
