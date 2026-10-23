#!/usr/bin/env python3
"""Read-only actual selected-cover diagnostics; no added mapping candidates/advice."""
import argparse,json,hashlib,collections
from pathlib import Path
from analyze_sqrt_paths import analyze
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('campaign',type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();assert not a.output.exists();c=json.loads(a.campaign.read_text());rows=[]
 for k in [3,4]:
  for b in ['sqrt','bar','sin']:
   jobs=[next(j for j in c['jobs'] if j['k']==k and j['bench']==b and j['profile']==p) for p in ['legacy_control','functional_cuts']]
   ds=[json.loads(Path(j['candidates']).read_text()) for j in jobs]
   old={(x['class'],tuple(x['leaves']),x['function']) for x in ds[0]['cuts']}
   new={(x['class'],tuple(x['leaves']),x['function']) for x in ds[1]['cuts']}
   d=ds[1];aliases=d['canonical_signals'];by_target=collections.defaultdict(list)
   for cut in d['cuts']:
    if cut['trivial']:continue
    for phase in range(2):by_target[aliases[cut['class']*2+phase]].append((cut,phase))
   witnesses=[]
   for m in d['selected_instances']:
    for o in m['outputs']:
     for target in o['selected_targets']:
      matches=[]
      for cut,phase in by_target[target]:
       leaves=cut['leaves'];inputs=m['inputs']
       if len(leaves)!=len(inputs):continue
       positions=[]
       for sig in inputs:
        pair=next(((i,neg) for i,l in enumerate(leaves) for neg in range(2) if aliases[l*2+neg]==sig),None)
        if pair is None:break
        positions.append(pair)
       if len(positions)!=len(inputs) or len({i for i,n in positions})!=len(leaves):continue
       function=0
       for x in range(1<<len(leaves)):
        arg=sum(((((x>>i)&1)^neg)<<p) for p,(i,neg) in enumerate(positions))
        function|=((o['function']>>arg)&1)<<x
       expected=cut['function']^(((1<<(1<<len(leaves)))-1) if phase else 0)
       if function==expected:matches.append((cut['class'],tuple(leaves),cut['function']))
      if matches and not any(x in old for x in matches):witnesses.append(dict(instance=m['instance'],cell=m['cell'],target=target,exact_matches=matches))
   paths=[]
   for j in jobs:
    r=json.loads(Path(j['report']).read_text())['results'][0];paths.append(analyze(Path(j['report']).parent,b,r['method']))
   rows.append(dict(k=k,bench=b,new_final_cuts=len(new-old),old_final_cuts_displaced=len(old-new),selected_targets_matching_only_newly_retained_cuts=witnesses,paths=paths,sources={j['candidates']:sha(j['candidates']) for j in jobs}))
 a.output.write_text(json.dumps(dict(rows=rows,scope='Read-only selected exact full-care match diagnostics. Source graph/aliases equality independently checked. New final cut may be derived syntactically from new boundaries; counts do not imply all are direct quotient proposals or prove timing causality. Does not audit omitted unselected physical pool. SDC-only matches excluded from exact-match attribution.'),indent=2)+'\n')
if __name__=='__main__':main()
