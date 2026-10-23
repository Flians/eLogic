#!/usr/bin/env python3
"""Read-only exact existing source/phase-boundary restoration audit for K4 bar."""
import itertools,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];O=R/'results_v2/compound_boundary_20261016'
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cp=R/'results_v2/pin_cut_full_20261016/campaign.json';campaign=load(cp);jobs={j['profile']:j for j in campaign['jobs'] if j['k']==4 and j['bench']=='bar'}
a,b=[load(jobs[p]['candidates']) for p in ['legacy_control','protected_area']]
assert all(a[k]==b[k] for k in ['egraph_nodes','canonical_signals','inputs','roots'])
reference=R/'results_v2/compound_boundary_refined_20261016/bar_attempt00/bar.json';r=load(reference);assert r['correspondence_proved'];old=load(R/'results_v2/phase_delay_diagnosis_20261016/paths.json');ref=next(v for v in old['records'] if v['bench']=='bar')['emap'];lib=load(next(p for p in ref['sources'] if p.endswith('library.json')));rows=[]
for row in r['rows']:
 p=next(p for p in lib if p['name']==row['type']);tt=sum(x<<(64*i) for i,x in enumerate(p['tt']));matches=[]
 for d in [a,b]:
  banks=[cut for cut in d['cuts'] if not cut['trivial'] and cut['class'] in dict(row['output_classes'])];found=[]
  lookup=[{cid:phase for cid,phase in group} for group in row['input_classes']]
  for cut in banks:
   if len(cut['leaves'])!=len(p['vars']):continue
   for perm in itertools.permutations(range(len(p['vars']))):
    if not all(l in lookup[pin] for l,pin in zip(cut['leaves'],perm)):continue
    truth=0;output_phase=dict(row['output_classes'])[cut['class']]
    for x in range(1<<len(cut['leaves'])):
     pins=[0]*len(p['vars'])
     for position,(leaf,pin) in enumerate(zip(cut['leaves'],perm)):pins[pin]=((x>>position)&1)^lookup[pin][leaf]
     idx=sum(v<<i for i,v in enumerate(pins));truth|=(((tt>>idx)&1)^output_phase)<<x
    if truth==cut['function']:found.append(dict(cut=cut,physical_pin_order=[p['vars'][i] for i in perm]))
  matches.append(found)
 rows.append(dict(position=row['position'],cell=row['type'],original_output_literal=row['output_original_literal'],original_input_literals=row['input_original_literals'],control_exact_variants=len(matches[0]),candidate_exact_variants=len(matches[1]),restored=not matches[0] and bool(matches[1]),candidate_matches=matches[1]))
frontier={p:load(j['pool_frontier']) for p,j in jobs.items()}
for f in frontier.values():assert f['before']['pool_delay_certificate_pass'] and f['after']['pool_delay_certificate_pass'];assert abs(f['before']['attaining_cover_delay']-f['after']['attaining_cover_delay'])<1e-6
snapshot=O/'bar_campaign_snapshot.json';snapshot.write_text(cp.read_text())
report=dict(same_logical_graph_and_interfaces=True,original_correspondence_independently_proved=True,rows=rows,restored=[r['cell'] for r in rows if r['restored']],frontier=frontier,limitation='Restoration of these exact boundaries and lower whole-pool attainable delay establish a cut-coverage effect of this fixed factor; do not attribute all gain to one gate or assume unchanged final critical endpoint.',sources={str(p):sha(p) for p in [reference,snapshot,Path(__file__),*[Path(j['candidates']) for j in jobs.values()]]})
(O/'bar_restoration.json').write_text(json.dumps(report,indent=2)+'\n');print(report['restored']);print([(k,f['before']['attaining_cover_delay']) for k,f in frontier.items()])
