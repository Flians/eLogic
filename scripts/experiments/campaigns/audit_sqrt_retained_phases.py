#!/usr/bin/env python3
"""Read-only exact single-output cut/phase replay on frozen sqrt diagnostics."""
import hashlib,itertools,json,struct
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def variants(cell):
    n=len(cell['vars']);table=defaultdict(list)
    for perm in itertools.permutations(range(n)):
      for neg in range(1<<n):
        tt=0
        for a in range(1<<n):
            original=sum((((a>>i)&1)^((neg>>i)&1))<<perm[i] for i in range(n))
            tt|=((cell['tt'][0]>>original)&1)<<a
        table[tt].append((cell['name'],tuple(cell['vars'][p] for p in perm),neg))
    return table

def analyze(k):
    base=ROOT/f'results_v2/flow_saturation_main_checkpoint_20261014/{k}in_sqrt_flow_sat';folder=base/'sqrt.egraph_staged_control_resyn2'
    files=[folder/'candidate-diagnostics.json',folder/'pool-diagnostics.json',base/'library.json',base/'report.json']
    d=json.loads(files[0].read_text());pool=json.loads(files[1].read_text());lib=json.loads(files[2].read_text())
    report=json.loads(files[3].read_text());assert report['results'][0]['measured']['cec']==1
    families=['AO31','AOI21','OAI21'];cells=[c for c in lib if any(c['name'].startswith(f+'x') for f in families)]
    tables={c['name']:variants(c) for c in cells};arities={c['name']:len(c['vars']) for c in cells}
    canonical=d['canonical_signals'];critical={r['signal']:r for r in d['critical_signals'] if r['selected'] is not None};selected={m['instance']:m for m in d['selected_instances']}
    retained=set()
    for signal,r in critical.items():
      for p in r['producers']:
        m=selected.get(p['instance'])
        # Candidate rows omit pin names; recover exact names from the library function/ordered delays.
        c=next(c for c in lib if c['name']==p['cell']);n=len(c['vars'])
        for pins in itertools.permutations(c['vars']):
            physical=0
            for a in range(1<<n):
                original=sum(((a>>i)&1)<<c['vars'].index(pin) for i,pin in enumerate(pins));physical|=((c['tt'][0]>>original)&1)<<a
            if physical==p['function'] and [struct.unpack('f',struct.pack('f',c['pin_delay'][pin]))[0] for pin in pins]==p['delays']:
                retained.add((p['cell'],tuple(p['inputs']),tuple(pins),signal))
    eligible=defaultdict(set);kept=defaultdict(set);self_dep=Counter();cut_counts=Counter()
    for cut in d['cuts']:
        n=len(cut['leaves']);mask=(1<<(1<<n))-1
        for ph in [0,1]:
            target=canonical[2*cut['class']+ph]
            if target not in critical:continue
            function=cut['function']^(mask if ph else 0)
            for cell,table in tables.items():
                if arities[cell]!=n:continue
                for name,pins,neg in table.get(function,[]):
                    ins=tuple(canonical[2*leaf+((neg>>i)&1)] for i,leaf in enumerate(cut['leaves']))
                    if target in ins:self_dep[name]+=1;continue
                    key=(name,ins,pins,target);eligible[name].add(key);cut_counts[name]+=1
                    if key in retained:kept[name].add(key)
    pool_by={r['signal']:r for r in pool['signals']};rows=[]
    for c in cells:
        name=c['name'];e=eligible[name];r=kept[name]
        targets={x[-1] for x in e};retained_targets={x[-1] for x in r}
        used={s for s,x in critical.items() if selected[x['selected'][0]]['cell']==name}
        # Before/after estimator uses the same area-witness arrivals, distinct from best legal cover.
        lost=[s for s in critical if s in pool_by and pool_by[s]['best_before'] and pool_by[s]['best_before']['cell']==name and not pool_by[s]['best_before']['kept']]
        rows.append({'cell':name,'exact_cut_phase_variants_replayed':len(e),'exact_variants_retained':len(r),'critical_targets_with_replayed_variant':len(targets),'critical_targets_with_retained_variant':len(retained_targets),'critical_targets_with_no_replayed_variant_retained':len(targets-retained_targets),'selected_on_critical_signals':len(used),'fastest_area_witness_estimated_producer_removed_by_cap':len(lost),'removed_estimator_examples':[pool_by[s] for s in lost[:8]],'not_a_legal_trial_or_delay_lower_bound':True})
    return {'k':k,'area':d['area'],'delay':d['delay'],'critical_bound_signals':len(critical),'pool_before':pool['instances_before'],'pool_after':pool['instances_after'],'witness_reserved':pool['witness_reserved_instances'],'cells':rows,'sources_sha256':{str(p):sha(p) for p in files},'scope':'Exact library truth tables, pin permutations, both output phases and input inversions replayed on saved retained cuts; canonical inputs/targets normalized. Candidate function/delay pin permutation recovered from saved producers. No new mapping, baseline change, CEC or legal whole-cover substitution.'}

def main():
    out=ROOT/'results_v2/sqrt_retained_phase_audit_v2_20261014';out.mkdir(exist_ok=False)
    records=[analyze(k) for k in [3,4]]
    (out/'report.json').write_text(json.dumps({'analyzer_sha256':sha(Path(__file__)),'records':records,'limitations':['No discarded-cut bank exists in this snapshot; missing retained-cut variants cannot prove original cut-generation absence.','Replayed eligibility is before pool selection; no independent replay of every transient generation/cone filter.','Area-witness fastest estimates do not prove global timing improvement; replacing a producer requires cycle/phase/full-cover validation.']},indent=2)+'\n')
    for r in records:
        print(r['k'],r['pool_before'],r['pool_after'])
        for c in r['cells']:print({k:v for k,v in c.items() if k!='removed_estimator_examples'})
if __name__=='__main__':main()
