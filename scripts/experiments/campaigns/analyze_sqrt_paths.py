#!/usr/bin/env python3
"""Read-only reconstruction of public unloaded pin-delay paths, not new PPA."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
f32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def analyze(folder, bench, method):
    report = folder/'report.json'
    r = next(r for r in json.loads(report.read_text())['results'] if r['method'] == method)
    assert r['status'] == 'ok' and r['measured']['cec'] == 1
    path = folder/f'{bench}.{method}'/'mapped.json'
    library = folder/'library.json'
    pins = {}
    for c in json.loads(library.read_text()): pins.setdefault(c['name'], []).append(c)
    m = json.loads(path.read_text())['modules']['mapped']
    arrival, previous, drivers = {'0': 0., '1': 0.}, {}, {}
    tree_area = {'0':0.,'1':0.};tree_units={'0':0,'1':0}
    area = 0.
    for name, p in m['ports'].items():
        if p['direction'] == 'input':
            for b in p['bits']: arrival[b] = 0.; previous[b] = {'input_port': name};tree_area[b]=0.;tree_units[b]=0
    for name, c in m['cells'].items():
        model = pins[c['type']]
        assert all(abs(p['area']-model[0]['area']) < 1e-6 for p in model)
        area += f32(model[0]['area'])
        for p in model:
            bits = c['connections'][p['out_pin']]; assert len(bits) == 1
            assert bits[0] not in drivers
            drivers[bits[0]] = (name, c, p)
    from collections import defaultdict, deque
    dependents = defaultdict(list); missing = {}; ready = deque()
    for bit, (_, c, p) in drivers.items():
        unknown = set(c['connections'][pin][0] for pin in p['vars']) - set(arrival)
        missing[bit] = len(unknown)
        if not unknown: ready.append(bit)
        for dep in unknown: dependents[dep].append(bit)
    processed = 0
    while ready:
        bit = ready.popleft(); name, c, p = drivers[bit]
        deps = [(pin,c['connections'][pin][0]) for pin in p['vars']]
        best = max(((arrival[b]+f32(p['pin_delay'].get(pin,0.)),pin,b) for pin,b in deps),default=(0.,None,None))
        arrival[bit]=best[0]
        unique=set(b for _,b in deps)
        tree_area[bit]=f32(p['area'])+sum(tree_area[b] for b in unique)
        tree_units[bit]=1+sum(tree_units[b] for b in unique)
        previous[bit]={'cell':name,'type':c['type'],'output_pin':p['out_pin'],'input_pin':best[1],'input_bit':best[2],'arc_delay':f32(p['pin_delay'].get(best[1],0.)),'arrival':best[0]}
        processed += 1
        for consumer in dependents[bit]:
            missing[consumer]-=1
            if missing[consumer]==0:ready.append(consumer)
    if processed != len(drivers):raise ValueError('cycle or undriven dependency')
    outputs = [(arrival[b], name, b) for name, p in m['ports'].items()
               if p['direction'] == 'output' for b in p['bits']]
    delay, port, bit = max(outputs)
    assert abs(area-r['measured']['area']) < 1e-6, (area, r['measured']['area'])
    assert abs(delay-r['measured']['delay']) < 1e-6, (delay, r['measured']['delay'])
    chain = []
    while 'cell' in previous.get(bit, {}):
        a = previous[bit]; chain.append(a); bit = a['input_bit']
    chain.reverse()
    return {'method': method, 'bench': bench, 'area': area, 'delay': delay,
            'endpoint': port, 'critical_path_cell_count': len(chain),
            'output_tree_unit_bits':max(tree_units[b].bit_length() for _,_,b in outputs),
            'output_tree_area_infinite_count':sum(not __import__('math').isfinite(tree_area[b]) for _,_,b in outputs),
            'tree_proxy_caution':'Reference-one additive tree estimate on selected physical cover, not actual area and not direct capture of native flow labels.',
            'critical_path_types': dict(Counter(a['type'] for a in chain)),
            'all_cell_types': dict(Counter(c['type'] for c in m['cells'].values())),
            'path': chain, 'sources': {str(p): sha(p) for p in [report, library, path]},
            'input_sha256': r['input_sha256'], 'prepared_sha256': r['preprocessing']['sha256'],
            'library_sha256': json.loads(report.read_text())['library_sha256']}


def main():
    global OUT
    OUT=ROOT/'results_v2/flow_cost_implementation_20261014'
    campaign=json.loads((ROOT/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/campaign.json').read_text())
    records=[]
    for k in [3,4]:
        job=next(j for j in campaign['jobs'] if j['name']==f'{k}in_core_sqrt')
        baseline=Path(job['report']).parent
        group=[analyze(baseline,'sqrt','abc_map_delay_resyn2'),analyze(baseline,'sqrt','mockturtle_emap_delay_resyn2')]
        for profile in ['control','flow_cost']:
            folder=ROOT/f'results_v2/flow_cost_checkpoint_20261014/{k}in_sqrt_{profile}'
            r=analyze(folder,'sqrt','egraph_staged_control_resyn2');r['profile']=profile;group.append(r)
        for key in ['input_sha256','prepared_sha256','library_sha256']:assert len({r[key] for r in group})==1
        for r in group:r['k']=k;records.append(r)
    (OUT/'sqrt_critical_paths.json').write_text(json.dumps({'scope':'sqrt two inventories; independently CEC-passed existing observations','public_metric_reproduced':True,'baseline_algorithms_changed':False,'limitation':'Path descriptions do not prove candidate absence or causal benefit; no new PPA measurements or run selection','records':records},indent=2)+'\n')
    for r in records:print(r['k'],r['method'],r.get('profile'),r['delay'],r['critical_path_cell_count'],r['critical_path_types'])
if __name__=='__main__':main()
