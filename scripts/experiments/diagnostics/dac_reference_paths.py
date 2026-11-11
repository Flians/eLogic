"""Read existing same-library physical graphs; no synthesis or baseline reruns.

Reproduce all-instance area and unloaded pin delay before describing a selected
critical path. A path difference alone is never a candidate-absence proof.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import shutil
import struct

R=Path(__file__).resolve().parents[3]
S=R/'results_v2/emapper_dac_convergence_20261111'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def f32(v):return struct.unpack('<f',struct.pack('<f',v))[0]


def physical_path(path, library):
    modules=read(path)['modules'];tops=[m for m in modules.values()if m.get('cells')]
    assert len(tops)==1,'ambiguous physical top'
    top=tops[0];cells=top['cells'];arrival={'0':0.,'1':0.};drivers={};users=defaultdict(list)
    for p in top['ports'].values():
        if p['direction']=='input':
            for bit in p['bits']:arrival[bit]=0.
    pending={};ready=deque();counts=Counter();area=0.;parents={}
    for key,c in cells.items():
        spec=library[c['type']];counts[c['type']]+=1;area+=f32(spec['area'])
        assert set(spec['vars'])<=set(c['connections']) and spec['out_pin'] in c['connections']
        inp=[c['connections'][pin][0]for pin in spec['vars']]
        assert all(len(c['connections'][pin])==1 for pin in [*spec['vars'],spec['out_pin']])
        unknown=set(inp)-arrival.keys();pending[key]=len(unknown)
        for bit in unknown:users[bit].append(key)
        if not unknown:ready.append(key)
    visited=0
    while ready:
        key=ready.popleft();visited+=1;c=cells[key];spec=library[c['type']]
        arcs=[(arrival[c['connections'][pin][0]]+f32(spec['pin_delay'][pin]),pin,c['connections'][pin][0])for pin in spec['vars']]
        if arcs:t,pin,previous=max(arcs,key=lambda a:a[0])
        else:t,pin,previous=0.,None,None
        bit=c['connections'][spec['out_pin']][0];assert bit not in drivers and bit not in arrival,'multiple physical drivers'
        arrival[bit]=t;drivers[bit]=(key,c['type'],pin,previous,t)
        for user in users[bit]:
            pending[user]-=1
            if pending[user]==0:ready.append(user)
    assert visited==len(cells),'cycle or unconnected input in physical graph'
    outputs=[(arrival[bit],name,bit)for name,p in top['ports'].items()if p['direction']=='output'for bit in p['bits']]
    delay,name,bit=max(outputs,key=lambda x:x[0]);path=[];seen=set()
    while bit in drivers:
        assert bit not in seen;seen.add(bit);key,cell,pin,previous,t=drivers[bit]
        path.append(dict(instance=key,cell=cell,input_pin=pin,output_signal=bit,input_signal=previous,arrival=t,
                         library_inputs=len(library[cell]['vars'])))
        if previous is None:break
        bit=previous
    path.reverse()
    return dict(area=area,delay=delay,instances=len(cells),critical_output=name,critical_path=path,
                path_cells=len(path),path_max_library_inputs=max((x['library_inputs']for x in path),default=0),
                all_cell_counts=dict(counts),wide_cell_counts={k:v for k,v in counts.items()if len(library[k]['vars'])>6})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    shutil.copy2(Path(__file__),a.output/'reader.py')
    base=read(R/'results_v2/full_baselines_20261025/baselines.json');current=read(S/'full-epfl/campaign.json')
    audit=read(S/'full-epfl-final-audit/validation.json');assert not audit['invalid_artifacts']
    library={c['name']:c for c in read(R/'emapper/libraries/asap7/library.json')};rows=[];errors=[]
    for b in base['results']:
        if b['method']!='mockturtle_emap_delay' or b['status']!='ok':continue
        name=b['bench'];row=dict(design=name,baseline='mockturtle_emap_delay',candidate_absence_proven=False)
        try:
            adopted=read(Path(b['source_report']));obs=adopted['source_observation'];source=Path(obs['source_report'])
            assert sha(source)==obs['source_report_sha256'];raw=read(source)
            assert raw['library_sha256']==base['library_sha256']
            command=next(x['argv']for x in raw['commands']if x.get('bench')==name and x.get('method')=='mockturtle_emap_delay')
            baseline_net=Path(command[3]).parent/'mapped.json'
            native_net=Path(current['jobs'][name]['case'])/'mapped.json'
            before=physical_path(baseline_net,library);after=physical_path(native_net,library)
            assert [before['area'],before['delay']]==[b['measured']['area'],b['measured']['delay']]
            assert [after['area'],after['delay']]==[current['jobs'][name]['A'],current['jobs'][name]['D']]
            row.update(status='verified',baseline_graph=before,native_graph=after,
                       baseline_netjson=str(baseline_net),baseline_netjson_sha256=sha(baseline_net),
                       native_netjson=str(native_net),native_netjson_sha256=sha(native_net),source_report_sha256=sha(source))
        except Exception as e:row.update(status='unresolved_preserved',error=repr(e));errors.append(name)
        rows.append(row)
    report=dict(scope='existing physical graph descriptions only; exact A/D required, not candidate-coverage or equivalence proof',
                rows=rows,unresolved=errors,baseline_runs=0,mapping_runs=0,GitHub_push=False)
    (a.output/'references.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(cases=len(rows),unresolved=errors)))


if __name__=='__main__':main()
