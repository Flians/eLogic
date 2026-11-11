"""Compare existing emap critical functions with the audited native logic model.

Unequal sampled vectors provide concrete counterexamples. Equal vectors are
only potential matches, never equivalence proofs. No mapping/SDC inference.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import shutil

R=Path(__file__).resolve().parents[3]
S=R/'results_v2/emapper_dac_convergence_20261111'
WIDTH=256
MASK=(1<<WIDTH)-1


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def word(v):return format(v,'064x') if v is not None else None
def truth_bits(truth,args):
    assert len(args)<=9,'unexpected library/logical expression width'
    result=0
    for pattern in range(1<<len(args)):
        if not(truth>>pattern)&1:continue
        term=MASK
        for j,value in enumerate(args):term &= value if pattern&(1<<j) else MASK^value
        result |= term
    return result


def logic_values(nodes,pi_words):
    values={};users=defaultdict(list);pending={};ready=deque()
    for i,n in enumerate(nodes):
        if 'pi'in n:values[i]=pi_words[n['pi']]
    for i,n in enumerate(nodes):
        if i in values or 'args'not in n:continue
        assert all(0<=a<len(nodes)for a in n['args'])
        missing=set(n['args'])-values.keys();pending[i]=len(missing)
        for a in missing:users[a].append(i)
        if not missing:ready.append(i)
    while ready:
        i=ready.popleft();n=nodes[i];values[i]=truth_bits(n['function'],[values[a]for a in n['args']])
        for user in users[i]:
            pending[user]-=1
            if pending[user]==0:ready.append(user)
    return [values.get(i)for i in range(len(nodes))]


def physical_values(path,library,named_words):
    tops=[m for m in read(path)['modules'].values()if m.get('cells')];assert len(tops)==1
    top=tops[0];values={'0':0,'1':MASK};pending={};users=defaultdict(list);ready=deque()
    for name,p in top['ports'].items():
        if p['direction']=='input':assert len(p['bits'])==1;values[p['bits'][0]]=named_words[name]
    for key,c in top['cells'].items():
        spec=library[c['type']];args=[c['connections'][pin][0]for pin in spec['vars']]
        missing=set(args)-values.keys();pending[key]=len(missing)
        for bit in missing:users[bit].append(key)
        if not missing:ready.append(key)
    while ready:
        key=ready.popleft();c=top['cells'][key];spec=library[c['type']]
        truth=sum(v<<(64*j)for j,v in enumerate(spec['tt']))
        args=[values[c['connections'][pin][0]]for pin in spec['vars']]
        bit=c['connections'][spec['out_pin']][0];assert bit not in values
        values[bit]=truth_bits(truth,args)
        for user in users[bit]:
            pending[user]-=1
            if pending[user]==0:ready.append(user)
    assert all(v==0 for v in pending.values()),'physical simulation has unresolved dependencies'
    outputs={name:values[p['bits'][0]]for name,p in top['ports'].items()if p['direction']=='output'and len(p['bits'])==1}
    return values,outputs,top['cells']


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--references',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    shutil.copy2(Path(__file__),a.output/'reader.py')
    audit=read(S/'full-epfl-final-audit/validation.json');assert audit['completed']==20 and not audit['invalid_artifacts']
    current=read(S/'full-epfl/campaign.json');refs=read(a.references)
    library={c['name']:c for c in read(R/'emapper/libraries/asap7/library.json')}
    rows=[]
    for ref in refs['rows']:
        name=ref['design'];row=dict(design=name)
        try:
            assert ref['status']=='verified'
            report=read(Path(current['jobs'][name]['case'])/'report.json');manifest=Path(report['incumbent']['manifest'])
            selected_file=manifest.parent/report['incumbent']['selected_evidence']['file']
            assert sha(selected_file)==report['incumbent']['selected_evidence']['sha256'];selected=read(selected_file)
            logic_path=Path(current['jobs'][name]['case'])/'selected-provenance/objects'/(selected['logic']+'.json')
            assert sha(logic_path)==selected['logic'];logic=read(logic_path);nodes=logic['nodes']
            named={name:(int.from_bytes(hashlib.sha256(('dac-function-boundary-v1:'+name).encode()).digest(),'little')&~3)|2 for _,_,name in selected['ports']['inputs']}
            pi={pi:named[name]for _,pi,name in selected['ports']['inputs']}
            values=logic_values(nodes,pi);complete=all(v is not None for v in values)
            def signal(s):
                value=values[s//2];return None if value is None else value^(MASK if s%2 else 0)
            assert all(signal(i)==signal(v)for i,v in enumerate(logic['signed_aliases'])if signal(i)is not None and signal(v)is not None),'alias model disagrees on concrete PI assignments'
            path=Path(ref['baseline_netjson']);assert sha(path)==ref['baseline_netjson_sha256']
            physical,outputs,cells=physical_values(path,library,named)
            native_outputs={name:signal(s)for s,name in selected['ports']['outputs']}
            assert native_outputs==outputs,'native model / baseline outputs disagree on concrete PI assignments'
            index=defaultdict(list)
            for i,value in enumerate(values):
                if value is not None:index[value].append(2*i);index[MASK^value].append(2*i+1)
            targets=[]
            for step in ref['baseline_graph']['critical_path']:
                c=cells[step['instance']];spec=library[c['type']];bits=physical[step['output_signal']]
                inputs={pin:physical[c['connections'][pin][0]]for pin in spec['vars']}
                targets.append(dict(instance=step['instance'],cell=c['type'],output_function_samples=word(bits),
                    potential_signed_output_classes=index.get(bits,[]),output_absent_in_complete_model=complete and bits not in index,
                    input_functions={pin:word(v)for pin,v in inputs.items()},
                    input_function_absent_pins=[pin for pin,v in inputs.items()if complete and v not in index]))
            payload=dict(schema=1,sample_count=WIDTH,pi_assignment_words={k:word(v)for k,v in named.items()},
                         native_class_function_words=[word(v)for v in values],targets=targets,
                         logic_sha256=selected['logic'],logic_path=str(logic_path),baseline_netjson_sha256=sha(path),
                         complete_model=complete,primary_outputs_agree_on_samples=True,
                         proof_scope='sample inequalities yield explicit counterexamples for each exported class/phase; matching signatures are potential only; not independent CEC of every eclass and not proof no other fast cover exists; no sampled dont-care inference')
            dst=a.output/(name+'.json');dst.write_text(json.dumps(payload,indent=2)+'\n')
            row.update(status='verified_model'if complete else'incomplete_model_no_global_absence_claim',logical_classes=len(nodes),
                       evaluated_classes=sum(v is not None for v in values),critical_cells=len(targets),
                       critical_output_functions_absent=sum(v['output_absent_in_complete_model']for v in targets),
                       critical_cells_with_absent_input_function=sum(bool(v['input_function_absent_pins'])for v in targets),
                       evidence=str(dst),evidence_sha256=sha(dst),candidate_pool_membership_not_checked=True)
        except Exception as error:row.update(status='unresolved_preserved',error=repr(error))
        rows.append(row)
    out=dict(rows=rows,baseline_runs=0,mapping_runs=0,CEC_runs=0,GitHub_push=False,
             scope='post-AIG logical-model function boundaries of existing emap critical paths; not candidate pool, pin-match or recovery availability proof')
    (a.output/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(rows))


if __name__=='__main__':main()
