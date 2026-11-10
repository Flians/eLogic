#!/usr/bin/env python3
"""Compare mapped core DAGs modulo identity buffers and wire/output naming."""
import argparse, hashlib, json, re, subprocess
from pathlib import Path
from run_epmapping_supplement import sha,save

def core_fingerprint(netlist,library):
    def digest(value):return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()
    cells={}
    for block in re.split(r'(?m)^GATE\s+',library.read_text())[1:]:
        fields=block.splitlines()[0].split();cells[fields[0]]=fields[2]
    text=netlist.read_text()
    inputs=[n.strip() for g in re.findall(r'\binput\s+(.*?);',text,re.S) for n in g.split(',')]
    outputs=[n.strip() for g in re.findall(r'\boutput\s+(.*?);',text,re.S) for n in g.split(',')]
    values={n:digest(['PI',i]) for i,n in enumerate(inputs)}
    values.update({"1'b0":digest(['constant',0]),"1'b1":digest(['constant',1])})
    pending=[]
    for name,instance,ports in re.findall(r'^\s*(\w+)\s+(\w+)\s*\((.*?)\)\s*;',text,re.M|re.S):
        if name not in cells:continue
        ports=dict((p,v.strip()) for p,v in re.findall(r'\.(\w+)\s*\(\s*([^)]*?)\s*\)',ports))
        function=cells[name];output=ports.pop(function.split('=')[0])
        if function.endswith('=CONST0;'):values[output]=values["1'b0"];continue
        if function.endswith('=CONST1;'):values[output]=values["1'b1"];continue
        pending.append((name,output,ports,function=='Y=A;'))
    for output,source in re.findall(r'\bassign\s+(\S+)\s*=\s*(\S+)\s*;',text):
        assert '~' not in source,('unmapped inversion',source)
        pending.append(('wire',output,{'A':source},True))
    core=[]
    while pending:
        rest=[]
        for name,output,ports,alias in pending:
            if not all(v in values for v in ports.values()):rest.append((name,output,ports,alias));continue
            ordered=sorted((p,values[v]) for p,v in ports.items())
            value=ordered[0][1] if alias else digest(['cell',name,ordered])
            if not alias:core.append(value)
            values[output]=value
        assert len(rest)<len(pending),('unresolved core',rest[:2]);pending=rest
    # Multiset preserves multiplicity: sharing versus duplicated physical cells differs.
    return dict(core_dag_sha256=digest([sorted(core),[values[o] for o in outputs]]),core_cells=len(core),outputs=len(outputs))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    root=a.root.resolve();analysis=json.loads((root/'analysis.json').read_text())
    reference=Path(__file__).resolve().parents[2]/'results_epmapping_supplement_v1_20261110/analysis.json'
    old=json.loads(reference.read_text())['records'];proofs=[]
    for r in analysis['records']:
        previous=next(x for x in old if x['part']=='common' and x['arm']=='ours6' and x['design']==r['design'] and x['mode']==r['mode'])
        before=Path(previous['source_campaign'])/previous['selected_netlist']
        after=root/f"common/ours/{r['mode']}.{r['design']}/epmapping_mapped.v"
        old_graph=core_fingerprint(before,root/'inputs/asap7_6in.genlib')
        new_graph=core_fingerprint(after,root/'inputs/asap7_6in.genlib')
        assert old_graph==new_graph,('core DAG changed',r['design'],r['mode'],old_graph,new_graph)
        checkpoint=json.loads((after.parent/'epmapping_checkpoint.json').read_text())
        assert abs(checkpoint['area']-r['ours']['area'])<.0001 and abs(checkpoint['delay']-r['ours']['delay'])<.001
        assert sha(after.parent/checkpoint['netlist'])==sha(after),('final checkpoint bytes differ',r['design'],r['mode'])
        command=[str(root/'inputs/abc'),'-c',f'read_genlib {root/"inputs/asap7_6in.genlib"}; read -m {after}; cec -n {root/"inputs/benchmarks"/(r["design"]+".aig")}']
        result=subprocess.run(command,capture_output=True,text=True,timeout=3600)
        log=after.parent/'selected_cec.log';log.write_text(result.stdout+result.stderr)
        assert result.returncode==0 and 'Networks are equivalent' in log.read_text(),('selected CEC failed',after)
        proofs.append(dict(cec=1,cec_log_sha256=sha(log),design=r['design'],mode=r['mode'],before_sha256=sha(before),after_sha256=sha(after),**new_graph,identical=True,final_checkpoint_metrics_match=True))
    save(root/'core_dag_audit.json',dict(complete=True,analysis_sha256=sha(root/'analysis.json'),script_sha256=sha(Path(__file__)),proofs=proofs))
    print('PASS: all six buffer-free mapped core DAGs match the previous six-round configuration')
if __name__=='__main__':main()
