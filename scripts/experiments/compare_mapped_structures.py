#!/usr/bin/env python3
"""Compare mapped DAGs by physical pins and ordered PI identities, ignoring wires."""
import argparse,collections,hashlib,json,re
from pathlib import Path

def graph(path):
    text=path.read_text();inputs=[x.strip()for x in re.search(r'\binput\s+(.*?);',text,re.S)[1].split(',')];outputs=[x.strip()for x in re.search(r'\boutput\s+(.*?);',text,re.S)[1].split(',')]
    drivers={};aliases=dict(re.findall(r'\bassign\s+(\S+)\s*=\s*(\S+)\s*;',text));keys={x:'pi:'+str(i)for i,x in enumerate(inputs)};keys.update({"1'b0":'0',"1'b1":'1'})
    for name,ports in re.findall(r'^\s*(\w+)\s+\w+\s*\((.*?)\)\s*;',text,re.M|re.S):
        if name in ['module','input','output','wire']:continue
        pins=dict((p,w.strip())for p,w in re.findall(r'\.(\w+)\s*\(\s*([^)]*)\s*\)',ports))
        if name in ['_const0_','_const1_']:
            keys[pins['z']]='1'if name=='_const1_'else'0';continue
        if 'Y'not in pins:continue
        out=pins.pop('Y');drivers[out]=(name,sorted(pins.items()))
    def resolve(w):
        while w in aliases:w=aliases[w]
        return w
    for w in list(drivers)+outputs:
        stack=[(resolve(w),False)]
        while stack:
            node,ready=stack.pop();node=resolve(node)
            if node in keys:continue
            name,pins=drivers[node]
            if ready:keys[node]=hashlib.sha256(json.dumps([name,[(p,keys[resolve(c)])for p,c in pins]],separators=(',',':')).encode()).hexdigest()
            else:
                stack.append((node,True))
                for _,c in reversed(pins):
                    c=resolve(c)
                    if c not in keys:stack.append((c,False))
    return dict(inputs=inputs,outputs=outputs,drivers=drivers,keys=keys,output_keys=[keys[resolve(w)]for w in outputs],aliases=aliases)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('old',type=Path);ap.add_argument('new',type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();old=graph(a.old);new=graph(a.new);assert len(old['inputs'])==len(new['inputs'])and len(old['outputs'])==len(new['outputs'])
    changed=[i for i,(u,v)in enumerate(zip(old['output_keys'],new['output_keys']))if u!=v];common=set(old['keys'].values())&set(new['keys'].values())
    def unique(g):return [(w,n)for w,(n,_)in g['drivers'].items()if g['keys'][w]not in common]
    def counts(g):return collections.Counter(n for n,_ in g['drivers'].values())
    c0,c1=counts(old),counts(new)
    out=dict(old=str(a.old),new=str(a.new),changed_outputs=changed,cell_count_changes={k:c1[k]-c0[k]for k in c0.keys()|c1.keys()if c1[k]!=c0[k]},old_unique=unique(old),new_unique=unique(new))
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(changed_outputs=len(changed),first_changed=changed[:30],cell_count_changes=out['cell_count_changes'],old_unique_cells=len(out['old_unique']),new_unique_cells=len(out['new_unique'])),indent=2))
if __name__=='__main__':main()
