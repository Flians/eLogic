"""Canonical ordered-root traversal retaining physical node identity and sharing."""
import hashlib,json,re

def exact_core_fingerprint(netlist,library):
    functions={}
    for block in re.split(r'(?m)^GATE\s+',library.read_text())[1:]:
        row=block.splitlines()[0].split();functions[row[0]]=row[2]
    text=netlist.read_text()
    inputs=[n.strip() for g in re.findall(r'\binput\s+(.*?);',text,re.S) for n in g.split(',')]
    outputs=[n.strip() for g in re.findall(r'\boutput\s+(.*?);',text,re.S) for n in g.split(',')]
    ids={n:-3-i for i,n in enumerate(inputs)};ids.update({"1'b0":0,"1'b1":1})
    drivers={}
    for name,instance,ports in re.findall(r'^\s*(\w+)\s+(\w+)\s*\((.*?)\)\s*;',text,re.M|re.S):
        if name not in functions:continue
        ports=dict((p,v.strip()) for p,v in re.findall(r'\.(\w+)\s*\(\s*([^)]*?)\s*\)',ports))
        function=functions[name];out=ports.pop(function.split('=')[0]);assert out not in drivers
        if function.endswith('=CONST0;'):drivers[out]=('alias',"1'b0")
        elif function.endswith('=CONST1;'):drivers[out]=('alias',"1'b1")
        elif function=='Y=A;':drivers[out]=('alias',ports['A'])
        else:drivers[out]=('cell',name,tuple(sorted(ports.items())))
    for out,value in re.findall(r'\bassign\s+(\S+)\s*=\s*(\S+)\s*;',text):
        assert out not in drivers and '~' not in value
        drivers[out]=('alias',value)
    aliases={}
    def resolve(w):
        seen=[]
        while w in drivers and drivers[w][0]=='alias':
            if w in aliases:w=aliases[w];break
            assert w not in seen,('alias cycle',w)
            seen.append(w);w=drivers[w][1]
        for name in seen:aliases[name]=w
        return w
    cells=[];visiting=set()
    for output in outputs:
        stack=[(resolve(output),False)]
        while stack:
            wire,expanded=stack.pop();wire=resolve(wire)
            if wire in ids:continue
            assert wire in drivers and drivers[wire][0]=='cell',('undriven core signal',wire)
            _,name,ports=drivers[wire]
            if expanded:
                ids[wire]=2+len(cells)
                cells.append([name,[[pin,ids[resolve(value)]] for pin,value in ports]])
                visiting.remove(wire)
            else:
                assert wire not in visiting,('core cycle',wire)
                visiting.add(wire);stack.append((wire,True))
                # LIFO order gives a deterministic traversal by physical pin name.
                for pin,value in reversed(ports):
                    source=resolve(value)
                    if source not in ids:stack.append((source,False))
    count=sum(v[0]=='cell' for v in drivers.values())
    assert len(cells)==count,('unreachable physical core cells',count-len(cells))
    payload=[len(inputs),cells,[ids[resolve(o)] for o in outputs]]
    return dict(core_dag_sha256=hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest(),core_cells=len(cells),outputs=len(outputs))
