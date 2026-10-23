"""Independent raw-genlib-vs-Verilog logic oracle, without the mapper's expression parser."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--models',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cells=[]
    for line in a.library.read_text().splitlines():
        m=re.match(r'GATE\s+(\w+)\s+\S+\s+(\w+)\s*=\s*(.*);\s*$',line.strip())
        if not m:continue
        name,pin,expr=m.groups()
        variables=sorted(set(re.findall(r'\b[A-Za-z_]\w*\b',expr)) - {'CONST0','CONST1'})
        if len(variables)>6:
            assert len(variables)<=9
            cells.append((name,pin,expr,variables))
    assert len(cells)==7
    lines=['module equivalence(input [8:0] x, output fail);','wire [6:0] raw, actual;']
    for i,(name,pin,expr,variables) in enumerate(cells):
        mapping={v:f'x[{p}]' for p,v in enumerate(variables)}
        direct=re.sub(r'\b[A-Za-z_]\w*\b',lambda m:mapping[m[0]],expr)
        direct=direct.replace('*','&').replace('+','|').replace('!','~')
        lines.append(f'assign raw[{i}] = {direct};')
        connections=[f'.{v}({mapping[v]})'for v in variables]+[f'.{pin}(actual[{i}])']
        lines.append(f'{name} g{i} ({", ".join(connections)});')
    lines+=['assign fail = |(raw ^ actual);','endmodule']
    positive=a.output/'miter.v';positive.write_text('\n'.join(lines)+'\n')
    negative=a.output/'negative.v'
    # Build the negative with an explicit output inversion, without changing parser syntax.
    negative.write_text(positive.read_text().replace('assign fail = |(raw ^ actual);','assign fail = |(raw ^ actual ^ 7\'b0000001);'))
    commands=[]
    for path,expected in [(positive,0),(negative,1)]:
        script=f'read_verilog "{a.models.resolve()}" "{path.resolve()}"; hierarchy -top equivalence; proc; flatten; opt_clean; sat -verify -prove fail 0 -show-inputs'
        cmd=['/usr/local/bin/yosys','-Q','-T','-p',script];r=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
        log=a.output/(path.stem+'.log');log.write_text(r.stdout+r.stderr)
        assert (r.returncode==0)==(expected==0),log
        if expected:
            assert 'proof did fail' in (r.stdout+r.stderr), 'negative was not a SAT counterexample'
        commands.append(dict(argv=cmd,exit_code=r.returncode,log=str(log),log_sha256=sha(log)))
    (a.output/'validation.json').write_text(json.dumps(dict(cells=[c[0]for c in cells],free_boolean_inputs=9,
        equivalent_assignments=512,raw_genlib_not_mapper_ast=True,positive=True,wrong_output_negative_rejected=True,
        library_sha256=sha(a.library),models_sha256=sha(a.models),checker_sha256=sha(__file__),
        yosys_sha256=sha('/usr/local/bin/yosys'),commands=commands),indent=2)+'\n')

if __name__=='__main__':main()
