#!/usr/bin/env python3
"""The post-mapping deadline must kill search, exclude CEC, and preserve an outer alarm."""
import ast
import os
from pathlib import Path
import subprocess
import tempfile

root=Path(__file__).resolve().parents[3]
engine=root/'build/epmapping/epmapping_check'
tree=ast.parse(Path(__file__).with_name('test_path_objective.py').read_text())
library=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='library' for t in n.targets))
with tempfile.TemporaryDirectory(prefix='epm-post-timeout-') as directory:
    tmp=Path(directory)
    env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','MAP_EGG_'))}
    env.update(EPMAPPING_BASELINE='direct',EPMAPPING_OBJECTIVE='area_delay_bound',
               EPMAPPING_POST_TIMEOUT='1',EPMAPPING_SUPER='',EPMAPPING_WINDOW_SYN='0',RAYON_NUM_THREADS='1')
    result=subprocess.run([str(engine),str(root/'results_v8_area_bound/inputs/library.genlib'),
                           str(root/'tools/mockturtle/experiments/benchmarks/aes_core.aig'),str(tmp/'large')],
                          env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==-14, (result.returncode,result.stderr)
    assert 'post_mapping_timeout=1 seconds' in result.stdout and '[i] cec=1' not in result.stdout
    # Deliberately slow validation stub isolates timer scope; it is not an equivalence test.
    (tmp/'input.v').write_text('module top(a,b,c,d,y0,y1);\ninput a,b,c,d;\noutput y0,y1;\nassign y0 = a ^ b;\nassign y1 = c & d;\nendmodule\n')
    (tmp/'library.genlib').write_text(library)
    subprocess.run(['abc','-c',f'read_verilog {tmp / "input.v"}; strash; write_aiger {tmp / "input.aig"}'],check=True,capture_output=True)
    stub=tmp/'slow-validation'; stub.write_text('#!/bin/sh\nsleep 3\necho "Networks are equivalent"\n'); stub.chmod(0o755)
    env.update(ABC=str(stub),EPMAPPING_SYN='0',EPMAPPING_EGG='0',EPMAPPING_ROUNDS='1')
    command=[str(engine),str(tmp/'library.genlib'),str(tmp/'input.aig'),str(tmp/'tiny')]
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0 and '[i] cec=1' in result.stdout, (result.returncode,result.stdout,result.stderr)
    result=subprocess.run(['perl','-e','alarm shift; exec @ARGV','2',*command],env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==-14, result.stdout
print('Post deadline kills active optimization, excludes validation, and restores the outer process alarm')
