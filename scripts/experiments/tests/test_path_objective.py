#!/usr/bin/env python3
"""Critical XOR may grow for timing; a noncritical AND must not grow for local speed."""
import os
from pathlib import Path
import re
import subprocess
import tempfile

root=Path(__file__).resolve().parents[3]
engine=root/'build/epmapping/epmapping_check'
abc=os.environ.get('ABC','abc')
library='''GATE ZERO 0 Y=CONST0;
GATE ONE 0 Y=CONST1;
GATE BUF 1 Y=A;
PIN A UNKNOWN 1 999 1 0 1 0
GATE INV 1 Y=!A;
PIN A UNKNOWN 1 999 1 0 1 0
GATE ANDslow 1 Y=A*B;
PIN A UNKNOWN 1 999 10 0 10 0
PIN B UNKNOWN 1 999 10 0 10 0
GATE ANDfast 2 Y=A*B;
PIN A UNKNOWN 1 999 1 0 1 0
PIN B UNKNOWN 1 999 1 0 1 0
GATE XORslow 1 Y=(A*!B)+(!A*B);
PIN A UNKNOWN 1 999 100 0 100 0
PIN B UNKNOWN 1 999 100 0 100 0
GATE XORfast 3 Y=(A*!B)+(!A*B);
PIN A UNKNOWN 1 999 60 0 60 0
PIN B UNKNOWN 1 999 60 0 60 0
'''
with tempfile.TemporaryDirectory(prefix='epm-path-objective-') as directory:
    tmp=Path(directory)
    source,aig,lib=tmp/'input.v',tmp/'input.aig',tmp/'library.genlib'
    source.write_text('module top(a,b,c,d,y0,y1);\ninput a,b,c,d;\noutput y0,y1;\n'
                      'assign y0 = a ^ b;\nassign y1 = c & d;\nendmodule\n')
    lib.write_text(library)
    subprocess.run([abc,'-c',f'read_verilog {source}; strash; write_aiger {aig}'],check=True,capture_output=True)
    environment={k:v for k,v in os.environ.items() if not k.startswith('EPMAPPING_')}
    environment.update(ABC=abc,EPMAPPING_OBJECTIVE='delay_area',EPMAPPING_BASELINE='direct',
        EPMAPPING_AREA_MAPPING='1',EPMAPPING_RELAX_REQUIRED='100',EPMAPPING_WINDOW_SYN='0',
        EPMAPPING_SUPER='',EPMAPPING_SYN='0',EPMAPPING_EGG='0',EPMAPPING_SINGLE_MATCHER='brute',
        EPMAPPING_ROUNDS='1',EPMAPPING_NO_DELAY_REGRESSION='0')
    logs={}
    for hybrid in ['0','1']:
        environment['EPMAPPING_PATH_OBJECTIVE']=hybrid
        completed=subprocess.run([str(engine),str(lib),str(aig),str(tmp/hybrid)],env=environment,
                                 text=True,capture_output=True,check=True)
        assert '[i] cec=1' in completed.stdout
        logs[hybrid]=completed.stdout
    assert 'epmapping area 2 -> 5 delay 100 -> 60' in logs['0'],logs['0']
    assert 'epmapping area 2 -> 4 delay 100 -> 60' in logs['1'],logs['1']
    assert 'critical_replacements=1 noncritical_replacements=0' in logs['1']
    assert 'window_synth windows=0 applied=0' in logs['1']
print('Critical timing improves while noncritical upsizing is rejected; both outputs pass CEC')
