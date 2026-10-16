#!/usr/bin/env python3
"""A faster noncritical gate must not inflate area at unchanged global delay."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[3]
engine = root / 'build/epmapping/epmapping_check'
abc = os.environ.get('ABC', 'abc')
library = '''GATE ZERO 0 Y=CONST0;
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
'''
with tempfile.TemporaryDirectory(prefix='epm-global-guard-') as directory:
    tmp = Path(directory)
    source, aig, lib = tmp / 'input.v', tmp / 'input.aig', tmp / 'library.genlib'
    source.write_text('module top(a,b,c,d,y0,y1);\ninput a,b,c,d;\noutput y0,y1;\n'
                      'assign y0 = a ^ b;\nassign y1 = c & d;\nendmodule\n')
    lib.write_text(library)
    prepared = subprocess.run([abc, '-c', f'read_verilog {source}; strash; write_aiger {aig}'],
                              check=True, capture_output=True, text=True)
    assert aig.exists(), prepared.stdout + prepared.stderr
    environment = dict(os.environ, ABC=abc, EPMAPPING_OBJECTIVE='delay_area',
                       EPMAPPING_BASELINE='direct', EPMAPPING_AREA_MAPPING='1',
                       EPMAPPING_RELAX_REQUIRED='100', EPMAPPING_WINDOW_SYN='0',
                       EPMAPPING_SUPER='', EPMAPPING_SYN='0', EPMAPPING_EGG='0',
                       EPMAPPING_SINGLE_MATCHER='brute', EPMAPPING_ROUNDS='1',
                       EPMAPPING_NO_DELAY_REGRESSION='0')
    logs = {}
    for guard in ('0', '1'):
        environment['EPMAPPING_GLOBAL_OBJECTIVE_GUARD'] = guard
        result = subprocess.run([str(engine), str(lib), str(aig), str(tmp / ('run'+guard))],
                                env=environment, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert '[i] cec=1' in result.stdout
        logs[guard] = result.stdout
    assert 'epmapping area 2 -> 3 delay 100 -> 100' in logs['0']
    assert 'objective_guard rollbacks=1 discarded_replacements=1' in logs['1']
    assert 'epmapping area 2 -> 2 delay 100 -> 100' in logs['1']
    # A hard area budget is enforced even when the lexicographic guard is off.
    environment['EPMAPPING_GLOBAL_OBJECTIVE_GUARD']='0'
    environment['EPMAPPING_MAX_AREA_GROWTH']='0'
    result=subprocess.run([str(engine),str(lib),str(aig),str(tmp/'budget')],
                          env=environment,capture_output=True,text=True,check=True)
    assert '[i] cec=1' in result.stdout
    assert 'objective_guard rollbacks=1 discarded_replacements=1' in result.stdout
    assert 'epmapping area 2 -> 2 delay 100 -> 100' in result.stdout
print('Real-network rollback restores area at tied global delay; both outputs pass CEC')
