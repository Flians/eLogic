#!/usr/bin/env python3
"""The constrained-area goal must reject critical upsizing and override an old path flag."""
import os
from pathlib import Path
import runpy
import subprocess
import tempfile

root=Path(__file__).resolve().parents[3]
fixture=runpy.run_path(str(Path(__file__).with_name('test_path_objective.py')))
library=fixture['library']
abc=os.environ.get('ABC','abc')
engine=root/'build/epmapping/epmapping_check'
with tempfile.TemporaryDirectory(prefix='epm-area-bound-') as directory:
    tmp=Path(directory)
    source,aig,lib=tmp/'input.v',tmp/'input.aig',tmp/'library.genlib'
    source.write_text('module top(a,b,c,d,y0,y1);\ninput a,b,c,d;\noutput y0,y1;\n'
                      'assign y0 = a ^ b;\nassign y1 = c & d;\nendmodule\n')
    lib.write_text(library)
    subprocess.run([abc,'-c',f'read_verilog {source}; strash; write_aiger {aig}'],check=True,capture_output=True)
    for mode in ['area_delay_bound','default','override']:
        environment={k:v for k,v in os.environ.items() if not k.startswith('EPMAPPING_')}
        environment.update(ABC=abc,EPMAPPING_BASELINE='direct',EPMAPPING_AREA_MAPPING='1',
            EPMAPPING_RELAX_REQUIRED='100',EPMAPPING_PATH_OBJECTIVE='1',EPMAPPING_WINDOW_SYN='0',
            EPMAPPING_GLOBAL_OBJECTIVE_GUARD='0',EPMAPPING_SUPER='',EPMAPPING_SYN='0',
            EPMAPPING_EGG='0',EPMAPPING_SINGLE_MATCHER='brute',EPMAPPING_ROUNDS='1')
        if mode=='area_delay_bound':
            environment['EPMAPPING_OBJECTIVE']=mode
        elif mode=='override':
            environment.update(EPMAPPING_OBJECTIVE='delay_area',EPMAPPING_AREA_UNDER_DELAY='1')
        result=subprocess.run([str(engine),str(lib),str(aig),str(tmp/mode)],env=environment,
                              capture_output=True,text=True,check=True)
        assert '[i] cec=1' in result.stdout
        assert 'objective_order=area_delay, global_objective_guard=1' in result.stdout
        assert 'area_under_delay=1, path_objective=0' in result.stdout
        assert 'max_area_growth=0, max_delay_growth=0, local_area_growth=0' in result.stdout
        assert 'epmapping area 2 -> 2 delay 100 -> 100' in result.stdout
        assert 'window_synth windows=0 applied=0' in result.stdout
print('Default and explicit bounded-area goals reject timing-only upsizing and supersede path mode')
