#!/usr/bin/env python3
"""Native post-mapping regression: identity outputs become wires, core remains."""
import os, subprocess, tempfile, re, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from audit_mapped_ppa import audit
f=ROOT/'results_epmapping_supplement_v1_20261110/inputs'
env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','MAP_EGG_'))}
env.update(ABC=str(f/'abc'),EPMAPPING_BASELINE='direct',EPMAPPING_SUPER='',
 EPMAPPING_EGG='0',EPMAPPING_SYN='0',EPMAPPING_WINDOW_SYN='0',EPMAPPING_ROUNDS='1',
 EPMAPPING_AREA_UNDER_DELAY='1',EPMAPPING_POST_TIMEOUT='1000',EPMAPPING_WRITE_BASELINE='1')
with tempfile.TemporaryDirectory(prefix='epm-buffer-') as d:
 p=Path(d);v=p/'input.v';aig=p/'input.aig'
 v.write_text('module top(a,b,y0,y1,y2,y3,y4,y5,y6);\ninput a,b;\noutput y0,y1,y2,y3,y4,y5,y6;\nassign y0 = a;\nassign y1 = a;\nassign y2 = ~a;\nassign y3 = a & b;\nassign y4 = a & b;\nassign y5 = 1\'b0;\nassign y6 = 1\'b1;\nendmodule\n')
 prep=subprocess.run([str(f/'abc'),'-c',f'read_verilog {v}; strash; write_aiger {aig}'],capture_output=True,text=True,check=True)
 assert aig.exists(),prep.stdout+prep.stderr
 command=[str(ROOT/'build/epmapping/epmapping_check'),str(f/'asap7_6in.genlib'),str(aig)]
 env['EPMAPPING_BUFFER_CLEANUP']='0'
 off=subprocess.run(command+[str(p/'off')],env=env,capture_output=True,text=True,timeout=20)
 assert off.returncode==0 and '[i] cec=1' in off.stdout,off.stdout+off.stderr
 env['EPMAPPING_BUFFER_CLEANUP']='1'
 q=subprocess.run(command+[str(p/'run')],env=env,capture_output=True,text=True,timeout=20)
 assert q.returncode==0,q.stdout+q.stderr
 assert '[i] cec=1' in q.stdout,q.stdout+q.stderr
 base=(p/'run/epmapping_baseline.v').read_text();out=(p/'run/epmapping_mapped.v').read_text()
 assert 'BUFx' in base,'fixture must expose real initial buffers'
 assert 'BUFx' not in out, 'identity buffers remain in delivered network'
 assert len(re.findall(r'\bassign\s+y\d+\s*=',out))>=3,out
 assert (p/'off/epmapping_baseline.v').read_bytes()==(p/'run/epmapping_baseline.v').read_bytes()
 old=audit(p/'off/epmapping_mapped.v',f/'asap7_6in.genlib')
 final=audit(p/'run/epmapping_mapped.v',f/'asap7_6in.genlib')
 assert final['area']<old['area']-.0001 and final['delay']<=old['delay']+.001,(old,final)
 record=json.loads((p/'run/epmapping_checkpoint.json').read_text())
 saved=audit(p/'run'/record['netlist'],f/'asap7_6in.genlib')
 assert abs(saved['area']-final['area'])<.0001 and abs(saved['delay']-final['delay'])<.001,(saved,final)
 assert 'BUFx' not in (p/'run'/record['netlist']).read_text()
 print('PASS: native buffer cleanup, PI/output aliases, retained inverter and shared logic, CEC')
