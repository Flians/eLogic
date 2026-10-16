#!/usr/bin/env python3
"""A root may differ internally only when its difference cannot reach an output."""
from pathlib import Path
import os,re,subprocess,tempfile
root=Path(__file__).resolve().parents[3];abc=os.environ.get('ABC','abc')
library='''GATE ZERO 0 Y=CONST0;
GATE ONE 0 Y=CONST1;
GATE BUF 1 Y=A;
PIN A UNKNOWN 1 999 1 0 1 0
GATE INV 1 Y=!A;
PIN A UNKNOWN 1 999 1 0 1 0
GATE AND2 1 Y=A*B;
PIN A UNKNOWN 1 999 10 0 10 0
PIN B UNKNOWN 1 999 10 0 10 0
GATE OR2 1 Y=A+B;
PIN A UNKNOWN 1 999 10 0 10 0
PIN B UNKNOWN 1 999 10 0 10 0
GATE AO21 2 Y=A*B+C;
PIN A UNKNOWN 1 999 10 0 10 0
PIN B UNKNOWN 1 999 10 0 10 0
PIN C UNKNOWN 1 999 10 0 10 0
'''
fixtures={
 'masked':('output z;\nwire y;','',3,2),
 'direct_po':('output z,y;','',3,3),
 'branch':('output z,w;\nwire y;','assign w = y & b;\n',4,4),
}
for name,(declarations,extra,base,optimized) in fixtures.items():
 with tempfile.TemporaryDirectory(prefix='epm-odc-') as folder:
  p=Path(folder);ports='a,b,c,z'+(',y' if name=='direct_po' else ',w' if name=='branch' else '')
  source='module top('+ports+');\ninput a,b,c;\n'+declarations+'\nwire ab;\nassign ab = a & b;\nassign y = ab | c;\nassign z = y & a;\n'+extra+'endmodule\n'
  (p/'input.v').write_text(source);(p/'lib.genlib').write_text(library)
  q=subprocess.run([abc,'-c',f'read_verilog {p}/input.v; strash; write_aiger {p}/input.aig'],capture_output=True,text=True,check=True)
  assert (p/'input.aig').exists(),q.stdout
  for flag in ['0','1']:
   env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','MAP_EGG_'))}
   env.update(ABC=abc,EPMAPPING_BASELINE='direct',EPMAPPING_AREA_MAPPING='1',EPMAPPING_AREA_UNDER_DELAY='1',EPMAPPING_WINDOW_SYN='0',EPMAPPING_POST_TIMEOUT='1000',EPMAPPING_OBSERVABILITY_CARE=flag,EPMAPPING_OBSERVABILITY_BUDGET='20',EPMAPPING_FEASIBLE_COVERS='1',EPMAPPING_SUPER='',EPMAPPING_SYN='0',EPMAPPING_EGG='0',EPMAPPING_ROUNDS='6',EPMAPPING_WRITE_BASELINE='1')
   q=subprocess.run([str(root/'build/epmapping/epmapping_check'),str(p/'lib.genlib'),str(p/'input.aig'),str(p/flag)],env=env,capture_output=True,text=True,check=True)
   expected=base if flag=='0' else optimized
   assert f'epmapping area {base} -> {expected} delay 20 -> 20' in q.stdout,q.stdout
   assert '[i] cec=1' in q.stdout and 'window_synth windows=0 applied=0' in q.stdout
   if flag=='1' and name=='masked':
    m=re.search(r'observability cuts=(\d+) queries=(\d+) cleared=(\d+) projected=(\d+) unknown=\d+ wins=(\d+)',q.stdout)
    assert m and all(int(m[i])>0 for i in [1,2,3,4,5]),q.stdout
  assert (p/'0/epmapping_baseline.v').read_bytes()==(p/'1/epmapping_baseline.v').read_bytes()
print('ODC replacement saves area at fixed delay with CEC; direct PO and branched observers remain protected')
