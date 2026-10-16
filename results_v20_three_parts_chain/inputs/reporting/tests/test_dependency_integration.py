from pathlib import Path
import subprocess,os,tempfile,re
root=Path(__file__).resolve().parents[3];abc=os.environ.get('ABC','abc')
libtext='''GATE ZERO 0 Y=CONST0;
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
'''
fixtures={
 'side_branch':('assign z = b | c;', 4,2,'wins=1'),
 'identity':('wire bc;\nassign bc = b | c;\nassign z = a & bc;',5,3,'reuse=1'),
}
for name,(side,base,optimized,counter) in fixtures.items():
 with tempfile.TemporaryDirectory(prefix='epm-dependency-integration-') as folder:
  p=Path(folder)
  (p/'input.v').write_text('module top(a,b,c,y,z);\ninput a,b,c;\noutput y,z;\nwire ab,ac;\nassign ab = a & b;\nassign ac = a & c;\nassign y = ab | ac;\n'+side+'\nendmodule\n')
  (p/'lib.genlib').write_text(libtext)
  prepared=subprocess.run([abc,'-c',f'read_verilog {p}/input.v; strash; write_aiger {p}/input.aig'],capture_output=True,text=True,check=True)
  assert (p/'input.aig').exists(),prepared.stdout
  for flag in ['0','1','global']:
   env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','MAP_EGG_'))}
   env.update(ABC=abc,EPMAPPING_BASELINE='direct',EPMAPPING_AREA_MAPPING='1',EPMAPPING_AREA_UNDER_DELAY='1',EPMAPPING_WINDOW_SYN='0',EPMAPPING_POST_TIMEOUT='1000',EPMAPPING_DEPENDENCY_CUTS=('1' if flag=='1' else '0'),EPMAPPING_GLOBAL_DIVISORS=('1' if flag=='global' else '0'),EPMAPPING_SUPER='',EPMAPPING_SYN='0',EPMAPPING_EGG='0',EPMAPPING_ROUNDS='6',EPMAPPING_WRITE_BASELINE='1')
   q=subprocess.run([str(root/'build/epmapping/epmapping_check'),str(p/'lib.genlib'),str(p/'input.aig'),str(p/flag)],env=env,capture_output=True,text=True,check=True)
   expected=base if flag=='0' else optimized
   assert f'epmapping area {base} -> {expected} delay 20 -> 20' in q.stdout,q.stdout
   assert '[i] cec=1' in q.stdout and 'window_synth windows=0 applied=0' in q.stdout
   assert 'functional_rejections=0 build_rejections=0 cycle_rejections=0' in q.stdout
   if flag=='1':assert counter in q.stdout,q.stdout
   if flag=='global':
    import re
    m=re.search(r'global_divisors sampled=\d+ sat_queries=(\d+) proved=(\d+) unknown=\d+ wins=(\d+)',q.stdout)
    assert m and all(int(m[i])>0 for i in [1,2,3]),q.stdout
  for mode in ['1','global']:
   assert (p/'0/epmapping_baseline.v').read_bytes()==(p/mode/'epmapping_baseline.v').read_bytes()
print('Real mapped-netlist dependency replacement and direct divisor reuse reduce area, preserve delay, and pass CEC')
