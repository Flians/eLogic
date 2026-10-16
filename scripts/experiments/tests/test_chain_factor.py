#!/usr/bin/env python3
"""Shared affine chains save physical area; an unshared expansion must roll back."""
from pathlib import Path
import os,re,subprocess,tempfile
root=Path(__file__).resolve().parents[3];abc=os.environ.get('ABC','abc')
lib=root/'tools/mockturtle/experiments/cell_libraries/asap7.genlib'
for count in [1,8]:
 with tempfile.TemporaryDirectory(prefix='epm-chain-') as folder:
  p=Path(folder);length=16
  inputs=[f'p{i}' for i in range(length)]+[f'q{i}' for i in range(length)]+[f'b{j}' for j in range(count)]
  outputs=[f'y{j}' for j in range(count)];wires=[];assigns=[]
  for j in range(count):
   previous=f'b{j}'
   for i in range(length):
    term=f'm{j}_{i}';out=f'r{j}_{i}';wires += [term,out]
    assigns += [f'assign {term} = p{i} & {previous};',f'assign {out} = {term} | q{i};'];previous=out
   assigns.append(f'assign y{j} = {previous};')
  source='module top('+','.join(inputs+outputs)+');\ninput '+','.join(inputs)+';\noutput '+','.join(outputs)+';\nwire '+','.join(wires)+';\n'+'\n'.join(assigns)+'\nendmodule\n'
  (p/'input.v').write_text(source)
  q=subprocess.run([abc,'-c',f'read_verilog {p}/input.v; strash; write_aiger {p}/input.aig'],capture_output=True,text=True,check=True)
  assert (p/'input.aig').exists(),q.stdout
  logs={}
  for flag in ['0','1','noSHARE']:
   env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','MAP_EGG_'))}
   env.update(ABC=abc,EPMAPPING_BASELINE='direct',EPMAPPING_AREA_MAPPING='1',EPMAPPING_AREA_UNDER_DELAY='1',EPMAPPING_WINDOW_SYN='0',EPMAPPING_POST_TIMEOUT='1000',EPMAPPING_CHAIN_FACTOR='0' if flag=='0' else '1',EPMAPPING_CHAIN_SHARE='0' if flag=='noSHARE' else '1',EPMAPPING_CHAIN_BLOCK='4',EPMAPPING_SUPER='',EPMAPPING_SYN='0',EPMAPPING_EGG='0',EPMAPPING_ROUNDS='1',EPMAPPING_WRITE_BASELINE='1')
   q=subprocess.run([str(root/'build/epmapping/epmapping_check'),str(lib),str(p/'input.aig'),str(p/flag)],env=env,capture_output=True,text=True,check=True);logs[flag]=q.stdout
   assert '[i] cec=1' in q.stdout and 'window_synth windows=0 applied=0' in q.stdout
  assert (p/'0/epmapping_baseline.v').read_bytes()==(p/'1/epmapping_baseline.v').read_bytes()
  metrics=lambda s:tuple(map(float,re.search(r'epmapping area ([\d.]+) -> ([\d.]+) delay ([\d.]+) -> ([\d.]+)',s).groups()))
  a,aa,d,dd=metrics(logs['1']);control=metrics(logs['0'])
  assert dd<=d+.001
  assert metrics(logs['noSHARE'])[1]==control[1],logs['noSHARE']
  assert (p/'0/epmapping_mapped.v').read_bytes()==(p/'noSHARE/epmapping_mapped.v').read_bytes()
  if count==8:
   assert aa<control[1]-.001 and 'accepted=1' in logs['1'],logs['1']
   assert 'segment_hits=0 ' not in logs['1']
  else:
   assert aa==control[1] and 'accepted=0' in logs['1'],logs['1']
   assert (p/'0/epmapping_mapped.v').read_bytes()==(p/'1/epmapping_mapped.v').read_bytes()
print('Exact affine-chain sharing lowers area with CEC and delay bound; an area-increasing transaction restores the original netlist')
