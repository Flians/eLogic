#!/usr/bin/env python3
"""Imported shared mapping preserves cells/pins/aliases, fixes the cleaned D0."""
import json,os,subprocess,tempfile,sys,re
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'scripts/experiments'))
from audit_mapped_ppa import audit
from run_buffer_core_validation import buffer_area
from exact_core_graph import exact_core_fingerprint
from mapped_shared_blif import convert
F=R/'results_epmapping_three_parts_buffer_v1_20261111';lib=F/'inputs/asap7.genlib';exe=R/'build/epmapping/epmapping_check'
env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','MAP_EGG_','EPM_'))};env.update(ABC=str(F/'inputs/abc'),EPMAPPING_SUPER='',EPMAPPING_PRE_BUFFER_CLEANUP='1',RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1')
rows=[]
with tempfile.TemporaryDirectory(prefix='epm-prebuffer-') as tmp:
 p=Path(tmp)
 for design in ['ethernet','sasc','usb_phy']:
  source=F/'main/ours'/f'cold.{design}/epmapping_baseline.v';aig=F/'inputs/benchmarks'/(design+'.aig')
  env.update(EPMAPPING_MAPPED_INPUT=str(source),EPMAPPING_SKIP='1');w=p/design/'prepare'
  q=subprocess.run([str(exe),str(lib),str(aig),str(w)],env=env,capture_output=True,text=True,timeout=90)
  assert q.returncode==0,q.stdout+q.stderr
  clean=w/'epmapping_mapped.v';old=audit(source,lib);new=audit(clean,lib);buf,_=buffer_area(clean,lib)
  assert buf==0 and new['area']<=old['area']+.0001 and new['delay']<=old['delay']+.001
  assert exact_core_fingerprint(source,lib)==exact_core_fingerprint(clean,lib),'normalization changed core DAG/pins/sharing'
  shared=clean.with_suffix('.blif');convert(clean,lib,shared)
  env['EPMAPPING_MAPPED_INPUT']=str(shared);w2=p/design/'reimport'
  q=subprocess.run([str(exe),str(lib),str(aig),str(w2)],env=env,capture_output=True,text=True,timeout=90)
  assert q.returncode==0 and clean.read_bytes()==(w2/'epmapping_mapped.v').read_bytes(),q.stdout+q.stderr
  cec=subprocess.run([str(F/'inputs/abc'),'-c',f'read_genlib {lib}; read -m {clean}; cec -n {aig}'],capture_output=True,text=True,timeout=90)
  assert cec.returncode==0 and 'Networks are equivalent' in cec.stdout,cec.stdout+cec.stderr
  if design in ['sasc','usb_phy']:
   env['EPMAPPING_SKIP']='0';w3=p/design/'optimize'
   q=subprocess.run([str(exe),str(lib),str(aig),str(w3)],env=env,capture_output=True,text=True,timeout=120)
   assert q.returncode==0 and '[i] cec=1' in q.stdout,q.stdout+q.stderr
   assert clean.read_bytes()==(w3/'epmapping_baseline.v').read_bytes(),'optimizer changed supplied initial representation'
   final=audit(w3/'epmapping_mapped.v',lib)
   assert final['delay']<=new['delay']+.001 and final['area']<=new['area']+.0001
   assert buffer_area(w3/'epmapping_mapped.v',lib)[0]==0
  rows.append(dict(design=design,raw=old,normalized=new,CEC=1,core_identical=True,byte_exact_reimport=True))
 # Aliases, constants, shared outputs and distinct duplicate physical nodes.
 text="module top(x0,x1,y0,y1,y2,y3,y4,y5); input x0,x1; output y0,y1,y2,y3,y4,y5; wire a,b; AND2x2_ASAP7_75t_R a0(.A(x0),.B(x1),.Y(a)); AND2x2_ASAP7_75t_R b0(.A(x0),.B(x1),.Y(b)); assign y0=x0; assign y1=a; assign y2=a; assign y3=b; assign y4=1'b0; assign y5=1'b1; endmodule\n"
 fixture=p/'fixture.v';fixture.write_text(text.replace(";", ";\n"));blif=p/'fixture.blif';convert(fixture,lib,blif)
 env.update(EPMAPPING_SKIP='1',EPMAPPING_MAPPED_INPUT=str(blif))
 q=subprocess.run([str(exe),str(lib),str(F/'inputs/benchmarks/usb_phy.aig'),str(p/'fixture')],env=env,capture_output=True,text=True,timeout=30)
 assert q.returncode==0,q.stdout+q.stderr
 out=p/'fixture/epmapping_mapped.v';assert exact_core_fingerprint(fixture,lib)==exact_core_fingerprint(out,lib)
 assert buffer_area(out,lib)[0]==0 and len(re.findall('AND2x2_ASAP7_75t_R',out.read_text()))==2
 # Re-running the original strong preprocessing/mapping reaches the same cleaned
 # start as the frozen-input normalization; no mapped-input shortcut in this check.
 env.pop('EPMAPPING_MAPPED_INPUT');env['EPMAPPING_BASELINE']='high_effort'
 raw=p/'original-pipeline'
 q=subprocess.run([str(exe),str(lib),str(F/'inputs/benchmarks/usb_phy.aig'),str(raw)],env=env,capture_output=True,text=True,timeout=90)
 assert q.returncode==0,q.stdout+q.stderr
 assert (raw/'epmapping_mapped.v').read_bytes()==(p/'usb_phy/prepare/epmapping_mapped.v').read_bytes(),'original preprocessing/mapping input changed'
 print(json.dumps(dict(complete=True,original_preprocessing_matches=True,shared_input_roundtrip=True,wire_constant_duplicate_nodes=True,records=rows),indent=2))
