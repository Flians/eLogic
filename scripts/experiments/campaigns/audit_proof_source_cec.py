#!/usr/bin/env python3
"""Replay frozen source proof and independently CEC its serialized source AIG.
These are source-only diagnostic checks, never mapped covers or free advice.
"""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();O=a.output.resolve();O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();save=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n')
helper=Path('/tmp/emapper-source-proof-audit-20261016/target/release/source-proof-audit');abc=Path('/usr/bin/abc');cp=R/'results_v2/proof_source_pilot_20261016/campaign.json';campaign=json.loads(cp.read_text());records=[]
def run(argv,log,env=None):
 start=time.monotonic()
 try:r=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=60);rc=r.returncode;text=r.stdout+r.stderr;timedout=False
 except subprocess.TimeoutExpired as e:rc=None;text=(e.stdout or b'').decode(errors='replace')+(e.stderr or b'').decode(errors='replace')+'\nAudit60s timeout, unknown.\n';timedout=True
 log.write_text(text);return dict(argv=argv,exit_code=rc,timeout=timedout,elapsed=time.monotonic()-start,log=str(log),log_sha256=sha(log)),text

def port_names(path):
 raw=Path(path).read_bytes();header,rest=raw.split(b'\n',1);kind,m,ni,l,no,na=header.decode().split();assert kind=='aig' and int(l)==0;ni,no,na=map(int,(ni,no,na))
 for _ in range(no):_,rest=rest.split(b'\n',1)
 offset=0
 for _ in range(na*2):
  while True:
   value=rest[offset];offset+=1
   if value<128:break
 inputs={};outputs={}
 for line in rest[offset:].decode().splitlines():
  if line=='c':break
  if line.startswith(('i','o')):
   key,value=line.split(' ',1);(inputs if key[0]=='i' else outputs)[int(key[1:])]=value
 assert len(inputs)==ni and len(outputs)==no
 return dict(inputs=[inputs[i] for i in range(ni)],outputs=[outputs[i] for i in range(no)])

def check(name,original,negative=False,post_cleanup=False):
 compressed=O/(name+'.aig');diag=O/(name+'.diag.json');env={k:v for k,v in os.environ.items() if not k.startswith('EGM_')};env['EGM_POOL_DIAG_PATH']=str(diag)
 h,helper_stdout=run([str(helper),str(original),str(compressed)]+(['--post-cleanup'] if post_cleanup else []),O/(name+'.helper.log'),env);assert h['exit_code']==0,h
 helper_result=json.loads(helper_stdout.splitlines()[-1]);side=diag.with_suffix('.source-proof.json');q=json.loads(side.read_text())
 if negative:
  head,lit,tail=compressed.read_bytes().split(b'\n',2);compressed.write_bytes(head+b'\n'+str(int(lit)^1).encode()+b'\n'+tail)
 ports=port_names(original);assert ports==port_names(compressed),'exact named ports must survive source replay'
 argv=[str(abc),'-c',f'cec "{original}" "{compressed}"'];record,text=run(argv,O/(name+'.cec.log'));equivalent=('Networks are equivalent' in text and not record['timeout'] and record['exit_code']==0);different=('NOT EQUIVALENT' in text.upper() and not record['timeout'] and record['exit_code']==0)
 row=dict(name=name,post_cleanup=post_cleanup,helper_result=helper_result,exact_named_ports=ports,original=str(original),original_sha256=sha(original),replayed_source=str(compressed),replayed_source_sha256=sha(compressed),proof=q,proof_path=str(side),proof_sha256=sha(side),helper=h,cec=record,expected_equivalent=not negative,equivalent=equivalent,counterexample=different,passed=different if negative else equivalent,scope='source AIG only, NOT mapped-netlist CEC/PPA or budget-cover success; no advice supplied to a mapper')
 if name in ['div','hyp','mem_ctrl']:
  actual=[]
  for j in campaign['jobs']:
   if j['bench']==name and j['profile']=='protected_area':
    real=json.loads(Path(j['source_proof']).read_text());fields=['before_source_and','after_source_and','proved_aliases','examined_and','unknown','proof_decisions','proof_calls'];assert all(real[k]==q[k] for k in fields),(name,j['k']);actual.append(dict(k=j['k'],path=j['source_proof'],sha256=j['source_proof_sha256']))
  assert len(actual)==2;row['matches_both_actual_native_proof_counts']=actual
 records.append(row);save(O/'report.json',dict(records=records,physical_mapping_attempts=0,physical_mapping_cec=0,algorithms_changed=False,helper_sha256=sha(helper),abc_sha256=sha(abc),campaign=str(cp),campaign_sha256=sha(cp),source_proof_sha256=sha(R/'results_v2/proof_source_implementation_20261016/frozen_source/emapper/src/source_proof.rs'),purpose='Independent exact-source replay and source-only equivalence, not changed official inputs or free cached preoptimization. Same frozen module; proof counter matches both library mapping attempts, all graph/egraph failures preserved.'))
 print(name,row['passed'],flush=True)
for inv in [0,1]:
 original=O/f'calibration_original_{inv}.aig';original.write_bytes(f'aig 4 2 0 1 2\n{8+inv}\n'.encode()+bytes([1,2,1,5])+b'i0 a[0]\ni1 b[0]\no0 y[0]\nc\nCalibration exact absorption and source phase.\n')
 check(f'calibration_{inv}_positive',original);check(f'calibration_{inv}_negative',original,True)
for name in ['div','hyp','mem_ctrl']:check(name,R/f'results_v2/epfl_resyn2_subjects_20261012/{name}.resyn2.aig')
check('div_post_cleanup',R/'results_v2/epfl_resyn2_subjects_20261012/div.resyn2.aig',post_cleanup=True)
assert all(x['passed'] for x in records),[(x['name'],x['passed']) for x in records]
