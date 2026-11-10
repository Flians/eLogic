#!/usr/bin/env python3
"""Frozen supplemental controls; no historical record is rewritten.

Retention is applied to the SAME vector-extracted root pool, not to intermediate
extraction fronts. Replay records offered candidates on live local snapshots;
it does not claim counterfactual full-network optimization trajectories.
"""
import argparse,concurrent.futures,datetime,hashlib,json,os,re,shutil,struct,subprocess,sys,time,traceback
from pathlib import Path
from audit_mapped_ppa import audit
from recover_checkpoint import recover
from report import read_counters
from run_three_parts import DESIGNS
PROJECT=Path(__file__).resolve().parents[2]
REFERENCE=PROJECT/'results_v28_three_parts_query_budget'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def call(cmd,env,log,timeout=3600):
 start=time.monotonic()
 with Path(log).open('w')as f:
  try:rc=subprocess.run(cmd,cwd=PROJECT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout).returncode
  except subprocess.TimeoutExpired:rc=124
 return rc,time.monotonic()-start

def audit_abc(netlist,library,float32=False):
 """Independent fixed-pin propagation, supporting ABC cell syntax and PO aliases."""
 gates={}
 for block in re.split(r'(?m)^GATE\s+',Path(library).read_text())[1:]:
  first=block.splitlines()[0].split()
  pins={p:max(float(r),float(f))for p,r,f in re.findall(r'PIN\s+(\S+)\s+\S+\s+\S+\s+\S+\s+(\S+)\s+\S+\s+(\S+)',block)}
  gates[first[0]]=(float(first[1]),pins,first[2])
 text=Path(netlist).read_text();inputs=re.findall(r'\binput\s+(.*?);',text,re.S);outputs=re.findall(r'\boutput\s+(.*?);',text,re.S)
 arr={n.strip():0. for group in inputs for n in group.split(',')};po={n.strip()for group in outputs for n in group.split(',')};arr.update({"1'b0":0.,"1'b1":0.})
 pending=[];area=0.;cells=0
 for name,instance,ports in re.findall(r'^\s*(\w+)\s+(\w+)\s*\((.*?)\)\s*;',text,re.M|re.S):
  if name not in gates:continue
  ports=dict((p,s.strip())for p,s in re.findall(r'\.(\w+)\s*\(\s*([^)]*?)\s*\)',ports));out=ports.pop('Y')
  cost,pins,fun=gates[name];assert set(ports)==set(pins),(name,ports,pins)
  alias=len(ports)==1 and fun=='Y=A;' and next(iter(ports.values()))in po and out in po
  pending.append((out,ports,pins,0. if alias else cost,alias));cells+=1
 for out,value in re.findall(r'\bassign\s+(\S+)\s*=\s*(\S+)\s*;',text):
  assert '~'not in value,('unmapped inversion',out,value)
  pending.append((out,{'A':value},{'A':0.},0.,False))
 while pending:
  rest=[]
  for out,ports,pins,cost,alias in pending:
   if not all(v in arr for v in ports.values()):rest.append((out,ports,pins,cost,alias));continue
   cast=lambda x:struct.unpack('f',struct.pack('f',x))[0]
   arr[out]=max((cast(arr[v]+cast(0. if alias else pins[k]))if float32 else arr[v]+(0. if alias else pins[k]))for k,v in ports.items());area+=cost
  assert len(rest)<len(pending),('unresolved cells',rest[:2]);pending=rest
 assert cells and all(o in arr for o in po),(cells,po-set(arr))
 return dict(area=area,delay=max(arr[o]for o in po),cells=cells)

def initialize(root,binary):
 f=root/'inputs';manifest=root/'manifest.json'
 if manifest.exists():
  m=json.loads(manifest.read_text())
  for n,h in m['input_sha256'].items():assert sha(f/n)==h,n
  assert sha(f/'engine')==m['engine_sha256']
  return m
 f.mkdir(parents=True);old=REFERENCE/'cold/inputs'
 for name in ['asap7.genlib','asap7_6in.genlib','database.super','abc']:shutil.copy2(old/name,f/name)
 shutil.copytree(old/'benchmarks',f/'benchmarks');shutil.copy2(binary,f/'engine')
 source=f/'sources';source.mkdir()
 for p in (PROJECT/'epmapping').glob('*'):
  if p.suffix in ['.hpp','.cpp']:shutil.copy2(p,source/p.name)
 shutil.copytree(PROJECT/'map_egg/src',source/'map_egg_src')
 for name in ['run_epmapping_supplement.py','recover_checkpoint.py','audit_mapped_ppa.py','report.py']:
  shutil.copy2(PROJECT/'scripts/experiments'/name,source/name)
 env={}
 for line in (REFERENCE/'cold/tab_main/ae.usb_phy/experiment_env.txt').read_text().splitlines():
  if line.startswith(('EPMAPPING_','MAP_EGG_')):
   k,v=line.split('=',1);env[k]=v
 env.update(EPMAPPING_SUPER=str(f/'database.super'),EPMAPPING_BASELINE='high_effort',EPMAPPING_POST_TIMEOUT='1000',EPMAPPING_WINDOW_SYN='0',EPMAPPING_DSE='0',MAP_EGG_PROFILE_TT_CACHE='0',EPMAPPING_STRUCTURAL_PIVOTS='0',EPMAPPING_ROUNDS='6')
 env.pop('MAP_EGG_PROFILE_CACHE_DIR',None)
 m=dict(schema=1,recorded_at=datetime.datetime.now().astimezone().isoformat(),designs=DESIGNS,profile_arms=['vector','scalar','single','noegg'],post_timeout=1000,abc_timeout=1000,common_rounds=1,profile_rounds=6,whole_window_synthesis=False,preprocessing='fraig; resyn2rs; fraig; compress2rs until AND count stops decreasing; cap 20',goal=env,engine_sha256=sha(f/'engine'),input_sha256={str(p.relative_to(f)):sha(p)for p in f.rglob('*')if p.is_file()},retention_scope='same bounded vector-extracted root pool; final retention only',replay_scope='up to 64 offered-candidate roots per round, local snapshot ranking only; instrumentation timing excluded from profile comparisons',cold='empty private pool',warm='paired cold pool; new process; OS caches not cleared',abc_modes='two independently timed processes; ABC has no e-graph pool',original_campaign=str(REFERENCE))
 save(manifest,m);return m

def engine(root,m,part,arm,design,mode,cache,rounds=None):
 wd=root/part/arm/f'{mode}.{design}';result=wd/'result.json'
 if result.exists():return json.loads(result.read_text())
 wd.mkdir(parents=True,exist_ok=True);f=root/'inputs';env={k:v for k,v in os.environ.items()if not k.startswith(('EPMAPPING_','MAP_EGG_','EPM_'))}
 options=dict(m['goal'],MAP_EGG_PROFILE_CACHE_DIR=str(cache),EPMAPPING_PROFILE_RETAIN=arm if arm in ['vector','scalar','single']else'vector')
 options.update(EPMAPPING_ROUNDS='1'if part=='common'else'6')
 if rounds is not None:options['EPMAPPING_ROUNDS']=str(rounds)
 if arm=='noegg':options['EPMAPPING_EGG']='0'
 if part=='replay':options.update(EPMAPPING_SELECTION_REPLAY=str(wd/'candidates.jsonl'),EPMAPPING_REPLAY_ROOTS='64')
 if mode=='cold':assert not cache.exists(),('nonempty cold pool',cache)
 else:assert cache.is_dir()
 cache.mkdir(parents=True,exist_ok=True)
 env.update(options,ABC=str(f/'abc'),RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1')
 library=f/('asap7_6in.genlib'if part=='common'else'asap7.genlib');bench=f/'benchmarks'/f'{design}.aig'
 cmd=[str(f/'engine'),str(library),str(bench),str(wd)];save(wd/'command.json',dict(command=cmd,env=options,engine_sha256=m['engine_sha256'],input_sha256=sha(bench),library_sha256=sha(library)))
 rc,wall=call(cmd,env,wd/'log');log=(wd/'log').read_text();out=dict(part=part,arm=arm,design=design,mode=mode,rc=rc,wall_seconds=wall,status='error',cache_files_before_warm=None)
 assert 'ABC baseline: high_effort'in log and 'ands (no reduction); fixpoint after'in log and 'stopped at the iteration cap'not in log,wd
 assert 'window_synth windows=0 applied=0'in log or rc==-14,wd
 assert 'post_mapping_timeout=1000'in log
 initial=audit(wd/'epmapping_baseline.v',library)
 if part!='common':
  old=REFERENCE/mode/'tab_main'/f'ae.{design}'/'epmapping_baseline.v'
  assert sha(old)==sha(wd/'epmapping_baseline.v'),('changed full-library initial mapping',wd)
 out.update(initial=initial,initial_sha256=sha(wd/'epmapping_baseline.v'))
 if rc==0:
  assert '[i] cec=1'in log,wd
  selected=wd/'epmapping_mapped.v';out.update(status='complete',cec=1,optimizer_seconds=read_counters(str(wd/'log'))['time_optimizer'])
 elif rc==-14:
  proof=recover(wd,library,bench,f/'abc',rc);selected=wd/proof['checkpoint']['netlist'];out.update(status='timeout',cec=1,optimizer_seconds=1000,selection=proof['selection'])
 else:
  save(result,out);raise RuntimeError((wd,rc))
 final=audit(selected,library)
 assert final['area']<=initial['area']+.0001 and final['delay']<=initial['delay']+.001,(wd,initial,final)
 assert final['serialized_area']<=initial['serialized_area']+.0001 and final['serialized_delay']<=initial['serialized_delay']+.001,(wd,'serialized regression')
 assert out['optimizer_seconds']<=1000.1
 out.update(final=final,selected_netlist=str(selected.relative_to(root)),selected_sha256=sha(selected),profile_stats=re.findall(r'\[i\] egraph_profile (.*)',log),cache_files=sum(1 for _ in cache.rglob('*.json')))
 if part=='profile':
  prior=audit(REFERENCE/mode/'tab_main'/f'ae.{design}'/'epmapping_mapped.v',library)
  out['v28_same_ppa']=abs(prior['area']-final['area'])<.0001 and abs(prior['delay']-final['delay'])<.001
 save(result,out);print(json.dumps({k:out[k]for k in ['part','arm','design','mode','status','optimizer_seconds']}),flush=True);return out

def abc(root,m,design,mode,baseline):
 wd=root/'common/abc'/f'{mode}.{design}';result=wd/'result.json'
 if result.exists():return json.loads(result.read_text())
 wd.mkdir(parents=True,exist_ok=True);f=root/'inputs';lib=f/'asap7_6in.genlib';env=dict(os.environ,RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1');cmd=lambda s:[str(f/'abc'),'-c',s]
 rc,_=call(cmd(f'read_genlib {lib}; read -m {baseline}; print_stats'),env,wd/'baseline.log');assert rc==0
 def metrics(path):
  t=path.read_text();return{k:float(re.findall(r'\b'+k+r'\s*=\s*([0-9.]+)',t)[-1])for k in ['area','delay']}
 native=metrics(wd/'baseline.log');initial=audit(baseline,lib)
 script=f'read_genlib {lib}; read -m {baseline}; mfs3 -a; print_stats; write_verilog {wd/"mapped.v"}'
 save(wd/'command.json',dict(script=script,input_sha256=sha(baseline),library_sha256=sha(lib),timeout=1000))
 rc,secs=call(cmd(script),env,wd/'log',1000)
 out=dict(part='common',arm='abc',design=design,mode=mode,rc=rc,status='timeout'if rc==124 else'complete',optimizer_seconds=min(secs,1000.)if rc==124 else secs,initial=initial,native_initial=native,initial_sha256=sha(baseline),cec=1,guard_selected='initial')
 if rc==0:
  crc,_=call(cmd(f'read_genlib {lib}; read -m {wd/"mapped.v"}; cec -n {f/"benchmarks"/(design+".aig")}'),env,wd/'cec.log');assert crc==0 and'Networks are equivalent'in(wd/'cec.log').read_text()
  raw=audit_abc(wd/'mapped.v',lib);out.update(raw=raw,native_final=metrics(wd/'log'),raw_sha256=sha(wd/'mapped.v'))
  feasible=raw['area']<=initial['area']+.0001 and raw['delay']<=initial['delay']+.001
  native_feasible=out['native_final']['area']<=native['area']+.0001 and out['native_final']['delay']<=native['delay']+.001
  out.update(raw_feasible=feasible,native_feasible=native_feasible,final=raw if feasible and native_feasible else initial,guard_selected='optimized'if feasible and native_feasible else'initial')
 else:
  assert rc==124,(wd,rc)
  # Initial mapping has independent CEC from the paired method's final proof and
  # immutable original inputs; explicitly validate the fallback as well.
  crc,_=call(cmd(f'read_genlib {lib}; read -m {baseline}; cec -n {f/"benchmarks"/(design+".aig")}'),env,wd/'cec.log');assert crc==0 and'Networks are equivalent'in(wd/'cec.log').read_text();out['final']=initial
 save(result,out);print(json.dumps({k:out[k]for k in ['part','arm','design','mode','status','optimizer_seconds','guard_selected']}),flush=True);return out

def pair(root,m,part,arm,design):
 cache=root/'cache'/part/arm/design
 for mode in ['cold','warm']:
  out=engine(root,m,part,arm,design,mode,cache)
  if part=='common':abc(root,m,design,mode,root/part/arm/f'{mode}.{design}'/'epmapping_baseline.v')
 if part=='common':
  assert sha(root/part/arm/f'cold.{design}'/'epmapping_baseline.v')==sha(root/part/arm/f'warm.{design}'/'epmapping_baseline.v')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--binary',type=Path,default=PROJECT/'build/epmapping/epmapping_check');ap.add_argument('--jobs',type=int,default=6);ap.add_argument('--designs',nargs='+',default=DESIGNS);ap.add_argument('--parts',default='profile,replay,common');a=ap.parse_args()
 root=a.root.resolve();root.mkdir(parents=True,exist_ok=True);m=initialize(root,a.binary)
 jobs=[]
 for d in a.designs:
  if'profile'in a.parts.split(','):
   jobs.extend(('profile',arm,d)for arm in m['profile_arms'])
  if'replay'in a.parts.split(','):jobs.append(('replay','vector',d))
  if'common'in a.parts.split(','):jobs.append(('common','ours',d))
 errors=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs)as pool:
  futures={pool.submit(pair,root,m,*job):job for job in jobs}
  for future in concurrent.futures.as_completed(futures):
   try:future.result()
   except Exception as e:errors.append((futures[future],repr(e)));traceback.print_exc()
 save(root/'execution_status.json',dict(jobs=len(jobs),errors=errors,complete=not errors));assert not errors,errors
if __name__=='__main__':main()
