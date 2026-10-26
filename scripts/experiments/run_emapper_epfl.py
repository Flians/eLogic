#!/usr/bin/env python3
"""Fixed full-EPFL comparison from the sole working-tree eMapper entrypoint."""
import argparse,fcntl,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/emapper'))
from archive import sha,dump,seal,verify_files
from provenance import source_files,hashes,git_revision
from frontend import verify_receipt
ORDER=['ctrl','dec','cavlc','int2float','priority','router','i2c','adder','bar','max','sin','sqrt','multiplier','square','arbiter','voter','log2','mem_ctrl','div','hyp']

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--subjects',type=Path,default=ROOT/'results_v2/epfl_resyn2_subjects_20261012/report.json');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-jobs',type=int,required=True,help='finite execution batch, not a mapping search restriction');a=ap.parse_args();assert a.max_jobs>0
 out=a.output.resolve();assert out.is_relative_to(ROOT),'results must be persistent workspace files';out.mkdir(exist_ok=True,parents=True);locks=[]
 for p in [out/'runner.lock',ROOT/'results_v2/independent_components_20261019/global.lock',ROOT/'results_v2/eqmap_search500_20261024/family.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 lib=ROOT/'emapper/libraries/asap7/asap7_full.genlib';native=ROOT/'target/release/emapper_kernel';measure=ROOT/'target/release/emapper_measure';profile=ROOT/'emapper/profiles/delivery.json';build=ROOT/'target/release/emapper-build.json'
 assert json.loads(build.read_text())==dict(source=hashes(source_files()),binaries=hashes([native,measure]))
 files=source_files()+list((ROOT/'scripts/emapper').glob('*.py'))+[profile,build,lib,lib.parent/'cells.v',lib.parent/'library.json',native,measure]
 dependencies=hashes(files);tools={n:dict(path=shutil.which(n),sha256=sha(Path(shutil.which(n))))for n in ['abc','yosys']}
 subjects=json.loads(a.subjects.read_text());assert subjects['preprocess']=='resyn2';by={r['bench']:r for r in subjects['results']};assert set(by)==set(ORDER)
 inputs={}
 for n in ORDER:
  r=by[n];assert r['status']=='ok'and r['preprocessing']['cec']==1
  inp=a.subjects.resolve().parent/(n+'.resyn2.aig');golden=ROOT/'tools/mockturtle/experiments/benchmarks'/(n+'.aig');assert sha(inp)==r['preprocessing']['sha256']and sha(golden)==r['input_sha256'];inputs[n]=dict(source=str(inp),source_sha256=sha(inp),golden=str(golden),golden_sha256=sha(golden))
 frozen=dict(dependencies=dependencies,tools=tools,inputs=inputs,driver_sha256=sha(Path(__file__)),subjects_sha256=sha(a.subjects.resolve()))
 cp=out/'campaign.json'
 if cp.exists():
  state=json.loads(cp.read_text());assert state['frozen']==frozen,'Working code changed: use new batch, never mix source versions'
 else:state=dict(schema=1,scope='full20, one main source and one binary; SDC layout only differs; no baseline execution',git_head_at_start=git_revision(),mapping_seconds=600,CEC_in_clock=False,order=ORDER,arms=['functional-fine','functional-lazy'],expected_jobs=40,baseline_runs=0,frozen=frozen,jobs={})
 def check_runtime():
  for rel,h in frozen['dependencies'].items():assert sha(ROOT/rel)==h,rel
  for t in frozen['tools'].values():assert sha(Path(t['path']))==t['sha256']
 def verify(v):
  p=Path(v['report']);assert sha(p)==v['report_sha256'];r=json.loads(p.read_text());assert r['dependencies']==dependencies and r['tools']==tools;verify_files(p.parent,r['artifacts'],v.get('capsule'))
  if r.get('incumbent'):
   got=verify_receipt(p.parent/'raw.incumbent.receipt.json',lib,Path(inputs[v['design']]['source']),native);assert got==r['incumbent']and sha(p.parent/'raw.v')==got['netlist_sha256']
  return r
 def save():
  vs=list(state['jobs'].values());state.update(completed_jobs=len(vs),CEC_valid=sum(v.get('CEC')==1 for v in vs),optimized_CEC_valid=sum(v.get('status')=='ok'and v.get('CEC')==1 and v.get('optimized_cover')for v in vs),full_scope_attempted=len(vs)==40,full_EPFL_acceptance=False);dump(cp,state)
 for v in state['jobs'].values():
  if not v.get('capsule')and(out/'capsules'/(Path(v['report']).parent.name+'.json')).exists():
   v['capsule']=seal(Path(v['report']).parent,out/'capsules');save()
  verify(v)
  if not v.get('capsule'):
   v['capsule']=seal(Path(v['report']).parent,out/'capsules');verify(v);save()
 save();count=0
 for n in ORDER:
  inp=inputs[n]
  for arm in state['arms']:
   key=n+'/'+arm
   if key in state['jobs']:continue
   if count==a.max_jobs:print(json.dumps(dict(completed=state['completed_jobs'],expected=40)),flush=True);return
   check_runtime();assert sha(Path(inp['source']))==inp['source_sha256']and sha(Path(inp['golden']))==inp['golden_sha256'];folder=out/'jobs'/f'{n}-{arm}';folder.parent.mkdir(exist_ok=True)
   if folder.exists():raise RuntimeError('Unrecorded partial preserved; audit before adoption: '+str(folder))
   argv=[sys.executable,str(ROOT/'scripts/emapper/run.py'),inp['source'],'--golden',inp['golden'],'--output',str(folder),'--ablation',arm];dump(out/'active.json',dict(key=key,phase='mapping',argv=argv))
   with folder.with_suffix('.log').open('w')as log:code=subprocess.run(['/usr/bin/time','-v','-o',str(folder.with_suffix('.resource.txt'))]+argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
   p=folder/'report.json';r=json.loads(p.read_text());assert r['dependencies']==dependencies and r['tools']==tools
   v=dict(design=n,arm=arm,argv=argv,exit=code,report=str(p),report_sha256=sha(p),status=r['status'],CEC=r.get('cec'),optimized_cover=r.get('selection',{}).get('kind')=='optimized',search_finished=r.get('search_finished'),search_timeout=r.get('search_timeout'),measured=r.get('measured'),mapping_wall_secs=r.get('mapping_wall_secs'),error=r.get('error'))
   verify(v);state['jobs'][key]=v;save();dump(out/'active.json',dict(key=key,phase='durable_archive_outside_mapping_clock'));v['capsule']=seal(folder,out/'capsules');verify(v);save();dump(out/'active.json',dict(key=None,previous=key));count+=1;print(json.dumps({k:v[k]for k in ['design','arm','status','CEC','optimized_cover','measured','mapping_wall_secs']}),flush=True)
 print(json.dumps(dict(completed=state['completed_jobs'],expected=40)),flush=True)
if __name__=='__main__':main()
