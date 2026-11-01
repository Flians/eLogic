#!/usr/bin/env python3
"""Frozen fresh EPFL20 seeded-refinement campaign. No baseline algorithm is run.

One serial queue,600s mapping/search per design, CEC/format conversion excluded.
A supplied seed's historical generation time is disclosed separately, not hidden
inside claims of end-to-end direct-mapping speed. Interrupted partials stop safely.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'scripts/emapper'))
from provenance import sha,hashes,source_files,git_revision
from physical import measure_cover,contribution,METRIC_MODEL
from source_reference import bundle_verilog
from verify import verify

ORDER=['ctrl','priority','div','hyp','sqrt','mem_ctrl','log2','multiplier','square','sin',
       'bar','voter','adder','max','arbiter','cavlc','dec','i2c','int2float','router']
COMMON=['--ablation','source-editable','--recovery-inputs','overlay','--recovery-noop-prefilter','on',
        '--recovery-cache','legacy','--recovery-schedule','legacy','--area-reject-cache','off']


def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.pending');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)

def frozen_files():
 return source_files()+list((R/'scripts/emapper').glob('*.py'))+[Path(__file__).resolve(),
  R/'emapper/profiles/delivery.json',R/'target/release/emapper-build.json',
  R/'target/release/emapper_kernel',R/'target/release/emapper_measure']+list((R/'emapper/libraries/asap7').glob('*'))

def current_fingerprint():return hashes([p for p in frozen_files()if p.is_file()])

def initialize(out):
 out.mkdir(parents=True,exist_ok=True);ref=read(R/'results_v2/emapper_reject_cache_20261031/epfl/campaign.json')
 baseline=read(R/'results_v2/full_baselines_20261025/baselines.json');ledger=read(R/'results_v2/full_baselines_20261025/verified_artifacts.json')
 seed_paths=read(R/'target/emap_seeds.json');assert set(seed_paths)==set(ORDER)==set(ref['inputs'])
 inputs={}
 for name in ORDER:
  original=ref['inputs'][name];origin=next(x for x in baseline['results']if x['bench']==name and x['method']=='mockturtle_emap_delay')
  report=Path(origin['source_report']);provider=read(report);seed=Path(seed_paths[name]).resolve()
  assert origin['status']=='ok'and origin['measured']['cec']==1
  assert provider['prepared_sha256']==original['source_sha256']
  assert provider['input_sha256']==original['golden_sha256']
  assert provider['library_sha256']==sha(R/'emapper/libraries/asap7/asap7_full.genlib')
  assert str(seed)in ledger and sha(seed)==ledger[str(seed)],'seed is not the verified baseline artifact'
  folder=out/'subjects'/name;folder.mkdir(parents=True,exist_ok=False)
  paths={}
  for side,source,target in [('prepared',Path(original['source']),'prepared.aig'),
                             ('golden',Path(original['golden']),'golden.aig'),('seed',seed,'seed.v')]:
   expected=sha(source)
   if side!='seed':assert expected==original['source_sha256'if side=='prepared'else'golden_sha256']
   dest=folder/target;shutil.copy2(source,dest);assert sha(dest)==expected
   paths[side]=str(dest);paths[side+'_sha256']=expected
  shutil.copy2(report,folder/'provider-report.json')
  observation=provider.get('source_observation',{});source_report=Path(observation['source_report'])
  assert sha(source_report)==observation['source_report_sha256']
  shutil.copy2(source_report,folder/'provider-original-report.json')
  paths.update(original_input_paths={k:original[k]for k in ['source','golden']},original_seed_path=str(seed),
     seed_provider='original verified emap-delay observation; not recomputed',
     seed_provider_report=str(folder/'provider-report.json'),seed_provider_report_sha256=sha(folder/'provider-report.json'),
     seed_original_report=str(folder/'provider-original-report.json'),seed_original_report_sha256=sha(folder/'provider-original-report.json'),
     historical_seed_mapping_seconds=provider['mapping_seconds'],historical_seed_timing_scope='original recorded mapper wall observation; not a fresh end-to-end measurement',
     seed_expected_measured=origin['measured'])
  inputs[name]=paths
 protocol=dict(schema=1,scope='externally supplied emap seed + eGraphMap refinement; not standalone direct mapping',
    expected_designs=20,repetitions_per_configuration=1,order=ORDER,common_cli=COMMON,mapping_seconds=600,
    seed_generation='pre-existing verified outputs; no baseline execution; historical mapper time separately disclosed',
    excluded_from_mapping_clock=['pure format conversion','public physical measurement','independent CEC'],
    verification_reference='re-emit already recorded native physical choices; no matching or search; never feeds candidate production',
    objective='exact delay-area; lexicographic wins are not Pareto dominance',
    profile='source-editable/overlay/noop-on/legacy cache/legacy schedule/reject off; observation off',
    AIG_rounds=5,alpha=10,wide_rounds=2,beta=2,new_limits_or_pruning=False,
    metrics=METRIC_MODEL,baseline_algorithm_runs=0,GitHub_push=False,source_commit=git_revision(),inputs=inputs)
 dump(out/'protocol.json',protocol)
 c=dict(schema=1,source_commit=git_revision(),expected=20,order=ORDER,protocol_sha256=sha(out/'protocol.json'),
        frozen=current_fingerprint(),tools={k:dict(path=shutil.which(k),sha256=sha(shutil.which(k)))for k in ['yosys','abc']},
        python=dict(path=sys.executable,sha256=sha(sys.executable),version=sys.version),jobs={},baseline_runs=0)
 dump(out/'campaign.json',c);return c


def validate(out,c,key,deep=False):
 j=c['jobs'][key];assert sha(j['case'])==j['case_sha256'];record=read(j['case'])
 assert sha(record['report'])==record['report_sha256']
 if deep:
  for p,digest in record['artifact_hashes'].items():assert sha(out/p)==digest,p
 return record


def summarize(out,c):
 rows=[];counts={};observations=[]
 for name in ORDER:
  if name not in c['jobs']:
   rows.append(dict(design=name,status='missing'));continue
  r=validate(out,c,name);observations.append(r);p=read(r['report']);s=p.get('external_seed',{}).get('measured',{});v=p.get('measured',{})
  row=dict(design=name,status=r['status'],output_CEC=p.get('cec'),seed_CEC=p.get('external_seed',{}).get('CEC'),
   source_CEC=r.get('constructive',{}).get('CEC'),mapping_seconds=p.get('mapping_wall_secs'),
   historical_seed_seconds=r['historical_seed_seconds'],search_timeout=p.get('search_timeout'),
   search_finished=p.get('search_finished'),delivery_origin=r.get('delivery_origin'),
   seed_A=s.get('area'),seed_D=s.get('delay'),output_A=v.get('area'),output_D=v.get('delay'),
   source_A=r.get('constructive',{}).get('measured',{}).get('area'),source_D=r.get('constructive',{}).get('measured',{}).get('delay'),
   seed_contribution=r.get('seed_vs_constructive',{}).get('outcome'),
   output_vs_seed=p.get('search_vs_seed',{}).get('outcome'),
   validated_search_gain=p.get('search_vs_seed',{}).get('validated_search_gain',False),error=r.get('error',p.get('error')))
  rows.append(row)
 fields=['design','status','source_A','source_D','source_CEC','seed_A','seed_D','seed_CEC','output_A','output_D','output_CEC',
 'seed_contribution','output_vs_seed','validated_search_gain','delivery_origin','mapping_seconds','historical_seed_seconds','search_timeout','search_finished','error']
 with(out/'current.csv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 for r in rows:
  status=r['status'];counts[status]=counts.get(status,0)+1
 summary=dict(completed=len(c['jobs']),expected=20,complete=len(c['jobs'])==20,
    case_status=counts,output_CEC=sum(r.get('output_CEC')==1 for r in rows),
    seed_CEC=sum(r.get('seed_CEC')==1 for r in rows),constructive_CEC=sum(r.get('source_CEC')==1 for r in rows),
    validated_search_gains=sum(bool(r.get('validated_search_gain'))for r in rows),
    search_deadlines=sum(bool(r.get('search_timeout'))for r in rows),
    scope=read(out/'protocol.json')['scope'],source_commit=c['source_commit'],baseline_runs=0,
    formal_paper_updated=False,GitHub_push=False,per_design_configuration_selection=False,
    provider_generation_cost='historical timing reported separately; current T is refinement/search only',results=rows)
 dump(out/'summary.json',summary)
 if summary['complete']:
  (out/'COMPLETE.txt').write_text('All20 fixed cases attempted. See summary.json for failures, CEC and seed/search attribution; completion is not universal improvement. Paper integration requires review of this complete cohort.\n')
 return summary


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-jobs',type=int,default=20);a=ap.parse_args()
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);locks=[]
 for p in [R/'results_v2/independent_components_20261019/global.lock',out/'runner.lock']:
  p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 assert a.max_jobs>0
 c=read(out/'campaign.json')if(out/'campaign.json').exists()else initialize(out);protocol=read(out/'protocol.json')
 assert sha(out/'protocol.json')==c['protocol_sha256']
 assert c['frozen']==current_fingerprint(),'source/binary/configuration drift; do not mix versions'
 for name in c['jobs']:validate(out,c,name,deep=True)
 def state(**v):dump(out/'state.json',dict(completed=len(c['jobs']),expected=20,baseline_runs=0,GitHub_push=False,**v))
 jobs=0
 try:
  for name in ORDER:
   if name in c['jobs']:continue
   if jobs>=a.max_jobs:break
   assert c['frozen']==current_fingerprint(),'frozen dependency drift'
   assert sha(out/'protocol.json')==c['protocol_sha256']
   for tool in c['tools'].values():assert sha(tool['path'])==tool['sha256']
   inp=protocol['inputs'][name]
   for side in ['prepared','golden','seed']:assert sha(inp[side])==inp[side+'_sha256']
   for side in ['seed_provider_report','seed_original_report']:assert sha(inp[side])==inp[side+'_sha256']
   folder=out/'cases'/name;assert not folder.exists(),'unrecorded partial: audit and retain before recovery'
   argv=[sys.executable,str(R/'scripts/emapper/run.py'),inp['prepared'],'--golden',inp['golden'],
         '--output',str(folder),*COMMON,'--seed-verilog',inp['seed']]
   dump(out/(name+'-attempt.json'),dict(argv=argv,input=inp,source_commit=c['source_commit'],frozen_manifest_sha256=sha(out/'campaign.json')))
   state(status='running_mapping_or_output_CEC',active=name,complete=False);print(json.dumps(dict(event='start',design=name)),flush=True)
   # Keep the global/case lock alive in the supervising child if this controller
   # is interrupted. An ambiguous partial still requires explicit recovery.
   with(out/(name+'.log')).open('x')as log:p=subprocess.run(argv,cwd=R,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(f.fileno()for f in locks))
   report=folder/'report.json';r=read(report);assert r['status']!='running','interrupted output: preserve partial'
   assert c['frozen']==current_fingerprint(),'source changed during the timed run'
   record=dict(design=name,report=str(report),report_sha256=sha(report),runner_exit=p.returncode,
      historical_seed_seconds=inp['historical_seed_mapping_seconds'],scope=protocol['scope'],status='failed')
   try:
    if r['status']not in ['ok','source_only']:raise ValueError('mapper/output validation did not succeed: '+r['status'])
    assert r['cec']==1 and r['external_seed']['CEC']==1
    assert r['configuration']['mapping_seconds']==600 and r['CEC_in_mapping_clock']is False
    assert r['evaluation']['exact_agreement']is True
    assert all(r['external_seed']['measured'][k]==inp['seed_expected_measured'][k]for k in ['area','delay','instances'])
    assert r['search_vs_seed']['outcome']!='regressed','seed guard regressed; retain adverse case'
    bundle_path=folder/'source.bundle.json';bundle=read(bundle_path)
    reference=folder/'constructive-reference.v';reference.write_text(bundle_verilog(bundle,R/'emapper/libraries/asap7/library.json'))
    measured=measure_cover(R/'emapper/libraries/asap7/asap7_full.genlib',reference,folder/'constructive-reference',
                         R/'target/release/emapper_measure',c['tools']['yosys']['path'])
    os.environ['EGM_YOSYS']=c['tools']['yosys']['path'];os.environ['BENCH_RS_EMAP_ABC']=c['tools']['abc']['path']
    good=verify(R/'emapper/libraries/asap7/cells.v',inp['golden'],measured['netjson'],folder/'constructive-reference/cec')
    measured.update(CEC=int(good),reference_kind='export of recorded physical choices; never fed back to search',
       source_bundle_sha256=sha(bundle_path),search_or_matching_called=False)
    dump(folder/'constructive-reference/audit.json',measured);record['constructive']=measured
    assert good,'recorded constructive reference failed CEC'
    assert all(measured['measured'][k]==bundle[k]for k in ['area','delay'])
    record['seed_vs_constructive']=contribution(measured['measured'],r['external_seed']['measured'])
    record['output_vs_constructive']=contribution(measured['measured'],r['measured'])
    record['delivery_origin']='search'if r['search_vs_seed']['validated_search_gain']else'provided_seed'if r['search_vs_seed']['outcome']=='equal'else'native_constructive_or_other_nonsearch'
    record['status']='verified'
   except Exception as e:record['error']=repr(e)
   # Freeze case evidence including all source/pin proofs and failed records.
   record['artifact_hashes']={str(p.relative_to(out)):sha(p)for p in folder.rglob('*')if p.is_file()}
   record['artifact_hashes'][str(report.relative_to(out))]=sha(report)
   dump(out/(name+'-case.json'),record)
   c['jobs'][name]=dict(case=str(out/(name+'-case.json')),case_sha256=sha(out/(name+'-case.json')),status=record['status'])
   dump(out/'campaign.json',c);jobs+=1;s=summarize(out,c)
   print(json.dumps(dict(event='end',design=name,status=record['status'],output_CEC=r.get('cec'),search_vs_seed=r.get('search_vs_seed'))),flush=True)
  s=summarize(out,c)
  if s['complete']:
   for name in ORDER:validate(out,c,name,deep=True)
   assert c['frozen']==current_fingerprint(),'final frozen source check failed'
   dump(out/'full-artifact-validation.json',dict(completed=20,all_case_artifact_hashes_verified=True,
      records=sum(len(read(j['case'])['artifact_hashes'])for j in c['jobs'].values()),
      source_and_binaries_still_frozen=True,
      protocol_sha256=sha(out/'protocol.json'),baseline_runs=0))
  state(status='complete_ready_for_paper_review'if s['complete']else'finite_batch_complete',active=None,complete=s['complete'])
 except BaseException as e:state(status='stopped_preserve_partial',active=name if'name'in locals()else None,complete=False,error=repr(e));raise

if __name__=='__main__':main()
