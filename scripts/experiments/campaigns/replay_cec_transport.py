#!/usr/bin/env python3
"""Verification-only replay of saved netlists; no mapping or baseline optimization."""
import argparse,copy,json,subprocess,time
from pathlib import Path
from run_epfl_baselines import ROOT,sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--job',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 out=a.output.resolve();out.mkdir(exist_ok=False);c=json.loads(a.campaign.read_text());j=next(j for j in c['jobs'] if j['name']==a.job)
 for key in ['report','audit']:assert sha(Path(j[key]))==j[key+'_sha256']
 report=json.loads(Path(j['report']).read_text());audit=json.loads(Path(j['audit']).read_text());records=[];old=ROOT/'results_v2/cec_scriptfile_transport_20261013/bench_rs_netlist_eval.before';new=ROOT/'target/release/bench_rs_netlist_eval'
 folder=Path(j['report']).parent;genlib=next(folder.glob('asap7_*in.genlib'))
 argv=[str(new),str(genlib),'--models',str(out/'cells.v'),str(out/'library.json')]
 p=subprocess.run(argv,capture_output=True,text=True,timeout=60);assert p.returncode==0,p.stderr
 assert sha(out/'cells.v')==sha(folder/'cells.v');assert sha(out/'library.json')==sha(folder/'library.json')
 for r in report['results']:
  commands=audit['states'][r['bench']+'/'+r['method']]['commands']
  argv=next(cmd['argv'] for cmd in commands if Path(cmd['argv'][0]).name=='bench_rs_netlist_eval' and len(cmd['argv'])==5 and '--models' not in cmd['argv'])
  files={name:sha(Path(name)) for name in argv[1:]};method=r['method'];base=out/method;base.mkdir()
  previous=[str(old),*argv[1:]];t=time.monotonic();before=subprocess.run(previous,capture_output=True,text=True,timeout=60)
  (base/'before.stdout.log').write_text(before.stdout);(base/'before.stderr.log').write_text(before.stderr)
  assert before.returncode!=0 and 'Argument list too long' in before.stderr
  current=[str(new),*argv[1:]];t=time.monotonic();after=subprocess.run(current,capture_output=True,text=True,timeout=60);wall=time.monotonic()-t
  (base/'after.stdout.log').write_text(after.stdout);(base/'after.stderr.log').write_text(after.stderr)
  parsed=[json.loads(line) for line in after.stdout.splitlines() if line.startswith('{"area":')]
  for path,digest in files.items():assert sha(Path(path))==digest
  record={'method':method,'before_argv':previous,'before_exit':before.returncode,'after_argv':current,'after_exit':after.returncode,'after_verification_secs':wall,'unchanged_artifact_sha256':files}
  r['prior_verification_failure']=copy.deepcopy(r);r['verification_replay']=record
  if after.returncode==0:
   assert len(parsed)==1 and parsed[0]['cec']==1;r['measured']=parsed[0];r['status']='ok';r.pop('error',None)
  else:r['status']='failed';r['error']='verification-only replay failed; original mapping/failure retained'
  records.append(record);print(method,r['status'],parsed,flush=True)
 report['verification_transport_replay']={'source_report':j['report'],'source_sha256':j['report_sha256'],'source_audit':j['audit'],'source_audit_sha256':j['audit_sha256'],'new_evaluator_sha256':sha(new),'old_evaluator_sha256':sha(old),'baseline_algorithms_changed':False,'mapping_rerun':False,'public_metrics_code_changed':False,'models_library_bit_identical':True}
 (out/'report.json').write_text(json.dumps(report,indent=2));(out/'validation.json').write_text(json.dumps({'job':j['name'],'records':records,'report_sha256':sha(out/'report.json'),'mapping_rerun':False,'baseline_algorithms_changed':False},indent=2))
if __name__=='__main__':main()
