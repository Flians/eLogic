"""Independent provenance checks for actual mapped metrics and original named ports."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[3]
def audit(job,campaign):
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 d=json.loads(Path(job['report']).read_text());r=d['results'][0];a=json.loads(Path(job['audit']).read_text());prepared=r['preprocessing'];subjects=R/'results_v2/epfl_resyn2_subjects_20261012/report.json'
 assert prepared['cec']==1 and prepared['source_report_sha256']==campaign['fingerprint'][str(subjects)]==sha(subjects)
 assert prepared['recipe']=='balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance'
 assert sha(prepared['reused_from'])==prepared['sha256']
 assert a['mapping_seconds']==300 and not a['native_parameters_changed'] and d['timeout_secs']==60
 assert a['driver_sha256']==campaign['fingerprint'][a['driver']]
 lib=Path(d['library']);lib=lib if lib.is_absolute() else Path(job['report']).parent/lib;assert sha(lib)==d['library_sha256']
 base=json.loads((R/f'results_v2/guarded_joint_evidence_20261016/baseline_scope/baselines_{job["k"]}in.json').read_text());assert d['metric_model']==base['metric_model'] and d['library_sha256']==base['library_sha256']
 cmds=[cmd for state in a['states'].values() for cmd in state['commands']];mapping=[cmd for cmd in cmds if cmd['mapping_charged']];assert len(mapping)==1
 argv=mapping[0]['argv'];assert sha(argv[0])==campaign['fingerprint'][argv[0]] and sha(argv[1])==d['library_sha256'] and sha(argv[2])==prepared['sha256']
 assert all(not s['exhausted'] and s['mapping_wall_secs']<=300 for s in a['states'].values())
 proof=dict(original_input_sha256=r['input_sha256'],prepared_input_sha256=prepared['sha256'],library_sha256=d['library_sha256'],actual_mapping_argv=argv,mapper_sha256=sha(argv[0]),mapping_calls=1,output_hashes_first_captured_by_audit=True)
 if r['status']=='ok':
  evals=[x for x in cmds if Path(x['argv'][0]).name=='bench_rs_netlist_eval' and len(x['argv'])==5 and not x['argv'][2].startswith('--')];assert len(evals)==1
  e=evals[0];v=e['argv'];assert not e['mapping_charged'] and sha(v[1])==d['library_sha256'] and sha(v[2])==r['input_sha256'] and sha(v[0])==campaign['fingerprint'][v[0]]
  folder=Path(job['report']).parent/(r['bench']+'.'+r['method']);log=folder/'evaluate.log';records=[json.loads(x) for x in log.read_text().splitlines() if x.startswith('{"area":')];assert records==[r['measured']] and r['measured']['cec']==1
  proof.update(evaluator_argv=v,evaluator_sha256=sha(v[0]),mapped_verilog_sha256=sha(v[3]),mapped_json_sha256=sha(v[4]),evaluation_log_sha256=sha(log),independent_original_named_port_cec=1,cec_excluded_from_mapping_clock=True)
 return proof
