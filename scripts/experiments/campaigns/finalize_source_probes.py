#!/usr/bin/env python3
"""Audit two independent source probes; never substitute missing/failed covers."""
import argparse,copy,hashlib,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();O=a.output;O.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text());save=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n')
old=R/'results_v2/guarded_joint_evidence_20261016';audit=old/'baseline_integrity.json';aa=load(audit);assert aa['checked_files']==13403 and not aa['drift'];save(O/'baseline_audit_reference.json',dict(path=str(audit),sha256=sha(audit),scope='Reference prior13403-file zero-drift audit; NOT a new full baseline audit or rerun. This work changes only independent candidate sources/harness and paper.'))
checks={};allrows={};gates=[]
primary=['abc_map_area_resyn2','abc_map_delay_resyn2','mockturtle_emap_area_resyn2','mockturtle_emap_delay_resyn2','eqmap_lut_amap','esyn_frozen_model_amap_bounded_dag','emorphic_final_sa']
for version,folder in [('signed','signed_source_pilot_20261016'),('joint','joint_source_pilot_20261016')]:
 c=load(R/'results_v2'/folder/'campaign.json');assert c['status']=='attempted' and len(c['jobs'])==c['expected_jobs'];errors=[];entries=[]
 for file,h in c['fingerprint'].items():
  if sha(Path(file))!=h:errors.append(['source',file])
 for j in c['jobs']:
  for key in ['report','audit','pool','candidates','trace','pool_frontier','pool_guard','helper_allocation','diagnostic_profile']:
   if key in j and sha(Path(j[key]))!=j[key+'_sha256']:errors.append(['artifact',j['name'],key])
  r=load(Path(j['report']))['results'][0];d=Path(j['report']).parent/(r['bench']+'.'+r['method']);diag={}
  for suffix in ['import','canonical-import','helper-allocation','joint-source']:
   f=d/('pool-diagnostics.'+suffix+'.json')
   if f.exists():diag[suffix]=dict(path=str(f),sha256=sha(f),record=load(f),first_captured_by_readonly_audit=suffix=='joint-source')
  if version=='joint':
   assert r['settings']['EGM_JOINT_SOURCE']==str(int(j['profile']=='protected_area')) and r['settings']['EGM_SHARED_MAJORITY']=='1'
   assert r['status']=='failed' and not r.get('partial_measured')
   imp=diag['import']['record'];assert imp['import_complete'] is False and imp['cumulative_import_admitted_enodes']==imp['node_allowance']==20000
   q=diag['joint-source']['record'];assert q['before']>=q['after_single']>=q['after_pairs'] and q['pair_trial_cap_per_root']==16
  entries.append(dict(name=j['name'],k=j['k'],bench=j['bench'],profile=j['profile'],row=r,diagnostics=diag,report=j['report'],audit=j['audit']))
 assert not errors,errors
 checks[version]=dict(fresh_attempts=len(entries),cec=sum(e['row']['status']=='ok' for e in entries),errors=errors,actual_failures_retained=True,promoted=False);allrows[version]=entries;save(O/(version+'_rows.json'),entries)
 G=O/(version+'_baseline_scope');G.mkdir()
 for k in [3,4]:
  b=old/f'baseline_scope/baselines_{k}in.json';base=load(b);bybench={e['bench']:e for e in entries if e['k']==k and e['profile']=='protected_area'};d=copy.deepcopy(base);d['results']=[]
  # Exactly20 candidate status slots, including explicitly unmeasured scope.
  benches=list(dict.fromkeys(r['bench'] for r in base['results']));assert len(benches)==20
  for bench in benches:
   if bench in bybench:r=copy.deepcopy(bybench[bench]['row'])
   else:r=dict(bench=bench,input_sha256=next(r['input_sha256'] for r in base['results'] if r['bench']==bench),status='missing',error='Not measured in this explicitly bounded source probe. No old/main cover substituted.')
   r['method']='egraph_'+version+'_source_probe';d['results'].append(r)
  d['objective']='delay-area';d['area_delay_check_is_posthoc']=True;cp=G/f'candidate_{k}in.json';save(cp,d)
  for objective in ['delay-area','area-delay','pareto']:
   dest=G/f'{k}in_{objective}.json';argv=[sys.executable,str(R/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(cp),'--baselines',str(b),'--required-baselines',*primary,'--objective',objective,'--output',str(dest)];r=subprocess.run(argv,capture_output=True,text=True);dest.with_suffix('.log').write_text(r.stdout+r.stderr);gates.append(dict(version=version,k=k,objective=objective,argv=argv,exit_code=r.returncode,all_passed=load(dest)['all_passed'],baseline_sha256=sha(b)))
signed=load(R/'results_v2/signed_source_implementation_20261016/validation.json');assert signed['budget_verified'] and signed['direct_cec']==4 and signed['fresh_attempts']==16
save(O/'validation.json',dict(studies=checks,unique_fresh_attempts=28,unique_direct_cec=4,failures=24,baseline_gates=gates,production_qualified=False,production_promoted=False,baseline_algorithms_changed=False,unconnected=['MtM'],warning='All source-count reductions are pre-import representation estimates. Actual failed prefixes are not full imports or physical covers. No full20 PPA/repeat qualification, strict delay advantage or impossibility proof.'))
print(json.dumps(checks))
