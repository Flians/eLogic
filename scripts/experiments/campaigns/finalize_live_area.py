#!/usr/bin/env python3
"""Reconcile complete fixed-version physical-area and source-budget studies."""
import argparse,copy,hashlib,json,subprocess,sys
from audit_live_area_commands import audit
from pathlib import Path
R=Path(__file__).resolve().parents[3];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();O=a.output;O.mkdir(exist_ok=False);load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();save=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n')
inputs={'pilot':('live_shared_area_pilot_20261016','check_live_shared_area.py'),'allocation':('live_area_allocation_20261016','check_live_area_allocation.py'),'source':('zero_cost_source_pilot_20261016','check_zero_source.py'),'full_source':('full_source_pilot_20261016','check_full_source.py'),'full':('live_area_full_20261016','check_live_area_full.py'),'repeats':('live_area_allocation_repeats_20261016','check_live_area_allocation_repeats.py')}
studies={};native_rows={};pairs={};artifacts={}
for label,(folder,checker) in inputs.items():
 cp=R/'results_v2'/folder/'campaign.json';dest=O/(label+'_validation.json');argv=[sys.executable,str(Path(__file__).with_name(checker)),'--campaign',str(cp),'--output',str(dest)];r=subprocess.run(argv,capture_output=True,text=True);dest.with_suffix('.log').write_text(r.stdout+r.stderr);assert r.returncode==0,(argv,r.stdout,r.stderr)
 check=load(dest);c=load(cp);data=[]
 for j in c['jobs']:
  report=load(Path(j['report']));row=report['results'][0];proof=audit(j,c);artifacts[j['name']+'_'+label]=proof;data.append(dict(k=j['k'],bench=j['bench'],profile=j['profile'],name=j['name'],row=row,report=j['report'],report_sha256=j['report_sha256'],audit=j['audit'],audit_sha256=j['audit_sha256'],pool_sha256=j.get('pool_sha256')))
 native_rows[label]=data;save(O/(label+'_rows.json'),data);studies[label]=dict(fresh_attempts=len(data),direct_cec=sum(d['row']['status']=='ok' for d in data),failures=sum(d['row']['status']!='ok' for d in data),campaign=str(cp),campaign_sha256=sha(cp),checks=str(dest),checks_sha256=sha(dest))
 if label in ['pilot','full','full_source']:
  pp=[]
  for k in [4,3]:
   for b in c['designs']:
    d={i['profile']:i for i in data if i['k']==k and i['bench']==b};x=d['legacy_control'];y=d['protected_area'];item=dict(k=k,bench=b,control=x['row']['status'],candidate=y['row']['status'])
    if x['row']['status']==y['row']['status']=='ok':
     m=x['row']['measured'];n=y['row']['measured'];item.update(control_area=m['area'],candidate_area=n['area'],control_delay=m['delay'],candidate_delay=n['delay'],area_delta=n['area']-m['area'],delay_delta=n['delay']-m['delay'],retained_pools_identical=x['pool_sha256']==y['pool_sha256'])
    pp.append(item)
  pairs[label]=pp;save(O/(label+'_pairs.json'),pp)
# Balanced three-profile sqrt observations: screen plus two new alternating orders.
observations=[]
for label in ['allocation','repeats']:
 for item in native_rows[label]:
  item=copy.deepcopy(item);item['observation']='screen' if label=='allocation' else item['name'].rsplit('_',1)[1];observations.append(item)
assert len(observations)==18;save(O/'sqrt_repeat_observations.json',observations)
for k in [4,3]:
 for obs in ['screen','r0','r1']:
  d={i['profile']:i for i in observations if i['k']==k and i['observation']==obs};assert set(d)=={'legacy_control','allocation_only','protected_area'}
  valid=[i for i in d.values() if i['row']['status']=='ok'];assert len({i['pool_sha256'] for i in valid})<=1,'same-source allocation factor changed retained pool'
# Public baseline scope: full fixed candidate only, never best-case profiles.
old=R/'results_v2/guarded_joint_evidence_20261016';audit=old/'baseline_integrity.json';oldcheck=load(audit);assert oldcheck['checked_files']==13403 and not oldcheck['drift'];save(O/'baseline_audit_reference.json',dict(path=str(audit),sha256=sha(audit),scope='Prior13403-file zero-drift audit referenced bySHA, not a new audit/rerun; no original baseline algorithm/model/parameter/helper edits in this work.'))
G=O/'baseline_scope';G.mkdir();gates=[];required=['abc_map_area_resyn2','abc_map_delay_resyn2','mockturtle_emap_area_resyn2','mockturtle_emap_delay_resyn2','eqmap_lut_amap','esyn_frozen_model_amap_bounded_dag','emorphic_final_sa']
for k in [3,4]:
 base=old/f'baseline_scope/baselines_{k}in.json';d=copy.deepcopy(load(base));d['results']=[dict(copy.deepcopy(i['row']),method='egraph_live_area_proto') for i in native_rows['full'] if i['k']==k and i['profile']=='protected_area'];assert len(d['results'])==20;d['objective']='delay-area';d['area_delay_check_is_posthoc']=True;cp=G/f'candidate_{k}in.json';save(cp,d)
 for objective in ['delay-area','area-delay','pareto']:
  dest=G/f'{k}in_{objective}.json';argv=[sys.executable,str(R/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(cp),'--baselines',str(base),'--required-baselines',*required,'--objective',objective,'--output',str(dest)];r=subprocess.run(argv,capture_output=True,text=True);dest.with_suffix('.log').write_text(r.stdout+r.stderr);gates.append(dict(k=k,objective=objective,argv=argv,exit_code=r.returncode,all_passed=load(dest)['all_passed'],baseline_source=str(base),baseline_sha256=sha(base)))
# Main comparison separate from the same-binary control contrast.
main=load(R/'results_v2/scored_main_full_report_20261015/all_rows.json');reference={(i['k'],i['bench']):i['row'] for i in main if i['profile']=='protected_area'};main_pairs=[]
for item in native_rows['full']:
 if item['profile']!='protected_area':continue
 oldrow=reference[(item['k'],item['bench'])];new=item['row'];d=dict(k=item['k'],bench=item['bench'],main_status=oldrow['status'],candidate_status=new['status'])
 if oldrow['status']==new['status']=='ok':d.update(area_delta=new['measured']['area']-oldrow['measured']['area'],delay_delta=new['measured']['delay']-oldrow['measured']['delay'])
 main_pairs.append(d)
save(O/'canonical_main_pairs.json',main_pairs)
def counts(data,field):
 values=[d[field] for d in data if field in d];return dict(better=sum(x<-1e-6 for x in values),equal=sum(abs(x)<=1e-6 for x in values),worse=sum(x>1e-6 for x in values))
qualification=dict(studies=studies,unique_fresh_attempts=sum(s['fresh_attempts'] for s in studies.values()),unique_direct_cec=sum(s['direct_cec'] for s in studies.values()),unique_failures=sum(s['failures'] for s in studies.values()),full_pairs=dict(delay=counts(pairs['full'],'delay_delta'),area=counts(pairs['full'],'area_delta')),main_common_legal=dict(delay=counts(main_pairs,'delay_delta'),area=counts(main_pairs,'area_delta')),main_equal_delay_area_regressions=[d for d in main_pairs if 'delay_delta' in d and abs(d['delay_delta'])<=1e-6 and d['area_delta']>1e-6],baseline_gates=[g['all_passed'] for g in gates],production_qualified=False,production_promoted=False,baseline_algorithms_changed=False,unconnected=['MtM'],reason='All unsuccessful observations retained; fixed full-scope and small target repetitions do not establish universal old-final-incumbent protection or full-baseline dominance. No main promotion without verified coverage/area/delay qualification. Generated transient candidates are not a retained-memory cap; node units depend on explicitly recorded representation.')
save(O/'independent_command_audit.json',artifacts);save(O/'qualification.json',qualification);save(G/'validation.json',dict(gates=gates,original_status_slots=520,baseline_algorithms_changed=False,unconnected=['MtM']));print(json.dumps({k:qualification[k] for k in ['unique_fresh_attempts','unique_direct_cec','unique_failures','full_pairs','main_common_legal','production_promoted']}))
