"""Audit actual wide-library bounds and runtime phase fractions, preserving failed/native-source-only cases."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();state=json.loads(a.campaign.read_text());a.output.mkdir(exist_ok=False);rows=[];inputs={};libs=set();trace_counts={};verification={}
 for job in state['jobs']:
  assert all(sha(p)==h for p,h in job['sha256'].items()),job['name']
  if not job.get('report'):rows.append({'job':job['name'],'status':'unmeasured failure'});continue
  report=json.loads(Path(job['report']).read_text());libs.add(report['library_sha256']);v=report['results'][0];p=v.get('resource_policy',{});s=v.get('scalability',{});n=s.get('optimizer_native',{});optimized=s.get('selected_attempt') is not None
  assert v.get('mapping_wall_secs',0)<=300
  if p:
   assert p['search_ratio']=='1/2' and p['pool_limit']==100000 and p['pool_extra']==0 and not p['refund_on_rebuild']
   assert p['node_extra']==(p['base_enode_upper']+1)//2 and p['node_limit']==p['base_enode_upper']+p['node_extra']
   assert p['global_retained_candidate_limit']==p['base_enode_upper']+100000
   assert p['optimizer_guard_secs']==60 and p['frontend_process_guard_secs']==300
   assert s['cumulative_accounted_enode_upper']<=p['node_limit']
   assert s['source_plus_optimizer_retained_upper']<=p['global_retained_candidate_limit']
  if optimized:
   assert n['instances']<=100000 and n['max_cuts_per_class']<=16 and n['heuristic_secs']<=2
   assert n['cumulative_imported_enodes']<=p['node_limit'] and n['cumulative_admitted_enodes']<=p['node_limit'] and n['enodes']<=p['node_limit']
  assert not s or s['optimizer_address_space_limit_bytes']==16*1024**3
  folder=Path(job['report']).parent;actual=[];outcomes=[];wide=[];reuse=[]
  import_plans=[]
  for f in folder.rglob('*.import-plan.json'):
   d=json.loads(f.read_text());assert d['forced_compact'] and d['node_limit']==p['node_limit'];import_plans.append(d.get('forced_compact'))
  for f in folder.rglob('*.helper-allocation.json'):
   d=json.loads(f.read_text());assert not d['enabled'] and d['global_node_allowance']==p['node_limit']
  for f in folder.rglob('*.wide-generation.json'):
   d=json.loads(f.read_text());assert d['new_enodes']==0 and d['work_used']<=d['global_work_allowance']==1600000 and d['global_new_candidates']<=100000;wide.append({k:d.get(k)for k in ['work_used','global_new_candidates','seed_index_keys','new_enodes']})
  for f in folder.rglob('*.wire-pass.json'):
   d=json.loads(f.read_text());assert d['candidates_dropped']==0;reuse.append(d['reused_streamed_pool'])
  for f in folder.rglob('attempt*.trace.json'):
   for e in json.loads(f.read_text()).get('events',[]):
    if e['stage']=='fixed_area_window':
     d=e['data'];expected=[.30,.65][job['profile']];assert abs(d['fraction_of_remaining_search']-expected)<1e-9 and not d['global_deadline_extended'];assert e['seconds']+d['allowance_seconds']<=1.76;actual.append(expected)
    if e['stage']=='required_area_dag_outcome':outcomes.append(e['data']);reason=e['data']['reason'];trace_counts[reason]=trace_counts.get(reason,0)+1
  mode=v.get('verification_mode','unknown');verification[mode]=verification.get(mode,0)+1
  inputs.setdefault(v['bench'],set()).add(v['input_sha256'])
  rows.append(dict(job=job['name'],bench=v['bench'],profile=job['profile'],status=v['status'],native_optimized=optimized,source_only=not optimized,cec=v.get('measured',{}).get('cec'),A=v.get('measured',{}).get('area'),D=v.get('measured',{}).get('delay'),T=v.get('mapping_wall_secs'),actual_area_flow_fraction=actual,area_stage_not_reached=not actual,area_outcomes=outcomes,wide_matching=wide,pool_reuse=reuse,forced_compact_imports=import_plans,admission_charges=[{k:at.get(k)for k in ['index','charge_is_measured','charged_admission_upper','node_allowance']}for at in s.get('attempts',[])]))
 assert len(libs)<=1 and all(len(s)==1 for s in inputs.values())
 (a.output/'validation.json').write_text(json.dumps(dict(campaign=str(a.campaign.resolve()),campaign_sha256=sha(a.campaign),jobs=len(rows),expected_jobs=state['expected_jobs'],complete=len(rows)==state['expected_jobs'],native_optimized=sum(r.get('native_optimized',False)for r in rows),source_only=sum(r.get('source_only',False)for r in rows),final_cec=sum(r.get('cec')==1 for r in rows),actual_library_hashes=sorted(libs),verification_modes=verification,area_outcome_counts=trace_counts,all_recorded_bounds_pass=True,ordinary_cut_bank_limit=16,wide_implicit_boundaries_not_claimed_limited_to16=True,wide_matches_share_global_retained_pool_and_time=True,rows=rows),indent=2))
 print('audit',len(rows),'jobs, native optimized',sum(r.get('native_optimized',False)for r in rows),'area outcomes',trace_counts)
if __name__=='__main__':main()
