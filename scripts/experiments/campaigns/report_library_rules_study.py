"""Read-only, complete-scope one-way library-rule ablation report. No mapper/baseline mutation."""
import argparse,collections,csv,hashlib,json
from pathlib import Path
from library_rules_paths import analyze as path_analysis

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def cmp(x,y):
 for k in ['D','A']:
  t=1e-6*max(1,abs(y[k]))
  if x[k]<y[k]-t:return 'better'
  if x[k]>y[k]+t:return 'worse'
 return 'equal'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=False)
 state=json.loads(a.campaign.read_text());rows=[];coverage=[];hashes={str(a.campaign.resolve()):sha(a.campaign),str(Path(__file__).resolve()):sha(__file__),str(Path(__file__).with_name('library_rules_paths.py').resolve()):sha(Path(__file__).with_name('library_rules_paths.py'))};errors=[];libraries=set();inputs=collections.defaultdict(set)
 for job in state['jobs']:
  for path,digest in job['sha256'].items():
   if sha(path)!=digest:errors.append('artifact drift '+path)
  if not job.get('report'):rows.append(dict(job=job['name'],bench=job['bench'],profile=job['profile'],status='unmeasured_failure'));continue
  report=json.loads(Path(job['report']).read_text());libraries.add(report['library_sha256']);v=report['results'][0];inputs[v['bench']].add(v['input_sha256']);policy=v.get('resource_policy',{});search=v.get('scalability',{});n=search.get('optimizer_native',{});m=v.get('measured',{});root=Path(job['report']).parent
  assert v.get('mapping_wall_secs',0)<=300
  if policy:
   assert policy['search_ratio']=='1/2' and policy['pool_limit']==100000 and policy['node_extra']==(policy['base_enode_upper']+1)//2
   assert policy['node_limit']==policy['base_enode_upper']+policy['node_extra'] and not policy['refund_on_rebuild']
   assert search['cumulative_accounted_enode_upper']<=policy['node_limit']
   assert search['source_plus_optimizer_retained_upper']<=policy['global_retained_candidate_limit']
  if search.get('optimizer_succeeded'):
   assert n['instances']<=100000 and n['max_cuts_per_class']<=16 and n['heuristic_secs']<=2
   assert n['cumulative_admitted_enodes']-n['cumulative_imported_enodes']<=policy['node_extra']
  rules=[]
  for p in root.rglob('*.library-rules-stage.json'):
   x=json.loads(p.read_text());assert x['logical_rounds']+x['rounds']<=16
   assert x['post_import_used']<=x['post_import_allowance']==policy['node_extra']
   assert x['work_used']<=x['work_limit']==(800000 if job['profile']==2 else 1600000);rules.append(x)
  for p in root.rglob('*.library-rules.json'):coverage.append(dict(job=job['name'],kind='rule_generation_before_common_retention',data=json.loads(p.read_text())))
  # Candidate and selected-path descriptions are per graph; dense class IDs are not joined across profiles.
  if search.get('selected_attempt')is not None:
   attempt=search['attempts'][search['selected_attempt']];diag=Path(attempt['candidate_path'])
   if diag.exists():
    x=json.loads(diag.read_text());selected={r['instance']:r for r in x['selected_instances']};crit=collections.Counter();producer=collections.Counter()
    for s in x['critical_signals']:
     if s['selected']is not None:crit[selected[s['selected'][0]]['cell']]+=1
     for p in s['producers']:producer[p['cell']]+=1
    coverage.append(dict(job=job['name'],kind='critical_signal_bindings_in_this_selected_cover',selected=dict(crit),retained_producers=dict(producer),critical_signal_count=len(x['critical_signals']),not_cross_profile_boundary_identity=True))
  rows.append(dict(job=job['name'],bench=job['bench'],profile=job['profile'],status=v['status'],cec=m.get('cec'),A=m.get('area'),D=m.get('delay'),T=v.get('mapping_wall_secs'),native_completed=bool(search.get('optimizer_succeeded')),native_selected=search.get('selected_attempt')is not None,selected=search.get('selected'),rules=rules,report=job['report']))
 assert len(libraries)<=1 and all(len(v)==1 for v in inputs.values()) and not errors
 groups=collections.defaultdict(dict)
 for r in rows:
  repetition=r['job'].split('_',1)[0];groups[(repetition,r['bench'])][r['profile']]=r
 pairs=[]
 for (rep,b),g in groups.items():
  if 0 not in g:continue
  for profile in [1,2,3]:
   if profile not in g:continue
   x,y=g[profile],g[0];valid=all(z.get('status')=='ok' and z.get('cec')==1 and z.get('A')is not None and z.get('D')is not None for z in [x,y]);pairs.append(dict(repetition=rep,bench=b,profile=profile,valid=valid,DA=cmp(x,y)if valid else 'invalid',control={k:y.get(k)for k in ['A','D','T','native_selected']},candidate={k:x.get(k)for k in ['A','D','T','native_selected']}))
 summary={str(p):dict(collections.Counter(r['DA']for r in pairs if r['profile']==p))for p in sorted({r["profile"]for r in pairs})}
 validation=dict(jobs=len(rows),expected=state['expected_jobs'],complete=len(rows)==state['expected_jobs'],CEC=sum(r.get('cec')==1 for r in rows),native_completed=sum(r.get('native_completed',False)for r in rows),native_selected=sum(r.get('native_selected',False)for r in rows),summary=summary,rows=rows,pairs=pairs,hashes=hashes,all_recorded_budgets_pass=True,scope=state['configuration'],baseline_algorithms_changed=False,main_integrated=False,critical_dense_ids_not_joined_across_profiles=True)
 (a.output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n');(a.output/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
 with (a.output/'A_D_T.csv').open('w')as f:
  keys=['job','bench','profile','status','cec','A','D','T','native_completed','native_selected','selected'];w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
 lines=['# One-way library-rule ablation',f"Attempted {len(rows)}/{state['expected_jobs']}; final CEC {validation['CEC']}; native completed {validation['native_completed']}; selected {validation['native_selected']}.",f'Paired DA summary: {summary}', '', 'Profiles are fixed across designs; no per-design winner selection. Source-only fallbacks are not native optimization successes. Failed and missing slots remain in the scope. Critical-path candidate descriptions do not prove cross-profile AIG boundary correspondence.', '', '|Design|Profile|A|D|T (s)|Native selected|Status|','|---|---:|---:|---:|---:|---|---|']
 for r in rows:lines.append('|'+ '|'.join(str(r.get(k,''))for k in ['bench','profile','A','D','T','native_selected','status'])+'|')
 (a.output/'report.md').write_text('\n'.join(lines)+'\n');print({k:validation[k]for k in ['jobs','expected','complete','CEC','native_completed','native_selected','summary']})
if __name__=='__main__':main()
