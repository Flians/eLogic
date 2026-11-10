"""Audit completed real-mapping cases; first-cover CEC and original baselines."""
import argparse,csv,fcntl,json,pathlib,sys
R=pathlib.Path(__file__).resolve().parents[4];O=R/'results_v2/emapper_selected_witness_20261111/benchmark-v2'
sys.path.insert(0,str(R/'scripts/emapper'))
from observations import read as read_observations
from provenance import sha
from physical import measure_cover
from verify import verify
from selected_evidence import replay

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 locks=[]
 for p in [R/'results_v2/independent_components_20261019/global.lock',O/'runner.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 c=json.loads((O/'campaign.json').read_text());assert json.loads((O/'state.json').read_text())['active']is None
 invalid=[];checked=0;rows=[];pairs=[]
 basepath=R/'results_v2/full_baselines_20261025/baselines.json';b=json.loads(basepath.read_text());csvpath=basepath.parent/'baseline_snapshot.csv';times=list(csv.DictReader(csvpath.open()))
 labels={'abc_map':'ABC','mockturtle_emap_delay':'emap-delay','mockturtle_emap_area':'emap-area'}
 for key,r in c['jobs'].items():
  for n,h in r['artifacts'].items():
   p=pathlib.Path(r['case'])/n;checked+=1
   if sha(p)!=h:invalid.append(str(p))
  row={k:r.get(k)for k in ['key','design','rep','variant','status','A','D','T','CEC','search_timeout','search_finished','initial_AD','search_net_gain','first_native_cover','error','audit_error','exclusive_seconds','attempt_key','interrupted_attempt','interruption_audit','original_harness_status','original_harness_error']}
  observed=None
  obs=pathlib.Path(r['case'])/'observation.jsonl'
  if obs.exists():
   events,tail=read_observations(obs);row['observation_tail_fragment']=tail
   for ev in events:
    ad=ev.get('details',{}).get('AD')
    if ad and (observed is None or (ad[1],ad[0])<(observed[1],observed[0])):observed=ad
  row['best_observed_AD']=observed;row['observed_scope']='internal observations, not independently CEC verified unless also delivered/audited'
  row['delivery_worse_than_best_observed']=bool(r['status']=='ok'and observed and(r['D'],r['A'])>(observed[1],observed[0]))
  if r['status']=='ok':
   first=r['first_native_cover'];folder=a.output/'first-cover'/key
   measured=measure_cover(R/'emapper/libraries/asap7/asap7_full.genlib',pathlib.Path(first['netlist']),folder,R/'target/release/emapper_measure','yosys')
   cec=verify(R/'emapper/libraries/asap7/cells.v',pathlib.Path(c['subjects'][r['design']]['golden']),pathlib.Path(measured['netjson']),folder/'cec')
   proof=replay(pathlib.Path(first['manifest']),pathlib.Path(r['case'])/'selected-provenance/objects',pathlib.Path(measured['netjson']))
   row.update(first_public_AD=[measured['measured']['area'],measured['measured']['delay']],first_independent_CEC=cec,first_selected_evidence_valid=proof['valid'])
   assert cec and proof['valid']and row['first_public_AD']==r['initial_AD']
   compare=[]
   for old in b['results']:
    if old['bench']!=r['design']or old['method']not in labels:continue
    oldrow={k:old.get(k)for k in ['method','status','measured','source_report','error']}
    if old['status']=='ok'and old.get('measured',{}).get('cec')==1:
     assert old['input_sha256']==c['subjects'][r['design']]['golden_sha256']and b['library_sha256']==sha(R/'emapper/libraries/asap7/asap7_full.genlib')
     oldrow['DA']='better'if(r['D'],r['A'])<(old['measured']['delay'],old['measured']['area'])else('equal'if(r['D'],r['A'])==(old['measured']['delay'],old['measured']['area'])else'worse')
     matches=[x for x in times if x['bench']==r['design']and x['method']==labels[old['method']]and x['source']==old['source_report']]
     if len(matches)==1:oldrow['original_T']=float(matches[0]['T'])if matches[0]['T']else None
    compare.append(oldrow)
   row['original_baselines']=compare
  if r['status']=='ok':
   report=json.loads((pathlib.Path(r['case'])/'report.json').read_text());evidence=json.loads((pathlib.Path(report['incumbent']['manifest']).parent/'selected-evidence.json').read_text())
   counts=[len(x['proof_ids'])for x in evidence['selected']];row['delivered_selected_bindings']=len(counts);row['delivered_proof_references']=sum(counts)
   row['delivered_evidence_bytes']=(pathlib.Path(report['incumbent']['manifest']).parent/'selected-evidence.json').stat().st_size
   if r['variant']=='witness':assert all(n==1 for n in counts)
  rows.append(row)
 for n,rep in [('dec',0),('sin',0),('dec',1),('sin',1)]:
  rr=[c['jobs'].get(f'{n}_r{rep}_{v}')for v in ['control','witness']]
  if all(r and r['status']=='ok'for r in rr):
   x,y=rr;pairs.append(dict(design=n,rep=rep,control_AD=[x['A'],x['D']],candidate_AD=[y['A'],y['D']],control_T=x['T'],candidate_T=y['T'],T_change_percent=100*(y['T']/x['T']-1),DA='better'if(y['D'],y['A'])<(x['D'],x['A'])else('equal'if(y['D'],y['A'])==(x['D'],x['A'])else'worse'),both_deadline=x['search_timeout']and y['search_timeout'],same_output=x['output_sha256']==y['output_sha256']))
 interruptions=[]
 for item in c.get('interruptions',[]):
  path=pathlib.Path(item['audit_file']);assert sha(path)==item['sha256'];old=json.loads(path.read_text())
  for n,h in old['artifacts'].items():
   p=pathlib.Path(old['original_case'])/n;checked+=1
   if sha(p)!=h:invalid.append(str(p))
  interruptions.append({k:old.get(k)for k in ['original_case','original_key','reason','status','last_complete_observation_elapsed','last_observed_AD','CEC','not_a_full_budget_result','retry_key','old_attempt_retained','not_a_search_continuation']})
 result=dict(completed=len(c['jobs']),planned=len(c['plan']),interruptions=interruptions,scope='full native mapping, generation included, no replay/no external seed; once-resyn2/ASAP7,600s mapping only, CEC excluded',rows=rows,pairs=pairs,missing=[p['key']for p in c['plan']if p['key']not in c['jobs']],verified_artifact_references=checked,invalid_artifacts=invalid,baseline_runs=0,GitHub_push=False,default_promoted=False,receipts={str(p.relative_to(R)):sha(p)for p in [O/'campaign.json',basepath,csvpath]})
 (a.output/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['rows','missing']},indent=2));assert not invalid
if __name__=='__main__':main()
