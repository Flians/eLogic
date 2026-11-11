"""Compare observed physical-cache work; never turn a prefix into a final total."""
import argparse,json,pathlib,sys,hashlib
from observations import read

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();assert not a.output.exists()
 c=json.loads(a.campaign.read_text());rows=[]
 for key,r in c['jobs'].items():
  frames,tail=read(pathlib.Path(r['case'])/'observation.jsonl');eligible=[f for f in frames if f.get('details',{}).get('selected_evidence',{}).get('cost')is not None]
  if not eligible:rows.append(dict(key=key,status=r['status'],missing_native_cache_observation=True));continue
  f=eligible[-1];d=f['details'];e=d['selected_evidence'];cost=e['cost'];mode=r['stamp']
  assert e['borrow_physical']==(mode=='borrow')and cost['physical_stamp_builds']==cost['physical_exports']+(cost['physical_hits']if mode=='construct'else 0)
  rows.append(dict(key=key,design=r['design'],rep=r['rep'],variant=r['variant'],status=r['status'],A=r.get('A'),D=r.get('D'),T=r.get('T'),CEC=r.get('CEC'),prefix_elapsed=f.get('elapsed_seconds'),physical_cache_cost=cost,exclusive_prefix_seconds=f.get('exclusive_seconds'),telemetry_active_publication_seconds=d.get('telemetry',{}).get('publication_secs'),matched_cuts=d.get('matched_cuts'),pool=d.get('pool'),tail=tail))
 by={r['key']:r for r in rows};pairs=[]
 work=['captures','source_visits','exported_proof_refs','generator_replays','physical_exports','physical_hits','object_serializations','source_hits']
 for n in ['dec','sin']:
  for rep in [0,1]:
   keys=[f'{n}_r{rep}_{v}'for v in ['control','borrow']]
   if all(k in by and 'physical_cache_cost'in by[k]for k in keys):
    x,y=[by[k]for k in keys];pairs.append(dict(design=n,rep=rep,same_observed_capture_work=all(x['physical_cache_cost'][k]==y['physical_cache_cost'][k]for k in work),constructor_calls=[x['physical_cache_cost']['physical_stamp_builds'],y['physical_cache_cost']['physical_stamp_builds']],T_percent=100*(y['T']/x['T']-1),A_percent=100*(y['A']/x['A']-1),D_percent=100*(y['D']/x['D']-1)))
 result=dict(completed=len(c['jobs']),planned=len(c['plan']),rows=rows,pairs=pairs,scope='last complete selected-evidence observation per run; not final counters; distinct search progress may change work. The publication observation scope also includes early no-op calls; telemetry publication_secs starts only after pending guards. Different scope times are not added. Constructor counts are not allocator bytes.',helper_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),default_promoted=False)
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(completed=result['completed'],pairs=pairs),indent=2))
if __name__=='__main__':main()
