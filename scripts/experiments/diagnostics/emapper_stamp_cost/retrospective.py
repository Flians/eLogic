"""Replay audited cost receipts; counts do not stand in for missing CPU evidence."""
import argparse,json,pathlib,hashlib
R=pathlib.Path(__file__).resolve().parents[4]
P=R/'results_v2/emapper_physical_stamp_20261111'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--require-no-regression',action='store_true');a=ap.parse_args();assert not a.output.exists()
 c=json.loads((P/'benchmark/campaign.json').read_text());x=json.loads((P/'prefix-008.json').read_text());rows={r['key']:r for r in x['rows']};pairs=[]
 for n in ['dec','sin']:
  for rep in [0,1]:
   ka,kb=[f'{n}_r{rep}_{v}'for v in ['control','borrow']];u,v=c['jobs'][ka],c['jobs'][kb];p,q=rows[ka],rows[kb]
   ep,eq=[r.get('exclusive_prefix_seconds')or{}for r in [p,q]];names=set(ep)|set(eq)
   pu,pv=[r['physical_cache_cost']for r in [p,q]]
   pairs.append(dict(design=n,rep=rep,T_control=u['T'],T_borrow=v['T'],T_delta=v['T']-u['T'],T_percent=100*(v['T']/u['T']-1),AD_control=[u['A'],u['D']],AD_borrow=[v['A'],v['D']],same_final_netlist=u['output_sha256']==v['output_sha256'],matched_cuts=[p['matched_cuts'],q['matched_cuts']],pool=[p['pool'],q['pool']],capture_work_equal=all(pu[k]==pv[k]for k in ['captures','source_visits','generator_replays','physical_exports','physical_hits','object_serializations']),capture_scope_delta=eq.get('selected_evidence_capture',0)-ep.get('selected_evidence_capture',0),scope_delta_desc=sorted([(k,eq.get(k,0)-ep.get(k,0))for k in names],key=lambda v:v[1],reverse=True),prefix_elapsed=[p['prefix_elapsed'],q['prefix_elapsed']],whole_map_work_equal_proven=False))
 result=dict(pairs=pairs,old_native_cpu_or_runqueue_samples_present=False,load_attribution='unidentifiable retrospectively: affinity and wall time are not CPU/IO/scheduler counters',scope='historical observed prefixes, not final exclusive totals; requires fresh same-work/resource observations for causal attribution',sources={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest()for f in [P/'benchmark/campaign.json',P/'prefix-008.json',P/'final-validation.json']})
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(pairs,indent=2))
 if a.require_no_regression:assert all(r['T_delta']<=0 for r in pairs if r['design']=='dec'),'observed full mapping runtime regression remains in both dec pairs'
if __name__=='__main__':main()
