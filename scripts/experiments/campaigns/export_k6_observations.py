"""Export verified common physical A/D and mapping-only clock; never select winners."""
import argparse,csv,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--scope',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 scope=json.loads(a.scope.read_text());campaign=json.loads(a.checkpoint.read_text());jobs={j['name']:j for j in campaign['jobs']};cache={};rows=[]
 for value in scope['results']:
  name=value['source_job'];job=jobs[name];audit_path=Path(job['argv'][job['argv'].index('--audit')+1]);assert sha(audit_path)==job['sha256'][str(audit_path)]
  if name not in cache:cache[name]=json.loads(audit_path.read_text())
  audit=cache[name];state=audit['states'].get(value['bench']+'/'+value['method']);charged=state.get('mapping_wall_secs') if state else None
  successful=value['status']=='ok';measured=value.get('measured',{});assert not successful or measured['cec']==1
  upper=value.get('mapping_budget_upper_bound_secs');t=upper if upper is not None else charged
  if successful and t is not None:assert t<=300 and not (state and state['exhausted'])
  objective=next((x for x in ['area-delay','delay-area'] if x in name),'native-final' if '_emorphic_' in name else 'area-delay' if '_area_' in value['method'] else 'delay-area')
  rows.append(dict(bench=value['bench'],method=value['method'],objective=objective,status=value['status'],A=measured.get('area','') if successful else '',D=measured.get('delay','') if successful else '',T=t if successful and t is not None else '',T_kind='upper_bound_enclosing_LUT_conversion' if upper is not None else 'mapping_only_active_interval_wall' if charged is not None else 'unavailable',mapping_charged_secs=charged if charged is not None else '',error=value.get('error',''),source_job=name,observation_report=value.get('observation_report',''),audit=str(audit_path),audit_sha256=sha(audit_path),verification_history=json.dumps(value.get('verification_history',[]),separators=(',',':'))))
 assert len(rows)==260
 a.output.mkdir(exist_ok=False)
 with (a.output/'baseline_A_D_T.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 metadata=dict(observations=len(rows),verified=sum(r['status']=='ok' for r in rows),failed=sum(r['status']!='ok' for r in rows),verified_T_unavailable=sum(r['status']=='ok' and r['T']=='' for r in rows),mapping_limit_seconds=300,policy_excludes_CEC_conversion_resyn2_reporting=True,exact_T_excludes_CEC_conversion_resyn2_reporting=True,EqMap_T_is_conservative_upper_including_enclosing_conversion_not_exact=True,unseeded_native_SA_winner_preserved=True,per_design_winner_selection=False,baseline_algorithms_changed=False,scope=str(a.scope),scope_sha256=sha(a.scope),checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),exporter_sha256=sha(Path(__file__)))
 (a.output/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata))
if __name__=='__main__':main()
