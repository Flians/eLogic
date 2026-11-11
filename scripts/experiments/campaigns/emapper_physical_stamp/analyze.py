"""Read completed audited representative runs; preserve prefix and deadline scope."""
import argparse,json,pathlib
from observations import read
R=pathlib.Path(__file__).resolve().parents[4]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--audit',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();assert not a.output.exists()
 v=json.loads(a.audit.read_text());c=json.loads((R/'results_v2/emapper_physical_stamp_20261111/benchmark/campaign.json').read_text());rows=[]
 assert v['completed']>0 and not v['invalid_artifacts']
 for key,r in c['jobs'].items():
  events,tail=read(pathlib.Path(r['case'])/'observation.jsonl');ad=[x for x in events if x.get('details',{}).get('AD')];last=ad[-1]if ad else{};details=last.get('details',{})
  reached=next((x for x in ad if(x['details']['AD'][1],x['details']['AD'][0])<=(r['D'],r['A'])),None)if r['status']=='ok'else None
  rows.append(dict(key=key,design=r['design'],variant=r['variant'],rep=r['rep'],status=r['status'],A=r.get('A'),D=r.get('D'),T=r.get('T'),first_native_cover_seconds=(r.get('first_native_cover')or{}).get('mapping_elapsed_seconds'),internal_first_reach_delivered_quality_seconds=reached['elapsed_seconds']if reached else None,reach_scope='internal observer, not independent proof of an early deliverable receipt',prefix_elapsed=last.get('elapsed_seconds'),exclusive_prefix=last.get('exclusive_seconds'),capture_cost=details.get('selected_evidence',{}).get('cost'),telemetry=details.get('telemetry'),publication=details.get('publication'),materialization=details.get('cover_materialization'),pool=details.get('pool'),matched_cuts=details.get('matched_cuts'),batches=details.get('batches'),joint=details.get('joint_windows'),trial=details.get('trial_diagnostics'),tail_fragment=tail,counters_scope='last complete AD observation prefix, not same workload or final total'))
 pairs=[]
 for p in v['pairs']:
  q=dict(p);x,y=q['control_AD'],q['candidate_AD'];q.update(A_change_percent=100*(y[0]/x[0]-1),D_change_percent=100*(y[1]/x[1]-1),runtime_lower_in_pair=not q['both_deadline']and q['candidate_T']<q['control_T'],isolated_runtime_pair=True);pairs.append(q)
 result=dict(interruptions=v.get('interruptions',[]),completed=v['completed'],planned=v['planned'],scope='representative dec/sin pairs only; actual counts and missing repetitions in audit; not full EPFL; same native source/binary/profile and CPU6; 600s mapping, CEC excluded',rows=rows,pairs=pairs,invalid=v['invalid_artifacts'],first_and_final_CEC_all=all(x.get('first_independent_CEC')and x.get('CEC')==1 for x in v['rows']),default_promoted=False,full_EPFL_started=False,baseline_runs=0,GitHub_push=False,note='No concurrent diagnostic/compilation or baseline reruns; both-deadline T rounding is never a speedup; no configuration/repetition winner selection',proof_details=v['rows'],audit=str(a.audit))
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(pairs=pairs,all_CEC=result['first_and_final_CEC_all']),indent=2))
if __name__=='__main__':main()
