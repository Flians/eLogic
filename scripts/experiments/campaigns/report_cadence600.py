"""Frozen cadence study; preserve full20 missing/failure scope and version distinctions."""
import argparse,csv,json,hashlib,subprocess,sys
from pathlib import Path
from report_online600 import load
ROOT=Path(__file__).resolve().parents[3]
def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--baselines',type=Path,required=True);a=p.parse_args();r=a.study.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 names=['v3_epfl_legacy','v3_epfl_control','v3_epfl_combined','v3_epfl_repeat1','v3_epfl_repeat2'];states={};data={};flat=[]
 for n in names:
  states[n],data[n]=load(r/n/'campaign.json')
  for b in states[n]['configuration']['designs']:
   row=dict(cohort=n,bench=b,status='missing')
   if b in data[n]:
    v=data[n][b];m=v.get('measured',{});row.update(status=v['status'],A=m.get('area'),D=m.get('delay'),T=v['mapping']['wall_secs'],CEC=m.get('cec'),search_finished=v.get('native',{}).get('search_finished'),raw_timeout=v['mapping']['timeout'])
   flat.append(row)
 ref=states['v3_epfl_combined'];norm=lambda c:{k:v for k,v in c.items()if k not in ['stream_small','incremental_delay','small_area_period']}
 assert all(norm(v['configuration'])==norm(ref['configuration'])and v['fingerprint']==ref['fingerprint']for v in states.values())
 assert ref['configuration']['small_area_period']==16 and states['v3_epfl_control']['configuration']['small_area_period']==1
 with(out/'all_scope.csv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=['cohort','bench','status','A','D','T','CEC','search_finished','raw_timeout']);w.writeheader();w.writerows(flat)
 base=json.loads(a.baselines.read_text());candidate={k:v for k,v in base.items()if k!='results'};candidate['results']=[];current=data['v3_epfl_combined']
 for b in ref['configuration']['designs']:
  row=dict(bench=b,method='cooperative600-cadence16-v3',status='missing',optimizer_succeeded=False,input_sha256=hashlib.sha256((ROOT/f'tools/mockturtle/experiments/benchmarks/{b}.aig').read_bytes()).hexdigest())
  if b in current:
   v=current[b];assert v['native']['library_sha256']==base['library_sha256'];row.update(status=v['status'],measured=v.get('measured'),optimizer_succeeded=v['native'].get('optimizer_succeeded',False),search_finished=v['native'].get('search_finished'),raw_mapping_timeout=v['mapping']['timeout'],mapping_wall_secs=v['mapping']['wall_secs'])
  candidate['results'].append(row)
 (out/'candidate.json').write_text(json.dumps(candidate,indent=2)+'\n');(out/'baselines.json').write_text(json.dumps(base,indent=2)+'\n');gates={}
 for obj in ['delay-area','area-delay','pareto']:
  cmd=[sys.executable,str(ROOT/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(out/'candidate.json'),'--baselines',str(out/'baselines.json'),'--required-baselines',*sorted({v['method']for v in base['results']}),'--objective',obj,'--output',str(out/(obj+'.json'))];q=subprocess.run(cmd,capture_output=True,text=True);(out/(obj+'.log')).write_text(q.stdout+q.stderr);assert q.returncode in [0,1];gates[obj]=json.loads((out/(obj+'.json')).read_text())['all_passed']
 repeats={}
 for b in ['adder','bar']:
  vals=[data[n][b]['measured']for n in ['v3_epfl_combined','v3_epfl_repeat1','v3_epfl_repeat2']if b in data[n]];repeats[b]=dict(runs=len(vals),identical_AD=len(vals)==3 and len({(v['area'],v['delay'])for v in vals})==1)
 _,older=load(r/'v2_epfl_control/campaign.json');hist=[]
 for b,v in older.items():
  if b in current:hist.append(dict(bench=b,reference_version='source_v2 legacy',candidate_version='source_v3 cadence16',reference=v['measured'],candidate=current[b].get('measured'),not_a_same_binary_ablation=True))
 (out/'version_comparison.json').write_text(json.dumps(hist,indent=2)+'\n')
 summary=dict(completed={n:len(v)for n,v in data.items()},unique_designs=len({b for v in data.values()for b in v}),independent_CEC_outputs=sum(v.get('measured',{}).get('cec')==1 for d in data.values()for v in d.values()),repeats=repeats,full_EPFL_complete=len(current)==20,large_design_repeats_complete=False,full_EPFL_ablations_complete=False,gates=gates,Rust_related_passed=268,ignored=6,filtered=16,main_merged=False,paper_updated=False,baseline_algorithms_changed=False,mapping_seconds=600,CEC_in_T=False,general_lazy_materialization=False,full_area_flow_resumable=False)
 (out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n');text=['# Cadence16: limited-scope validation','','Same source_v3 binary; P1/P16 differ only in ordinary-loop area recovery cadence. A scheduling period yields and retains pending work; it is not a trial limit or stage allowance. All mapping-related work shares600s; independentCEC excluded.','','|Design|Legacy batch A/D/T|P1 A/D/T|P16 A/D/T|','|---|---|---|---|']
 for b in ref['configuration']['designs']:
  if not any(b in data[n]for n in names[:3]):continue
  vals=[]
  for n in names[:3]:
   v=data[n].get(b);m=v.get('measured',{})if v else{};vals.append(f"{m['area']:.2f}/{m['delay']:.2f}/{v['mapping']['wall_secs']:.2f}"if m else(v['status']if v else'missing'))
  text.append('|'+b+'|'+'|'.join(vals)+'|')
 text += ['',f"Completed case outputs: {summary['independent_CEC_outputs']}; unique designs: {summary['unique_designs']}. Repeats: {repeats}.",'','Log2 prior-version reference is separate in version_comparison.json; no same-binary log2 ablation/repeat is claimed unless that current cohort actually contains the pair. Failed, timed-out and missing cases stay visible. Optimized predeadline receipt and search completion are distinct. No main/paper adoption. General lazy physical materialization, fully schedulable area-flow/shared transactions and full20 ablations remain unfinished.']
 (out/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
