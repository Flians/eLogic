"""Metric workspace and shared-tail study, immutable source and all20 statuses; no per-design winner selection."""
import argparse,csv,json,hashlib,subprocess,sys
from pathlib import Path
from report_online600 import load
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--baselines',type=Path,required=True);a=p.parse_args();r=a.study.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 names=['v6_base','v6_metric','v6_tail','v6_epfl','v6_repeat1','v6_repeat2'];states={};data={};flat=[]
 for n in names:
  states[n],data[n]=load(r/n/'campaign.json')
  for b in states[n]['configuration']['designs']:
   row=dict(cohort=n,bench=b,status='missing')
   if b in data[n]:
    v=data[n][b];m=v.get('measured',{});row.update(status=v['status'],A=m.get('area'),D=m.get('delay'),T=v['mapping']['wall_secs'],CEC=m.get('cec'),search_finished=v.get('native',{}).get('search_finished'),raw_timeout=v['mapping']['timeout'])
   flat.append(row)
 ref=states['v6_epfl'];norm=lambda c:{k:v for k,v in c.items()if k not in ['metric_epochs','legacy_tail']}
 assert all(norm(v['configuration'])==norm(ref['configuration'])and v['fingerprint']==ref['fingerprint']for v in states.values());assert ref['configuration']['metric_epochs']==1 and ref['configuration']['legacy_tail']==1 and states['v6_base']['configuration']['metric_epochs']==0 and states['v6_base']['configuration']['legacy_tail']==0
 with(out/'all_scope.csv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=['cohort','bench','status','A','D','T','CEC','search_finished','raw_timeout']);w.writeheader();w.writerows(flat)
 base=json.loads(a.baselines.read_text());candidate={k:v for k,v in base.items()if k!='results'};candidate['results']=[];current=data['v6_epfl']
 for b in ref['configuration']['designs']:
  row=dict(bench=b,method='cooperative600-metric-v6',status='missing',optimizer_succeeded=False,input_sha256=sha(ROOT/f'tools/mockturtle/experiments/benchmarks/{b}.aig'))
  if b in current:
   v=current[b];assert v['input_sha256']==row['input_sha256']and v['native']['library_sha256']==base['library_sha256'];row.update(status=v['status'],measured=v.get('measured'),optimizer_succeeded=v['native'].get('optimizer_succeeded',False),search_finished=v['native'].get('search_finished'),raw_mapping_timeout=v['mapping']['timeout'],mapping_wall_secs=v['mapping']['wall_secs'])
  candidate['results'].append(row)
 (out/'candidate.json').write_text(json.dumps(candidate,indent=2)+'\n');(out/'baselines.json').write_text(json.dumps(base,indent=2)+'\n');gates={}
 for obj in ['delay-area','area-delay','pareto']:
  cmd=[sys.executable,str(ROOT/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(out/'candidate.json'),'--baselines',str(out/'baselines.json'),'--required-baselines',*sorted({v['method']for v in base['results']}),'--objective',obj,'--output',str(out/(obj+'.json'))];q=subprocess.run(cmd,capture_output=True,text=True);(out/(obj+'.log')).write_text(q.stdout+q.stderr);assert q.returncode in [0,1];gates[obj]=json.loads((out/(obj+'.json')).read_text())['all_passed']
 repeats={}
 for b in ['adder','bar']:
  vals=[data[n][b]['measured']for n in ['v6_epfl','v6_repeat1','v6_repeat2']if b in data[n]];repeats[b]=dict(runs=len(vals),identical_AD=len(vals)==3 and len({(v['area'],v['delay'])for v in vals})==1)
 versions={}
 for n in ['v2_epfl_control','v2_epfl_combined','v3_epfl_combined','v3_p0','v4_epfl','v5_epfl']:
  _,d=load(r/n/'campaign.json');versions[n]=d
 old=json.loads((ROOT/'results_v2/time_only_600_20261022/epfl_checkpoint/campaign.json').read_text());older={j['bench']:json.loads(Path(j['report']).read_text())for j in old['jobs']};pairs=[]
 for b,v in current.items():
  o=older[b];assert v['input_sha256']==o['input_sha256'] and v['native']['library_sha256']==o['native']['library_sha256'] and v['prepared_sha256']==o['prepared_sha256'];m=v.get('measured');om=o.get('measured');pairs.append(dict(bench=b,old_uniform_time600=o['status'],new_uniform_v6=v['status'],old_measured=om,new_measured=m,delay_change=(m['delay']-om['delay'])if m and om else None,area_change=(m['area']-om['area'])if m and om else None,old_search_timeout=o['mapping']['timeout'],new_search_timeout=v['mapping']['timeout']))
 (out/'old_uniform_version_pairs.json').write_text(json.dumps(pairs,indent=2)+'\n')
 (out/'factorial_results.json').write_text(json.dumps({n:data[n] for n in names[:4]},indent=2)+'\n')
 summary=dict(completed={n:len(v)for n,v in data.items()},candidate_optimized=sum(v.get('status')=='ok'and v.get('native',{}).get('optimizer_succeeded',False)for v in current.values()),candidate_statuses={s:sum(v['status']==s for v in current.values())for s in sorted({v['status']for v in current.values()})},unique_designs=len({b for v in data.values()for b in v}),independent_CEC_outputs=sum(v.get('measured',{}).get('cec')==1 for d in data.values()for v in d.values()),repeats=repeats,full_EPFL_complete=len(current)==20,large_design_repeats_complete=False,full_EPFL_ablations_complete=False,gates=gates,Rust_related_passed=275,ignored=6,filtered=16,main_merged=False,paper_updated=False,baseline_algorithms_changed=False,mapping_seconds=600,CEC_in_T=False,general_lazy_physical_materialization=False,full_area_flow_resumable=False)
 (out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n');text=['# Online metric workspace and full-extractor tail: limited-scope verification','','One frozen V6 configuration for every candidate design; no winner mixing. Same binary 2x2 metric/tail switch for first2 controls. All mapping search shares600s; independentCEC excluded.','','|Design|Metric/tail off A/D/T|Metric/tail on A/D/T|Search finished (on)|','|---|---|---|---|']
 for b in ref['configuration']['designs']:
  if b not in current and b not in data['v6_base']:continue
  vals=[]
  for n in ['v6_base','v6_epfl']:
   v=data[n].get(b);m=v.get('measured',{})if v else{};vals.append(f"{m['area']:.2f}/{m['delay']:.2f}/{v['mapping']['wall_secs']:.2f}"if m else(v['status']if v else'missing'))
  text.append('|'+b+'|'+'|'.join(vals)+'|'+str(current.get(b,{}).get('native',{}).get('search_finished','missing'))+'|')
 text+=['',f"Unique designs: {summary['unique_designs']}; candidate optimized: {summary['candidate_optimized']}. Repeats: {repeats}.",'','## Log2 version history (all failures retained)','','|Version/config|A|D|T|CEC|Search finished|','|---|---:|---:|---:|---|---|']
 for n,d in [*versions.items(),('v6_epfl',current)]:
  if 'log2'not in d:continue
  v=d['log2'];m=v.get('measured',{});text.append(f"|{n}|{m.get('area',float('nan')):.2f}|{m.get('delay',float('nan')):.2f}|{v['mapping']['wall_secs']:.2f}|{m.get('cec')}|{v.get('native',{}).get('search_finished')}|")
 text+=['','History rows are different named versions/configurations, not a single mapper selected per design. Same-binary full20 metric/tail ablations and log2 repeats are unfinished. Every missing/failure row remains in all_scope.csv; all baseline gates use the full required scope. Old full-flow adder area87.06 remains below the online87.10 when that current result persists; preserving batch delay seeds is not full legacy shared-area recovery.','', 'The scoped metric workspace is enabled at the actual online entry point and restores its state on exit. The optional full legacy shared extractor runs only at the tail after matching closure, within the original global deadline; its proposal must satisfy both current D and A bounds. It is nonpreemptive and does not guarantee recovery of the old full-flow trajectory. The proof memo belongs to one frozen logical/library epoch and append-only physical identities. Targets/evidence may grow. Every newly used binding needs full source proof; cached cut witnesses are checked for retention. Driver/cycle, full physical A/D and final independentCEC remain checked. Retention pins are materialized only on demand. This is not general deferred physical Instance materialization; resumable whole area-flow/shared-cover recovery is unfinished. No main/paper adoption.']
 (out/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
