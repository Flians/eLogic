"""Read-only cooperative ordinary matching/dirty-delay evidence; no missing case removal."""
import argparse,csv,json,sys,subprocess,hashlib
from pathlib import Path
from report_online600 import load
ROOT=Path(__file__).resolve().parents[3]
def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--baselines',type=Path,default=ROOT/'results_v2/streaming_mapping_20261022/baseline39_scope/baselines.json');a=p.parse_args();r=a.study.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 names=['v2_epfl_control','v2_epfl_small_only','v2_epfl_delay_only','v2_epfl_combined','v2_epfl_repeat1','v2_epfl_repeat2','v2_epfl_batch_off'];states={};data={};rows=[]
 for name in names:
  path=r/name/'campaign.json'
  if not path.exists():continue
  states[name],data[name]=load(path)
  for bench in states[name]['configuration']['designs']:
   row=dict(cohort=name,bench=bench,status='missing')
   if bench in data[name]:
    v=data[name][bench];m=v.get('measured',{});row.update(status=v['status'],A=m.get('area'),D=m.get('delay'),T=v['mapping']['wall_secs'],CEC=m.get('cec'),search_finished=v.get('native',{}).get('search_finished'))
   rows.append(row)
 def norm(x):return {k:v for k,v in x.items()if k not in ['stream_small','incremental_delay','protect_batch_seed']}
 control=states['v2_epfl_control'];assert all(norm(v['configuration'])==norm(control['configuration']) for v in states.values());assert all(v['fingerprint']==control['fingerprint']for v in states.values())
 with(out/'all_scope.csv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=['cohort','bench','status','A','D','T','CEC','search_finished']);w.writeheader();w.writerows(rows)
 traces={}
 for n,rs in data.items():
  for b in rs:
   path=r/n/b/'online.stages.jsonl'
   if not path.exists():continue
   events=[json.loads(x)for x in path.read_text().splitlines()];active=False;yielded=False;recover=False;flow=False;spans=[];start=None
   for e in events:
    s=e['stage']
    if s=='online_delta_match_start':active=True;start=e['unix_secs']
    elif s=='online_delta_match_done':
     if start is not None:spans.append(e['unix_secs']-start)
     active=False
    elif active:
     yielded|=s=='online_small_yield';recover|=s=='online_recovery_slice';flow|=s=='online_incremental_delay_yield'
   traces[n+'/'+b]=dict(ordinary_yields=yielded,recovery_inside_ordinary= recover,incremental_flow_inside_ordinary=flow,whole_delta_intervals=spans)
 (out/'traces.json').write_text(json.dumps(traces,indent=2)+'\n')
 base=json.loads(a.baselines.read_text());candidate={k:v for k,v in base.items()if k!='results'};candidate['results']=[];current=data.get('v2_epfl_combined',{})
 for bench in control['configuration']['designs']:
  row=dict(bench=bench,method='cooperative600-batch-v2',status='missing',optimizer_succeeded=False,input_sha256=hashlib.sha256((ROOT/f'tools/mockturtle/experiments/benchmarks/{bench}.aig').read_bytes()).hexdigest())
  if bench in current:
   v=current[bench];assert v['native']['library_sha256']==base['library_sha256'];row.update(status=v['status'],measured=v.get('measured'),optimizer_succeeded=v['native'].get('optimizer_succeeded',False),search_finished=v['native'].get('search_finished'),raw_mapping_timeout=v['mapping']['timeout'],mapping_wall_secs=v['mapping']['wall_secs'])
  candidate['results'].append(row)
 (out/'candidate.json').write_text(json.dumps(candidate,indent=2)+'\n');(out/'baselines.json').write_text(json.dumps(base,indent=2)+'\n');gates={}
 for obj in ['delay-area','area-delay','pareto']:
  args=[sys.executable,str(ROOT/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(out/'candidate.json'),'--baselines',str(out/'baselines.json'),'--required-baselines',*sorted({x['method']for x in base['results']}),'--objective',obj,'--output',str(out/(obj+'.json'))];q=subprocess.run(args,capture_output=True,text=True);(out/(obj+'.log')).write_text(q.stdout+q.stderr);assert q.returncode in [0,1];gates[obj]=json.loads((out/(obj+'.json')).read_text())['all_passed']
 repeat={}
 for b in ['adder','bar']:
  vals=[data[n][b]['measured']for n in ['v2_epfl_combined','v2_epfl_repeat1','v2_epfl_repeat2']if b in data.get(n,{})];repeat[b]=dict(runs=len(vals),identical_AD=len(vals)==3 and len({(v['area'],v['delay'])for v in vals})==1)
 summary=dict(completed={n:len(x)for n,x in data.items()},unique_designs=len({b for x in data.values()for b in x}),independent_CEC_outputs=sum(x.get('measured',{}).get('cec')==1 for d in data.values()for x in d.values()),repeats=repeat,full_EPFL_complete=len(current)==20,full_EPFL_ablations_complete=all(len(data.get(n,{}))==20 for n in names[:4]),gates=gates,main_merged=False,paper_updated=False,baseline_algorithms_changed=False,total_mapping_seconds=600,CEC_in_T=False,general_lazy_materialization=False,full_area_flow_resumable=False)
 (out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n');text=['# Cooperative v2 with independent batch predecessor: limited-scope evidence','','Same binary and library, one600s mapping clock; independent CEC outside clock. No baseline algorithm changes.','','|Design|Control A/D/T|Small cursor A/D/T|Dirty delay A/D/T|Combined A/D/T|','|---|---|---|---|---|']
 for b in control['configuration']['designs']:
  if not any(b in data.get(n,{})for n in names[:4]):continue
  vals=[]
  for n in names[:4]:
   v=data.get(n,{}).get(b);m=v.get('measured',{})if v else{};vals.append(f"{m['area']:.2f}/{m['delay']:.2f}/{v['mapping']['wall_secs']:.2f}"if m else (v['status']if v else'missing'))
  text.append('|'+b+'|'+'|'.join(vals)+'|')
 text+=['',f"Unique designs: {summary['unique_designs']}; independentCEC outputs: {summary['independent_CEC_outputs']}. Repeats: {repeat}.",'','Cursor setup/care/individual target collection and whole-cover validation remain nonpreemptive operations. Delay labels use conservative readiness including all physical inputs; a finite acyclic witness is checked before a cover can be committed. The batch frontier predecessor runs once per delta; this is not a completely scan-free mapper. Physical matches remain eager; general lazy materialization and full area-flow scheduling are unfinished. All failed, timed-out, regressed and missing cases remain in all_scope.csv. No main adoption or full-suite advantage claim.']
 
 if 'v2_epfl_batch_off' in data:
  text+=['','|Design|Batch archive off A/D|Batch archive on A/D|','|---|---|---|']
  for b in ['adder','bar']:
   v=data['v2_epfl_batch_off'].get(b,{}).get('measured');w=current.get(b,{}).get('measured')
   if v and w:text.append(f"|{b}|{v['area']:.2f}/{v['delay']:.2f}|{w['area']:.2f}/{w['delay']:.2f}|")
 (out/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
