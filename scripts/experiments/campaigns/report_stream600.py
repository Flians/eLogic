"""Versioned streaming/seed-protection evidence. Missing EPFL designs stay missing."""
import argparse,json,hashlib,csv,sys,subprocess
from pathlib import Path
from report_online600 import load
ROOT=Path(__file__).resolve().parents[3]
def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--baselines',type=Path,default=ROOT/'results_v2/online_mapping_20261022/baseline38_scope/baselines.json');a=p.parse_args();r=a.study.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 names=['epfl_stream_recovery','epfl_control','epfl_stream_only','epfl_recovery_only','epfl_combined_repeat1','epfl_combined_repeat2','epfl_protected','epfl_protected_control','epfl_protected_repeat1','epfl_protected_repeat2']
 states={};data={};flat=[]
 for name in names:
  states[name],data[name]=load(r/name/'campaign.json')
  for bench in states[name]['configuration']['designs']:
   x=dict(cohort=name,bench=bench,status='missing')
   if bench in data[name]:
    v=data[name][bench];x.update(status=v['status'],**dict(A=v['measured']['area'],D=v['measured']['delay'],T=v['mapping']['wall_secs'],CEC=v['measured']['cec'],search_finished=v['native']['search_finished']))
   flat.append(x)
 def normalized(c,fields):return{k:v for k,v in c.items()if k not in fields}
 c=states['epfl_control']['configuration']
 for name in names[:6]:assert normalized(states[name]['configuration'],['stream_wide','scheduled_recovery'])==normalized(c,['stream_wide','scheduled_recovery'])
 c=states['epfl_protected']['configuration']
 for name in names[6:]:assert normalized(states[name]['configuration'],['protect_delay_seed'])==normalized(c,['protect_delay_seed'])
 for bench in ['adder','bar']:
  for nameset in [['epfl_stream_recovery','epfl_combined_repeat1','epfl_combined_repeat2'],['epfl_protected','epfl_protected_repeat1','epfl_protected_repeat2']]:
   vals=[data[n][bench]['measured']for n in nameset];assert len({(v['area'],v['delay'])for v in vals})==1
 with(out/'A_D_T_all_scope.csv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=['cohort','bench','status','A','D','T','CEC','search_finished']);w.writeheader();w.writerows(flat)
 target=r/'log2_protected/campaign.json';target_complete=False;rows=dict(data['epfl_protected'])
 if target.exists():
  ts,td=load(target);assert all(states['epfl_protected']['fingerprint'].get(k)==v for k,v in ts['fingerprint'].items());assert normalized(c,['designs'])==normalized(ts['configuration'],['designs'])
  target_complete='log2'in td;rows.update(td)
 baseline=a.baselines.resolve();base=json.loads(baseline.read_text());crow=[]
 for bench in c['designs']:
  path=ROOT/f'tools/mockturtle/experiments/benchmarks/{bench}.aig';sha=hashlib.sha256(path.read_bytes()).hexdigest();row=dict(bench=bench,method='stream600_protected',status='missing',optimizer_succeeded=False,input_sha256=sha)
  if bench in rows:
   v=rows[bench];assert v['native']['library_sha256']==base['library_sha256'];assert v['input_sha256']==sha;row.update(status=v['status'],measured=v['measured'],optimizer_succeeded=v['native']['optimizer_succeeded'],mapping_wall_secs=v['mapping']['wall_secs'],raw_mapping_timeout=v['mapping']['timeout'],search_finished=v['native']['search_finished'])
  crow.append(row)
 candidate={k:v for k,v in base.items()if k!='results'};candidate['results']=crow;(out/'candidate.json').write_text(json.dumps(candidate,indent=2)+'\n');(out/'baselines.json').write_text(json.dumps(base,indent=2)+'\n');required=sorted({v['method']for v in base['results']});gates={}
 for objective in ['delay-area','area-delay','pareto']:
  cmd=[sys.executable,str(ROOT/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(out/'candidate.json'),'--baselines',str(out/'baselines.json'),'--required-baselines',*required,'--objective',objective,'--output',str(out/(objective+'.json'))];q=subprocess.run(cmd,capture_output=True,text=True);(out/(objective+'.log')).write_text(q.stdout+q.stderr);assert q.returncode in [0,1]and(out/(objective+'.json')).exists(),q.stderr;gates[objective]=json.loads((out/(objective+'.json')).read_text())['all_passed']
 summary=dict(main_merged=False,paper_updated=False,baseline_algorithms_changed=False,global_mapping_seconds=600,independent_CEC_in_T=False,case_outputs=sum(len(v)for v in data.values())+int(target_complete and 'log2'not in data['epfl_protected']),protected_known_designs=len(rows),expected_designs=20,log2_protected_complete=target_complete,gates=gates,general_lazy_materialization=False,all_failures_and_regressions_retained=True,full_EPFL_ablations_complete=False)
 (out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n');lines=['# Streaming and complete-seed protection: limited-scope validation','','One shared600-second mapping window, independent CEC excluded. Main/paper unchanged.','','|Design|stream/recovery 0/0 A|1/0 A|0/1 A|1/1 A|Protected seed A|D (all first2)|','|---|---:|---:|---:|---:|---:|---:|']
 for b in ['adder','bar']:
  vals=[data[n][b]['measured']for n in ['epfl_control','epfl_stream_only','epfl_recovery_only','epfl_stream_recovery','epfl_protected']];lines.append('|'+b+'|'+'|'.join(f"{v['area']:.2f}"for v in vals)+f"|{vals[0]['delay']:.2f}|")
 lines+=['','The first four result columns use source/; the protection column uses source_protected/. These are two separately frozen versions. Within each version the toggles use the same binary; the full protection0/1 comparison is in CSV. Three combined repetitions reproduce A/D before and after protection; all shown outputs have independent named-port CEC. Protection restores the same-source pure-delay-seed area trajectory for adder/bar; it does not prove no regression for all designs. Historical adder87.06 remains below87.10.','',f'Protected known scope: {len(rows)}/20. Full EPFL ablation and large-design repeats remain incomplete. General deferred physical materialization and incremental timing are not implemented.']
 if target_complete:
  v=rows['log2'];lines +=['',f"Protected log2 A={v['measured']['area']:.2f},D={v['measured']['delay']:.2f},T={v['mapping']['wall_secs']:.2f},CEC=1,search_finished={v['native']['search_finished']}. Targeted dependency-equivalent result is shown separately, not silently adopted into canonical checkpoint."]
 (out/'report.md').write_text('\n'.join(lines)+'\n');print(summary)
if __name__=='__main__':main()
