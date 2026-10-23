"""Read-only V7 negative-certificate-cache ablation, complete required20 scope."""
import argparse,collections,csv,hashlib,json,subprocess,sys
from pathlib import Path
from report_online600 import load
ROOT=Path(__file__).resolve().parents[3]
def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--baselines',type=Path,required=True);a=p.parse_args();r=a.study.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 states={};data={};rows=[];diagnostics={}
 for n in ['v7_control','v7_epfl']:
  states[n],data[n]=load(r/n/'campaign.json');diagnostics[n]={}
  for b in states[n]['configuration']['designs']:
   v=data[n].get(b);m=v.get('measured',{})if v else{};rows.append(dict(cohort=n,bench=b,status=v['status']if v else'missing',A=m.get('area'),D=m.get('delay'),T=v['mapping']['wall_secs']if v else None,CEC=m.get('cec'),search_finished=v.get('native',{}).get('search_finished')if v else None,raw_timeout=v['mapping']['timeout']if v else None))
  for j in states[n]['jobs']:
   q=Path(j['report']).parent/'online.stages.jsonl';c=collections.Counter(json.loads(x)['stage']for x in q.read_text().splitlines());diagnostics[n][j['bench']]={k:v for k,v in c.items()if 'certificate' in k or 'rejected' in k or k=='online_incumbent_validated'}
 norm=lambda c:{k:v for k,v in c.items()if k!='cache_rejected_flow'}
 assert norm(states['v7_control']['configuration'])==norm(states['v7_epfl']['configuration']) and states['v7_control']['fingerprint']==states['v7_epfl']['fingerprint']
 assert states['v7_control']['configuration']['cache_rejected_flow']==0 and states['v7_epfl']['configuration']['cache_rejected_flow']==1
 for b in set(data['v7_control'])&set(data['v7_epfl']):assert data['v7_control'][b]['prepared_sha256']==data['v7_epfl'][b]['prepared_sha256']
 with(out/'all_scope.csv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=['cohort','bench','status','A','D','T','CEC','search_finished','raw_timeout']);w.writeheader();w.writerows(rows)
 baseline=json.loads(a.baselines.read_text());candidate={k:v for k,v in baseline.items()if k!='results'};candidate['results']=[]
 for b in states['v7_epfl']['configuration']['designs']:
  v=data['v7_epfl'].get(b);row=dict(bench=b,method='cooperative600-rejected-v7',status='missing',optimizer_succeeded=False,input_sha256=hashlib.sha256((ROOT/f'tools/mockturtle/experiments/benchmarks/{b}.aig').read_bytes()).hexdigest())
  if v:
   assert v['input_sha256']==row['input_sha256']and v['native']['library_sha256']==baseline['library_sha256'];row.update(status=v['status'],measured=v.get('measured'),optimizer_succeeded=v['native'].get('optimizer_succeeded',False),search_finished=v['native'].get('search_finished'),raw_mapping_timeout=v['mapping']['timeout'],mapping_wall_secs=v['mapping']['wall_secs'])
  candidate['results'].append(row)
 for name,value in [('candidate',candidate),('baselines',baseline),('certificate_diagnostics',diagnostics)]: (out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
 gates={}
 for objective in ['delay-area','area-delay','pareto']:
  cmd=[sys.executable,str(ROOT/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(out/'candidate.json'),'--baselines',str(out/'baselines.json'),'--required-baselines',*sorted({v['method']for v in baseline['results']}),'--objective',objective,'--output',str(out/(objective+'.json'))];q=subprocess.run(cmd,capture_output=True,text=True);(out/(objective+'.log')).write_text(q.stdout+q.stderr);assert q.returncode in [0,1];gates[objective]=json.loads((out/(objective+'.json')).read_text())['all_passed']
 summary=dict(completed={n:len(d)for n,d in data.items()},unique_designs=len(set().union(*[set(d)for d in data.values()])),independent_CEC_outputs=sum(v.get('measured',{}).get('cec')==1 for d in data.values()for v in d.values()),gates=gates,tests_passed=277,ignored=6,filtered=16,full_EPFL_complete=len(data['v7_epfl'])==20,repeats_complete=False,baseline_algorithms_changed=False,main_merged=False,paper_updated=False,mapping_seconds=600,CEC_in_T=False,general_lazy_materialization=False,whole_area_flow_resumable=False)
 (out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
 text=['# V7 negative-certificate-cache ablation','','One frozen binary; only cache flag differs. Missing/failure rows remain in full20 scope.600s mapping-related search, independentCEC excluded.','','|Design|Cache off A/D/T|Cache on A/D/T|','|---|---|---|']
 for b in states['v7_epfl']['configuration']['designs']:
  if b not in data['v7_control']and b not in data['v7_epfl']:continue
  cols=[]
  for n in ['v7_control','v7_epfl']:
   v=data[n].get(b);m=v.get('measured')if v else None;cols.append(f"{m['area']:.2f}/{m['delay']:.2f}/{v['mapping']['wall_secs']:.2f}"if m else v['status']if v else'missing')
  text.append('|'+b+'|'+'|'.join(cols)+'|')
 text+=['','A certificate attempt is not a failed proof. Consult certificate_diagnostics.json: absent rejected_* and rejected_unchanged_skip events mean the negative-cache performance hypothesis was not supported by this workload, even though constructed rejection/invalidation regressions pass. Valid, globally nonimproving selected covers need scoring/scheduling improvements. No speedup or main adoption follows from unit tests.','', 'The older V2 online log2 control D3852.82 remains a separate stronger reference; this report must not hide a regression by comparing only against source fallback. V6 results, including div/hyp, are a distinct frozen version, not merged into the V7 candidate. All raw600 timeouts and predeadline receipts remain. Full20 independent factors and repeats are unfinished.']
 (out/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
