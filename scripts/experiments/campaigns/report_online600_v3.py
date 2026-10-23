"""Read-only paired V3 experiment report; first2 and repetitions never imply full20 coverage."""
import json,argparse
from pathlib import Path
from report_online600 import load

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 records=[];cohorts={};frozen=None
 for rep,suffix in enumerate(['','_repeat1','_repeat2']):
  ca,cr=load(a.study/('epfl_v3'+suffix)/'campaign.json');co,br=load(a.study/('epfl_v3_control'+suffix)/'campaign.json')
  assert {k:v for k,v in ca['configuration'].items()if k!='area_recovery'}=={k:v for k,v in co['configuration'].items()if k!='area_recovery'}
  native={k:v for k,v in ca['fingerprint'].items()if '/source_v3/'in k};assert native
  if frozen is None:frozen=native
  assert native==frozen=={k:v for k,v in co['fingerprint'].items()if '/source_v3/'in k}
  cohorts[suffix or 'first']={'candidate_completed':len(cr),'control_completed':len(br),'expected':20}
  for bench in ca['configuration']['designs']:
   row={'bench':bench,'repetition':rep,'candidate_status':'missing','control_status':'missing'}
   for label,d in [('candidate',cr),('control',br)]:
    if bench in d:
     r=d[bench];row[label+'_status']=r['status'];row[label]=dict(r.get('measured')or{},T=r['mapping']['wall_secs'],search_timeout=r['mapping']['timeout'])
   if 'candidate'in row and 'control'in row:
    c=row['candidate'];b=row['control'];row.update(delta_A=c['area']-b['area'],delta_D=c['delay']-b['delay'])
   records.append(row)
 actual=[r for r in records if 'candidate'in r and 'control'in r];assert len(actual)==6
 for b in ['adder','bar']:
  vals=[r['candidate']for r in actual if r['bench']==b];assert len(vals)==3 and len({(r['area'],r['delay'])for r in vals})==1
 out={'source_same_binary':True,'mapping_seconds':600,'CEC_in_runtime':False,'main_merged':False,'baseline_algorithms_changed':False,'cohorts':cohorts,'rows':records,'actual_valid_CEC_pairs':len(actual),'all_candidate_AD_reproduced':True,'full20_complete':False,'old_area_regression_adder_retained':True,'limitations':['only adder/bar repeated; other18 missing in every full20 cohort','all delta matches eagerly materialized; delayed materialization pending','area recovery after cut closure; no proof of optimality or baseline dominance']}
 (a.output/'validation.json').write_text(json.dumps(out,indent=2)+'\n')
 lines=['# V3: full-cover area recovery, limited-scope paired validation','','One600-second mapping-search window, CEC excluded. Same source/binary per pair; three rounds with alternated configuration order.','','|Design|Rep|Control A|Candidate A|D control / candidate|T control / candidate|CEC both|','|---|---:|---:|---:|---:|---:|---:|']
 for r in actual:
  c=r['candidate'];b=r['control'];lines.append(f"|{r['bench']}|{r['repetition']}|{b['area']:.2f}|{c['area']:.2f}|{b['delay']:.2f} / {c['delay']:.2f}|{b['T']:.2f} / {c['T']:.2f}|1|")
 lines+=['','Old full-flow areas are adder87.06 and bar262.77. The new adder remains about0.04 larger; bar improves. These are separate historical full-flow comparisons, not the same-binary ablation above. All regressions and missing designs remain. No main adoption or final paper-table update. Deferred materialization is not implemented.']
 (a.output/'report.md').write_text('\n'.join(lines)+'\n');print('6 paired tests; 12CEC outputs; full20 incomplete')
if __name__=='__main__':main()
