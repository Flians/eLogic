#!/usr/bin/env python3
"""Independent full-care cut factor audit; changed pools are NOT assumed equivalent."""
import argparse,json,subprocess,sys,hashlib,collections
from pathlib import Path
from analyze_single_output_structure import slice_field
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);a=p.parse_args();c=json.loads(a.campaign.read_text())
 r=subprocess.run([sys.executable,str(Path(__file__).with_name('check_single_output_components.py')),str(a.campaign)],capture_output=True,text=True)
 (a.campaign.parent/'base_validation.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr
 v=json.loads((a.campaign.parent/'validation.json').read_text());proof=[];graphs={}
 for j in c['jobs']:
  r=json.loads(Path(j['report']).read_text())['results'][0];s=r['settings'];flag=int(j['profile']!='legacy_control');assert s['EGM_FUNCTIONAL_PHASE_SCREEN']==str(int(j['profile']=='phase_screen'));assert s['EGM_FUNCTIONAL_CUTS']==str(flag) and s['EGM_TERMINAL_AREA']=='1'
  if r['status']!='ok':continue
  n=r['native'];assert n['slack_rounds_started']<=4 and n['slack_local_trials']<=8000
  events=json.loads(Path(j['trace']).read_text())['events'];terminal=[e['data'] for e in events if e['stage']=='terminal_area']
  assert bool(terminal)==n['terminal_area_attempted']
  for d in terminal:
   assert d['after_all_legacy_cover_comparisons'] and d['no_new_deadline']
   assert d['area']<=d['old_area'] and d['delay']<=d['old_delay']
   assert d['old_area_rounds']<=d['total_area_rounds']<=4 and d['remaining_rounds']==4-d['old_area_rounds']
   assert d['total_area_trials']-d['old_area_trials']<=2000*d['remaining_rounds']
   assert (d['area'],d['delay'])==(r['measured']['area'],r['measured']['delay'])
   assert d['accepted']==n['terminal_area_accepted']
  assert n['functional_cut_trials']<=32*32*n['eclasses'];assert n['functional_cut_proofs']<=n['functional_cut_trials']
  if not flag:assert n['functional_cut_trials']==n['functional_cut_proofs']==n['functional_cut_admitted']==0
  raw=Path(j['candidates']).read_bytes();graphs[j['k'],j['bench'],j['profile']]={f:hashlib.sha256(slice_field(raw,f,g)).hexdigest() for f,g in [('egraph_nodes','free'),('canonical_signals','canonical_wire_only')]}
  proof.append(dict(k=j['k'],bench=j['bench'],profile=j['profile'],trials=n['functional_cut_trials'],proofs=n['functional_cut_proofs'],admissions_across_rounds=n['functional_cut_admitted']))
 for k,b,p in graphs:
  other=k,b,'functional_cuts' if p=='legacy_control' else 'legacy_control'
  if other in graphs:assert graphs[k,b,p]==graphs[other],(k,b,'unexpected graph or alias mutation')
 v.update(functional_cuts=proof,unchanged_graph_and_aliases=True,changed_cut_and_physical_pools_expected=True,no_cross_pool_incumbent_preservation_claim=True,terminal_own_run_old_final_nonincrease_checked=True,proof_scope='quotients checked for all original independent support assignments; complete quotient required; finite whole-cover cycle/port CEC still required; counter counts admissions across rounds, not final retained novel cuts')
 (a.campaign.parent/'phase_screen_validation.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v[k] for k in ['complete','attempts','cec','summary']},indent=2))
if __name__=='__main__':main()
