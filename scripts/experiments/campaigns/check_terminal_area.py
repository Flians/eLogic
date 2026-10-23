#!/usr/bin/env python3
"""Independent accounting for terminal continuation and its old final incumbent."""
import argparse,subprocess,sys,json,hashlib,re
from pathlib import Path
from analyze_single_output_structure import slice_field
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('campaign',type=Path);a=ap.parse_args()
 p=subprocess.run([sys.executable,str(Path(__file__).with_name('check_single_output_components.py')),str(a.campaign)],capture_output=True,text=True)
 (a.campaign.parent/'base_validation.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
 c=json.loads(a.campaign.read_text());v=json.loads((a.campaign.parent/'validation.json').read_text());proof=[];diags={}
 for j in c['jobs']:
  r=json.loads(Path(j['report']).read_text())['results'][0];assert r['settings']['EGM_TERMINAL_AREA']==str(int(j['profile']=='terminal_area'))
  if r['status']!='ok':continue
  n=r['native'];assert n['slack_rounds_started']<=4 and n['slack_local_trials']<=8000
  trace=json.loads(Path(j['trace']).read_text())['events'];ev=[e for e in trace if e['stage']=='terminal_area']
  if j['profile']=='legacy_control':assert not ev and not n['terminal_area_attempted']
  assert bool(ev)==n['terminal_area_attempted']
  if ev:
   assert len(ev)==1;d=ev[0]['data'];assert d['configured_rounds']==4 and d['configured_trials_per_round']==2000
   assert d['old_area_rounds']<=d['total_area_rounds']<=4 and d['remaining_rounds']==4-d['old_area_rounds']
   assert d['total_area_trials']-d['old_area_trials']<=2000*d['remaining_rounds']
   assert d['area']<=d['old_area'] and d['delay']<=d['old_delay'] and d['no_new_deadline'] and d['after_all_legacy_cover_comparisons']
   stages=[e['stage'] for e in trace];assert stages.index('terminal_area')>stages.index('live_refs_and_slack')
   assert d['accepted']==n['terminal_area_accepted'];assert (d['area'],d['delay'])==(r['measured']['area'],r['measured']['delay'])
   proof.append(dict(k=j['k'],bench=j['bench'],**d))
  raw=Path(j['candidates']).read_bytes();diags[j['k'],j['bench'],j['profile']]={f:hashlib.sha256(slice_field(raw,f,g)).hexdigest() for f,g in [('egraph_nodes','free'),('cuts','delay'),('canonical_signals','canonical_wire_only')]}
  diags[j['k'],j['bench'],j['profile']]['pool_fp']=re.search(rb'"retained_pool_fingerprint":"([^"]+)"',raw).group(1).decode()
 for k,b,p in diags:
  other=(k,b,'terminal_area' if p=='legacy_control' else 'legacy_control')
  if other in diags:assert diags[k,b,p]==diags[other],(k,b,'candidate-source drift')
 v.update(terminal_proofs=proof,terminal_attempted_cases=len(proof),terminal_accepted_cases=sum(d['accepted'] for d in proof),unchanged_graph_cuts_and_aliases=True,old_complete_cover_nonincrease_certified=True,per_run_certificate_not_cross_run_guarantee=True)
 (a.campaign.parent/'terminal_validation.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v[k] for k in ['complete','attempts','cec','summary','terminal_attempted_cases','terminal_accepted_cases']},indent=2))
if __name__=='__main__':main()
