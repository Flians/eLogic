#!/usr/bin/env python3
"""Audit cached already-scored covers, no extra oracle calls/search allowances."""
import argparse,subprocess,sys,json,hashlib,re
from pathlib import Path
from analyze_single_output_structure import slice_field

def main():
 ap=argparse.ArgumentParser();ap.add_argument('campaign',type=Path);a=ap.parse_args();r=subprocess.run([sys.executable,str(Path(__file__).with_name('check_single_output_components.py')),str(a.campaign)],capture_output=True,text=True);(a.campaign.parent/'base_validation.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 c=json.loads(a.campaign.read_text());v=json.loads((a.campaign.parent/'validation.json').read_text());proof=[];diags={}
 for j in c['jobs']:
  r=json.loads(Path(j['report']).read_text())['results'][0];assert r['settings']['EGM_SCORED_AREA_ARCHIVE']==str(int(j['profile']=='scored_archive')) and r['settings']['EGM_TERMINAL_AREA']=='0'
  if r['status']!='ok':continue
  n=r['native'];assert not n['terminal_area_attempted'] and n['slack_rounds_started']<=4 and n['slack_local_trials']<=8000
  events=json.loads(Path(j['trace']).read_text())['events'];ev=[e for e in events if e['stage']=='scored_area_archive'];assert bool(ev)==(j['profile']=='scored_archive' and any(e['stage']=='live_refs_and_slack' for e in events))
  if ev:
   assert len(ev)==1;d=ev[0]['data'];assert d['area']<=d['old_area'] and d['delay']<=d['old_delay'] and d['after_all_legacy_cover_comparisons'];assert d['extra_scores']==d['extra_trials']==d['extra_rounds']==0
   assert 0<=d['updates']<=d['observed']<=n['slack_local_scored']+n['slack_local_trials'];assert (d['area'],d['delay'])==(r['measured']['area'],r['measured']['delay'])
   stages=[e['stage'] for e in events]
   if 'deferred_incumbent_selection' in stages:assert stages.index('scored_area_archive')>stages.index('deferred_incumbent_selection')
   assert stages.index('scored_area_archive')<stages.index('final_validation');proof.append(dict(k=j['k'],bench=j['bench'],**d))
  raw=Path(j['candidates']).read_bytes();diags[j['k'],j['bench'],j['profile']]={f:hashlib.sha256(slice_field(raw,f,g)).hexdigest() for f,g in [('egraph_nodes','free'),('cuts','delay'),('canonical_signals','canonical_wire_only')]}
  diags[j['k'],j['bench'],j['profile']]['pool_fp']=re.search(rb'"retained_pool_fingerprint":"([^"]+)"',raw).group(1).decode()
 for k,b,p in diags:
  other=(k,b,'scored_archive' if p=='legacy_control' else 'legacy_control')
  if other in diags:assert diags[k,b,p]==diags[other]
 v.update(archive_proofs=proof,archive_observed_cases=sum(d['observed']>0 for d in proof),archive_saved_cases=sum(d['updates']>0 for d in proof),archive_accepted_cases=sum(d['accepted'] for d in proof),same_graph_cuts_aliases=True,old_complete_cover_nonincrease_certified=True,extra_oracle_scores_trials_rounds=0,per_run_not_inter_run_guarantee=True)
 (a.campaign.parent/'archive_validation.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v[k] for k in ['complete','attempts','cec','summary','archive_observed_cases','archive_saved_cases','archive_accepted_cases']},indent=2))
if __name__=='__main__':main()
