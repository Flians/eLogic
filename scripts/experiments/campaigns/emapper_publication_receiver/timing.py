"""Monotonic commit/receipt evidence; read-only, never repairs an old receipt."""
import argparse,json,pathlib,statistics,sys
from observations import read
R=pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0,str(R/'scripts/emapper'))
from provenance import sha

def summary(values):
 return dict(count=len(values),minimum=min(values),median=statistics.median(values),maximum=max(values))if values else dict(count=0)
def case(folder):
 native,nt=read(folder/'native.publication-timing.jsonl');front,ft=read(folder/'frontend.publication-timing.jsonl')
 commits={};stages={}
 for r in native:
  assert r['clock']=='CLOCK_MONOTONIC'and r['event']=='native_committed'
  assert r['prepare_started_monotonic']<=r['commit_before_monotonic']<=r['commit_after_monotonic']
  assert r['sequence']not in commits;commits[r['sequence']]=r
 for r in front:
  assert r['clock']=='CLOCK_MONOTONIC'
  # A retry may discover an incomplete IO path again; retain the last complete stage.
  stages.setdefault(r['sequence'],{})[r['event']]=r
 receipt=json.loads((folder/'raw.incumbent.receipt.json').read_text());receipt_hash=sha(folder/'raw.incumbent.receipt.json')
 selected=receipt['sequence'];rows=[];lags=[];late=[];missing=[]
 for seq,n in sorted(commits.items()):
  deadline=n['caller_deadline_monotonic'];assert deadline==receipt['deadline_monotonic']
  s=stages.get(seq,{})
  item=dict(sequence=seq,native=n,frontend=s,commit_definitely_before_deadline=n['commit_after_monotonic']<deadline,official_selected=seq==selected)
  if 'discovered'in s:
   discovered=s['discovered']['monotonic']
   item['commit_to_discovery_interval_seconds']=[discovered-n['commit_after_monotonic'],discovered-n['commit_before_monotonic']]
   lags.append(item['commit_to_discovery_interval_seconds'][1])
  if 'receipt_written'in s:
   written=s['receipt_written']['monotonic'];item['receipt_completed_before_deadline']=written<deadline
   if not item['receipt_completed_before_deadline']:late.append(seq)
   assert s['verified']['monotonic']<=s['receipt_write_begin']['monotonic']<=written
   item['verify_to_receipt_seconds']=written-s['verified']['monotonic']
  elif seq==selected:missing.append('selected receipt has no completed write trace')
  rows.append(item)
 selected_trace=stages.get(selected,{})
 if selected not in commits:missing.append('selected receipt has no native commit trace')
 if selected_trace.get('receipt_written',{}).get('receipt_sha256')!=receipt_hash:missing.append('selected receipt trace SHA missing/mismatch')
 if selected in commits and commits[selected]['commit_after_monotonic']>=receipt['deadline_monotonic']:missing.append('selected native commit not proven pre-deadline')
 if selected in late:missing.append('selected receipt write completed late')
 committed_dirs={int(p.name)for p in (folder/'incumbents').glob('[0-9]'*8)if p.is_dir()}
 return dict(case=str(folder),selected_sequence=selected,qualified_complete_monotonic_chain=not missing,qualification_issues=missing,native_records=len(native),commit_to_discovery_upper_seconds=summary(lags),late_receipt_sequences=late,untraced_committed_sequences=sorted(committed_dirs-set(commits)),native_tail=nt,frontend_tail=ft,rows=rows,source_hashes={p.name:sha(p)for p in [folder/'native.publication-timing.jsonl',folder/'frontend.publication-timing.jsonl',folder/'raw.incumbent.receipt.json']},note='Commit instant lies inside native before/after bracket; discovery minus after can be negative under scheduling. Missing tail is unknown, never inferred from wall time. This audit does not replace official results.')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();assert not a.output.exists()
 c=json.loads(a.campaign.read_text());rows=[]
 for r in c['jobs'].values():
  if r['status']=='ok':rows.append(case(pathlib.Path(r['case'])))
  else:rows.append(dict(case=r['case'],status=r['status'],qualified_complete_monotonic_chain=False,reason='failed attempt retained'))
 out=dict(completed=len(c['jobs']),planned=len(c['plan']),cases=rows,all_completed_qualified=bool(rows)and all(r['qualified_complete_monotonic_chain']for r in rows),baseline_runs=0,old_results_modified=False)
 a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(completed=out['completed'],qualified=out['all_completed_qualified'],cases=[{k:r[k]for k in ['case','selected_sequence','native_records','qualification_issues']if k in r}for r in rows]),indent=2))
if __name__=='__main__':main()
