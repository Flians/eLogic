"""Classify complete but unreceipted transactions without selecting/replacing them."""
import argparse,json,pathlib,sys
R=pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0,str(R/'scripts/emapper'))
from provenance import sha
from frontend import valid_integrity
from observations import read

def inspect(case):
 receipt=json.loads((case/'raw.incumbent.receipt.json').read_text());deadline=receipt['deadline_monotonic']
 native,nt=read(case/'native.publication-timing.jsonl');front,ft=read(case/'frontend.publication-timing.jsonl')
 byseq={r['sequence']:r for r in native};stages={}
 for r in front:stages.setdefault(r['sequence'],[]).append(r)
 misses=[];unknown=[];best=None
 for folder in sorted((case/'incumbents').glob('[0-9]'*8)):
  m=json.loads((folder/'manifest.json').read_text());seq=m['sequence'];n=byseq.get(seq)
  if n is None:unknown.append(seq);continue
  if not n['commit_after_monotonic']<deadline:unknown.append(seq);continue
  if best is None or(m['delay'],m['area'])<(best['delay'],best['area']):best=m
  if(m['delay'],m['area'])>=(receipt['delay'],receipt['area']):continue
  data=(folder/'mapped.v').read_bytes()
  misses.append(dict(sequence=seq,AD=[m['area'],m['delay']],full_byte_integrity=valid_integrity(m,data),manifest_sha256=sha(folder/'manifest.json'),netlist_sha256=sha(folder/'mapped.v'),native=n,frontend_events=stages.get(seq,[]),independent_CEC=False,classification='pre-deadline complete native transaction with better declared metrics; diagnostic only, no result replacement'))
 return dict(case=str(case),selected_sequence=receipt['sequence'],selected_AD=[receipt['area'],receipt['delay']],best_proven_predeadline_declared_AD=[best['area'],best['delay']]if best else None,predeadline_better_committed_than_selected=misses,time_unknown_sequences=unknown,native_tail=nt,frontend_tail=ft,CEC_of_unselected_not_implied=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();assert not a.output.exists()
 c=json.loads(a.campaign.read_text());result=dict(completed=len(c['jobs']),planned=len(c['plan']),rows=[inspect(pathlib.Path(r['case']))if r['status']=='ok'else dict(case=r['case'],status=r['status'])for r in c['jobs'].values()],helper_sha256=sha(pathlib.Path(__file__)),official_values_modified=False)
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(completed=result['completed'],cases=[dict(case=r['case'],misses=len(r.get('predeadline_better_committed_than_selected',[])),unknown=len(r.get('time_unknown_sequences',[])))for r in result['rows']]),indent=2))
if __name__=='__main__':main()
