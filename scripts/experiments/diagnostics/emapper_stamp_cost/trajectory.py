"""Align retained work counters and committed cover identities, not just A/D."""
import json,pathlib,sys,hashlib
R=pathlib.Path(__file__).resolve().parents[4];P=R/'results_v2/emapper_physical_stamp_20261111'
sys.path.insert(0,str(R/'scripts/experiments/campaigns/emapper_physical_stamp'));from observations import read

def covers(folder):
 out=[]
 for d in sorted((folder/'incumbents').glob('[0-9]'*8)):
  m=json.loads((d/'manifest.json').read_text());out.append({k:m.get(k)for k in ['sequence','area','delay','batches','matched_cuts','instances','sha256','selected_evidence']})
 return out

def intervals(folder):
 frames,tail=read(folder/'observation.jsonl');prev=None;deltas=[]
 for f in frames:
  t=f.get('details',{}).get('telemetry');d=f.get('details',{})
  if t is None:continue
  if prev:
   old,old_t=prev;delta=t.get('consider_outcome_secs',{}).get('metric_rejected',0)-old_t.get('consider_outcome_secs',{}).get('metric_rejected',0)
   if delta>0:deltas.append(dict(before=old['elapsed_seconds'],after=f['elapsed_seconds'],metric_reject_wall=delta,calls=t['consider_calls']-old_t['consider_calls'],event=d.get('event'),pool=d.get('pool'),cuts=d.get('matched_cuts')))
  prev=(f,t)
 return sorted(deltas,key=lambda r:r['metric_reject_wall'],reverse=True)[:12],tail

def main():
 out=R/'results_v2/emapper_stamp_diagnosis_20261111/trajectory.json';assert not out.exists();c=json.loads((P/'benchmark/campaign.json').read_text());reports=[]
 for rep in [0,1]:
  a,b=[pathlib.Path(c['jobs'][f'dec_r{rep}_{v}']['case'])for v in ['control','borrow']];u,v=covers(a),covers(b);at=next((i for i in range(min(len(u),len(v)))if u[i]!=v[i]),None)
  ia,ta=intervals(a);ib,tb=intervals(b)
  reports.append(dict(rep=rep,all_committed_cover_records_identical=u==v,count=[len(u),len(v)],first_difference=at,first_different_records=[u[at],v[at]]if at is not None else None,control_metric_reject_intervals=ia,borrow_metric_reject_intervals=ib,scope='same published trajectory is stronger evidence than same final netlist, but not a hash of all unselected candidates'))
 out.write_text(json.dumps(dict(rows=reports,helper_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n');print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
