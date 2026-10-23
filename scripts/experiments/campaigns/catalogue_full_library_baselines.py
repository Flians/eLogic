"""Index a uniform stock9 core and unchanged original noncore; never select a per-design winner."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--core',type=Path,required=True);ap.add_argument('--noncore',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists();core=json.loads(a.core.read_text());other=json.loads(a.noncore.read_text());assert core['configuration']['stock_emap_library_inputs']==9 and core['configuration']['stock_structural_cut_capacity']==9
 jobs=[];libs=set()
 for label,state in [('stock9_core',core),('original_noncore',other)]:
  for job in state['jobs']:
   if label=='stock9_core':assert '_core_' in job['name']
   elif '_core_' in job['name']:continue
   assert all(sha(p)==h for p,h in job['sha256'].items()),job['name']
   entry=dict(job,source_cohort=label);jobs.append(entry)
   if entry.get('report'):libs.add(json.loads(Path(entry['report']).read_text())['library_sha256'])
 assert len({j['name']for j in jobs})==len(jobs) and len(libs)<=1
 out=dict(expected_jobs=120,jobs=jobs,status='attempted'if len(jobs)==120 else'checkpointed',configuration={'scope':'full47 single-output library; stock emap library9/structural9; original author noncore settings','baseline_algorithms_changed':False,'same_library_hashes':sorted(libs),'selection':'all core cases use stock9 configuration, all noncore cases use original algorithms; no per-design or best-run selection','old_capacity6_core_and_library9_cut6_preflight':'preserved separate historical/sensitivity studies'},fingerprint={'files':{str(p.resolve()):sha(p)for p in [a.core,a.noncore,Path(__file__)]}},source_cohorts={'core':str(a.core.resolve()),'noncore':str(a.noncore.resolve())})
 a.output.write_text(json.dumps(out,indent=2));print('uniform primary baseline catalogue',len(jobs),'/120 jobs')
if __name__=='__main__':main()
