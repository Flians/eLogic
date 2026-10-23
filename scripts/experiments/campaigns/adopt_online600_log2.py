"""Explicit exact-dependency adoption, retaining source checkpoint/raw timeout; no mapper run."""
import argparse,copy,fcntl,hashlib,json
from pathlib import Path
from report_online600 import load
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args();r=a.study.resolve();cp=r/'epfl_v3/campaign.json';sp=r/'log2_v3_probe/campaign.json';locks=[]
 if a.apply:
  for f in [cp.parent/'runner.lock',ROOT/'results_v2/independent_components_20261019/global.lock',ROOT/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock']:
   h=f.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 full,fr=load(cp);sub,sr=load(sp);assert not full.get('current_job') and not sub.get('current_job')
 c=full['configuration'].copy();d=sub['configuration'].copy();c.pop('designs');d.pop('designs');assert c==d
 assert all(full['fingerprint'].get(k)==v for k,v in sub['fingerprint'].items());assert list(sr)==['log2'];assert 'log2'in full['configuration']['designs']
 if 'log2'in fr:print('already explicitly adopted; no mutation');return
 job=copy.deepcopy(sub['jobs'][0]);assert job['bench']=='log2';proof={'qualification':'same kernel/harness/actual native dependencies and parameters; only design selection list differs','source_checkpoint':str(sp),'source_checkpoint_sha256':sha(sp),'canonical_before_sha256':sha(cp),'actual_dependency_subset_equal':True,'parameters_except_design_list_equal':True,'all_source_jobs_adopted_without_quality_selection':True,'source_status':sr['log2']['status'],'source_search_timeout_retained':sr['log2']['mapping']['timeout'],'baseline_algorithms_changed':False,'main_kernel_changed':False}
 if not a.apply:print(json.dumps(proof,indent=2));return
 directory=r/'log2_adoption';directory.mkdir(exist_ok=False);before=directory/'canonical.before.json';before.write_bytes(cp.read_bytes());proof['canonical_before_snapshot']=str(before);pf=directory/'proof.json';pf.write_text(json.dumps(proof,indent=2)+'\n')
 job.update(observation_kind='explicit_same_dependency_targeted_adoption',adopted_from=str(sp),dependency_proof=str(pf));job['sha256'].update({str(pf):sha(pf),str(before):sha(before),str(sp):sha(sp)})
 full['jobs'].append(job);full['status']='checkpointed';temp=cp.with_suffix('.pending');temp.write_text(json.dumps(full,indent=2)+'\n');temp.replace(cp);print('canonical3/20; log2 raw600timeout and independentCEC retained; no rerun/main change')
if __name__=='__main__':main()
