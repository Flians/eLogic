"""Explicit checkpoint adoption of every targeted job, exact same mapper/harness dependencies."""
import argparse,copy,fcntl,hashlib,json
from pathlib import Path
from report_online600 import load
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--canonical',required=True);p.add_argument('--target',required=True);p.add_argument('--apply',action='store_true');a=p.parse_args();r=a.study.resolve();cp=r/a.canonical/'campaign.json';sp=r/a.target/'campaign.json';locks=[]
 if a.apply:
  for f in [cp.parent/'runner.lock',ROOT/'results_v2/independent_components_20261019/global.lock',ROOT/'results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock']:
   h=f.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 full,fr=load(cp);sub,sr=load(sp);assert not full.get('current_job')and not sub.get('current_job');c=full['configuration'].copy();d=sub['configuration'].copy();c.pop('designs');d.pop('designs');assert c==d;assert all(full['fingerprint'].get(k)==v for k,v in sub['fingerprint'].items());assert set(sr)<=set(full['configuration']['designs']);jobs=[copy.deepcopy(j)for j in sub['jobs']if j['bench']not in fr]
 if not jobs:print('already recorded');return
 proof=dict(source_checkpoint=str(sp),source_checkpoint_sha256=sha(sp),canonical_before_sha256=sha(cp),same_executed_dependencies=True,same_parameters_except_design_selection=True,all_new_target_jobs_adopted_without_quality_filter=True,source_results={j['bench']:dict(status=sr[j['bench']]['status'],raw_mapping=sr[j['bench']]['mapping'])for j in jobs},main_merged=False)
 if not a.apply:print(json.dumps(proof,indent=2));return
 folder=r/(a.canonical+'_'+a.target+'_adoption');folder.mkdir(exist_ok=False);before=folder/'canonical.before.json';before.write_bytes(cp.read_bytes());proof['canonical_before_snapshot']=str(before);pf=folder/'proof.json';pf.write_text(json.dumps(proof,indent=2)+'\n')
 for j in jobs:
  j.update(observation_kind='explicit_same_dependency_target_adoption',adopted_from=str(sp),dependency_proof=str(pf));j['sha256'].update({str(pf):sha(pf),str(before):sha(before),str(sp):sha(sp)});full['jobs'].append(j)
 full['status']='checkpointed';temp=cp.with_suffix('.pending');temp.write_text(json.dumps(full,indent=2)+'\n');temp.replace(cp);print('canonical',len(full['jobs']),'/',full['expected_jobs'],'all raw outcomes retained')
if __name__=='__main__':main()
