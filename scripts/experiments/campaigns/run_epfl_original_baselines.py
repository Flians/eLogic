#!/usr/bin/env python3
"""Original baseline algorithms, 300s mapping policy, auditable prior observations."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import sys
import run_epfl_mapping_campaign as policy
from run_epfl_baselines import sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--subjects',type=Path,required=True)
    ap.add_argument('--max-jobs',type=int,default=2)
    ap.add_argument('--adopt-campaign',type=Path)
    a=ap.parse_args()
    original_jobs=policy.jobs
    policy.jobs=lambda: (j for j in original_jobs() if '_egraph_' not in j[0])
    original_fingerprint=policy.fingerprint
    def fingerprint(subjects):
        f=original_fingerprint(subjects);f['files'][str(Path(__file__).resolve())]=sha(Path(__file__).resolve());return f
    policy.fingerprint=fingerprint
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);sp=out/'campaign.json'
    if a.adopt_campaign and not sp.exists():
        oldpath=a.adopt_campaign.resolve()
        adoption_lock=(oldpath.parent/'runner.lock').open('a')
        fcntl.flock(adoption_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=json.loads(oldpath.read_text())
        if old.get('current_job'):raise ValueError('prior campaign not quiescent')
        if old['mapping_seconds']!=300 or old['baseline_algorithms_changed']:
            raise ValueError('prior policy differs')
        f=fingerprint(a.subjects.resolve())
        if old['fingerprint']['native']!=f['native']:raise ValueError('native baseline sources changed')
        checked={};excluded={}
        for p,digest in old['fingerprint']['files'].items():
            # Only unused emapper mapper sources/binary may change. All executed
            # baseline native binaries, models, root adapters, evaluator, clocks,
            # inputs, subject manifest and library must remain byte-identical.
            candidate_only=('/emapper/src/' in p or p.endswith('/src/bin/bench_rs_egraph_map.rs')
                            or p.endswith('/target/release/bench_rs_egraph_map'))
            actual=sha(Path(p))
            if actual!=digest:
                if not candidate_only:raise ValueError('executed baseline dependency changed: '+p)
                excluded[p]={'old':digest,'new':actual,'reason':'unused Rust candidate mapper'}
            else:checked[p]=digest
        # Match the policy runner's complete frozen source set for future batches.
        for p in [Path(policy.__file__).resolve(),Path(policy.__file__).with_name('run_with_mapping_budget.py'),
                  Path(policy.__file__).with_name('nested_mapping_clock.py'),
                  *sorted((policy.ROOT/'emapper/src').rglob('*.rs')),policy.ROOT/'src/bin/bench_rs_egraph_map.rs']:
            f['files'][str(p)]=sha(p)
        names={j[0] for j in policy.jobs()};jobs=[]
        for j in old['jobs']:
            if j['name'] not in names:continue
            if 'audit' not in j:raise ValueError('missing audit')
            for key in ['report','audit']:
                if key in j and sha(Path(j[key]))!=j[key+'_sha256']:raise ValueError('modified observation')
            if 'report' in j:
                report=json.loads(Path(j['report']).read_text())
                if any(r['method'].startswith('egraph') for r in report['results']):raise ValueError('candidate output cannot be adopted as baseline')
            elif j['exit_code']==0:raise ValueError('successful job missing report')
            entry=copy.deepcopy(j);entry['adopted_from_campaign']=str(oldpath)
            if 'report' not in j:entry.update(ok=0,failed=1,unmeasured=True,error='original failed process produced no report; original log/audit/argv retained')
            jobs.append(entry)
        proof={'prior_campaign':str(oldpath),'prior_campaign_sha256':sha(oldpath),
            'checked_identical_dependencies':checked,'unused_candidate_changes':excluded,
            'native_fingerprint_equal':True,'status_preserved':True,'adoption_script_sha256':sha(Path(__file__).resolve())}
        (out/'adoption.json').write_text(json.dumps(proof,indent=2))
        state={'suite':'EPFL20 arithmetic/control','subjects':str(a.subjects.resolve()),'mapping_seconds':300,
            'policy':'mapping only; CEC/resyn2/adapters/metrics excluded','existing_process_limits_preserved':True,
            'baseline_algorithms_changed':False,'fingerprint':f,'expected_jobs':240,'jobs':jobs,'status':'checkpointed',
            'adoption_report':str(out/'adoption.json'),'adoption_sha256':sha(out/'adoption.json')}
        sp.write_text(json.dumps(state,indent=2))
        fcntl.flock(adoption_lock,fcntl.LOCK_UN);adoption_lock.close()
    sys.argv=[policy.__file__,'--output',str(out),'--subjects',str(a.subjects.resolve()),'--max-jobs',str(a.max_jobs)]
    policy.main()

if __name__=='__main__':main()
