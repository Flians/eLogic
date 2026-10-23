#!/usr/bin/env python3
"""Original baseline continuation excluding three audited, unused paper helpers.

No native algorithm, driver, clock, library, input or evaluator is changed.
The previous broad root-Python fingerprint caught unrelated paper work.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import sys
import run_epfl_mapping_campaign as policy
from run_epfl_baselines import sha

UNUSED = {'plot_paper.py', 'run_three_parts.py', 'report_three_parts.py'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subjects', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--adopt-campaign', type=Path)
    ap.add_argument('--max-jobs', type=int, default=12)
    a = ap.parse_args()
    if a.max_jobs < 1: ap.error('max-jobs must be positive')
    original_jobs, original_fingerprint = policy.jobs, policy.fingerprint
    policy.jobs = lambda: (j for j in original_jobs() if '_egraph_' not in j[0])
    excluded = {str(policy.ROOT/'scripts/experiments'/n) for n in UNUSED}

    def fingerprint(subjects):
        f = original_fingerprint(subjects)
        f['files'] = {p: h for p, h in f['files'].items() if p not in excluded}
        f['files'][str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
        # Keep the actual predecessor runner frozen as well.
        p = Path(__file__).with_name('run_epfl_original_baselines.py')
        f['files'][str(p)] = sha(p)
        return f
    policy.fingerprint = fingerprint
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    path = out/'campaign.json'
    if a.adopt_campaign and not path.exists():
        oldpath = a.adopt_campaign.resolve()
        lock = (oldpath.parent/'runner.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = json.loads(oldpath.read_text())
        if old.get('current_job'): raise ValueError('prior campaign not quiescent')
        f = fingerprint(a.subjects.resolve())
        if old['mapping_seconds'] != 300 or old['baseline_algorithms_changed']:
            raise ValueError('different baseline or clock policy')
        if old['fingerprint']['native'] != f['native']:
            raise ValueError('native source/commit/model/binary dependencies changed')
        checked, unused = {}, {}
        for p, digest in old['fingerprint']['files'].items():
            actual = sha(Path(p))
            if p in excluded:
                unused[p] = dict(old=digest, new=actual, reason='unused paper plotting/reporting/other experiment helper')
            elif actual != digest:
                raise ValueError('executed dependency changed: '+p)
            else:
                checked[p] = digest
        # None of the still-frozen scripts imports or invokes these three names.
        # Do not generalize this exemption to future files or driver changes.
        invocation_audit = {}
        for p in sorted(set(checked) | set(f['files'])):
            q = Path(p)
            if q.suffix == '.py' and q.resolve() != Path(__file__).resolve():
                text = q.read_text()
                for name in UNUSED:
                    if name in text or name.removesuffix('.py') in text:
                        raise ValueError('excluded helper referenced by frozen script: '+p)
                invocation_audit[p] = sha(q)
        names = {j[0] for j in policy.jobs()}; jobs = []
        for j in old['jobs']:
            if j['name'] not in names: raise ValueError('unexpected prior job')
            for key in ['report', 'audit']:
                if key in j and sha(Path(j[key])) != j[key+'_sha256']:
                    raise ValueError('modified prior observation')
            if 'report' not in j and j['exit_code'] == 0:
                raise ValueError('successful job without report')
            entry = copy.deepcopy(j)
            entry['adopted_dependency_campaign'] = str(oldpath)
            jobs.append(entry)
        # The policy runner adds these to its complete startup fingerprint.
        for p in [Path(policy.__file__).resolve(), Path(policy.__file__).with_name('run_with_mapping_budget.py'),
                  Path(policy.__file__).with_name('nested_mapping_clock.py'),
                  *sorted((policy.ROOT/'emapper/src').rglob('*.rs')),
                  policy.ROOT/'src/bin/bench_rs_egraph_map.rs']:
            f['files'][str(p)] = sha(p)
        proof = {'prior_campaign': str(oldpath), 'prior_campaign_sha256': sha(oldpath),
                 'identical_executed_dependencies': checked, 'unused_helpers': unused,
                 'invocation_audit': invocation_audit, 'native_fingerprint_equal': True,
                 'new_unused_files': {str(policy.ROOT/'scripts/experiments'/n): sha(policy.ROOT/'scripts/experiments'/n)
                                      for n in UNUSED if str(policy.ROOT/'scripts/experiments'/n) not in old['fingerprint']['files']},
                 'statuses_preserved': True, 'algorithm_changed': False, 'clock_changed': False,
                 'failed_resume': 'Previous broad-fingerprint runner rejected unused plot_paper changes before any new mapping.'}
        (out/'adoption.json').write_text(json.dumps(proof, indent=2)+'\n')
        state = {k: copy.deepcopy(v) for k, v in old.items()
                 if k not in ['fingerprint', 'jobs', 'adoption_report', 'adoption_sha256']}
        state.update(fingerprint=f, jobs=jobs, status='checkpointed',
                     adoption_report=str(out/'adoption.json'), adoption_sha256=sha(out/'adoption.json'))
        path.write_text(json.dumps(state, indent=2)+'\n')
        lock.close()
    sys.argv = [policy.__file__, '--subjects', str(a.subjects.resolve()), '--output', str(out),
                '--max-jobs', str(a.max_jobs)]
    policy.main()


if __name__ == '__main__':
    main()
