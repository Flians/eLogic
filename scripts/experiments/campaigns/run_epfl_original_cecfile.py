#!/usr/bin/env python3
"""Original baseline continuation excluding three audited, unused paper helpers.

No native algorithm, driver, clock, library or input is changed.
The common CEC backend changes only Yosys command transport.
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
        f['files'][str(Path(__file__).with_name('run_epfl_original_dependencies.py'))] = sha(Path(__file__).with_name('run_epfl_original_dependencies.py'))
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
        proofpath=policy.ROOT/'results_v2/cec_scriptfile_transport_20261013/integration.json'
        transport=json.loads(proofpath.read_text())
        src=policy.ROOT/'emapper/src/egraph_map.rs';exe=policy.ROOT/'target/release/bench_rs_netlist_eval'
        assert sha(src)==transport['new_source_sha256'] and sha(exe)==transport['new_evaluator_sha256']
        previous_source=(proofpath.parent/'egraph_map.before.rs').read_text().split('\n#[cfg(test)]')[0]
        current_source=src.read_text().split('\n#[cfg(test)]')[0]
        diff=transport['production_source_difference_exactly']
        assert current_source==previous_source.replace(diff['before'],diff['after'])
        assert old['fingerprint']['files'][str(src)]==transport['old_source_sha256']
        assert old['fingerprint']['files'][str(exe)]==transport['old_evaluator_sha256']
        allowed_transport={str(src),str(exe)}
        checked, unused, transport_changes = {}, {}, {}
        for p, digest in old['fingerprint']['files'].items():
            actual = sha(Path(p))
            if p in excluded:
                unused[p] = dict(old=digest, new=actual, reason='unused paper plotting/reporting/other experiment helper')
            elif actual != digest:
                if p not in allowed_transport:raise ValueError('executed dependency changed: '+p)
                transport_changes[p]={'old':digest,'new':actual,'purpose':'CEC transport only; exact production-source patch and positive/negative tests verified'}
            else:
                checked[p] = digest
        # None of the still-frozen scripts imports or invokes these three names.
        # Do not generalize this exemption to future files or driver changes.
        invocation_audit = {}
        for p in sorted(set(checked) | set(f['files'])):
            q = Path(p)
            if q.suffix == '.py' and q.resolve() not in [Path(__file__).resolve(),Path(__file__).with_name('run_epfl_original_dependencies.py').resolve()]:
                text = q.read_text()
                for name in UNUSED:
                    if name in text or name.removesuffix('.py') in text:
                        raise ValueError('excluded helper referenced by frozen script: '+p)
                invocation_audit[p] = sha(q)
        names = {j[0] for j in policy.jobs()}; jobs = []; retry=[]
        replaypath=policy.ROOT/'results_v2/cec_scriptfile_mem_ctrl_replay_20261013/report.json'
        replay=json.loads(replaypath.read_text());assert all(r['status']=='ok' and r['measured']['cec']==1 for r in replay['results'])
        assert replay['verification_transport_replay']['mapping_rerun'] is False
        retry_names={'3in_eqmap_area-delay_mem_ctrl','3in_eqmap_delay-area_mem_ctrl'}
        for j in old['jobs']:
            if j['name'] not in names: raise ValueError('unexpected prior job')
            for key in ['report', 'audit']:
                if key in j and sha(Path(j[key])) != j[key+'_sha256']:
                    raise ValueError('modified prior observation')
            if 'report' not in j and j['exit_code'] == 0:
                raise ValueError('successful job without report')
            if j['name'] in retry_names:
                report=json.loads(Path(j['report']).read_text())
                eqrow=next(r for r in report['results'] if r['method']=='eqmap_lut_amap')
                log=Path(j['report']).parent/'mem_ctrl/eqmap_lut_amap/lut-cec.log'
                assert eqrow['status']!='ok' and 'Argument list too long' in log.read_text()
                retry.append({'prior_job':j,'reason':'initial LUT CEC transport failed before author EqMap execution; same original parameters under corrected verifier','lut_cec_log_sha256':sha(log)})
                continue
            entry = copy.deepcopy(j)
            if j['name']=='3in_core_mem_ctrl':
                entry['prior_verification_report']=j['report'];entry['prior_verification_report_sha256']=j['report_sha256']
                entry['report']=str(replaypath);entry['report_sha256']=sha(replaypath);entry['ok']=4;entry['failed']=0
                entry['verification_only_replay']=True
                entry['verification_replay_validation']=str(replaypath.parent/'validation.json')
                entry['verification_replay_validation_sha256']=sha(replaypath.parent/'validation.json')
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
                 'statuses_preserved_except_explicit_verification_replay': True, 'transport_only_changes':transport_changes,'transport_proof':str(proofpath),'transport_proof_sha256':sha(proofpath),'retry_pre_native_verification_jobs':retry,'algorithm_changed': False, 'clock_changed': False,
                 'verification_policy':'Yosys -s passes the exact old command text through a file. Original positive CEC remains valid; all old failures retained, only explicit saved-netlist verification replay and pre-native verifier failure retries change records.'}
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
