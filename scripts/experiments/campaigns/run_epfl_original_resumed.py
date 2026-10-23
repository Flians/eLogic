#!/usr/bin/env python3
"""Explicit continuation after audited unrelated report/experiment helper changes."""
import copy,fcntl,json,sys
from pathlib import Path
import run_epfl_original_queuebridge as bridge
from run_epfl_baselines import sha
UNUSED={'export_paper.py','report.py','run_observability.py','run_chain_factor.py','run_observability_depth.py','run_chain_controls.py','confirm_arbiter_method.py','run_global_divisors.py','audit_mapped_ppa.py'}
original=bridge.policy.main

def resume():
    oldfinger=bridge.policy.fingerprint
    def fingerprint(subjects):
        f=oldfinger(subjects)
        for n in UNUSED:f['files'].pop(str(bridge.policy.ROOT/'scripts/experiments'/n),None)
        f['files'][str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
        return f
    bridge.policy.fingerprint=fingerprint
    out=Path(sys.argv[sys.argv.index('--output')+1]).resolve();out.mkdir(exist_ok=True)
    subjects=Path(sys.argv[sys.argv.index('--subjects')+1]).resolve();sp=out/'campaign.json'
    if not sp.exists():
        prior=bridge.policy.ROOT/'results_v2/epfl_original_baselines_queuebridge_checkpoint_20261013/campaign.json'
        lock=(prior.parent/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=json.loads(prior.read_text());assert not old.get('current_job')
        f=fingerprint(subjects)
        for p in [Path(bridge.policy.__file__).resolve(),Path(__file__).with_name('run_with_mapping_budget.py'),Path(__file__).with_name('nested_mapping_clock.py'),*sorted((bridge.policy.ROOT/'emapper/src').rglob('*.rs')),bridge.policy.ROOT/'src/bin/bench_rs_egraph_map.rs']:f['files'][str(p)]=sha(p)
        assert f['native']==old['fingerprint']['native']
        excluded={str(bridge.policy.ROOT/'scripts/experiments'/n) for n in UNUSED}
        for p,h in old['fingerprint']['files'].items():
            if p not in excluded:assert f['files'].get(p)==h,('dependency changed',p)
        added=set(f['files'])-set(old['fingerprint']['files']);assert added=={str(Path(__file__).resolve())},added
        # Each frozen root helper and nested executable helper must not refer to
        # excluded modules or paths. This is a specific dependency audit, not a
        # blanket exemption for arbitrary future source changes.
        checked={}
        for p,h in old['fingerprint']['files'].items():
            path=Path(p)
            if path.name in {'bench_egraph_baselines.py','bench_eqmap_baseline.py','bench_esyn_baseline.py','bench_emorphic_sa.py','bench_emorphic_baseline.py','shared_subject.py','port_identifiers.py','benchmark_suites.py','restrict_genlib.py','capture_emorphic_abc.py','run_with_mapping_budget.py','nested_mapping_clock.py','run_epfl_mapping_campaign.py'}:
                text=path.read_text()
                for n in UNUSED:
                    assert n not in text and ('import '+n[:-3]) not in text and ('from '+n[:-3]+' import') not in text,(p,n)
                checked[p]=h
        for j in old['jobs']:
            for k in ['report','audit']:
                if j.get(k):assert sha(Path(j[k]))==j[k+'_sha256']
        proof={'prior':str(prior),'prior_sha256':sha(prior),'adopted_jobs':len(old['jobs']),'excluded_unused_files':{n:sha(bridge.policy.ROOT/'scripts/experiments'/n) for n in sorted(UNUSED)},'reference_audit':checked,'native_identical':True,'executed_dependencies_identical':True,'baseline_algorithm_or_budget_changed':False,'interrupted_attempt':old.get('interruption_recovery')}
        pp=out/'adoption.json';pp.write_text(json.dumps(proof,indent=2)+'\n')
        state=copy.deepcopy(old);state.update(fingerprint=f,adoption_report=str(pp),adoption_sha256=sha(pp))
        for j in state['jobs']:j['adopted_dependency_campaign']=str(prior)
        sp.write_text(json.dumps(state,indent=2)+'\n');lock.close()
    original()
bridge.policy.main=resume
if __name__=='__main__':bridge.main()
