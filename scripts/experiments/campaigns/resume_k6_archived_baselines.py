"""Original baseline runner with archived tracked-source provenance after /tmp loss.

Author files and executables must exactly match the archived frozen fingerprint.
No author Git metadata is fabricated; commit identities are archived provenance.
New attempts write directly into the persistent workspace, with lost suffix
explicitly excluded rather than silently adopted from the pause counter.
"""
import copy
import json
from pathlib import Path
import sys

import run_single_output_bfd_campaign as runner
from run_epfl_baselines import ROOT, sha

BASE = ROOT / 'results_v2/single_output_k6_20261020'
OUTPUT = BASE / 'resumed_baselines_persistent_v2'


def verify_archived(reference):
    for path, wanted in reference['files'].items():
        assert sha(Path(path)) == wanted, 'executed dependency drift: ' + path
    for source, native in reference['native'].items():
        for relative, wanted in native['tracked_sha256'].items():
            assert sha(Path(source) / relative) == wanted, 'author source drift: ' + source + '/' + relative
    return copy.deepcopy(reference)


def main():
    assert Path(sys.executable).resolve() == Path('/home/flynn/miniconda3/bin/python3').resolve(), 'use the original frozen cohort Python runtime'
    assert '--output' in sys.argv
    output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
    assert output == OUTPUT.resolve(), 'this recovery has one explicitly named cohort'
    reference_path = BASE / 'baseline36_checkpoint.json'
    donor = json.loads(reference_path.read_text())
    reference = donor['fingerprint']
    verify_archived(reference)
    output.mkdir(exist_ok=True)
    checkpoint = output / 'campaign.json'
    proof = BASE / 'resume_recovery/dependency_recovery_v2.json'
    if not checkpoint.exists():
        assert len(donor['jobs']) == 36 and not donor.get('current_job')
        for job in donor['jobs']:
            assert all(sha(Path(path)) == wanted for path, wanted in job['sha256'].items()), job['name']
        record = dict(donor=str(reference_path), donor_sha256=sha(reference_path),
            adopted_jobs=[j['name'] for j in donor['jobs']], adopted_jobs_count=36,
            pause_counter_observed=107, lost_unarchived_suffix_count=71,
            interrupted_case='r0_6in_emorphic_priority', interrupted_artifacts_available=False,
            lost_suffix_adopted=False, lost_suffix_will_be_new_attempts=True,
            native_git_metadata_recreated=False, author_commits='archived provenance with every tracked byte verified',
            baseline_algorithms_changed=False, baseline_parameters_changed=False,
            budgets_changed=False, output_policy='new artifacts directly in persistent workspace',
            recovery_runner_sha256=sha(Path(__file__)),
            frozen_runner_sha256=sha(Path(runner.__file__)))
        assert not proof.exists()
        proof.write_text(json.dumps(record, indent=2) + '\n')
        donor['recovery_provenance'] = dict(path=str(proof), sha256=sha(proof))
        donor['status'] = 'checkpointed'
        checkpoint.write_text(json.dumps(donor, indent=2) + '\n')
    runner.fingerprint = lambda subjects: verify_archived(reference)
    runner.main()


if __name__ == '__main__':
    main()
