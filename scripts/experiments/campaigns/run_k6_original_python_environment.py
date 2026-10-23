"""Restore the original author's literal python command, without algorithm edits.

Adopt only byte-verified completed jobs; retain the four confirmed missing-python
failures and the interrupted case in the donor cohort. The original frozen
runner executes every new job with its original parameters and timeout policy.
"""
import json
import os
from pathlib import Path
import shutil
import sys

import run_single_output_bfd_campaign as runner
from run_epfl_baselines import ROOT, sha

BASE = ROOT / 'results_v2/single_output_k6_20261020'
OUTPUT = BASE / 'resumed_baselines_python_command'


def main():
    assert Path(sys.executable).resolve() == Path('/home/flynn/miniconda3/bin/python3').resolve()
    assert '--kind' in sys.argv and sys.argv[sys.argv.index('--kind') + 1] == 'baseline'
    assert Path(sys.argv[sys.argv.index('--output') + 1]).resolve() == OUTPUT
    policy_path = BASE / 'resume_recovery/python_command_policy.json'
    policy = json.loads(policy_path.read_text())
    os.environ['PATH'] = policy['shim'] + ':' + os.environ['PATH']
    for name, expected in policy['after'].items():
        resolved = shutil.which(name)
        assert resolved == expected and sha(Path(resolved)) == policy['executable_sha256'][name], name
    checkpoint = OUTPUT / 'campaign.json'
    proof_path = BASE / 'resume_recovery/python_environment_adoption.json'
    if not checkpoint.exists():
        donor_path = BASE / 'baseline77_environment_interrupted_checkpoint.json'
        donor = json.loads(donor_path.read_text())
        assert len(donor['jobs']) == 77
        for path, wanted in donor['fingerprint']['files'].items():
            assert sha(Path(path)) == wanted, 'frozen dependency drift: ' + path
        for source, native in donor['fingerprint']['native'].items():
            for relative, wanted in native['tracked_sha256'].items():
                assert sha(Path(source) / relative) == wanted, 'author source drift'
        adopted, excluded = [], []
        for job in donor['jobs']:
            assert all(sha(Path(p)) == wanted for p, wanted in job['sha256'].items()), job['name']
            native_logs = list(Path(job['report']).parent.rglob('native.log')) if job.get('report') else []
            missing = [p for p in native_logs if 'Failed to execute command: Os { code: 2' in p.read_text()]
            if missing:
                assert '_esyn_' in job['name'] and job['ok'] == 0
                excluded.append(dict(name=job['name'], reason='author literal python command absent from PATH',
                    original_job=job, native_logs={str(p): sha(p) for p in missing}))
            else:
                adopted.append(dict(job, environment_adoption='literal python command not responsible for this observation'))
        assert len(adopted) == 73 and {j['name'] for j in excluded} == {
            'r0_6in_esyn_area-delay_cavlc', 'r0_6in_esyn_delay-area_cavlc',
            'r0_6in_esyn_area-delay_ctrl', 'r0_6in_esyn_delay-area_ctrl'}
        assert donor['current_job']['name'] == 'r0_6in_emorphic_ctrl'
        record = dict(donor=str(donor_path), donor_sha256=sha(donor_path), adopted_count=73,
            adopted_jobs=[j['name'] for j in adopted], excluded_environment_failures=excluded,
            interrupted_case=donor['current_job'], interrupted_report_complete=False,
            policy=str(policy_path), policy_sha256=sha(policy_path), launcher_sha256=sha(Path(__file__)),
            original_runner_sha256=sha(Path(runner.__file__)), baseline_algorithms_changed=False,
            budgets_changed=False, no_environment_failure_used_as_algorithm_comparison=True)
        assert not proof_path.exists()
        proof_path.write_text(json.dumps(record, indent=2) + '\n')
        OUTPUT.mkdir(exist_ok=False)
        donor.update(jobs=adopted, status='checkpointed',
                     environment_recovery=dict(path=str(proof_path), sha256=sha(proof_path)))
        donor.pop('current_job')
        checkpoint.write_text(json.dumps(donor, indent=2) + '\n')
    proof = json.loads(proof_path.read_text())
    assert proof['launcher_sha256'] == sha(Path(__file__))
    assert proof['policy_sha256'] == sha(policy_path)
    runner.main()


if __name__ == '__main__':
    main()
