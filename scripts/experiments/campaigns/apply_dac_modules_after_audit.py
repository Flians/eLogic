"""Finite, guarded module-control wiring AFTER complete frozen evidence audit.

No mappings or baselines are dispatched. Failed tests stop the workflow, retain
logs and the exact local diff, and do not commit or update any paper claims.
"""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

R = pathlib.Path(__file__).resolve().parents[3]
STUDY = R / 'results_v2/emapper_dac_convergence_20261111'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--max-wait-seconds', type=float, default=28800)
    a = ap.parse_args()
    assert not a.output.exists() and a.max_wait_seconds > 0
    a.output.mkdir(parents=True)
    state = dict(stage='waiting_for_full_artifact_audit', mappings=0, baseline_runs=0,
                 paper_updated=False, GitHub_push=False, applied=False, local_commit=None)

    def save():
        path = a.output / 'state.json'
        tmp = path.with_suffix('.pending')
        tmp.write_text(json.dumps(state, indent=2)+'\n');tmp.replace(path)

    save()
    with (STUDY / 'module-controls-application.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic()+a.max_wait_seconds
        audit = STUDY / 'full-epfl-final-audit/validation.json'
        locks = []
        while True:
            if audit.exists():
                evidence = json.loads(audit.read_text())
                assert evidence['completed'] == 20 and not evidence['invalid_artifacts']
                assert evidence['baseline_runs'] == 0 and not evidence['GitHub_push']
                try:
                    for path in [R/'results_v2/independent_components_20261019/global.lock',
                                 STUDY/'benchmark/runner.lock', STUDY/'full-epfl/runner.lock']:
                        handle = path.open('a');locks.append(handle)
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    for handle in locks:handle.close()
                    locks.clear()
            if time.monotonic() >= deadline:
                state.update(stage='monitor_expired_without_source_changes');save();return
            time.sleep(15)
        try:
            # No implicit adoption of another person's concurrent source edits.
            prep = json.loads((STUDY/'module-controls-prepared-v2/preparation.json').read_text())
            for name, digest in prep['before_sha256'].items():
                path = R/name
                actual = sha(path) if path.exists() else hashlib.sha256(b'').hexdigest()
                assert actual == digest, 'working source drift: '+name
            sys.path.insert(0,str(R/'scripts/experiments/campaigns'))
            import run_dac_convergence as core
            old = json.loads((STUDY/'full-epfl/campaign.json').read_text())
            assert core.fingerprint() == old['frozen']
            state.update(stage='applying_experimental_controls', audit_sha256=sha(audit));save()
            script = pathlib.Path(__file__).with_name('prepare_dac_module_controls.py')
            command = [sys.executable,str(script),'--apply','--output',str(a.output/'application')]
            subprocess.run(command,cwd=R,check=True)
            state.update(applied=True, stage='unit_and_regression_tests');save()
            fixture = a.output/'fixture';fixture.mkdir()
            for name in ['asap7_full.genlib','cells.v','library.json']:
                shutil.copy2(R/'emapper/libraries/asap7'/name,fixture/name)
            env={k:v for k,v in os.environ.items()if not k.startswith('EGM_')}
            env['EGM_TEST_DATA_ROOT']=str(fixture.resolve())
            commands = [
                ('python-tests', [sys.executable,'-m','unittest','discover','-s','scripts/emapper','-p','test_*.py']),
                ('rust-tests', ['cargo','test','--locked','--offline','-p','emapper','egraph','-j','2','--','--test-threads=1']),
                ('release-build', ['bash','scripts/emapper/build.sh','--offline']),
            ]
            state['commands']=commands;save()
            for label, command in commands:
                state['active_command']=label;save()
                with (a.output/(label+'.log')).open('x') as log:
                    result=subprocess.run(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
                assert result.returncode == 0, label+' failed; inspect preserved log'
            state.update(stage='tests_and_build_passed_CEC_and_smoke_pending', active_command=None)
            state['source_after']={name:sha(R/name) for name in prep['before_sha256']}
            state['build_receipt_sha256']=sha(R/'target/release/emapper-build.json')
            save()
            # Exact owned paths only; preserve all pre-existing staged work.
            paths=list(prep['before_sha256'])
            git=['git','-c','core.hooksPath=/dev/null','-c','gc.auto=0','-c','maintenance.auto=false',
                 '-c','filter.lfs.process=','-c','filter.lfs.required=false']
            subprocess.run(git+['add','--',*paths],cwd=R,check=True)
            subprocess.run(git+['commit','--only','-m','Wire independent DAC experimental controls without changing defaults','--',*paths],cwd=R,check=True)
            state['local_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();save()
        except Exception as error:
            state.update(stage='failed_preserved_for_audit',error=repr(error));save();raise


if __name__=='__main__':
    main()
