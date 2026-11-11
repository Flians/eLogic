"""Finite post-worker evidence audit; no mapping dispatch or periodic schedule.

Wait for the already-running full20 worker to release its lock. If its completed
checkpoint is incomplete, preserve it and stop instead of restarting any case.
"""
import argparse
import fcntl
import json
import pathlib
import subprocess
import sys
import time


R = pathlib.Path(__file__).resolve().parents[3]
DEFAULT = R / 'results_v2/emapper_dac_convergence_20261111/full-epfl'


def ready(study):
    with (study / 'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 'worker_running'
        campaign = json.loads((study / 'campaign.json').read_text())
        state = json.loads((study / 'state.json').read_text())
        return ('ready' if len(campaign['jobs']) == campaign['expected'] == 20
                and state.get('active') is None else 'incomplete_checkpoint')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study', type=pathlib.Path, default=DEFAULT)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--max-wait-seconds', type=float, default=14400,
                    help='monitor lifetime only; does not change mapping budget')
    a = ap.parse_args()
    assert a.max_wait_seconds > 0 and not a.output.exists()
    with (a.study / 'final-audit-waiter.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + a.max_wait_seconds
        while True:
            status = ready(a.study)
            if status == 'ready':
                break
            if status == 'incomplete_checkpoint':
                raise SystemExit('worker ended with incomplete checkpoint; preserve partials, audit before resume')
            if time.monotonic() >= deadline:
                raise SystemExit('finite monitor expired; mapping worker left untouched')
            time.sleep(15)
        command = [sys.executable, str(pathlib.Path(__file__).with_name('audit_dac_frozen_epfl.py')),
                   '--study', str(a.study), '--output', str(a.output)]
        print(json.dumps(dict(stage='full20_evidence_audit', argv=command,
                              mapping_dispatches=0, baseline_runs=0)), flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
