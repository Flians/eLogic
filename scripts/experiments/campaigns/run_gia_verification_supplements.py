"""Recheck completed baseline netlists only; no baseline mapping is executed."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from scaled_verification_cache import sha

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--measure-binary", required=True, type=Path)
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument("--priority-method",default="mockturtle_emap_area_resyn2")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(exist_ok=True)
    plan = json.loads(args.plan.read_text())
    verifier = ROOT / "scripts/experiments/campaigns/fast_original_cec_gia.py"
    fp = {str(p.resolve()): sha(p) for p in [args.plan, Path(__file__), verifier, args.measure_binary,
          ROOT / "emapper/src/netlist_metrics.rs", ROOT / "emapper/src/techlib.rs",
          ROOT / "scripts/experiments/campaigns/fast_original_cec.py", Path("/usr/bin/abc"), Path("/usr/local/bin/yosys")]}
    for case in plan["cases"]:
        for p in [*case["argv"], str(Path(case["argv"][1]).parent/"cells.v"), case["raw_verilog"], case["original_report"], case["original_audit"]]:
            fp[str(Path(p).resolve())] = sha(p)
    checkpoint = out / "campaign.json"
    campaign = json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(
        fingerprint=fp, expected_jobs=len(plan["cases"]), jobs=[], mapping_rerun=False,
        baseline_algorithms_changed=False, verification_guard_secs=600, checker="ABC GIA &cec",
        priority_method=args.priority_method, status="running")
    assert campaign["fingerprint"] == fp and campaign["priority_method"] == args.priority_method
    locks = []
    for path in [out / "runner.lock", ROOT / "results_v2/independent_components_20261019/global.lock"]:
        stream = path.open("a")
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(stream)
    for job in campaign["jobs"]:
        assert all(sha(p) == h for p, h in job["sha256"].items())

    def save():
        pending = checkpoint.with_suffix(".pending")
        pending.write_text(json.dumps(campaign, indent=2))
        pending.replace(checkpoint)

    completed, done = 0, {j["name"] for j in campaign["jobs"]}
    save()
    for case in sorted(plan["cases"], key=lambda c:c["method"] != args.priority_method):
        name = f'{case["k"]}in_{case["bench"]}_{case["method"]}'
        if name in done:
            continue
        if completed >= args.max_jobs:
            return
        assert all(sha(p) == h for p, h in fp.items())
        log = out / (name + ".stdout")
        assert not log.exists(), "partial output must be preserved and audited"
        proof_dir = out / (name + ".proof")
        assert not proof_dir.exists(), "unrecorded proof partial must be audited"
        argv = [str(verifier), "--proof-dir", str(proof_dir), *case["argv"][1:]]
        env = dict(os.environ, EGM_MEASURE_BINARY=str(args.measure_binary.resolve()))
        campaign["in_progress"] = dict(name=name, argv=argv)
        save()
        started = time.monotonic()
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, env=env, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=600)
            timeout = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            timeout = True
        log.write_text(stdout)
        error = out / (name + ".stderr")
        error.write_text(stderr)
        records = [json.loads(line) for line in stdout.splitlines() if line.startswith('{"area":')]
        good = not timeout and process.returncode == 0 and len(records) == 1 and records[0]["cec"] == 1
        artifacts = [log, error, *[p for p in proof_dir.rglob("*") if p.is_file()]]
        campaign["jobs"].append(dict(name=name, original=case, argv=argv,
            exit_code=process.returncode, timeout=timeout, wall_secs=time.monotonic() - started, proof_dir=str(proof_dir),
            status="ok" if good else "failed", measured=records[0] if good else None,
            old_status_retained=case["old_status"], mapping_rerun=False,
            sha256={str(p): sha(p) for p in artifacts}))
        campaign.pop("in_progress")
        save()
        completed += 1
        print(name, "verified" if good else "failed", flush=True)
    campaign["status"] = "attempted"
    save()


if __name__ == "__main__":
    main()
