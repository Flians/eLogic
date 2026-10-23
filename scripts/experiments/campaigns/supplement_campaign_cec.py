#!/usr/bin/env python3
"""Checkpointed verification-only supplements, with no repeated closed attempts."""
import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--existing", nargs="*", type=Path, default=[])
    ap.add_argument("--max-jobs", type=int, default=4)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(exist_ok=True)
    driver = Path(__file__).with_name("certify_prepared_chain.py")
    campaign = json.loads(args.campaign.read_text())
    fingerprint = dict(campaign_sha256=sha(args.campaign), verifier_sha256=sha(driver),
                       orchestrator_sha256=sha(Path(__file__)))
    checkpoint = out/"campaign.json"
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(
        fingerprint=fingerprint, jobs=[], scope="Additional CEC only; no mapping run or parameter change.")
    assert state["fingerprint"] == fingerprint
    locks = []
    for path in [out/"runner.lock",
                 ROOT/"results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock",
                 ROOT/"results_v2/small_cut_quota_cases_20261014/runner.lock"]:
        handle = path.open("a")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    processed = set()
    for path in args.existing:
        data = json.loads(path.read_text())
        if data.get("results"):
            processed.add(data["original_report_sha256"])
    for job in state["jobs"]:
        assert sha(Path(job["report"])) == job["report_sha256"]
        processed.add(job["original_report_sha256"])
    count = 0
    for job in campaign["jobs"]:
        if job["report_sha256"] in processed:
            continue
        path = Path(job["report"])
        assert sha(path) == job["report_sha256"]
        data = json.loads(path.read_text())
        assert len(data["results"]) == 1
        row = data["results"][0]
        if row["status"] == "ok" or "mapping_wall_secs" not in row:
            continue
        if row.get("partial_status") or row.get("partial_measured"):
            continue
        folder = path.parent/f"{row['bench']}.{row['method']}"
        if not all((folder/name).is_file() for name in ["mapped.v", "mapped.json", "evaluate.log"]):
            continue
        if "Timeout after 60s" not in (folder/"evaluate.log").read_text():
            continue
        if count >= args.max_jobs:
            state["status"] = "checkpointed"
            checkpoint.write_text(json.dumps(state, indent=2))
            return
        target = out/job["name"]
        assert not target.exists(), "Unrecorded verification partial must be audited separately"
        argv = [sys.executable, str(driver), "--report", str(path), "--output", str(target)]
        with (out/(job["name"]+".log")).open("w") as log:
            result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        report = target/"report.json"
        assert report.exists(), "Verifier rejected metadata; keep diagnostic log and audit"
        state["jobs"].append(dict(name=job["name"], original_report_sha256=sha(path),
                                 argv=argv, exit_code=result.returncode, report=str(report),
                                 report_sha256=sha(report)))
        checkpoint.write_text(json.dumps(state, indent=2))
        count += 1
        print(job["name"], result.returncode, flush=True)
    state["status"] = "attempted_all_eligible"
    checkpoint.write_text(json.dumps(state, indent=2))

if __name__ == "__main__":
    main()
