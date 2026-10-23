#!/usr/bin/env python3
"""Sequential, resumable EPFL campaign; record failures and never silently omit coverage."""
import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from benchmark_suites import EPFL


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--subjects", type=Path, required=True)
    ap.add_argument("--max-jobs", type=int, help="Finish this many checkpoint jobs per invocation, then yield")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    subjects = args.subjects.resolve()
    source = json.loads(subjects.read_text())
    if [r["bench"] for r in source["results"]] != list(EPFL):
        raise ValueError("subject preparation has not finished all 20 cases")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    jobs = []
    for k in [3, 4]:
        for objective in ["area-delay", "delay-area"]:
            methods = ["abc_map_area", "abc_map_delay", "mockturtle_emap_area", "mockturtle_emap_delay", "egraph"]
            if objective == "delay-area":
                methods += ["egraph_library_slack", "egraph_timing_cap"]
            jobs.append((f"{k}in_core_{objective}", ["bench_egraph_baselines.py", "--k", str(k),
                        "--objective", objective, "--preprocess", "resyn2", "--methods", *methods]))
    for k in [3, 4]:
        for objective in ["area-delay", "delay-area"]:
            jobs.append((f"{k}in_eqmap_{objective}", ["bench_eqmap_baseline.py", "--k", str(k),
                         "--eqmap-source", "/tmp/eqmap-baseline", "--objective", objective]))
        jobs.append((f"{k}in_esyn", ["bench_esyn_baseline.py", "--k", str(k), "--source", "/tmp/emapper-esyn-baseline",
                     "--representation", "dag", "--native-seconds", "2", "--native-nodes", "20000", "--native-iterations", "16"]))
        jobs.append((f"{k}in_emorphic", ["bench_emorphic_sa.py", "--k", str(k), "--source", "/tmp/emapper-emorphic-baseline"]))
    jobs = [(f"{name}_{bench}", options, bench) for name, options in jobs for bench in EPFL]
    sources = [root / "scripts/experiments" / name for name in [
        "run_epfl_campaign.py", "benchmark_suites.py", "shared_subject.py", "bench_egraph_baselines.py",
        "bench_eqmap_baseline.py", "bench_esyn_baseline.py", "bench_emorphic_sa.py", "bench_emorphic_baseline.py", "port_identifiers.py"]]
    sources += [root / "emapper/src/egraph_map.rs", root / "target/release/bench_rs_egraph_map"]
    frozen = {str(p): sha(p) for p in sources}
    state_path = out / "campaign.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["source_sha256"] != frozen or state["subjects_sha256"] != sha(subjects):
            raise ValueError("campaign sources changed; create a new campaign rather than mix configurations")
    else:
        state = {"suite": "epfl", "expected_benches": list(EPFL), "preprocess": "resyn2", "subjects": str(subjects),
                 "subjects_sha256": sha(subjects), "source_sha256": frozen, "jobs": [], "status": "running"}

    def save():
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    completed = 0
    for name, options, bench in jobs:
        if any(j["name"] == name for j in state["jobs"]):
            continue
        folder = out / name
        if folder.exists():
            raise ValueError(f"interrupted unrecorded output {folder}; preserve and audit before retry")
        command = [sys.executable, str(root / "scripts/experiments" / options[0]), *options[1:],
                   "--output", str(folder), "--prepared-report", str(subjects), "--suite", "epfl", "--benches", bench]
        state["current_job"] = {"name": name, "argv": command}
        save()
        with (out / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        entry = {"name": name, "argv": command, "exit_code": result.returncode}
        report = folder / "report.json"
        if report.exists():
            d = json.loads(report.read_text())
            rows = d["results"]
            entry.update(report_sha256=sha(report), ok=sum(r["status"] == "ok" for r in rows),
                         failed=sum(r["status"] != "ok" for r in rows),
                         expected_benches=[bench], missing_benches=sorted({bench} - {r["bench"] for r in rows}))
        else:
            entry.update(error="no report; inspect retained log", expected_benches=[bench], missing_benches=[bench])
        state["jobs"].append(entry)
        save()
        print(name, entry, flush=True)
        completed += 1
        if args.max_jobs and completed >= args.max_jobs:
            break
    state.pop("current_job", None)
    state["expected_jobs"] = len(jobs)
    state["status"] = "checkpointed" if len(state["jobs"]) < len(jobs) else "evaluated" if all(j["exit_code"] == 0 and not j["missing_benches"] for j in state["jobs"]) else "incomplete"
    state["note"] = "Evaluated means all jobs attempted; failures and unverified covers never count as wins. Baseline collection, dominance and paper refresh remain separate."
    save()


if __name__ == "__main__":
    main()
