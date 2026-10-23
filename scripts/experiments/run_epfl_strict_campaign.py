#!/usr/bin/env python3
"""Frozen, per-design strict-profile matrix; adopt only identical prior observations."""
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
    ap.add_argument("--mapper-root", type=Path, required=True)
    ap.add_argument("--subjects", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-jobs", type=int, default=2)
    ap.add_argument("--adopt-area", type=Path)
    ap.add_argument("--adopt-delay", type=Path)
    args = ap.parse_args()
    root, out, subjects = args.mapper_root.resolve(), args.output.resolve(), args.subjects.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    files = sorted((root / "emapper/src").rglob("*.rs")) + sorted((root / "scripts/experiments").glob("*.py"))
    files += [root / "src/bin/bench_rs_egraph_map.rs", root / "target/release/bench_rs_egraph_map",
              root / "target/release/bench_rs_netlist_eval", Path(__file__).resolve(), subjects]
    frozen = {str(p): sha(p) for p in files}
    sp = out / "campaign.json"
    state = json.loads(sp.read_text()) if sp.exists() else {
        "suite": "EPFL20 arithmetic/control", "method": "egraph_bounded_resyn2", "expected_jobs": 80,
        "source_sha256": frozen, "subjects": str(subjects), "jobs": [], "status": "checkpointed"}
    if state["source_sha256"] != frozen:
        raise ValueError("source/binary changed; use an explicitly new matrix")
    metadata = {r["bench"]: r for r in json.loads(subjects.read_text())["results"]}
    adopted = {}
    for objective, path in [("area-delay", args.adopt_area), ("delay-area", args.adopt_delay)]:
        if not path:
            continue
        path = path.resolve()
        d = json.loads(path.read_text())
        if d["binary_sha256"][str(root / "target/release/bench_rs_egraph_map")] != sha(root / "target/release/bench_rs_egraph_map"):
            raise ValueError("prior mapper binary differs")
        for name, digest in d["source_sha256"].items():
            if sha(root / name) != digest:
                raise ValueError("prior harness/source differs")
        for r in d["results"]:
            settings = r["settings"]
            if (r["input_sha256"] != metadata[r["bench"]]["input_sha256"] or
                r["preprocessing"]["sha256"] != metadata[r["bench"]]["preprocessing"]["sha256"] or
                settings["EGM_OBJECTIVE"] != objective or settings.get("EGM_STRICT_BUDGETS") != "1"):
                raise ValueError("prior subject/profile differs")
            if settings["EGM_K"] != "3":
                raise ValueError("prior library input limit differs")
            adopted[(3, objective, r["bench"])] = (path, r)

    def save():
        sp.write_text(json.dumps(state, indent=2) + "\n")

    completed = 0
    for k in [3, 4]:
        for objective in ["area-delay", "delay-area"]:
            for bench in EPFL:
                name = f"{k}in_{objective}_{bench}"
                if any(j["name"] == name for j in state["jobs"]):
                    continue
                key = (k, objective, bench)
                if key in adopted:
                    report, row = adopted[key]
                    state["jobs"].append({"name": name, "bench": bench, "k": k, "objective": objective,
                        "adopted_from": str(report), "report_sha256": sha(report), "status": row["status"],
                        "note": "same source, binary, profile and input; includes retained failed observation"})
                    save()
                    continue
                folder = out / name
                if folder.exists():
                    raise ValueError("unrecorded interrupted directory; preserve and audit: " + str(folder))
                cmd = [sys.executable, str(root / "scripts/experiments/bench_egraph_baselines.py"),
                       "--output", str(folder), "--k", str(k), "--objective", objective, "--preprocess", "resyn2",
                       "--prepared-report", str(subjects), "--methods", "egraph_bounded", "--benches", bench]
                state["current_job"] = {"name": name, "argv": cmd}
                save()
                with (out / f"{name}.log").open("w") as log:
                    r = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                report = folder / "report.json"
                entry = {"name": name, "bench": bench, "k": k, "objective": objective,
                         "argv": cmd, "exit_code": r.returncode, "report": str(report)}
                if report.exists():
                    d = json.loads(report.read_text())
                    if len(d["results"]) != 1 or d["results"][0]["bench"] != bench:
                        raise ValueError("unexpected report coverage")
                    entry.update(report_sha256=sha(report), status=d["results"][0]["status"])
                else:
                    entry.update(status="failed", error="no report; retained command/log")
                state["jobs"].append(entry)
                state.pop("current_job", None)
                save()
                completed += 1
                if completed >= args.max_jobs:
                    state["status"] = "checkpointed"
                    save()
                    print(f"{len(state['jobs'])}/80 recorded, including adopted failures")
                    return
    state["status"] = "evaluated_with_failures" if any(j["status"] != "ok" for j in state["jobs"]) else "evaluated"
    save()


if __name__ == "__main__":
    main()
