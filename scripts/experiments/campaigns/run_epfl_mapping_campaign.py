#!/usr/bin/env python3
"""EPFL checkpoints under the explicit 300-second mapping-only policy."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from run_epfl_baselines import ROOT, HELPERS, EPFL, fingerprint, sha


def jobs():
    for k in (3, 4):
        for bench in EPFL:
            common = ["--k", str(k)]
            yield f"{k}in_core_{bench}", bench, ["bench_egraph_baselines.py", *common,
                "--objective", "area-delay", "--preprocess", "resyn2", "--methods",
                "abc_map_area", "abc_map_delay", "mockturtle_emap_area", "mockturtle_emap_delay"]
            for objective in ("area-delay", "delay-area"):
                yield f"{k}in_egraph_{objective}_{bench}", bench, ["bench_egraph_baselines.py", *common,
                    "--objective", objective, "--preprocess", "resyn2", "--methods", "egraph_bounded"]
                yield f"{k}in_eqmap_{objective}_{bench}", bench, ["bench_eqmap_baseline.py", *common,
                    "--eqmap-source", "/tmp/eqmap-baseline", "--objective", objective]
                yield f"{k}in_esyn_{objective}_{bench}", bench, ["bench_esyn_baseline.py", *common,
                    "--source", "/tmp/emapper-esyn-baseline", "--objective", objective,
                    "--representation", "dag", "--native-seconds", "2", "--native-nodes", "20000",
                    "--native-iterations", "16"]
            yield f"{k}in_emorphic_{bench}", bench, ["bench_emorphic_sa.py", *common,
                "--source", "/tmp/emapper-emorphic-baseline"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--subjects", type=Path, required=True)
    ap.add_argument("--max-jobs", type=int, default=2)
    a = ap.parse_args()
    if a.max_jobs <= 0:
        ap.error("max-jobs must be positive")
    out, subjects = a.output.resolve(), a.subjects.resolve()
    out.mkdir(parents=True, exist_ok=True)
    locks = []
    # Hold legacy locks too: no concurrent timing campaigns or silent resume of
    # a frozen old policy while this policy's bounded batch is running.
    for path in [ROOT / "results_v2/epfl_strict_main_checkpoint_20261013/runner.lock",
                 ROOT / "results_v2/epfl_baseline_checkpoint_20261013/runner.lock", out / "runner.lock"]:
        lock = path.open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    prep = json.loads(subjects.read_text())
    if [r["bench"] for r in prep["results"]] != list(EPFL) or not all(
        r["status"] == "ok" and r["preprocessing"]["cec"] == 1 for r in prep["results"]):
        raise ValueError("requires all20 CEC-passing cached resyn2 subjects")
    frozen = fingerprint(subjects)
    for p in [Path(__file__).resolve(), Path(__file__).with_name("run_with_mapping_budget.py"),
              Path(__file__).with_name("nested_mapping_clock.py"),
              *sorted((ROOT / "emapper/src").rglob("*.rs")), ROOT / "src/bin/bench_rs_egraph_map.rs"]:
        frozen["files"][str(p)] = sha(p)
    path = out / "campaign.json"
    plan = list(jobs())
    if path.exists():
        state = json.loads(path.read_text())
        if state["fingerprint"] != frozen:
            raise ValueError("sources changed: create a new explicit version matrix")
        for j in state["jobs"]:
            for key in ("report", "audit"):
                if j.get(key) and sha(Path(j[key])) != j[key + "_sha256"]:
                    raise ValueError("completed checkpoint changed")
    else:
        state = {"suite": "EPFL20 arithmetic/control", "subjects": str(subjects),
            "mapping_seconds": 300, "policy": "mapping only; CEC/resyn2/adapters/metrics excluded",
            "existing_process_limits_preserved": True, "baseline_algorithms_changed": False,
            "fingerprint": frozen, "expected_jobs": len(plan), "jobs": [], "status": "checkpointed"}

    def save():
        tmp = out / "campaign.json.tmp"
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(path)

    count = 0
    for name, bench, options in plan:
        if any(j["name"] == name for j in state["jobs"]):
            continue
        folder, audit = out / name, out / f"{name}.audit.json"
        log_path = out / f"{name}.log"
        if any(p.exists() for p in (folder, audit, log_path)):
            raise ValueError(f"interrupted artifacts for {name}; preserve and audit before retry")
        argv = [sys.executable, str(Path(__file__).with_name("run_with_mapping_budget.py")),
            "--mapping-seconds", "300", "--driver", str(HELPERS / options[0]), "--audit", str(audit),
            "--", *options[1:], "--output", str(folder), "--prepared-report", str(subjects),
            "--suite", "epfl", "--benches", bench]
        state.update(status="running", current_job={"name": name, "argv": argv})
        save()
        with log_path.open("w") as log:
            result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        entry = {"name": name, "bench": bench, "argv": argv, "exit_code": result.returncode}
        for key, p in [("report", folder / "report.json"), ("audit", audit)]:
            if p.exists():
                entry.update({key: str(p), key + "_sha256": sha(p)})
        if entry.get("report"):
            rows = json.loads(Path(entry["report"]).read_text())["results"]
            entry.update(ok=sum(r["status"] == "ok" for r in rows),
                         failed=sum(r["status"] != "ok" for r in rows),
                         missing_benches=sorted({bench} - {r["bench"] for r in rows}))
        else:
            entry.update(error="no report; retained log/audit", missing_benches=[bench])
        state["jobs"].append(entry)
        state.pop("current_job", None)
        state["status"] = "checkpointed" if len(state["jobs"]) < len(plan) else "attempted"
        save()
        print(name, entry.get("ok", 0), entry.get("failed", 0), flush=True)
        count += 1
        if count >= a.max_jobs:
            break


if __name__ == "__main__":
    main()
