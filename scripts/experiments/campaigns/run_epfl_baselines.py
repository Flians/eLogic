#!/usr/bin/env python3
"""Frozen baseline-only EPFL checkpoints; never rerun preprocessing or select winners."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HELPERS = ROOT / "scripts/experiments"
sys.path.insert(0, str(HELPERS))
from benchmark_suites import EPFL


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan():
    for k in (3, 4):
        for bench in EPFL:
            common = ["--k", str(k)]
            yield f"{k}in_core_{bench}", bench, ["bench_egraph_baselines.py", *common,
                "--preprocess", "resyn2", "--objective", "area-delay", "--methods",
                "abc_map_area", "abc_map_delay", "mockturtle_emap_area", "mockturtle_emap_delay"]
            for objective in ("area-delay", "delay-area"):
                yield f"{k}in_eqmap_{objective}_{bench}", bench, ["bench_eqmap_baseline.py", *common,
                    "--eqmap-source", "/tmp/eqmap-baseline", "--objective", objective]
                yield f"{k}in_esyn_{objective}_{bench}", bench, ["bench_esyn_baseline.py", *common,
                    "--source", "/tmp/emapper-esyn-baseline", "--objective", objective,
                    "--representation", "dag", "--native-seconds", "2", "--native-nodes", "20000",
                    "--native-iterations", "16"]
            yield f"{k}in_emorphic_{bench}", bench, ["bench_emorphic_sa.py", *common,
                "--source", "/tmp/emapper-emorphic-baseline"]


def fingerprint(subjects):
    paths = [Path(__file__).resolve(), subjects, ROOT / "build/baseline_mapper",
             ROOT / "target/release/bench_rs_netlist_eval", ROOT / "target/release/bench_rs_egraph_map",
             ROOT / "tools/mockturtle/experiments/cell_libraries/asap7.genlib",
             Path("/usr/bin/abc"), Path("/usr/local/bin/yosys")]
    paths += list(HELPERS.glob("*.py"))
    native = {}
    for source, binaries in [
        ("/tmp/eqmap-baseline", ["target/release/eqmap_fpga"]),
        ("/tmp/emapper-esyn-baseline", ["e-rewriter/target/release/e-rewriter"]),
        ("/tmp/emapper-emorphic-baseline", ["e-rewriter/target/release/e-rewriter",
            "extraction-gym/target/release/extraction-gym", "grpc_communicator/proto/service.proto"]),
    ]:
        src = Path(source)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src, text=True).strip()
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=src).decode().split("\0")
        native[source] = {"commit": commit, "tracked_sha256": {
            name: sha(src / name) for name in tracked if name and (src / name).is_file()}}
        paths.extend(src / name for name in binaries)
    return {"files": {str(p): sha(p) for p in sorted(set(paths))}, "native": native}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--subjects", type=Path, required=True)
    ap.add_argument("--max-jobs", type=int, default=2)
    a = ap.parse_args()
    if a.max_jobs < 1:
        ap.error("max-jobs must be positive")
    out, subjects = a.output.resolve(), a.subjects.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepared = json.loads(subjects.read_text())
        if [r["bench"] for r in prepared["results"]] != list(EPFL):
            raise ValueError("requires the complete, ordered EPFL20 prepared report")
        frozen, jobs = fingerprint(subjects), list(plan())
        path = out / "campaign.json"
        if path.exists():
            state = json.loads(path.read_text())
            if state["fingerprint"] != frozen:
                raise ValueError("frozen sources changed; use a new explicit version matrix")
            for entry in state["jobs"]:
                if entry.get("report") and sha(Path(entry["report"])) != entry["report_sha256"]:
                    raise ValueError("checkpoint report changed")
        else:
            state = {"suite": "EPFL20 arithmetic/control", "subjects": str(subjects),
                "preprocess": "one cached resyn2; never rerun", "fingerprint": frozen,
                "expected_jobs": len(jobs), "expected_benches": list(EPFL), "jobs": [],
                "note": "Native budgets differ and remain reported; attempted is not CEC success or dominance."}

        def save():
            tmp = out / "campaign.json.tmp"
            tmp.write_text(json.dumps(state, indent=2) + "\n")
            tmp.replace(path)

        count = 0
        for name, bench, opts in jobs:
            if any(j["name"] == name for j in state["jobs"]):
                continue
            folder = out / name
            if folder.exists() or (out / f"{name}.log").exists():
                raise ValueError(f"unrecorded interrupted job {name}: preserve and audit before retry")
            argv = [sys.executable, str(HELPERS / opts[0]), *opts[1:], "--output", str(folder),
                "--prepared-report", str(subjects), "--suite", "epfl", "--benches", bench]
            state.update(status="running", current_job={"name": name, "argv": argv})
            save()
            # Driver timeouts and process-group cleanup retain their original budgets.
            env = {k: v for k, v in os.environ.items() if not k.startswith(("EGM_", "ESYN_"))}
            env.update(ABC="/usr/bin/abc", EGM_YOSYS="/usr/local/bin/yosys")
            with (out / f"{name}.log").open("w") as log:
                run = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            entry = {"name": name, "bench": bench, "argv": argv, "exit_code": run.returncode}
            report = folder / "report.json"
            if report.exists():
                rows = json.loads(report.read_text())["results"]
                entry.update(report=str(report), report_sha256=sha(report),
                    ok=sum(r["status"] == "ok" for r in rows),
                    failed=sum(r["status"] != "ok" for r in rows),
                    missing_benches=sorted({bench} - {r["bench"] for r in rows}))
            else:
                entry.update(error="no report; retained process log", missing_benches=[bench])
            state["jobs"].append(entry)
            state.pop("current_job", None)
            state["status"] = "checkpointed" if len(state["jobs"]) < len(jobs) else "attempted"
            save()
            print(name, "ok", entry.get("ok", 0), "failed", entry.get("failed", 0), flush=True)
            count += 1
            if count >= a.max_jobs:
                break


if __name__ == "__main__":
    main()
