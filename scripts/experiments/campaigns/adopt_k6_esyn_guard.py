"""Reuse only an objective-independent input-conversion failure, never a mapped result.

The frozen original driver reads the objective only for report metadata and
post-native selection. Its DAG-expression guard executes before native search.
Keep the source execution, failed reports, argv and hashes; do not rerun or
change a baseline. Refuse all other failures, successful results and partials.
"""
import argparse
import fcntl
import json
from pathlib import Path

from run_epfl_baselines import ROOT, fingerprint, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--subjects", type=Path, required=True)
    ap.add_argument("--bench", required=True)
    a = ap.parse_args()
    cp = a.campaign.resolve()
    out = cp.parent
    locks = []
    for p in [out / "runner.lock", ROOT / "results_v2/independent_components_20261019/global.lock",
              ROOT / "results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock"]:
        stream = p.open("a")
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(stream)
    state = json.loads(cp.read_text())
    assert state["status"] != "running" and "current_job" not in state
    assert state["configuration"]["k"] == 6 and state["configuration"]["rounds"] == 1
    source_name = f"r0_6in_esyn_area-delay_{a.bench}"
    name = f"r0_6in_esyn_delay-area_{a.bench}"
    assert not any(j["name"] == name for j in state["jobs"])
    source = next(j for j in state["jobs"] if j["name"] == source_name)
    assert all(sha(Path(p)) == h for p, h in source["sha256"].items())
    assert all(sha(Path(p)) == h for p, h in state["fingerprint"]["files"].items())
    assert fingerprint(a.subjects.resolve())["native"] == state["fingerprint"]["native"]
    driver = ROOT / "scripts/experiments/bench_esyn_baseline.py"
    # Pin the audited driver, rather than accepting arbitrary code with a
    # similarly worded exception. All objective reads precede metadata or
    # follow the native call; the expression converter has no objective input.
    assert sha(driver) == state["fingerprint"]["files"][str(driver)]
    import ast
    tree = ast.parse(driver.read_text())
    reads = sorted(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                   and isinstance(n.value, ast.Name) and n.value.id == "a" and n.attr == "objective")
    assert reads == [190, 273], "re-audit objective dependencies after driver changes"
    report = json.loads(Path(source["report"]).read_text())
    assert len(report["results"]) == 1
    row = report["results"][0]
    assert row["status"] == "failed" and row["error"] == "input expression exceeds resource guard"
    assert not list((out / source_name).rglob("native.log"))
    audit_path = Path(source["argv"][source["argv"].index("--audit") + 1])
    audit = json.loads(audit_path.read_text())
    assert all(s["mapping_wall_secs"] == 0 and not s["exhausted"] for s in audit["states"].values())
    assert not any(c["mapping_charged"] for s in audit["states"].values() for c in s["commands"])
    folder, new_audit = out / name, out / (name + ".audit.json")
    assert not folder.exists() and not new_audit.exists() and not (out / (name + ".log")).exists()
    argv = list(source["argv"])
    for flag, value in [("--objective", "delay-area"), ("--output", str(folder)), ("--audit", str(new_audit))]:
        argv[argv.index(flag) + 1] = value
    proof = dict(purpose="objective-independent failure adoption; not a new execution or runtime sample",
                 source_job=source_name, source_artifact_sha256=source["sha256"],
                 driver_sha256=sha(driver), objective_reads=reads,
                 proof="DAG converter has no objective input; guard failed before native; only later selection differs",
                 declared_target_argv=argv, baseline_algorithms_changed=False,
                 mapping_rerun=False, native_rerun=False, mapping_wall_secs=0,
                 original_status_retained=dict(status=row["status"], error=row["error"]),
                 adoption_tool_sha256=sha(Path(__file__)))
    folder.mkdir()
    proof_path = folder / "failure_adoption.json"
    proof_path.write_text(json.dumps(proof, indent=2) + "\n")
    report.update(objective="delay-area", adoption=proof, target_elapsed_not_measured=True)
    row.update(adopted_failure=True, source_execution_objective="area-delay")
    report_path = folder / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    audit.update(adoption=proof, target_elapsed_not_measured=True)
    new_audit.write_text(json.dumps(audit, indent=2) + "\n")
    artifacts = {**source["sha256"], **{str(p): sha(p) for p in [proof_path, report_path, new_audit]}}
    state["jobs"].append(dict(name=name, bench=a.bench, profile=None, argv=argv,
        exit_code=source["exit_code"], report=str(report_path), ok=0, failed=1,
        executed=False, adopted_source_job=source_name, adoption_proof=str(proof_path), sha256=artifacts))
    temporary = cp.with_suffix(".pending")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(cp)
    print(name, "adopted as failure; no mapping or native execution")


if __name__ == "__main__":
    main()
