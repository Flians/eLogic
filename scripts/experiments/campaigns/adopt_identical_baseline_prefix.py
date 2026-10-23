"""Adopt every unseen observation from an identical-dependency baseline subset.

Only scheduling/design coverage may differ. Never replace an existing point,
choose successful outcomes, change a budget or promote a different native build.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path

from run_epfl_baselines import ROOT, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    a = ap.parse_args()
    primary, source = a.primary.resolve(), a.source.resolve()
    locks = []
    for p in [primary.parent / "runner.lock", source.parent / "runner.lock",
              ROOT / "results_v2/independent_components_20261019/global.lock",
              ROOT / "results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock"]:
        f = p.open("a")
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(f)
    p, s = json.loads(primary.read_text()), json.loads(source.read_text())
    assert all(c["status"] != "running" and "current_job" not in c for c in [p, s])
    assert p["fingerprint"] == s["fingerprint"], "actual dependency fingerprints differ"
    pc, sc = copy.deepcopy(p["configuration"]), copy.deepcopy(s["configuration"])
    pd, sd = pc.pop("designs"), sc.pop("designs")
    assert pc == sc and pc["kind"] == "baseline" and set(sd) <= set(pd)
    assert all(sha(Path(path)) == h for path, h in p["fingerprint"]["files"].items())
    seen = {j["name"]: j for j in p["jobs"]}
    fresh = []
    for job in s["jobs"]:
        assert job["bench"] in sd
        assert all(sha(Path(path)) == h for path, h in job["sha256"].items())
        if job["name"] in seen:
            old = seen[job["name"]]
            assert old["argv"] == job["argv"] and old["report"] == job["report"]
            assert all(old["sha256"].get(path) == h for path, h in job["sha256"].items())
            continue
        fresh.append(copy.deepcopy(job))
    assert fresh, "no new observations"
    folder = primary.parent / "adoptions"
    folder.mkdir(exist_ok=True)
    stem = f"{source.parent.name}-prefix{len(s['jobs'])}"
    snapshot, proof = folder / (stem + ".source.json"), folder / (stem + ".proof.json")
    assert not snapshot.exists() and not proof.exists()
    snapshot.write_text(source.read_text())
    proof.write_text(json.dumps(dict(source=str(source), source_snapshot=str(snapshot),
        source_snapshot_sha256=sha(snapshot), primary_before_sha256=sha(primary),
        dependency_fingerprints_identical=True, only_configuration_difference="designs / scheduling",
        adopted_jobs=[j["name"] for j in fresh], adopted_all_unseen_results_including_failures=True,
        baseline_algorithms_changed=False, budgets_changed=False, mapping_rerun=False,
        adoption_tool_sha256=sha(Path(__file__))), indent=2) + "\n")
    for job in fresh:
        job.update(adopted_from=str(source), identical_dependency_adoption=str(proof))
        job["sha256"].update({str(path): sha(path) for path in [snapshot, proof]})
        p["jobs"].append(job)
    assert len(p["jobs"]) <= p["expected_jobs"]
    p["status"] = "attempted" if len(p["jobs"]) == p["expected_jobs"] else "checkpointed"
    pending = primary.with_suffix(".pending")
    pending.write_text(json.dumps(p, indent=2) + "\n")
    pending.replace(primary)
    print("adopted", len(fresh), "observations; canonical", len(p["jobs"]), "/", p["expected_jobs"])


if __name__ == "__main__":
    main()
