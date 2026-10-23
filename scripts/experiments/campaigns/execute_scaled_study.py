"""Sequential finite study. Wait for verification; never overlap mapping jobs."""
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from scaled_ratio_selection import select
from scaled_verification_cache import sha

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results_v2/proportional_sweep_20261019"
BASELINES = Path("/tmp/emapper-baseline-verification-20261019")
RESEARCH = Path("/tmp/emapper-proportional-research-v3-20261020")
NATIVE = ROOT / "target/release/bench_rs_egraph_scale_v3"
MEASURE = ROOT / "target/release/bench_rs_scale_measure"
CACHE = EVIDENCE / "verification_cache.json"
PROBE = Path("/tmp/emapper-proportional-v3-div-probe-20261020")
FULL = Path("/tmp/emapper-proportional-full-reserved-v3-20261020")
REPEATS = Path("/tmp/emapper-proportional-target-repeats-v3-20261020")
LARGE = Path("/tmp/emapper-proportional-large-repeats-v3-20261020")


def main():
    state = dict(status="waiting", stage="baseline-verification-only",
                 primary_full=str(FULL), target_repeats=str(REPEATS), large_repeats=str(LARGE),
                 native=str(NATIVE), research=str(RESEARCH), budgets=dict(mapping=300, native=60, recovery=2),
                 baseline_algorithms_changed=False, paper_updated=False, finished_stages=[])
    status_path = EVIDENCE / "execution.json"

    def save():
        status_path.write_text(json.dumps(state, indent=2))

    save()
    previous = None
    while True:
        checkpoint = json.loads((BASELINES / "campaign.json").read_text())
        observed = (len(checkpoint["jobs"]), checkpoint.get("in_progress", {}).get("name"))
        if observed != previous:
            print("verification-only progress", observed, flush=True)
            previous = observed
        if checkpoint["status"] == "attempted" and len(checkpoint["jobs"]) == 8:
            stream = (ROOT / "results_v2/independent_components_20261019/global.lock").open("a")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                stream.close()
                break
            except BlockingIOError:
                stream.close()
        time.sleep(3)

    def run(name, argv):
        state.update(status="running", stage=name, argv=argv)
        save()
        with (EVIDENCE / (name + ".log")).open("a") as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line.rstrip(), flush=True)
            code = process.wait()
        if code:
            state.update(status="stopped-on-error", exit_code=code)
            save()
            raise RuntimeError(name + " failed; outputs preserved")
        state["finished_stages"].append(name)
        save()

    runner = [sys.executable, str(HERE / "run_scaled_reserved_sweep.py"),
              "--native", str(NATIVE), "--research-root", str(RESEARCH),
              "--measure-binary", str(MEASURE), "--verification-cache", str(CACHE)]
    checker = [sys.executable, str(HERE / "check_scaled_sweep.py")]
    run("v3-div-preflight", runner + ["--output", str(PROBE), "--designs", "div", "--ratios", "1", "--max-jobs", "2"])
    run("v3-div-preflight-audit", checker + [str(PROBE / "campaign.json")])
    probe = json.loads((PROBE / "validation.json").read_text())
    if not (probe["complete"] and probe["counts"]["1"]["optimized_selected"] == 2):
        state.update(status="stopped-on-budget-or-quality-preflight", reason="fresh in-window availability did not produce two checked optimized div covers")
        save()
        return
    fingerprint = json.loads((PROBE / "campaign.json").read_text())["fingerprint"]
    frozen = EVIDENCE / "frozen_v3_source"
    frozen.mkdir(exist_ok=True)
    manifest = []
    for index, (original, expected) in enumerate(fingerprint.items()):
        path = Path(original)
        assert sha(path) == expected
        target = frozen / f"{index:03d}_{path.name}"
        if not target.exists():
            target.write_bytes(path.read_bytes())
        assert sha(target) == expected
        manifest.append(dict(original=original, snapshot=str(target), sha256=expected))
    (frozen / "index.json").write_text(json.dumps(manifest, indent=2))
    run("full120", runner + ["--output", str(FULL), "--max-jobs", "120"])
    run("full120-audit", checker + [str(FULL / "campaign.json")])
    full = json.loads((FULL / "validation.json").read_text())
    assert full["complete"] and full["attempts"] == 120
    chosen, summary, common = select(full["rows"], ["1/2", "1", "2"])
    selection = dict(selected_ratio=chosen, summary=summary, common_verified=len(common),
                     single_global_ratio=True, repetitions_not_used_to_replace_primary_points=True)
    (FULL / "selected_ratio.json").write_text(json.dumps(selection, indent=2))
    (EVIDENCE / "selected_ratio.json").write_text(json.dumps(selection, indent=2))
    state["selected_ratio"] = chosen
    save()
    run("target-repeats54", runner + ["--output", str(REPEATS), "--designs", "sqrt", "sin", "bar", "--rounds", "3", "--max-jobs", "54"])
    run("target-repeats54-audit", checker + [str(REPEATS / "campaign.json")])
    run("large-repeats18", runner + ["--output", str(LARGE), "--designs", "div", "hyp", "mem_ctrl", "--ratios", chosen, "--rounds", "3", "--max-jobs", "18"])
    run("large-repeats18-audit", checker + [str(LARGE / "campaign.json")])
    run("qualification", [sys.executable, str(HERE / "finalize_scaled_sweep.py"), "--full", str(FULL),
         "--repeats", str(REPEATS), "--large-repeats", str(LARGE),
         "--baseline-supplements", str(BASELINES), "--output", str(EVIDENCE / "qualification")])
    state.update(status="study-finished-paper-pending", stage="paper-and-persistence", paper_updated=False)
    save()
    print("Complete measured study; paper update and permanent archives remain.", flush=True)


if __name__ == "__main__":
    main()
