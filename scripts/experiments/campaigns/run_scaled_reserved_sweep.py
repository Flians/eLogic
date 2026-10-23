"""Checkpointed sequential EPFL rho sweep with frozen algorithm and proof cache."""
import argparse
import fcntl
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from run_epfl_baselines import EPFL
from scaled_verification_cache import sha

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--research-root", required=True, type=Path)
    parser.add_argument("--verification-cache", required=True, type=Path)
    parser.add_argument("--measure-binary",required=True,type=Path)
    parser.add_argument("--max-jobs", type=int, default=120)
    parser.add_argument("--ratios", nargs="+", default=["1/2", "1", "2"])
    parser.add_argument("--designs", nargs="+", default=EPFL)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    ratios = [str(Fraction(r)) for r in args.ratios]
    assert len(set(ratios)) == len(ratios) and all(Fraction(r) >= 0 for r in ratios)
    assert len(set(args.designs)) == len(args.designs) and set(args.designs) <= set(EPFL)
    output = args.output.resolve()
    output.mkdir(exist_ok=True)
    template = json.loads(Path("/tmp/emapper-independent-full-20261019/campaign.json").read_text())
    fingerprint = dict(template["fingerprint"])
    assert all(sha(p) == h for p, h in fingerprint.items()), "old actual dependencies changed"
    sources = [Path(__file__).resolve(), args.native.resolve(), args.verification_cache.resolve(),
               args.research_root / "Cargo.toml", args.research_root / "Cargo.lock",
               args.research_root / "main.rs", args.research_root / "measure.rs", args.measure_binary.resolve(), args.research_root / "emapper/Cargo.toml"]
    sources += list((args.research_root / "emapper/src").glob("*.rs"))
    sources += [HERE / name for name in ["bench_scaled_sweep.py", "egraphmap_scaled.py",
                "run_scaled_mapping_budget.py", "scaled_verification_cache.py",
                "input_capacity_proportional.py", "input_capacity_policy_v2.py", "source_cover.py", "fast_original_cec.py"]]
    sources += [ROOT / "scripts/experiments/prototypes/region_stitch.py",
                ROOT / "scripts/experiments/analyze_epfl_structural_budget.py"]
    fingerprint.update({str(p.resolve()): sha(p) for p in sources})
    assert sha(args.research_root/"emapper/src/netlist_metrics.rs")==sha(ROOT/"emapper/src/netlist_metrics.rs")
    configuration = dict(ratios=ratios, designs=list(args.designs), rounds=args.rounds,return_reserve_secs=0.25,
                         native=str(args.native.resolve()), verification_cache=str(args.verification_cache.resolve()),
                         mapping_secs=300, native_guard_secs=60, recovery_secs=2,
                         verification_guard_secs=600, source_first=True, measure_binary=str(args.measure_binary.resolve()),
                         compact_source=True, mandatory_helper_cap=2048, helper_fill=False,
                         cut_dependencies=True, streamed_matches=True, root_closure_retention=True,
                         all_failures_retained=True, baseline_algorithms_changed=False)
    checkpoint = output / "campaign.json"
    campaign = json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(
        fingerprint=fingerprint, configuration=configuration,
        expected_jobs=2 * len(args.designs) * len(ratios) * args.rounds,
        jobs=[], status="running")
    assert campaign["fingerprint"] == fingerprint and campaign["configuration"] == configuration
    locks = []
    for path in [output / "runner.lock", ROOT / "results_v2/independent_components_20261019/global.lock",
                 ROOT / "results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock",
                 ROOT / "results_v2/small_cut_quota_cases_20261014/runner.lock"]:
        stream = path.open("a")
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(stream)
    for job in campaign["jobs"]:
        assert all(sha(p) == h for p, h in job["sha256"].items()), job["name"]

    def save():
        pending = checkpoint.with_suffix(".pending")
        pending.write_text(json.dumps(campaign, indent=2))
        pending.replace(checkpoint)

    done = {job["name"] for job in campaign["jobs"]}
    completed = 0
    save()
    for repetition in range(args.rounds):
        for ordinal, bench in enumerate(args.designs):
            offset = (ordinal + repetition) % len(ratios)
            order = ratios[offset:] + ratios[:offset]
            for ratio in order:
                for k in ([3, 4] if (ordinal + repetition) % 2 == 0 else [4, 3]):
                    name = f"rep{repetition}_r{ratio.replace('/', 'd')}_{k}in_{bench}"
                    if name in done:
                        continue
                    if completed >= args.max_jobs:
                        save()
                        return
                    assert all(sha(p) == h for p, h in fingerprint.items()), "frozen source drift"
                    folder, audit = output / name, output / (name + ".audit.json")
                    assert not folder.exists() and not audit.exists(), "unrecorded partial requires audit"
                    original = next(j for j in template["jobs"] if j["bench"] == bench and j["k"] == k and j["profile"] == "full")
                    argv = list(original["argv"])
                    argv[0], argv[1] = sys.executable, str(HERE / "run_scaled_mapping_budget.py")
                    for flag, value in [("--driver", HERE / "bench_scaled_sweep.py"),
                                        ("--audit", audit), ("--output", folder)]:
                        argv[argv.index(flag) + 1] = str(value)
                    argv[argv.index("--return-reserve")+1]="0.25"
                    argv += ["--search-ratio", ratio, "--pool-extra", "0", "--native-optimizer", str(args.native.resolve()),
                             "--verification-cache", str(args.verification_cache.resolve()), "--verification-seconds", "600", "--measure-binary",str(args.measure_binary.resolve())]
                    campaign["in_progress"] = dict(name=name, argv=argv)
                    save()
                    with (output / (name + ".log")).open("w") as log:
                        result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    artifacts = [p for p in folder.rglob("*") if p.is_file()]
                    artifacts += [audit, output / (name + ".log")]
                    campaign["jobs"].append(dict(name=name, bench=bench, k=k, ratio=ratio,
                        repetition=repetition, profile="scaled", argv=argv, exit_code=result.returncode,
                        report=str(folder / "report.json"), audit=str(audit),
                        sha256={str(p): sha(p) for p in artifacts if p.is_file()}))
                    campaign.pop("in_progress")
                    completed += 1
                    save()
                    print(name, result.returncode, flush=True)
    campaign["status"] = "attempted"
    save()


if __name__ == "__main__":
    main()
