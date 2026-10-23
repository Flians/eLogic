"""Full-care quotient single factor; original node/time/pool budgets unchanged."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from run_epfl_baselines import ROOT, HELPERS, EPFL, fingerprint, sha, plan

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["mapper"], required=True)
    parser.add_argument("--mode",choices=["quotient"],required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--frontend-root",type=Path,required=True)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--verification-cache", type=Path)
    parser.add_argument("--designs", nargs="+", default=EPFL)
    parser.add_argument("--max-jobs", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--k",type=int,choices=[3,4,6],default=6)
    parser.add_argument("--profiles",type=int,nargs="+",choices=[0,1],default=[0,1])
    args = parser.parse_args()
    args.designs = list(args.designs)
    assert set(args.designs) <= set(EPFL) and len(set(args.designs)) == len(args.designs)
    assert args.max_jobs > 0 and args.rounds > 0
    assert len(set(args.profiles))==len(args.profiles)
    subjects, output = args.subjects.resolve(), args.output.resolve()
    frontend_root=args.frontend_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    locks = []
    for path in [output / "runner.lock", ROOT / "results_v2/independent_components_20261019/global.lock",
                 ROOT / "results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock"]:
        stream = path.open("a")
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(stream)
    preparation = json.loads(subjects.read_text())
    assert [row["bench"] for row in preparation["results"]] == list(EPFL)
    assert all(row["status"] == "ok" and row["preprocessing"]["cec"] == 1 for row in preparation["results"])
    files = [Path(__file__).resolve(), HERE / "run_single_output_budget.py", HERE / "run_k6_quotient_budget.py",
             HERE / "run_with_mapping_budget.py", HERE / "nested_mapping_clock.py", subjects,frontend_root/"scripts/experiments/campaigns/egraphmap_scaled.py",frontend_root/"scripts/experiments/campaigns/source_cover.py"]
    for row in preparation["results"]:
        # The shared-subject helper also checks these bytes before every run.
        for value in [row.get("subject"), row.get("input")]:
            if value and Path(value).is_file():
                files.append(Path(value))
    configuration = dict(kind=args.kind, designs=args.designs, rounds=args.rounds, k=args.k,
                         mapping_seconds=300, baseline_algorithms_changed=False,frontend_root=str(frontend_root),
                         native_guard_seconds=60, eqmap_guard_seconds=90,
                         library_scope=f"K{args.k}: "+str({3:19,4:30,6:40}[args.k])+" of47 single-output cells",profile_values=list(args.profiles),
                         profile_parameter="EGM_WORD_QUOTIENT",word_truth=1,study_mode=args.mode,
                         profile_actual_values={"0":"0","1":"1"},
                         fixed_native_configuration={"EGM_FAST_ADMISSION_DELTA":"1","EGM_ADMISSION_SHADOW":"0","EGM_WORD_TRUTH":"1","EGM_WORD_REDUCE":"1","EGM_EXACT_CUT_OPS":"1","EGM_EXCHANGE_TIME_FRACTION":"0.70","EGM_FORCE_COMPACT_SOURCE":"1"})
    jobs = []
    if args.kind == "baseline":
        frozen = fingerprint(subjects)
        # Candidate mapper is not executed by these baseline-only jobs.
        frozen["files"].pop(str(ROOT / "target/release/bench_rs_egraph_map"), None)
        for name, bench, options in plan():
            if not name.startswith("3in_") or bench not in args.designs:
                continue
            options = list(options)
            options[options.index("--k") + 1] = "6"
            for repetition in range(args.rounds):
                jobs.append((f"r{repetition}_6in_{name[4:]}", bench, None, options))
    else:
        assert all([args.research_root, args.native, args.template, args.verification_cache])
        source, native = args.research_root.resolve(), args.native.resolve()
        template = json.loads(args.template.read_text())
        frozen = {"files": {p: sha(Path(p)) for p in template["fingerprint"]}}
        files += [source / "Cargo.toml", source / "Cargo.lock", source / "main.rs", native,
                  source / "emapper/Cargo.toml", args.template.resolve(), args.verification_cache.resolve()]
        files += list((source / "emapper/src").glob("*.rs"))
        assert sha(source / "emapper/src/netlist_metrics.rs") == sha(ROOT / "emapper/src/netlist_metrics.rs")
        for repetition in range(args.rounds):
            for ordinal, bench in enumerate(args.designs):
                profiles = list(args.profiles) if (repetition + ordinal) % 2 == 0 else list(reversed(args.profiles))
                original = next(job for job in template["jobs"] if job["bench"] == bench and job["k"] == 3)
                options = list(original["argv"])[original["argv"].index("--") + 1:]
                for flag, value in [("--k", str(args.k)), ("--native-optimizer", str(native)),
                                    ("--verification-cache", str(args.verification_cache.resolve())),("--mapper-root",str(frontend_root))]:
                    options[options.index(flag) + 1] = value
                for profile in profiles:
                    jobs.append((f"r{repetition}_{args.mode}{profile}_{args.k}in_{bench}", bench, profile, options))
        configuration.update(rho="1/2", recovery_seconds=2, return_reserve_seconds=.25,
                             profiles=["0: scalar full-care quotient", "1: exact word-masked full-care quotient"],
                             native=str(native), research_root=str(source))
    frozen["files"].update({str(p.resolve()): sha(p) for p in files})
    checkpoint = output / "campaign.json"
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(
        fingerprint=frozen, configuration=configuration, expected_jobs=len(jobs), jobs=[], status="checkpointed")
    assert state["fingerprint"] == frozen and state["configuration"] == configuration, "frozen study drift"
    for job in state["jobs"]:
        assert all(sha(Path(p)) == h for p, h in job["sha256"].items()), job["name"]

    def save():
        temporary = checkpoint.with_suffix(".pending")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(checkpoint)

    count = 0
    done = {job["name"] for job in state["jobs"]}
    for name, bench, profile, original_options in jobs:
        if name in done:
            continue
        folder, audit, log = output / name, output / (name + ".audit.json"), output / (name + ".log")
        assert not any(path.exists() for path in [folder, audit, log]), "unrecorded partial: audit before recovery"
        assert all(sha(Path(p)) == h for p, h in frozen["files"].items()), "executed dependency drift"
        options = list(original_options)
        argv = [sys.executable, str(HERE / "run_k6_quotient_budget.py"), "--mode", args.mode, "--profile", str(profile)]
        if args.kind == "baseline":
            driver, options = HELPERS / options[0], options[1:]
            options += ["--output", str(folder), "--prepared-report", str(subjects),
                        "--suite", "epfl", "--benches", bench]
        else:
            driver = HERE / "bench_scaled_sweep.py"
            for flag, value in [("--output", folder), ("--prepared-report", subjects)]:
                options[options.index(flag) + 1] = str(value)
            argv += ["--scaled", "--word-truth", "1"]
        argv += ["--mapping-seconds", "300", "--driver", str(driver), "--audit", str(audit), "--", *options]
        state.update(status="running", current_job=dict(name=name, argv=argv))
        save()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("EGM_", "ESYN_"))}
        with log.open("w") as stream:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        artifacts = [path for path in folder.rglob("*") if path.is_file()] + [audit, log]
        entry = dict(name=name, bench=bench, profile=profile, argv=argv, exit_code=result.returncode,
                     sha256={str(path): sha(path) for path in artifacts if path.is_file()})
        report = folder / "report.json"
        if report.exists():
            rows = json.loads(report.read_text())["results"]
            entry.update(report=str(report), ok=sum(row["status"] == "ok" for row in rows),
                         failed=sum(row["status"] != "ok" for row in rows))
        else:
            entry.update(ok=0, failed=1, unmeasured=True)
        state["jobs"].append(entry)
        state.pop("current_job")
        state["status"] = "attempted" if len(state["jobs"]) == len(jobs) else "checkpointed"
        save()
        print(name, entry["ok"], entry["failed"], flush=True)
        count += 1
        if count >= args.max_jobs:
            break


if __name__ == "__main__":
    main()
