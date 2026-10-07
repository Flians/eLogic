#!/usr/bin/env python3
"""Compare finite seed and shared-area recovery, requiring CEC for every run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=root / "target/release/bench_rs_egraph_map")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("benches", nargs="*", default=["c17", "c432", "c499"])
    args = parser.parse_args()
    library = root / "tools/mockturtle/experiments/cell_libraries/asap7.genlib"
    env = {key: value for key, value in os.environ.items() if not key.startswith("EGM_")}
    settings = dict(EGM_K="3", EGM_CUTS="8", EGM_TIME="0", EGM_DC_SUPPORT="4",
                    EGM_REFINE_ROUNDS="4", EGM_REFINE_TRIALS="2000")
    env.update(settings)
    report = {"settings": settings, "library": str(library), "results": []}
    with tempfile.TemporaryDirectory(prefix="egm-recovery-") as directory:
        for bench in args.benches:
            if Path(bench).name != bench:
                parser.error("bench names must be filenames without directory components")
            row = {"bench": bench}
            for mode, budget in [("seed", "0"), ("recovered", "2")]:
                run_env = dict(env, EGM_REFINE_TIME=budget)
                circuit = root / "tools/mockturtle/experiments/benchmarks" / f"{bench}.aig"
                result = subprocess.run(
                    [str(args.binary.resolve()), str(library), str(circuit),
                     str(Path(directory) / f"{bench}-{mode}.v")],
                    env=run_env, text=True, capture_output=True, timeout=args.timeout,
                )
                if result.returncode:
                    raise RuntimeError(f"{bench}/{mode}: {result.stderr}\n{result.stdout[-4000:]}")
                records = [json.loads(line) for line in result.stdout.splitlines()
                           if line.startswith('{"area":')]
                if len(records) != 1 or records[0].get("cec") != 1:
                    raise RuntimeError(f"{bench}/{mode}: missing successful equivalence check")
                row[mode] = records[0]
            baseline = row["seed"]["area"]
            row["area_reduction_percent"] = (
                100 * (baseline - row["recovered"]["area"]) / baseline if baseline else 0
            )
            report["results"].append(row)
            print(f"{bench}: {baseline:.4f} -> {row['recovered']['area']:.4f}, CEC passed", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
