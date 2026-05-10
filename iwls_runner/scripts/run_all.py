#!/usr/bin/env python3
"""Run run_one_bench.py against many IWLS contest benchmarks in parallel,
then aggregate per-bench JSON outputs into a CSV.

All output stays under <package>/results/<run_id>/.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
RUN_ONE = PACKAGE_ROOT / "scripts" / "run_one_bench.py"
BENCH_DIR = PACKAGE_ROOT / "benchmarks"


def run_one(bench, out_root, simplify_timeout, cec_timeout, first_depth):
    t0 = time.time()
    cmd = [
        "python3", str(RUN_ONE), bench,
        "--out-root", str(out_root),
        "--simplify-timeout", str(simplify_timeout),
        "--cec-timeout", str(cec_timeout),
        "--first-depth", first_depth,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=simplify_timeout + cec_timeout + 120)
        return {
            "bench": bench, "ok": proc.returncode == 0, "rc": proc.returncode,
            "elapsed": time.time() - t0,
            "stdout_tail": proc.stdout[-500:],
            "stderr_tail": proc.stderr[-500:],
        }
    except subprocess.TimeoutExpired as e:
        return {
            "bench": bench, "ok": False, "rc": -1,
            "elapsed": time.time() - t0,
            "stdout_tail": "", "stderr_tail": f"WALL TIMEOUT: {e}",
        }
    except Exception as e:
        return {
            "bench": bench, "ok": False, "rc": -2,
            "elapsed": time.time() - t0,
            "stdout_tail": "", "stderr_tail": f"PYTHON EXC: {e}",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="full")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--benchmarks", default="",
                    help="comma-separated subset; default all 100")
    ap.add_argument("--simplify-timeout", type=int, default=900,
                    help="seconds for the mig_egg test invocation per bench")
    ap.add_argument("--cec-timeout", type=int, default=1800)
    ap.add_argument("--first-depth", default="true",
                    choices=["true", "false"])
    ap.add_argument("--resume", action="store_true",
                    help="skip benches whose results/<bench>.json with ok=true exists")
    args = ap.parse_args()

    if not os.environ.get("ELOGIC_REPO_DIR"):
        sys.exit("ELOGIC_REPO_DIR not set; export it to your eLogic repo root")

    out_root = PACKAGE_ROOT / "results" / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)
    res_dir = out_root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmarks:
        benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    else:
        benches = sorted([p.stem for p in BENCH_DIR.glob("ex*.truth")])

    if args.resume:
        skipped, kept = [], []
        for b in benches:
            p = res_dir / f"{b}.json"
            if p.exists():
                try:
                    j = json.load(open(p))
                    if j.get("ok"):
                        skipped.append(b)
                        continue
                except Exception:
                    pass
            kept.append(b)
        print(f"# resume: skipping {len(skipped)} already-ok benches, "
              f"running {len(kept)}", flush=True)
        benches = kept

    total = len(benches)
    print(f"# elogic_iwls run-id={args.run_id} workers={args.workers} "
          f"total={total} simplify_timeout={args.simplify_timeout}s "
          f"cec_timeout={args.cec_timeout}s first_depth={args.first_depth}",
          flush=True)
    t_start = time.time()

    done = 0
    n_ok = 0
    n_eq = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_one, b, out_root, args.simplify_timeout,
                      args.cec_timeout, args.first_depth): b
            for b in benches
        }
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            bench = r["bench"]
            if r["ok"]:
                n_ok += 1
            try:
                j = json.load(open(res_dir / f"{bench}.json"))
                eq = j.get("stages", {}).get("cec", {}).get("equivalent", False)
                init = j.get("stages", {}).get("init", {})
                final = j.get("stages", {}).get("final", {})
                if eq:
                    n_eq += 1
                tail = (
                    f"init=(and={init.get('and')}, lev={init.get('lev')}) "
                    f"final=(and={final.get('and')}, lev={final.get('lev')}) "
                    f"eq={eq}"
                )
            except Exception:
                tail = "(no result json)"
            elapsed_total = time.time() - t_start
            eta = (elapsed_total / done) * (total - done) if done > 0 else 0
            print(f"[{done}/{total}] {bench} took={r['elapsed']:.1f}s "
                  f"rc={r['rc']} {tail} | n_ok={n_ok} n_eq={n_eq} "
                  f"elapsed={elapsed_total/60:.1f}m eta={eta/60:.1f}m",
                  flush=True)
            if not r["ok"] and r["stderr_tail"]:
                print(f"    STDERR_TAIL: {r['stderr_tail'][-200:]!r}", flush=True)

    csv_path = out_root / "summary.csv"
    rows = []
    for b in sorted(benches):
        p = res_dir / f"{b}.json"
        if not p.exists():
            rows.append([b, "", "", "", "", "", "", ""])
            continue
        j = json.load(open(p))
        init = j.get("stages", {}).get("init", {})
        final = j.get("stages", {}).get("final", {})
        cec = j.get("stages", {}).get("cec", {})
        rows.append([
            b,
            init.get("and", ""), init.get("lev", ""),
            final.get("and", ""), final.get("lev", ""),
            cec.get("equivalent", ""), j.get("ok", ""),
            f"{j.get('total_s', 0):.1f}",
        ])
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["bench", "init_and", "init_lev",
                    "final_and", "final_lev", "equivalent", "ok", "total_s"])
        w.writerows(rows)
    print(f"# wrote {csv_path}", flush=True)
    print(f"# done in {(time.time() - t_start) / 60:.1f}m. "
          f"ok={n_ok}/{total} cec_equivalent={n_eq}/{total}", flush=True)


if __name__ == "__main__":
    main()
