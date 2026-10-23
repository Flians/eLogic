#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from statistics import median


def run_cmd(cmd, env=None, timeout=None):
    p = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=timeout,
    )
    out = p.stdout.strip().splitlines()
    last = out[-1] if out else ""
    return p.returncode, last, p.stderr


def geom_mean(xs):
    import math

    xs = [x for x in xs if x is not None and x > 0.0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bench-dir",
        default="tools/mockturtle/experiments/benchmarks",
        help="Directory with .aig benchmarks",
    )
    ap.add_argument(
        "--genlib",
        default="tools/mockturtle/experiments/cell_libraries/asap7.genlib",
        help="Genlib path",
    )
    ap.add_argument(
        "--mt-emap",
        default="./build/emap/emap",
        help="Mockturtle emap executable",
    )
    ap.add_argument(
        "--rs-emap",
        default="./target/debug/bench_rs_emap",
        help="Rust emap bench executable",
    )
    ap.add_argument(
        "--use-egg",
        action="store_true",
        help="Enable egg in Rust mapper (default off)",
    )
    ap.add_argument(
        "--pattern",
        default="*.aig",
        help="Glob pattern under bench-dir (default *.aig)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of benchmarks (0 = no limit)",
    )
    ap.add_argument(
        "--out",
        default="compare_emap.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--timeout-sec",
        type=float,
        default=120.0,
        help="Per-command timeout seconds (default 120s)",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV (resume style). Duplicates not removed.",
    )
    args = ap.parse_args()

    bench_paths = sorted(glob.glob(os.path.join(args.bench_dir, args.pattern)))
    if args.limit and args.limit > 0:
        bench_paths = bench_paths[: args.limit]

    if not bench_paths:
        print(f"No benchmarks matched under {args.bench_dir}/{args.pattern}", file=sys.stderr)
        return 2

    env_rs = os.environ.copy()
    env_rs["EMAP_USE_EGG"] = "1" if args.use_egg else "0"

    rows = []

    fieldnames = [
        "name",
        "mt_ok",
        "rs_ok",
        "mt_area",
        "mt_gates",
        "mt_delay",
        "mt_depth",
        "mt_runtime",
        "rs_area",
        "rs_gates",
        "rs_delay",
        "rs_depth",
        "rs_runtime",
        "rs_wall",
        "area_ratio_rs_over_mt",
        "delay_ratio_rs_over_mt",
        "runtime_ratio_rs_over_mt",
        "error",
    ]

    out_exists = os.path.exists(args.out)
    out_mode = "a" if args.append else "w"
    out_file = open(args.out, out_mode, newline="")
    out_writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    if (not out_exists) or (not args.append):
        out_writer.writeheader()
        out_file.flush()

    for p in bench_paths:
        name = os.path.basename(p).removesuffix(".aig")
        mt_v = f"/tmp/mt_{name}.v"

        try:
            rc_mt, mt_line, mt_err = run_cmd(
                f"{args.mt_emap} {args.genlib} {p} {mt_v}", timeout=args.timeout_sec
            )
        except subprocess.TimeoutExpired:
            row = {"name": name, "mt_ok": 0, "rs_ok": 0, "error": "mockturtle timeout"}
            rows.append(row)
            out_writer.writerow(row)
            out_file.flush()
            print(f"[skip] {name}: mockturtle timeout", flush=True)
            continue
        if rc_mt != 0:
            row = {
                "name": name,
                "mt_ok": 0,
                "rs_ok": 0,
                "error": f"mockturtle rc={rc_mt}: {mt_err.strip()[:200]}",
            }
            rows.append(row)
            out_writer.writerow(row)
            out_file.flush()
            print(f"[skip] {name}: mockturtle failed", flush=True)
            continue

        try:
            mt = json.loads(mt_line)
        except Exception as e:
            row = {
                "name": name,
                "mt_ok": 0,
                "rs_ok": 0,
                "error": f"mockturtle bad json: {e}",
            }
            rows.append(row)
            out_writer.writerow(row)
            out_file.flush()
            print(f"[skip] {name}: mockturtle bad json", flush=True)
            continue

        t0 = time.time()
        try:
            rc_rs, rs_line, rs_err = run_cmd(
                f"{args.rs_emap} {args.genlib} {p}",
                env=env_rs,
                timeout=args.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            rs_wall = time.time() - t0
            row = {
                "name": name,
                "mt_ok": 1,
                "rs_ok": 0,
                "mt_area": mt.get("area"),
                "mt_delay": mt.get("delay"),
                "mt_depth": mt.get("depth"),
                "mt_runtime": mt.get("runtime"),
                "rs_wall": rs_wall,
                "error": "rust timeout",
            }
            rows.append(row)
            out_writer.writerow(row)
            out_file.flush()
            print(f"[skip] {name}: rust timeout", flush=True)
            continue
        rs_wall = time.time() - t0
        if rc_rs != 0:
            row = {
                "name": name,
                "mt_ok": 1,
                "rs_ok": 0,
                "mt_area": mt.get("area"),
                "mt_delay": mt.get("delay"),
                "mt_depth": mt.get("depth"),
                "mt_runtime": mt.get("runtime"),
                "rs_wall": rs_wall,
                "error": f"rust rc={rc_rs}: {rs_err.strip()[:200]}",
            }
            rows.append(row)
            out_writer.writerow(row)
            out_file.flush()
            print(f"[skip] {name}: rust failed", flush=True)
            continue

        try:
            rs = json.loads(rs_line)
        except Exception as e:
            row = {
                "name": name,
                "mt_ok": 1,
                "rs_ok": 0,
                "mt_area": mt.get("area"),
                "mt_delay": mt.get("delay"),
                "mt_depth": mt.get("depth"),
                "mt_runtime": mt.get("runtime"),
                "rs_wall": rs_wall,
                "error": f"rust bad json: {e}",
            }
            rows.append(row)
            out_writer.writerow(row)
            out_file.flush()
            print(f"[skip] {name}: rust bad json", flush=True)
            continue

        row = {
            "name": name,
            "mt_ok": 1,
            "rs_ok": 1,
            "mt_area": mt.get("area"),
            "mt_gates": mt.get("gates"),
            "mt_delay": mt.get("delay"),
            "mt_depth": mt.get("depth"),
            "mt_runtime": mt.get("runtime"),
            "rs_area": rs.get("area"),
            "rs_gates": rs.get("gates"),
            "rs_delay": rs.get("delay"),
            "rs_depth": rs.get("depth"),
            "rs_runtime": rs.get("runtime"),
            "rs_wall": rs_wall,
        }
        row["area_ratio_rs_over_mt"] = (
            row["rs_area"] / row["mt_area"] if row["mt_area"] else None
        )
        row["delay_ratio_rs_over_mt"] = (
            row["rs_delay"] / row["mt_delay"] if row["mt_delay"] else None
        )
        row["runtime_ratio_rs_over_mt"] = (
            row["rs_runtime"] / row["mt_runtime"] if row["mt_runtime"] else None
        )
        rows.append(row)
        out_writer.writerow(row)
        out_file.flush()
        print(
            f"[ok] {name}: area {row['mt_area']:.2f}->{row['rs_area']:.2f} "
            f"delay {row['mt_delay']:.2f}->{row['rs_delay']:.2f} "
            f"rt {row['mt_runtime']:.4f}->{row['rs_runtime']:.4f}",
            flush=True,
        )

    out_file.close()

    # Print summary (only successful rows)
    ok = [r for r in rows if r.get("mt_ok") == 1 and r.get("rs_ok") == 1]
    if ok:
        area_ratios = [r["area_ratio_rs_over_mt"] for r in ok]
        delay_ratios = [r["delay_ratio_rs_over_mt"] for r in ok]
        rt_ratios = [r["runtime_ratio_rs_over_mt"] for r in ok]
        print("\nSummary (Rust / mockturtle ratios):", flush=True)
        print(f"- circuits: {len(ok)}", flush=True)
        print(f"- area:   median {median(area_ratios):.3f}, gmean {geom_mean(area_ratios):.3f}", flush=True)
        print(f"- delay:  median {median(delay_ratios):.3f}, gmean {geom_mean(delay_ratios):.3f}", flush=True)
        print(f"- runtime median {median(rt_ratios):.3f}, gmean {geom_mean(rt_ratios):.3f}", flush=True)
        print(f"\nWrote {args.out}", flush=True)
    else:
        print("\nNo successful comparisons. See CSV for errors.", flush=True)
        print(f"Wrote {args.out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

