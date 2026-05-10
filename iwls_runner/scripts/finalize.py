#!/usr/bin/env python3
"""Collect AIGs from one or more run dirs, optionally cec-verify each,
dedupe by (bench, lev), Pareto-filter per bench, and emit a contest-shaped
zip + summary CSV.

Usage:
    finalize.py --runs <run_id_1>,<run_id_2>,... [--out-dir DIR] [--no-verify]

The IWLS contest expects a flat zip (no subdirs) of files named
`exNNN_LLL.aig`, where LLL is the AIG's logic level (3-digit padded).
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ABC = os.environ.get("ABC", "abc")
BENCH_DIR = PACKAGE_ROOT / "benchmarks"

_STATS_RE = re.compile(
    r"\bi/o\s*=\s*(\d+)\s*/\s*(\d+).*?\band\s*=\s*(\d+).*?\blev\s*=\s*(\d+)",
    re.DOTALL,
)


def aig_stats(path):
    out = subprocess.run(
        [str(ABC), "-c", f"read_aiger {path}; print_stats"],
        cwd=str(PACKAGE_ROOT), capture_output=True, text=True, timeout=120,
    ).stdout
    m = _STATS_RE.search(out)
    if not m:
        return None
    return {"and": int(m.group(3)), "lev": int(m.group(4))}


def cec_verify(bench, aig_path, timeout=1800):
    proc = subprocess.run(
        [str(ABC), "-c",
         f"read_truth -xf benchmarks/{bench}.truth; cec -n {aig_path}"],
        cwd=str(PACKAGE_ROOT), capture_output=True, text=True, timeout=timeout,
    )
    return "Networks are equivalent" in proc.stdout


def gather_aigs(run_dirs):
    """For each run dir, walk <dir>/aigs/ and collect all AIG paths."""
    out = []
    for d in run_dirs:
        aig_dir = d / "aigs"
        if not aig_dir.exists():
            print(f"# warn: {aig_dir} missing, skip", file=sys.stderr)
            continue
        for p in sorted(aig_dir.glob("*.aig")):
            m = re.match(r"(ex\d+)_(\d{3})\.aig$", p.name)
            if m:
                out.append((m.group(1), int(m.group(2)), d.name, p))
    return out


def pareto_filter(points):
    survivors = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if (q["lev"] <= p["lev"] and q["and"] < p["and"]) or \
               (q["lev"] < p["lev"] and q["and"] <= p["and"]):
                dominated = True
                break
        if not dominated:
            survivors.append(p)
    return survivors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="comma-separated run-ids under <package>/results/")
    ap.add_argument("--out-dir",
                    default=str(PACKAGE_ROOT / "results" / "final"),
                    help="output directory for combined zip + summary")
    ap.add_argument("--cec-timeout", type=int, default=900)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip cec verification (trust per-run cec results)")
    args = ap.parse_args()

    run_dirs = [PACKAGE_ROOT / "results" / r.strip()
                for r in args.runs.split(",") if r.strip()]
    out_dir = Path(args.out_dir)
    submit_dir = out_dir / "submit"
    submit_dir.mkdir(parents=True, exist_ok=True)
    non_eq_dir = out_dir / "non_equiv"
    non_eq_dir.mkdir(parents=True, exist_ok=True)

    all_pts = gather_aigs(run_dirs)
    print(f"# gathered {len(all_pts)} AIGs from {len(run_dirs)} runs",
          flush=True)

    by_bench = defaultdict(list)
    for bench, lev, src, path in all_pts:
        stats = aig_stats(path)
        if stats is None:
            continue
        by_bench[bench].append({
            "bench": bench, "lev": stats["lev"], "and": stats["and"],
            "src": src, "path": path,
        })

    summary_rows = []
    n_kept = 0
    for bench in sorted(by_bench):
        pts = by_bench[bench]
        seen = set()
        uniq = []
        for p in pts:
            k = (p["lev"], p["and"])
            if k in seen:
                continue
            seen.add(k)
            uniq.append(p)
        pareto = pareto_filter(uniq)
        for p in pareto:
            ok = True
            if not args.no_verify:
                ok = cec_verify(bench, p["path"], timeout=args.cec_timeout)
            if ok:
                dst = submit_dir / p["path"].name
                if not dst.exists():
                    shutil.copy(p["path"], dst)
                n_kept += 1
                summary_rows.append([bench, p["lev"], p["and"], p["src"], "OK"])
            else:
                ne_dst = non_eq_dir / f"{p['src']}__{p['path'].name}"
                shutil.copy(p["path"], ne_dst)
                summary_rows.append([bench, p["lev"], p["and"], p["src"],
                                     "NOT_EQ"])
        print(f"  {bench}: {len(uniq)} candidates -> {len(pareto)} pareto",
              flush=True)

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["bench", "lev", "and", "src_run", "status"])
        w.writerows(summary_rows)

    zip_path = out_dir / "elogic_iwls_submit.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(submit_dir.glob("*.aig")):
            zf.write(p, arcname=p.name)
    n_zip = len(list(submit_dir.glob("*.aig")))

    benches_covered = set(r[0] for r in summary_rows if r[4] == "OK")
    print(f"\n=== finalize done ===", flush=True)
    print(f"  zip path:      {zip_path}", flush=True)
    print(f"  zip files:     {n_zip}", flush=True)
    print(f"  bench covered: {len(benches_covered)}/100", flush=True)
    print(f"  source runs:   {','.join(d.name for d in run_dirs)}", flush=True)


if __name__ == "__main__":
    main()
