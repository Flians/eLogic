#!/usr/bin/env python3
"""Run the eLogic e-graph rewrite pipeline on one IWLS contest benchmark.

Stages
------
  1. ABC: read_truth -xf benchmarks/B.truth ; strash ; print_stats
         ; write_aiger init.aig
  2. aig_to_prefix.py init.aig -> init.prefix       (one PO/line, AIG form)
  3. mig_egg test binary (env IWLS_PREFIX_FILE=init.prefix
         + IWLS_FIRST_DEPTH={true|false}) -> per-PO best_expr,
         captured from stdout into optimized.prefix
  4. prefix_to_aig.py optimized.prefix -> final.aig
         (via EQN -> ABC strash + resyn2)
  5. ABC: read_truth -xf benchmarks/B.truth ; cec -n final.aig
  6. ABC: read_aiger final.aig ; print_stats           (final and / lev)
  7. Write results/<bench>.json with init/final stats + equiv flag + timings

Paths are resolved relative to this package (the parent of this script's
directory) for benchmarks / abc.rc; the path to your eLogic clone is read
from $ELOGIC_REPO_DIR (it must contain a release-built mig_egg test binary
at $ELOGIC_REPO_DIR/target/release/deps/mig_egg-<hash>).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ABC = os.environ.get("ABC", "abc")
ELOGIC_REPO_DIR = os.environ.get("ELOGIC_REPO_DIR")
AIG2PFX = PACKAGE_ROOT / "scripts" / "aig_to_prefix.py"
PFX2AIG = PACKAGE_ROOT / "scripts" / "prefix_to_aig.py"
BENCH_DIR = PACKAGE_ROOT / "benchmarks"


def find_mig_egg_bin():
    """Locate the release-built mig_egg test binary."""
    if not ELOGIC_REPO_DIR:
        sys.exit("ELOGIC_REPO_DIR not set; point it at your eLogic repo root")
    deps = Path(ELOGIC_REPO_DIR) / "target" / "release" / "deps"
    if not deps.is_dir():
        sys.exit(f"missing {deps}; build with: cd {ELOGIC_REPO_DIR}/mig_egg "
                 f"&& cargo build --release --tests --no-default-features")
    candidates = []
    for p in deps.iterdir():
        if p.name.startswith("mig_egg-") and "." not in p.name[len("mig_egg-"):] \
                and p.is_file() and os.access(p, os.X_OK):
            candidates.append(p)
    if not candidates:
        sys.exit(f"no mig_egg test binary in {deps}; "
                 f"run `cargo test --release --no-default-features --no-run`")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def run_abc(cmd_str, log_path, cwd=PACKAGE_ROOT, timeout=600):
    """Run `abc -c <cmd_str>` with cwd containing abc.rc."""
    proc = subprocess.run(
        [str(ABC), "-c", cmd_str],
        cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
    )
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(f"\n$ abc -c {cmd_str!r}\n")
            f.write(proc.stdout)
            f.write(proc.stderr)
            f.write(f"# exit={proc.returncode}\n")
    return proc.stdout, proc.stderr, proc.returncode


_STATS_RE = re.compile(
    r"\bi/o\s*=\s*(\d+)\s*/\s*(\d+).*?\band\s*=\s*(\d+).*?\blev\s*=\s*(\d+)",
    re.DOTALL,
)


def parse_print_stats(out, which="last"):
    matches = list(_STATS_RE.finditer(out))
    if not matches:
        return None
    m = matches[-1] if which == "last" else matches[0]
    return {"inputs": int(m.group(1)), "outputs": int(m.group(2)),
            "and": int(m.group(3)), "lev": int(m.group(4))}


def parse_optimized_prefix(test_stdout):
    """Pull `iwls case <id> best_expr[0] = ...` lines from the test binary."""
    out = {}
    case_re = re.compile(r"^iwls case (\S+) best_expr\[0\] = (.+)$")
    for line in test_stdout.splitlines():
        m = case_re.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", help="benchmark id, e.g. ex212")
    ap.add_argument("--out-root",
                    default=str(PACKAGE_ROOT / "results" / "default"),
                    help="root directory for this run's outputs")
    ap.add_argument("--simplify-timeout", type=int, default=900,
                    help="seconds; cap on the mig_egg test invocation")
    ap.add_argument("--cec-timeout", type=int, default=1800,
                    help="seconds; cec -n timeout per benchmark")
    ap.add_argument("--first-depth", default="true",
                    choices=["true", "false"],
                    help="forwarded as IWLS_FIRST_DEPTH env var to mig_egg "
                         "test (true = delay-first, false = area-first)")
    args = ap.parse_args()

    bench = args.bench
    out_root = Path(args.out_root)
    work = out_root / "per_bench" / bench
    log_dir = out_root / "log"
    res_dir = out_root / "results"
    aig_out_dir = out_root / "aigs"
    for d in (work, log_dir, res_dir, aig_out_dir):
        d.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{bench}.log"

    truth = BENCH_DIR / f"{bench}.truth"
    if not truth.exists():
        sys.exit(f"missing truth: {truth}")

    init_aig = work / "init.aig"
    init_pfx = work / "init.prefix"
    opt_pfx = work / "optimized.prefix"
    final_aig = work / "final.aig"

    result = {"bench": bench, "stages": {}, "ok": False}
    t0 = time.time()

    # --- Stage 1: ABC strash ---
    cmd1 = (
        f"read_truth -xf benchmarks/{bench}.truth; "
        f"strash; print_stats; "
        f"write_aiger {init_aig};"
    )
    out, err, rc = run_abc(cmd1, log_path)
    if rc != 0:
        result["stages"]["init_aig"] = {"ok": False, "rc": rc}
        json.dump(result, open(res_dir / f"{bench}.json", "w"), indent=2)
        return 1
    init_stats = parse_print_stats(out)
    result["stages"]["init"] = {"ok": True, **(init_stats or {})}

    # --- Stage 2: AIG -> AIG-form prefix expressions ---
    proc = subprocess.run(
        ["python3", str(AIG2PFX), str(init_aig)],
        capture_output=True, text=True,
    )
    with open(log_path, "a") as f:
        f.write(f"\n$ aig_to_prefix.py {init_aig}\n")
        f.write(proc.stderr)
        f.write(f"# exit={proc.returncode}\n")
    if proc.returncode != 0:
        result["stages"]["aig2pfx"] = {"ok": False, "stderr": proc.stderr}
        json.dump(result, open(res_dir / f"{bench}.json", "w"), indent=2)
        return 1
    init_pfx.write_text(proc.stdout)
    n_outputs = len([l for l in proc.stdout.splitlines()
                     if l and not l.startswith("#")])
    result["stages"]["aig2pfx"] = {"ok": True, "n_outputs": n_outputs}

    # --- Stage 3: e-graph rewrite via mig_egg test binary ---
    bin_path = find_mig_egg_bin()
    env = dict(os.environ)
    env["IWLS_PREFIX_FILE"] = str(init_pfx)
    env["IWLS_FIRST_DEPTH"] = args.first_depth
    t1 = time.time()
    try:
        proc = subprocess.run(
            [str(bin_path), "iwls_run", "--nocapture", "--test-threads=1"],
            env=env, capture_output=True, text=True,
            timeout=args.simplify_timeout,
        )
    except subprocess.TimeoutExpired:
        with open(log_path, "a") as f:
            f.write(f"\n# mig_egg simplify TIMEOUT after {args.simplify_timeout}s\n")
        result["stages"]["simplify"] = {"ok": False, "reason": "timeout"}
        json.dump(result, open(res_dir / f"{bench}.json", "w"), indent=2)
        return 1
    t_simplify = time.time() - t1
    with open(log_path, "a") as f:
        f.write(f"\n$ mig_egg iwls_run (release) IWLS_FIRST_DEPTH={args.first_depth}\n")
        f.write("--- stdout (tail) ---\n")
        f.write(proc.stdout[-4000:])
        f.write("\n--- stderr (full) ---\n")
        f.write(proc.stderr)
        f.write(f"\n# exit={proc.returncode} took={t_simplify:.1f}s\n")
    optimized = parse_optimized_prefix(proc.stdout)
    if proc.returncode != 0 or not optimized:
        result["stages"]["simplify"] = {
            "ok": False, "rc": proc.returncode, "took_s": t_simplify,
            "n_optimized": len(optimized),
        }
        json.dump(result, open(res_dir / f"{bench}.json", "w"), indent=2)
        return 1
    lines = []
    for line in init_pfx.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        idx, _ = line.split("\t", 1)
        if idx not in optimized:
            sys.stderr.write(f"# missing optimized expr for PO {idx}\n")
            return 1
        lines.append(f"{idx}\t{optimized[idx]}")
    opt_pfx.write_text("\n".join(lines) + "\n")
    result["stages"]["simplify"] = {"ok": True, "took_s": t_simplify,
                                    "n_optimized": len(optimized)}

    # --- Stage 4: optimized prefix -> AIG (via EQN + ABC strash) ---
    proc = subprocess.run(
        ["python3", str(PFX2AIG), str(opt_pfx), str(final_aig),
         "--abc", str(ABC), "--n-inputs", str(init_stats["inputs"]),
         "--abc-rc-dir", str(PACKAGE_ROOT), "--keep-eqn"],
        capture_output=True, text=True,
    )
    with open(log_path, "a") as f:
        f.write(f"\n$ prefix_to_aig.py\n")
        f.write(proc.stdout)
        f.write(proc.stderr)
        f.write(f"# exit={proc.returncode}\n")
    if proc.returncode != 0:
        result["stages"]["pfx2aig"] = {"ok": False, "stderr": proc.stderr}
        json.dump(result, open(res_dir / f"{bench}.json", "w"), indent=2)
        return 1
    final_stats = parse_print_stats(proc.stdout)
    result["stages"]["final"] = {"ok": True, **(final_stats or {})}

    # --- Stage 5: cec verify ---
    cmd_cec = (
        f"read_truth -xf benchmarks/{bench}.truth; "
        f"cec -n {final_aig};"
    )
    try:
        out, err, rc = run_abc(cmd_cec, log_path, timeout=args.cec_timeout)
    except subprocess.TimeoutExpired:
        with open(log_path, "a") as f:
            f.write(f"\n# cec TIMEOUT after {args.cec_timeout}s\n")
        result["stages"]["cec"] = {"ok": False, "reason": "timeout"}
        json.dump(result, open(res_dir / f"{bench}.json", "w"), indent=2)
        return 1
    equivalent = "Networks are equivalent" in out
    result["stages"]["cec"] = {"ok": rc == 0, "equivalent": equivalent}

    # --- Stage 6: contest-shaped AIG (only if equivalent) ---
    if equivalent and final_stats is not None:
        lev = final_stats["lev"]
        contest_name = f"{bench}_{lev:03d}.aig"
        dst = aig_out_dir / contest_name
        shutil.copy(final_aig, dst)
        result["aig_path"] = str(dst)
        result["aig_name"] = contest_name

    result["ok"] = equivalent
    result["total_s"] = time.time() - t0
    json.dump(result, open(res_dir / f"{bench}.json", "w"), indent=2)

    init = result["stages"].get("init", {})
    final = result["stages"].get("final", {})
    print(f"{bench}: init=(and={init.get('and')}, lev={init.get('lev')}) "
          f"-> final=(and={final.get('and')}, lev={final.get('lev')}) "
          f"equiv={equivalent} took={result['total_s']:.1f}s")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
