#!/usr/bin/env bash
# Compare QoR: emapper (bench_rs_emap), mockturtle emap_single, yosys-abc map.
# Usage: ./scripts/compare_mapping_qor.sh <genlib> <aig> [--resyn2]
# Env: COMPARE_RESYN2=1 or third arg --resyn2 → yosys-abc strash; resyn2 on <aig> first.
#      COMPARE_BUILD=1 → cargo + cmake build first. MOCKTURTLE_EMAP_SINGLE overrides path to emap_single. VERBOSE=1.
#      mockturtle cmake: -DMOCKTURTLE_BUILD_EXPERIMENTS=ON (not MOCKTURTLE_EXPERIMENTS).
#
# nec: emapper uses BENCH_RS_EMAP_ABC (default: yosys-abc if in PATH). mockturtle emap_single uses `abc` if in PATH; else -1. yosys line has nec=-1.

set -uo pipefail

COMPARE_BUILD=1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GENLIB="${1:?genlib}"
AIG="${2:?aig}"
OPT_AIG="$AIG"
AIG_TMP=""

cleanup() { [[ -n "$AIG_TMP" && -f "$AIG_TMP" ]] && rm -f "$AIG_TMP"; }
trap cleanup EXIT

YOSYS_ABC="$(command -v yosys-abc 2>/dev/null || true)"
[[ -n "$YOSYS_ABC" ]] && export BENCH_RS_EMAP_ABC="${BENCH_RS_EMAP_ABC:-$YOSYS_ABC}"

if [[ "${COMPARE_RESYN2:-0}" == 1 || "${3:-}" == "--resyn2" || "${3:-}" == "resyn2" ]] && [[ -n "$YOSYS_ABC" ]]; then
  base="$(mktemp /tmp/migbalance_opt_XXXXXX)"
  out="${base}.aig"
  printf 'read_genlib %s\nread_aiger %s\nstrash\nresyn2\nwrite_aiger %s\nquit\n' "$GENLIB" "$AIG" "$out" >"${base}.abc"
  if "$YOSYS_ABC" -f "${base}.abc" >/dev/null 2>&1 && [[ -f "$out" ]]; then
    OPT_AIG="$out"
    AIG_TMP="$out"
  fi
  rm -f "${base}.abc"
fi

if [[ "${COMPARE_BUILD:-0}" == 1 ]]; then
  ( cd "$ROOT" && cargo build --release --bin bench_rs_emap )
  MT="$ROOT/tools/mockturtle"
  MB="$ROOT/tools/mockturtle/build"
  cmake -S "$MT" -B "$MB" -DMOCKTURTLE_BUILD_EXPERIMENTS=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build "$MB" --target emap_single -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi

BENCH="$ROOT/target/release/bench_rs_emap"
EMAP="${MOCKTURTLE_EMAP_SINGLE:-$ROOT/tools/mockturtle/build/experiments/emap_single}"

run_bench() {
  if [[ -x "$BENCH" ]]; then "$BENCH" "$@"; else ( cd "$ROOT" && cargo run -q --release --bin bench_rs_emap -- "$@" ); fi
}

echo "=== emapper (bench_rs_emap) ==="
if [[ "${VERBOSE:-}" == 1 ]]; then
  run_bench "$GENLIB" "$OPT_AIG" | grep '^{' || true
else
  run_bench "$GENLIB" "$OPT_AIG" 2>/dev/null | grep '^{' || true
fi

echo "=== mockturtle (emap_single) ==="
if [[ -x "$EMAP" ]]; then
  "$EMAP" "$GENLIB" "$OPT_AIG" 2>/dev/null | grep '^{' || "$EMAP" "$GENLIB" "$OPT_AIG"
else
  echo "skip: $EMAP" >&2
fi

echo "=== yosys-abc map ==="
if [[ -n "$YOSYS_ABC" ]]; then
  b="$(mktemp /tmp/migbalance_abcmap_XXXXXX)"
  abc="${b}.abc"
  blif="${b}.mapped.blif"
  out="${b}.out"
  rm -f "$b"
  printf 'read_genlib %s\nread_aiger %s\nstrash\nmap\nwrite_blif %s\nprint_stats\nquit\n' "$GENLIB" "$OPT_AIG" "$blif" >"$abc"
  t0="$(python3 -c 'import time; print(time.perf_counter())')"
  "$YOSYS_ABC" -f "$abc" >"$out" 2>&1
  rt="$(python3 -c 'import sys,time; print(f"{time.perf_counter()-float(sys.argv[1]):.6f}")' "$t0")"

  python3 - "$out" "$blif" "$rt" <<'PY'
import json, re, sys
from pathlib import Path
txt = Path(sys.argv[1]).read_text()
txt = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", txt)
m = re.search(r"nd\s*=\s*(\d+).*?area\s*=\s*([0-9.]+).*?delay\s*=\s*([0-9.]+).*?lev\s*=\s*(\d+)", txt, re.S)
mp = Path(sys.argv[2])
rt = float(sys.argv[3])
gates = -1.0
if mp.exists():
    gates = float(sum(1 for line in mp.read_text().splitlines() if line.startswith(".gate ")))
if m:
    print(json.dumps({
        "mapper": "yosys-abc-map",
        "area": float(m.group(2)),
        "gates": gates if gates >= 0 else float(m.group(1)),
        "delay": float(m.group(3)),
        "depth": float(m.group(4)),
        "runtime": rt,
        "nec": -1.0,
    }))
else:
    sys.stdout.write(txt)
    if gates >= 0:
        print(json.dumps({"mapper": "yosys-abc-map", "gates": gates}))
PY

  rm -f "$out" "$abc" "$blif"
else
  echo "skip: yosys-abc not in PATH"
fi
