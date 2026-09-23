#!/usr/bin/env bash
# Run EPFL combinational benchmarks (20 designs) with the default binary
# `MIGBalance` (`src/main.rs`): sets `EMAPPER` / `EMAPPER_GENLIB` per case when
# `tools/mockturtle/experiments/benchmarks/<name>.aig` is present.
#
# Usage (from repo root):
#   ./scripts/run_epfl_combinational.sh
# Optional: first arg = path to genlib (default: asap7 in mockturtle experiments).

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

GENLIB="${1:-$ROOT/tools/mockturtle/experiments/cell_libraries/asap7.genlib}"
BENCH_DIR="$ROOT/tools/mockturtle/experiments/benchmarks"
# Respect custom target dir (same layout as `cargo build --release`).
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
MAIN_BIN="$TARGET_DIR/release/MIGBalance"

if [[ ! -x "$MAIN_BIN" ]]; then
  echo "Building MIGBalance (release, src/main.rs)…"
  (cd "$ROOT" && cargo build --release --bin MIGBalance) || exit 1
fi

# EPFL combinational suite (same names as tools/mockturtle/experiments/mapper.json).
EPFL=(
  adder bar div hyp log2 max multiplier sin sqrt square
  arbiter cavlc ctrl dec i2c int2float mem_ctrl priority router voter
)

echo "genlib: $GENLIB"
echo "bench dir: $BENCH_DIR"
echo ""

for name in "${EPFL[@]}"; do
  aig="$BENCH_DIR/${name}.aig"
  if [[ ! -f "$aig" ]]; then
    echo "[SKIP] missing $aig"
    continue
  fi
  echo "========================================"
  echo "[RUN] $name"
  echo "========================================"
  set +e
  EMAPPER="$aig" EMAPPER_GENLIB="$GENLIB" "$MAIN_BIN"
  st=$?
  set -e
  if [[ "$st" -eq 0 ]]; then
    echo "[OK] $name"
  elif [[ "$st" -eq 2 ]]; then
    # main.rs: CEC could not run (no ABC on PATH / BENCH_RS_EMAP_ABC)
    echo "[WARN] $name: CEC unavailable (exit 2); mapping finished." >&2
    echo "[OK] $name"
  else
    echo "[FAIL] $name (exit $st)" >&2
  fi
  echo ""
done
