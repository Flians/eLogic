#!/usr/bin/env bash
# Run epmapping and print baseline (emap) vs optimized QoR per design, with gains.
#
# Usage: scripts/run_epmapping.sh [workdir] [benchmark ...]
# Env:   EPMAPPING_OBJECTIVE (default 1), EPMAPPING_ROUNDS (default 4),
#        EPMAPPING_AREA_MAPPING, EPMAPPING_ABC_SCRIPT, ... (forwarded to the binary)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
GENLIB="$ROOT/tools/mockturtle/experiments/cell_libraries/asap7.genlib"
BIN="$ROOT/build/epmapping/epmapping_check"
WD="${1:-/tmp/epmap_bench}"
mkdir -p "$WD"
shift || true
BENCHES=("$@")
if [ ${#BENCHES[@]} -eq 0 ]; then
  BENCHES=(c432 c499 c880 c1355 c1908 c2670)
fi

pct() { # pct <before> <after>
  awk -v a="$1" -v b="$2" 'BEGIN { if (a+0 == 0) { print "n/a" } else { printf "%+.2f%%", 100.0*(b-a)/a } }'
}

printf '%-8s %9s %9s %8s %10s %10s %8s %5s %5s %6s\n' \
  bench base_area opt_area dA% base_del opt_del dD% rnd cec secs

for b in "${BENCHES[@]}"; do
  aig="$ROOT/tools/mockturtle/experiments/benchmarks/$b.aig"
  [ -f "$aig" ] || { echo "[SKIP] $b"; continue; }
  log="$WD/$b.log"
  start=$SECONDS
  EPMAPPING_OBJECTIVE="${EPMAPPING_OBJECTIVE:-1}" "$BIN" "$GENLIB" "$aig" "$WD" >"$log" 2>&1
  secs=$((SECONDS - start))
  base_area=$(sed -n 's/.*emap area=\([0-9.]*\).*/\1/p' "$log" | head -1)
  base_del=$(sed -n 's/.*emap area=[0-9.]* delay=\([0-9.]*\).*/\1/p' "$log" | head -1)
  ep_line=$(grep -m1 'epmapping area' "$log")
  opt_area=$(echo "$ep_line" | sed -n 's/.*area \([0-9.]*\) -> \([0-9.]*\).*/\2/p')
  opt_del=$(echo "$ep_line" | sed -n 's/.*delay \([0-9.]*\) -> \([0-9.]*\).*/\2/p')
  rnd=$(grep -m1 'epmapping rounds=' "$log" | sed -n 's/.*rounds=\([0-9]*\).*/\1/p')
  cec=$(grep -m1 'cec=' "$log" | sed -n 's/.*cec=\([0-9]*\).*/\1/p')
  printf '%-8s %9s %9s %8s %10s %10s %8s %5s %5s %6s\n' \
    "$b" "${base_area:--}" "${opt_area:--}" "$(pct "${base_area:-0}" "${opt_area:-0}")" \
    "${base_del:--}" "${opt_del:--}" "$(pct "${base_del:-0}" "${opt_del:-0}")" \
    "${rnd:--}" "${cec:--}" "$secs"
done
