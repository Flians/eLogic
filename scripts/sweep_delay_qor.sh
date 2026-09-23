#!/usr/bin/env bash
# Compare mapper max_delay / area on one AIG (JSON line from bench_rs_emap).
# Usage:
#   scripts/sweep_delay_qor.sh [GENLIB] [AIG]
# Example:
#   scripts/sweep_delay_qor.sh tools/mockturtle/experiments/cell_libraries/asap7.genlib \
#     tools/mockturtle/experiments/benchmarks/c432.aig

set -euo pipefail

GENLIB="${1:-tools/mockturtle/experiments/cell_libraries/asap7.genlib}"
AIG="${2:-tools/mockturtle/experiments/benchmarks/c432.aig}"

if [[ ! -f "$GENLIB" || ! -f "$AIG" ]]; then
  echo "usage: $0 [genlib] [aig]" >&2
  exit 1
fi

run_case() {
  local name="$1"
  shift
  echo "== $name ==" >&2
  set +e
  local json
  json="$(env "$@" cargo run -q --package MIGBalance --bin bench_rs_emap -- "$GENLIB" "$AIG" 2>/dev/null | head -1)"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "FAILED (exit=$rc)" >&2
    return 0
  fi
  echo "$json"
}

echo "GENLIB=$GENLIB AIG=$AIG"
echo "case	delay	area	gates	depth	nec"
echo "------------------------------------------------"

parse() {
  local name="$1" line="$2"
  # {"mapper":"...","area": x, "gates": y, "delay": z, "depth": w, "runtime":..., "nec":...}
  local area gates delay depth nec
  area="$(echo "$line" | sed -n 's/.*"area": *\([0-9.]*\).*/\1/p')"
  gates="$(echo "$line" | sed -n 's/.*"gates": *\([0-9.]*\).*/\1/p')"
  delay="$(echo "$line" | sed -n 's/.*"delay": *\([0-9.]*\).*/\1/p')"
  depth="$(echo "$line" | sed -n 's/.*"depth": *\([0-9.]*\).*/\1/p')"
  nec="$(echo "$line" | sed -n 's/.*"nec": *\([0-9.-]*\).*/\1/p')"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$delay" "$area" "$gates" "$depth" "$nec"
}

line="$(run_case bench_default EMAPPER_AREA_FLOW_PASSES=1 EMAPPER_EXACT_AREA_PASSES=1 EMAPPER_MAX_CUTS=16)"
[[ -n "$line" ]] && parse bench_default "$line"

line="$(run_case delay_lean EMAPPER_AREA_FLOW_PASSES=0 EMAPPER_EXACT_AREA_PASSES=0 EMAPPER_MAX_CUTS=16)"
[[ -n "$line" ]] && parse delay_lean "$line"

line="$(run_case delay_cuts32_phys EMAPPER_AREA_FLOW_PASSES=0 EMAPPER_EXACT_AREA_PASSES=0 EMAPPER_MAX_CUTS=32 EMAPPER_MAPPING_AREA_COST=physical)"
[[ -n "$line" ]] && parse delay_cuts32_phys "$line"

line="$(run_case delay_mixed EMAPPER_AREA_FLOW_PASSES=0 EMAPPER_EXACT_AREA_PASSES=0 EMAPPER_MIXED_OBJECTIVE=1)"
[[ -n "$line" ]] && parse delay_mixed "$line"
