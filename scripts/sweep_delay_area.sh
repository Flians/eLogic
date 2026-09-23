#!/usr/bin/env bash
set -euo pipefail

# Quick QoR sweep for MIGBalance mapper settings.
# For meaningful delay differences, use c432 (or larger); c17/adder are too small.
# Usage:
#   scripts/sweep_delay_area.sh [AIG_PATH]
# Example:
#   scripts/sweep_delay_area.sh tools/mockturtle/experiments/benchmarks/c432.aig

AIG_PATH="${1:-tools/mockturtle/experiments/benchmarks/c432.aig}"
GENLIB_PATH="${GENLIB_PATH:-tools/mockturtle/experiments/cell_libraries/asap7.genlib}"

if [[ ! -f "$AIG_PATH" ]]; then
  echo "AIG not found: $AIG_PATH" >&2
  exit 1
fi
if [[ ! -f "$GENLIB_PATH" ]]; then
  echo "GENLIB not found: $GENLIB_PATH" >&2
  exit 1
fi

run_case() {
  local name="$1"
  shift
  echo "== $name =="
  local out
  set +e
  out="$(env MIGBALANCE_AIG="$AIG_PATH" "$@" cargo run --package MIGBalance --bin MIGBalance 2>&1)"
  local rc=$?
  set -e
  local qor
  qor="$(printf "%s\n" "$out" | awk '/mapper internal QoR:/ {print; exit}')"
  if [[ -z "$qor" ]]; then
    echo "FAILED to parse QoR line (exit=$rc)"
    return 1
  fi
  local area delay
  area="$(printf "%s\n" "$qor" | sed -E 's/.*total_area = ([0-9.]+), max_delay = ([0-9.]+).*/\1/')"
  delay="$(printf "%s\n" "$qor" | sed -E 's/.*total_area = ([0-9.]+), max_delay = ([0-9.]+).*/\2/')"
  printf "  area=%s delay=%s (exit=%d)\n" "$area" "$delay" "$rc"
  printf "%s\t%s\t%s\n" "$name" "$area" "$delay"
}

echo "Sweeping on AIG=$AIG_PATH GENLIB=$GENLIB_PATH"
echo
printf "case\tarea\tdelay\n"

run_case baseline \
  MIGBALANCE_MAX_INPUTS=9 \
  MIGBALANCE_MAX_CUTS=16 \
  MIGBALANCE_AREA_FLOW_PASSES=3 \
  MIGBALANCE_EXACT_AREA_PASSES=2 \
  MIGBALANCE_MAPPING_AREA_COST=flow

run_case fast_delay_a \
  MIGBALANCE_MAX_INPUTS=9 \
  MIGBALANCE_MAX_CUTS=16 \
  MIGBALANCE_AREA_FLOW_PASSES=1 \
  MIGBALANCE_EXACT_AREA_PASSES=0 \
  MIGBALANCE_MAPPING_AREA_COST=flow

run_case fast_delay_b \
  MIGBALANCE_MAX_INPUTS=9 \
  MIGBALANCE_MAX_CUTS=32 \
  MIGBALANCE_AREA_FLOW_PASSES=1 \
  MIGBALANCE_EXACT_AREA_PASSES=0 \
  MIGBALANCE_MAPPING_AREA_COST=flow

run_case fast_delay_c \
  MIGBALANCE_MAX_INPUTS=9 \
  MIGBALANCE_MAX_CUTS=32 \
  MIGBALANCE_AREA_FLOW_PASSES=1 \
  MIGBALANCE_EXACT_AREA_PASSES=0 \
  MIGBALANCE_MAPPING_AREA_COST=physical
