#!/usr/bin/env bash
# Generic alternative initial-AIG flow; post-mapping remains constrained and cut-only.
# Usage: run_structural_input.sh <output-root> <design> [syn4|sopb]
set -euo pipefail
out="${1:?output root}"; design="${2:?benchmark name}"; preset="${3:-syn4}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
export EPM_ROOT="$out"
case "$preset" in
  syn4) round='fraig; &get; &syn4; &put; compress2rs; resyn2rs' ;;
  sopb) round='fraig; &get; &sopb; &put; compress2rs; resyn2rs' ;;
  *) echo 'preset must be syn4 or sopb' >&2; exit 2 ;;
esac
export EPM_TIMEOUT="${EPM_TIMEOUT:-3600}"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mkdir -p "$out"
tsv_header
run_engine "$out/$preset.$design" "$design" 6 high_effort \
  'EPMAPPING_OBJECTIVE=area_delay_bound' 'EPMAPPING_AREA_UNDER_DELAY=1' \
  'EPMAPPING_POST_TIMEOUT=1000' 'EPMAPPING_WINDOW_SYN=0' 'EPMAPPING_DSE=0' \
  'EPMAPPING_SAT_DC=0' 'EPMAPPING_EGG_REQ=1' 'EPMAPPING_WRITE_BASELINE=1' \
  "EPMAPPING_ABC_ROUND=$round"
