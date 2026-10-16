#!/usr/bin/env bash
# Rebuild and reproduce the paired three-part experiment; choose a fresh root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RESULT_ROOT="${EPM_ROOT:-$ROOT/results_reproduced_query_budget}"
ORIGINAL_ROOT="${EPM_ORIGINAL_ROOT:-$ROOT/results_v20_three_parts_chain}"
bash scripts/build.sh -j "${EPM_BUILD_JOBS:-4}"
python3 scripts/experiments/run_three_parts_paired.py --root "$RESULT_ROOT" --original-root "$ORIGINAL_ROOT" --jobs "${EPM_JOBS:-6}"
python3 scripts/experiments/finalize_build_campaign.py --root "$RESULT_ROOT"
# Final delivery requires visual inspection of the rendered 6+1 PDF.
