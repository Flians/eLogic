#!/usr/bin/env bash
# Run the full IWLS contest set (100 benchmarks) through the eLogic pipeline.
# Wall clock: ~30 min on 16 cores for the easier benches; long-tail bench
# (ex299, ex230, etc.) may hit simplify-timeout.
#
# Required env vars:
#   ABC               path to abc binary (default: 'abc')
#   ELOGIC_REPO_DIR   absolute path to your eLogic clone

set -euo pipefail

PKG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

: "${ABC:=abc}"
: "${ELOGIC_REPO_DIR:?ELOGIC_REPO_DIR not set}"

export ABC ELOGIC_REPO_DIR

WORKERS="${WORKERS:-16}"
RUN_ID="${RUN_ID:-full}"
SIMPLIFY_TIMEOUT="${SIMPLIFY_TIMEOUT:-900}"
CEC_TIMEOUT="${CEC_TIMEOUT:-1800}"

echo "==> running full 100-bench (run-id=$RUN_ID, workers=$WORKERS)"
python3 "$PKG_ROOT/scripts/run_all.py" \
    --run-id "$RUN_ID" \
    --workers "$WORKERS" \
    --simplify-timeout "$SIMPLIFY_TIMEOUT" \
    --cec-timeout "$CEC_TIMEOUT" \
    --resume

echo
echo "==> finalizing (Pareto + cec verify + zip)"
python3 "$PKG_ROOT/scripts/finalize.py" --runs "$RUN_ID"

echo
echo "==> done. outputs under $PKG_ROOT/results/"
