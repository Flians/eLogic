#!/usr/bin/env bash
# End-to-end smoke test on one small benchmark (ex212, 11 inputs / 16 outputs).
# Expected wall-clock: ~30s on a recent machine. cec should report equivalent.
#
# Required env vars:
#   ABC               path to abc binary (default: 'abc' in PATH)
#   ELOGIC_REPO_DIR   absolute path to your eLogic clone (must contain
#                     a release-built mig_egg test binary)

set -euo pipefail

PKG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

: "${ABC:=abc}"
: "${ELOGIC_REPO_DIR:?ELOGIC_REPO_DIR not set}"

export ABC ELOGIC_REPO_DIR

echo "==> ABC=$ABC"
echo "==> ELOGIC_REPO_DIR=$ELOGIC_REPO_DIR"
echo "==> running ex212"

python3 "$PKG_ROOT/scripts/run_one_bench.py" ex212 \
    --out-root "$PKG_ROOT/results/smoke" \
    --simplify-timeout 120 \
    --cec-timeout 300

echo
echo "==> result JSON:"
cat "$PKG_ROOT/results/smoke/results/ex212.json"
echo
echo "==> AIGs produced:"
ls -la "$PKG_ROOT/results/smoke/aigs/"
