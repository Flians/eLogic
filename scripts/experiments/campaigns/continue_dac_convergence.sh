#!/usr/bin/env bash
# A finite serial queue, not a schedule. Every case keeps the same 600s policy.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p results_v2/emapper_dac_convergence_20261111
exec 9>results_v2/emapper_dac_convergence_20261111/continuation.lock
flock -n 9
taskset -c 2 python3 scripts/experiments/campaigns/run_dac_convergence.py --phase representatives --max-jobs 8
taskset -c 2 python3 scripts/experiments/campaigns/run_dac_convergence.py --phase repeats --max-jobs 8
taskset -c 2 python3 scripts/experiments/campaigns/run_dac_frozen_epfl.py --max-jobs 20
