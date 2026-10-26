#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
cargo build --locked --release --bin emapper_kernel --bin emapper_measure -j 2 "$@"
python3 scripts/emapper/provenance.py
