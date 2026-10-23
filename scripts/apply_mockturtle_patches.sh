#!/usr/bin/env bash
# Compatibility entry point for archived ePMapping commands.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/tools/apply_mockturtle_overlay.py" "$@"
