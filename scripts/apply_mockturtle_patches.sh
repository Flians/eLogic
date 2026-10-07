#!/usr/bin/env bash
# Apply only the checked-in ePMapping patches; preserve other submodule edits.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE="$ROOT/tools/mockturtle"
PATCH="$ROOT/patches/mockturtle/tech-library-function-buckets.patch"
if git -C "$SUBMODULE" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "[epmapping] mockturtle patch already applied"
elif git -C "$SUBMODULE" apply --check "$PATCH"; then
  git -C "$SUBMODULE" apply "$PATCH"
  echo "[epmapping] applied mockturtle function-bucket filtering"
else
  echo "[epmapping] mockturtle patch conflicts with the current checkout; existing edits were preserved." >&2
  exit 1
fi
