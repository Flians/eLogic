#!/usr/bin/env bash
# Apply only the checked-in ePMapping patches; preserve other submodule edits.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE="$ROOT/tools/mockturtle"
for PATCH in "$ROOT/patches/mockturtle/tech-library-function-buckets.patch" \
             "$ROOT/patches/mockturtle/shared-mapped-library.patch"; do
if git -C "$SUBMODULE" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "[epmapping] ${PATCH##*/} already applied"
elif git -C "$SUBMODULE" apply --check "$PATCH"; then
  git -C "$SUBMODULE" apply "$PATCH"
  echo "[epmapping] applied ${PATCH##*/}"
else
  echo "[epmapping] mockturtle patch conflicts with the current checkout; existing edits were preserved." >&2
  exit 1
fi
done
