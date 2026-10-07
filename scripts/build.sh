#!/usr/bin/env bash
# Build ePMapping (C++ engine + Rust e-graph oracle) in one command.
#
#   scripts/build.sh                 # Release build into ./build
#   scripts/build.sh --debug         # Debug build into ./build-debug
#   scripts/build.sh --clean         # remove the build directory first
#   scripts/build.sh -j 8            # limit parallelism (default: all cores)
#   BUILD_DIR=out scripts/build.sh   # explicit build directory
#
# Requires: cmake >= 3.16, a C++17 compiler, cargo/rustc (for map_egg), and the
# ABC binary used by the experiments (brew install berkeley-abc; see README).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_TYPE=Release
CLEAN=0
JOBS="$( { sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4; } )"
while [ $# -gt 0 ]; do
  case "$1" in
    --debug)     BUILD_TYPE=Debug ;;
    --release)   BUILD_TYPE=Release ;;
    --clean)     CLEAN=1 ;;
    -j|--jobs)   JOBS="$2"; shift ;;
    -j*)         JOBS="${1#-j}" ;;
    -h|--help)   sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)           echo "[e] unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ "$BUILD_TYPE" = Debug ]; then
  BUILD_DIR="${BUILD_DIR:-build-debug}"
else
  BUILD_DIR="${BUILD_DIR:-build}"
fi

if [ "$CLEAN" = 1 ]; then
  echo "[i] removing $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

mkdir -p "$ROOT/$BUILD_DIR"
echo "[i] configuring  $BUILD_DIR ($BUILD_TYPE, -j$JOBS)"
cmake -S "$ROOT" -B "$ROOT/$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" > "$ROOT/$BUILD_DIR/configure.log" 2>&1 || {
  tail -30 "$ROOT/$BUILD_DIR/configure.log" >&2; exit 1; }

echo "[i] building     epmapping_check"
cmake --build "$ROOT/$BUILD_DIR" --target epmapping_check -j "$JOBS" 2>&1 | tail -5

BIN="$ROOT/$BUILD_DIR/epmapping/epmapping_check"
[ -x "$BIN" ] || { echo "[e] build produced no binary at $BIN" >&2; exit 1; }

# The experiments read the engine path from EPM_BIN; record it so a later
# `source scripts/experiments/common.sh` picks up the same build.
printf '%s\n' "$BIN" > "$ROOT/$BUILD_DIR/.epm_bin"
echo "[i] binary:     $BIN"
if command -v shasum >/dev/null; then
  echo "[i] sha256:     $(shasum -a 256 "$BIN" | cut -d' ' -f1)"
fi
echo "[i] done"
