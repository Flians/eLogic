#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
fixture_root=$(mktemp -d /tmp/emapper-tests.XXXXXX)
trap 'rm -rf "$fixture_root"' EXIT
cp emapper/libraries/asap7/{asap7_full.genlib,library.json,cells.v} "$fixture_root/"
EGM_TEST_DATA_ROOT="$fixture_root" cargo test --offline -p emapper egraph -- --test-threads=1
