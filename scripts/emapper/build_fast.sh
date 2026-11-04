#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
cargo build --locked --release --bin emapper_fast --bin emapper_measure -j 2 "$@"
python3 - <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('scripts/emapper').resolve()))
from provenance import source_files,hashes,git_revision
extra=[Path('src/bin/emapper_fast.rs').resolve(),Path('scripts/emapper/build_fast.sh').resolve(),Path('scripts/emapper/run_fast.py').resolve(),Path('scripts/emapper/physical.py').resolve(),Path('scripts/emapper/verify.py').resolve(),Path('scripts/emapper/provenance.py').resolve()]
r=dict(source=hashes(source_files()+extra),binaries=hashes([Path('target/release/emapper_fast').resolve(),Path('target/release/emapper_measure').resolve()]),git_revision=git_revision(),scope='experimental native fast; no external mapper seed; not the old600s profile')
Path('target/release/emapper-fast-build.json').write_text(json.dumps(r,indent=2)+'\n')
PY
