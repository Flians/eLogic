#!/usr/bin/env bash
# Rebuild, reproduce the paper's data, verify ABC outputs and export numeric LaTeX.
# EPM_ROOT selects a fresh/resumable root. Completed stages are reused; interrupted stages rerun.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export EPM_ROOT="${EPM_ROOT:-$ROOT/results_paper_verified}"
export EPM_JOBS="${EPM_JOBS:-4}" RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}" ABC="${ABC:-abc}"
export EPM_TIMEOUT="${EPM_TIMEOUT:-1800}" EPM_DC_BUDGET="${EPM_DC_BUDGET:-60}" EPM_ROUNDS="${EPM_ROUNDS:-6}"
export EPM_BIN="$ROOT/build/epmapping/epmapping_check"
bash scripts/build.sh -j "${EPM_BUILD_JOBS:-4}"
if [ -f "$EPM_ROOT/manifest.json" ]; then
  python3 - "$EPM_ROOT" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
m=json.loads((Path(sys.argv[1])/'manifest.json').read_text())
config={k:v for k,v in sorted(os.environ.items()) if k.startswith(('EPM_','EPMAPPING_','MAP_EGG_','RAYON_')) or k=='ABC'}
if config!=m.get('config'):raise SystemExit('Resume refused: experiment environment changed; select a fresh EPM_ROOT.')
for file in ('build/epmapping/epmapping_check','scripts/experiments/common.sh','scripts/experiments/run_all.sh'):
    current=hashlib.sha256(Path(file).read_bytes()).hexdigest()
    if current!=m['sha256'][file]:raise SystemExit(f'Resume refused: {file} changed; select a fresh EPM_ROOT.')
PY
else
  python3 scripts/experiments/record_manifest.py --root "$EPM_ROOT"
fi
bash scripts/experiments/run_all.sh --no-build --only tab_main,tab_mfs3,funnel,ablation,lever3,db,report
mkdir -p "$EPM_ROOT/validation"
(
  source scripts/experiments/common.sh
  tsv_header
  for d in c7552 tv80; do
    rounds=1; window=0
    if [ "$d" = tv80 ]; then rounds=6; window=1; fi
    run_engine "$EPM_ROOT/validation/cache_off.$d" "$d" "$rounds" high_effort EPMAPPING_SYN_CACHE=0 "EPMAPPING_WINDOW_SYN=$window"
    run_engine "$EPM_ROOT/validation/cache_on.$d" "$d" "$rounds" high_effort EPMAPPING_SYN_CACHE=8192 "EPMAPPING_WINDOW_SYN=$window"
  done
) > "$EPM_ROOT/validation/summary.tsv"
# Build a matched-start ABC batch from this run's own baseline files.
python3 - "$EPM_ROOT" "$ROOT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,'scripts/experiments')
from report import PAPER16
out,root=map(Path,sys.argv[1:]);lib=out/'tab_mfs3/asap7_6in.genlib'
with (out/'abc.specs').open('w') as f:
    for d in ['div','mem_ctrl','sqrt']+[d for d in PAPER16 if d not in ['div','mem_ctrl','sqrt']]:
        args=[out/'tab_mfs3'/f'abc.{d}',d,lib,out/'tab_mfs3'/f'base.{d}'/'epmapping_mapped.v',root/'tools/mockturtle/experiments/benchmarks'/f'{d}.aig',1000]
        f.write(' '.join(map(str,args))+'\n')
(out/'tab_mfs3/abc.tsv').write_text('design\tarea\tdelay\tsecs\trc\tcec\tvalidation_secs\n')
PY
xargs -P "$EPM_JOBS" -L 1 bash scripts/experiments/abc_mfs3_one.sh < "$EPM_ROOT/abc.specs" >> "$EPM_ROOT/tab_mfs3/abc.tsv"
python3 scripts/experiments/measure_abc_baseline.py --root "$EPM_ROOT"
python3 scripts/experiments/export_paper.py --root "$EPM_ROOT" --out epmapping_paper/DATE27-ePMapping-sub/doc/generated_results.tex
python3 scripts/experiments/report.py --root "$EPM_ROOT" --markdown "$EPM_ROOT/tables.md" --latex "$EPM_ROOT/tables.tex"
