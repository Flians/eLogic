#!/usr/bin/env bash
# One design, three configs (baseline / single round / iterate), for the paper-style comparison.
# Prints one CSV line. Used by compare_with_paper.sh via xargs -P.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${EPM_BIN:-$ROOT/build/epmapping/epmapping_check}"
GENLIB="$ROOT/tools/mockturtle/experiments/cell_libraries/asap7.genlib"
BENCH="$ROOT/tools/mockturtle/experiments/benchmarks"

design="${1:?design}"; outdir="${2:?outdir}"; tmo="${3:-300}"
# One work directory per design: the Verilog hand-off (`epmapping_mapped.v`) has a fixed name, so
# designs running in parallel must not share a directory.
outdir="$outdir/$design"
mkdir -p "$outdir"
aig="$BENCH/$design.aig"
if [ ! -f "$aig" ]; then echo "$design,,,,,,,,,,,MISSING"; exit 0; fi

export EPMAPPING_AREA_MAPPING=1 EPMAPPING_RELAX_REQUIRED=100 EPMAPPING_OBJECTIVE=1
# Subject-graph preprocessing. The engine default is the high-effort baseline (`EPMAPPING_BASELINE=
# high_effort`: `read_aiger` + `fraig; resyn2rs; fraig; compress2rs` repeated until the AIG stops
# shrinking + area-oriented `emap`), which is what the ASPDAC'25 comparison is reported against, so it
# is left to the engine here. The alternatives are `EPMAPPING_BASELINE=direct` (`read_aiger` + emap,
# no ABC optimization) and a single-pass script through `EPMAPPING_ABC_SCRIPT`, e.g. the IWLS'24 flow
# `EPMAPPING_ABC_SCRIPT="fraig; compress2rs; resyn2rs; compress2rs"` or the one-pass ASPDAC'25 flow
# `"fraig; rw; rs; rf; resyn2rs; compress2rs"`; both imply `EPMAPPING_BASELINE=custom`.
# Recommended engine configuration: pre-mapped multi-cell sub-network database
# (`epmapping/db/*.super`, generated with ABC's `super` command; see README). It is worth ~0.2-0.25%
# area on the designs where it fires and costs ~2 s per run. The optional SAT sharpener
# (`EPMAPPING_SAT_DC=1`) buys at most ~0.09% for up to a minute of don't-care time per design, so it
# stays off, and the legacy `EPMAPPING_DC` switch is no longer read by the engine. Either can be
# overridden from the environment.
if [ -z "${EPMAPPING_SUPER:-}" ] && [ -f "$ROOT/epmapping/db/asap7_4in_3lvl_a050.super" ]; then
  export EPMAPPING_SUPER="$ROOT/epmapping/db/asap7_4in_3lvl_a050.super"
fi
# `EPMAPPING_SUPER=off` explicitly disables the database (fast exploration runs: the ~34 s load
# dominates short sweeps and the database only changes *which* cuts reach the later stages).
if [ "${EPMAPPING_SUPER:-}" = "off" ]; then
  unset EPMAPPING_SUPER
fi
# Dependency cuts (IWLS'24 III-B) were dropped with the folded engine: `EPMAPPING_DEPCUT` is no longer
# read, and every candidate now stays inside the target cut's cone.

# Design-space exploration cycle (the paper's Section IV-D): unmap the optimized mapped netlist to an
# AIG, re-optimize it, map and optimize again, and keep the smaller of the two results. The chosen
# netlist is always CEC-checked against the *original* benchmark.
dse_run() { # dse_run <tag> <extra env...>
  # One or more design-space-exploration cycles. Each cycle tries every unmap pipeline in
  # EPMAPPING_DSE_PIPES, keeps the best remapped+optimized result, and the next cycle starts from it.
  local tag="$1"; shift
  [ "${EPMAPPING_DSE:-1}" = "0" ] && return 1
  local cycles="${EPMAPPING_DSE_CYCLES:-1}"
  local pipes="${EPMAPPING_DSE_PIPES:-plain dc2}"
  local cur_v="$outdir/epmapping_mapped.v" cur_a
  cur_a=$(sed -n 's/.*epmapping area [0-9.]* -> \([0-9.]*\).*/\1/p' "$outdir/${design}.${tag}.log" 2>/dev/null | head -1)
  [ -f "$cur_v" ] && [ -n "$cur_a" ] || return 1
  local c p ops w a best_a best_dir
  for c in $(seq 1 "$cycles"); do
    best_a="$cur_a"; best_dir=""
    for p in $pipes; do
      case "$p" in
        plain) ops="fraig; compress2rs; resyn2rs" ;;
        dc2)   ops="fraig; dc2; compress2rs; resyn2rs" ;;
        bal)   ops="fraig; balance; compress2rs; resyn2rs" ;;
        more)  ops="fraig; compress2rs; resyn2rs; compress2rs; resyn2rs" ;;
        # Structural synthesis (ABC `&syn4` = SOP balancing + resubstitution) before re-mapping.
        # This is the mechanism that reaches the ASPDAC'25 numbers on `arbiter` (557.7 -> 272.6, their
        # 268.8) and `mem_ctrl` (1895 -> 1640, their 1799); the local candidate set finds ~0 there.
        syn4)  ops="fraig; &get; &syn4; &put; compress2rs; resyn2rs" ;;
        sopb)  ops="fraig; &get; &sopb; &put; compress2rs; resyn2rs" ;;
        *)     ops="$p" ;;
      esac
      w="$outdir/dse${c}_${p}"; rm -rf "$w"; mkdir -p "$w"
      rm -f "$outdir/${design}.dse${c}_${p}.aig"
      abc -c "read_genlib $GENLIB; read -m $cur_v; strash; $ops; write_aiger $outdir/${design}.dse${c}_${p}.aig; quit" \
          >"$outdir/${design}.${tag}.dse${c}_${p}.abc" 2>&1
      [ -f "$outdir/${design}.dse${c}_${p}.aig" ] || continue
      perl -e 'alarm shift; exec @ARGV' "$tmo" env "$@" "$BIN" "$GENLIB" \
          "$outdir/${design}.dse${c}_${p}.aig" "$w" >"$w/run.log" 2>&1
      a=$(sed -n 's/.*epmapping area [0-9.]* -> \([0-9.]*\).*/\1/p' "$w/run.log" 2>/dev/null | head -1)
      [ -n "$a" ] || continue
      if python3 -c "import sys; sys.exit(0 if float('$a') < float('$best_a') else 1)" 2>/dev/null; then
        best_a="$a"; best_dir="$w"
      fi
    done
    [ -n "$best_dir" ] || break
    cur_a="$best_a"; cur_v="$best_dir/epmapping_mapped.v"
    cp "$best_dir/run.log" "$outdir/${design}.${tag}.dse.log"
    echo "pipeline $(basename $best_dir)" > "$outdir/${design}.${tag}.dse.pick"
    rm -rf "$outdir/dse_final"; cp -r "$best_dir" "$outdir/dse_final"
  done
  return 0
}

dse_area() { sed -n 's/.*epmapping area [0-9.]* -> \([0-9.]*\).*/\1/p' "$outdir/${design}.${1}.dse.log" 2>/dev/null | head -1; }
dse_delay() { sed -n 's/.*delay [0-9.]* -> \([0-9.]*\).*/\1/p' "$outdir/${design}.${1}.dse.log" 2>/dev/null | head -1; }

# Does the DSE netlist CEC against the original benchmark? (the binary itself only checked it against
# the intermediate AIG).
dse_cec() {
  local vv; vv=$(ls "$outdir"/dse_final/cec_*.v 2>/dev/null | head -1)
  [ -n "$vv" ] || { echo 0; return; }
  local out; out=$(abc -c "read_genlib $GENLIB; read -m $vv; cec -n $BENCH/$design.aig" 2>&1)
  case "$out" in *"equivalent"*) echo 1 ;; *) echo 0 ;; esac
}

run_one() { # run_one <tag> <extra env...>
  local tag="$1"; shift
  local log="$outdir/${design}.${tag}.log"
  local st=$SECONDS
  perl -e 'alarm shift; exec @ARGV' "$tmo" env "$@" "$BIN" "$GENLIB" "$aig" "$outdir" >"$log" 2>&1
  local rc=$?
  echo "$((SECONDS - st))" >"$outdir/${design}.${tag}.secs"
  echo "$rc" >"$outdir/${design}.${tag}.rc"
}

run_one base EPMAPPING_SKIP=1 EPMAPPING_BASELINE_CEC=1
run_one r1 EPMAPPING_ROUNDS=1
dse_run r1 EPMAPPING_ROUNDS=1
run_one rN EPMAPPING_ROUNDS=6
dse_run rN EPMAPPING_ROUNDS=6

# Keep the better of the direct and the design-space-explored result for each configuration.
pick() { # pick <tag>
  # Take the design-space-explored result only when it is smaller *and* (under
  # EPMAPPING_NO_DELAY_REGRESSION=1) not slower than the direct one.
  local tag="$1" a b da db
  a=$(sed -n 's/.*epmapping area [0-9.]* -> \([0-9.]*\).*/\1/p' "$outdir/${design}.${tag}.log" 2>/dev/null | head -1)
  da=$(sed -n 's/.*delay [0-9.]* -> \([0-9.]*\).*/\1/p' "$outdir/${design}.${tag}.log" 2>/dev/null | head -1)
  b=$(dse_area "$tag"); db=$(dse_delay "$tag")
  if [ -n "$b" ] && [ -n "$a" ] && A="$a" B="$b" DA="$da" DB="$db" NDR="${EPMAPPING_NO_DELAY_REGRESSION:-0}" \
     python3 -c "import os,sys
a=float(os.environ['A']); b=float(os.environ['B'])
if b >= a: sys.exit(1)
if os.environ.get('NDR') == '1':
    da=os.environ.get('DA') or ''; db=os.environ.get('DB') or ''
    if da and db and float(db) > float(da) + 1e-6: sys.exit(1)
sys.exit(0)" 2>/dev/null; then
    echo "dse"
  else
    echo "direct"
  fi
}
picked_1=$(pick r1); picked_n=$(pick rN)
a1=$( [ "$picked_1" = dse ] && dse_area r1 || sed -n 's/.*epmapping area [0-9.]* -> \([0-9.]*\).*/\1/p' "$outdir/${design}.r1.log" | head -1)
d1=$( [ "$picked_1" = dse ] && dse_delay r1 || sed -n 's/.*delay [0-9.]* -> \([0-9.]*\).*/\1/p' "$outdir/${design}.r1.log" | head -1)
c1=$( [ "$picked_1" = dse ] && dse_cec || sed -n 's/.*cec=\([01]\).*/\1/p' "$outdir/${design}.r1.log" | head -1)
an=$( [ "$picked_n" = dse ] && dse_area rN || sed -n 's/.*epmapping area [0-9.]* -> \([0-9.]*\).*/\1/p' "$outdir/${design}.rN.log" | head -1)
dn=$( [ "$picked_n" = dse ] && dse_delay rN || sed -n 's/.*delay [0-9.]* -> \([0-9.]*\).*/\1/p' "$outdir/${design}.rN.log" | head -1)
cn=$( [ "$picked_n" = dse ] && dse_cec || sed -n 's/.*cec=\([01]\).*/\1/p' "$outdir/${design}.rN.log" | head -1)

field() { sed -n "$2" "$outdir/${design}.${1}.log" 2>/dev/null | head -1; }
area_base()  { field "$1" 's/.*emap area=\([0-9.]*\).*/\1/p'; }
delay_base() { field "$1" 's/.*emap area=[0-9.]* delay=\([0-9.]*\).*/\1/p'; }
area_opt()   { field "$1" 's/.*epmapping area [0-9.]* -> \([0-9.]*\).*/\1/p'; }
delay_opt()  { field "$1" 's/.*delay [0-9.]* -> \([0-9.]*\).*/\1/p'; }

ba=$(area_base base); bd=$(delay_base base)
ra="$a1"; rd="$d1"
na="$an"; nd="$dn"
rnd=$(field rN 's/.*epmapping rounds=\([0-9]*\).*/\1/p')
cec_b=$(field base 's/.*baseline_cec=\([0-9]*\).*/\1/p')
cec_1="$c1"
cec_n="$cn"
bs=$(cat "$outdir/${design}.base.secs" 2>/dev/null)
s1=$(cat "$outdir/${design}.r1.secs" 2>/dev/null)
sN=$(cat "$outdir/${design}.rN.secs" 2>/dev/null)
status="OK"
[ -n "$na" ] || status="TIMEOUT"
[ -n "$ba" ] || status="BASE_FAIL"

echo "$design,$ba,$bd,$ra,$rd,$na,$nd,$rnd,$cec_b,$cec_1,$cec_n,$bs,$s1,$sN,$status"
