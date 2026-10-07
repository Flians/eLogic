#!/usr/bin/env bash
# One-command reproduction of every experiment behind the DATE'27 paper.
#
#   scripts/experiments/run_all.sh                      # build if needed, all stages, then report
#   scripts/experiments/run_all.sh --list               # list the stages
#   scripts/experiments/run_all.sh --only tab_main,tab_mfs3_abc
#   scripts/experiments/run_all.sh --skip runtime,lever3
#   scripts/experiments/run_all.sh --force tab_mfs3     # ignore that stage's ".done" marker
#
# Everything lands under $EPM_ROOT (default <project>/results), one directory per stage:
#
#   results/<stage>/<config>.<design>/log    engine stdout/stderr
#   results/<stage>/summary.tsv              one row per engine run (parsed area/delay/cec/rounds)
#   results/<stage>/results.csv              compare_with_paper.sh output for the sweep stages
#   results/<stage>/.done                    completion marker (redo with --force)
#   results/tables.md, results/tables.tex    aggregated tables
#
# Completed stages are resumable; an interrupted stage is rerun with a fresh summary. A full run takes hours
# (the 27-design sweeps dominate); run it in the background or under nohup.
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON="$SELF_DIR/common.sh"
PROJECT_ROOT="$(cd "$SELF_DIR/../.." && pwd)"

# Paper tables first (all built on the same 16 circuits of ASPDAC'25's Table 1), then the README's
# extended 27-design tables (not part of the paper), then the aggregation.
STAGES=(db_build tab_main tab_mfs3 tab_mfs3_abc iscas_ae iscas_iwls funnel lever1 lever1_micro lever3 runtime ablation oracle db \
        report ext27_ae ext27_iwls ext27_ai)

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; }

ONLY=""; SKIP=""; DO_BUILD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --only)     ONLY="$2"; shift ;;
    --skip)     SKIP="$2"; shift ;;
    --force)    export EPM_FORCE=1 ;;
    --dry-run)  export EPM_DRY_RUN=1 ;;
    --no-build) DO_BUILD=0 ;;
    --list)     printf '%s\n' "${STAGES[@]}"; exit 0 ;;
    -h|--help)  usage; exit 0 ;;
    all|"")     ;;
    *) echo "[e] unknown stage or option: $1 (see --list / --help)" >&2; exit 2 ;;
  esac
  shift
done

# shellcheck disable=SC1090
source "$COMMON"
mkdir -p "$EPM_ROOT"
# children (xargs workers) must see the same paths and knobs
export PROJECT_ROOT EPM_ROOT EPM_JOBS RAYON_NUM_THREADS EPM_TIMEOUT EPM_DC_BUDGET EPM_ROUNDS EPM_RETRIES EPM_BIN

# stage names must be real, so a typo fails loudly instead of running nothing
is_stage() { local x; for x in "${STAGES[@]}"; do [ "$x" = "$1" ] && return 0; done; return 1; }
validate_list() { # validate_list <option> <comma list>
  local opt="$1" list="$2" tok
  [ -z "$list" ] && return 0
  for tok in ${list//,/ }; do
    is_stage "$tok" || { echo "[e] $opt: unknown stage '$tok' (see --list)" >&2; exit 2; }
  done
}
validate_list --only "$ONLY"
validate_list --skip "$SKIP"

wanted() {
  local s="$1"
  if [ -n "$ONLY" ]; then case ",$ONLY," in *",$s,"*) ;; *) return 1 ;; esac; fi
  case ",$SKIP," in *",$s,"*) return 1 ;; esac
  return 0
}

if [ "$DO_BUILD" = 1 ]; then
  echo "== build =="
  if [ ! -x "$BIN" ]; then
    "$PROJECT_ROOT/scripts/build.sh" || { echo "[e] build failed" >&2; exit 1; }
    BIN="$(_epm_resolve_bin)"; export BIN
  fi
  echo "[i] engine: $BIN"
  command -v shasum >/dev/null && echo "[i] sha256: $(shasum -a 256 "$BIN" | cut -d' ' -f1)"
fi

# --- design sets -------------------------------------------------------------
# The paper's tables are all on these 16 circuits (ASPDAC'25's Table 1): tab:main and tab:mfs3.
PAPER16="aes_core arbiter div ethernet iwls05_i2c mem_ctrl RISC sasc simple_spi spi sqrt systemcaes systemcdes tv80 usb_funct usb_phy"
ISCAS="c432 c499 c880 c1355 c1908 c2670 c3540 c5315 c6288 c7552"
ABL11="aes_core arbiter div iwls05_i2c mem_ctrl sasc simple_spi spi systemcdes tv80 usb_phy"
# README-only extended set (the paper does not report these 27 designs).
DES27="adder bar max sin log2 div cavlc multiplier sqrt ctrl dec arbiter int2float i2c priority router sasc iwls05_i2c simple_spi voter usb_phy c6288 systemcaes square systemcdes spi mem_ctrl"

# --- helpers -----------------------------------------------------------------
# spec <stage> <config> <design> <rounds> <recipe> [extra env...] -> one spec line
spec() { # spec <stage> <config> <design> <rounds> <recipe> [extra env...]
  # No trailing blanks: BSD xargs treats a line ending in a blank as a continuation line,
  # which would merge every spec into a single invocation.
  local line; line="$(printf '%s/%s %s %s %s %s' "$EPM_ROOT/$1" "$2.$3" "$3" "$4" "$5" "${6:-}")"
  line="${line%"${line##*[![:space:]]}"}"
  printf '%s
' "$line"
}

# run_specs <stage> <specfile>: EPM_JOBS specs at a time, TSV rows appended to the stage summary.
run_specs() {
  local stage="$1" specs="$2"
  if [ "${EPM_DRY_RUN:-0}" = 1 ]; then
    echo "[dry-run] stage $stage: $(wc -l < "$specs" | tr -d ' ') runs"
    sed 's/^/DRY\t/' "$specs"
    return 0
  fi
  mkdir -p "$EPM_ROOT/$stage"
  if [ -f "$EPM_ROOT/$stage/summary.tsv" ]; then
    cp "$EPM_ROOT/$stage/summary.tsv" "$EPM_ROOT/$stage/summary.previous.tsv"
  fi
  tsv_header > "$EPM_ROOT/$stage/summary.tsv"
  xargs -P "$EPM_JOBS" -L 1 bash -c "source '$COMMON' >/dev/null 2>&1; run_engine \"\$@\"" _ \
    < "$specs" >> "$EPM_ROOT/$stage/summary.tsv"
}

run_stage() { # run_stage <stage> <specfile>
  local stage="$1" specs="$2"
  if [ "${EPM_DRY_RUN:-0}" = 1 ]; then
    run_specs "$stage" "$specs"
    return 0
  fi
  if stage_done "$stage"; then echo "[=] $stage already done (use --force to redo)"; return 0; fi
  echo "== $stage: $(wc -l < "$specs" | tr -d ' ') runs ($(date '+%H:%M:%S'))"
  run_specs "$stage" "$specs" || { echo "[e] $stage interrupted; no completion marker written" >&2; exit 1; }
  mark_done "$stage"
  echo "   done ($(date '+%H:%M:%S'))"
}

sweep_stage() { # sweep_stage <stage> <recipe> <designs...>
  local stage="$1" recipe="$2"; shift 2
  if [ "${EPM_DRY_RUN:-0}" = 1 ]; then
    sweep "$EPM_ROOT/$stage" "$EPM_TIMEOUT" "$recipe" "$@"
    return 0
  fi
  if stage_done "$stage"; then echo "[=] $stage already done (use --force to redo)"; return 0; fi
  echo "== $stage ($recipe, $(date '+%H:%M:%S'))"
  sweep "$EPM_ROOT/$stage" "$EPM_TIMEOUT" "$recipe" "$@" && mark_done "$stage"
}

# --- stages ------------------------------------------------------------------
# README's extended tables (27 designs); the paper's tables are tab_main / tab_mfs3 above.
stage_ext27_ae()   { sweep_stage ext27_ae   high_effort $DES27; }
stage_ext27_iwls() { sweep_stage ext27_iwls iwls        $DES27; }
stage_ext27_ai()   { sweep_stage ext27_ai   direct      $DES27; }
stage_iscas_ae()   { sweep_stage iscas_ae   high_effort $ISCAS; }
stage_iscas_iwls() { sweep_stage iscas_iwls iwls        $ISCAS; }

# tab:main: the paper's comparison with ASPDAC'25 over its 16 benchmarks, both recipes.
stage_tab_main() {
  local s="$EPM_ROOT/.specs_tab_main"; : > "$s"
  local d
  for d in $PAPER16; do spec tab_main "ae.r1" "$d" 1             high_effort >> "$s"; done
  for d in $PAPER16; do spec tab_main "ae"    "$d" "$EPM_ROUNDS" high_effort >> "$s"; done
  for d in $PAPER16; do spec tab_main "ai"    "$d" "$EPM_ROUNDS" direct      >> "$s"; done
  run_stage tab_main "$s"
}

# tab:mfs3: ABC's `mfs3 -a` against ePMapping from the same start. Both map the raw AIG with
# area-oriented emap using the ASAP7 subset that mfs3 accepts (40 cells, i.e. at most six inputs),
# then get one pass and 1000 s per design. The engine side and ABC's side are separate stages so
# that redrawing one does not repeat the other.
stage_tab_mfs3() {
  local s="$EPM_ROOT/.specs_mfs3"; : > "$s"
  local lib="$EPM_ROOT/tab_mfs3/asap7_${EPM_MFS3_MAXPINS:-6}in.genlib"
  mkdir -p "$(dirname "$lib")"
  if [ "${EPM_DRY_RUN:-0}" = 1 ]; then
    echo "[dry-run] restrict_genlib --in $GENLIB --max-pins ${EPM_MFS3_MAXPINS:-6} --out $lib"
  else
    python3 "$SELF_DIR/restrict_genlib.py" --in "$GENLIB" --max-pins "${EPM_MFS3_MAXPINS:-6}" --out "$lib"
  fi
  local d
  for d in $PAPER16; do spec tab_mfs3 base "$d" 0 direct "EPMAPPING_SKIP=1 EPMAPPING_WRITE_BASELINE=1 EPM_GENLIB=$lib" >> "$s"; done
  # The DATE'27 `tab:mfs3` numbers were measured with the whole-window (MFFW) rewrite ON, while
  # EPM_BASE_ENV switches it off for the paper's other stages.  Re-running the identical arm on
  # 2026-09-28 reproduced all 16 rows to two decimals only with `EPMAPPING_WINDOW_SYN=1`, so the
  # `ours` arm of this stage turns it back on (see epmapping/README.md "口径 differences" and
  # results/paper_data_check.md).
  for d in $PAPER16; do spec tab_mfs3 ours "$d" 1 direct "EPM_GENLIB=$lib EPMAPPING_WINDOW_SYN=1 EPM_TIMEOUT=${EPM_MFS3_TIMEOUT:-1000}" >> "$s"; done
  run_stage tab_mfs3 "$s"
  stage_note tab_mfs3 "engine side only: the 16 A_i baselines (EPMAPPING_WRITE_BASELINE=1) and the 16 \
one-pass runs; ABC's side of the comparison is the next stage, tab_mfs3_abc."
}

# tab:mfs3, ABC side: `mfs3 -a` from each baseline netlist that `tab_mfs3` wrote, under the same
# 1000 s cap. Records area/delay/runtime in abc.tsv so the comparison table is reproducible from the
# logs; the baseline netlist exists because the engine writes it when the run is skipped.
stage_tab_mfs3_abc() {
  local lib="$EPM_ROOT/tab_mfs3/asap7_${EPM_MFS3_MAXPINS:-6}in.genlib"
  local abc="${ABC:-abc}" tsv="$EPM_ROOT/tab_mfs3/abc.tsv" cap="${EPM_MFS3_TIMEOUT:-1000}"
  if [ "${EPM_DRY_RUN:-0}" = 1 ]; then
    echo "[dry-run] abc -c 'read_genlib $lib; read -m <baseline>/epmapping_mapped.v; mfs3 -a; print_stats' on $PAPER16"
    return 0
  fi
  if stage_done tab_mfs3_abc; then echo "[=] tab_mfs3_abc already done (use --force to redo)"; return 0; fi
  if [ ! -f "$lib" ]; then
    echo "[!] $lib is missing; run --only tab_mfs3 first"
    return 1
  fi
  echo "== tab_mfs3_abc: mfs3 -a on $PAPER16 (cap ${cap}s, $(date '+%H:%M:%S'))"
  printf 'design\tarea\tdelay\tsecs\trc\tcec\tvalidation_secs\n' > "$tsv"
  local d v t0 t1 arc a dl cec validation_secs out n=0
  for d in $PAPER16; do
    v="$EPM_ROOT/tab_mfs3/base.$d/epmapping_mapped.v"
    if [ ! -f "$v" ]; then echo "[w] $d: no baseline netlist, skipped"; continue; fi
    t0="$SECONDS"
    out="$EPM_ROOT/tab_mfs3/abc.$d.v"
    ${EPM_AWAKE[@]+"${EPM_AWAKE[@]}"} perl -e 'alarm shift; exec @ARGV' "$cap" \
        "$abc" -c "read_genlib $lib; read -m $v; mfs3 -a; print_stats; write_verilog $out" \
        > "$EPM_ROOT/tab_mfs3/abc.$d.log" 2>&1
    arc=$?
    t1=$(( SECONDS - t0 ))
    # print_stats is only reached after mfs3 returns, so a capped run has no area/delay line.
    a="$(sed -n 's/.*[^a-z]area *= *\([0-9.]*\).*/\1/p' "$EPM_ROOT/tab_mfs3/abc.$d.log" | tail -1)"
    dl="$(sed -n 's/.*[^a-z]delay *= *\([0-9.]*\).*/\1/p' "$EPM_ROOT/tab_mfs3/abc.$d.log" | tail -1)"
    cec="-"; validation_secs=0
    if [ "$arc" = 0 ] && [ -f "$out" ]; then
      t0="$SECONDS"
      perl -e 'alarm shift; exec @ARGV' "$cap" "$abc" \
        -c "read_genlib $lib; read -m $out; cec -n $BENCH/$d.aig" \
        > "$EPM_ROOT/tab_mfs3/abc.$d.cec.log" 2>&1
      validation_secs=$(( SECONDS - t0 ))
      cec=0
      if grep -q 'Networks are equivalent' "$EPM_ROOT/tab_mfs3/abc.$d.cec.log"; then cec=1; fi
    fi
    [ "$arc" = 142 ] && echo "[i] mfs3 -a on $d hit the ${cap} s cap (TO@$cap)" >> "$EPM_ROOT/tab_mfs3/abc.$d.log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$d" "${a:--}" "${dl:--}" "$t1" "$arc" "$cec" "$validation_secs" >> "$tsv"
    echo "   $d: area=${a:--} delay=${dl:--} ${t1}s rc=$arc"
    n=$(( n + 1 ))
  done
  stage_note tab_mfs3_abc "ABC mfs3 -a ran on $n designs under the ${cap} s cap; per-design area/delay/\
runtime in abc.tsv, full logs in abc.<design>.log."
  mark_done tab_mfs3_abc
}

# tab:oracle: the cut stages alone, with synthesis, and with the window fallback, on both recipes.
stage_oracle() {
  local s="$EPM_ROOT/.specs_oracle"; : > "$s"
  local rec tag d
  for rec in high_effort iwls; do
    [ "$rec" = high_effort ] && tag=ae || tag=iwls
    for d in c432 c3540 c5315 c2670; do
      spec oracle "$tag.cuts" "$d" 1 "$rec" "EPMAPPING_SYN=0 EPMAPPING_EGG=0 EPMAPPING_WINDOW_SYN=0" >> "$s"
      spec oracle "$tag.syn"  "$d" 1 "$rec" "EPMAPPING_SYN=1 EPMAPPING_EGG=0 EPMAPPING_WINDOW_SYN=0" >> "$s"
      spec oracle "$tag.win"  "$d" 1 "$rec" "EPMAPPING_SYN=1 EPMAPPING_EGG=0 EPMAPPING_WINDOW_SYN=1" >> "$s"
    done
  done
  run_stage oracle "$s"
}

# Database variants (supergate libraries). Producing one costs minutes, so `db_build` runs once and
# writes them next to the shipped libraries; every later run reuses the files, and the measurement
# stage (`db`) never generates anything.
EPM_DB_DIR="${EPM_DB_DIR:-$PROJECT_ROOT/epmapping/db}"

stage_db_build() {
  local dd="$EPM_DB_DIR" ABC="${ABC:-/opt/homebrew/bin/abc}"
  if [ "${EPM_DRY_RUN:-0}" = 1 ]; then
    echo "[dry-run] db_build: abc -c 'super -I 4 -L 3 -A <cap> -T 240 -o asap7.genlib' for a045/a055/a060 into $dd (only if missing)"
    return 0
  fi
  if stage_done db_build; then echo "[=] db_build already done (use --force to redo)"; return 0; fi
  mkdir -p "$dd" "$EPM_ROOT/db_build"
  local g; g="$(mktemp -d "${TMPDIR:-/tmp}/epm-dbgen.XXXXXX")"
  cp "$GENLIB" "$g/asap7.genlib"
  local A out f
  for A in 0.45 0.55 0.60; do
    out="$dd/asap7_4in_3lvl_a${A/./}.super"
    if [ -f "$out" ]; then echo "[=] $(basename "$out") exists ($(du -h "$out" | cut -f1)), reused"; continue; fi
    echo "== db_build: abc super -A $A -> $(basename "$out") ($(date '+%H:%M:%S'))"
    ( cd "$g" && rm -f ./*.super && "$ABC" -c "super -I 4 -L 3 -A $A -T 240 -o asap7.genlib" ) > "$g/gen_a$A.log" 2>&1
    f="$(ls -S "$g"/*.super 2>/dev/null | head -1)"
    if [ -n "$f" ]; then mv "$f" "$out"; else echo "[e] abc super -A $A produced no library (see $g/gen_a$A.log)" >&2; fi
  done
  rm -rf "$g"
  { printf 'file\tsize_bytes\tsha256\n'
    for f in "$dd"/*.super; do
      printf '%s\t%s\t%s\n' "$(basename "$f")" "$(wc -c < "$f" | tr -d ' ')" "$(shasum -a 256 "$f" | cut -d' ' -f1)"
    done
  } > "$EPM_ROOT/db_build/libraries.tsv"
  mark_done db_build
  echo "[i] databases in $dd:"; ls -lh "$dd"/*.super | awk '{print "    " $5, $9}'
}

# tab:db: c7552 against six database variants -- single cells only (`none`), the two-level database,
# and the three-level ones at four area caps. Pure measurement: it never generates a library.
stage_db() {
  local dd="$EPM_DB_DIR" s="$EPM_ROOT/.specs_db"; mkdir -p "$dd"; : > "$s"
  local dbs="none $dd/asap7_4in_2lvl.super $dd/asap7_4in_3lvl_a045.super $dd/asap7_4in_3lvl_a050.super $dd/asap7_4in_3lvl_a055.super $dd/asap7_4in_3lvl_a060.super"
  local db name sup rec tag missing=0
  for db in $dbs; do
    [ "$db" = none ] && continue
    if [ ! -f "$db" ]; then echo "[e] missing database $db -- run the db_build stage first" >&2; missing=1; fi
  done
  [ "$missing" = 1 ] && return 1
  for db in $dbs; do
    name="$( [ "$db" = none ] && echo none || basename "$db" .super )"
    [ "$db" = none ] && sup="EPMAPPING_SUPER=off" || sup="EPMAPPING_SUPER=$db"
    for rec in high_effort iwls; do
      [ "$rec" = high_effort ] && tag=ae || tag=iwls
      spec db "$tag.$name.stats" c7552 1             "$rec" "$sup EPMAPPING_SKIP=1 EPMAPPING_DB_STATS=1" >> "$s"
      spec db "$tag.$name"       c7552 "$EPM_ROUNDS" "$rec" "$sup EPMAPPING_DB_STATS=1" >> "$s"
    done
  done
  run_stage db "$s"
}

# fig:funnel: one pass, with the whole-run wall clock in the "secs" column.
stage_funnel() {
  local s="$EPM_ROOT/.specs_funnel"; : > "$s"
  local d
  for d in c7552 sin div multiplier log2; do spec funnel r1 "$d" 1 high_effort >> "$s"; done
  run_stage funnel "$s"
}

# Lever 1: the SAT refinement and the window-cap knob.
stage_lever1() {
  local s="$EPM_ROOT/.specs_lever1"; : > "$s"
  local d cap
  for d in sasc arbiter; do
    spec lever1 sat_on  "$d" 1 high_effort >> "$s"
    spec lever1 sat_off "$d" 1 high_effort "EPMAPPING_SAT_DC=0" >> "$s"
  done
  for cap in 0 2 8; do spec lever1 "cap$cap" sasc 1 high_effort "EPMAPPING_DC_MAX_PIS=$cap" >> "$s"; done
  run_stage lever1 "$s"
}

# Lever 1, the micro-benchmarks: the door of the window (reconvergence cap), the window scheme and
# the don't-care budget, each one knob at a time.
stage_lever1_micro() {
  local s="$EPM_ROOT/.specs_lever1_micro"; : > "$s"
  local d cap
  for cap in 8 10; do spec lever1_micro "cap$cap" sin 1 high_effort "EPMAPPING_ROUNDS=1 EPMAPPING_DC_MAX_PIS=$cap" >> "$s"; done
  for cap in 4 12; do spec lever1_micro "free$cap" sasc 1 high_effort "EPMAPPING_ROUNDS=1 EPMAPPING_DC_FREE=$cap" >> "$s"; done
  # the "exact refinement" arm: lift the SAT budget so the same window is refined without pruning
  spec lever1_micro exact div 1 high_effort "EPMAPPING_ROUNDS=1 EPMAPPING_SAT_DC_CALLS=100000000 EPMAPPING_SAT_DC_MAX_NODES=100000000" >> "$s"
  spec lever1_micro bounded div 1 high_effort "EPMAPPING_ROUNDS=1" >> "$s"
  for d in $ABL11; do spec lever1_micro per_cut "$d" 1 high_effort "EPMAPPING_ROUNDS=1 EPMAPPING_DC_WINDOW=per_cut" >> "$s"; done
  for d in $ABL11; do spec lever1_micro per_node "$d" 1 high_effort "EPMAPPING_ROUNDS=1" >> "$s"; done
  run_stage lever1_micro "$s"
}

# Lever 3: searching harder (saturation budget, node limit) and the full configuration.
stage_lever3() {
  local s="$EPM_ROOT/.specs_lever3"; : > "$s"
  local d
  for d in c7552 c3540 c5315 c2670; do
    spec lever3 base     "$d" 1             high_effort >> "$s"
    spec lever3 iter5    "$d" 1             high_effort "MAP_EGG_EGG_ITER_LIMIT=5" >> "$s"
    spec lever3 iter10   "$d" 1             high_effort "MAP_EGG_EGG_ITER_LIMIT=10" >> "$s"
    spec lever3 node2000 "$d" 1             high_effort "MAP_EGG_EGG_NODE_LIMIT=2000" >> "$s"
    spec lever3 bundle   "$d" 1             high_effort "MAP_EGG_FULL_BUNDLE=1" >> "$s"
    spec lever3 full     "$d" "$EPM_ROUNDS" high_effort >> "$s"
  done
  run_stage lever3 "$s"
}

# Where the runtime goes: one pass with and without the e-graph, plus convergence.
stage_runtime() {
  local s="$EPM_ROOT/.specs_runtime"; : > "$s"
  local d
  for d in sqrt div mem_ctrl; do
    spec runtime full  "$d" 1             high_effort >> "$s"
    spec runtime noegg "$d" 1             high_effort "EPMAPPING_EGG=0" >> "$s"
    spec runtime rN    "$d" "$EPM_ROUNDS" high_effort >> "$s"
  done
  run_stage runtime "$s"
}

# Lever 2: the stage ablation over the eleven designs the paper reports.
stage_ablation() {
  local s="$EPM_ROOT/.specs_abl"; : > "$s"
  local d
  for d in $ABL11; do
    # The ablation studies the window as one of the removable stages, so its arms set the window on
    # again (`EPM_BASE_ENV` runs the paper's stages with it off): without the explicit `=1` here the
    # `noWINDOW` arm would measure nothing.
    spec ablation full     "$d" "$EPM_ROUNDS" high_effort "EPMAPPING_WINDOW_SYN=1" >> "$s"
    spec ablation noSYN    "$d" "$EPM_ROUNDS" high_effort "EPMAPPING_WINDOW_SYN=1 EPMAPPING_SYN=0" >> "$s"
    spec ablation noWINDOW "$d" "$EPM_ROUNDS" high_effort "EPMAPPING_WINDOW_SYN=0" >> "$s"
    spec ablation noEGG    "$d" "$EPM_ROUNDS" high_effort "EPMAPPING_WINDOW_SYN=1 EPMAPPING_EGG=0" >> "$s"
    spec ablation noCARE   "$d" "$EPM_ROUNDS" high_effort "EPMAPPING_WINDOW_SYN=1 EPMAPPING_DC=0 EPMAPPING_SAT_DC=0" >> "$s"
    spec ablation noDB     "$d" "$EPM_ROUNDS" high_effort "EPMAPPING_WINDOW_SYN=1 EPMAPPING_SUPER=off" >> "$s"
  done
  run_stage ablation "$s"
}

stage_report() {
  echo "== report ($(date '+%H:%M:%S'))"
  python3 "$SELF_DIR/report.py" --root "$EPM_ROOT" \
      --markdown "$EPM_ROOT/tables.md" --latex "$EPM_ROOT/tables.tex" \
    || echo "[w] report.py failed; the raw stage data is still under $EPM_ROOT"
}

# --- dispatch ----------------------------------------------------------------
for st in "${STAGES[@]}"; do
  wanted "$st" || continue
  "stage_$st"
done
echo "[i] done ($(date '+%H:%M:%S')); data under $EPM_ROOT"
