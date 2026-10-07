#!/usr/bin/env bash
# Shared helpers for the experiment scripts. Source it, do not execute it:
#
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# Environment knobs (all optional, defaults match the DATE'27 paper):
#   EPM_ROOT            output root for all stage data   (default: <project>/results)
#   EPM_BIN             engine binary                    (default: <project>/build/epmapping/epmapping_check)
#   EPM_JOBS            designs in flight per sweep      (default: 3)
#   RAYON_NUM_THREADS   threads per engine process       (default: 3)
#   EPM_TIMEOUT         seconds per engine run           (default: 1800)
#   EPM_DC_BUDGET       EPMAPPING_DC_TIME_BUDGET         (default: 60)
#   EPM_ROUNDS          rounds for the converged config  (default: 6)
#   EPM_RETRIES         attempts after SIGSEGV (rc=139)  (default: 3)

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GENLIB="$PROJECT_ROOT/tools/mockturtle/experiments/cell_libraries/asap7.genlib"
BENCH="$PROJECT_ROOT/tools/mockturtle/experiments/benchmarks"
SUPER_DB="${EPMAPPING_SUPER_DB:-$PROJECT_ROOT/epmapping/db/asap7_4in_3lvl_a050.super}"

_epm_resolve_bin() {
  if [ -n "${EPM_BIN:-}" ] && [ -x "${EPM_BIN}" ]; then printf '%s\n' "$EPM_BIN"; return 0; fi
  local d
  for d in build build-debug; do
    if [ -f "$PROJECT_ROOT/$d/.epm_bin" ]; then cat "$PROJECT_ROOT/$d/.epm_bin"; return 0; fi
  done
  for d in build build-debug; do
    if [ -x "$PROJECT_ROOT/$d/epmapping/epmapping_check" ]; then
      printf '%s\n' "$PROJECT_ROOT/$d/epmapping/epmapping_check"; return 0
    fi
  done
  return 1
}
BIN="$(_epm_resolve_bin)" || {
  echo "[e] no engine binary found; run scripts/build.sh first (or set EPM_BIN)" >&2
  exit 1
}

EPM_ROOT="${EPM_ROOT:-$PROJECT_ROOT/results}"
EPM_JOBS="${EPM_JOBS:-3}"
RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-3}"
EPM_TIMEOUT="${EPM_TIMEOUT:-1800}"
EPM_DC_BUDGET="${EPM_DC_BUDGET:-60}"
EPM_ROUNDS="${EPM_ROUNDS:-6}"
EPM_RETRIES="${EPM_RETRIES:-3}"

# The settings every DATE'27 number uses on top of the engine defaults.
# `EPMAPPING_WINDOW_SYN=0`: main and funnel runs study cut rewriting alone. The matched-start
# comparison and extended ablations explicitly enable whole-window (MFFW) rewriting. This belongs
# in the driver rather than in the engine default.
# `EPMAPPING_CUT_LIMIT=12`: the paper's §Setup reports "cut size 6 and at most 12 cuts per node", but
# the engine's default is now 15 (`epmapping/epmapping.hpp`; README knob table calls 12 "the runtime
# knee"). Without this the paper's stages silently run at 15 and `systemcdes` no longer converges:
# on 2026-09-30 the same configuration took 808 s at 12 (area 130.79, the paper's row) against
# >3600 s at 15, with identical `egg_calls`/`win_calls`/`replacements` at 12.
EPM_BASE_ENV=(
  "EPMAPPING_AREA_MAPPING=1"
  "EPMAPPING_WINDOW_SYN=0"
  "EPMAPPING_CUT_LIMIT=${EPM_CUT_LIMIT:-12}"
  "EPMAPPING_RELAX_REQUIRED=100"
  "EPMAPPING_OBJECTIVE=1"
  "EPMAPPING_SUPER=$SUPER_DB"
  "EPMAPPING_DSE=0"
  "EPMAPPING_NO_DELAY_REGRESSION=1"
  "EPMAPPING_DC_TIME_BUDGET=$EPM_DC_BUDGET"
  "RAYON_NUM_THREADS=$RAYON_NUM_THREADS"
)

# recipe_set <name> fills the global RECIPE_ENV array with the subject graph selection.
#   high_effort (A_e, default) | direct (A_i) | iwls (single-pass IWLS'24 graph) | <abc script>
recipe_set() {
  case "${1:-high_effort}" in
    high_effort|ae|A_e) RECIPE_ENV=( "EPMAPPING_BASELINE=high_effort" ) ;;
    direct|ai|A_i)      RECIPE_ENV=( "EPMAPPING_BASELINE=direct" ) ;;
    iwls|single_pass)   RECIPE_ENV=( "EPMAPPING_ABC_SCRIPT=fraig; compress2rs; resyn2rs; compress2rs" ) ;;
    custom)             RECIPE_ENV=( "EPMAPPING_ABC_SCRIPT=${EPM_ABC_SCRIPT:-fraig; resyn2rs; compress2rs}" ) ;;
    *)                  RECIPE_ENV=( "EPMAPPING_ABC_SCRIPT=$1" ) ;;
  esac
}

# --- parsing -----------------------------------------------------------------
log_area_base() { sed -n 's/.*emap area=\([0-9.]*\).*/\1/p'                "$1" | head -1; }
# EPM_GENLIB overrides the technology library for a single run (e.g. the 40-cell ASAP7 variant
# used by the tab:mfs3 comparison).
log_delay_base(){ sed -n 's/.*emap area=[0-9.]* delay=\([0-9.]*\).*/\1/p'   "$1" | head -1; }
log_area_out()  { sed -n 's/.*epmapping area [0-9.]* -> \([0-9.]*\).*/\1/p' "$1" | tail -1; }
log_delay_out() { sed -n 's/.*delay [0-9.]* -> \([0-9.]*\).*/\1/p'          "$1" | tail -1; }
log_rounds()    { sed -n 's/.*epmapping rounds=\([0-9]*\).*/\1/p'           "$1" | tail -1; }
log_cec()       { sed -n 's/.*[^_]cec=\([01]\).*/\1/p'                      "$1" | tail -1; }

pct() {
  awk -v a="${1:-}" -v b="${2:-}" 'BEGIN{ if(a=="" || b=="" || a+0==0) print "n/a"; else printf "%+.2f", 100.0*(b-a)/a }'
}

# --- running -----------------------------------------------------------------
# macOS suspends an idle machine, and a suspended run is not computing: in the 2026-09-23 batch the
# machine slept through `mfs3 -a` on `sqrt`, which was then recorded as 6571 s although only ~780 s
# were spent computing (the kernel suspends `alarm` timers across sleep, so the 1000 s cap had not
# expired either). Every measured run is wrapped in `caffeinate`, so the reported runtimes and the
# caps count time the machine was actually awake. Set EPM_CAFFEINATE=0 to disable.
EPM_CAFFEINATE="${EPM_CAFFEINATE:-1}"
EPM_AWAKE=()
if [ "$EPM_CAFFEINATE" = 1 ] && command -v caffeinate >/dev/null 2>&1; then
  EPM_AWAKE=( caffeinate -i -s )
fi

# run_engine <workdir> <design> <rounds> <recipe> [extra env assignments...]
# Prints one TSV row: <config> <design> <recipe> <rounds> <base_a> <base_d> <out_a> <out_d>
#                    <dA%> <dD%> <rounds_used> <cec> <secs> <rc>
# `EPMAPPING_SUPER=off` in the extras unsets the database for that run.
run_engine() {
  # A cooperative stop prevents xargs from launching further engine jobs. Active jobs finish;
  # raw outputs remain available. Remove the marker before explicitly resuming the batch.
  if [ -f "$EPM_ROOT/.stop" ]; then
    echo "[i] batch stopped before launching another engine: $EPM_ROOT" >&2
    return 255
  fi
  local wd="$1" design="$2" rounds="$3" recipe="$4"; shift 4
  local aig="$BENCH/$design.aig"
  local cfg; cfg="$(basename "$wd")"; cfg="${cfg%.$design}"
  mkdir -p "$wd"
  if [ ! -f "$aig" ]; then printf '%s\t%s\tMISSING\n' "$cfg" "$design"; return 1; fi
  local -a extra=() unset_env=()
  local e alarm="$EPM_TIMEOUT" genlib_run="${EPM_GENLIB:-$GENLIB}"
  for e in "$@"; do
    [ -z "$e" ] && continue
    case "$e" in
      EPMAPPING_SUPER=off) unset_env+=( -u EPMAPPING_SUPER ) ;;   # unset the database for this run
      EPM_TIMEOUT=*)       alarm="${e#EPM_TIMEOUT=}" ;;            # per-run wall-clock budget
      EPM_GENLIB=*)        genlib_run="${e#EPM_GENLIB=}" ;;        # per-run technology library
      *)                   extra+=( "$e" ) ;;
    esac
  done
  recipe_set "$recipe"
  local -a round_env=()
  [ -n "$rounds" ] && round_env=( "EPMAPPING_ROUNDS=$rounds" )
  local log="$wd/log" try=0 rc=0 st
  st=$(date +%s)
  while :; do
    try=$(( try + 1 ))
    ( cd "$PROJECT_ROOT" && env ${unset_env[@]+"${unset_env[@]}"} "${EPM_BASE_ENV[@]}" "${RECIPE_ENV[@]}" \
        ${round_env[@]+"${round_env[@]}"} ${extra[@]+"${extra[@]}"} \
        ${EPM_AWAKE[@]+"${EPM_AWAKE[@]}"} perl -e 'alarm shift; exec @ARGV' "$alarm" "$BIN" "$genlib_run" "$aig" "$wd" ) > "$log" 2>&1
    rc=$?
    [ "$rc" = 139 ] && [ "$try" -lt "$EPM_RETRIES" ] && continue
    break
  done
  local secs=$(( $(date +%s) - st ))
  local ba bd oa od rnd cec
  ba=$(log_area_base "$log"); bd=$(log_delay_base "$log")
  oa=$(log_area_out "$log");  od=$(log_delay_out "$log")
  rnd=$(log_rounds "$log");   cec=$(log_cec "$log")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$cfg" "$design" "$recipe" "${rounds:-default}" \
    "${ba:--}" "${bd:--}" "${oa:--}" "${od:--}" "$(pct "$ba" "$oa")" "$(pct "$bd" "$od")" \
    "${rnd:--}" "${cec:--}" "$secs" "$rc"
}

# sweep <outdir> <timeout> <recipe> <design...>
# Wraps scripts/compare_with_paper.sh, whose results.csv has the base/r1/rN columns
# the paper tables use. EPM_ROUNDS sets the converged ("rN") round count.
sweep() {
  local out="$1" tmo="$2" recipe="$3"; shift 3
  recipe_set "$recipe"
  if [ "${EPM_DRY_RUN:-0}" = 1 ]; then
    echo "[dry-run] sweep $out  recipe=$recipe  timeout=$tmo  designs=$*"
    return 0
  fi
  mkdir -p "$out"
  ( cd "$PROJECT_ROOT" && env EPM_OUT="$out" EPM_JOBS="$EPM_JOBS" RAYON_NUM_THREADS="$RAYON_NUM_THREADS" \
      EPMAPPING_DSE=0 EPMAPPING_NO_DELAY_REGRESSION=1 EPMAPPING_DC_TIME_BUDGET="$EPM_DC_BUDGET" \
      EPMAPPING_RN_ROUNDS="$EPM_ROUNDS" "${RECIPE_ENV[@]}" \
      ./scripts/compare_with_paper.sh "$tmo" "$@" ) > "$out.table" 2>&1
  local rc=$?
  echo "[i] swept $out (rc=$rc, $(wc -l < "$out/results.csv" 2>/dev/null | tr -d ' ') designs)"
  return $rc
}

# --- stage bookkeeping -------------------------------------------------------
tsv_header() {
  printf 'stage\tdesign\trecipe\trounds\tbase_a\tbase_d\tout_a\tout_d\tdA%%\tdD%%\trnd\tcec\tsecs\trc\n'
}
stage_done() { [ -f "$EPM_ROOT/$1/.done" ] && [ "${EPM_FORCE:-0}" != 1 ]; }
mark_done()  { mkdir -p "$EPM_ROOT/$1"; printf '%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" > "$EPM_ROOT/$1/.done"; }
# stage_note <stage> <text...>: append a human-readable note to the stage directory.
stage_note() { local st="$1"; shift; mkdir -p "$EPM_ROOT/$st"; printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$EPM_ROOT/$st/notes.txt"; }
