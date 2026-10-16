#!/usr/bin/env bash
# Validate the experiment scripts without running the engine.
#
#   scripts/experiments/selftest.sh            # syntax + spec + report parsing checks
#
# Checks
#   1. every shell script parses (`bash -n`), every python file parses (ast)
#   2. a full dry run generates a well-formed spec for every stage: the working
#      directory is under EPM_ROOT, the design exists as a benchmark, the round
#      count is numeric, the recipe is one the driver knows, and extra arguments
#      are KEY=VALUE assignments
#   3. no (stage, config, design) spec is generated twice
#   4. report.py parses synthetic logs (counters -> fig:share data, sweeps -> tables)
#      and emits every expected section, so a broken log format fails here instead
#      of silently producing an empty paper table
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SELF_DIR/../.." && pwd)"
COMMON="$SELF_DIR/common.sh"
fails=0
ok()   { printf '  ok    %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; fails=$(( fails + 1 )); }

echo "== 1. syntax =="
while IFS= read -r f; do
  bash -n "$f" && ok "bash -n ${f#$PROJECT_ROOT/}" || fail "bash -n ${f#$PROJECT_ROOT/}"
done < <(find "$PROJECT_ROOT/scripts" -name '*.sh' | sort)
for f in "$SELF_DIR/report.py" "$SELF_DIR/restrict_genlib.py"; do
  python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f" \
    && ok "ast ${f#$PROJECT_ROOT/}" || fail "ast ${f#$PROJECT_ROOT/}"
done

echo "== 2. dry run: specs =="
DRY_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/epm-selftest.XXXXXX")"
DRY="$DRY_ROOT/dry.txt"
EPM_ROOT="$DRY_ROOT/results" EPM_DRY_RUN=1 "$SELF_DIR/run_all.sh" --no-build > "$DRY" 2>&1
rc=$?
[ $rc = 0 ] && ok "run_all.sh --dry-run exits 0" || fail "run_all.sh --dry-run exit $rc"

BENCH="$PROJECT_ROOT/tools/mockturtle/experiments/benchmarks"
stages_seen=$(grep -c '^\[dry-run\] stage\|^\[dry-run\] sweep' "$DRY")
[ "$stages_seen" -ge 15 ] && ok "$stages_seen stages generated specs" || fail "only $stages_seen stages"

bad=0; n=0
while read -r tag wd design rounds recipe extra; do
  [ "$tag" = DRY ] || continue
  n=$(( n + 1 ))
  case "$wd" in "$DRY_ROOT/results"/*) ;; *) fail "wd outside EPM_ROOT: $wd"; bad=$((bad+1)) ;; esac
  [ -f "$BENCH/$design.aig" ] || { fail "no benchmark for '$design'"; bad=$((bad+1)); }
  case "$rounds" in ''|*[!0-9]*) fail "round count '$rounds' for $design"; bad=$((bad+1)) ;; esac
  case "$recipe" in
    high_effort|direct|iwls|single_pass|custom|ae|ai|A_e|A_i) ;;
    *) fail "unknown recipe '$recipe' for $design"; bad=$((bad+1)) ;;
  esac
  for tok in ${extra:-}; do
    case "$tok" in *=*) ;; *) fail "extra arg '$tok' is not KEY=VALUE ($design)"; bad=$((bad+1)) ;; esac
  done
done < "$DRY"
[ "$bad" = 0 ] && ok "$n specs well-formed (wd, design, rounds, recipe, env)" || fail "$bad malformed specs"

dups=$(grep '^DRY' "$DRY" | awk -F'\t' '{print $2}' | sort | uniq -d | wc -l | tr -d ' ')
[ "$dups" = 0 ] && ok "no duplicate (config, design) work directories" || fail "$dups duplicated work directories"
care_specs=$(grep '^DRY' "$DRY" | grep '/noCARE\.')
if [ -n "$care_specs" ] && ! printf '%s\n' "$care_specs" | grep -v 'EPMAPPING_DC=0' >/dev/null; then
  ok "every noCARE arm disables all cut care masks"
else
  fail "noCARE must set EPMAPPING_DC=0 (SAT refinement alone is not the care mask)"
fi

if grep '^DRY' "$DRY" | grep 'EPMAPPING_WINDOW_SYN=1' >/dev/null; then
  fail "DATE stage enables whole-window synthesis"
else
  ok "every DATE stage keeps whole-window synthesis disabled"
fi
if grep '^DRY' "$DRY" | grep '/noWINDOW\.' >/dev/null; then
  fail "redundant noWINDOW arm must not appear in cut-only ablations"
else
  ok "cut-only ablations contain no redundant noWINDOW arm"
fi

echo "== 3. report.py on synthetic logs =="
R="$DRY_ROOT/report"; mkdir -p "$R/funnel/r1.sasc" "$R/funnel/r1.c7552" "$R/main_ae" \
  "$R/ablation/full.sasc" "$R/ablation/full.c7552" "$R/db/ae.asap7_4in_3lvl_a050.stats.sasc"
printf 'stage\tdesign\trecipe\trounds\tbase_a\tbase_d\tout_a\tout_d\tdA%%\tdD%%\trnd\tcec\tsecs\trc\n' > "$R/db/summary.tsv"
printf 'ae.asap7_4in_3lvl_a050.stats\tsasc\thigh_effort\t1\t27.28\t191.2\t-\t-\tn/a\tn/a\t-\t-\t1\t0\n' >> "$R/db/summary.tsv"
cat > "$R/funnel/r1.sasc/log" <<'LOG'
[i] map_egg paths library_hits=3 egg_calls=100 egg_cache_hits=84
[i] epmapping replacements=42 (egg=7, single=30, multi=2, synth=2, window=1)
[i] epmapping staged library_cuts=595 mffc_cut_egg_roots=82 egg_covers=82
[i] epmapping staged_secs cut_enum=0.5 care=4.0 stage1=2.0 stage2=1.0 egg=3.0 mffc=0.1 synth=6.0 window=0.2
[i] epmapping sat_dc cuts=9 calls=100 cleared=5 budget_hits=0 secs=1.5 dc_time_budget_stops=0
[i] emap area=27.28 delay=191.2 gates=315 area_oriented=1 multioutput=0
[i] epmapping area 27.28 -> 27.18 delay 191.2 -> 191.2
[i] epmapping rounds=1
[i] cec=1
LOG
cat > "$R/funnel/r1.c7552/log" <<'LOG'
[i] map_egg paths library_hits=1 egg_calls=200 egg_cache_hits=150
[i] epmapping replacements=84 (egg=52, single=5, multi=18, synth=9, window=0)
[i] epmapping staged library_cuts=595 mffc_cut_egg_roots=443 egg_covers=443
[i] epmapping staged_secs cut_enum=1.0 care=2.0 stage1=3.0 stage2=1.0 egg=4.0 mffc=0.1 synth=5.0 window=0.3
[i] emap area=81.87 delay=657.42 gates=900 area_oriented=1 multioutput=0
[i] epmapping area 81.87 -> 81.74 delay 657.42 -> 627.15
[i] epmapping rounds=1
[i] cec=1
LOG
for d in sasc c7552; do cp "$R/funnel/r1.$d/log" "$R/ablation/full.$d/log"; done
cat > "$R/db/ae.asap7_4in_3lvl_a050.stats.sasc/log" <<'LOG'
[i] db4 coverage=29491/65536 with_multi_cell=1234 avg_best_area=0.123 max_best_area=0.900
LOG
printf 'stage\tdesign\trecipe\trounds\tbase_a\tbase_d\tout_a\tout_d\tdA%%\tdD%%\trnd\tcec\tsecs\trc\n' > "$R/ablation/summary.tsv"
printf 'full\tsasc\thigh_effort\t6\t27.28\t191.2\t27.18\t191.2\t-0.37\t+0.00\t6\t1\t40\t0\n' >> "$R/ablation/summary.tsv"
printf 'noSYN\tsasc\thigh_effort\t6\t27.28\t191.2\t27.60\t191.2\t+1.17\t+0.00\t6\t1\t38\t0\n' >> "$R/ablation/summary.tsv"
printf 'full\tc7552\thigh_effort\t6\t81.87\t657.42\t81.74\t627.15\t-0.16\t-4.60\t6\t1\t40\t0\n' >> "$R/ablation/summary.tsv"
cat > "$R/main_ae/results.csv" <<'CSV'
sasc,27.28,191.2,27.18,191.2,27.18,191.2,4,1,1,1,41,41,43,OK
CSV
if python3 "$SELF_DIR/report.py" --root "$R" --markdown "$R/tables.md" --latex "$R/tables.tex" >/dev/null 2>&1; then
  ok "report.py runs on synthetic data"
else
  fail "report.py crashed on synthetic data"
fi
for want in "fig:share data" "L1 wins" "share of oracle time" "signed area change when a stage is removed" "extended tables" "database coverage" "memo cache" ; do
  grep -q "$want" "$R/tables.md" && ok "report contains: $want" || fail "report misses: $want"
done
grep -q "27.18" "$R/tables.md" && ok "sweep row parsed from results.csv" || fail "sweep row missing"
grep -q "84.0" "$R/tables.md" && ok "memo-cache hit rate parsed" || fail "cache share missing"

rm -rf "$DRY_ROOT"
echo
if [ "$fails" = 0 ]; then echo "[i] selftest passed"; else echo "[e] selftest: $fails failure(s)"; fi
exit $(( fails > 0 ))
