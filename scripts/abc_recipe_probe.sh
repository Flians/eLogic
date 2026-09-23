#!/usr/bin/env bash
# Which ABC recipe reproduces the ASPDAC'25 Table 1 area `A_e`?
#
# The paper states three different subject-graph recipes for different experiments:
#
#   §4.1 / Table 1   "an aggressive optimization flow running the following area-oriented commands
#                     and scripts in ABC: rw, rs, rf, resyn2rs, and compress2rs" ... "Before applying
#                     any optimization, the AIG is functionally reduced using command fraig" ...
#                     "We start with one-pass commands rw, rs, rf to slow down convergence" ...
#                     "We iterate this procedure as long as there are changes in the subject graph."
#   §4.3 / Table 2   "High-effort optimization is achieved by iteratively applying resyn2rs and
#                     compress2rs until the subject graph cannot be reduced, and merging equivalent
#                     nodes using command fraig before applying any script."
#   §4.4 / Table 2   "The subject graphs are optimized using two iterations of the script
#                     fraig; compress2rs; resyn2rs".
#
# This script runs each candidate recipe through the engine's baseline (`EPMAPPING_SKIP=1`, so only
# ABC + area-oriented `emap`) on the designs whose `A_e` the paper publishes, and scores the mean
# relative deviation from it.
#
# Usage: scripts/abc_recipe_probe.sh [outdir]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${EPM_BIN:-$ROOT/build/epmapping/epmapping_check}"
GENLIB="$ROOT/tools/mockturtle/experiments/cell_libraries/asap7.genlib"
BENCH="$ROOT/tools/mockturtle/experiments/benchmarks"
OUT="${1:-/tmp/abc_recipe_probe}"
JOBS="${EPM_JOBS:-4}"
mkdir -p "$OUT"; : > "$OUT/results.txt"

# tag|area_mapping|script, where `script` is either a one-pass ABC script or `baseline:<preset>`
# (`direct` = read_aiger + emap; `high_effort` = the `{fraig; resyn2rs; fraig; compress2rs}` loop run
# to its node-count fixpoint, which is the engine's default initial netlist).
RECIPES=$(cat <<'EOF'
HFI|1|baseline:high_effort
AEd|1|baseline:direct
AEdD|0|baseline:direct
AI|1|strash
AID|0|strash
HF1|1|fraig; resyn2rs; fraig; compress2rs
HF2|1|fraig; resyn2rs; fraig; compress2rs; fraig; resyn2rs; fraig; compress2rs
HF3|1|fraig; resyn2rs; fraig; compress2rs; fraig; resyn2rs; fraig; compress2rs; fraig; resyn2rs; fraig; compress2rs
HF4|1|fraig; resyn2rs; fraig; compress2rs; fraig; resyn2rs; fraig; compress2rs; fraig; resyn2rs; fraig; compress2rs; fraig; resyn2rs; fraig; compress2rs
HF5|1|fraig; resyn2rs; fraig; compress2rs; fraig; resyn2rs; fraig; compress2rs; fraig; resyn2rs; fraig; compress2rs; fraig; resyn2rs; fraig; compress2rs; fraig; resyn2rs; fraig; compress2rs
P1|1|fraig; rw; rs; rf; resyn2rs; compress2rs
P2|1|fraig; rw; rs; rf; resyn2rs; compress2rs; rw; rs; rf; resyn2rs; compress2rs
P2b|1|fraig; rw; rs; rf; resyn2rs; compress2rs; rw; rs; rf; resyn2rs; compress2rs; rw; rs; rf; resyn2rs; compress2rs
Q1|1|fraig; resyn2rs; compress2rs
Q1b|1|fraig; resyn2rs; compress2rs; fraig; resyn2rs; compress2rs
Q1c|1|fraig; resyn2rs; compress2rs; fraig; resyn2rs; compress2rs; fraig; resyn2rs; compress2rs
Q1d|1|fraig; resyn2rs; compress2rs; fraig; resyn2rs; compress2rs; fraig; resyn2rs; compress2rs; fraig; resyn2rs; compress2rs
S44|1|fraig; compress2rs; resyn2rs; fraig; compress2rs; resyn2rs
S44b|1|fraig; compress2rs; resyn2rs; fraig; compress2rs; resyn2rs; fraig; compress2rs; resyn2rs
F1|1|fraig
C1|1|compress2rs
R1|1|resyn2rs
EOF
)
# `EPM_RECIPES` filters by tag prefix (e.g. `EPM_RECIPES='HFI|AEd' python3`-style regex not needed:
# it is an extended regex matched against the tag), so a new preset can be scored without re-running
# the whole matrix.
if [ -n "${EPM_RECIPES:-}" ]; then
  RECIPES=$(printf '%s\n' "$RECIPES" | grep -E "^($EPM_RECIPES)\|")
fi
DESIGNS="arbiter sasc simple_spi usb_phy iwls05_i2c spi systemcdes aes_core tv80 mem_ctrl sqrt div"

one() { # one <tag> <script> <design> <area_mapping>
  local tag="$1" script="$2" d="$3" area="$4" w="$OUT/$1.$3"
  rm -rf "$w"; mkdir -p "$w"
  case "$script" in
    baseline:*) env EPMAPPING_SKIP=1 EPMAPPING_AREA_MAPPING="$area" \
                    EPMAPPING_BASELINE="${script#baseline:}" \
      perl -e 'alarm 300; exec @ARGV' "$BIN" "$GENLIB" "$BENCH/$d.aig" "$w" >"$w/log" 2>&1 ;;
    *)          env EPMAPPING_SKIP=1 EPMAPPING_AREA_MAPPING="$area" EPMAPPING_ABC_SCRIPT="$script" \
      perl -e 'alarm 300; exec @ARGV' "$BIN" "$GENLIB" "$BENCH/$d.aig" "$w" >"$w/log" 2>&1 ;;
  esac
  local rc=$? a
  a=$(sed -n 's/.*emap area=\([0-9.]*\).*/\1/p' "$w/log" | head -1)
  echo "$tag|$d|${a:-ERR}|$rc" >>"$OUT/results.txt"
}

while IFS='|' read -r tag area script; do
  [ -z "$tag" ] && continue
  for d in $DESIGNS; do
    one "$tag" "$script" "$d" "$area" &
    while [ "$(jobs -r | wc -l)" -ge "$JOBS" ]; do sleep 0.3; done
  done
done <<<"$RECIPES"
wait

python3 - "$OUT/results.txt" <<'PY'
import collections, sys
AE = dict(arbiter=557.84, sasc=31.72, simple_spi=41.58, usb_phy=23.97, iwls05_i2c=49.96,
          spi=167.59, systemcdes=142.22, aes_core=1106.60, tv80=356.38, mem_ctrl=2063.01,
          sqrt=1171.15, div=1296.90)
AI = dict(arbiter=557.84, sasc=40.46, simple_spi=54.77, usb_phy=28.82, iwls05_i2c=66.32,
          spi=205.45, systemcdes=172.17, aes_core=1198.55, tv80=512.39, mem_ctrl=2547.32,
          sqrt=1372.25, div=3914.60)
res = collections.defaultdict(dict); failed = []
for line in open(sys.argv[1]):
    t, d, a, rc = line.strip().split('|')
    (res[t].__setitem__(d, float(a)) if a != 'ERR' else failed.append((t, d, rc)))

def table(title, ref, tags):
    print("\n" + title)
    print(f"{'recipe':<6}{'mean|dA|':>9}{'max|dA|':>9}   " +
          "".join(f"{d:>13}" for d in ['arbiter', 'mem_ctrl', 'sqrt', 'div']))
    rows = []
    for t in tags:
        m = res.get(t, {})
        devs = [(m[d] - ref[d]) / ref[d] for d in m if d in ref]
        if devs:
            rows.append((sum(abs(v) for v in devs) / len(devs), max(abs(v) for v in devs), t, m))
    for mean, mx, t, m in sorted(rows):
        ex = "".join(f"{m[d]:>8.1f}/{ref[d]:<5.0f}" if d in m else f"{'-':>13}"
                     for d in ['arbiter', 'mem_ctrl', 'sqrt', 'div'])
        print(f"{t:<6}{100*mean:>8.2f}%{100*mx:>8.2f}%   {ex}")

table("A_e (optimized subject graph) vs Table 1 A_e",
      AE, ['HFI','P1','P2','P2b','Q1','Q1b','Q1c','Q1d','S44','S44b','HF1','HF2','HF3','HF4','HF5','F1','C1','R1'])
table("A_i (unoptimized AIG) vs Table 1 A_i", AI, ['AEd','AEdD','AI','AID'])
print("\nlower is better; the last four columns are ours/paper [um^2]")
print("HFI = EPMAPPING_BASELINE=high_effort (default initial netlist); AEd = EPMAPPING_BASELINE=direct;")
print("AI = read_aiger + area-oriented emap via `strash`; AID = the same with delay-oriented emap")
if failed:
    print("failed runs:", failed)
PY
