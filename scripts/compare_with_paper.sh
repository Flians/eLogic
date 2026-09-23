#!/usr/bin/env bash
# Fair comparison with the IWLS'24 post-mapping resubstitution paper:
#   baseline = area-oriented emap on the paper's aggressively optimized subject graph, then
#   r1       = one epmapping pass,
#   rN       = iterate until convergence (max 6 passes),
# and report dA1 = (r1-base)/base, dAinf = (rN-base)/base (same for delay).
#
# The subject graph comes from the engine's `EPMAPPING_BASELINE` preset: the default `high_effort`
# (ASPDAC'25 `A_e`: `{fraig; resyn2rs; fraig; compress2rs}` iterated to a node-count fixpoint) or
# `direct` (the paper's unoptimized `A_i`); setting `EPMAPPING_ABC_SCRIPT` selects a single-pass
# `custom` graph instead (e.g. the IWLS'24 `fraig; compress2rs; resyn2rs; compress2rs`).
#
# Usage: scripts/compare_with_paper.sh [timeout_seconds] [design ...]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${EPM_OUT:-/tmp/epmap_paper}"
TMO="${1:-300}"
mkdir -p "$OUT"
shift || true
DESIGNS=("$@")
if [ ${#DESIGNS[@]} -eq 0 ]; then
  # EPFL combinational suite (hyp excluded: the paper excludes it).
  DESIGNS=(adder bar div log2 max multiplier sin sqrt square arbiter
           cavlc ctrl dec i2c int2float mem_ctrl priority router voter)
fi
JOBS="${EPM_JOBS:-5}"

printf '%s\n' "${DESIGNS[@]}" \
  | EPM_OUT="$OUT" TMO="$TMO" xargs -P "$JOBS" -I{} \
      bash -c 'EPM_OUT="$EPM_OUT" "$0" "$1" "$EPM_OUT" "$TMO"' \
      "$ROOT/scripts/_epm_one.sh" {} >"$OUT/results.csv"

python3 - "$OUT/results.csv" <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
hdr = ["design","base_a","base_d","r1_a","r1_d","rN_a","rN_d","rnd","cec_b","cec_1","cec_n","base_s","r1_s","rN_s","status"]
rows = [r for r in rows if len(r) == len(hdr)]
def f(x):
    try: return float(x)
    except: return None
def pct(a,b):
    return "n/a" if (a in (None,0) or b is None) else f"{100.0*(b-a)/a:+.2f}%"
print(f"{'design':<11}{'base_a':>9}{'r1_a':>9}{'rN_a':>9}{'dA1':>8}{'dAinf':>8}"
      f"{'base_d':>10}{'rN_d':>10}{'dDinf':>8}{'rnd':>4}{'cec':>7}{'secs':>7}  status")
tot1=[]; totN=[]
for r in rows:
    d, ba, bd, ra, rd, na, nd = r[0], f(r[1]), f(r[2]), f(r[3]), f(r[4]), f(r[5]), f(r[6])
    d1 = None if (ba in (None,0) or ra is None) else 100.0*(ra-ba)/ba
    dN = None if (ba in (None,0) or na is None) else 100.0*(na-ba)/ba
    if d1 is not None: tot1.append(d1)
    if dN is not None: totN.append(dN)
    cecs = f"{r[8] or '-'}/{r[9] or '-'}/{r[10] or '-'}"
    print(f"{d:<11}{(ba if ba is not None else 0):>9.2f}{(ra if ra is not None else 0):>9.2f}"
          f"{(na if na is not None else 0):>9.2f}{('' if d1 is None else f'{d1:+.2f}%'):>8}"
          f"{('' if dN is None else f'{dN:+.2f}%'):>8}{(bd if bd is not None else 0):>10.2f}"
          f"{(nd if nd is not None else 0):>10.2f}{pct(bd, nd):>8}{r[7]:>4}{cecs:>7}{r[13]:>7}  {r[14]}")
if tot1:
    print(f"\naverage over {len(tot1)} designs:  dA1 = {sum(tot1)/len(tot1):+.2f}%   "
          f"dAinf = {sum(totN)/len(totN):+.2f}%   (paper: dA1e = -1.06%, dAinf_e = -2.50%)")
PY
