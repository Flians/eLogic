#!/usr/bin/env bash
# Side-by-side comparison with the ASPDAC'25 baseline ("Area-Oriented Optimization After
# Standard-Cell Mapping", Costamagna et al., 10.1145/3658617.3697722, Table 1).
#
# The paper generates its initial netlist with a high-effort area-oriented AIG optimization (once
# `fraig` before each script, iterated until the subject graph cannot be reduced further) and
# area-oriented `emap` (Mockturtle) for the mapping, then reports the area/delay of its post-mapping
# resubstitution after one pass (t_e1) and to convergence (t_e∞). The engine's default initial netlist
# is exactly that (`EPMAPPING_BASELINE=high_effort`; `direct` selects the paper's unoptimized A_i
# graph), so a plain run is the matched baseline:
#
#   EPM_OUT=/tmp/epmap_asp EPM_JOBS=2 RAYON_NUM_THREADS=4 EPMAPPING_DSE=0 \
#     EPMAPPING_NO_DELAY_REGRESSION=1 \
#     ./scripts/compare_with_paper.sh 1200 <designs>
#   scripts/compare_with_aspdac25.sh /tmp/epmap_asp/results.csv
#
# One-pass readings (the older tables) use `EPMAPPING_ABC_SCRIPT="fraig; rw; rs; rf; resyn2rs;
# compress2rs"` (§4.1) or `"fraig; compress2rs; resyn2rs; compress2rs"` (IWLS'24).
#
# Usage: compare_with_aspdac25.sh <results.csv>
set -uo pipefail
CSV="${1:?usage: compare_with_aspdac25.sh <results.csv>}"

python3 - "$CSV" <<'PY'
import csv, sys

# Paper Table 1: A_e [um^2], dA_e1 [%], dA_einf [%], D_e [ps], dD_e1 [%], dD_einf [%], t_e1 [s], t_einf [s]
PAPER = {
 "div":        (1296.90, -7.81,  -9.30, 60248.23,  0.00,   0.01,   7.09,  38.40),
 "sqrt":       (1171.15, -3.16,  -5.22, 78957.63,  0.00,  -1.21,   4.09,  50.83),
 "arbiter":    ( 557.84,-12.01, -51.81,   999.95,  0.00, -39.57,   1.59,  17.38),
 "mem_ctrl":   (2063.01, -5.86, -12.79,  1649.46, -5.82, -13.00,  18.41, 183.16),
 "aes_core":   (1106.60, -1.07,  -1.81,   434.52, -2.29,  -2.05,  11.41, 165.45),
 "ethernet":   (3123.35, -0.96,  -3.68,   588.34,  0.00,   0.00,  54.60, 214.07),
 "iwls05_i2c": (  49.96, -1.46,  -2.00,   288.07,  0.00,   0.00,   0.19,   0.58),
 "RISC":       (3172.54, -1.03,  -1.03,  1304.74,  0.00,   0.00, 350.90, 350.90),
 "sasc":       (  31.72, -1.07,  -1.32,   191.00,  0.00,   0.00,   0.16,   0.48),
 "simple_spi": (  41.58, -0.53,  -1.39,   287.00,  0.00,   0.00,   0.17,   1.06),
 "spi":        ( 167.59, -1.52,  -2.08,   489.49,  0.00,   0.00,   0.43,   3.66),
 "systemcaes": ( 530.89, -3.25,  -3.62,   784.00,  0.00,   0.00,   1.23,  11.28),
 "systemcdes": ( 142.22, -1.21,  -2.38,   530.29,  0.00,   0.00,   0.39,   2.22),
 "tv80":       ( 356.38, -0.70,  -1.48,   907.86,  0.00,   0.00,   1.03,   6.84),
 "usb_funct":  ( 702.45, -0.70,  -0.98,   722.00, -2.65,  -2.65,   1.64,  12.45),
 "usb_phy":    (  23.97, -0.83,  -1.42,   179.06, -7.79,  -8.51,   0.15,   0.61),
}
# Paper Table 1 averages over all 39 benchmarks it used.
PAPER_AVG = dict(dA1=-1.21, dAinf=-2.81, dD1=-0.48, dDinf=-1.81, t1=17.62, tinf=47.15)

def f(x):
    try: return float(x)
    except: return None

rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) == 15]
sel = [r for r in rows if r[0] in PAPER]
hdr = (f"{'design':<11}{'A_e pap':>9}{'A_e ours':>9}{'dA1 pap':>9}{'dA1 our':>9}"
       f"{'dAinf pap':>11}{'dAinf our':>11}{'D_e pap':>10}{'D_e our':>10}"
       f"{'dDinf pap':>10}{'dDinf our':>11}{'t1 pap':>8}{'t1 our':>8}{'tinf pap':>9}{'tinf our':>9}")
print(hdr); print("-" * len(hdr))
acc = dict(dA1=[], dAinf=[], dDinf=[], t1=[], tinf=[], aep=[], aeo=[], dep=[], deo=[], t1p=[], tinfp=[])
for r in sorted(sel, key=lambda r: r[0].lower()):
    n = r[0]
    pae, pa1, painf, pde, pd1, pdinf, pt1, ptinf = PAPER[n]
    ba, bd, ra, rd, na, nd = f(r[1]), f(r[2]), f(r[3]), f(r[4]), f(r[5]), f(r[6])
    sa1 = None if not ba or ra is None else 100 * (ra - ba) / ba
    sainf = None if not ba or na is None else 100 * (na - ba) / ba
    sD1 = None if not bd or rd is None else 100 * (rd - bd) / bd
    sDinf = None if not bd or nd is None else 100 * (nd - bd) / bd
    st1 = f(r[12]); stinf = f(r[13])
    fm = lambda v, w=9, p=2: "n/a".rjust(w) if v is None else f"{v:+.{p}f}%".rjust(w)
    fs = lambda v, w=9: "n/a".rjust(w) if v is None else f"{v:.2f}".rjust(w)
    print(f"{n:<11}{pae:>9.2f}{fs(ba,9)}{pa1:>9.2f}{('n/a' if sa1 is None else f'{sa1:+.2f}'):>9}"
          f"{painf:>11.2f}{('n/a' if sainf is None else f'{sainf:+.2f}'):>11}"
          f"{pde:>10.2f}{fs(bd,10)}{pdinf:>10.2f}{('n/a' if sDinf is None else f'{sDinf:+.2f}'):>11}"
          f"{pt1:>8.2f}{fs(st1,8)}{ptinf:>9.2f}{fs(stinf,9)}")
    if sainf is not None:
        acc['dA1'].append(sa1); acc['dAinf'].append(sainf); acc['aep'].append(pae); acc['aeo'].append(ba)
        if sDinf is not None: acc['dDinf'].append(sDinf)
        if st1 is not None: acc['t1'].append(st1)
        if stinf is not None: acc['tinf'].append(stinf)
        acc['t1p'].append(pt1); acc['tinfp'].append(ptinf)

m = lambda v: sum(v) / len(v) if v else float('nan')
print("-" * len(hdr))
print(f"{'our avg('+str(len(acc['dAinf']))+')':<11}{m(acc['aep']):>9.2f}{m(acc['aeo']):>9.2f}"
      f"{m([PAPER[r[0]][1] for r in sel]):>9.2f}{m(acc['dA1']):>9.2f}"
      f"{m([PAPER[r[0]][2] for r in sel]):>11.2f}{m(acc['dAinf']):>11.2f}"
      f"{'':>10}{'':>10}{m([PAPER[r[0]][5] for r in sel]):>10.2f}{m(acc['dDinf']):>11.2f}"
      f"{m(acc['t1p']):>8.2f}{m(acc['t1']):>8.2f}{m(acc['tinfp']):>9.2f}{m(acc['tinf']):>9.2f}")
print(f"{'paper avg(39)':<11}{'':>9}{'':>9}{PAPER_AVG['dA1']:>9.2f}{'':>9}{PAPER_AVG['dAinf']:>11.2f}{'':>11}"
      f"{'':>10}{'':>10}{PAPER_AVG['dDinf']:>10.2f}{'':>11}{PAPER_AVG['t1']:>8.2f}{'':>8}{PAPER_AVG['tinf']:>9.2f}")
print("\nA_e/dA/D_e are the paper's; 'our' columns are epmapping's baseline and post-mapping results on")
print("the paper's initial netlist. dA1/dAinf = one pass / to convergence (rounds=6).")
print("'our avg' averages the paper columns over exactly the designs listed above; 'paper avg(39)' is")
print("Table 1's own average over all 39 benchmarks it used.")
PY
