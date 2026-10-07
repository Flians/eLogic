#!/usr/bin/env python3
"""Aggregate the stage data produced by run_all.sh into the paper's tables.

    scripts/experiments/report.py --root results --markdown results/tables.md --latex results/tables.tex

Sections:
  * tab:main vs ASPDAC'25        (its published Table 1 vs ours, A_e recipe)
  * tab:main internal            (A_e / single-pass / A_i over the 27-design set)
  * ISCAS subset                 (both recipes)
  * per-stage summaries          (oracle, db, funnel, levers, runtime, ablation)

Nothing is recomputed here: every number is read from the stage logs, so the report can be rebuilt at
any time without re-running the experiments.
"""
import argparse
import csv
import os
import re
import sys

DES27 = ("adder bar max sin log2 div cavlc multiplier sqrt ctrl dec arbiter int2float i2c priority router "
         "sasc iwls05_i2c simple_spi voter usb_phy c6288 systemcaes square systemcdes spi mem_ctrl").split()
ISCAS = "c432 c499 c880 c1355 c1908 c2670 c3540 c5315 c6288 c7552".split()
# the paper's 16 circuits (ASPDAC'25 Table 1) -- tab:main and tab:mfs3 are both on this set
PAPER16 = ("aes_core arbiter div ethernet iwls05_i2c mem_ctrl RISC sasc simple_spi spi sqrt systemcaes "
           "systemcdes tv80 usb_funct usb_phy").split()
ASP16 = PAPER16  # backwards-compatible alias


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean(v):
    return sum(v) / len(v) if v else float('nan')


def verified(row):
    return row.get('rc') == '0' and row.get('cec') == '1'


def esc(s):
    return s.replace('_', r'\_')


# --- input readers -----------------------------------------------------------
def read_results(path):
    """compare_with_paper.sh results.csv -> {design: row}"""
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.reader(open(path)):
        if len(r) >= 15 and r[0]:
            out[r[0]] = r
    return out


def read_summary(path):
    """stage summary.tsv -> list of dicts"""
    rows = []
    if not os.path.exists(path):
        return rows
    hdr = None
    for line in open(path):
        c = line.rstrip('\n').split('\t')
        if hdr is None:
            hdr = c
            continue
        if len(c) == len(hdr):
            rows.append(dict(zip(hdr, c)))
    return rows


def read_paper_table(script):
    """ASPDAC'25 Table 1 as transcribed in scripts/compare_with_aspdac25.sh."""
    if not os.path.exists(script):
        return {}
    body = re.search(r"^PAPER = \{(.*?)^\}", open(script).read(), re.S | re.M)
    if not body:
        return {}
    ns = {}
    exec("PAPER = {" + body.group(1) + "}", ns)
    return ns["PAPER"]


# --- table emitters ----------------------------------------------------------
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def tex_table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    out = [r"\begin{tabular}{" + align + "}", r"  \toprule",
           "  " + " & ".join(headers) + r" \\", r"  \midrule"]
    out += ["  " + " & ".join(str(c) for c in r) + r" \\" for r in rows]
    out += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def sweep_table(rows_by_design, order, latex=False):
    """The README/paper delta table for one subject graph."""
    body, t1, tN, tD = [], [], [], []
    for d in order:
        r = rows_by_design.get(d)
        if not r:
            continue
        ba, bd, ra, rd, na, nd = (num(r[1]), num(r[2]), num(r[3]), num(r[4]), num(r[5]), num(r[6]))
        d1 = 100 * (ra - ba) / ba if (ra is not None and ba) else None
        dN = 100 * (na - ba) / ba if (na is not None and ba) else None
        dD = 100 * (nd - bd) / bd if (nd is not None and bd) else None
        for v, acc in ((d1, t1), (dN, tN), (dD, tD)):
            if v is not None:
                acc.append(v)
        p = lambda v: "---" if v is None else f"{v:+.2f}"
        c = lambda v: "---" if v is None else f"{v:.2f}"
        body.append([esc(d) if latex else d, c(ba), c(ra), p(d1), c(na), p(dN), p(dD),
                     r[7] or "---", f"{r[8] or '-'}/{r[9] or '-'}/{r[10] or '-'}",
                     r[12] or "---", r[13] or "---", r[14]])
    hdr = ["design", "base (um2)", "r1", "dA1", "rN", "dAinf", "dDinf", "rounds", "CEC b/1/N",
           "t1 (s)", "tinf (s)", "status"]
    avgs = (f"**Averages**: dA1 = {mean(t1):+.2f}% ({len(t1)} designs), "
            f"dAinf = {mean(tN):+.2f}% ({len(tN)}), dDinf = {mean(tD):+.2f}% ({len(tD)})")
    return (tex_table(hdr, body) if latex else md_table(hdr, body)) + "\n\n" + avgs



# --- staged counters (fig:share and the runtime claims) ----------------------
_STAGE_KEYS = ("cut_enum", "care", "stage1", "stage2", "egg", "mffc", "synth", "window")


def read_counters(log):
    """Parse one engine log into the counters the paper's figures quote."""
    out = {}
    if not os.path.exists(log):
        return out
    txt = open(log, errors='replace').read()

    def last(pat):
        m = re.findall(pat, txt)
        return float(m[-1]) if m else None

    m = re.findall(r"replacements=\d+ \(egg=(\d+), single=(\d+), multi=(\d+), synth=(\d+), window=(\d+)\)", txt)
    if m:
        e, si, mu, sy, wi = (int(x) for x in m[-1])
        out.update(egg_wins=e, single_wins=si, multi_wins=mu, synth_wins=sy, window_wins=wi)
    m = re.findall(r"mffc_cut_egg_roots=(\d+)", txt)
    if m:
        out['egg_admitted'] = int(m[-1])
    m = re.findall(r"egg_calls=(\d+) egg_cache_hits=(\d+)", txt)
    if m:
        calls, hits = (int(x) for x in m[-1])
        out['egg_calls'], out['egg_cache_hits'] = calls, hits
        out['cache_share'] = 100.0 * hits / calls if calls else 0.0
    m = re.findall(r"staged_secs ([^\n]*)", txt)
    if m:
        for k, v in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([0-9.eE+-]+)", m[-1]):
            if k in _STAGE_KEYS:
                out['secs_' + k] = float(v)
    # `synth` and `window` are not part of the `staged_secs` line; they print their own.
    m = re.findall(r"synth roots=\d+ covers=\d+ wins=\d+ secs=([0-9.eE+-]+)", txt)
    if m:
        out['secs_synth'] = float(m[-1])
    m = re.findall(r"window_synth windows=\d+ applied=\d+ gain=[0-9.eE+-]+ actual=[0-9.eE+-]+ "
                   r"gates=\d+ abandoned=\d+ delay_rejected=\d+ secs=([0-9.eE+-]+)", txt)
    if m:
        out['secs_window'] = float(m[-1])
    m = re.findall(r"sat_dc cuts=(\d+) calls=(\d+) cleared=(\d+) budget_hits=(\d+) secs=([0-9.]+)", txt)
    if m:
        c, ca, cl, bh, sec = m[-1]
        out.update(sat_cuts=int(c), sat_calls=int(ca), sat_cleared=int(cl), sat_budget_hits=int(bh), sat_secs=float(sec))
    m = re.findall(r"db4 coverage=(\d+)/65536 with_multi_cell=(\d+) avg_best_area=([0-9.]+) max_best_area=([0-9.]+)", txt)
    if m:
        cov, wm, avg, mx = m[-1]
        out.update(db_coverage=int(cov), db_multi=int(wm), db_avg=float(avg), db_max=float(mx))
    for line in re.findall(r"epmapping timing ([^\n]+)", txt):
        for key, value in re.findall(r"([a-z_]+)=([0-9.eE+-]+)", line):
            out['time_' + key] = float(value)
    m = re.findall(r"synth_cache lookups=(\d+) hits=(\d+)", txt)
    if m:
        out['synth_cache_lookups'], out['synth_cache_hits'] = map(int, m[-1])
    return out


def stage_log(root, stage, cfg, design):
    return os.path.join(root, stage, f"{cfg}.{design}", "log")


def aggregate_counters(root, stage, cfg, designs, keys):
    tot = {k: 0.0 for k in keys}
    per = {}
    for d in designs:
        c = read_counters(stage_log(root, stage, cfg, d))
        per[d] = c
        for k in keys:
            tot[k] += c.get(k, 0.0)
    return per, tot


def figure_data(root, md):
    """fig:share: funnel counts, oracle-time shares and removed-stage area shares."""
    md.append("## fig:share data\n")
    # (a) funnel, one pass
    funnel = "c7552 sin div multiplier log2".split()
    rows_a, plots = [], [[], [], []]
    have_a = False
    for d in funnel:
        c = read_counters(stage_log(root, 'funnel', 'r1', d))
        if not c:
            continue
        have_a = True
        l1 = int(c.get('single_wins', 0))
        adm = int(c.get('egg_admitted', 0))
        l4 = int(c.get('egg_wins', 0))
        rows_a.append([d, l1, adm, l4])
        short = {'multiplier': 'mult'}.get(d, d)
        plots[0].append((short, l1))
        plots[1].append((short, adm))
        plots[2].append((short, l4))
    if have_a:
        md.append("### (a) layer funnel, one pass (L1 wins / L4 admitted / L4 wins)\n")
        md.append(md_table(["design", "L1 wins", "L4 admitted", "L4 wins"], rows_a))
        md.append("")
        for name, pl in zip(("L1 wins", "L4 admitted", "L4 wins"), plots):
            md.append(f"    % {name}")
            md.append("    \\addplot coordinates {" + " ".join(f"({d},{v})" for d, v in pl) + "}")

    # memo cache (the "what pays is not searching twice" claim)
    cache = []
    cache_designs = list(dict.fromkeys(funnel + [r['design'] for r in read_summary(os.path.join(root, 'ablation', 'summary.tsv'))]))
    for d in cache_designs:
        c = read_counters(stage_log(root, 'funnel', 'r1', d)) or read_counters(stage_log(root, 'ablation', 'full', d))
        if c.get('egg_calls'):
            cache.append([d, int(c['egg_calls']), int(c.get('egg_cache_hits', 0)), f"{c.get('cache_share', 0):.1f}"])
    if cache:
        md.append("\n### e-graph memo cache (one pass)\n")
        md.append(md_table(["design", "egg calls", "cache hits", "hit rate (%)"], cache))
        md.append("")

    # database coverage (\u00a7Lever 1: "three quarters of all four-input functions have no entry")
    cov = []
    for d in sorted({r['design'] for r in read_summary(os.path.join(root, 'db', 'summary.tsv'))}):
        for r in read_summary(os.path.join(root, 'db', 'summary.tsv')):
            if r['design'] != d or 'stats' not in r['stage']:
                continue
            c = read_counters(stage_log(root, 'db', r['stage'], d))
            if 'db_coverage' in c:
                cov.append([d, r['stage'], f"{100.0*c['db_coverage']/65536.0:.1f}", c['db_multi'], f"{c['db_avg']:.3f}"])
    if cov:
        md.append("\n### database coverage (`EPMAPPING_DB_STATS=1`)\n")
        md.append(md_table(["design", "database", "coverage (%)", "best area > 0.2 (proxy)", "avg best area"], cov))
        md.append("")

    # (b) oracle time shares over the ablation set, six rounds
    # (c) signed marginal area effects of removing each stage
    # Both are restricted to the designs carrying a number in *every* ablation arm: summing each arm
    # over whatever happened to finish compares different design sets as soon as one run times out
    # (defect recorded in results/paper_data_check.md).
    abl = read_summary(os.path.join(root, 'ablation', 'summary.tsv'))
    designs = sorted({r['design'] for r in abl}) or \
        "aes_core arbiter div iwls05_i2c mem_ctrl sasc simple_spi spi systemcdes tv80 usb_phy".split()
    arms = ('full', 'noSYN', 'noWINDOW', 'noEGG', 'noCARE', 'noDB')
    areas = {c: {d: num(read_summary_area(root, 'ablation', c, d)) for d in designs} for c in arms}
    have = [c for c in arms if any(v is not None for v in areas[c].values())]
    common = [d for d in designs if all(areas[c][d] is not None for c in have)] if have else []
    keys = [f"secs_{k}" for k in _STAGE_KEYS]
    per, tot = aggregate_counters(root, 'ablation', 'full', common, keys)
    if any(tot.values()):
        labels = {'secs_stage1': 'L1 single cell', 'secs_stage2': 'L2 database', 'secs_synth': 'L3 synthesis',
                  'secs_egg': 'L4 e-graph', 'secs_care': 'care mask'}
        ssum = sum(tot.get(k, 0.0) for k in labels)
        rows_b = [[labels[k], f"{tot[k]:.1f}", f"{100*tot[k]/ssum:.1f}"] for k in labels if k in tot]
        md.append(f"\n### (b) share of oracle time ({len(common)} designs complete in every arm, six rounds)\n")
        md.append(md_table(["stage", "seconds", "share (%)"], rows_b))
        md.append(f"\nTotal {ssum:.1f} s; cut enumeration, MFFC accounting and whole-window synthesis "
                  f"are outside this denominator.\n")

    stage_labels = {'noSYN': 'L3 synthesis', 'noEGG': 'L4 e-graph', 'noDB': 'L2 database',
                    'noCARE': 'care mask', 'noWINDOW': 'window'}
    delta_arms = [c for c in ('noSYN', 'noEGG', 'noDB', 'noCARE', 'noWINDOW') if c in have]
    if common and 'full' in have and delta_arms:
        base = sum(areas['full'][d] for d in common)
        deltas = {stage_labels[c]: sum(areas[c][d] for d in common) - base for c in delta_arms}
        rows_c = [[k, f"{v:+.2f}"] for k, v in deltas.items()]
        md.append(f"\n### (c) signed area change when a stage is removed "
                  f"({len(common)} designs complete in every arm)\n")
        md.append(md_table(["stage", "area delta"], rows_c))
        md.append("These greedy-path marginal effects are not additive contributions.\n")
    md.append("")


def read_summary_area(root, stage, cfg, design):
    for r in read_summary(os.path.join(root, stage, 'summary.tsv')):
        if r['stage'] == cfg and r['design'] == design:
            return r.get('out_a') if verified(r) else None
    return None


# --- report ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results')
    ap.add_argument('--markdown', default=None)
    ap.add_argument('--latex', default=None)
    ap.add_argument('--script', default=None, help="compare_with_aspdac25.sh (for its Table 1)")
    args = ap.parse_args()

    root = args.root
    here = os.path.dirname(os.path.abspath(__file__))
    script = args.script or os.path.join(here, '..', 'compare_with_aspdac25.sh')
    paper = read_paper_table(script)

    md, tex = [], []

    md.append("# ePMapping experiment report\n")
    md.append(f"Generated from `{root}` by `scripts/experiments/report.py`.\n")

    # 1) the internal 27-design sweeps
    md.append("## extended tables (27 designs, README only -- the paper uses the 16 below)\n")
    for stage, label in (("ext27_ae", "A_e (iterated high-effort)"),
                         ("ext27_iwls", "single-pass graph"),
                         ("ext27_ai", "A_i (direct, no ABC preprocessing)")):
        rows = read_results(os.path.join(root, stage, 'results.csv'))
        if not rows:
            continue
        md.append(f"### {label}\n")
        md.append(sweep_table(rows, DES27))
        md.append("")
        tex.append(f"% {label}\n" + sweep_table(rows, DES27, latex=True) + "\n")

    # 2) ISCAS
    md.append("## ISCAS subset (10 designs)\n")
    for stage, label in (("iscas_ae", "A_e"), ("iscas_iwls", "single-pass")):
        rows = read_results(os.path.join(root, stage, 'results.csv'))
        if not rows:
            continue
        md.append(f"### {label}\n")
        md.append(sweep_table(rows, ISCAS))
        md.append("")

    # 3) against ASPDAC'25's published Table 1
    asp = {}
    for row in read_summary(os.path.join(root, 'tab_main', 'summary.tsv')):
        asp[(row['design'], row['stage'])] = row
    if asp and paper:
        md.append("## tab:main vs ASPDAC'25 (its published Table 1 vs ours, A_e recipe)\n")
        body, texbody, da, dd = [], [], [], []
        for d in ASP16:
            p = paper.get(d)
            ours = asp.get((d, 'ae'))
            if not p or not ours or not verified(ours):
                continue
            to, tod = p[0] * (1 + p[2] / 100.0), p[3] * (1 + p[5] / 100.0)
            oa, od = num(ours['out_a']), num(ours['out_d'])
            if oa is None or od is None:
                continue
            a, e = 100 * (oa - to) / to, 100 * (od - tod) / tod
            da.append(a)
            dd.append(e)
            body.append([d, f"{to:.2f}", f"{tod:.2f}", f"{oa:.2f}", f"{od:.2f}", f"{a:+.2f}", f"{e:+.2f}"])
            texbody.append([esc(d), f"{to:.2f}", f"{tod:.2f}", f"{oa:.2f}", f"{od:.2f}", f"{a:+.2f}", f"{e:+.2f}"])
        hdr = ["design", "ASPDAC'25 area", "ASPDAC'25 delay", "ours area", "ours delay", "dA (%)", "dD (%)"]
        md.append(md_table(hdr, body))
        md.append(f"\n**Means**: dA = {mean(da):+.2f}%, dD = {mean(dd):+.2f}% "
                  f"({len(da)} designs; negative = we are smaller/faster)\n")
        tex.append(tex_table(hdr, texbody) + "\n")

    # 3b) tab:mfs3 -- ABC's `mfs3 -a` and ePMapping from the same A_i start
    mfs = read_summary(os.path.join(root, 'tab_mfs3', 'summary.tsv'))
    abc = {r['design']: r for r in read_summary(os.path.join(root, 'tab_mfs3', 'abc.tsv'))}
    abc_starts = {r['design']: r for r in read_summary(os.path.join(root, 'tab_mfs3', 'abc_baseline_metrics.tsv'))}
    if mfs:
        base = {r['design']: r for r in mfs if r['stage'] == 'base'}
        ours = {r['design']: r for r in mfs if r['stage'] == 'ours'}
        body, texbody, mda, mdd, oda, odd = [], [], [], [], [], []
        paired_area, paired_delay = [], []
        for d in PAPER16:
            b, o = base.get(d), ours.get(d)
            if not b or not o or not verified(o):
                continue
            ba, bd = num(b['base_a']), num(b['base_d'])
            oa, od = num(o['out_a']), num(o['out_d'])
            if ba is None or oa is None or not bd:
                continue
            # ABC's side has area/delay only when `mfs3 -a` returned before the time cap.
            a = abc.get(d, {})
            native = abc_starts.get(d, {})
            aba, abd = num(native.get('abc_area')) or ba, num(native.get('abc_delay')) or bd
            aa, ad = num(a.get('area')), num(a.get('delay'))
            if aa is not None and ad is not None and verified(a):
                mrow = [f"{100 * (aa - aba) / aba:+.2f}", f"{100 * (ad - abd) / abd:+.2f}", a.get('secs', '')]
                mda.append(100 * (aa - aba) / aba)
                mdd.append(100 * (ad - abd) / abd)
            else:
                mrow = ["--", "--", "TO" if a.get('rc') == '142' else "unverified"]
            da, dd = 100 * (oa - ba) / ba, 100 * (od - bd) / bd
            if aa is not None and ad is not None and verified(a):
                paired_area.append(da)
                paired_delay.append(dd)
            oda.append(da)
            odd.append(dd)
            row = [f"{ba:.2f}", f"{bd:.2f}", f"{aba:.2f}", f"{abd:.2f}"] + mrow + [f"{da:+.2f}", f"{dd:+.2f}", o['secs']]
            body.append([d] + row)
            texbody.append([esc(d)] + row)
        if body:
            hdr = ["design", "ours A_i area", "ours A_i delay", "ABC A_i area", "ABC A_i delay", "mfs3 dA (%)", "mfs3 dD (%)", "mfs3 s",
                   "ours dA (%)", "ours dD (%)", "ours s"]
            mstr = (f"mean dA = {mean(mda):+.2f}%, mean dD = {mean(mdd):+.2f}% over the {len(mda)} of "
                    f"{len(body)} it completes") if mda else "no completed run"
            md.append("## tab:mfs3 (common serialized A_i input on 40 ASAP7 cells, one pass each)\n")
            if abc_starts:
                md.append("Native parsed baseline accounting can differ through output buffers; each percentage uses its own measured start.\n")
            md.append(md_table(hdr, body))
            md.append(f"\n**`mfs3 -a`**: {mstr}; **ePMapping**: mean dA = {mean(oda):+.2f}%, "
                      f"mean dD = {mean(odd):+.2f}% over all {len(oda)}. ABC's numbers come from "
                      f"`tab_mfs3/abc.tsv`, its logs from `tab_mfs3/abc.<design>.log`.\n")
            if paired_area:
                md.append(f"On the same {len(paired_area)} completed designs, ePMapping has mean "
                          f"dA = {mean(paired_area):+.2f}% and dD = {mean(paired_delay):+.2f}%. "
                          "Ours s is whole-process wall time; ABC s excludes its separate CEC.\n")
            tex.append(tex_table([esc(h) for h in hdr], texbody) + "\n")

    # 3c) the figure data
    figure_data(root, md)

    # 4) stage summaries
    for stage, cols in (('tab_main', ['stage', 'design', 'base_a', 'out_a', 'dA%']),
                        ('oracle', ['stage', 'design', 'base_a', 'out_a', 'dA%']),
                        ('db', ['stage', 'design', 'base_a', 'out_a', 'dA%']),
                        ('funnel', ['stage', 'design', 'base_a', 'out_a', 'secs']),
                        ('lever1', ['stage', 'design', 'base_a', 'out_a', 'dA%']),
                        ('lever1_micro', ['stage', 'design', 'base_a', 'out_a', 'dA%', 'secs']),
                        ('lever3', ['stage', 'design', 'base_a', 'out_a', 'dA%']),
                        ('runtime', ['stage', 'design', 'base_a', 'out_a', 'secs']),
                        ('ablation', ['stage', 'design', 'base_a', 'out_a', 'dA%'])):
        rows = read_summary(os.path.join(root, stage, 'summary.tsv'))
        if not rows:
            continue
        md.append(f"## stage `{stage}`\n")
        md.append(md_table(cols, [[r.get(c, '') for c in cols] for r in rows]))
        md.append("")

    text = "\n".join(md)
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        open(args.markdown, 'w').write(text + "\n")
        print(f"[i] wrote {args.markdown}")
    if args.latex:
        os.makedirs(os.path.dirname(os.path.abspath(args.latex)), exist_ok=True)
        open(args.latex, 'w').write("\n\n".join(tex) + "\n")
        print(f"[i] wrote {args.latex}")
    if not args.markdown and not args.latex:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
