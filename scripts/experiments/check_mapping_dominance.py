#!/usr/bin/env python3
"""Check verified candidate covers against every measured, same-library baseline.

Do not equate beating minimum area with dominating all area/delay tradeoffs.
This gate checks both metrics, inputs, library, CEC and requested coverage.
"""
import argparse
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--baselines", type=Path, nargs="+", required=True)
    ap.add_argument("--required-baselines", nargs="+", default=[])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--objective", choices=["pareto", "area-delay", "delay-area"], default="pareto")
    ap.add_argument("--candidate-methods", nargs="+", help="Explicit candidate filter for reports with matched baseline controls")
    ap.add_argument("--comparison", choices=["exact", "legacy-relative"], default="exact",
                    help="exact final lexicographic/physical comparison; legacy mode only reproduces old gates")
    a = ap.parse_args()
    candidate = json.loads(a.candidate.read_text())
    methods = {}
    for path in a.baselines:
        report = json.loads(path.read_text())
        for key in ["library_sha256", "metric_model"]:
            if report[key] != candidate[key]:
                raise ValueError(f"incompatible {key}: {path}")
        for r in report["results"]:
            if r["method"].startswith(("abc_", "mockturtle_", "eqmap_", "esyn_", "emorphic_", "lut_stage_")):
                # Two objective profiles or randomized runs can share a method
                # name but have different covers. Never overwrite the earlier
                # observation; coalesce only identical metric/status records.
                fingerprint = json.dumps({k: r.get(k) for k in
                    ["method", "input_sha256", "status", "measured", "error"]}, sort_keys=True)
                key = (r["bench"], fingerprint)
                if key not in methods:
                    methods[key] = dict(r, baseline_sources=[])
                methods[key]["baseline_sources"].append(str(path))

    def validated(r):
        m = r.get("measured", {})
        return (r["status"] == "ok" and m.get("cec") == 1 and
                all(isinstance(m.get(k), (int, float)) and math.isfinite(m[k]) and m[k] >= 0
                    for k in ["area", "delay"]))

    def leq(x, y):
        return x <= y if a.comparison == "exact" else x <= y + 1e-6 * max(1, abs(y))

    def wins_against(m, baseline):
        if a.objective == "pareto":
            return all(leq(m[k], baseline[k]) for k in ["area", "delay"]) and any(
                not leq(baseline[k], m[k]) for k in ["area", "delay"])
        first, second = a.objective.split("-")
        return (not leq(baseline[first], m[first]) or
                (leq(m[first], baseline[first]) and leq(baseline[first], m[first]) and
                 not leq(baseline[second], m[second])))

    rows = []
    for r in candidate["results"]:
        if a.candidate_methods and r["method"] not in a.candidate_methods:
            continue
        references = [b for (bench, _), b in methods.items() if bench == r["bench"]]
        if any(b["input_sha256"] != r["input_sha256"] for b in references):
            raise ValueError("benchmark inputs differ")
        missing = sorted(set(a.required_baselines) - {b["method"] for b in references})
        invalid = [b["method"] for b in references if not validated(b)]
        valid = [b for b in references if validated(b)]
        row = {"bench": r["bench"], "method": r["method"], "missing": missing,
               "invalid_baselines": invalid, "candidate_valid": validated(r)}
        if valid and validated(r):
            m = r["measured"]
            wins = [b["method"] for b in valid if wins_against(m, b["measured"])]
            row.update(measured=m, baseline_count=len(references), dominated_baselines=wins,
                       baseline_observations=[{"method": b["method"], "measured": b["measured"],
                           "sources": b["baseline_sources"], "wins": wins_against(m, b["measured"])}
                           for b in valid],
                       minimum_baseline_area=min(b["measured"]["area"] for b in valid),
                       minimum_baseline_delay=min(b["measured"]["delay"] for b in valid))
            row["dominates_all"] = len(wins) == len(references) and not missing and not invalid
        else:
            row["dominates_all"] = False
        rows.append(row)
        print(f"{row['bench']}: dominates_all={row['dominates_all']}; "
              f"dominated={len(row.get('dominated_baselines', []))}/{len(references)}")
    missing_designs = sorted({b for b, _ in methods} - {r["bench"] for r in rows})
    result = {"scope": "only measured baselines; unconnected tools are not covered",
              "objective": a.objective, "comparison_policy": a.comparison,
              "candidate_methods_filter": a.candidate_methods,
              "definition": ("strict Pareto or named lexicographic comparison; exact finite physical metrics" if a.comparison=="exact" else "legacy strict comparison within 1e-6 relative tolerance")+"; lexicographic wins are not Pareto dominance",
              "missing_candidate_designs": missing_designs,
              "all_passed": bool(rows) and not missing_designs and all(r["dominates_all"] for r in rows),
              "candidate_report": str(a.candidate), "baseline_reports": list(map(str, a.baselines)),
              "results": rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
