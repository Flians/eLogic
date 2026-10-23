"""Select one global rho from complete verified data; retain every failure."""
import argparse
import copy
import csv
import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from scaled_verification_cache import sha

ROOT = Path(__file__).resolve().parents[3]


def geometric(values):
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--repeats", required=True, type=Path)
    parser.add_argument("--baseline-supplements", required=True, type=Path)
    parser.add_argument("--large-repeats",required=True,type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    full = json.loads((args.full / "validation.json").read_text())
    repeats = json.loads((args.repeats / "validation.json").read_text())
    campaign = json.loads((args.full / "campaign.json").read_text())
    repeat_campaign = json.loads((args.repeats / "campaign.json").read_text())
    assert full["complete"] and full["attempts"] == 120 and repeats["complete"]
    assert repeats["attempts"] == 54
    large=json.loads((args.large_repeats/"validation.json").read_text());assert large["complete"] and large["attempts"]==18
    assert json.loads((args.large_repeats/"campaign.json").read_text())["fingerprint"]==campaign["fingerprint"]
    assert campaign["fingerprint"] == repeat_campaign["fingerprint"]
    assert all(sha(p) == h for p, h in campaign["fingerprint"].items())
    baseline_supp = json.loads((args.baseline_supplements / "campaign.json").read_text())
    assert baseline_supp["status"] == "attempted" and len(baseline_supp["jobs"]) == 8
    assert all(sha(p) == h for p, h in baseline_supp["fingerprint"].items())
    for job in baseline_supp["jobs"]:
        assert all(sha(p) == h for p, h in job["sha256"].items())
    by_ratio = {r: [x for x in full["rows"] if x["ratio"] == r] for r in campaign["configuration"]["ratios"]}
    valid = {r: {(x["bench"], x["k"]): x for x in rows if x["status"] == "ok"} for r, rows in by_ratio.items()}
    common = set.intersection(*(set(v) for v in valid.values()))
    summary = {}
    for ratio, rows in by_ratio.items():
        assert len(rows) == 40
        summary[ratio] = dict(full["counts"][ratio],
            common_verified_cases=len(common),
            delay_geomean=geometric([valid[ratio][key]["delay"] for key in common]),
            area_geomean=geometric([valid[ratio][key]["area"] for key in common]),
            mapping_seconds_geomean=geometric([x["mapping_secs"] for x in rows]),
            missing=[dict(k=x["k"], bench=x["bench"], error=x["error"]) for x in rows if x["status"] != "ok"])
    eligible = [r for r in summary if summary[r]["verified"] == max(v["verified"] for v in summary.values())]
    if common:
        best = min(summary[r]["delay_geomean"] for r in eligible)
        eligible = [r for r in eligible if summary[r]["delay_geomean"] <= best * (1 + 1e-6)]
        best = min(summary[r]["area_geomean"] for r in eligible)
        eligible = [r for r in eligible if summary[r]["area_geomean"] <= best * (1 + 1e-6)]
    chosen = min(eligible, key=Fraction)
    from scaled_ratio_selection import select
    assert select(full["rows"],campaign["configuration"]["ratios"])[0]==chosen
    assert all(r["ratio"]==chosen for r in large["rows"])
    scope = out / "baseline_scope"
    scope.mkdir()
    required = ["abc_map_area_resyn2", "abc_map_delay_resyn2", "mockturtle_emap_area_resyn2",
                "mockturtle_emap_delay_resyn2", "eqmap_lut_amap", "esyn_frozen_model_amap_bounded_dag", "emorphic_final_sa"]
    gates = []
    overlays = []
    for k in [3, 4]:
        original = ROOT / f"results_v2/guarded_joint_evidence_20261016/baseline_scope/baselines_{k}in.json"
        baseline = json.loads(original.read_text())
        for supplement in baseline_supp["jobs"]:
            case = supplement["original"]
            if case["k"] != k or supplement["status"] != "ok":
                continue
            matches = [r for r in baseline["results"] if r["bench"] == case["bench"] and r["method"] == case["method"]
                       and r["observation_report"] == case["original_report"]]
            assert len(matches) == 1
            row = matches[0]
            row["original_verification_failure"] = dict(status=row["status"], error=row.get("error"))
            row["status"], row["measured"] = "ok", supplement["measured"]
            row["mapping_wall_secs"] = case["mapping_wall_secs"]
            row.pop("error", None)
            row["verification_supplement"] = supplement
            overlays.append(dict(k=k, bench=case["bench"], method=case["method"], mapping_rerun=False))
        baseline["verification_only_supplements"] = True
        base_path = scope / f"baselines_{k}in.json"
        base_path.write_text(json.dumps(baseline, indent=2))
        candidate = {key: baseline[key] for key in ["library_sha256", "metric_model"]}
        candidate["results"] = []
        for point in by_ratio[chosen]:
            if point["k"] != k:
                continue
            doc = json.loads(Path(point["report"]).read_text())
            assert doc["library_sha256"] == baseline["library_sha256"]
            row = copy.deepcopy(doc["results"][0])
            row["method"] = "eGraphMap"
            candidate["results"].append(row)
        assert len(candidate["results"]) == 20
        candidate_path = scope / f"egraphmap_{k}in.json"
        candidate_path.write_text(json.dumps(candidate, indent=2))
        for objective in ["delay-area", "area-delay", "pareto"]:
            dest = scope / f"{k}in_{objective}.json"
            argv = [sys.executable, str(ROOT / "scripts/experiments/check_mapping_dominance.py"),
                    "--candidate", str(candidate_path), "--baselines", str(base_path),
                    "--required-baselines", *required, "--objective", objective, "--output", str(dest)]
            result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            dest.with_suffix(".log").write_text(result.stdout)
            assert dest.exists()
            gates.append(dict(k=k, objective=objective, passed=json.loads(dest.read_text())["all_passed"], argv=argv))
    ranges = []
    for ratio in by_ratio:
        for bench, k in sorted({(r["bench"], r["k"]) for r in repeats["rows"]}):
            rows = [r for r in repeats["rows"] if r["ratio"] == ratio and r["bench"] == bench and r["k"] == k]
            good = [r for r in rows if r["status"] == "ok"]
            item = dict(ratio=ratio, bench=bench, k=k, attempts=len(rows), verified=len(good),
                        optimized_selected=sum(r["optimized_selected_and_CEC"] for r in rows))
            for metric in ["area", "delay", "mapping_secs"]:
                values = [r[metric] for r in good]
                item[metric] = dict(values=values, min=min(values), max=max(values)) if values else None
            ranges.append(item)
    for label, rows in [("full", full["rows"]), ("repeats", repeats["rows"])]:
        with (out / (label + ".csv")).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    report = dict(complete=True, selected_ratio=chosen, per_design_selection=False,
        selection_rule="coverage, common-scope geometric delay, area, lower rho; relative1e-6 ties",
        full_scope=120, common_verified_cases=len(common), summary=summary, gates=gates,
        baseline_verification_only_overlays=overlays, baseline_algorithms_changed=False,
        native_main_not_promoted=True, new_resource_protocol_not_unchanged20k=True,
        repetitions=54, repeat_ranges=ranges, large_repeat_attempts=18,large_repeat_rows=large["rows"],main_rows_from_first_full_sweep_only=True,
        source_only_fallbacks_not_optimization_successes=True)
    (out / "qualification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ["repeat_ranges", "gates"]}, indent=2))


if __name__ == "__main__":
    main()
