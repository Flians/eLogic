#!/usr/bin/env python3
"""Merge equivalent-library reports, keeping source provenance and latest method measurements."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", nargs="+", type=Path, required=True, help="oldest first, latest last")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    records, sources, library, model, inputs = {}, {}, None, None, {}
    for path in args.reports:
        report = json.loads(path.read_text())
        if library is not None and library != report["library_sha256"]:
            raise ValueError("cannot merge different physical libraries")
        if model is not None and model != report["metric_model"]:
            raise ValueError("cannot merge different timing models")
        library, model = report["library_sha256"], report["metric_model"]
        source = str(path.resolve())
        sources[source] = {k: v for k, v in report.items() if k != "results"}
        for row in report["results"]:
            previous = inputs.setdefault(row["bench"], row["input_sha256"])
            if previous != row["input_sha256"]:
                raise ValueError("benchmark contents differ")
            # Failed replacements remain failed; do not silently retain an older successful result.
            records[(row["bench"], row["method"])] = dict(row, source_report=source)
    comparisons = []
    for bench in sorted(inputs):
        candidates = [r for (b, method), r in records.items() if b == bench and
                      method.startswith(("abc_", "mockturtle_")) and r["status"] == "ok"]
        candidate = records.get((bench, "egraph"))
        if not candidates or candidate is None or candidate["status"] != "ok":
            continue
        best = min(candidates, key=lambda r: (round(r["measured"]["area"], 6), r["measured"]["delay"]))
        row = {"bench": bench, "egraph": candidate["measured"],
               "minimum_area_baseline": best["method"], "baseline": best["measured"]}
        area = best["measured"]["area"]
        row["area_vs_minimum_baseline_percent"] = 100 * (candidate["measured"]["area"] - area) / area if area else None
        comparisons.append(row)
    result = {"library_sha256": library, "metric_model": model, "sources": sources,
              "results": list(records.values()), "comparisons": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in comparisons:
        print(f"{row['bench']}: egraph {row['egraph']['area']:.2f}/{row['egraph']['delay']:.2f}; "
              f"{row['minimum_area_baseline']} {row['baseline']['area']:.2f}/{row['baseline']['delay']:.2f}")


if __name__ == "__main__":
    main()
