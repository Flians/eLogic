#!/usr/bin/env python3
"""Assemble fixed-subject results, preserving originals and explicit adapter repairs."""
import hashlib
import json
from pathlib import Path
import subprocess
import platform
import os

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results_v2"
OUT = R / "fixed_complete_20261012"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def read(folder):
    p = R / folder / "report.json"
    d = json.loads(p.read_text())
    d["collection_source"] = {"path": str(p.relative_to(ROOT)), "sha256": sha(p)}
    return d


def write(name, d):
    (OUT / name).write_text(json.dumps(d, indent=2)+"\n")


def main():
    OUT.mkdir(exist_ok=False)
    inventory = {"scope": "7 designs with 3-input library; 3 designs with 4-input library; fixed resyn2",
                 "uncovered": "industrial multi-output library; ODC; unbounded/published vendor flows",
                 "hardware": {"platform": platform.platform(), "cpu_affinity_count": len(os.sched_getaffinity(0))},
                 "runtime_note": "independent experiments ran concurrently; wall times are recorded, not speedup claims",
                 "collector_sha256": sha(Path(__file__)), "suites": [], "historical_failures": []}
    for k, prefix, expected in [(3, "fixed_all", 7), (4, "fixed_4in", 3)]:
        area, delay, eqarea, eqdelay, esyn, sa = [read(prefix + n + "_20261012") for n in
            ["_area", "_delay", "_eqmap", "_eqmap_delay", "_esyn", "_emorphic"]]
        sources = [d["collection_source"] for d in [area, delay, eqarea, eqdelay, esyn, sa]]
        if len({d["library_sha256"] for d in [area, delay, eqarea, eqdelay, esyn, sa]}) != 1:
            raise ValueError("library mismatch")
        if k == 3:
            repaired_eq = read("fixed_eqmap_delay_c2670_repair2_20261012")
            repaired_esyn = read("fixed_esyn_c2670_repair_20261012")
            sources += [d["collection_source"] for d in [repaired_eq, repaired_esyn]]
            # The native algorithms are never rerun or selected differently for repairs.
            replay = next(r for r in repaired_eq["results"] if r["method"] == "eqmap_lut_amap")
            original = R / "fixed_all_eqmap_delay_20261012/c2670/eqmap_lut_amap/eqmap.v"
            assert sha(original) == replay["replayed_native_sha256"]
            old_candidates = {r["candidate"]: r["sexpr_sha256"] for r in esyn["candidates"] if r["bench"] == "c2670"}
            new_candidates = {r["candidate"]: r["sexpr_sha256"] for r in repaired_esyn["candidates"]}
            assert old_candidates == new_candidates and len(new_candidates) == 30
            for d, repair, method in [(eqdelay, repaired_eq, "eqmap_lut_amap"), (esyn, repaired_esyn, None)]:
                for i, row in enumerate(d["results"]):
                    if row["bench"] == "c2670" and (method is None or row["method"] == method):
                        assert row["status"] == "failed"
                        inventory["historical_failures"].append(dict(row, source=d["collection_source"]))
                        replacement = next(r for r in repair["results"] if r["bench"] == row["bench"] and r["method"] == row["method"])
                        assert replacement["status"] == "ok" and replacement["input_sha256"] == row["input_sha256"]
                        d["results"][i] = dict(replacement, repaired_from=row,
                            repair_source=repair["collection_source"])
            esyn["candidates"] = [r for r in esyn["candidates"] if r["bench"] != "c2670"] + repaired_esyn["candidates"]
        benches = sorted({r["bench"] for r in area["results"]})
        assert len(benches) == expected
        # Shared hashes are required in addition to independent golden-input CEC.
        subjects = {b: next(r["preprocessing"]["sha256"] for r in area["results"] if r["bench"] == b) for b in benches}
        for d in [delay, eqarea, eqdelay, esyn, sa]:
            for row in d["results"]:
                assert row["status"] == "ok" and row["measured"]["cec"] == 1
                assert row["preprocessing"]["sha256"] == subjects[row["bench"]]
        assert len(esyn["candidates"]) == expected * 30
        assert len(sa["candidates"]) == expected * 70
        assert all(r["status"] == "ok" and r["measured"]["cec"] == 1 for d in [esyn, sa] for r in d["candidates"])
        baseline_rows = [r for r in area["results"] if not r["method"].startswith("egraph")]
        baseline_rows += eqarea["results"] + eqdelay["results"] + sa["results"]
        for objective in ["area-delay", "delay-area"]:
            for bench in benches:
                candidates = [r for r in esyn["candidates"] if r["bench"] == bench]
                first, second = objective.split("-")
                best = min(candidates, key=lambda r: (r["measured"][first], r["measured"][second]))
                source = next(r for r in esyn["results"] if r["bench"] == bench)
                baseline_rows.append(dict(source, measured=best["measured"], selected_candidate=best["candidate"],
                    selection_objective=objective, selection_note="same verified upstream ranked pool; no new native search"))
        baseline = {"library_sha256": area["library_sha256"], "metric_model": area["metric_model"],
                    "preprocess": "resyn2", "source_reports": sources, "results": baseline_rows}
        write(f"baselines_{k}in.json", baseline)
        write(f"eqmap_{k}in_area.json", eqarea)
        write(f"eqmap_{k}in_delay.json", eqdelay)
        for objective, data in [("area-delay", area), ("delay-area", delay)]:
            for check in [objective, "pareto"]:
                output = OUT / f"{k}in_{objective}_{check}.json"
                argv = ["python3", str(ROOT / "scripts/experiments/check_mapping_dominance.py"),
                        "--candidate", str(R / (prefix + ("_area" if objective == "area-delay" else "_delay") + "_20261012/report.json")),
                        "--candidate-methods", "egraph_resyn2", "--baselines", str(OUT / f"baselines_{k}in.json"),
                        "--required-baselines", "mockturtle_emap_area_resyn2", "mockturtle_emap_delay_resyn2",
                        "eqmap_lut_amap", "esyn_frozen_model_amap_bounded_dag", "emorphic_final_sa",
                        "--objective", check, "--output", str(output)]
                run = subprocess.run(argv, text=True, capture_output=True)
                print(run.stdout.strip())
                write(output.stem + ".command.json", {"argv": argv, "exit_code": run.returncode, "stderr": run.stderr})
        all_records = area["results"] + delay["results"] + eqarea["results"] + eqdelay["results"]
        # Counts exclude retained failed adapter attempts and duplicate repair controls.
        successes = len(all_records) + len(esyn["candidates"]) + len(sa["candidates"])
        inventory["suites"].append({"library_max_inputs": k, "designs": benches, "source_reports": sources,
             "successful_netlist_records_including_internal_candidates": successes,
             "native_esyn_candidates": len(esyn["candidates"]), "native_final_sa_candidates": len(sa["candidates"]),
             "emap_configurations": 2, "expected_coverage_complete": True})
    inventory["successful_netlist_records"] = sum(s["successful_netlist_records_including_internal_candidates"] for s in inventory["suites"])
    write("inventory.json", inventory)


if __name__ == "__main__": main()
