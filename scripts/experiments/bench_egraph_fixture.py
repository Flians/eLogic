#!/usr/bin/env python3
"""Same-binary synthetic three-disjoint-output ablation with common metrics/CEC."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from bench_egraph_baselines import quoted, record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture = root / "emapper/tests/fixtures"
    lib, golden = fixture / "disjoint_triple.genlib", fixture / "disjoint_triple.aag"
    mapper, evaluator = [root / "target/release" / n for n in
                         ["bench_rs_egraph_map", "bench_rs_netlist_eval"]]
    env = {k: v for k, v in os.environ.items() if not k.startswith("EGM_")}
    env.update(EGM_K="3", EGM_CUTS="16", EGM_TIME="0", EGM_DC_SUPPORT="0",
               EGM_NOCEC="1", EGM_MULTI_CUMULATIVE="1", EGM_STRUCTURAL_REFS="0",
               EGM_FLOW_DELAY_TIES="0", EGM_REFINE_TIME="2", EGM_REFINE_ROUNDS="4",
               EGM_REFINE_TRIALS="2000", EGM_BOOLEAN="0", EGM_MULTIOUTPUT="1",
               EGM_YOSYS="/usr/local/bin/yosys", BENCH_RS_EMAP_ABC="/usr/bin/abc")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {"scope": "synthetic mechanism ablation only; not production library performance",
              "library_sha256": digest(lib), "input_sha256": digest(golden),
              "metric_model": "all physical instances; max(rise,fall) block pin delay; no load or slew",
              "binary_sha256": {str(p): digest(p) for p in [mapper, evaluator]},
              "source_sha256": {str(p.relative_to(root)): digest(p) for p in
                                [Path(__file__), root / "emapper/src/egraph_map.rs"]},
              "commands": [], "results": []}

    def run(argv, log, local=env):
        report["commands"].append(argv)
        start = time.perf_counter()
        p = subprocess.run(argv, cwd=root, env=local, capture_output=True, text=True, timeout=30)
        log.write_text(p.stdout + p.stderr)
        if p.returncode:
            raise RuntimeError(f"exit {p.returncode}: {log}")
        return p.stdout, time.perf_counter() - start

    models = out / "models.v"
    run([str(evaluator), str(lib), "--models", str(models), str(out / "library.json")], out / "models.log")
    for name, cumulative, seconds in [("pair_only", "0", "0"),
                                      ("cumulative_heuristic", "1", "0"),
                                      ("cumulative_cbc", "1", "5")]:
        work = out / name
        work.mkdir()
        row = {"bench": "disjoint_triple", "method": name, "input_sha256": digest(golden)}
        local = dict(env, EGM_MULTI_CUMULATIVE=cumulative, EGM_TIME=seconds)
        row["settings"] = {k: v for k, v in local.items() if k.startswith("EGM_")}
        try:
            mapped, netjson = work / "mapped.v", work / "mapped.json"
            text, row["mapping_wall_secs"] = run([str(mapper), str(lib), str(golden), str(mapped)],
                                                  work / "map.log", local)
            row["native"] = record(text)
            run([env["EGM_YOSYS"], "-Q", "-T", "-p",
                 f"read_verilog -lib {quoted(models)}; read_verilog {quoted(mapped)}; "
                 f"hierarchy -check -top mapped; write_json {quoted(netjson)}"], work / "normalize.log")
            text, row["evaluation_wall_secs"] = run(
                [str(evaluator), str(lib), str(golden), str(mapped), str(netjson)], work / "evaluate.log")
            row["measured"] = record(text)
            if row["measured"]["cec"] != 1:
                raise RuntimeError("independent CEC failed")
            row["status"] = "ok"
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            row.update(status="failed", error=str(e))
        report["results"].append(row)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, row.get("measured", row.get("error")), flush=True)
    if any(r["status"] != "ok" for r in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
