#!/usr/bin/env python3
"""Compare stock standard-cell mappers with common physical metrics and named-port CEC."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import time

RECIPES = {
    "abc_map_delay": "map",
    "abc_map_area": "map -a",
    "abc_amap": "amap",
    "abc_nf_delay": "&get -n; &nf -K {k} -C 16; &put",
    "abc_nf_area": "&get -n; &nf -a -K {k} -C 16; &put",
    "abc_amap_mfs3": "amap; mfs3 -a",
    "abc_resyn2_nf": ("balance; rewrite; refactor; balance; rewrite; rewrite -z; "
                      "balance; refactor -z; rewrite -z; balance; &get -n; &nf -K {k} -C 16; &put"),
}
SOURCES = {
    "abc": "https://github.com/berkeley-abc/abc",
    "amap": "https://github.com/berkeley-abc/abc/blob/master/src/map/amap/amapCore.c",
    "nf": "https://github.com/berkeley-abc/abc/blob/master/src/aig/gia/giaNf.c",
    "mfs3": "https://people.eecs.berkeley.edu/~alanmi/publications/2016/iwls16_mfs3.pdf",
    "mockturtle_map": "https://mockturtle.readthedocs.io/en/latest/algorithms/mapper.html",
}


def quoted(path):
    value = str(path)
    if any(c in value for c in '\n\r;"'):
        raise ValueError("unsupported script path")
    return '"' + value + '"'


def record(text):
    records = [json.loads(line) for line in text.splitlines() if line.startswith('{"area":')]
    if len(records) != 1:
        raise RuntimeError("expected one JSON metric record")
    return records[0]


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True, help="new result directory")
    ap.add_argument("--k", type=int, choices=range(2, 7), default=3)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--methods", nargs="+", default=[*RECIPES, "mockturtle_map_area",
                                                   "mockturtle_map_delay", "egraph", "egraph_no_sdc"])
    ap.add_argument("--benches", nargs="+", default=["c17", "c432", "c499"])
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evaluator = root / "target/release/bench_rs_netlist_eval"
    mapper = root / "target/release/bench_rs_egraph_map"
    cpp = root / "build/baseline_mapper"
    abc = shutil.which(os.environ.get("ABC", "abc"))
    yosys = shutil.which(os.environ.get("EGM_YOSYS", "yosys"))
    if not abc or not yosys:
        ap.error("ABC and Yosys are required")
    env = {k: v for k, v in os.environ.items() if not k.startswith("EGM_")}
    env.update(EGM_K=str(a.k), EGM_CUTS="16", EGM_TIME="0", EGM_DC_SUPPORT="4",
               EGM_REFINE_TIME="2", EGM_REFINE_ROUNDS="4", EGM_REFINE_TRIALS="2000",
               EGM_BOOLEAN="0", EGM_LIVE_REFS="1", EGM_DELAY_WEIGHT="0",
               EGM_CONE_RECOVERY="1",
               EGM_MULTIOUTPUT="1", EGM_ITER="16", EGM_NODES="20000",
               EGM_CUT_ROUNDS="32", EGM_DC_CONE="1024", EGM_INSTANCES="100000",
               EGM_YOSYS=yosys, BENCH_RS_EMAP_ABC=abc)

    def run(command, log, environment=env):
        started = time.perf_counter()
        with subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, start_new_session=True) as process:
            try:
                stdout, stderr = process.communicate(timeout=a.timeout)
            except subprocess.TimeoutExpired as error:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
                log.write_text(stdout + stderr + f"\nTimeout after {a.timeout}s\n")
                raise RuntimeError(f"timeout after {a.timeout}s") from error
            log.write_text(stdout + stderr)
            if process.returncode:
                raise RuntimeError(f"exit {process.returncode}; see {log.name}")
            return stdout, time.perf_counter() - started

    library = out / f"asap7_{a.k}in.genlib"
    run(["python3", str(root / "scripts/experiments/restrict_genlib.py"), "--in",
         str(root / "tools/mockturtle/experiments/cell_libraries/asap7.genlib"),
         "--out", str(library), "--max-pins", str(a.k)], out / "library.log")
    models = out / "cells.v"
    run([str(evaluator), str(library), "--models", str(models), str(out / "library.json")],
        out / "models.log")
    manifest = {"library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "library": library.name, "settings": {k: v for k, v in env.items() if k.startswith("EGM_")},
                "sources": SOURCES, "timeout_secs": a.timeout, "commands": [], "results": [],
                "metric_model": "all physical instances; max(rise,fall) block pin delay; no load or slew"}
    for label, command in [("abc", [abc, "-c", "version"]), ("yosys", [yosys, "-V"])]:
        manifest[label + "_version"], _ = run(command, out / f"{label}-version.log")
    manifest["binary_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(abc), Path(yosys), evaluator, mapper, cpp] if path.is_file()
    }
    manifest["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), root / "emapper/src/egraph_map.rs",
                     root / "emapper/src/netlist_metrics.rs", root / "emapper/baseline_mapper.cpp",
                     root / "tools/mockturtle/include/mockturtle/algorithms/mapper.hpp"]
    }
    for bench in a.benches:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", bench):
            ap.error("invalid benchmark name")
        golden = root / f"tools/mockturtle/experiments/benchmarks/{bench}.aig"
        for method in a.methods:
            folder = out / f"{bench}.{method}"
            folder.mkdir()
            mapped = folder / "raw.v"
            row = {"bench": bench, "method": method,
                   "input_sha256": hashlib.sha256(golden.read_bytes()).hexdigest()}
            try:
                if method in RECIPES:
                    recipe = RECIPES[method].format(k=a.k)
                    script = (f"read_genlib {quoted(library)}; read {quoted(golden)}; strash; "
                              f"{recipe}; print_stats; write_verilog {quoted(mapped)}")
                    command = [abc, "-c", script]
                elif method.startswith("mockturtle_map_"):
                    command = [str(cpp), str(library), str(golden), str(mapped), method.rsplit('_', 1)[1]]
                elif method in ("egraph", "egraph_no_sdc", "egraph_factor", "egraph_associate",
                                "egraph_no_live_refs", "egraph_timing", "egraph_previous"):
                    command = [str(mapper), str(library), str(golden), str(mapped)]
                else:
                    raise RuntimeError("unknown method")
                manifest["commands"].append({"bench": bench, "method": method, "argv": command})
                local_env = dict(env, EGM_NOCEC="1")
                if method == "egraph_no_sdc":
                    local_env["EGM_DC_SUPPORT"] = "0"
                if method in ("egraph_factor", "egraph_associate"):
                    local_env["EGM_BOOLEAN"] = "1" if method == "egraph_factor" else "2"
                if method == "egraph_no_live_refs":
                    local_env["EGM_LIVE_REFS"] = "0"
                if method == "egraph_previous":
                    local_env["EGM_LIVE_REFS"] = "0"
                    local_env["EGM_CONE_RECOVERY"] = "0"
                if method == "egraph_timing":
                    local_env["EGM_DELAY_WEIGHT"] = "0.01"
                row["settings"] = {k: v for k, v in local_env.items() if k.startswith("EGM_")}
                text, row["mapping_wall_secs"] = run(command, folder / "map.log", local_env)
                if not mapped.is_file():
                    raise RuntimeError("mapper produced no netlist (ABC may return zero on errors)")
                if method.startswith("egraph") or method.startswith("mockturtle"):
                    row["native"] = record(text)
                else:
                    row["native"] = {key: float(re.findall(r'\b' + key + r'\s*=\s*([0-9.]+)', text)[-1])
                                     for key in ("area", "delay")}
                normalized, netjson = folder / "mapped.v", folder / "mapped.json"
                script = (f"read_verilog -lib {quoted(models)}; read_verilog {quoted(mapped)}; "
                          "hierarchy -check -auto-top; rename -top mapped; "
                          f"write_verilog -noattr -noexpr {quoted(normalized)}; write_json {quoted(netjson)}")
                run([yosys, "-Q", "-T", "-p", script], folder / "normalize.log")
                text, row["evaluation_wall_secs"] = run(
                    [str(evaluator), str(library), str(golden), str(normalized), str(netjson)], folder / "evaluate.log")
                row["measured"] = record(text)
                if row["measured"]["cec"] != 1:
                    raise RuntimeError("CEC did not pass")
                row["status"] = "ok"
                print(f"{bench}/{method}: area={row['measured']['area']:.4f}, "
                      f"delay={row['measured']['delay']:.2f}, CEC passed", flush=True)
            except (RuntimeError, ValueError, IndexError) as error:
                row["status"], row["error"] = "failed", str(error)
                print(f"{bench}/{method}: {error}", flush=True)
            manifest["results"].append(row)
            (out / "report.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if any(r["status"] != "ok" for r in manifest["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
