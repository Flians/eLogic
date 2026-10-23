#!/usr/bin/env python3
"""Pinned EqMap LUT remapping plus a common ABC standard-cell backend.

This is a cross-target adapter, not a native ASIC EqMap reproduction. Compare
the LUT staging control to isolate EqMap from the staging/backend effects.
"""
import argparse
from benchmark_suites import add_suite_arguments, resolve_suite
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from bench_egraph_baselines import quoted, record
from shared_subject import reuse_subject

PINNED_COMMIT = "d3144c45d3441530cd8d240c5fc6ea6645b8a2fb"


def lut_models():
    text = ""
    for k in range(1, 7):
        inputs = ", ".join(f"input I{i}" for i in range(k))
        index = ",".join(f"I{i}" for i in reversed(range(k)))
        text += (f"module LUT{k} #(parameter [{(1 << k)-1}:0] INIT=0) "
                 f"({inputs}, output O); assign O=INIT[{{{index}}}]; endmodule\n")
    return text


def lut_adapter():
    text = '(* techmap_celltype = "$lut" *) module lut_adapter(A,Y);\n'
    text += "parameter WIDTH=1; parameter [2**WIDTH-1:0] LUT=0;\n"
    text += "input [WIDTH-1:0] A; output Y; generate\n"
    for k in range(1, 7):
        pins = ",".join(f".I{i}(A[{i}])" for i in range(k))
        text += f"if (WIDTH=={k}) begin LUT{k} #(.INIT(LUT)) _TECHMAP_REPLACE_({pins},.O(Y)); end\n"
    return text + "endgenerate endmodule\n"


def lut_metrics(path):
    """Count every LUT instance; measure worst output depth on the used DAG."""
    top = json.loads(path.read_text())["modules"]["mapped"]
    drivers, inputs = {}, set()
    for port in top["ports"].values():
        if port["direction"] == "input":
            inputs.update(port["bits"])
    counts, primitives = {}, {}
    for cell in top["cells"].values():
        kind = cell["type"]
        pins = {"MUX": ["S", "A", "B"], "AND": ["A", "B"], "OR": ["A", "B"],
                "NAND": ["A", "B"], "NOR": ["A", "B"], "XOR": ["A", "B"],
                "AND2": ["A", "B"], "OR2": ["A", "B"], "NAND2": ["A", "B"],
                "NOR2": ["A", "B"], "XOR2": ["A", "B"], "NOT": ["A"], "INV": ["I"]}
        is_lut = bool(re.fullmatch(r"LUT[1-6]", kind))
        if not is_lut and kind not in pins:
            raise RuntimeError(f"non-LUT cell in LUT stage: {kind}")
        table = counts if is_lut else primitives
        table[kind] = table.get(kind, 0) + 1
        output = cell["connections"]["O" if is_lut or kind == "INV" else "Y"]
        if len(output) != 1 or output[0] in drivers:
            raise RuntimeError("invalid or multiply-driven LUT output")
        inputs_used = [f"I{i}" for i in range(int(kind[-1]))] if is_lut else pins[kind]
        drivers[output[0]] = [cell["connections"][pin][0] for pin in inputs_used]
    memo, visiting = {}, set()

    def depth(bit):
        if bit in inputs or bit in ("0", "1"):
            return 0
        if bit in memo:
            return memo[bit]
        if bit in visiting or bit not in drivers:
            raise RuntimeError("cyclic or undriven LUT net")
        visiting.add(bit)
        memo[bit] = 1 + max(map(depth, drivers[bit]), default=0)
        visiting.remove(bit)
        return memo[bit]

    for bit in drivers:
        depth(bit)
    worst = max((depth(bit) for port in top["ports"].values()
                 if port["direction"] == "output" for bit in port["bits"]), default=0)
    return {"lut_count": sum(counts.values()), "lut_depth": worst, "lut_types": counts,
            "primitive_types": primitives,
            "max_lut_inputs": max((int(t[-1]) for t in counts), default=0)}


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eqmap-source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=int, choices=range(2, 7), default=3)
    ap.add_argument("--lut-k", type=int, choices=range(2, 7), default=6,
                    help="LUT width, independent of standard-cell library input limit")
    add_suite_arguments(ap)
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--prepared-report", type=Path)
    ap.add_argument("--objective", choices=["area-delay", "delay-area"], default="area-delay")
    ap.add_argument("--replay-netlist", type=Path, help="Recheck one retained native output after an adapter repair")
    a = ap.parse_args()
    resolve_suite(a)
    if a.replay_netlist and len(a.benches) != 1: ap.error("replay requires one design")
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = a.eqmap_source.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != PINNED_COMMIT or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source, text=True).strip():
        ap.error("EqMap source must be clean and at the pinned commit")
    binary = source / "target/release/eqmap_fpga"
    evaluator = root / "target/release/bench_rs_netlist_eval"
    env = dict(os.environ, EGM_YOSYS="/usr/local/bin/yosys", BENCH_RS_EMAP_ABC="/usr/bin/abc")
    commands = []

    def run(argv, log):
        commands.append(argv)
        start = time.perf_counter()
        with subprocess.Popen(argv, cwd=root, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, start_new_session=True) as p:
            try:
                stdout, stderr = p.communicate(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                stdout, stderr = p.communicate()
                log.write_text(stdout + stderr)
                raise RuntimeError("process-group timeout")
            log.write_text(stdout + stderr)
            if p.returncode:
                raise RuntimeError(f"exit {p.returncode}: {log}")
        return stdout, time.perf_counter() - start

    def yosys(script, log):
        return run([env["EGM_YOSYS"], "-Q", "-T", "-p", script], log)

    lib, models = out / f"asap7_{a.k}in.genlib", out / "cells.v"
    run(["python3", str(root / "scripts/experiments/restrict_genlib.py"), "--in",
         str(root / "tools/mockturtle/experiments/cell_libraries/asap7.genlib"),
         "--out", str(lib), "--max-pins", str(a.k)], out / "library.log")
    run([str(evaluator), str(lib), "--models", str(models), str(out / "library.json")],
        out / "models.log")
    lmodels, adapter = out / "luts.v", out / "lut_adapter.v"
    lmodels.write_bytes((source / "verilog/simlib.v").read_bytes())
    adapter.write_text(lut_adapter())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {"eqmap_commit": commit, "eqmap_source": "https://github.com/cornell-zhang/eqmap",
              "flow": "Yosys abc -lut K -> EqMap greedy arc-set -> ABC amap",
              "k": a.k, "lut_k": a.lut_k,
              "eqmap_limits": {"seconds": 2, "nodes": 20000, "iterations": 16},
              "library_sha256": digest(lib), "binary_sha256": digest(binary),
              "adapter_sha256": digest(adapter), "models_sha256": digest(lmodels),
              "harness_sha256": digest(Path(__file__)), "commands": commands, "results": [],
              "metric_model": "all physical instances; max(rise,fall) block pin delay; no load or slew",
              "k_semantics": "EqMap current KLUTCostFn uses finite penalties above K, not a hard bound",
              "preprocess": "resyn2" if a.prepared_report else "none",
              "objective": a.objective,
              "tool_binary_sha256": {str(p): digest(p) for p in [binary, evaluator,
                                      Path(env["EGM_YOSYS"]), Path(env["BENCH_RS_EMAP_ABC"])]}}
    for name, argv in [("eqmap", [str(binary), "--version"]),
                       ("yosys", [env["EGM_YOSYS"], "-V"]),
                       ("abc", [env["BENCH_RS_EMAP_ABC"], "-c", "version"])]:
        report[name + "_version"], _ = run(argv, out / f"{name}-version.log")
    for bench in a.benches:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", bench):
            ap.error("invalid benchmark name")
        golden = root / f"tools/mockturtle/experiments/benchmarks/{bench}.aig"
        folder = out / bench
        folder.mkdir()
        subject = golden
        preprocessing = {}
        if a.prepared_report:
            subject = folder / "subject.aig"
            preprocessing = reuse_subject(a.prepared_report, bench, golden, lib, subject)
        stage = folder / "input_luts.v"
        yosys(f"read_aiger -module_name mapped {quoted(subject)}; abc -lut {a.lut_k}; "
              f"techmap -map {quoted(adapter)}; splitnets -ports -format _; "
              f"write_verilog -noattr -noexpr {quoted(stage)}", folder / "prepare.log")
        for method in ["abc_amap", "lut_stage_amap", "eqmap_lut_amap"]:
            work = folder / method
            work.mkdir()
            row = {"bench": bench, "method": method, "input_sha256": digest(golden), "preprocessing": preprocessing}
            try:
                aig = subject
                if method != "abc_amap":
                    current = stage
                    if method == "eqmap_lut_amap":
                        current = work / "eqmap.v"
                        if a.replay_netlist:
                            current.write_bytes(a.replay_netlist.read_bytes())
                            row["replayed_native_path"] = str(a.replay_netlist.resolve())
                            row["replayed_native_sha256"] = digest(current)
                        else:
                            _, row["eqmap_wall_secs"] = run(
                            [str(binary), str(stage), str(current), "-k", str(a.lut_k),
                             "-t", "2", "-s", "20000", "-n", "16", "--non-ansi",
                             "--report", str(work / "eqmap-native.json")] +
                            (["--min-depth"] if a.objective == "delay-area" else []), work / "eqmap.log")
                    _, row["lut_cec_wall_secs"] = run(
                        [str(evaluator), "--cec-models", str(lmodels), str(golden), str(current)],
                        work / "lut-cec.log")
                    row["lut_cec"] = 1
                    lutjson = work / "luts.json"
                    yosys(f"read_verilog -lib {quoted(lmodels)}; read_verilog {quoted(current)}; "
                          f"hierarchy -check -top mapped; write_json {quoted(lutjson)}",
                          work / "lut-metrics.log")
                    row["native_lut"] = lut_metrics(lutjson)
                    row["native_lut"]["above_requested_k"] = sum(
                        count for kind, count in row["native_lut"]["lut_types"].items()
                        if int(kind[-1]) > a.lut_k)
                    aig = work / "stage.aig"
                    yosys(f"read_verilog {quoted(lmodels)} {quoted(current)}; hierarchy -top mapped; "
                          f"proc; flatten; techmap; opt; aigmap; write_aiger -symbols {quoted(aig)}",
                          work / "flatten.log")
                raw = work / "raw.v"
                _, row["backend_wall_secs"] = run(
                    [env["BENCH_RS_EMAP_ABC"], "-c", f"read_genlib {quoted(lib)}; "
                     f"read {quoted(aig)}; strash; amap; write_verilog {quoted(raw)}"], work / "map.log")
                if not raw.is_file():
                    raise RuntimeError("ABC produced no netlist")
                mapped, netjson = work / "mapped.v", work / "mapped.json"
                yosys(f"read_verilog -lib {quoted(models)}; read_verilog {quoted(raw)}; "
                      f"hierarchy -check -auto-top; rename -top mapped; "
                      f"write_verilog -noattr -noexpr {quoted(mapped)}; write_json {quoted(netjson)}",
                      work / "normalize.log")
                text, row["evaluation_wall_secs"] = run(
                    [str(evaluator), str(lib), str(golden), str(mapped), str(netjson)], work / "evaluate.log")
                row["measured"] = record(text)
                row["status"] = "ok"
                print(f"{bench}/{method}: {row['measured']}", flush=True)
            except (RuntimeError, ValueError, KeyError) as error:
                row.update(status="failed", error=str(error))
                print(f"{bench}/{method}: {error}", flush=True)
            report["results"].append(row)
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(r["status"] != "ok" for r in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
