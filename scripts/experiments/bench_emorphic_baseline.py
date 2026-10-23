#!/usr/bin/env python3
"""Pinned E-morphic random extraction + common-library ABC amap adapter.

This tests the upstream random extractor, not the published SA/golden-model flow.
All sampled covers, including failures, are retained and independently checked.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from bench_egraph_baselines import quoted, record
from port_identifiers import identifier

COMMIT = "26c128b5b0453795917474d118b8f89dd7a2ebc9"
PROTO_COMMIT = "1a513f89886408df0c94471de0590df7ab16dcd2"


def input_eqn(path):
    module = json.loads(path.read_text())["modules"]["mapped"]
    if any(len(p["bits"]) != 1 for p in module["ports"].values()):
        raise ValueError("scalar ports required")
    inputs = [(n, p["bits"][0]) for n, p in module["ports"].items() if p["direction"] == "input"]
    outputs = [(n, p["bits"][0]) for n, p in module["ports"].items() if p["direction"] == "output"]
    if not inputs or not outputs:
        raise ValueError("unsupported ports")
    for n, _ in inputs + outputs:
        identifier(n)
    names = {b: f"p{i}" for i, (_, b) in enumerate(inputs)}
    names.update({"0": "0", "1": "1"})
    drivers = {}
    for cell in module["cells"].values():
        c, kind = cell["connections"], cell["type"]
        if kind not in ("$_AND_", "$_NOT_") or c["Y"][0] in drivers:
            raise ValueError("unsupported or multiply driven AIG")
        drivers[c["Y"][0]] = (kind, c["A"][0], c.get("B", [None])[0])
    lines, active = [], set()
    def visit(bit):
        if bit in names: return names[bit]
        if bit in active or bit not in drivers: raise ValueError("cyclic or undriven AIG")
        active.add(bit)
        kind, a, b = drivers[bit]
        rhs = "!" + visit(a) if kind == "$_NOT_" else visit(a) + "*" + visit(b)
        active.remove(bit)
        names[bit] = f"new_n{bit}"
        lines.append(f"{names[bit]} = {rhs};")
        return names[bit]
    result = [f"o{i} = {visit(b)};" for i, (_, b) in enumerate(outputs)]
    text = "INORDER = " + " ".join(f"p{i}" for i in range(len(inputs))) + ";\n"
    text += "OUTORDER = " + " ".join(f"o{i}" for i in range(len(outputs))) + ";\n"
    return text + "\n".join(lines + result) + "\n", inputs, outputs


def cover_verilog(graph, extraction, inputs, outputs):
    nodes, choices = graph["nodes"], extraction["choices"]
    active, wires, assignments = set(), {}, []
    symbols = {f"p{i}": n for i, (n, _) in enumerate(inputs)}
    used_names = {n for n, _ in inputs + outputs}
    def selected(cid):
        if cid not in choices or choices[cid] not in nodes:
            raise ValueError("missing extraction choice")
        node = nodes[choices[cid]]
        if node["eclass"] != cid: raise ValueError("choice has wrong eclass")
        return node
    def child_class(nid):
        if nid not in nodes: raise ValueError("missing child node")
        return nodes[nid]["eclass"]
    def signal(cid):
        if cid in wires: return wires[cid]
        if cid in active: raise ValueError("cyclic extracted cover")
        active.add(cid)
        node = selected(cid)
        op, children = node["op"], node["children"]
        if not children:
            if op in ("0", "false"): value = "1'b0"
            elif op in ("1", "true"): value = "1'b1"
            elif op in symbols: value = symbols[op]
            else: raise ValueError("unknown input symbol: " + op)
        else:
            args = [signal(child_class(n)) for n in children]
            if op == "!" and len(args) == 1: rhs = "~" + args[0]
            elif op in ("*", "+", "->") and len(args) == 2:
                a, b = args
                rhs = f"(~{a} | {b})" if op == "->" else f"({a} {'&' if op == '*' else '|'} {b})"
            else: raise ValueError("unsupported Boolean node: " + op)
            value = "__emorphic_" + str(len(assignments))
            while value in used_names: value += "_"
            used_names.add(value)
            assignments.append((value, rhs))
        active.remove(cid); wires[cid] = value
        return value
    bundle_active = set()
    def bundle(cid):
        if cid in bundle_active: raise ValueError("cyclic output bundle")
        node = selected(cid)
        if node["op"] != "&": return [signal(cid)]
        if len(node["children"]) != 2: raise ValueError("invalid concat")
        bundle_active.add(cid)
        parts = sum((bundle(child_class(n)) for n in node["children"]), [])
        bundle_active.remove(cid)
        return parts
    values = sum((bundle(cid) for cid in graph["root_eclasses"]), [])
    if len(values) != len(outputs): raise ValueError("output count changed")
    text = "module mapped(" + ",".join(n for n, _ in inputs + outputs) + ");\n"
    text += "input " + ",".join(n for n, _ in inputs) + ";\n"
    text += "output " + ",".join(n for n, _ in outputs) + ";\n"
    if assignments: text += "wire " + ",".join(n for n, _ in assignments) + ";\n"
    text += "".join(f"assign {n} = {rhs};\n" for n, rhs in assignments)
    text += "".join(f"assign {n} = {v};\n" for (n, _), v in zip(outputs, values))
    return text + "endmodule\n"


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--benches", nargs="+", default=["c17"])
    ap.add_argument("--objective", choices=["area-delay", "delay-area"], default="area-delay")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--random-prob", type=float, default=0.8)
    ap.add_argument("--timeout", type=float, default=60)
    a = ap.parse_args()
    if a.iterations < 1 or a.samples < 1 or not 0 <= a.random_prob <= 1 or a.timeout <= 0:
        ap.error("invalid budget")
    src, out = a.source.resolve(), a.output.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src, text=True).strip()
    if commit != COMMIT: ap.error("source commit differs")
    out.mkdir(parents=True, exist_ok=False)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    rewrite = src / "e-rewriter/target/release/e-rewriter"
    extract = src / "extraction-gym/target/release/extraction-gym"
    evaluator = root / "target/release/bench_rs_netlist_eval"
    env = dict(os.environ, EGM_YOSYS="/usr/local/bin/yosys", BENCH_RS_EMAP_ABC="/usr/bin/abc",
               RAYON_NUM_THREADS="2", OMP_NUM_THREADS="2")
    report = {"commit": commit, "source": "https://github.com/Gy-Hu/E-Syn2",
              "branch": "SA-golden-model-delaytuning", "objective": a.objective,
              "scope": "native random extractor + common ABC amap; not published SA reproduction",
              "settings": {"iterations": a.iterations, "samples": a.samples, "random_prob": a.random_prob,
                           "native_seconds": 10, "native_nodes": 200000000, "process_timeout": a.timeout,
                           "rng_seed": "upstream thread_rng; not externally seedable"},
              "binary_sha256": {"rewriter": digest(rewrite), "extractor": digest(extract)},
              "proto_source_commit": PROTO_COMMIT,
              "proto_sha256": digest(src / "grpc_communicator/proto/service.proto"),
              "linked_libabc_sha256": digest(src / "extraction-gym/libabc.a"),
              "harness_sha256": digest(Path(__file__)),
              "source_patch": subprocess.check_output(["git", "diff"], cwd=src, text=True),
              "metric_model": "all physical instances; max(rise,fall) block pin delay; no load or slew",
              "commands": [], "results": [], "candidates": []}
    def save(): (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    def run(argv, log, cwd=root):
        report["commands"].append({"argv": argv, "cwd": str(cwd)}); save()
        start = time.perf_counter()
        with subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, start_new_session=True) as p:
            try: stdout, stderr = p.communicate(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); stdout, stderr = p.communicate()
                log.write_text(stdout + stderr + "\nTIMEOUT\n"); raise RuntimeError("process-group timeout")
            log.write_text(stdout + stderr)
            if p.returncode: raise RuntimeError(f"exit {p.returncode}: {log.name}")
        return stdout, time.perf_counter() - start
    def yosys(script, log): return run([env["EGM_YOSYS"], "-Q", "-T", "-p", script], log)
    lib, models = out / "asap7_3in.genlib", out / "cells.v"
    run(["python3", str(root / "scripts/experiments/restrict_genlib.py"), "--in",
         str(root / "tools/mockturtle/experiments/cell_libraries/asap7.genlib"), "--out", str(lib),
         "--max-pins", "3"], out / "library.log")
    run([str(evaluator), str(lib), "--models", str(models), str(out / "library.json")], out / "models.log")
    report["library_sha256"] = digest(lib)
    for bench in a.benches:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", bench): ap.error("invalid benchmark")
        golden = root / f"tools/mockturtle/experiments/benchmarks/{bench}.aig"
        work = out / bench; work.mkdir()
        row = {"bench": bench, "method": "emorphic_random_amap", "input_sha256": digest(golden)}
        try:
            for n in ["rewritten_circuit", "random_graph", "input", "out_json", "out_dag_json", "random_out_dag_json"]:
                (work / n).mkdir()
            netjson = work / "original.json"
            yosys(f"read_aiger -module_name mapped {quoted(golden)}; write_json {quoted(netjson)}", work / "import.log")
            text, inputs, outputs = input_eqn(netjson)
            eqn = work / "input.eqn"; eqn.write_text(text)
            row["eqn_sha256_before"] = digest(eqn)
            _, row["rewrite_secs"] = run([str(rewrite), "input.eqn", str(a.iterations)], work / "rewrite.log", work)
            row["eqn_sha256_after"] = digest(eqn)
            expected = " ".join(f"o{i}" for i in range(len(outputs)))
            actual = re.search(r"OUTORDER = ([^;]+);", eqn.read_text()).group(1)
            if actual != expected: raise ValueError("upstream preprocessing reordered outputs")
            graphfile = work / "input/rewritten_egraph_with_weight_cost_serd.json"
            graphfile.write_bytes((work / "rewritten_circuit/rewritten_egraph_with_weight_cost_serd.json").read_bytes())
            graph = json.loads(graphfile.read_text()); row["graph_sha256"] = digest(graphfile)
            cost = "node_sum_cost" if a.objective == "area-delay" else "node_depth_cost"
            _, row["extract_secs"] = run([str(extract), "input/rewritten_egraph_with_weight_cost_serd.json",
                "--extractor=random-based-faster-bottom-up", "--cost-function=" + cost,
                "--num-samples=" + str(a.samples), "--random-prob=" + str(a.random_prob),
                "--out=extract.log.json"], work / "extract.log", work)
            successful, local = [], []
            for candidate in sorted((work / "random_out_dag_json").glob("*.json")):
                folder = work / candidate.stem; folder.mkdir()
                item = {"bench": bench, "candidate": candidate.name, "choices_sha256": digest(candidate)}
                try:
                    logic = folder / "logic.v"
                    logic.write_text(cover_verilog(graph, json.loads(candidate.read_text()), inputs, outputs))
                    aig = folder / "logic.aig"
                    yosys(f"read_verilog {quoted(logic)}; hierarchy -top mapped; proc; flatten; techmap; opt; "
                          f"aigmap; write_aiger -symbols {quoted(aig)}", folder / "logic.log")
                    raw, mapped, netjson = folder / "raw.v", folder / "mapped.v", folder / "mapped.json"
                    run([env["BENCH_RS_EMAP_ABC"], "-c", f"read_genlib {quoted(lib)}; read {quoted(aig)}; "
                         f"strash; amap; write_verilog {quoted(raw)}"], folder / "map.log")
                    yosys(f"read_verilog -lib {quoted(models)}; read_verilog {quoted(raw)}; hierarchy -check -auto-top; "
                          f"rename -top mapped; write_verilog -noattr -noexpr {quoted(mapped)}; write_json {quoted(netjson)}",
                          folder / "normalize.log")
                    measured_text, _ = run([str(evaluator), str(lib), str(golden), str(mapped), str(netjson)], folder / "evaluate.log")
                    measured = record(measured_text)
                    if measured["cec"] != 1: raise RuntimeError("independent CEC failed")
                    item.update(status="ok", measured=measured); successful.append(item)
                except (RuntimeError, ValueError, KeyError, RecursionError) as e: item.update(status="failed", error=str(e))
                report["candidates"].append(item); local.append(item); save()
            if len(local) != a.samples: raise RuntimeError("upstream sample count differs")
            if not successful: raise RuntimeError("no valid equivalent candidates")
            first, second = a.objective.split("-")
            best = min(successful, key=lambda x: (x["measured"][first], x["measured"][second]))
            row.update(status="ok", measured=best["measured"], selected_candidate=best["candidate"],
                       valid_candidates=len(successful), failed_candidates=len(local)-len(successful))
        except (RuntimeError, ValueError, KeyError, RecursionError) as e: row.update(status="failed", error=str(e))
        report["results"].append(row); save(); print(bench, row, flush=True)
    if any(r["status"] != "ok" for r in report["results"]): raise SystemExit(1)


if __name__ == "__main__": main()
