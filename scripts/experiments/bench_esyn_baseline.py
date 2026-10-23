#!/usr/bin/env python3
"""E-Syn new-branch frozen-model candidates + common ABC amap; all outputs CEC checked.

This adapter is not a claim of reproducing published full-library numbers.
It preserves upstream rules/models and records the model-path portability patch.
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
from port_identifiers import identifier

COMMIT = "72f5568ab61bac9b8de224199c45fd975ab2399c"


def input_expression(path, dag=False):
    m = json.loads(path.read_text())["modules"]["mapped"]
    inputs = [(n, p["bits"][0]) for n, p in m["ports"].items() if p["direction"] == "input"]
    outputs = [(n, p["bits"][0]) for n, p in m["ports"].items() if p["direction"] == "output"]
    if any(len(p["bits"]) != 1 for p in m["ports"].values()):
        raise ValueError("scalar input/output ports required")
    for n, _ in inputs + outputs:
        identifier(n)
    memo = {b: f"p{i}" for i, (_, b) in enumerate(inputs)}
    drivers = {}
    for c in m["cells"].values():
        kind = c["type"]
        if kind not in ("$_AND_", "$_NOT_"):
            raise ValueError(f"unsupported AIG cell: {kind}")
        bits = c["connections"]
        y = bits["Y"][0]
        if y in drivers:
            raise ValueError("multiple drivers")
        drivers[y] = (kind, bits["A"][0], bits.get("B", [None])[0])
    visiting = set()

    def expr(b):
        if b in ("0", "1"):
            return b
        if b in memo:
            return memo[b]
        if b in visiting or b not in drivers:
            raise ValueError("cyclic or undriven input")
        visiting.add(b)
        kind, a, c = drivers[b]
        text = f"(! {expr(a)})" if kind == "$_NOT_" else f"(* {expr(a)} {expr(c)})"
        visiting.remove(b)
        if len(text) > 1_000_000:
            raise ValueError("unfolded expression exceeds resource guard")
        memo[b] = text
        return text
    if dag:
        def name(b):
            if b in ("0", "1"): return b
            if b in memo: return memo[b]
            if b in drivers: return f"n{b}"
            raise ValueError("undriven compact AIG")
        parts = []
        for b, (kind, a, c) in sorted(drivers.items()):
            rhs = f"(! {name(a)})" if kind == "$_NOT_" else f"(* {name(a)} {name(c)})"
            parts.append(f"(let n{b} {rhs})")
        parts.extend(f"(let o{i} {name(b)})" for i, (_, b) in enumerate(outputs))
    else:
        parts = [f"(let o{i} {expr(b)})" for i, (_, b) in enumerate(outputs)]
    text = parts[0]
    for part in parts[1:]:
        text = f"(& {text} {part})"
    if len(text) > 1_000_000:
        raise ValueError("input expression exceeds resource guard")
    return text, inputs, outputs


def parse(text):
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    stack, roots = [], []
    for token in tokens:
        if token == "(": stack.append([])
        elif token == ")":
            if not stack: raise ValueError("unexpected close")
            value = stack.pop()
            (stack[-1] if stack else roots).append(value)
        else: (stack[-1] if stack else roots).append(token)
    if stack: raise ValueError("unclosed expression")
    if len(roots) != 1: raise ValueError("trailing or missing expression")
    return roots[0]


def verilog(text, inputs, outputs, dag=False):
    inputs = [(identifier(n), b) for n, b in inputs]
    outputs = [(identifier(n), b) for n, b in outputs]
    bindings = {}

    pending = [parse(text)]
    while pending:
        tree = pending.pop()
        if isinstance(tree, list) and len(tree) == 3 and tree[0] == "&":
            pending.extend([tree[2], tree[1]])
        elif isinstance(tree, list) and len(tree) == 3 and tree[0] == "let":
            name = tree[1]
            if not isinstance(name, str) or name in bindings:
                raise ValueError("invalid output binding")
            bindings[name] = tree[2]
        else:
            raise ValueError("candidate lost output bindings")
    expected = {f"o{i}" for i in range(len(outputs))}
    internals = set(bindings) - expected
    if not expected.issubset(bindings) or (not dag and internals) or any(
            not re.fullmatch(r"n[0-9]+", n) for n in internals):
        raise ValueError("candidate output interface changed")
    symbols = {f"p{i}": n for i, (n, _) in enumerate(inputs)}
    symbols.update({n: "__esyn_" + n for n in internals})
    symbols.update({f"o{i}": n for i, (n, _) in enumerate(outputs)})

    def deps(t):
        if isinstance(t, str): return {t} if t in bindings else set()
        return set().union(*(deps(x) for x in t[1:]))
    active, complete = set(), set()
    def visit(n):
        if n in active: raise ValueError("cyclic candidate assignments")
        if n in complete: return
        active.add(n)
        for b in deps(bindings[n]): visit(b)
        active.remove(n); complete.add(n)
    for n in bindings: visit(n)

    def expr(t):
        if isinstance(t, str):
            if t in ("0", "false"): return "1'b0"
            if t in ("1", "true"): return "1'b1"
            if t not in symbols: raise ValueError(f"unknown symbol {t}")
            return symbols[t]
        if len(t) == 2 and t[0] == "!": return f"(~{expr(t[1])})"
        if len(t) == 3 and t[0] in ("*", "+", "->"):
            a, b = expr(t[1]), expr(t[2])
            return f"((~{a}) | {b})" if t[0] == "->" else f"({a} {'&' if t[0]=='*' else '|'} {b})"
        raise ValueError("unsupported Boolean expression")
    ports = [n for n, _ in inputs + outputs]
    result = "module mapped(" + ",".join(ports) + ");\n"
    result += "input " + ",".join(n for n, _ in inputs) + ";\n"
    result += "output " + ",".join(n for n, _ in outputs) + ";\n"
    if internals: result += "wire " + ",".join(symbols[n] for n in sorted(internals)) + ";\n"
    for n in sorted(internals): result += f"assign {symbols[n]} = {expr(bindings[n])};\n"
    for i, (n, _) in enumerate(outputs): result += f"assign {n} = {expr(bindings[f'o{i}'])};\n"
    return result + "endmodule\n"


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    add_suite_arguments(ap)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--objective", choices=["area-delay", "delay-area"], default="area-delay")
    ap.add_argument("--native-seconds", type=int)
    ap.add_argument("--native-nodes", type=int)
    ap.add_argument("--native-iterations", type=int)
    ap.add_argument("--representation", choices=["unfolded", "dag"], default="unfolded")
    ap.add_argument("--prepared-report", type=Path)
    ap.add_argument("--k", type=int, choices=[3, 4], default=3)
    ap.add_argument("--replay-candidates", type=Path,
                    help="Recheck one retained native candidate directory without rerunning search")
    a = ap.parse_args()
    resolve_suite(a)
    if a.replay_candidates and len(a.benches) != 1: ap.error("replay requires one design")
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    src = a.source.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src, text=True).strip()
    if commit != COMMIT: ap.error("source commit differs")
    binary = src / "e-rewriter/target/release/e-rewriter"
    evaluator = root / "target/release/bench_rs_netlist_eval"
    env = dict(os.environ, EGM_YOSYS="/usr/local/bin/yosys", BENCH_RS_EMAP_ABC="/usr/bin/abc",
               OMP_NUM_THREADS="2", RAYON_NUM_THREADS="2")
    for key, value in [("ESYN_TIME", a.native_seconds), ("ESYN_NODES", a.native_nodes),
                       ("ESYN_ITER", a.native_iterations)]:
        env.pop(key, None)
        if value is not None:
            if value <= 0: ap.error("native budgets must be positive")
            env[key] = str(value)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {"commit": commit, "source": "https://github.com/Gy-Hu/E-Syn", "branch": "new",
              "objective": a.objective, "timeout_secs": a.timeout, "commands": [], "results": [],
              "representation": a.representation,
              "preprocess": "resyn2" if a.prepared_report else "none",
              "candidate_selection": "common physical metric lexicographic minimum over upstream ranked candidates",
              "binary_sha256": digest(binary), "harness_sha256": digest(Path(__file__)),
              "source_patch": subprocess.check_output(["git", "diff"], cwd=src, text=True),
              "model_sha256": {n: digest(src / "e-rewriter/src/model" / n) for n in ["xgb_delay.model", "xgb.model"]},
              "metric_model": "all physical instances; max(rise,fall) block pin delay; no load or slew",
              "scope": "frozen upstream model candidates plus ABC amap; restricted final library; not published-flow reproduction",
              "native_limits": {"seconds": a.native_seconds or 300, "nodes": a.native_nodes or 25000000,
                                "iterations": a.native_iterations or 10000000},
              "candidates": []}

    def run(argv, log, cwd=root):
        report["commands"].append(argv)
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
    lib, models = out / f"asap7_{a.k}in.genlib", out / "cells.v"
    run(["python3", str(root / "scripts/experiments/restrict_genlib.py"), "--in",
         str(root / "tools/mockturtle/experiments/cell_libraries/asap7.genlib"),
         "--out", str(lib), "--max-pins", str(a.k)], out / "library.log")
    run([str(evaluator), str(lib), "--models", str(models), str(out / "library.json")], out / "models.log")
    report["library_sha256"] = digest(lib)
    for bench in a.benches:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", bench): ap.error("invalid benchmark")
        golden = root / f"tools/mockturtle/experiments/benchmarks/{bench}.aig"
        work = out / bench; work.mkdir()
        label = "esyn_frozen_model_amap" + ("_bounded" if any(v is not None for v in
                                              [a.native_seconds, a.native_nodes, a.native_iterations]) else "")
        if a.representation == "dag": label += "_dag"
        row = {"bench": bench, "method": label, "input_sha256": digest(golden)}
        try:
            for n in ["out_dot", "graph_info", "test_data_beta_runner"]: (work / n).mkdir()
            (work / "graph_info.py").write_bytes((src / "graph_info.py").read_bytes())
            netjson = work / "input.json"
            subject = golden
            if a.prepared_report:
                subject = work / "subject.aig"
                row["preprocessing"] = reuse_subject(a.prepared_report, bench, golden, lib, subject)
            yosys(f"read_aiger -module_name mapped {quoted(subject)}; write_json {quoted(netjson)}", work / "import.log")
            text, inputs, outputs = input_expression(netjson, dag=a.representation == "dag")
            (work / "input.sexpr").write_text(text)
            if a.replay_candidates:
                row["replayed_native_directory"] = str(a.replay_candidates.resolve())
                for retained in a.replay_candidates.glob("output_from_egg*.txt"):
                    (work / "test_data_beta_runner" / retained.name).write_bytes(retained.read_bytes())
            else:
                _, row["native_wall_secs"] = run([str(binary), "input.sexpr", "unused.sexpr"], work / "native.log", work)
            successful, local_candidates = [], []
            for candidate in sorted((work / "test_data_beta_runner").glob("output_from_egg*.txt")):
                folder = work / candidate.stem; folder.mkdir()
                item = {"bench": bench, "candidate": candidate.name, "sexpr_sha256": digest(candidate)}
                try:
                    logic = folder / "logic.v"; logic.write_text(verilog(candidate.read_text(), inputs, outputs, dag=a.representation == "dag"))
                    aig = folder / "logic.aig"
                    yosys(f"read_verilog {quoted(logic)}; hierarchy -top mapped; proc; flatten; "
                          f"techmap; opt; aigmap; write_aiger -symbols {quoted(aig)}", folder / "logic.log")
                    raw, mapped, netjson = folder / "raw.v", folder / "mapped.v", folder / "mapped.json"
                    run([env["BENCH_RS_EMAP_ABC"], "-c", f"read_genlib {quoted(lib)}; read {quoted(aig)}; "
                         f"strash; amap; write_verilog {quoted(raw)}"], folder / "map.log")
                    yosys(f"read_verilog -lib {quoted(models)}; read_verilog {quoted(raw)}; "
                          f"hierarchy -check -auto-top; rename -top mapped; "
                          f"write_verilog -noattr -noexpr {quoted(mapped)}; write_json {quoted(netjson)}",
                          folder / "normalize.log")
                    text, _ = run([str(evaluator), str(lib), str(golden), str(mapped), str(netjson)], folder / "evaluate.log")
                    measured = record(text)
                    if measured["cec"] != 1: raise RuntimeError("independent CEC did not pass")
                    item.update(status="ok", measured=measured); successful.append(item)
                except (RuntimeError, ValueError, RecursionError) as e: item.update(status="failed", error=str(e))
                report["candidates"].append(item)
                local_candidates.append(item)
                (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            if not successful: raise RuntimeError("no valid equivalent candidates")
            first, second = a.objective.split("-")
            best = min(successful, key=lambda x: (x["measured"][first], x["measured"][second]))
            row.update(status="ok", measured=best["measured"], selected_candidate=best["candidate"],
                       valid_candidates=len(successful), failed_candidates=sum(x["status"] != "ok" for x in local_candidates))
        except (RuntimeError, ValueError, RecursionError) as e: row.update(status="failed", error=str(e))
        report["results"].append(row)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(bench, row, flush=True)
    if any(r["status"] != "ok" for r in report["results"]): raise SystemExit(1)


if __name__ == "__main__": main()
