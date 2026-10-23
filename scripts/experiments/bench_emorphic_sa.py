#!/usr/bin/env python3
"""E-morphic final native SA method, fixed common cell inventory, independent CEC.

Uses the documented paper branch's SA/ABC sizing flow after fixed resyn2.
Internal Liberty timing and common genlib block-delay metrics are kept distinct.
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
import sys
import time
from bench_egraph_baselines import quoted, record
from shared_subject import reuse_subject
from port_identifiers import identifier
from bench_emorphic_baseline import COMMIT, PROTO_COMMIT, input_eqn


def restrict_liberty(text, names):
    """Keep complete named cell blocks and all original global settings/templates."""
    parts, last, seen = [], 0, set()
    for match in re.finditer(r"\bcell\s*\(\s*([^\s)]+)\s*\)\s*\{", text):
        i, depth, in_string, escaped = match.end(), 1, False, False
        while depth and i < len(text):
            c = text[i]
            if escaped: escaped = False
            elif c == "\\" and in_string: escaped = True
            elif c == '"': in_string = not in_string
            elif not in_string: depth += (c == "{") - (c == "}")
            i += 1
        if depth: raise ValueError("unclosed Liberty cell")
        if match.start() < last: raise ValueError("nested Liberty cell")
        parts.append(text[last:match.start()])
        if match.group(1) in names:
            if match.group(1) in seen: raise ValueError("duplicate Liberty cell")
            seen.add(match.group(1)); parts.append(text[match.start():i])
        last = i
    parts.append(text[last:])
    missing = names - seen - {"_const0_", "_const1_"}
    if missing: raise ValueError("missing physical cells: " + str(sorted(missing)))
    return "".join(parts), sorted(seen)


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    add_suite_arguments(ap)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--prepared-report", type=Path,
                    help="Reuse exact CEC-checked resyn2 AIGs from a common baseline report")
    ap.add_argument("--k", type=int, choices=[3, 4], default=3)
    a = ap.parse_args()
    resolve_suite(a)
    if a.timeout <= 0: ap.error("invalid timeout")
    src, out = a.source.resolve(), a.output.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src, text=True).strip()
    if commit != COMMIT: ap.error("source commit differs")
    out.mkdir(parents=True, exist_ok=False)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    rewrite, extract = [src / n for n in ["e-rewriter/target/release/e-rewriter",
                                         "extraction-gym/target/release/extraction-gym"]]
    evaluator = root / "target/release/bench_rs_netlist_eval"
    abc, yosys = "/usr/bin/abc", "/usr/local/bin/yosys"
    env = dict(os.environ, EGM_YOSYS=yosys, BENCH_RS_EMAP_ABC=abc, RAYON_NUM_THREADS="2", OMP_NUM_THREADS="2")
    report = {"source": "https://github.com/Gy-Hu/E-Syn2", "commit": commit,
        "branch": "SA-golden-model-delaytuning", "method": "emorphic_final_sa",
        "scope": "final native SA + fixed resyn2 preprocessing + ABC sizing; restricted physical cell inventory",
        "settings": {"rewrite_iterations": 5, "rewrite_seconds": 10, "rewrite_nodes": 200000000,
                     "sa_rounds": 6, "samples_per_round": 10, "random_prob": 0.8,
                     "cost_function": "node_depth_cost", "preprocess": "resyn2",
                     "native_feedback": "Liberty ABC stime delay",
                     "external_process_timeout": a.timeout, "rng_seed": "upstream unseeded thread_rng"},
        "metric_model": "all physical instances; max(rise,fall) block pin delay; no load or slew",
        "binary_sha256": {"rewriter": sha(rewrite), "extractor": sha(extract), "abc": sha(Path(abc)),
                          "evaluator": sha(evaluator)},
        "harness_sha256": sha(Path(__file__)),
        "capture_sha256": sha(root / "scripts/experiments/capture_emorphic_abc.py"),
        "proto_source_commit": PROTO_COMMIT, "proto_sha256": sha(src / "grpc_communicator/proto/service.proto"),
        "source_patch": subprocess.check_output(["git", "diff"], cwd=src, text=True),
        "commands": [], "results": [], "candidates": []}
    def save(): (out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    def run(argv, log, cwd=root, local_env=env):
        report["commands"].append({"argv": argv, "cwd": str(cwd)}); save()
        start = time.perf_counter()
        with subprocess.Popen(argv, cwd=cwd, env=local_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, start_new_session=True) as p:
            try: stdout, stderr = p.communicate(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); stdout, stderr = p.communicate()
                log.write_text(stdout+stderr+"\nTIMEOUT\n"); raise RuntimeError("process-group timeout")
            log.write_text(stdout+stderr)
            if p.returncode: raise RuntimeError(f"exit {p.returncode}: {log.name}")
        return stdout, time.perf_counter()-start
    def yrun(script, log): return run([yosys, "-Q", "-T", "-p", script], log)
    lib, models = out / f"asap7_{a.k}in.genlib", out / "cells.v"
    run([sys.executable, str(root / "scripts/experiments/restrict_genlib.py"), "--in",
         str(root / "tools/mockturtle/experiments/cell_libraries/asap7.genlib"), "--out", str(lib),
         "--max-pins", str(a.k)], out / "library.log")
    run([str(evaluator), str(lib), "--models", str(models), str(out / "library.json")], out / "models.log")
    report["library_sha256"] = sha(lib)
    shared = None
    if a.prepared_report:
        a.prepared_report = a.prepared_report.resolve()
        shared = json.loads(a.prepared_report.read_text())
        if shared.get("preprocess") != "resyn2" or (shared.get("kind") != "logic_subjects" and shared.get("library_sha256") != sha(lib)):
            raise ValueError("incompatible prepared report")
        report["prepared_report"] = str(a.prepared_report)
        report["prepared_report_sha256"] = sha(a.prepared_report)
    names = set(re.findall(r"^GATE\s+(\S+)", lib.read_text(), re.M))
    native_lib = out / "native_same_cells.lib"
    text, physical_cells = restrict_liberty((src / "abc/asap7_clean.lib").read_text(), names)
    native_lib.write_text(text)
    report.update(native_liberty_sha256=sha(native_lib), native_liberty_source_sha256=sha(src / "abc/asap7_clean.lib"),
                  native_physical_cells=physical_cells,
                  timing_note="same named cell inventory; upstream Liberty internal score differs from common unloaded block metric")
    for bench in a.benches:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", bench): ap.error("invalid benchmark")
        golden = root / f"tools/mockturtle/experiments/benchmarks/{bench}.aig"
        work = out / bench; work.mkdir()
        row = {"bench": bench, "method": "emorphic_final_sa", "input_sha256": sha(golden)}
        try:
            rw, ex = work / "e-rewriter", work / "extraction-gym"
            for folder in [rw / "rewritten_circuit", rw / "random_graph", ex / "input", work / "abc",
                           work / "trace", work / "bin"]: folder.mkdir(parents=True)
            (work / "abc/asap7_clean.lib").write_bytes(native_lib.read_bytes())
            recipe = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance"
            subject = work / "subject.resyn2.aig"
            if a.prepared_report:
                row["preprocessing"] = reuse_subject(a.prepared_report, bench, golden, lib, subject)
            else:
                _, prep_secs = run([abc, "-c", f"read {quoted(golden)}; strash; {recipe}; "
                                   f"write_aiger -s {quoted(subject)}"], work / "prepare.log")
                cec_text, cec_secs = run([abc, "-c", f"cec {quoted(golden)} {quoted(subject)}"], work / "prepare-cec.log")
                if "Networks are equivalent" not in cec_text: raise RuntimeError("prepared subject CEC failed")
                row["preprocessing"] = {"recipe": recipe, "sha256": sha(subject), "cec": 1,
                                        "wall_secs": prep_secs, "cec_wall_secs": cec_secs}
            netjson = work / "original.json"
            yrun(f"read_aiger -module_name mapped {quoted(subject)}; write_json {quoted(netjson)}", work / "import.log")
            text, inputs, outputs = input_eqn(netjson)
            initial, prepared = rw / "input.eqn", rw / "circuit0_opt.eqn"
            initial.write_text(text)
            # The subject already received the one fixed common preparation.
            prepared.write_text(text)
            row["prepared_eqn_sha256"] = sha(prepared)
            _, row["rewrite_secs"] = run([str(rewrite), "circuit0_opt.eqn", "5"], work / "rewrite.log", rw)
            graphfile = ex / "input/rewritten_egraph_with_weight_cost_serd.json"
            graphfile.write_bytes((rw / "rewritten_circuit/rewritten_egraph_with_weight_cost_serd.json").read_bytes())
            row["graph_sha256"] = sha(graphfile)
            launcher = work / "bin/abc"
            launcher.write_text("#!" + sys.executable + "\n" +
                (root / "scripts/experiments/capture_emorphic_abc.py").read_text())
            launcher.chmod(0o755)
            native_env = dict(env, PATH=str(work / "bin")+os.pathsep+env["PATH"],
                              EMORPHIC_TRACE=str(work / "trace"), EMORPHIC_REAL_ABC=abc)
            _, row["sa_secs"] = run([str(extract), "input/rewritten_egraph_with_weight_cost_serd.json",
                "--extractor=random-sim-ann-based-faster-bottom-up-fast-par", "--cost-function=node_depth_cost",
                "--num-samples=10", "--random-prob=0.8", "--out=extract.log.json"], work / "sa.log", ex, native_env)
            retained = list(ex.glob("tmp_*.v"))
            if len(retained) != 1: raise RuntimeError("native SA did not retain exactly one final netlist")
            final_name = retained[0].stem
            log = (work / "sa.log").read_text()
            if not all(f"iter {i}/6" in log for i in range(1, 7)): raise RuntimeError("SA rounds incomplete")
            successful, local = [], []
            for folder in sorted((work / "trace").iterdir()):
                item = {"bench": bench, "candidate": folder.name, "selected_by_native_sa": folder.name == final_name}
                try:
                    raw, mapped, netjson = folder / "raw.v", folder / "mapped.v", folder / "mapped.json"
                    item["raw_sha256"] = sha(raw)
                    renames = "; ".join(f"rename {'p' if kind == 'input' else 'o'}{i} {identifier(name)}"
                        for kind, ports in [("input", inputs), ("output", outputs)]
                        for i, (name, _) in enumerate(ports))
                    yrun(f"read_verilog -lib {quoted(models)}; read_verilog {quoted(raw)}; hierarchy -check -auto-top; "
                         f"rename -top mapped; cd mapped; {renames}; cd ..; write_verilog -noattr -noexpr {quoted(mapped)}; "
                         f"write_json {quoted(netjson)}", folder / "normalize.log")
                    text, _ = run([str(evaluator), str(lib), str(golden), str(mapped), str(netjson)], folder / "evaluate.log")
                    measured = record(text)
                    if measured["cec"] != 1: raise RuntimeError("independent CEC failed")
                    item.update(status="ok", measured=measured); successful.append(item)
                except (RuntimeError, ValueError, KeyError, OSError) as e: item.update(status="failed", error=str(e))
                local.append(item); report["candidates"].append(item); save()
            if len(local) != 70: raise RuntimeError("expected 10 initial + 6*10 SA candidates")
            selected = [x for x in successful if x["selected_by_native_sa"]]
            if len(selected) != 1: raise RuntimeError("native selected netlist failed independent verification")
            row.update(status="ok", measured=selected[0]["measured"], selected_candidate=final_name,
                       valid_candidates=len(successful), failed_candidates=len(local)-len(successful))
        except (RuntimeError, ValueError, KeyError, OSError) as e: row.update(status="failed", error=str(e))
        report["results"].append(row); save(); print(bench, row, flush=True)
    if any(r["status"] != "ok" for r in report["results"]): raise SystemExit(1)


if __name__ == "__main__": main()
