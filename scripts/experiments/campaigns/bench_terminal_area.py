#!/usr/bin/env python3
"""Compare stock standard-cell mappers with common physical metrics and named-port CEC."""
import argparse
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from benchmark_suites import add_suite_arguments, resolve_suite
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import time
from shared_subject import reuse_subject
from port_identifiers import escape_indexed_scalars

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
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ['single-structure-cuts','no-input-phase-matches','canonical-pin-matching','no-actual-area-tail']:
        ap.add_argument('--'+flag,type=int,choices=[0,1],default=0)
    ap.add_argument('--dc-support',type=int,default=4)
    ap.add_argument("--refresh-area-timing",type=int,choices=[0,1],default=0)
    ap.add_argument("--terminal-area",type=int,choices=[0,1],default=0)
    ap.add_argument("--tail-shared-area",type=int,choices=[0,1],default=0)
    ap.add_argument("--cut-pin-flow",type=int,choices=[0,1],default=0)
    ap.add_argument("--delta-area-filter",type=int,choices=[0,1],default=0)
    ap.add_argument("--online-shared-bound",type=int,choices=[0,1],default=0)
    ap.add_argument("--preserve-shared-trajectory",type=int,choices=[0,1],default=0)
    ap.add_argument("--canonical-source-inputs",type=int,choices=[0,1],default=0)
    ap.add_argument("--introduce-phase-inverters",type=int,choices=[0,1],default=0)
    ap.add_argument("--import-helper-fill",type=int,choices=[0,1],default=0)
    ap.add_argument("--shared-area-bound",type=int,choices=[0,1],default=0)
    ap.add_argument("--fused-cover-scoring",type=int,choices=[0,1],default=0)
    ap.add_argument("--online-area",type=int,choices=[0,1],default=0)
    ap.add_argument("--activate-deferred",type=int,choices=[0,1],default=0)
    ap.add_argument("--timing-witness-pool",type=int,choices=[0,1],default=0)
    ap.add_argument("--joint-cones",type=int,choices=[0,1],default=0)
    ap.add_argument("--scored-incumbent",type=int,choices=[0,1],default=0)
    ap.add_argument("--cone-import",type=int,choices=[0,1],default=0)
    ap.add_argument("--metric-epochs",type=int,choices=[0,1],default=0)
    ap.add_argument("--preserve-legacy-seeds",type=int,choices=[0,1],default=0)
    ap.add_argument("--defer-area-dag",type=int,choices=[0,1],default=0)
    ap.add_argument("--required-area-dag",type=int,choices=[0,1],default=0)
    ap.add_argument("--fast-delay-seed",type=int,choices=[0,1],default=0)
    ap.add_argument("--critical-recovery", type=int, choices=[0,1], default=0)
    ap.add_argument("--bounded-proposals", type=int, choices=[0,1], default=0)
    ap.add_argument("--seed-window-fraction", type=float, default=1.0)
    ap.add_argument("--mapper-root", type=Path, required=True)
    ap.add_argument("--small-cut-quota",type=int,choices=[0,8],default=0)
    ap.add_argument("--literal-auto",type=int,choices=[0,1],default=0)
    ap.add_argument("--fast-area",type=int,choices=[0,1],default=0)
    ap.add_argument("--area-time-fraction",type=float,default=0.0)
    ap.add_argument("--area-cut-quota",type=int,choices=[0,4],default=0)
    ap.add_argument("--cap-diversity",type=int,choices=[0,1],default=0)
    ap.add_argument("--canonical-phase",type=int,choices=[0,1],default=0)
    ap.add_argument("--canonical-wires",type=int,choices=[0,1],default=0)
    ap.add_argument("--structural-refs",type=int,choices=[0,1],default=0)
    ap.add_argument("--slack-flow",type=int,choices=[0,1],default=0)
    ap.add_argument("--region-classes", type=int, default=256)
    ap.add_argument("--library-timing", type=int, choices=[0,1], default=0)
    ap.add_argument("--timing-cap", type=int, choices=[0,1], default=0)
    ap.add_argument("--literal-import",type=int,choices=[0,1],default=0)
    ap.add_argument("--early-slack-flow",type=int,choices=[0,1],default=0)
    ap.add_argument("--return-reserve",type=float,default=0.05)
    ap.add_argument("--coupled-event-flow", type=int, choices=[0,1], default=0)
    ap.add_argument("--staged-event-flow", type=int, choices=[0,1], default=0)
    ap.add_argument("--staged-critical-order", type=int, choices=[0,1], default=0)
    ap.add_argument("--staged-recovery", type=int, choices=[0,1], default=0)
    ap.add_argument("--phase-time-fraction", type=float, default=1.0)
    ap.add_argument("--phase-recovery", type=int, choices=[0,1], default=1)
    ap.add_argument("--output", type=Path, required=True, help="new result directory")
    ap.add_argument("--k", type=int, choices=range(2, 7), default=3)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--cuts", type=int, default=16)
    ap.add_argument("--refine-seconds", type=float, default=2)
    ap.add_argument("--solver-seconds", type=float, default=0)
    ap.add_argument("--max-delay", type=float)
    ap.add_argument("--objective", choices=["weighted", "area-delay", "delay-area"], default="weighted")
    ap.add_argument("--preprocess", choices=["none", "resyn2"], default="resyn2",
                    help="Fixed shared ABC preparation (default resyn2); none reproduces historical raw-input controls")
    ap.add_argument("--prepared-report", type=Path)
    ap.add_argument("--methods", nargs="+", default=["abc_map_area", "abc_map_delay",
                                                   "mockturtle_emap_area", "mockturtle_emap_delay",
                                                   "egraph", "egraph_no_sdc"],
                    help="Primary mappers; supplemental controls remain available explicitly")
    add_suite_arguments(ap)
    a = ap.parse_args()
    resolve_suite(a)
    if a.preprocess == "resyn2" and "abc_resyn2_nf" in a.methods:
        ap.error("shared resyn2 must run once; use abc_nf_delay instead of abc_resyn2_nf")
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evaluator = root / "target/release/bench_rs_netlist_eval"
    mapper_root = a.mapper_root.resolve()
    mapper = mapper_root / "target/release/bench_rs_egraph_map"
    if a.methods != ["egraph_bounded"] or a.region_classes < 0:
        ap.error("regional harness only supports strict egraph mapper")
    cpp = root / "build/baseline_mapper"
    abc = shutil.which(os.environ.get("ABC", "abc"))
    yosys = shutil.which(os.environ.get("EGM_YOSYS", "yosys"))
    if not abc or not yosys:
        ap.error("ABC and Yosys are required")
    env = {k: v for k, v in os.environ.items() if not k.startswith("EGM_")}
    env["EGM_DELTA_AREA_FILTER"]=str(a.delta_area_filter)
    env["EGM_ONLINE_SHARED_BOUND"]=str(a.online_shared_bound)
    env["EGM_CONE_IMPORT"]=str(a.cone_import)
    env["EGM_SCORED_INCUMBENT"]=str(a.scored_incumbent)
    env["EGM_JOINT_CONES"]=str(a.joint_cones)
    env["EGM_TIMING_WITNESS_POOL"]=str(a.timing_witness_pool)
    env["EGM_ONLINE_AREA"]=str(a.online_area)
    env["EGM_FUSED_COVER_SCORING"]=str(a.fused_cover_scoring)
    env["EGM_SHARED_AREA_BOUND"]=str(a.shared_area_bound)
    env["EGM_INTRODUCE_PHASE_INVERTERS"]=str(a.introduce_phase_inverters)
    env["EGM_IMPORT_HELPER_FILL"]=str(a.import_helper_fill)
    env["EGM_CANONICAL_SOURCE_INPUTS"]=str(a.canonical_source_inputs)
    env["EGM_PRESERVE_SHARED_TRAJECTORY"]=str(a.preserve_shared_trajectory)
    env["EGM_ACTIVATE_DEFERRED"]=str(a.activate_deferred)
    env["EGM_POOL_FRONTIER_AUDIT"]="1"
    if not 0.0 <= a.seed_window_fraction <= 1.0: ap.error("invalid seed window fraction")
    env["EGM_METRIC_EPOCHS"]=str(a.metric_epochs)
    env["EGM_REFRESH_AREA_TIMING"]=str(a.refresh_area_timing)
    env["EGM_TAIL_SHARED_AREA"]=str(a.tail_shared_area)
    env["EGM_TERMINAL_AREA"]=str(a.terminal_area)
    env["EGM_CUT_PIN_FLOW"]=str(a.cut_pin_flow)
    env["EGM_PRESERVE_LEGACY_SEEDS"]=str(a.preserve_legacy_seeds)
    env["EGM_DEFER_AREA_DAG"]=str(a.defer_area_dag)
    env["EGM_REQUIRED_AREA_DAG"]=str(a.required_area_dag)
    env["EGM_FAST_DELAY_SEED"]=str(a.fast_delay_seed)
    env.update(EGM_CRITICAL_RECOVERY=str(a.critical_recovery),EGM_BOUNDED_PROPOSALS=str(a.bounded_proposals),EGM_SEED_WINDOW_FRACTION=str(a.seed_window_fraction))
    if a.cuts < 1 or a.refine_seconds < 0 or a.solver_seconds < 0:
        ap.error("cuts must be positive and refinement time nonnegative")
    for key in ['single_structure_cuts','no_input_phase_matches','canonical_pin_matching','no_actual_area_tail']:
        env['EGM_'+key.upper()]=str(getattr(a,key))
    env.update(EGM_K=str(a.k), EGM_CUTS=str(a.cuts), EGM_TIME=str(a.solver_seconds), EGM_DC_SUPPORT=str(a.dc_support),
               EGM_REFINE_TIME=str(a.refine_seconds), EGM_REFINE_ROUNDS="4", EGM_REFINE_TRIALS="2000",
               EGM_BOOLEAN="0", EGM_LIVE_REFS="1", EGM_DELAY_WEIGHT="0",
               EGM_FLOW_DELAY_TIES="0",
               EGM_OBJECTIVE=a.objective,
               EGM_STRUCTURAL_REFS="0",
               EGM_CONE_RECOVERY="1",
               EGM_MULTIOUTPUT="0", EGM_ITER="16", EGM_NODES="20000",
               EGM_MULTI_CUMULATIVE="0",
               EGM_CUT_ROUNDS="32", EGM_DC_CONE="1024", EGM_INSTANCES="100000",
               EGM_YOSYS=yosys, BENCH_RS_EMAP_ABC=abc)
    env["EGM_SMALL_CUT_QUOTA"] = str(a.small_cut_quota)
    env["EGM_AREA_CUT_QUOTA"] = str(a.area_cut_quota)
    env["EGM_CAP_DIVERSITY"] = str(a.cap_diversity)
    env["EGM_CANONICAL_PHASE"] = str(a.canonical_phase)
    env["EGM_LITERAL_AUTO"] = str(a.literal_auto)
    env["EGM_FAST_AREA"] = str(a.fast_area)
    env["EGM_AREA_TIME_FRACTION"] = str(a.area_time_fraction)
    env["EGM_CANONICAL_WIRES"] = str(a.canonical_wires)
    env["EGM_STRUCTURAL_REFS"] = str(a.structural_refs)
    env["EGM_SLACK_FLOW"] = str(a.slack_flow)
    env.update(EGM_EARLY_SLACK_FLOW=str(a.early_slack_flow), EGM_RETURN_RESERVE=str(a.return_reserve), EGM_LITERAL_IMPORT=str(a.literal_import), EGM_FLOW_WORKLIST=str(a.coupled_event_flow), EGM_REGION_CLASSES=str(a.region_classes), EGM_PHASE_RECOVERY=str(a.phase_recovery), EGM_PHASE_TIME_FRACTION=str(a.phase_time_fraction), EGM_STAGED_RECOVERY=str(a.staged_recovery), EGM_STAGED_EVENT_FLOW=str(a.staged_event_flow), EGM_STAGED_CRITICAL_ORDER=str(a.staged_critical_order),
               EGM_LIBRARY_TIMING=str(a.library_timing), EGM_TIMING_CAP=str(a.timing_cap))
    if a.max_delay is not None:
        if a.max_delay < 0:
            ap.error("maximum delay must be nonnegative")
        env["EGM_MAX_DELAY"] = str(a.max_delay)

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
         str((root / "tools/mockturtle/experiments/cell_libraries/asap7.genlib").resolve()),
         "--out", str(library), "--max-pins", str(a.k)], out / "library.log")
    models = out / "cells.v"
    run([str(evaluator), str(library), "--models", str(models), str(out / "library.json")],
        out / "models.log")
    manifest = {"library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "library": library.name, "settings": {k: v for k, v in env.items() if k.startswith("EGM_")},
                "sources": SOURCES, "timeout_secs": a.timeout, "commands": [], "results": [],
                "metric_model": "all physical instances; max(rise,fall) block pin delay; no load or slew",
                "preprocess": a.preprocess}
    for label, command in [("abc", [abc, "-c", "version"]), ("yosys", [yosys, "-V"])]:
        manifest[label + "_version"], _ = run(command, out / f"{label}-version.log")
    manifest["binary_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(abc), Path(yosys), evaluator, mapper, cpp] if path.is_file()
    }
    manifest["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), mapper_root / "emapper/src/egraph_map.rs", mapper_root / "emapper/src/egraph_sat.rs", mapper_root / "emapper/src/egraph_regions.rs",
                     root / "scripts/experiments/port_identifiers.py", mapper_root / "src/bin/bench_rs_egraph_map.rs",
                     root / "emapper/src/netlist_metrics.rs", root / "emapper/baseline_mapper.cpp",
                     root / "tools/mockturtle/include/mockturtle/algorithms/mapper.hpp"]
    }
    for bench in a.benches:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", bench):
            ap.error("invalid benchmark name")
        golden = root / f"tools/mockturtle/experiments/benchmarks/{bench}.aig"
        subject, preprocessing = golden, {}
        prep_error = None
        if a.preprocess == "resyn2":
            subject = out / f"{bench}.resyn2.aig"
            recipe = ("balance; rewrite; refactor; balance; rewrite; rewrite -z; "
                      "balance; refactor -z; rewrite -z; balance")
            try:
                if a.prepared_report:
                    preprocessing = reuse_subject(a.prepared_report, bench, golden, library, subject)
                else:
                    command = [abc, "-c", f"read {quoted(golden)}; strash; {recipe}; "
                               f"write_aiger -s {quoted(subject)}"]
                    manifest["commands"].append({"bench": bench, "stage": "preprocess", "argv": command})
                    _, preprocessing["wall_secs"] = run(command, out / f"{bench}.preprocess.log")
                if not subject.is_file():
                    raise RuntimeError("preprocessor produced no AIG")
                command = [abc, "-c", f"cec {quoted(golden)} {quoted(subject)}"]
                manifest["commands"].append({"bench": bench, "stage": "preprocess-cec", "argv": command})
                text, preprocessing["cec_wall_secs"] = run(command, out / f"{bench}.preprocess-cec.log")
                if "Networks are equivalent" not in text:
                    raise RuntimeError("preprocessing equivalence unknown or failed")
                preprocessing.update(cec=1, sha256=hashlib.sha256(subject.read_bytes()).hexdigest(),
                                     recipe=recipe)
            except (RuntimeError, ValueError) as error:
                prep_error = str(error)
        for method in a.methods:
            label = ("egraph_staged" if a.staged_recovery else "egraph_staged_control") + "_resyn2"
            folder = out / f"{bench}.{label}"
            folder.mkdir()
            mapped = folder / "raw.v"
            row = {"bench": bench, "method": label, "preprocessing": preprocessing,
                   "input_sha256": hashlib.sha256(golden.read_bytes()).hexdigest()}
            try:
                if prep_error:
                    raise RuntimeError(prep_error)
                if method in RECIPES:
                    recipe = RECIPES[method].format(k=a.k)
                    script = (f"read_genlib {quoted(library)}; read {quoted(subject)}; strash; "
                              f"{recipe}; print_stats; write_verilog {quoted(mapped)}")
                    command = [abc, "-c", script]
                elif method.startswith(("mockturtle_map_", "mockturtle_emap_")):
                    command = [str(cpp), str(library), str(subject), str(mapped), method.rsplit('_', 1)[1],
                               "emap" if method.startswith("mockturtle_emap_") else "map"]
                elif method in ("egraph", "egraph_no_sdc", "egraph_factor", "egraph_associate",
                                "egraph_no_live_refs", "egraph_timing", "egraph_previous", "egraph_delay_ties",
                                "egraph_structural_refs", "egraph_worklist", "egraph_flow_compare", "egraph_critical", "egraph_cut_depth", "egraph_cut_mix4", "egraph_cut_mix8", "egraph_library_timing", "egraph_library_slack", "egraph_timing_cap", "egraph_bounded"):
                    command = [str(mapper), str(library), str(subject), str(mapped)]
                else:
                    raise RuntimeError("unknown method")
                manifest["commands"].append({"bench": bench, "method": method, "argv": command})
                local_env = dict(env, EGM_NOCEC="1")
                if method == "egraph_bounded":
                    local_env["EGM_STRICT_BUDGETS"] = "1"
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
                if method == "egraph_delay_ties":
                    local_env["EGM_FLOW_DELAY_TIES"] = "1"
                if method == "egraph_worklist":
                    local_env["EGM_FLOW_WORKLIST"] = "1"
                if method in ("egraph_library_timing", "egraph_library_slack", "egraph_timing_cap"):
                    local_env["EGM_LIBRARY_TIMING"] = "1"
                    local_env["EGM_DIAG_PATH"] = str(folder / "candidate-diagnostics.json")
                if method in ("egraph_library_slack", "egraph_timing_cap"):
                    local_env["EGM_SLACK_FLOW"] = "1"
                if method == "egraph_timing_cap":
                    local_env["EGM_TIMING_CAP"] = "1"
                if method in ("egraph_cut_mix4", "egraph_cut_mix8"):
                    local_env["EGM_CUT_DEPTH_QUOTA"] = method[-1]
                if method == "egraph_cut_depth":
                    local_env["EGM_CUT_DEPTH"] = "1"
                if method == "egraph_critical":
                    local_env["EGM_CRITICAL_RECOVERY"] = "1"
                if method == "egraph_flow_compare":
                    local_env["EGM_FLOW_COMPARE"] = "1"
                if method == "egraph_structural_refs":
                    local_env["EGM_STRUCTURAL_REFS"] = "1"
                local_env["EGM_RECOVERY_TRACE"] = str(folder / "recovery-trace.json")
                local_env["EGM_POOL_DIAG_PATH"] = str(folder / "pool-diagnostics.json")
                local_env["EGM_DIAG_PATH"] = str(folder / "candidate-diagnostics.json")
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
                if method.startswith("mockturtle"):
                    escaped = folder / "escaped.v"
                    escaped.write_text(escape_indexed_scalars(mapped.read_text()))
                    row["port_adapter"] = {"kind": "escape indexed scalar names", "raw_sha256": hashlib.sha256(mapped.read_bytes()).hexdigest(),
                                           "escaped_sha256": hashlib.sha256(escaped.read_bytes()).hexdigest()}
                    mapped = escaped
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
