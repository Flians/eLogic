"""Audit rho-sweep provenance, whole-case clocks, independent proofs and budgets."""
import argparse
import collections
import json
from pathlib import Path

from input_capacity_proportional import input_capacity
from scaled_verification_cache import lookup, sha

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text())
    assert all(sha(p) == h for p, h in campaign["fingerprint"].items())
    rows = []
    for job in campaign["jobs"]:
        assert all(sha(p) == h for p, h in job["sha256"].items())
        document = json.loads(Path(job["report"]).read_text())
        row = document["results"][0]
        audit = json.loads(Path(job["audit"]).read_text())
        prepared = row["preprocessing"]
        assert prepared["cec"] == 1 and sha(prepared["reused_from"]) == prepared["sha256"]
        subject_report = ROOT / "results_v2/epfl_resyn2_subjects_20261012/report.json"
        assert prepared["source_report_sha256"] == sha(subject_report) == campaign["fingerprint"][str(subject_report)]
        assert audit["mapping_seconds"] == 300 and not audit["native_parameters_changed"]
        assert audit["driver_sha256"] == campaign["fingerprint"][audit["driver"]]
        commands = [c for s in audit["states"].values() for c in s["commands"]]
        mapping = [c for c in commands if c["mapping_charged"]]
        assert mapping and Path(mapping[0]["argv"][0]).name == "egraphmap_scaled.py"
        if len(mapping)>1:
            assert row["status"]!="ok" and all(c["argv"]==mapping[0]["argv"]for c in mapping)
        if row["status"]=="ok":
            assert len(mapping)==1
            assert all(not s["exhausted"] and s["mapping_wall_secs"] <= 300 for s in audit["states"].values())
            assert mapping[0]["wall_secs"] <= 300
        library = Path(job["report"]).parent / document["library"]
        assert sha(library) == document["library_sha256"]
        baseline = json.loads((ROOT / f'results_v2/guarded_joint_evidence_20261016/baseline_scope/baselines_{job["k"]}in.json').read_text())
        assert document["library_sha256"] == baseline["library_sha256"] and document["metric_model"] == baseline["metric_model"]
        folder = Path(job["report"]).parent / (row["bench"] + "." + row["method"])
        raw = folder / "raw.v"
        capacity = input_capacity(prepared["reused_from"], job["ratio"])
        if "scalability" not in row:
            assert row["status"]!="ok"
            rows.append(dict(native_failure=row.get("error"),native_last_stage=None,name=job["name"],bench=job["bench"],k=job["k"],ratio=job["ratio"],repetition=job["repetition"],status=row["status"],error=row.get("error"),selected="unreturned",optimized_selected_and_CEC=False,optimizer_returned_internal_cover=False,verification_mode="failed-before-auditable-completion",mapping_secs=sum(c["wall_secs"]for c in mapping),admitted=None,allowance=capacity["node_limit"],area=None,delay=None,source_area=None,source_delay=None,native_exit=None,native_timeout=any(s["exhausted"]for s in audit["states"].values()),report=job["report"],audit=job["audit"]))
            continue
        native = row["scalability"]
        assert native["global_cumulative_node_allowance"] == capacity["node_limit"]
        assert row["bench"]==job["bench"] and native["search_ratio"]==capacity["search_ratio"]
        assert native["primitive_input_node_bound"] == capacity["base_enode_upper"]
        assert native["proportional_search_allowance"] == capacity["search_allowance"]
        assert native["native_optimizer_sha256"] == campaign["fingerprint"][native["native_optimizer"]]
        assert native["optimizer_recovery_secs"] == 2 and native["optimizer_pool_limit"] == 100000
        assert native["optimizer_shared_window_secs"] == 60 and native["total_mapping_frontend_secs"] <= 300
        assert native["source_plus_optimizer_retained_upper"] <= native["global_retained_candidate_limit"] == capacity["base_enode_upper"] + 100000
        assert native["no_refund_on_retry"] and len(native["attempts"]) == 1
        attempt = native["attempts"][0]
        assert attempt["node_allowance"] == capacity["node_limit"]
        assert attempt["charged_admission_upper"] == native["cumulative_accounted_enode_upper"] <= capacity["node_limit"]
        if attempt["admission"] is not None:
            assert attempt["admission"]["cumulative_admitted_enodes"] == attempt["charged_admission_upper"]
            assert not attempt["admission"]["refund_on_rebuild"]
        else:
            assert attempt["charged_admission_upper"] == capacity["node_limit"]
        assert sha(raw) == native["selected_verilog_sha256"] and sha(native["source_verilog"]) == native["source_sha256"]
        if native["optimizer_succeeded"]:
            actual=native["optimizer_native"]
            assert actual["heuristic_secs"]<=2 and actual["instances"]<=100000
            assert actual["enodes"]<=actual["cumulative_admitted_enodes"]<=capacity["node_limit"]
            assert actual["max_cuts_per_class"]<=16
            recovery=Path(attempt["trace_path"])
            if recovery.exists():
                totals=[x for x in json.loads(recovery.read_text())["events"]if x["stage"]=="global_total"]
                assert totals and all(x["seconds"]<=2 for x in totals)

        original = mapping[0]["argv"][2]
        assert sha(original) == prepared["sha256"]
        original_path = ROOT / f'tools/mockturtle/experiments/benchmarks/{row["bench"]}.aig'
        assert sha(original_path) == row["input_sha256"]
        source_proof=lookup(campaign["configuration"]["verification_cache"],original_path,library,native["source_verilog"],ROOT/"target/release/bench_rs_netlist_eval")
        assert source_proof is not None
        for metric in ["area","delay"]:
            assert abs(source_proof["measured"][metric]-native["source"][metric])<1e-6
        proof_kind = row.get("verification_mode", "failed")
        if row["status"] == "ok":
            assert row["measured"]["cec"] == 1 and row["measured"]["multioutput_instances"] == 0
            for metric in ["area", "delay"]:
                assert abs(row["native"][metric] - row["measured"][metric]) < 1e-6
            if "reuse" in proof_kind:
                old_evaluator = ROOT / "target/release/bench_rs_netlist_eval"
                proof = lookup(campaign["configuration"]["verification_cache"], original_path, library, raw, old_evaluator)
                assert proof == row["verification_reuse"] and proof["measured"] == row["measured"]
            else:
                calls = [c for c in commands if Path(c["argv"][0]).name == "fast_original_cec.py"]
                assert len(calls) == 1 and not calls[0]["mapping_charged"]
                argv = calls[0]["argv"]
                assert sha(argv[0]) == campaign["fingerprint"][argv[0]]
                assert sha(argv[1]) == document["library_sha256"] and sha(argv[2]) == row["input_sha256"]
                normalize = [c for c in commands if Path(c["argv"][0]).name == "yosys" and "-p" in c["argv"] and str(raw) in c["argv"][-1]]
                assert len(normalize) == 1 and str(folder / "mapped.json") in normalize[0]["argv"][-1]
                check = json.loads((folder / "functional-cec/verification.json").read_text())
                assert check["cec"] == 1 and check["original_gate_bytes_unchanged"] and check["functional_models_from_verilog_not_mapper_truth_tables"]
                records = [json.loads(line) for line in (folder / "evaluate.log").read_text().splitlines() if line.startswith('{"area":')]
                assert records == [row["measured"]]
            if native["selected"] == "source_witness":
                assert sha(raw) == native["source_sha256"]
            else:
                assert native["optimizer_succeeded"] and attempt["complete_returned_cover"]
                n = native["optimizer_native"]
                assert n["instances"] <= 100000 and n["max_cuts_per_class"] <= 16
        native_error=Path(attempt["stderr_path"]).read_text().strip() if not attempt["complete_returned_cover"] else None
        stage_path=folder/"attempt0.stages.jsonl"
        stages=[json.loads(line)for line in stage_path.read_text().splitlines()] if stage_path.exists() else []
        rows.append(dict(native_failure=native_error,native_last_stage=stages[-1]["stage"] if stages else None,name=job["name"], bench=job["bench"], k=job["k"], ratio=job["ratio"], repetition=job["repetition"],
            status=row["status"], error=row.get("error"), selected=native["selected"],
            optimized_selected_and_CEC=row["status"] == "ok" and native["selected"] == "egraph_cover",
            optimizer_returned_internal_cover=native["optimizer_succeeded"], verification_mode=proof_kind,
            mapping_secs=mapping[0]["wall_secs"], admitted=native["cumulative_accounted_enode_upper"], allowance=capacity["node_limit"],
            area=row.get("measured", {}).get("area"), delay=row.get("measured", {}).get("delay"),
            source_area=native["source"]["area"], source_delay=native["source"]["delay"],
            native_exit=attempt["exit_code"], native_timeout=attempt["timeout"], report=job["report"], audit=job["audit"]))
    counts = {}
    for ratio in campaign["configuration"]["ratios"]:
        sub = [r for r in rows if r["ratio"] == ratio]
        counts[ratio] = dict(attempts=len(sub), verified=sum(r["status"] == "ok" for r in sub),
            optimized_selected=sum(r["optimized_selected_and_CEC"] for r in sub),
            source_only=sum(r["status"] == "ok" and r["selected"] == "source_witness" for r in sub),
            native_timeouts=sum(r["native_timeout"] for r in sub),
            verification_modes=dict(collections.Counter(r["verification_mode"] for r in sub)))
    result = dict(complete=campaign["status"] == "attempted" and len(rows) == campaign["expected_jobs"],
                  attempts=len(rows), expected=campaign["expected_jobs"], counts=counts,
                  baseline_algorithms_changed=False, new_resource_protocol=True,
                  cached_proofs_are_not_fresh_CEC_runs=True, rows=rows)
    (args.campaign.parent / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
