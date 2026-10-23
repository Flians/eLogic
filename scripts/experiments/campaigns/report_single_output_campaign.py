"""Read-only single-output scope: retain all designs, failures and missing data."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from run_epfl_baselines import ROOT, EPFL, sha
from report_epfl_comparison import PRIMARY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification-supplements", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    subjects = {row["bench"]: row for row in json.loads(args.subjects.read_text())["results"]}
    observations, library, hashes, time_bounds, native_checks = {}, None, {}, [], []
    candidate_campaign = json.loads(args.candidate.read_text())
    configuration = candidate_campaign["configuration"]
    parameter = configuration.get("profile_parameter", "EGM_WORD_TRUTH")
    profiles = configuration.get("profile_values", [0, 1])
    k = configuration.get("k", 6)
    for path in [Path(__file__).resolve(), args.subjects.resolve(),
                 ROOT / "scripts/experiments/check_mapping_dominance.py",
                 Path(__file__).with_name("report_epfl_comparison.py"),
                 Path(__file__).with_name("run_epfl_baselines.py")]:
        hashes[str(path)] = sha(path)
    for label, path in [("candidate", args.candidate), ("baseline", args.baseline)]:
        campaign = json.loads(path.read_text())
        hashes[str(path.resolve())] = sha(path)
        observations[label] = []
        for job in campaign["jobs"]:
            if not job.get("report"):
                methods = (["egraph_staged_control_resyn2"] if label == "candidate" else
                           ["eqmap_lut_amap", "lut_stage_amap", "abc_amap"] if "_eqmap_" in job["name"] else
                           ["esyn_frozen_model_amap_bounded_dag"] if "_esyn_" in job["name"] else
                           ["emorphic_final_sa"] if "_emorphic_" in job["name"] else PRIMARY[:4])
                for method in methods:
                    value = dict(bench=job["bench"], method=method, status="failed",
                        input_sha256=subjects[job["bench"]]["input_sha256"], unmeasured=True,
                        error="process failed without report; argv/log/audit retained", source_job=job["name"])
                    if label == "candidate":
                        value["profile"] = job["profile"]
                    observations[label].append(value)
                continue
            report_path = Path(job["report"])
            assert sha(report_path) == job["sha256"][str(report_path)]
            hashes[str(report_path)] = sha(report_path)
            report = json.loads(report_path.read_text())
            if library is None:
                library = report["library_sha256"]
            assert report["library_sha256"] == library, "mixed libraries"
            audit_path = Path(job["argv"][job["argv"].index("--audit") + 1])
            assert sha(audit_path) == job["sha256"][str(audit_path)]
            hashes[str(audit_path)] = sha(audit_path)
            audit = json.loads(audit_path.read_text())
            for row in report["results"]:
                value = copy.deepcopy(row)
                assert value["input_sha256"] == subjects[value["bench"]]["input_sha256"]
                value.update(source_job=job["name"], observation_report=str(report_path))
                if label == "candidate":
                    value["profile"] = job["profile"]
                    if parameter in ["EGM_FAST_ADMISSION_DELTA", "EGM_EXACT_CUT_OPS", "EGM_EXCHANGE_TIME_FRACTION", "EGM_SHARE_PHYSICAL_KEYS", "EGM_FORCE_COMPACT_SOURCE", "EGM_WORD_QUOTIENT", "EGM_FAST_TT3_SUPPORT", "EGM_ROOT_FRONTIER_STOP", "EGM_IMPORT_HELPER_FILL", "EGM_SKIP_REDUNDANT_WIRE_PASS", "EGM_REUSE_RETENTION_RANKS"] and "scalability" in value:
                        policy = value["resource_policy"]
                        search = value["scalability"]
                        assert search["cumulative_accounted_enode_upper"] <= policy["node_limit"]
                        assert search["source_plus_optimizer_retained_upper"] <= policy["global_retained_candidate_limit"]
                        assert not policy["refund_on_rebuild"]
                        assert value["mapping_wall_secs"] <= 300
                        for attempt in search["attempts"]:
                            stderr = Path(attempt["stderr_path"])
                            assert sha(stderr) == job["sha256"][str(stderr)]
                            hashes[str(stderr)] = sha(stderr)
                            text = stderr.read_text()
                            desired = dict(configuration.get("fixed_native_configuration",
                                {"EGM_WORD_TRUTH":"1", "EGM_ADMISSION_SHADOW":"0"}))
                            desired.update(configuration.get("coupled_profile_configuration", {}).get(str(job["profile"]), {}))
                            desired[parameter] = configuration.get("profile_actual_values", {}).get(str(job["profile"]), str(job["profile"]))
                            observed_import_plan = None
                            for flag, expected in desired.items():
                                if flag == "EGM_FORCE_COMPACT_SOURCE":
                                    ipath = Path(attempt.get("import_plan_path", str(Path(attempt["pool_path"]).with_suffix(".import-plan.json"))))
                                    assert search["forced_compact_source_requested"] == (expected == "1")
                                    if ipath.exists():
                                        assert sha(ipath) == job["sha256"][str(ipath)]
                                        hashes[str(ipath)] = sha(ipath)
                                        plan = json.loads(ipath.read_text())
                                        observed_import_plan = plan
                                        assert plan["forced_compact"] == (expected == "1")
                                        assert plan["node_limit"] == policy["node_limit"]
                                        if attempt.get("import_plan") is not None:
                                            assert attempt["import_plan"] == plan
                                    else:
                                        assert not attempt["complete_returned_cover"] and search["selected"] == "source_witness"
                                elif flag == "EGM_IMPORT_HELPER_FILL":
                                    # The driver records its pre-wrapper environment. This
                                    # explicit experiment overrides that one flag at spawn;
                                    # the native helper-allocation artifact is authoritative.
                                    value["driver_pre_wrapper_helper_fill"] = value["settings"][flag]
                                    hpath = Path(attempt["pool_path"]).with_suffix(".helper-allocation.json")
                                    assert hpath.exists(), "helper route must be actually observed"
                                    assert sha(hpath) == job["sha256"][str(hpath)]
                                    hashes[str(hpath)] = sha(hpath)
                                    helper = json.loads(hpath.read_text())
                                    assert helper["enabled"] == (expected == "1")
                                    value["actual_worker_helper_fill"] = helper["enabled"]
                                    value["actual_helper_sources_count"] = len(helper["actual_helper_sources"])
                                    value["helper_metadata_note"] = "native allocation plan checked; driver settings were captured before experiment wrapper"
                                    assert helper["global_node_allowance"] == policy["node_limit"]
                                    assert helper["planned_import_upper"] <= helper["initial_node_ceiling"] <= policy["node_limit"]
                                else:
                                    assert f"configuration: {flag}={expected}\n" in text
                            native_checks.append(dict(job=job["name"], profile=job["profile"],
                                node_admission=attempt["cumulative_accounted_upper"],
                                node_limit=policy["node_limit"], timeout=attempt.get("timeout", False),
                                selected=search["selected"], requested_configuration_verified=True,
                                configuration_verified=(parameter != "EGM_FORCE_COMPACT_SOURCE" or observed_import_plan is not None),
                                actual_native_import_plan=observed_import_plan))
                if value["status"] == "ok":
                    assert value["measured"]["cec"] == 1
                if value["method"] in ["eqmap_lut_amap", "lut_stage_amap"]:
                    enclosing = sum(command["wall_secs"] for state in audit["states"].values()
                        for command in state["commands"] if Path(command["argv"][0]).name == "yosys"
                        and "abc -lut" in " ".join(command["argv"]))
                    direct = audit["states"].get(value["bench"] + "/" + value["method"], {}).get("mapping_wall_secs", 0.)
                    upper = direct + enclosing
                    value.update(mapping_budget_upper_bound_secs=upper, mapping_time_exact=False,
                                 mapping_time_note="initial nested LUT mapping: enclosing Yosys wall upper bound")
                    if value["status"] == "ok" and upper > 300:
                        value.update(status="failed", error="mapping300s not proved by conservative upper bound",
                                     partial_measured=value.pop("measured"))
                    time_bounds.append(dict(job=job["name"], method=value["method"], upper_secs=upper))
                observations[label].append(value)
    assert library is not None
    model = "all physical instances; max(rise,fall) block pin delay; no load or slew"
    missing = lambda bench, method: dict(bench=bench, method=method, status="missing", error="not attempted",
                                        input_sha256=subjects[bench]["input_sha256"])
    baseline = observations["baseline"]
    supplements = []
    for supplement_path in args.verification_supplements:
        supplement = json.loads(supplement_path.read_text())
        assert supplement["mapping_rerun"] is False and supplement["baseline_algorithms_changed"] is False
        hashes[str(supplement_path.resolve())] = sha(supplement_path)
        for path, expected in supplement["fingerprint"].items():
            assert sha(Path(path)) == expected, path
            hashes[path] = expected
        for job in supplement["jobs"]:
            assert all(sha(Path(p)) == h for p, h in job["sha256"].items())
            case = job["original"]
            matches = [row for row in baseline if row.get("observation_report") == case["original_report"]
                       and row["bench"] == case["bench"] and row["method"] == case["method"]]
            assert len(matches) == 1, "supplement must name exactly one original observation"
            row = matches[0]
            original_failure = row.get("original_failure_retained", dict(status=row["status"], error=row.get("error")))
            assert original_failure["status"] == case["old_status"]["status"] == "failed"
            assert original_failure["error"] == case["old_status"]["error"] == "timeout after 60s"
            assert row["native"] and row["input_sha256"] == case["input_sha256"]
            assert case["library_sha256"] == library
            audit = json.loads(Path(case["original_audit"]).read_text())
            state = audit["states"][case["bench"] + "/" + case["method"]]
            assert not state["exhausted"] and state["mapping_wall_secs"] <= 300
            assert any(c["argv"] == case["argv"] and not c["mapping_charged"] for c in state["commands"])
            proof_dir = Path(job.get("proof_dir", str(Path(case["argv"][3]).parent / "functional-cec")))
            cec_log = proof_dir / "cec.log"
            negative = cec_log.is_file() and "not equivalent" in cec_log.read_text().lower()
            verdict = "equivalent" if job["status"] == "ok" else "not_equivalent" if negative else "unproved"
            history = row.setdefault("verification_history", [])
            assert not any({old["verdict"], verdict} == {"equivalent", "not_equivalent"} for old in history), "independent CEC engines disagree; investigate before qualifying"
            proof = dict(supplement=str(supplement_path), name=job["name"], status=job["status"],
                         checker=supplement.get("checker", "ABC cec"), verdict=verdict,
                         old_status_retained=case["old_status"], mapping_rerun=False,
                         original_report=case["original_report"], mapping_clock_unchanged=True,
                         verification_wall_secs=job["wall_secs"], sha256=job["sha256"])
            row["verification_supplement"] = proof
            history.append(proof)
            if job["status"] == "ok":
                assert job["measured"]["cec"] == 1 and not job["timeout"]
                witness = json.loads((proof_dir / "verification.json").read_text())
                assert witness["cec"] == 1 and witness["mapper_called"] is False
                assert witness["original_gate_bytes_unchanged"] is True
                if row["status"] == "ok":
                    assert row["measured"] == job["measured"], "same frozen physical netlist yielded different public metrics"
                row.update(original_failure_retained=original_failure,
                           status="ok", measured=job["measured"])
                row.pop("error", None)
            supplements.append(proof)
    for bench in EPFL:
        for method in PRIMARY:
            if not any(row["bench"] == bench and row["method"] == method for row in baseline):
                baseline.append(missing(bench, method))
    baseline_path = output / "baseline_scope.json"
    baseline_path.write_text(json.dumps(dict(library_sha256=library, metric_model=model, results=baseline), indent=2))
    gates = []
    for profile in profiles:
        candidate = [row for row in observations["candidate"] if row.get("profile") == profile]
        for bench in EPFL:
            if not any(row["bench"] == bench for row in candidate):
                candidate.append(missing(bench, "egraph_staged_control_resyn2"))
        path = output / f"candidate_profile{profile}.json"
        path.write_text(json.dumps(dict(library_sha256=library, metric_model=model, results=candidate), indent=2))
        for objective in ["delay-area", "area-delay", "pareto"]:
            target = output / f"profile{profile}_{objective}_gate.json"
            argv = [sys.executable, str(ROOT / "scripts/experiments/check_mapping_dominance.py"),
                "--candidate", str(path), "--baselines", str(baseline_path), "--required-baselines", *PRIMARY,
                "--objective", objective, "--output", str(target)]
            result = subprocess.run(argv, capture_output=True, text=True)
            target.with_suffix(".log").write_text(result.stdout + result.stderr)
            gate = json.loads(target.read_text())
            gates.append(dict(profile=profile, objective=objective, all_passed=gate["all_passed"], argv=argv,
                              exit_code=result.returncode, posthoc_only=objective == "area-delay"))
    summary = dict(hashes=hashes, gates=gates, baseline_algorithms_changed=False,
                   unconnected=["MtM"], eqmap_mapping_time_bounds=time_bounds,
                   profile_parameter=parameter, profile_values=profiles,
                   native_budget_configuration_checks=native_checks,
                   verification_supplements=supplements,
                   library_scope=f"K{k}: "+str({3:19,4:30,6:40}[k])+"/47 single-output cells",
                   objective="delay-area; area-delay gate is posthoc, not an area-first optimization run")
    (output / "validation.json").write_text(json.dumps(summary, indent=2))
    print("gates:", [gate["all_passed"] for gate in gates])


if __name__ == "__main__":
    main()
