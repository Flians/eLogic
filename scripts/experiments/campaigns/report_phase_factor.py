#!/usr/bin/env python3
"""Same-source recovery view with explicit, separately retained CEC supplements."""
import argparse
import hashlib
import json
from pathlib import Path
from report_delay_frontier import pool_hash
from certify_prepared_chain import validate_subject
from bench_epoch_incumbent import record

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def supplemental_rows(paths):
    table = {}
    for path in paths:
        data = json.loads(path.read_text())
        for row in data["results"]:
            if row["status"] != "ok":
                continue
            proof = row["verification_chain"]
            assert proof["prepared_to_mapped_cec"] and proof["identical_named_port_order"]
            origin = Path(row["original_report"])
            assert sha(origin) == row["original_report_sha256"]
            original = json.loads(origin.read_text())
            raw = next(r for r in original["results"]
                       if r["bench"] == row["bench"] and r["method"] == row["method"])
            assert raw["status"] == "failed" and not raw.get("partial_status")
            assert not raw.get("partial_measured") and "mapping_wall_secs" in raw
            folder = origin.parent / f"{row['bench']}.{row['method']}"
            assert "Timeout after 60s" in (folder/"evaluate.log").read_text()
            assert sha(folder/"mapped.v") == proof["mapped_verilog_sha256"]
            assert sha(folder/"mapped.json") == proof["mapped_json_sha256"]
            assert original["library_sha256"] == proof["library_sha256"]
            assert row["input_sha256"] == raw["input_sha256"]
            assert row["preprocessing"]["sha256"] == raw["preprocessing"]["sha256"]
            assert row["measured"]["cec"] == 1
            validate_subject(row, Path(proof["subject_report"]))
            commands = [c for c in data["commands"] if c["argv"][3] == str((folder/"mapped.v").resolve())]
            assert len(commands) == 1
            command = commands[0]
            assert sha(Path(command["log"])) == command["log_sha256"]
            assert record(Path(command["log"]).read_text()) == row["measured"]
            assert sha(Path(command["argv"][0])) == proof["evaluator_sha256"]
            assert sha(Path(command["argv"][1])) == proof["library_sha256"]
            assert sha(Path(command["argv"][2])) == proof["prepared_sha256"]
            assert command["per_process_guard_seconds"] == 60 and not command["mapping_charged"]
            first = Path(proof["subject_report"])
            assert sha(first) == proof["subject_report_sha256"]
            first_log = Path(proof["original_to_prepared_cec_log"])
            assert sha(first_log) == proof["original_to_prepared_cec_log_sha256"]
            assert "Networks are equivalent" in first_log.read_text()
            key = (sha(origin), row["bench"], row["method"])
            value = dict(row)
            value.pop("error", None)  # Original error remains in the direct row and original_error.
            if key in table:
                assert table[key][0]["measured"] == value["measured"]
            table[key] = (value, str(path), sha(path))
    return table

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--supplements", nargs="*", type=Path, default=[])
    parser.add_argument("--control", default="legacy_control")
    parser.add_argument("--candidate", default="protected_area")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-full", type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(exist_ok=False)
    campaign = json.loads(args.campaign.read_text())
    supplements = supplemental_rows(args.supplements)
    rows, groups, hashes, budgets = [], {}, [], []
    jobs=[dict(j,observation="repeat" if "repetition" in j else "full_r0") for j in campaign["jobs"]]
    expected=campaign["expected_jobs"]
    if args.reference_full:
        full=json.loads(args.reference_full.read_text())
        allowed={"run_unselected_phase.py","run_unselected_phase_repeats.py","run_helper_phase_full.py","run_helper_phase_repeats.py","run_guarded_joint_full.py","run_guarded_joint_repeats.py","run_joint_batch.py","run_joint_batch_repeats.py","run_library_source_plan.py","run_library_source_plan_repeats.py","run_epoch_report.py","run_epoch_report_repeats.py","run_epoch_report_main.py","run_epoch_report_main_repeats.py","run_tt3_import.py","run_tt3_import_repeats.py","run_tt3_main.py","run_tt3_main_repeats.py","run_tt3_helpers.py","run_tt3_helpers_repeats.py","run_scored_main.py","run_scored_main_repeats.py","run_scored_incumbent.py","run_scored_incumbent_repeats.py","run_cone_guard.py","run_cone_guard_repeats.py"}
        strip=lambda f:{p:h for p,h in f.items() if Path(p).name not in allowed}
        assert strip(full["fingerprint"])==strip(campaign["fingerprint"]),"Executed dependencies differ"
        selected=[dict(j,observation="full_r0") for j in full["jobs"] if j["bench"] in campaign["designs"] and j["profile"] in [args.control,args.candidate]]
        expected+=len(selected);jobs=selected+jobs
    for job in jobs:
        for field in ("report", "audit", "pool", "candidates", "trace", "diagnostic_profile"):
            if field in job and sha(Path(job[field])) != job[field+"_sha256"]:
                hashes.append(job["name"]+":"+field)
        parent = json.loads(Path(job["report"]).read_text())
        raw = parent["results"][0]
        row = raw
        key = (job["report_sha256"], raw["bench"], raw["method"])
        supplement = None
        if raw["status"] != "ok" and key in supplements:
            row, supplement, supplement_sha = supplements[key]
        item = dict(job=job["name"], k=job["k"], bench=job["bench"],
                    profile=job["profile"], repetition=job.get("repetition", "full_r0"), observation=job["observation"], row=row, direct_row=raw,
                    report=job["report"], report_sha256=job["report_sha256"],
                    library_sha256=parent["library_sha256"],
                    new_mapper_run=job.get("new_mapper_run", not bool(job.get("adopted_from"))),
                    adopted_from=job.get("adopted_from"),
                    verification_supplement=supplement)
        if supplement:
            item["verification_supplement_sha256"] = supplement_sha
        if "trace" in job:
            item["trace"] = json.loads(Path(job["trace"]).read_text())
        if "candidates" in job:
            item["pool_fingerprint"] = pool_hash(Path(job["candidates"]))
        if row["status"] == "ok":
            assert row["measured"]["cec"] == 1
            n = row["native"]
            if (n["enodes"] > 20000 or n["max_cuts_per_class"] > 16
                    or n["instances"] > 100000 or n["heuristic_secs"] > 2
                    or row["mapping_wall_secs"] > 300):
                budgets.append(job["name"])
        rows.append(item)
        groups.setdefault((job["observation"],job.get("repetition", "full_r0"),job["k"], job["bench"]), {})[job["profile"]] = item
    pairs, fingerprints = [], []
    counts = dict(delay=[0, 0, 0], area=[0, 0, 0])
    for (observation,repetition,k,bench), group in groups.items():
        x, y = group[args.control], group[args.candidate]
        pairs.append(dict(k=k, bench=bench, observation=observation, repetition=repetition, control=x, candidate=y))
        if x["row"]["status"] != "ok" or y["row"]["status"] != "ok":
            continue
        assert x["library_sha256"] == y["library_sha256"]
        assert x["row"]["input_sha256"] == y["row"]["input_sha256"]
        assert x["row"]["preprocessing"]["sha256"] == y["row"]["preprocessing"]["sha256"]
        if x.get("pool_fingerprint") != y.get("pool_fingerprint"):
            fingerprints.append([k, bench])
        for metric in counts:
            a, b = x["row"]["measured"][metric], y["row"]["measured"][metric]
            tolerance = 1e-6*max(1, abs(a), abs(b))
            counts[metric][0 if b < a-tolerance else 2 if b > a+tolerance else 1] += 1
    assert len(rows)==expected,"Incomplete declared scope"
    summary = dict(case_slots=len(rows), expected_slots=expected,
                   direct_cec=sum(r["direct_row"]["status"] == "ok" for r in rows),
                   additional_chain_cec=sum(bool(r["verification_supplement"]) for r in rows),
                   cec=sum(r["row"]["status"] == "ok" for r in rows),
                   failure_slots=sum(r["row"]["status"] != "ok" for r in rows),
                   eligible_pairs=sum(counts["delay"]), better_equal_worse=counts,
                   pool_fingerprint_mismatch=fingerprints, budget_errors=budgets,
                   artifact_hash_errors=hashes, baseline_algorithms_changed=False,
                   production_promoted=False, source_manifest=str(args.campaign),
                   source_manifest_sha256=sha(args.campaign),
                   verification_policy="Direct original named-port CEC or explicit original->fixed-resyn2->mapped chain; original failures retained; no partial mapping accepted.")
    (out/"all_rows.json").write_text(json.dumps(rows, indent=2))
    (out/"paired.json").write_text(json.dumps(pairs, indent=2))
    (out/"validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
