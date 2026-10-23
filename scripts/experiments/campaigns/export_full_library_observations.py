"""Export all full-library baseline observations, including failed and unproved cases."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scope = json.loads(args.scope.read_text())
    campaign = json.loads(args.checkpoint.read_text())
    assert len(campaign["jobs"]) == campaign["expected_jobs"] == 120
    assert not campaign.get("current_job")
    jobs = {job["name"]: job for job in campaign["jobs"]}
    assert len(jobs) == 120
    rows, audits = [], {}
    for value in scope["results"]:
        assert value["status"] != "missing", "finish the declared scope before exporting"
        name = value["source_job"]
        job = jobs[name]
        audit_path = Path(job["argv"][job["argv"].index("--audit") + 1])
        assert sha(audit_path) == job["sha256"][str(audit_path)]
        if name not in audits:
            audits[name] = json.loads(audit_path.read_text())
        state = audits[name]["states"].get(value["bench"] + "/" + value["method"])
        charged = state.get("mapping_wall_secs") if state else None
        upper = value.get("mapping_budget_upper_bound_secs")
        runtime = upper if upper is not None else charged
        good = value["status"] == "ok"
        measured = value.get("measured", {})
        if good:
            assert measured["cec"] == 1 and runtime is not None and runtime <= 300
            assert not (state and state["exhausted"])
        history = value.get("verification_history", [])
        # Preserve the original failure text and the latest proof outcome separately.
        # A verification timeout is not evidence of non-equivalence or mapping timeout.
        verdict = history[-1]["verdict"] if history else "equivalent" if good else "not_qualified"
        assert not good or verdict == "equivalent"
        objective = next((goal for goal in ["area-delay", "delay-area"] if goal in name),
                         "native-final" if "_emorphic_" in name else
                         "area-delay" if "_area_" in value["method"] else "delay-area")
        rows.append(dict(
            bench=value["bench"], method=value["method"], objective=objective,
            status=value["status"], qualification=verdict,
            observation_kind="unmeasured_driver_exit" if value.get("unmeasured") else "reported",
            A=measured.get("area", "") if good else "",
            D=measured.get("delay", "") if good else "",
            T=runtime if good else "",
            T_kind="upper_bound_enclosing_LUT_conversion" if upper is not None else
                   "mapping_only_active_interval_wall" if charged is not None else "unavailable",
            mapping_charged_secs=charged if charged is not None else "",
            original_error=value.get("error", ""),
            original_failure_retained=json.dumps(value.get("original_failure_retained"), separators=(",", ":")),
            latest_verification_wall_secs=history[-1]["verification_wall_secs"] if history else "",
            verification_history=json.dumps(history, separators=(",", ":")),
            source_job=name, source_cohort=job["source_cohort"],
            observation_report=value.get("observation_report", ""),
            audit=str(audit_path), audit_sha256=sha(audit_path),
        ))
    assert len(rows) == 260, len(rows)
    assert len({row["bench"] for row in rows}) == 20
    assert all(count == 13 for count in collections.Counter(row["bench"] for row in rows).values())
    args.output.mkdir(exist_ok=False)
    with (args.output / "baseline_A_D_T.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(
        observations=len(rows), verified=sum(row["status"] == "ok" for row in rows),
        failed=sum(row["status"] != "ok" for row in rows),
        unproved_verification=sum(row["qualification"] == "unproved" for row in rows),
        mapping_limit_seconds=300,
        exact_T_excludes_CEC_conversion_resyn2_reporting=True,
        EqMap_T_is_conservative_upper_including_enclosing_conversion_not_exact=True,
        library_scope="same provided47 single-output cells; not all foundry cells",
        core_emap="uniform stock upstream library9/structural9; legacy capacity6 is separate",
        noncore="original author models/search and LUT6 parameters, necessary format/port adapters only",
        baseline_algorithms_changed=False, per_design_winner_selection=False,
        failed_or_partial_metrics_not_used=True, unseeded_native_SA_winner_preserved=True,
        scope=str(args.scope.resolve()), scope_sha256=sha(args.scope),
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
        exporter_sha256=sha(Path(__file__)),
    )
    (args.output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
