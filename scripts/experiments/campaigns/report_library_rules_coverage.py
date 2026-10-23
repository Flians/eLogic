"""Read-only rule matching, independent pool additions and emitted path analysis.

Generation counters precede common retention; emitted cell types do not prove
which generator supplied a selected binding or identify a common AIG boundary.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from library_rules_paths import analyze


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.campaign.read_text())
    assert len(state["jobs"]) == state["expected_jobs"], "finish the finite campaign first"
    records = []
    summary = {p: {"matches": Counter(), "inverse": Counter(), "generated": Counter(),
                   "stage_records": 0, "phase_helpers": 0, "physical_cell_nodes": 0}
               for p in state["configuration"]["profile_values"]}
    for job in state["jobs"]:
        assert all(sha(path) == digest for path, digest in job["sha256"].items())
        if not job.get("report"):
            records.append({"job": job["name"], "unmeasured_failure": True})
            continue
        folder = Path(job["report"]).parent
        row = json.loads(Path(job["report"]).read_text())["results"][0]
        lib = json.loads((folder / "library.json").read_text())
        wide = {c["name"] for c in lib if len(c["vars"]) > 6}
        stages = []
        generation = []
        target = summary[job["profile"]]
        for file in folder.rglob("*.library-rules-stage.json"):
            value = json.loads(file.read_text())
            if "phase_helpers" in value:
                assert value["added_nodes"] == value["phase_helpers"] + value["physical_cell_nodes"]
            target["stage_records"] += 1
            target["matches"].update(value["matches_by_cell"])
            target["inverse"].update(value.get("inverse_matches_by_cell", {}))
            for key in ["phase_helpers", "physical_cell_nodes"]:
                target[key] += value.get(key, 0)
            stages.append({"source": str(file), "sha256": sha(file),
                           "wide_matches": {k: v for k, v in value["matches_by_cell"].items() if k in wide},
                           "wide_inverse": {k: v for k, v in value.get("inverse_matches_by_cell", {}).items() if k in wide}})
        for file in folder.rglob("*.library-rules.json"):
            value = json.loads(file.read_text())
            target["generated"].update(value["generated_by_cell"])
            generation.append({"source": str(file), "sha256": sha(file),
                               "new_instance_entries_before_common_retention": value["generated_by_cell"]})
        path = analyze(folder, row["measured"]) if row.get("status") == "ok" else {"available": False}
        records.append({"job": job["name"], "bench": job["bench"], "profile": job["profile"],
                        "stages": stages, "generation": generation, "emitted_path": path})
    args.output.mkdir(exist_ok=False)
    (args.output / "coverage.json").write_text(json.dumps({
        "campaign": str(args.campaign.resolve()), "campaign_sha256": sha(args.campaign),
        "reporter_sha256": sha(__file__), "path_reader_sha256": sha(Path(__file__).with_name("library_rules_paths.py")),
        "summary": summary, "records": records,
        "selection_generator_attribution_proven": False,
        "cross_profile_aig_boundary_correspondence_proven": False,
        "generation_precedes_retention": True,
    }, indent=2) + "\n")
    print({p: {k: sum(v.values()) if isinstance(v, Counter) else v for k, v in s.items()}
           for p, s in summary.items()})


if __name__ == "__main__":
    main()
