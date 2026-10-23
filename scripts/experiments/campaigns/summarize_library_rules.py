"""Summarize fixed-profile ablations without selecting a best run or design configuration."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from report_library_rules_study import cmp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar_cmp(x, y):
    eps = 1e-6 * max(1, abs(y))
    return "better" if x < y - eps else "worse" if x > y + eps else "equal"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--repeats", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full = json.loads(args.full.read_text())
    assert full["complete"] and full["all_recorded_budgets_pass"]
    result = {"full": {}, "native_failures": [], "repeat_targets": {},
              "input_hashes": {str(args.full): sha(args.full)},
              "summary_code_sha256": sha(__file__), "main_adoption": False,
              "best_run_selection": False, "configuration_per_design_selection": False}
    for profile in [1, 2]:
        pairs = [p for p in full["pairs"] if p["profile"] == profile]
        valid = [p for p in pairs if p["valid"]]
        stats = {"DA": dict(Counter(p["DA"] for p in pairs))}
        for key in ["A", "D", "T"]:
            stats[key] = dict(Counter(scalar_cmp(p["candidate"][key], p["control"][key]) for p in valid))
            stats[key + "_geometric_ratio_all_final_CEC_pairs"] = math.exp(sum(
                math.log(p["candidate"][key] / p["control"][key]) for p in valid) / len(valid))
        stats["pairs"] = len(pairs)
        result["full"][str(profile)] = stats
    for row in full["rows"]:
        if row.get("native_selected"):
            continue
        report = json.loads(Path(row["report"]).read_text())["results"][0]
        for attempt in report.get("scalability", {}).get("attempts", []):
            stderr = Path(attempt["stderr_path"])
            text = stderr.read_text()
            reason = ("native_timeout" if attempt.get("timeout") else
                      "memory_allocation_failure" if "memory allocation" in text and "failed" in text else
                      "native_failure_or_unselected")
            result["native_failures"].append({"job": row["job"], "reason": reason,
                                              "exit_code": attempt.get("exit_code"),
                                              "stderr": str(stderr), "stderr_sha256": sha(stderr)})
    if args.repeats:
        repeats = json.loads(args.repeats.read_text())
        assert repeats["complete"] and repeats["all_recorded_budgets_pass"]
        assert repeats["scope"]["native"] == full["scope"]["native"]
        result["input_hashes"][str(args.repeats)] = sha(args.repeats)
        targets = set(repeats["scope"]["designs"])
        groups = defaultdict(list)
        for label, report in [("full", full), ("repeat", repeats)]:
            for pair in report["pairs"]:
                if pair["bench"] in targets:
                    groups[(pair["bench"], pair["profile"])].append({"source": label, **pair})
        for (bench, profile), pairs in groups.items():
            result["repeat_targets"][f"{bench}/profile{profile}"] = {
                "DA": dict(Counter(p["DA"] for p in pairs)),
                "D": dict(Counter(scalar_cmp(p["candidate"]["D"], p["control"]["D"]) for p in pairs if p["valid"])),
                "A": dict(Counter(scalar_cmp(p["candidate"]["A"], p["control"]["A"]) for p in pairs if p["valid"])),
                "pairs": pairs}
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["full"], indent=2))


if __name__ == "__main__":
    main()
