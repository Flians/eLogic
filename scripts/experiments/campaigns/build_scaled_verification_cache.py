"""Index successful original-input CEC proofs; do not run or alter a mapper."""
import argparse
import json
from pathlib import Path

from audit_generation_commands import audit
from scaled_verification_cache import key, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", action="append", type=Path, default=[])
    parser.add_argument("--supplement", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    entries = {}
    for path in args.campaign:
        campaign = json.loads(path.read_text())
        assert all(sha(p) == h for p, h in campaign["fingerprint"].items())
        for job in campaign["jobs"]:
            assert all(sha(p) == h for p, h in job["sha256"].items())
            report = json.loads(Path(job["report"]).read_text())
            row = report["results"][0]
            if row["status"] != "ok":
                continue
            proof = audit(job, campaign)
            evaluator, library, original, mapped, netjson = proof["evaluator_argv"]
            folder = Path(job["report"]).parent / (row["bench"] + "." + row["method"])
            raw = folder / "raw.v"
            files = [path, Path(job["report"]), Path(job["audit"]), raw,
                     Path(evaluator), Path(library), Path(original), Path(mapped),
                     Path(netjson), folder / "evaluate.log"]
            entry = dict(measured=row["measured"], raw_verilog=str(raw),
                         proof_files={str(p.resolve()): sha(p) for p in files},
                         source="recorded independent original named-port CEC",
                         provenance=proof)
            entries.setdefault(key(original, library, raw, evaluator), entry)
    for path in args.supplement:
        document = json.loads(path.read_text())
        assert document["exit_code"] == 0 and not document["timeout"]
        assert all(sha(p) == h for p, h in document["proof_files"].items())
        evaluator, library, original, mapped, netjson = document["argv"]
        records = [json.loads(line) for line in Path(document["stdout_log"]).read_text().splitlines()
                   if line.startswith('{"area":')]
        assert len(records) == 1 and records[0]["cec"] == 1
        files = dict(document["proof_files"])
        files[str(path.resolve())] = sha(path)
        entry = dict(measured=records[0], raw_verilog=document["raw_verilog"],
                     proof_files=files, source="verification-only original-CEC supplement",
                     provenance=dict(argv=document["argv"],
                                     verification_guard_secs=document["guard_secs"],
                                     mapping_rerun=False))
        entries.setdefault(key(original, library, document["raw_verilog"], evaluator), entry)
    args.output.write_text(json.dumps(dict(entries=entries, fresh_cec=False,
                                          reuse_requires_identical_raw_netlist=True), indent=2))
    print("independent proof cache entries", len(entries))


if __name__ == "__main__":
    main()
