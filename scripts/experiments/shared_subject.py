"""Reuse an immutable, independently verified resyn2 subject across adapters."""
import hashlib
import json
from pathlib import Path

RECIPE = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance"


def reuse_subject(report_path, bench, golden, library, output):
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if report.get("preprocess") != "resyn2" or (
        report.get("kind") != "logic_subjects" and report.get("library_sha256") != digest(library)
    ):
        raise ValueError("incompatible shared subject report")
    rows = [r for r in report["results"] if r["bench"] == bench]
    if not rows: raise ValueError("missing shared subject: " + bench)
    if any(r.get("status") != "ok" for r in rows):
        raise ValueError("shared subject contains a failed result: " + bench)
    metadata = rows[0]["preprocessing"]
    subject = report_path.parent / (bench + ".resyn2.aig")
    if (metadata["cec"] != 1 or metadata["recipe"] != RECIPE or
        any(r["input_sha256"] != digest(golden) or r["preprocessing"] != metadata for r in rows) or
        digest(subject) != metadata["sha256"]):
        raise ValueError("shared subject provenance mismatch")
    Path(output).write_bytes(subject.read_bytes())
    return dict(metadata, reused_from=str(subject), source_report_sha256=digest(report_path))
