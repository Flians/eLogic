"""Reuse only an identical physical netlist's recorded independent original CEC.

Cache keys include the original input, library, raw mapped Verilog and common
evaluator binary. All proof artifacts are checked before use. Failed CECs never
enter this cache; cached checks are reported separately from fresh executions.
"""
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(original, library, raw, evaluator):
    return ":".join(sha(path) for path in (original, library, raw, evaluator))


def lookup(cache_path, original, library, raw, evaluator):
    document = json.loads(Path(cache_path).read_text())
    entry = document["entries"].get(key(original, library, raw, evaluator))
    if entry is None:
        return None
    if entry["measured"].get("cec") != 1:
        raise ValueError("verification cache contains an unproved result")
    for path, expected in entry["proof_files"].items():
        if sha(path) != expected:
            raise ValueError("verification proof artifact changed: " + path)
    if sha(entry["raw_verilog"]) != sha(raw):
        raise ValueError("verification raw netlist mismatch")
    return entry
