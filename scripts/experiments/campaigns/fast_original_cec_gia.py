#!/usr/bin/env python3
"""Independent GIA CEC of the same named, unchanged physical netlist.

Reuse the original functional-model conversion and common physical metrics.
Only the checker command changes. Require canonical PI/PO names *and order*
before using GIA; never guess positional port correspondence. Proof files go
to a new directory, preserving earlier timeout artifacts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import fast_original_cec as legacy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gia_script(golden, implementation):
    names = []
    for path in [golden, implementation]:
        if any(c in str(path) for c in '\n\r;"'):
            raise ValueError("unsupported verification path")
        _, ins, outs, _ = legacy.named_golden(Path(path).read_bytes())
        assert ins == [f"cec_i{i}" for i in range(len(ins))]
        assert outs == [f"cec_o{i}" for i in range(len(outs))]
        names.append((ins, outs))
    assert names[0] == names[1], "canonical named PI/PO order differs; refuse positional guessing"
    return f'&r "{golden}"; &cec -T 590 "{implementation}"'


def main():
    ap = argparse.ArgumentParser(description=__doc__, add_help=False)
    ap.add_argument("--proof-dir", type=Path, required=True)
    a, remaining = ap.parse_known_args()
    assert len(remaining) == 4
    folder = a.proof_dir.resolve()
    folder.mkdir(exist_ok=False, parents=True)
    original_run, original_verify = subprocess.run, legacy.verify

    def run(argv, *args, **kwargs):
        if isinstance(argv, list) and len(argv) == 3 and argv[1] == "-c" and argv[2].startswith('cec "'):
            tokens = shlex.split(argv[2])
            assert len(tokens) == 3 and tokens[0] == "cec"
            old = list(argv)
            argv[2] = gia_script(Path(tokens[1]), Path(tokens[2]))
            executable = shutil.which(argv[0])
            assert executable
            (folder / "gia_command.json").write_text(json.dumps(dict(
                argv=list(argv), original_legacy_command_not_run=old,
                abc_sha256=sha(executable), canonical_named_port_order_verified=True,
                physical_netlist_remapped=False, baseline_algorithms_changed=False,
            ), indent=2) + "\n")
        return original_run(argv, *args, **kwargs)

    def verify(models, golden, netjson, ignored_folder):
        return original_verify(models, golden, netjson, folder)

    protocol = dict(checker="ABC GIA &cec", mapping_rerun=False,
        baseline_algorithms_changed=False, physical_netlist_unchanged=True,
        metrics="same measure_json all-instance area and unloaded dependent-pin delay",
        legacy_verifier_sha256=sha(Path(legacy.__file__)), wrapper_sha256=sha(Path(__file__)),
        library_sha256=sha(remaining[0]), original_sha256=sha(remaining[1]),
        physical_verilog_sha256=sha(remaining[2]), physical_json_sha256=sha(remaining[3]),
        internal_approximate_cec_limit_secs=590, outer_verification_guard_secs=600,
        measurement_binary_sha256=sha(os.environ["EGM_MEASURE_BINARY"]))
    (folder / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    subprocess.run, legacy.verify = run, verify
    old_argv = sys.argv
    sys.argv = [legacy.__file__, *remaining]
    try:
        legacy.main()
    finally:
        subprocess.run, legacy.verify, sys.argv = original_run, original_verify, old_argv


if __name__ == "__main__":
    main()
