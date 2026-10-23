#!/usr/bin/env python3
"""Create one immutable resyn2 subject per EPFL circuit, independent of cell library."""
import argparse
import hashlib
import json
import signal
import subprocess
import time
from pathlib import Path
from benchmark_suites import EPFL
from shared_subject import RECIPE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {"kind": "logic_subjects", "suite": "epfl", "preprocess": "resyn2",
              "source": "https://github.com/lsils/benchmarks", "distribution": "vendored mockturtle AIGs; file hashes pinned below",
              "uncovered": ["MtM sixteen", "MtM twenty", "MtM twentythree"],
              "timeout_secs": args.timeout, "abc_sha256": sha(Path("/usr/bin/abc")),
              "commands": [], "results": []}

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(script, log):
        cmd = ["/usr/bin/abc", "-c", script]
        report["commands"].append(cmd)
        save()
        started = time.perf_counter()
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, start_new_session=True) as p:
            try:
                text, _ = p.communicate(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                import os
                os.killpg(p.pid, signal.SIGKILL)
                text, _ = p.communicate()
                log.write_text(text + "\nTIMEOUT\n")
                raise RuntimeError("process-group timeout")
        log.write_text(text)
        if p.returncode:
            raise RuntimeError(f"exit {p.returncode}: {log}")
        return text, time.perf_counter() - started

    for bench in EPFL:
        golden = root / "tools/mockturtle/experiments/benchmarks" / f"{bench}.aig"
        row = {"bench": bench, "method": "shared_resyn2", "preprocessing": {}}
        try:
            row["input_sha256"] = sha(golden)
            header = golden.open("rb").readline().decode().split()
            if header[0] != "aig" or int(header[3]) != 0:
                raise ValueError("expected combinational binary AIGER")
            row["original_aiger_header"] = header
            subject = out / f"{bench}.resyn2.aig"
            _, seconds = run(f'read "{golden}"; strash; {RECIPE}; write_aiger -s "{subject}"', out / f"{bench}.prepare.log")
            text, cec_secs = run(f'cec "{golden}" "{subject}"', out / f"{bench}.cec.log")
            if "Networks are equivalent" not in text:
                raise RuntimeError("preprocessing equivalence unknown or failed")
            row.update(status="ok", preprocessing={"recipe": RECIPE, "sha256": sha(subject),
                       "cec": 1, "wall_secs": seconds, "cec_wall_secs": cec_secs})
        except (OSError, RuntimeError, ValueError) as error:
            row.update(status="failed", error=str(error))
        report["results"].append(row)
        save()
        print(bench, row["status"], flush=True)


if __name__ == "__main__":
    main()
