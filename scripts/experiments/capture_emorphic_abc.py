#!/usr/bin/env python3
"""Transparent ABC launcher: archive E-morphic SA calls before upstream cleanup."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

script = sys.argv[sys.argv.index("-c") + 1]
output = re.search(r"(?:^|;)\s*write\s+(tmp_[0-9]+\.v)(?:;|$)", script)
if not output:
    raise SystemExit("unexpected native ABC call; cannot archive safely")
name = Path(output.group(1)).stem
folder = Path(os.environ["EMORPHIC_TRACE"]) / name
folder.mkdir(exist_ok=False)
input_path = re.search(r"(?:^|;)\s*read_eqn\s+([^;]+);", script)
if input_path:
    (folder / "input.eqn").write_bytes(Path(input_path.group(1).strip()).read_bytes())
start = time.perf_counter()
run = subprocess.run([os.environ["EMORPHIC_REAL_ABC"], *sys.argv[1:]], capture_output=True)
(folder / "stdout.log").write_bytes(run.stdout)
(folder / "stderr.log").write_bytes(run.stderr)
raw = Path(output.group(1))
if raw.exists(): (folder / "raw.v").write_bytes(raw.read_bytes())
metadata = {"argv": [os.environ["EMORPHIC_REAL_ABC"], *sys.argv[1:]],
            "cwd": os.getcwd(), "exit_code": run.returncode, "wall_secs": time.perf_counter()-start}
if raw.exists(): metadata["raw_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
(folder / "command.json").write_text(json.dumps(metadata, indent=2)+"\n")
sys.stdout.buffer.write(run.stdout); sys.stderr.buffer.write(run.stderr)
raise SystemExit(run.returncode)
