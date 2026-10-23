"""Replay one native attempt for profiling; never a source of qualified PPA."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

from run_epfl_baselines import ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--word-reduce",type=int,choices=[0,1],required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(exist_ok=False)
    locks = []
    for path in [ROOT / "results_v2/independent_components_20261019/global.lock",
                 ROOT / "results_v2/epfl_original_baselines_resumed_checkpoint_20261013/runner.lock"]:
        stream = path.open("a")
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(stream)
    report = json.loads(args.report.read_text())
    assert len(report["results"]) == 1
    row = report["results"][0]
    original = row["scalability"]["attempts"][0]["argv"]
    argv = [str(args.native.resolve()), original[1], original[2], str(folder / "raw.v")]
    env = {k: v for k, v in os.environ.items() if not k.startswith("EGM_")}
    env.update(row["settings"])
    for key in list(env):
        if key.startswith("EGM_") and str(args.report.parent) in env[key]:
            del env[key]
    env.update(EGM_WORD_TRUTH="1", EGM_WORD_REDUCE=str(args.word_reduce), EGM_FAST_ADMISSION_DELTA="1", EGM_ADMISSION_SHADOW="0", EGM_NATIVE_OPTIMIZER=str(args.native.resolve()), EGM_NOCEC="1", EGM_POOL_DIAG_DETAIL="0",
               EGM_POOL_DIAG_PATH=str(folder / "pool.json"),
               EGM_RECOVERY_TRACE=str(folder / "recovery.json"),
               EGM_STAGE_TRACE=str(folder / "stages.jsonl"))
    env.pop("EGM_DIAG_PATH", None)
    assert env["EGM_NODES"] == str(row["resource_policy"]["node_limit"])
    result = dict(purpose="profiling only; not verified PPA or a benchmark winner", argv=argv,
                  source_report=str(args.report.resolve()), source_report_sha256=sha(args.report),
                  native_sha256=sha(args.native), mapping_only_guard_seconds=60,
                  settings={k: v for k, v in env.items() if k.startswith("EGM_")},
                  input_sha256=sha(Path(original[2])), library_sha256=sha(Path(original[1])))
    (folder / "invocation.json").write_text(json.dumps(result, indent=2))

    def memory_limit():
        resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))

    started = time.monotonic()
    proc = subprocess.Popen(argv, cwd=ROOT, env=env, start_new_session=True,
                            preexec_fn=memory_limit, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    timeout = False
    try:
        stdout, stderr = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        timeout = True
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
    (folder / "stdout.log").write_text(stdout)
    (folder / "stderr.log").write_text(stderr)
    result.update(wall_seconds=time.monotonic() - started, timeout=timeout, exit_code=proc.returncode,
                  optimizer_address_space_limit_bytes=16 * 1024**3)
    (folder / "result.json").write_text(json.dumps(result, indent=2))
    print("profiling ended", result["wall_seconds"], "timeout", timeout, flush=True)


if __name__ == "__main__":
    main()
