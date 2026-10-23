#!/usr/bin/env python3
"""Harness-only limit on direct mapping commands; not a whole-flow deadline.

Native E-morphic SA feedback calls use a shared union-of-active-intervals clock.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
from nested_mapping_clock import Clock


class MappingBudget:
    def __init__(self, seconds):
        self.seconds = seconds
        self.states = {}
        self.current = "shared-setup"
        self.state(self.current)

    def state(self, key):
        if key not in self.states:
            self.states[key] = {"started": time.monotonic(), "excluded_other_secs": 0.,
                                "mapping_wall_secs": 0.,
                                "exhausted": False, "commands": []}
        return self.states[key]

    def used(self, key):
        return self.states[key]["mapping_wall_secs"]

    def remaining(self, key):
        return max(0., self.seconds - self.used(key))

    def alarm(self, *_):
        self.states[self.current]["exhausted"] = True
        raise RuntimeError("mapping wall budget exhausted (independent CEC excluded)")

    def arm(self):
        signal.setitimer(signal.ITIMER_REAL, max(.000001, self.remaining(self.current)))


def verification(argv):
    # The existing evaluator combines metrics and named-port CEC. Exclude that
    # complete verification/reporting call, not normalization or native feedback.
    if isinstance(argv, (str, bytes)):
        return False
    if Path(str(argv[0])).name == "bench_rs_netlist_eval":
        return len(argv) > 1 and "--models" not in argv
    return (Path(str(argv[0])).name == "abc" and "-c" in argv and
            str(argv[argv.index("-c") + 1]).strip().startswith("cec "))


def mapping(argv):
    if isinstance(argv, (str, bytes)):
        return False
    name = Path(str(argv[0])).name
    if name in ("bench_rs_egraph_map", "baseline_mapper", "eqmap_fpga"):
        return True
    if name == "abc" and "-c" in argv:
        script = str(argv[argv.index("-c") + 1])
        # Match actual ABC commands, not file names or preprocessing rewrite.
        return any(part.strip().split()[:1] in (["map"], ["amap"], ["&nf"])
                   for part in script.split(";"))
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--driver", type=Path, required=True)
    ap.add_argument("--mapping-seconds", type=float, default=300)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.mapping_seconds <= 0:
        ap.error("mapping-seconds must be positive")
    driver_args = a.args[1:] if a.args[:1] == ["--"] else a.args
    driver = a.driver.resolve()
    nested = driver.name == "bench_emorphic_sa.py"
    clock_path = a.audit.resolve().with_suffix(".clock.json")
    if nested and clock_path.exists():
        ap.error("existing nested clock: preserve interrupted output rather than overwrite")
    budget = MappingBudget(a.mapping_seconds)
    original = subprocess.Popen.communicate
    original_init = subprocess.Popen.__init__
    original_read_text = Path.read_text
    capture = driver.parent / "capture_emorphic_abc.py"

    def context_key():
        frame = inspect.currentframe().f_back
        while frame:
            row = frame.f_locals.get("row")
            if frame.f_code.co_filename == str(driver) and isinstance(row, dict) and "method" in row and "bench" in row:
                return row["bench"] + "/" + row["method"]
            frame = frame.f_back
        return "shared-setup"

    def init(process, *args, **kwargs):
        if nested:
            env = dict(kwargs.get("env") or os.environ)
            env.update(MAPPING_CLOCK_PATH=str(clock_path), MAPPING_CLOCK_SECONDS=str(a.mapping_seconds),
                       MAPPING_CLOCK_CASE=context_key())
            kwargs["env"] = env
        return original_init(process, *args, **kwargs)

    def read_text(path, *args, **kwargs):
        text = original_read_text(path, *args, **kwargs)
        if nested and path.resolve() == capture:
            # Instrument the staging launcher only; keep native source, ABC
            # argv, model, candidate selection and archive code unchanged.
            prefix = ("import sys\nsys.path.insert(0, " + repr(str(Path(__file__).resolve().parent)) + ")\n"
                      "from nested_mapping_clock import install_capture_clock\ninstall_capture_clock()\n")
            return prefix + text
        return text

    def communicate(process, *args, **kwargs):
        frame = inspect.currentframe().f_back
        while frame:
            row = frame.f_locals.get("row")
            if frame.f_code.co_filename == str(driver) and isinstance(row, dict) and "method" in row and "bench" in row:
                new_key = row["bench"] + "/" + row["method"]
                if new_key != budget.current:
                    budget.states[budget.current]["finished"] = time.monotonic()
                    budget.current = new_key
                    budget.state(new_key)
                break
            frame = frame.f_back
        key = budget.current
        charged = mapping(process.args)
        excluded = not charged
        signal.setitimer(signal.ITIMER_REAL, 0)
        start = time.monotonic()
        # Preserve existing process limits; only shorten mapping calls at the
        # remaining cumulative wall budget. CEC retains its original timeout.
        positional = list(args)
        existing = positional[1] if len(positional) > 1 else kwargs.get("timeout")
        cleanup = getattr(process, "_mapping_budget_cleanup", False)
        if not excluded and not cleanup:
            remaining = budget.remaining(key)
            clipped = remaining if existing is None else min(existing, remaining)
            if len(positional) > 1:
                positional[1] = clipped
            else:
                kwargs["timeout"] = clipped
        completed = False
        try:
            result = original(process, *positional, **kwargs)
            completed = True
            return result
        except subprocess.TimeoutExpired:
            process._mapping_budget_cleanup = True
            raise
        finally:
            elapsed = time.monotonic() - start
            if excluded:
                budget.states[key]["excluded_other_secs"] += elapsed
            else:
                budget.states[key]["mapping_wall_secs"] += elapsed
            budget.states[key]["commands"].append({"argv": process.args,
                "mapping_charged": charged, "wall_secs": elapsed})
            if budget.used(key) >= budget.seconds:
                budget.states[key]["exhausted"] = True
            # Leave the driver's TimeoutExpired handler enough opportunity to
            # kill its process group and drain output before raising a wall error.
            # Python adapters, input conversion and reporting are outside the
            # mapping clock. The next charged process gets only its remainder.

    old_handler = signal.signal(signal.SIGALRM, budget.alarm)
    subprocess.Popen.communicate = communicate
    subprocess.Popen.__init__ = init
    Path.read_text = read_text
    sys.path.insert(0, str(driver.parent))
    sys.argv = [str(driver), *driver_args]
    code = 0
    # No whole-flow alarm: only mapping subprocesses consume this allowance.
    try:
        runpy.run_path(str(driver), run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except RuntimeError as e:
        code = 1
        print(str(e), file=sys.stderr)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        subprocess.Popen.communicate = original
        subprocess.Popen.__init__ = original_init
        Path.read_text = original_read_text
        signal.signal(signal.SIGALRM, old_handler)
        budget.states[budget.current]["finished"] = time.monotonic()
        for key, state in budget.states.items():
            state["mapping_wall_secs"] = budget.used(key)
        # Freeze elapsed values at completion rather than charging later cases
        # to previously finished cases in the audit.
        audit = {"policy": f"{a.mapping_seconds:g}s direct mapping-command wall per design/method; other stages excluded",
            "mapping_seconds": a.mapping_seconds, "native_parameters_changed": False,
            "verification_scope": "resyn2, adapters, CEC and metrics excluded; nested SA actual ABC feedback mapping charged, other SA work excluded",
            "driver": str(driver), "argv": driver_args,
            "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
            "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "states": budget.states, "exit_code": code}
        if nested and clock_path.exists():
            clock = Clock(clock_path, a.mapping_seconds)
            for key in list(json.loads(clock_path.read_text())["cases"]):
                clock.update(key, "snapshot")
            audit["nested_mapping_clock"] = json.loads(clock_path.read_text())
            audit["nested_clock_sha256"] = hashlib.sha256(
                Path(__file__).with_name("nested_mapping_clock.py").read_bytes()).hexdigest()
            for key, s in audit["nested_mapping_clock"]["cases"].items():
                state = budget.state(key)
                state["mapping_wall_secs"] += s["used"]
                # The uncharged outer SA command contains the metered ABC
                # intervals. Remove that overlap from its non-mapping total.
                state["excluded_other_secs"] = max(0., state["excluded_other_secs"] - s["used"])
                state["nested_parent_overlap_removed_secs"] = s["used"]
                state["exhausted"] |= s["exhausted"] or s["incomplete"] or bool(s["active"])
        # A partial incumbent after exhaustion must never be exported as a
        # completed successful case, even if a driver catches a timeout.
        if "--output" in driver_args:
            report_path = Path(driver_args[driver_args.index("--output") + 1]) / "report.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                report["wall_budget_policy"] = {k: audit[k] for k in
                    ["mapping_seconds", "verification_scope", "wrapper_sha256"]}
                for row in report.get("results", []):
                    state = budget.states.get(row["bench"] + "/" + row["method"], {})
                    if state.get("exhausted") or budget.states["shared-setup"]["exhausted"]:
                        row["partial_status"] = row.get("status")
                        if "measured" in row:
                            row["partial_measured"] = row.pop("measured")
                        row.update(status="failed", error=f"{a.mapping_seconds:g}s cumulative mapping wall timeout; CEC excluded")
                        code = 1
                report_path.write_text(json.dumps(report, indent=2) + "\n")
        audit["exit_code"] = code
        a.audit.parent.mkdir(parents=True, exist_ok=True)
        a.audit.write_text(json.dumps(audit, indent=2) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
