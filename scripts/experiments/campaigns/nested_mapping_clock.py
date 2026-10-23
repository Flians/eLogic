"""Shared union-of-active-intervals clock for concurrent ABC mapping feedback."""
import fcntl
import json
import os
from pathlib import Path
import time
import uuid


class Clock:
    def __init__(self, path, limit):
        self.path, self.limit = Path(path), float(limit)

    def update(self, key, action, token=None, argv=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            text = f.read()
            data = json.loads(text) if text else {"mapping_seconds": self.limit, "cases": {}}
            if data["mapping_seconds"] != self.limit:
                raise RuntimeError("mapping clock policy changed")
            now = time.monotonic()
            s = data["cases"].setdefault(key, {"used": 0., "last": now, "active": {},
                                                "exhausted": False, "events": [], "incomplete": False})
            if s["active"]:
                s["used"] += now - s["last"]
            s["last"] = now
            # A worker killed by the original driver's process-group watchdog
            # cannot close its interval. Preserve uncertainty and reject the case.
            for t, owner in list(s["active"].items()):
                try:
                    os.kill(owner, 0)
                except ProcessLookupError:
                    s["incomplete"] = True
                    del s["active"][t]
            remaining = max(0., self.limit - s["used"])
            if remaining == 0:
                s["exhausted"] = True
            if action == "begin" and remaining > 0:
                s["active"][token] = os.getpid()
                s["events"].append({"event": "begin", "token": token, "at": now, "argv": argv})
            elif action == "end":
                s["active"].pop(token, None)
                s["events"].append({"event": "end", "token": token, "at": now})
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
            f.flush()
            return remaining

    def begin(self, key, argv):
        token = uuid.uuid4().hex
        return token, self.update(key, "begin", token, list(argv))

    def end(self, key, token):
        self.update(key, "end", token)


def install_capture_clock():
    """Wrap only the real ABC call inside the unchanged archive launcher."""
    import subprocess
    original = subprocess.run
    clock = Clock(os.environ["MAPPING_CLOCK_PATH"], os.environ["MAPPING_CLOCK_SECONDS"])
    key = os.environ["MAPPING_CLOCK_CASE"]

    def run(argv, *args, **kwargs):
        if str(argv[0]) != os.environ["EMORPHIC_REAL_ABC"]:
            return original(argv, *args, **kwargs)
        token, remaining = clock.begin(key, argv)
        if remaining <= 0:
            return subprocess.CompletedProcess(argv, 124, b"", b"mapping budget exhausted\n")
        timeout = kwargs.get("timeout")
        kwargs["timeout"] = remaining if timeout is None else min(timeout, remaining)
        try:
            return original(argv, *args, **kwargs)
        except subprocess.TimeoutExpired as e:
            # Keep the original launcher's archive path and stdout/stderr even
            # when the budget kills ABC. Native SA receives a real failed call.
            return subprocess.CompletedProcess(argv, 124, e.stdout or b"", e.stderr or b"")
        finally:
            clock.end(key, token)

    subprocess.run = run
