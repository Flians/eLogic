"""Select the mapper's admission-counter implementation; keep the old clock."""
import argparse
from pathlib import Path
import subprocess
import sys

import run_single_output_budget as delegated


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fast-admission", choices=[0, 1], type=int, required=True)
    options, remaining = parser.parse_known_args()
    original = subprocess.Popen.__init__

    def init(proc, argv, *args, **kwargs):
        if isinstance(argv, (tuple, list)) and argv and Path(str(argv[0])).name == "egraphmap_scaled.py":
            environment = dict(kwargs["env"])
            environment["EGM_FAST_ADMISSION_DELTA"] = str(options.fast_admission)
            kwargs["env"] = environment
        return original(proc, argv, *args, **kwargs)

    subprocess.Popen.__init__ = init
    sys.argv = [__file__, *remaining]
    try:
        delegated.main()
    finally:
        subprocess.Popen.__init__ = original


if __name__ == "__main__":
    main()
