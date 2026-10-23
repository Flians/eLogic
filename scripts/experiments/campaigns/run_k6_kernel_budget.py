"""Select exact cut operations or recovery allocation; keep the original mapping clock."""
import argparse
from pathlib import Path
import subprocess
import sys

import run_single_output_budget as delegated


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", choices=[0,1],type=int,required=True)
    parser.add_argument("--mode",choices=["kernels","allocation"],required=True)
    options, remaining = parser.parse_known_args()
    original = subprocess.Popen.__init__

    def init(proc, argv, *args, **kwargs):
        if isinstance(argv, (tuple, list)) and argv and Path(str(argv[0])).name == "egraphmap_scaled.py":
            environment = dict(kwargs["env"])
            environment.update(EGM_FAST_ADMISSION_DELTA="1", EGM_ADMISSION_SHADOW="0", EGM_WORD_REDUCE="1",
                EGM_EXACT_CUT_OPS=str(options.profile) if options.mode=="kernels" else "1",
                EGM_EXCHANGE_TIME_FRACTION="0.70" if options.mode=="kernels" or options.profile==0 else "0.40")
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
