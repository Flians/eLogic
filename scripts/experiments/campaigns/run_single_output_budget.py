"""Library-range and timing adapter; author algorithms and driver files unchanged.

The two historical E-Syn/E-morphic drivers accepted only K3/K4. Widen only
that argparse choice to K6. Keep their original filename/stack frames so the
existing nested mapping clock continues to attribute feedback calls correctly.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import run_with_mapping_budget as budget


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scaled", action="store_true")
    parser.add_argument("--word-truth", type=int, choices=[0, 1], default=0)
    parser.add_argument("--parse-only", action="store_true")
    options, remaining = parser.parse_known_args()
    add_argument = argparse.ArgumentParser.add_argument
    parse_args = argparse.ArgumentParser.parse_args

    def add(self, *flags, **kwargs):
        if "--k" in flags and kwargs.get("choices") == [3, 4]:
            kwargs["choices"] = [3, 4, 6]
        return add_argument(self, *flags, **kwargs)

    class Parsed(Exception):
        pass

    def parse(self, *args, **kwargs):
        value = parse_args(self, *args, **kwargs)
        if options.parse_only and hasattr(value, "k"):
            print(json.dumps({"k": value.k, "parsed_before_native_execution": True}))
            raise Parsed()
        return value

    argparse.ArgumentParser.add_argument = add
    argparse.ArgumentParser.parse_args = parse
    original_init = subprocess.Popen.__init__

    def init(proc, argv, *args, **kwargs):
        if isinstance(argv, (list, tuple)) and argv and Path(str(argv[0])).name == "egraphmap_scaled.py":
            # Inject only into our mapper frontend, never any baseline executable.
            environment = dict(kwargs["env"])
            environment["EGM_WORD_TRUTH"] = str(options.word_truth)
            kwargs["env"] = environment
        return original_init(proc, argv, *args, **kwargs)

    subprocess.Popen.__init__ = init
    if options.scaled:
        original_mapping, original_verification = budget.mapping, budget.verification
        budget.mapping = lambda argv: (isinstance(argv, (list, tuple)) and bool(argv) and
            Path(str(argv[0])).name == "egraphmap_scaled.py") or original_mapping(argv)
        budget.verification = lambda argv: (isinstance(argv, (list, tuple)) and bool(argv) and
            Path(str(argv[0])).name == "fast_original_cec.py") or original_verification(argv)
    sys.argv = [__file__, *remaining]
    try:
        budget.main()
    except Parsed:
        return
    finally:
        argparse.ArgumentParser.add_argument = add_argument
        argparse.ArgumentParser.parse_args = parse_args
        subprocess.Popen.__init__ = original_init


if __name__ == "__main__":
    main()
