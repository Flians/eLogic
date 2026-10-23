"""Full single-output library range adapter. Author algorithms and K settings unchanged."""
import argparse
from pathlib import Path
import subprocess
import sys
import run_single_output_budget as delegated


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--full-evaluator',type=Path,required=True)
    parser.add_argument('--mode', choices=['wide-area-reserve'], required=True)
    parser.add_argument('--profile', choices=[0, 1], type=int, required=True)
    options, remaining = parser.parse_known_args()
    original = subprocess.Popen.__init__
    def init(proc, argv, *args, **kwargs):
        if isinstance(argv, (tuple, list)):
            argv = list(argv)
            if argv and Path(str(argv[0])).name == "bench_rs_netlist_eval":
                argv[0] = str(options.full_evaluator.resolve())
            if any(Path(str(x)).name == 'restrict_genlib.py' for x in argv) and '--max-pins' in argv:
                argv[argv.index('--max-pins') + 1] = '9'
            if argv and Path(str(argv[0])).name == 'egraphmap_scaled.py':
                environment = dict(kwargs['env'])
                environment.update(EGM_WIDE_MATCH='1', EGM_WIDE_INDEX_FILTER='1', EGM_WIDE_REUSE_POOL='1', EGM_WIDE_AREA_FLOW_FRACTION=('0.30' if options.profile==0 else '0.65'), EGM_WORD_TRUTH='1', EGM_FAST_ADMISSION_DELTA='1', EGM_ADMISSION_SHADOW='0', EGM_WORD_REDUCE='1', EGM_EXACT_CUT_OPS='1', EGM_EXCHANGE_TIME_FRACTION='0.70',
                                   EGM_FORCE_COMPACT_SOURCE='1', EGM_IMPORT_HELPER_FILL='0')
                kwargs['env'] = environment
        return original(proc, argv, *args, **kwargs)
    subprocess.Popen.__init__ = init
    sys.argv = [__file__, *remaining]
    try:
        delegated.main()
    finally:
        subprocess.Popen.__init__ = original

if __name__ == '__main__':
    main()
