"""Root-output frontier completion single factor; same native binary and mapping clock."""
import argparse
from pathlib import Path
import subprocess
import sys
import run_single_output_budget as delegated


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile",choices=[0,1],type=int,required=True)
    parser.add_argument("--mode",choices=["root-frontier"],required=True)
    options,remaining=parser.parse_known_args()
    original=subprocess.Popen.__init__
    def init(proc,argv,*args,**kwargs):
        if isinstance(argv,(tuple,list)) and argv and Path(str(argv[0])).name=="egraphmap_scaled.py":
            environment=dict(kwargs["env"])
            environment.update(EGM_FAST_ADMISSION_DELTA="1",EGM_ADMISSION_SHADOW="0",EGM_WORD_REDUCE="1",EGM_EXACT_CUT_OPS="1",EGM_EXCHANGE_TIME_FRACTION="0.70",EGM_FORCE_COMPACT_SOURCE="1",EGM_WORD_QUOTIENT="0",EGM_FAST_TT3_SUPPORT="0",EGM_ROOT_FRONTIER_STOP=str(options.profile))
            kwargs["env"]=environment
        return original(proc,argv,*args,**kwargs)
    subprocess.Popen.__init__=init
    sys.argv=[__file__,*remaining]
    try:delegated.main()
    finally:subprocess.Popen.__init__=original


if __name__=="__main__":main()
