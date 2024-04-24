import copy
import re
import os
import math
import time
import queue
import circuitgraph as cg
from typing import Tuple
from collections import defaultdict

from vparser import parser, circuit_to_verilog

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '../target/tools/'))
import MIGPy  # type: ignore

import argparse

benchmarks = [
    ['c17'],
    ['full_adder_1', '4gt10', 'alu', 'c17', 'decoder_2_4', 'decoder_3_8', 'graycode4', 'ham3_28', 'mux_4'],
    ['4_49_7', 'graycode6_11', 'mod5adder_66', 'hwb8_64'] + [f'intdiv{i}' for i in range(4, 6)],
    [f'intdiv{i}' for i in range(6, 11)],
]

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--generation", type=int, default=50000000, help="the number of generation")
    argParser.add_argument("--optimized", type=int, default=2, choices=(0, 1, 2), help="0: MIGAQFPReSyn, 1: original CGP, 2: RCGP")
    argParser.add_argument("--benchmark", type=int, default=0, choices=(0, 1, 2), help="the index of the benchmark")
    args = argParser.parse_args()

    optimized = args.optimized
    generation = args.generation
    benchmark = benchmarks[args.benchmark]

    for case in benchmark:
        output_dir = f'results/{case}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        timer = time.time()
        aigpath = f'tests/{case}.aig'
        vpath = f'{output_dir}/{case}.v'
        MIGPy.MIGReSyn(aigpath, vpath)
        cir = parser(vpath)
        with open(f'{output_dir}/{case}_init.v', 'w', encoding='utf-8') as vfile:
            vfile.writelines(circuit_to_verilog(cir))

        print("--- Total %.2f seconds ---\n" % (time.time() - timer))
