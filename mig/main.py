import copy
import re
import os
import math
import time
import queue
import circuitgraph as cg
from typing import Tuple
from collections import defaultdict
import networkx as nx

import mig_egg
from vparser import parser, circuit_to_verilog
from util import process_not_buf, mig_cec
from mig_rewrite import kcuts_kcones_PIs_POs, rewrite_dp
from eggexpr import graph_to_egg_expr, graph_from_egg_expr

import sys

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(SCRIPT_DIR, 'target/tools/'))
import MIGPy  # type: ignore

import argparse

benchmarks = [
    ['adder1', 'c17', 'adder', 'arbiter', 'bar', 'c432', 'c499'],
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
        output_dir = f'{SCRIPT_DIR}/results/{case}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        aigpath = f'tools/mockturtle/experiments/benchmarks/{case}.aig'
        vpath = f'{output_dir}/{case}.v'
        init_cost = MIGPy.MIGReSub(aigpath, vpath)
        cir = parser(vpath)
        process_not_buf(cir.graph)
        with open(f'{output_dir}/{case}_init.v', 'w', encoding='utf-8') as vfile:
            vfile.writelines(circuit_to_verilog(cir, behavioral=True))
        assert mig_cec(vpath, f'{output_dir}/{case}_init.v')
        nx.drawing.nx_agraph.write_dot(cir.graph, f'{output_dir}/{case}_init.dot')

        timer = time.time()
        rewrite_dp(cir.graph, K=8)
        cir.remove_unloaded()
        with open(f'{output_dir}/{case}_opt.v', 'w', encoding='utf-8') as vfile:
            vfile.writelines(circuit_to_verilog(cir, behavioral=True))
        final_cost = MIGPy.MIGStatus(f'{output_dir}/{case}_opt.v')
        nx.drawing.nx_agraph.write_dot(cir.graph, f'{output_dir}/{case}_opt.dot')

        print(f'\nThe results of {case}:')
        print(f"initial cost {init_cost}")
        print(f"final cost {final_cost}")
        print("--- Total %.2f seconds ---\n" % (time.time() - timer))
        assert mig_cec(vpath, f'{output_dir}/{case}_opt.v')
