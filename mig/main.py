import os
import time
import networkx as nx

from vparser import parser, circuit_to_verilog
from util import process_not_buf, mig_cec
from mig_rewrite import rewrite_dp

import sys

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(SCRIPT_DIR, 'target/tools/'))
import MIGPy  # type: ignore

import argparse

benchmarks = [
    ['adder', 'arbiter', 'bar', 'cavlc', 'c432', 'c499', 'c1355', 'c6288', 'ctrl', 'dec', 'div', 'i2c', 'int2float', 'max', 'multiplier', 'priority', 'sin', 'sqrt'],
    ['full_adder_1', '4gt10', 'alu', 'c17', 'decoder_2_4', 'decoder_3_8', 'graycode4', 'ham3_28', 'mux_4'],
    ['4_49_7', 'graycode6_11', 'mod5adder_66', 'hwb8_64'] + [f'intdiv{i}' for i in range(4, 6)],
    [f'intdiv{i}' for i in range(6, 11)],
]

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--K", type=int, default=6, choices=(4, 6, 8), help="The input size of the feasible K-cut")
    argParser.add_argument("--obj", type=int, default=1, choices=(0, 1), help="0: depth, 1: area")
    argParser.add_argument("--benchmark", type=str, default='cavlc', help="the index of the benchmark")
    args = argParser.parse_args()

    obj_area = args.obj == 1
    benchmark = [args.benchmark]  # benchmarks[args.benchmark]
    print(f">>> The objective is {'area' if obj_area else 'depth'} oriented under K={args.K}-cuts.\n")

    for case in benchmark:
        output_dir = f'{SCRIPT_DIR}/results/{"area" if obj_area else "depth"}/{args.K}/{case}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        aigpath = f'tools/mockturtle/experiments/benchmarks/{case}.aig'
        vpath = f'{output_dir}/{case}.v'
        timer = time.time()
        init_cost = MIGPy.MIGReSub(aigpath, vpath)
        init_time = time.time() - timer
        cir = parser(vpath)
        process_not_buf(cir.graph)
        with open(f'{output_dir}/{case}_init.v', 'w', encoding='utf-8') as vfile:
            vfile.writelines(circuit_to_verilog(cir, behavioral=True))
        assert mig_cec(vpath, f'{output_dir}/{case}_init.v')
        nx.drawing.nx_agraph.write_dot(cir.graph, f'{output_dir}/{case}_init.dot')

        timer = time.time()
        rewrite_dp(cir.graph, K=args.K, obj_area=obj_area)
        rewrite_dp(cir.graph, K=args.K, obj_area=obj_area, independent=True)
        opt_time = time.time() - timer
        cir.remove_unloaded()
        with open(f'{output_dir}/{case}_opt.v', 'w', encoding='utf-8') as vfile:
            vfile.writelines(circuit_to_verilog(cir, behavioral=True))
        final_cost = MIGPy.MIGStatus(f'{output_dir}/{case}_opt.v')
        nx.drawing.nx_agraph.write_dot(cir.graph, f'{output_dir}/{case}_opt.dot')

        print(f'\n--- The results of {case} ---')
        print(f"initial cost {init_cost} with {init_time:.2f} seconds")
        print(f"final cost {final_cost} with {opt_time:.2f} seconds\n")
        assert mig_cec(vpath, f'{output_dir}/{case}_opt.v')
