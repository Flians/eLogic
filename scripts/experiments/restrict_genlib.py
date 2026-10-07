#!/usr/bin/env python3
"""Write a copy of a genlib containing only gates with at most N inputs.

    scripts/experiments/restrict_genlib.py --in asap7.genlib --max-pins 6 --out asap7_40.genlib

Motivation: the comparison against ABC's `mfs3 -a` (paper Table `tab:mfs3`) needs a library `mfs3`
can handle. Its timing engine asserts `Abc_ObjFaninNum(node) <= 6` (`sfmTim.c:103`), so a 7-input
gate makes `mfs3 -a` segfault -- even on a network ABC mapped itself. ASAP7 minus the three 7-input
and the four 8/9-input gates leaves **40** cells, which is what the paper's baselines correspond to.
"""
import argparse
import sys


def read_blocks(lines):
    """Yield (gate_line, pin_lines) for every GATE block."""
    cur = None
    for line in lines:
        if line.startswith('GATE '):
            if cur:
                yield cur
            cur = [line]
        elif cur is not None:
            if line.strip().startswith('PIN'):
                cur.append(line)
            elif line.strip() == '':
                continue
            else:
                yield cur
                cur = None
    if cur:
        yield cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', dest='dst', required=True)
    ap.add_argument('--max-pins', type=int, default=6)
    a = ap.parse_args()

    lines = open(a.src).read().splitlines()
    kept, dropped, header = [], [], []
    for line in lines:
        if line.startswith('#'):
            header.append(line)
    for block in read_blocks(lines):
        pins = sum(1 for l in block if l.strip().startswith('PIN'))
        if pins <= a.max_pins:
            kept.extend(block)
        else:
            dropped.append((block[0].split()[1], pins))
    n_kept = sum(1 for l in kept if l.startswith('GATE '))
    with open(a.dst, 'w') as fh:
        fh.write(f"# {n_kept} gates (<= {a.max_pins} inputs, including constants) from {a.src}\n")
        fh.write("\n".join(kept) + "\n")
    print(f"[i] {a.dst}: kept {n_kept} gates, dropped {len(dropped)} "
          f"({', '.join(f'{n}({p})' for n, p in dropped) or 'none'})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
