#!/usr/bin/env python3
"""Parse a binary AIGER file (no latches, no symbol table) and emit a Lisp
prefix expression per primary output, in the form expected by mig_egg's MIG
language: `&` for AND, `~` for NOT, lowercase letters or `i<N>` for vars,
`0`/`1` for constants.

Usage: aig_to_prefix.py <file.aig> [--po N]   (default: all POs, one per line)

Output line format: `<po_index>\t<prefix_expr>\n`
"""

import sys
import argparse


def read_delta(buf, pos):
    """Read variable-length delta (LEB128-ish) from binary AIGER body."""
    n = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise EOFError("unexpected EOF in AIG body")
        b = buf[pos]
        pos += 1
        n |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return n, pos
        shift += 7


def parse_aig(path):
    raw = open(path, "rb").read()
    nl = raw.index(b"\n")
    header = raw[:nl].decode()
    parts = header.split()
    assert parts[0] == "aig", f"not a binary aig: {header!r}"
    M, I, L, O, A = (int(x) for x in parts[1:6])
    assert L == 0, "this script doesn't handle latches"

    pos = nl + 1
    outputs = []
    for _ in range(O):
        nl2 = raw.index(b"\n", pos)
        outputs.append(int(raw[pos:nl2]))
        pos = nl2 + 1

    # AND gates: implicit lhs = 2*(I+L+1+i)
    ands = {}  # var -> (rhs0_lit, rhs1_lit)
    for i in range(A):
        lhs_var = I + L + 1 + i
        lhs_lit = 2 * lhs_var
        d0, pos = read_delta(raw, pos)
        d1, pos = read_delta(raw, pos)
        rhs0 = lhs_lit - d0
        rhs1 = rhs0 - d1
        ands[lhs_var] = (rhs0, rhs1)

    return I, O, outputs, ands


def var_name(v, n_inputs):
    """Variable id v (1..I) → letter a..z or i<N> if too many."""
    if v < 1 or v > n_inputs:
        raise ValueError(f"var {v} out of input range")
    if n_inputs <= 26:
        return chr(ord("a") + v - 1)
    return f"i{v}"


def lit_to_prefix(lit, ands, n_inputs, memo):
    """Convert AIG literal to Lisp prefix expression.

    lit even -> positive of var(lit//2);  lit odd -> negation.
    var 0 -> constant FALSE; with inversion, TRUE.
    """
    if lit in memo:
        return memo[lit]
    var = lit // 2
    inverted = lit & 1

    if var == 0:
        s = "1" if inverted else "0"
    elif var <= n_inputs:
        name = var_name(var, n_inputs)
        s = f"(~ {name})" if inverted else name
    else:
        rhs0, rhs1 = ands[var]
        e0 = lit_to_prefix(rhs0, ands, n_inputs, memo)
        e1 = lit_to_prefix(rhs1, ands, n_inputs, memo)
        and_expr = f"(& {e0} {e1})"
        s = f"(~ {and_expr})" if inverted else and_expr

    memo[lit] = s
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aig")
    ap.add_argument("--po", type=int, default=None,
                    help="emit only this PO index (default: all)")
    ap.add_argument("--as-mig", action="store_true",
                    help="emit (M 0 a b) instead of (& a b) so the e-graph "
                         "Majority rewrite rules can fire")
    args = ap.parse_args()

    n_inputs, n_outputs, outputs, ands = parse_aig(args.aig)
    sys.stderr.write(f"# inputs={n_inputs} outputs={n_outputs} ands={len(ands)}\n")

    memo = {}
    for i, out_lit in enumerate(outputs):
        if args.po is not None and i != args.po:
            continue
        expr = lit_to_prefix(out_lit, ands, n_inputs, memo)
        if args.as_mig:
            # AND a b ≡ Majority(0, a, b). Pure string substitution is safe
            # because `(& ` only ever appears as a syntactic AND opener.
            expr = expr.replace("(& ", "(M 0 ")
        sys.stdout.write(f"{i}\t{expr}\n")


if __name__ == "__main__":
    main()
