#!/usr/bin/env python3
"""Convert a Lisp-style prefix expression file (one PO per line) into an EQN
file, then invoke ABC to read the EQN, structurally hash, and write a binary
AIGER file.

Prefix language (output of mig_egg simplify):
- Constants: 0, 1
- Variables: lowercase letter (a, b, ...) or i<N>
- Negation: (~ X)
- AND:      (& X Y)
- Majority: (M X Y Z)        ← expanded into (X*Y) + (X*Z) + (Y*Z) in EQN

Hash-consing on the resulting DAG keeps the EQN compact even when the e-graph
result reuses subexpressions; ABC's strash collapses any remaining redundancy.

Usage:
    prefix_to_aig.py PREFIX_FILE OUT_AIG --abc PATH_TO_ABC --n-inputs N
"""

import argparse
import os
import subprocess
import sys
import tempfile


def tokenize(s):
    return s.replace("(", " ( ").replace(")", " ) ").split()


def parse_one(tokens, pos):
    tok = tokens[pos]
    pos += 1
    if tok == "(":
        head = tokens[pos]
        pos += 1
        if head == "~":
            child, pos = parse_one(tokens, pos)
            assert tokens[pos] == ")"
            return ("~", child), pos + 1
        if head == "&":
            a, pos = parse_one(tokens, pos)
            b, pos = parse_one(tokens, pos)
            assert tokens[pos] == ")"
            return ("&", a, b), pos + 1
        if head == "M":
            a, pos = parse_one(tokens, pos)
            b, pos = parse_one(tokens, pos)
            c, pos = parse_one(tokens, pos)
            assert tokens[pos] == ")"
            return ("M", a, b, c), pos + 1
        raise ValueError(f"unknown op {head!r}")
    return tok, pos  # leaf: 0, 1, or variable name


def parse(s):
    tokens = tokenize(s)
    expr, pos = parse_one(tokens, 0)
    if pos != len(tokens):
        raise ValueError(f"trailing tokens after parse: {tokens[pos:]!r}")
    return expr


def to_eqn(roots, output_names, var_names):
    """Emit EQN body lines + final z = ... assignments.

    Returns: (header_lines, body_lines, output_lines)
    """
    cache = {}      # canonical expr -> rhs symbol (variable, !var, 0/1, or nN)
    body = []
    counter = [0]

    def fresh():
        counter[0] += 1
        return f"n{counter[0]}"

    def neg(s):
        if s == "0":
            return "1"
        if s == "1":
            return "0"
        return s[1:] if s.startswith("!") else f"!{s}"

    def and_fold(a, b):
        if a == "0" or b == "0":
            return "0"
        if a == "1":
            return b
        if b == "1":
            return a
        if a == b:
            return a
        if a == neg(b):
            return "0"
        n = fresh()
        body.append(f"{n} = {a} * {b};")
        return n

    def or_fold(a, b):
        if a == "1" or b == "1":
            return "1"
        if a == "0":
            return b
        if b == "0":
            return a
        if a == b:
            return a
        if a == neg(b):
            return "1"
        n = fresh()
        body.append(f"{n} = {a} + {b};")
        return n

    def visit(e):
        # Leaf?
        if isinstance(e, str):
            return e
        if e in cache:
            return cache[e]

        op = e[0]
        if op == "~":
            inner = visit(e[1])
            if inner == "0":
                rhs = "1"
            elif inner == "1":
                rhs = "0"
            elif inner.startswith("!"):
                rhs = inner[1:]
            else:
                rhs = f"!{inner}"
            cache[e] = rhs
            return rhs

        if op == "&":
            a = visit(e[1])
            b = visit(e[2])
            rhs = and_fold(a, b)
            cache[e] = rhs
            return rhs

        if op == "M":
            a = visit(e[1])
            b = visit(e[2])
            c = visit(e[3])
            # Majority constant identities:
            #   M(0, X, Y) = X ∧ Y;   M(1, X, Y) = X ∨ Y
            #   M(X, X, Y) = X;       M(X, Y, X) = X;  M(Y, X, X) = X
            if a == "0":
                rhs = and_fold(b, c)
            elif a == "1":
                rhs = or_fold(b, c)
            elif b == "0":
                rhs = and_fold(a, c)
            elif b == "1":
                rhs = or_fold(a, c)
            elif c == "0":
                rhs = and_fold(a, b)
            elif c == "1":
                rhs = or_fold(a, b)
            elif a == b or a == c:
                rhs = a
            elif b == c:
                rhs = b
            else:
                n_ab = fresh()
                n_ac = fresh()
                n_bc = fresh()
                n_or = fresh()
                body.append(f"{n_ab} = {a} * {b};")
                body.append(f"{n_ac} = {a} * {c};")
                body.append(f"{n_bc} = {b} * {c};")
                body.append(f"{n_or} = {n_ab} + {n_ac} + {n_bc};")
                rhs = n_or
            cache[e] = rhs
            return rhs

        raise ValueError(f"unknown op {op}")

    output_lines = []
    for name, root in zip(output_names, roots):
        rhs = visit(root)
        output_lines.append(f"{name} = {rhs};")

    header = [
        f"INORDER = {' '.join(var_names)};",
        f"OUTORDER = {' '.join(output_names)};",
    ]
    return header, body, output_lines


def collect_var_names(roots, n_inputs):
    """Return canonical input list a..z or i1..iN matching aig_to_prefix.py."""
    if n_inputs <= 26:
        return [chr(ord("a") + i) for i in range(n_inputs)]
    return [f"i{i + 1}" for i in range(n_inputs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix_file")
    ap.add_argument("out_aig")
    ap.add_argument("--abc", required=True, help="path to ABC binary")
    ap.add_argument("--n-inputs", type=int, required=True)
    ap.add_argument("--abc-rc-dir", required=True,
                    help="cwd when running ABC (must contain abc.rc)")
    ap.add_argument("--keep-eqn", action="store_true",
                    help="keep the intermediate .eqn next to out_aig for debug")
    args = ap.parse_args()

    # Read all PO lines, sort by PO index, parse
    pairs = []
    with open(args.prefix_file) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            idx_s, expr_s = line.split("\t", 1)
            pairs.append((int(idx_s), parse(expr_s)))
    pairs.sort()
    n_outputs = len(pairs)

    output_names = [f"z{idx}" for idx, _ in pairs]
    roots = [e for _, e in pairs]
    var_names = collect_var_names(roots, args.n_inputs)

    header, body, out_lines = to_eqn(roots, output_names, var_names)

    eqn_path = args.out_aig + ".eqn" if args.keep_eqn else \
        tempfile.NamedTemporaryFile(suffix=".eqn", delete=False).name
    with open(eqn_path, "w") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(body) + "\n")
        f.write("\n".join(out_lines) + "\n")

    cmd = [
        args.abc, "-c",
        f"read_eqn {eqn_path}; strash; print_stats; "
        f"resyn2; print_stats; "
        f"write_aiger {args.out_aig}",
    ]
    proc = subprocess.run(cmd, cwd=args.abc_rc_dir, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(f"ABC failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}\n")
        sys.exit(1)
    sys.stdout.write(proc.stdout)
    sys.stdout.write(proc.stderr)


if __name__ == "__main__":
    main()
