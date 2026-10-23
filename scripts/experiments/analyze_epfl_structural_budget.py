#!/usr/bin/env python3
"""Structural witness/region estimates, not a mapped cover or a PPA lower bound."""
import argparse
import hashlib
import json
from pathlib import Path
from benchmark_suites import EPFL
from shared_subject import RECIPE


def read_binary_aig(data):
    header, tail = data.split(b"\n", 1)
    kind, maximum, inputs, latches, outputs, ands = header.decode().split()
    maximum, inputs, latches, outputs, ands = map(int, (maximum, inputs, latches, outputs, ands))
    if kind != "aig" or latches or maximum != inputs + ands:
        raise ValueError("expected canonical combinational binary AIGER")
    roots = []
    for _ in range(outputs):
        line, tail = tail.split(b"\n", 1)
        roots.append(int(line))
    position = 0

    def number():
        nonlocal position
        value, shift = 0, 0
        while True:
            if position >= len(tail) or shift > 63:
                raise ValueError("invalid delta encoding")
            byte = tail[position]
            position += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
            shift += 7

    gates = {}
    for variable in range(inputs + 1, maximum + 1):
        lhs = 2 * variable
        a = lhs - number()
        b = a - number()
        if not 0 <= b <= a < lhs:
            raise ValueError("non-topological AND")
        gates[variable] = (a, b)
    if any(not 0 <= literal <= 2 * maximum + 1 for literal in roots):
        raise ValueError("invalid output")
    return inputs, roots, gates


def estimate(data, node_limit):
    inputs, roots, gates = read_binary_aig(data)
    phases, live = {}, set()
    stack = list(roots)
    while stack:
        literal = stack.pop()
        variable = literal // 2
        phases.setdefault(variable, set()).add(literal & 1)
        if variable in gates and variable not in live:
            live.add(variable)
            stack.extend(gates[variable])
    both = sum(len(phases[v]) == 2 for v in live)
    pi_inverters = sum(v != 0 and v <= inputs and 1 in values for v, values in phases.items())
    # One AND2 or NAND2 per original AND plus an inverter when both phases are used.
    # This is a constructive gate-count estimate for this structure, not a minimum.
    witness = len(live) + both + pi_inverters
    regions, local, boundary = [], set(), set()

    def record():
        if local:
            regions.append({"ands": len(local), "boundary_variables": len(boundary),
                            "dual_phase_enode_upper_bound": 2 * (len(local) + len(boundary - {0}) + 1)})

    for variable in sorted(live):
        external = {literal // 2 for literal in gates[variable]} - local
        if local and 2 * (len(local) + 1 + len((boundary | external) - {0}) + 1) > node_limit:
            record()
            local, boundary = set(), set()
            external = {literal // 2 for literal in gates[variable]}
        if 2 * (len(local) + 1 + len((boundary | external) - {0}) + 1) > node_limit:
            raise ValueError("one gate region exceeds node limit")
        local.add(variable)
        boundary |= external
    record()
    return {"inputs": inputs, "outputs": len(roots), "live_original_ands": len(live),
            "both_phase_ands": both, "pi_inverters": pi_inverters,
            "structural_primitive_witness_estimate_excluding_ties": witness,
            "full_dual_phase_import_upper_bound": 2 * (inputs + len(live) + 1),
            "regions": regions, "peak_region_enode_upper_bound": max((r["dual_phase_enode_upper_bound"] for r in regions), default=0),
            "aggregate_region_enode_upper_bound": sum(r["dual_phase_enode_upper_bound"] for r in regions)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    source = json.loads(args.subjects.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    rows = []
    for row in source["results"]:
        bench = row["bench"]
        path = args.subjects.parent / f"{bench}.resyn2.aig"
        if row["status"] != "ok" or row["preprocessing"]["cec"] != 1 or row["preprocessing"]["recipe"] != RECIPE or sha(path) != row["preprocessing"]["sha256"]:
            raise ValueError("unverified subject")
        rows.append({"bench": bench, "subject_sha256": sha(path), **estimate(path.read_bytes(), 20000)})
    if {r["bench"] for r in rows} != set(EPFL):
        raise ValueError("incomplete EPFL set")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError("refusing to overwrite analysis")
    args.output.write_text(json.dumps({"subject_report_sha256": sha(args.subjects), "analyzer_sha256": sha(Path(__file__)),
        "scope": "structural feasibility estimates only; no library binding, mapped CEC, PPA or optimality claim",
        "budgets": {"nodes": 20000, "global_retained_instances": 100000, "global_recovery_secs": 2},
        "warning": "Region peak alone does not prove global budget compliance; aggregate work, candidates and one shared recovery deadline remain to implement. No per-region budget resets.",
        "results": rows}, indent=2))


if __name__ == "__main__":
    main()
