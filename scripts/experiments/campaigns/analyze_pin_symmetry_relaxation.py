"""Read-only fixed-topology pin-symmetry relaxation, never a mapped result.

Enumerate exact function-preserving pin permutations of each single-output
library cell. Forward propagate the minimum possible unloaded arrival on the
existing cell DAG. No cuts, inversions, cell substitutions, or source rewriting.
This is a topology-restricted timing diagnostic: retained-pool eligibility is
not enforced and no implementation/CEC/PPA result is produced.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import itertools
import json
from pathlib import Path
import struct


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def f32(v):
    return struct.unpack("f", struct.pack("f", v))[0]


def symmetries(model):
    n = len(model["vars"])
    assert n <= 6
    truth = model["tt"][0] & ((1 << (1 << n))-1)
    result = []
    for permutation in itertools.permutations(range(n)):
        transformed = 0
        for bits in range(1 << n):
            original = sum(((bits >> permutation[i]) & 1) << i for i in range(n))
            transformed |= ((truth >> original) & 1) << bits
        if transformed == truth:
            result.append(permutation)
    assert tuple(range(n)) in result
    return result


def analyze(report_path, method):
    report = json.loads(report_path.read_text())
    row = next(r for r in report["results"] if r["method"] == method)
    assert row["status"] == "ok" and row["measured"]["cec"] == 1
    bench = row["bench"]
    mapped = report_path.parent / f"{bench}.{method}" / "mapped.json"
    library = report_path.parent / "library.json"
    models = defaultdict(list)
    for model in json.loads(library.read_text()):
        models[model["name"]].append(model)
    net = json.loads(mapped.read_text())["modules"]["mapped"]
    original, relaxed = {"0": 0., "1": 0.}, {"0": 0., "1": 0.}
    drivers, waiters = {}, defaultdict(list)
    permutations, delays, direct_gains = {}, {}, []
    area = 0.
    for port in net["ports"].values():
        if port["direction"] == "input":
            for bit in port["bits"]:
                original[bit] = relaxed[bit] = 0.
    for name, cell in net["cells"].items():
        assert len(models[cell["type"]]) == 1, "single-output scope only"
        model = models[cell["type"]][0]
        assert len(cell["connections"][model["out_pin"]]) == 1
        bit = cell["connections"][model["out_pin"]][0]
        assert bit not in drivers
        drivers[bit] = (name, cell, model)
        area += f32(model["area"])
        if cell["type"] not in permutations:
            permutations[cell["type"]] = symmetries(model)
            delays[cell["type"]] = [f32(model["pin_delay"].get(pin, 0.)) for pin in model["vars"]]
    missing, ready = {}, deque()
    for bit, (_, cell, model) in drivers.items():
        deps = {cell["connections"][pin][0] for pin in model["vars"]} - original.keys()
        missing[bit] = len(deps)
        if not deps:
            ready.append(bit)
        for dep in deps:
            waiters[dep].append(bit)
    visited, changed, largest_local_gain = 0, 0, 0.
    while ready:
        bit = ready.popleft()
        name, cell, model = drivers[bit]
        inputs = [cell["connections"][pin][0] for pin in model["vars"]]
        ds = delays[cell["type"]]
        original[bit] = max((original[b]+d for b,d in zip(inputs,ds)), default=0.)
        identity = max((relaxed[b]+d for b,d in zip(inputs,ds)), default=0.)
        best, choice = min((max((relaxed[inputs[permutation[i]]]+ds[i] for i in range(len(inputs))),default=0.), permutation)
                           for permutation in permutations[cell["type"]])
        relaxed[bit] = best
        gain = identity-best
        if gain > 1e-9:
            changed += 1
            largest_local_gain = max(largest_local_gain,gain)
            direct_gains.append(dict(cell=name,type=cell["type"],local_gain=gain,permutation=choice))
        visited += 1
        for user in waiters[bit]:
            missing[user] -= 1
            if not missing[user]:
                ready.append(user)
    assert visited == len(drivers), "cycle or undriven input"
    outputs = [bit for port in net["ports"].values() if port["direction"] == "output" for bit in port["bits"]]
    old, new = max(original[b] for b in outputs), max(relaxed[b] for b in outputs)
    assert abs(old-row["measured"]["delay"]) < 1e-6
    assert abs(area-row["measured"]["area"]) < 1e-6
    assert new <= old+1e-9
    return dict(bench=bench,method=method,original_area=area,original_delay=old,
                relaxed_fixed_topology_delay=new,potential_delay_reduction=old-new,
                improved_internal_nodes=changed,max_local_gain=largest_local_gain,
                largest_local_gains=sorted(direct_gains,key=lambda r:-r["local_gain"])[:20],
                original_input_sha256=row["input_sha256"],prepared_sha256=row["preprocessing"]["sha256"],
                library_sha256=report["library_sha256"],sources={str(p):sha(p) for p in [report_path,mapped,library]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,action="append",required=True)
    parser.add_argument("--method",action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    assert len(args.report)==len(args.method)
    assert not args.output.exists()
    records = [analyze(p,m) for p,m in zip(args.report,args.method)]
    args.output.write_text(json.dumps(dict(diagnostic_only=True,mapper_calls=0,netlist_exports=0,
        source_sha256=sha(Path(__file__)),limitation="Fixed topology; exact pin symmetries only. Does not enforce retained candidate eligibility, is not a same-budget mapped/CEC result, and says nothing about cuts, phase changes or different cells.",records=records),indent=2)+"\n")


if __name__=="__main__":
    main()
