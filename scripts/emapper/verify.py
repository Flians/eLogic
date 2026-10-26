#!/usr/bin/env python3
"""Common physical metrics and named CEC via small functional-model BLIFs.

Only the checker's representation is converted. The physical netlist is never
remapped or optimized. Library BLIF truth comes from functional Verilog models
through Yosys, independently of mapper truth-table matching.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def named_golden(data):
    end = data.index(b"\n") + 1
    fields = data[:end].split()
    if len(fields) != 6 or fields[0] != b"aig":
        raise ValueError("requires combinational binary AIGER")
    maximum, inputs, latches, outputs, gates = map(int, fields[1:])
    if latches or min(maximum, inputs, outputs, gates) < 0:
        raise ValueError("unsupported AIGER header")
    for _ in range(outputs):
        end = data.index(b"\n", end) + 1
    for _ in range(2 * gates):
        while data[end] & 128:
            end += 1
        end += 1
    names = {"i": {}, "o": {}}
    for line in data[end:].decode().splitlines():
        if line == "c":
            break
        if line[:1] in names:
            index, value = line[1:].split(" ", 1)
            if int(index) in names[line[0]]:
                raise ValueError("duplicate AIGER port symbol")
            names[line[0]][int(index)] = value
    if set(names["i"]) != set(range(inputs)) or set(names["o"]) != set(range(outputs)):
        raise ValueError("complete named ports required; no positional guessing")
    ins, outs = [names["i"][i] for i in range(inputs)], [names["o"][i] for i in range(outputs)]
    if len(set(ins + outs)) != len(ins + outs):
        raise ValueError("duplicate original port names")
    symbols = [f"i{i} cec_i{i}" for i in range(inputs)] + [f"o{i} cec_o{i}" for i in range(outputs)]
    return data[:end] + ("\n".join(symbols) + "\nc\nverification symbols only\n").encode(), ins, outs, end


def top_blif(document, input_names, output_names, model_names):
    top = document["modules"]["mapped"]
    ports = top["ports"]
    if set(ports) != set(input_names + output_names):
        raise ValueError("mapped/original named port sets differ")
    token = re.compile(r"^[A-Za-z0-9_$.:/\[\]-]+$")

    def net(bit):
        if bit in ("0", "1"):
            return "constant" + bit
        if not isinstance(bit, int) or bit < 0:
            raise ValueError("unsupported unknown or non-scalar bit")
        return "wire" + str(bit)

    lines = [".model mapped", ".inputs " + " ".join(f"cec_i{i}" for i in range(len(input_names))),
             ".outputs " + " ".join(f"cec_o{i}" for i in range(len(output_names))),
             ".names constant0", ".names constant1", "1"]
    input_bits = []
    for i, name in enumerate(input_names):
        port = ports[name]
        if port["direction"] != "input" or len(port["bits"]) != 1:
            raise ValueError("input direction or scalar width mismatch")
        bit = port["bits"][0]
        if not isinstance(bit, int) or bit in input_bits:
            raise ValueError("constant or aliased primary input")
        input_bits.append(bit)
        lines += [f".names cec_i{i} {net(bit)}", "1 1"]
    for instance in top["cells"].values():
        name = instance["type"]
        if name not in model_names or not token.fullmatch(name):
            raise ValueError("unknown functional library model")
        pins = []
        for pin, bits in instance["connections"].items():
            if not token.fullmatch(pin) or len(bits) != 1:
                raise ValueError("unsupported library pin")
            pins.append(pin + "=" + net(bits[0]))
        lines.append(".subckt " + name + " " + " ".join(pins))
    for i, name in enumerate(output_names):
        port = ports[name]
        if port["direction"] != "output" or len(port["bits"]) != 1:
            raise ValueError("output direction or scalar width mismatch")
        lines += [f".names {net(port['bits'][0])} cec_o{i}", "1 1"]
    lines.append(".end")
    return "\n".join(lines) + "\n"


def verify(models, golden, netjson, folder):
    folder.mkdir(exist_ok=True)
    commands = []

    def run(argv, name):
        commands.append(argv)
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (folder / name).write_text(result.stdout)
        (folder / "commands.json").write_text(json.dumps(commands, indent=2))
        if result.returncode:
            raise ValueError("verification command failed: " + name)
        return result.stdout

    if any(c in str(models) for c in '\n\r;"'):
        raise ValueError("unsupported verification path")
    blif = folder / "functional_cells.blif"
    script = f'read_verilog "{models}"; proc; techmap; write_blif "{blif}"'
    run([os.environ.get("EGM_YOSYS", "yosys"), "-Q", "-T", "-p", script], "models.log")
    model_text = blif.read_text()
    model_names = set(re.findall(r"^\.model (\S+)", model_text, re.M))
    original = Path(golden).read_bytes()
    normalized, ins, outs, offset = named_golden(original)
    (folder / "golden.aig").write_bytes(normalized)
    (folder / "top.blif").write_text(top_blif(json.loads(Path(netjson).read_text()), ins, outs, model_names) + model_text)
    implementation = folder / "implementation.aig"
    command = f'read_blif "{folder / "top.blif"}"; strash; write_aiger -s "{implementation}"'
    abc = os.environ.get("BENCH_RS_EMAP_ABC", "abc")
    run([abc, "-c", command], "conversion.log")
    result = run([abc, "-c", f'cec "{folder / "golden.aig"}" "{implementation}"'], "cec.log")
    lower = result.lower()
    equivalent = "networks are equivalent" in lower and "not equivalent" not in lower
    prefix = hashlib.sha256(original[:offset]).hexdigest()
    assert hashlib.sha256(normalized[:offset]).hexdigest() == prefix
    (folder / "verification.json").write_text(json.dumps(dict(
        original_gate_bytes_unchanged=True, original_logic_prefix_sha256=prefix,
        functional_models_from_verilog_not_mapper_truth_tables=True,
        port_match="exact original names, direction and scalar width; golden only renames symbols",
        cec=int(equivalent), mapper_called=False), indent=2))
    return equivalent


def main():
    library, golden, mapped, netjson = map(Path, sys.argv[1:])
    measure = os.environ["EGM_MEASURE_BINARY"]
    result = subprocess.run([measure, str(library), str(netjson)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, check=True)
    metrics = json.loads(result.stdout)
    folder = mapped.parent / "functional-cec"
    good = verify(library.parent / "cells.v", golden, netjson, folder)
    metrics["cec"] = int(good)
    print(json.dumps(metrics, separators=(",", ":")), flush=True)
    if not good:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
