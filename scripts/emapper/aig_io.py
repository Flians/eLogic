"""Named combinational AIGER input adapters; no optimization."""
import re

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

def symbols(data, inputs, outputs, gates):
    tail = data.split(b'\n', 1)[1]
    for _ in outputs:
        tail = tail.split(b'\n', 1)[1]
    pos = 0
    for _ in range(2 * len(gates)):
        while tail[pos] & 128:
            pos += 1
        pos += 1
    names = {'i': {}, 'o': {}}
    for line in tail[pos:].decode().splitlines():
        if line == 'c':
            break
        if line[:1] in names:
            index, name = line[1:].split(' ', 1)
            names[line[0]][int(index)] = name
    # Reject missing names: do not guess a port correspondence for named CEC.
    if set(names['i']) != set(range(inputs)) or set(names['o']) != set(range(len(outputs))):
        raise ValueError('requires complete named AIGER ports')
    return [names['i'][i] for i in range(inputs)], [names['o'][i] for i in range(len(outputs))]

def identifier(name):
    if not name or any(c.isspace() or c in "\\;\"" for c in name):
        raise ValueError("unsupported port identifier")
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", name) else "\\" + name + " "
