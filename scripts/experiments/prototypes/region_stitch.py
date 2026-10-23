"""Logical region/stitch prototype. This does not perform technology mapping."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_epfl_structural_budget import read_binary_aig
from port_identifiers import identifier


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


def stitch(data, local_node_limit=20000, global_node_limit=20000, global_pool_limit=100000):
    inputs, roots, gates = read_binary_aig(data)
    input_names, output_names = symbols(data, inputs, roots, gates)
    live = set()
    stack = list(roots)
    while stack:
        v = stack.pop() // 2
        if v in gates and v not in live:
            live.add(v)
            stack.extend(gates[v])
    regions, local, boundary = [], set(), set()
    for v in sorted(live):
        external = {x // 2 for x in gates[v]} - local
        if local and 2 * (len(local) + 1 + len((boundary | external) - {0}) + 1) > local_node_limit:
            regions.append((local, boundary - {0}))
            local, boundary = set(), set()
            external = {x // 2 for x in gates[v]}
        if 2 * (len(local) + 1 + len((boundary | external) - {0}) + 1) > local_node_limit:
            raise ValueError('local import bound exceeded')
        local.add(v)
        boundary |= external
    if local:
        regions.append((local, boundary - {0}))
    # One global conservative accounting, including repeated boundary nodes and
    # each region's Zero/NotZero pair. No per-region reset of global allowances.
    counted_nodes = sum(2 * (len(a) + len(b) + 1) for a, b in regions)
    # Constructive AND2 + shared inverter upper bound. Complemented output
    # roots, including primary-input roots, need to be counted as well.
    negative_vars = {x // 2 for x in roots if x > 1 and x & 1}
    negative_vars.update(x // 2 for v in live for x in gates[v] if x > 1 and x & 1)
    primitive_candidates = len(live) + len(negative_vars) + 2
    if counted_nodes > global_node_limit:
        raise ValueError(f'global import bound {counted_nodes} exceeds {global_node_limit}')
    if primitive_candidates > global_pool_limit:
        raise ValueError('global primitive candidate upper bound exceeded')
    producers = {v: i for i, (a, _) in enumerate(regions) for v in a}
    exports = [set() for _ in regions]
    for _, boundary in regions:
        for v in boundary:
            if v in producers:
                exports[producers[v]].add(v)
    for literal in roots:
        if literal // 2 in producers:
            exports[producers[literal // 2]].add(literal // 2)

    def lit(x):
        if x < 2:
            return "1'b1" if x else "1'b0"
        return ('~' if x & 1 else '') + f'n{x // 2}'

    text = []
    for i, (a, b) in enumerate(regions):
        ports = [f'input n{v}' for v in sorted(b)] + [f'output n{v}' for v in sorted(exports[i])]
        text.append(f'module region_{i}({", ".join(ports)});')
        for v in sorted(a):
            if v not in exports[i]:
                text.append(f'wire n{v};')
            x, y = gates[v]
            text.append(f'assign n{v} = {lit(x)} & {lit(y)};')
        text.append('endmodule')
    ports = ['input ' + identifier(x) for x in input_names] + ['output ' + identifier(x) for x in output_names]
    text.append('module mapped(' + ', '.join(ports) + ');')
    used = {x // 2 for x in roots} | set().union(*(b | exports[i] for i, (_, b) in enumerate(regions)))
    for v in sorted(used - {0}):
        text.append(f'wire n{v};')
    for i, name in enumerate(input_names, 1):
        if i in used:
            text.append(f'assign n{i} = {identifier(name)};')
    for i, (_, b) in enumerate(regions):
        pins = sorted(b | exports[i])
        text.append(f'region_{i} r{i}(' + ', '.join(f'.n{v}(n{v})' for v in pins) + ');')
    for name, root in zip(output_names, roots):
        text.append(f'assign {identifier(name)} = {lit(root)};')
    text.append('endmodule')
    return '\n'.join(text) + '\n', {'regions': len(regions), 'live_original_ands': len(live),
        'counted_import_upper_bound': counted_nodes, 'primitive_candidate_upper_bound': primitive_candidates,
        'scope': 'logical stitching only; no egraph run, physical cover, PPA or recovery certificate'}
