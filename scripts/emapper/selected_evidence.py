"""Read-only selected-generator/care replay, independent of native object layout.

This replays local truth/care and identity. Global root equivalence is still
checked by independent named-port CEC; no claim of a complete egg proof trace.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replay(manifest, objects, netjson=None):
    manifest, objects = Path(manifest), Path(objects)
    info = json.loads(manifest.read_text())
    meta = info.get('selected_evidence')
    if not meta:
        raise ValueError('no persisted selected evidence: utilization unknown')
    snapshot = manifest.parent / meta['file']
    if digest(snapshot) != meta['sha256']:
        raise ValueError('selected evidence snapshot hash mismatch')
    selected = json.loads(snapshot.read_text())
    cache = {}

    def load(key):
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('invalid content identity')
        if key not in cache:
            p = objects / (key + '.json')
            if digest(p) != key:
                raise ValueError('proof object hash mismatch')
            cache[key] = json.loads(p.read_text())
        return cache[key]

    logic = load(selected['logic'])
    nodes, aliases = logic['nodes'], logic['signed_aliases']
    care_seen = {}

    def image(key):
        if key in care_seen:
            return care_seen[key]
        p = load(key)
        if p['logic'] != selected['logic']:
            raise ValueError('care context mismatch')
        frontier = p['frontier']
        expanded = ({c: (nodes[c]['args'], nodes[c]['function']) for c in p['expanded']}
                    if isinstance(p['expanded'], list) else {int(k): v for k, v in p['expanded'].items()})
        if len(set(frontier)) != len(frontier) or set(frontier) & set(expanded):
            raise ValueError('invalid care frontier')
        for c, (args, function) in expanded.items():
            if nodes[c] != dict(args=args, function=function):
                raise ValueError('care plan differs from exact representative')
        bits = p['frontier_bits']
        if bits is None:
            bits = list(range(len(frontier)))
        else:
            if len(bits) != len(frontier):
                raise ValueError('care PI assignment layout mismatch')
            variables = [nodes[c]['pi'] for c in frontier]
            for i, a in enumerate(variables):
                for j, b in enumerate(variables):
                    if (a == b) != (bits[i] == bits[j]):
                        raise ValueError('correlated PI assignment changed')
        width = max(bits, default=-1) + 1
        if width > 20 or min(bits, default=0) < 0:
            raise ValueError('unsupported proof width')
        table = 0
        for assignment in range(1 << width):
            values = {c: (assignment >> bit) & 1 for c, bit in zip(frontier, bits)}
            pending = dict(expanded)
            while pending:
                ready = [c for c, (args, _) in pending.items() if all(a in values for a in args)]
                if not ready:
                    raise ValueError('care plan cyclic or not closed')
                for c in ready:
                    args, truth = pending.pop(c)
                    tuple_ = sum(values[a] << i for i, a in enumerate(args))
                    values[c] = (truth >> tuple_) & 1
            tuple_ = sum(values[c] << i for i, c in enumerate(p['boundary']))
            table |= 1 << tuple_
        if table != p['care']:
            raise ValueError('replayed care mask mismatch')
        care_seen[key] = p
        return p

    physical = {v['pool_instance']: load(v['object']) for v in selected['instances']}
    rows = selected['selected']
    if len({r['signal'] for r in rows}) != len(rows):
        raise ValueError('duplicate selected signal')
    exact, conditional = 0, 0
    for row in rows:
        m = physical[row['pool_instance']]
        output = m['outputs'][row['pin']]
        has_exact = False
        for key in row['proof_ids']:
            p = load(key)
            source, assignment = p['source'], p['source']['assignment']
            if p['logic'] != selected['logic'] or source['target'] != row['signal'] or p['output_pin'] != row['pin']:
                raise ValueError('selected generator context/target mismatch')
            # Dedup preserves the canonical function+timing identity, not
            # necessarily the first generator's raw pin order.
            order = sorted(range(len(m['inputs'])), key=lambda i: m['inputs'][i])
            functions = []
            for out in m['outputs']:
                table = [0] * 8
                for a in range(1 << len(order)):
                    old = sum(((a >> i) & 1) << pos for i, pos in enumerate(order))
                    wide = out['wide_function']
                    if wide is None:
                        value = (out['function'] >> old) & 1
                    else:
                        pos = sum(((old >> p) & 1) << i for i, p in enumerate(wide['positions']))
                        value = (wide['truth'][pos // 64] >> (pos % 64)) & 1
                    table[a // 64] |= value << (a % 64)
                truth = {'Small': table[0]} if len(order) <= 6 else {'Wide': table}
                arcs = [struct.unpack('<I', struct.pack('<f', out['delays'][i]))[0] for i in order]
                functions.append([out['name'], truth, arcs])
            physical_key = [m['cell'], [m['inputs'][i] for i in order], functions]
            if assignment['key'] != physical_key:
                raise ValueError('selected canonical function/pin-delay identity changed')
            if assignment['area_bits'] != m['area_bits']:
                raise ValueError('selected area identity changed')
            if sorted(aliases[u] for u in assignment['raw_inputs']) != sorted(m['inputs']):
                raise ValueError('selected physical input assignment changed')
            if p['physical_output'] != output['name'] or p['physical_function'] != output['function'] or p['delays'] != output['delays'] or p['wide_function'] != output['wide_function']:
                raise ValueError('selected physical output identity changed')
            if source['wide'] or source['inverter']:
                full_valid = True
                if source['inverter'] and (assignment['raw_inputs'] != [source['raw_target'] ^ 1] or output['function'] != 1):
                    raise ValueError('invalid phase-complement proof')
            else:
                boundary = source['boundary']
                if boundary != sorted(set(boundary)) or len(boundary) > 6:
                    raise ValueError('invalid ordered cut boundary')
                actual = 0
                order = sorted(range(len(assignment['raw_inputs'])), key=lambda i: aliases[assignment['raw_inputs'][i]])
                truth = assignment['key'][2][row['pin']][1]['Small']
                for a in range(1 << len(boundary)):
                    tuple_ = sum(((((a >> boundary.index(assignment['raw_inputs'][pos] // 2)) & 1) ^ (assignment['raw_inputs'][pos] & 1)) << i) for i, pos in enumerate(order))
                    actual |= ((truth >> tuple_) & 1) << a
                mismatch = (actual ^ source['function']) & ((1 << (1 << len(boundary))) - 1)
                full_valid = mismatch == 0
                if mismatch != p['full_care_mismatch'] or mismatch & source['care']:
                    raise ValueError('selected truth/care mismatch')
                if not full_valid:
                    proof = image(p['care_witness'])
                    if proof['boundary'] != boundary or proof['care'] != source['care']:
                        raise ValueError('selected care witness belongs to another boundary')
            if full_valid != p['full_care_valid']:
                raise ValueError('incorrect care-essential classification')
            has_exact |= full_valid
        if not row['proof_ids'] or row['requires_conditional_proof'] != (not has_exact):
            raise ValueError('selected proof set incomplete')
        exact += has_exact
        conditional += not has_exact
    if (exact, conditional) != (selected['exact_supported_bindings'], selected['conditional_only_bindings']):
        raise ValueError('selected classification totals mismatch')
    if netjson is not None:
        import re
        ports = selected['ports']
        top = json.loads(Path(netjson).read_text())['modules']['mapped']
        names = [r[2] for r in ports['inputs']] + [r[1] for r in ports['outputs']]
        escaped = [n if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*', n) else '\\' + n + ' ' for n in names]
        prefix = '__egm_'
        while any(n.startswith(prefix) for n in escaped):
            prefix += '_'
        if set(top['ports']) != set(names):
            raise ValueError('selected proof interface differs from delivered netlist')
        nets = {s: top['ports'][name]['bits'] for s, _, name in ports['inputs']}
        nets.update({s: ['1' if value else '0'] for s, value in ports['constants']})
        bound = {}
        for row in rows:
            bound.setdefault((row['pool_instance'], row['pin']), []).append(row['signal'])
        for signals in bound.values():
            bits = top['netnames'][prefix + 's' + str(min(signals))]['bits']
            for s in signals:
                nets[s] = bits
        expected_cells = set()
        for item in selected['instances']:
            i = item['pool_instance']
            name = prefix + 'g' + str(item['compact_instance'])
            expected_cells.add(name)
            cell, m = top['cells'][name], physical[i]
            if cell['type'] != m['cell']:
                raise ValueError('selected proof cell differs from delivered netlist')
            for pin, signal in zip(m['input_pins'], m['inputs']):
                if cell['connections'][pin] != nets[signal]:
                    raise ValueError('selected proof input pin differs from delivered netlist')
            for pin, out in enumerate(m['outputs']):
                signals = bound.get((i, pin), [])
                expected = nets[signals[0]] if signals else []
                if cell['connections'].get(out['name'], []) != expected:
                    raise ValueError('selected proof output differs from delivered netlist')
        if expected_cells != set(top['cells']):
            raise ValueError('delivered netlist has unaccounted physical instances')
        for signal, name in ports['outputs']:
            if top['ports'][name]['bits'] != nets[signal]:
                raise ValueError('selected proof primary output differs from delivered netlist')
    return dict(valid=True, netlist_binding_agreement=netjson is not None, selected_bindings=len(rows), exact_supported=exact,
                conditional_only=conditional, care_images_replayed=len(care_seen),
                proof_objects_read=len(cache), scope='local generator identity/truth and complete care replay; global equivalence requires CEC')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--objects', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--netjson', type=Path)
    a = p.parse_args()
    report = replay(a.manifest, a.objects, a.netjson)
    if a.output:
        with a.output.open('x') as f:
            json.dump(report, f, indent=2)
            f.write('\n')
    print(json.dumps(report, indent=2))
