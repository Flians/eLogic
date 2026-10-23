"""Library-matched primitive region witness; no egraph saturation or optimization."""
import json
import re
import time
from pathlib import Path
from region_stitch import stitch, read_binary_aig, identifier


def physical_seed(data, library_json, local_node_limit=20000, global_node_limit=20000,
                  global_pool_limit=100000, recovery_seconds=2, negative_base_nodes=()):
    start = time.perf_counter()

    def deadline():
        if time.perf_counter() - start > recovery_seconds:
            raise ValueError('global primitive witness construction deadline exceeded')

    if recovery_seconds <= 0:
        raise ValueError('global construction deadline exhausted')
    logic, stats = stitch(data, local_node_limit, global_node_limit, global_pool_limit)
    inputs, roots, gates = read_binary_aig(data)
    # Import the actual common evaluator's library functions, area and pins.
    cells = json.loads(Path(library_json).read_text())
    names = [c['name'] for c in cells]

    def choose(arity, tt):
        matches = [c for c in cells if len(c['vars']) == arity and c['tt'][0] == tt]
        matches = [c for c in matches if names.count(c['name']) == 1]
        if not matches:
            raise ValueError('library lacks unique single-output primitive')
        return min(matches, key=lambda c: (c['area'], max(c['pin_delay'].values(), default=0), c['name']))

    and2, inv = choose(2, 8), choose(1, 1)
    live = {int(v) for v in re.findall(r'^assign n(\d+) = .* & .*;$', logic, re.M)}
    negative = {x // 2 for x in roots if x > 1 and x & 1}
    negative.update(x // 2 for v in live for x in gates[v] if x > 1 and x & 1)
    negative_base_nodes = set(negative_base_nodes)
    if not negative_base_nodes <= live & negative:
        raise ValueError('negative base requires a demanded AND output phase')
    nand2 = choose(2, 7) if negative_base_nodes else None
    count = 0

    def instance(cell, args, output, name):
        nonlocal count
        count += 1
        if count > global_pool_limit:
            raise ValueError('global physical primitive pool exceeded')
        deadline()
        pins = [f'.{identifier(p)}({x})' for p, x in zip(cell['vars'], args)]
        pins.append(f'.{identifier(cell["out_pin"])}({output})')
        return f'{identifier(cell["name"])} {name}(' + ', '.join(pins) + ');'

    def signed(text):
        return re.sub(r'~n(\d+)', lambda m: 'm' + m[1], text)

    lines, module = [], None
    output_neg, boundaries = set(), set()
    for line in logic.splitlines():
        deadline()
        if line.startswith('module region_'):
            module = line.split('(')[0].split()[1]
            outputs = {int(x) for x in re.findall(r'output n(\d+)', line)}
            boundaries = {int(x) for x in re.findall(r'input n(\d+)', line)}
            output_neg = outputs & negative
            extra = [f'input m{v}' for v in sorted(boundaries & negative)]
            extra += [f'output m{v}' for v in sorted(output_neg)]
            if extra:
                line = line[:-2] + ', ' + ', '.join(extra) + ');'
            lines.append(line)
            continue
        if line.startswith('module mapped('):
            module = 'mapped'
            lines.append(line)
            # One inverter per demanded PI phase globally, not per region.
            for v in sorted(negative & set(range(1, inputs + 1))):
                # Declarations may follow instances in generated Verilog.
                lines.append(instance(inv, [f'n{v}'], f'm{v}', f'pi_inv_{v}'))
            continue
        gate = re.fullmatch(r'assign n(\d+) = (.+) & (.+);', line)
        if gate:
            v = int(gate[1])
            if v in negative and v not in output_neg:
                lines.append(f'wire m{v};')
            if v in negative_base_nodes:
                lines.append(instance(nand2, [signed(gate[2]), signed(gate[3])], f'm{v}', f'nand_{v}'))
                lines.append(instance(inv, [f'm{v}'], f'n{v}', f'inv_positive_{v}'))
            else:
                lines.append(instance(and2, [signed(gate[2]), signed(gate[3])], f'n{v}', f'and_{v}'))
            if v in negative and v not in negative_base_nodes:
                lines.append(instance(inv, [f'n{v}'], f'm{v}', f'inv_{v}'))
            continue
        if module == 'mapped' and re.fullmatch(r'wire n\d+;', line):
            v = int(re.search(r'n(\d+)', line)[1])
            lines.append(line)
            if v in negative:
                lines.append(f'wire m{v};')
            continue
        if module == 'mapped' and line.startswith('region_'):
            connected = {int(x) for x in re.findall(r'\.n(\d+)\(', line)}
            extra = [f'.m{v}(m{v})' for v in sorted(connected & negative)]
            if extra:
                line = line[:-2] + ', ' + ', '.join(extra) + ');'
        if module == 'mapped' and line.startswith('assign '):
            line = signed(line)
        lines.append(line)
    # Verify the materialized global witness accounting before exporting it.
    if count != len(live) + len(negative):
        raise ValueError('global primitive instance accounting mismatch')
    deadline()
    stats.update(actual_primitive_instances=count, globally_shared_inverters=len(negative),
                 construction_seconds=time.perf_counter() - start,
                 primitive_library_cells=[and2['name'], inv['name']],
                 negative_base_nodes=sorted(negative_base_nodes),
                 scope='library-matched primitive region seed; no egraph or regional cut optimization; node counts remain conservative import bounds')
    return '\n'.join(lines) + '\n', stats


def recover_phases(data, library_json, objective='area-delay', local_node_limit=20000,
                   global_node_limit=20000, global_pool_limit=100000, recovery_seconds=2):
    """One greedy global round; both phases stay realizable, no optimality claim."""
    if objective not in ('area-delay', 'delay-area') or recovery_seconds <= 0:
        raise ValueError('invalid phase recovery objective/deadline')
    start = time.perf_counter()
    reserve = .05
    def remaining():
        return recovery_seconds - (time.perf_counter() - start)
    base_text, base = physical_seed(data, library_json, local_node_limit, global_node_limit,
                                   global_pool_limit, remaining())
    inputs, roots, gates = read_binary_aig(data)
    live = {int(v) for v in re.findall(r'^.* and_(\d+)\(', base_text, re.M)}
    negative = {x // 2 for x in roots if x > 1 and x & 1}
    negative.update(x // 2 for v in live for x in gates[v] if x > 1 and x & 1)
    cells = json.loads(Path(library_json).read_text())
    byname = {c['name']: c for c in cells}
    and2, inv = [byname[n] for n in base['primitive_library_cells']]
    matches = [c for c in cells if len(c['vars']) == 2 and c['tt'][0] == 7
               and sum(x['name'] == c['name'] for x in cells) == 1]
    if not matches:
        raise ValueError('library lacks single-output NAND phase alternative')
    nand2 = min(matches, key=lambda c: (c['area'], max(c['pin_delay'].values()), c['name']))
    eligible = sorted(live & negative)
    templates = []
    for v in sorted(live):
        templates.append({'node': v, 'phase': 0, 'cell': and2['name'], 'inputs': list(gates[v])})
    for v in sorted(negative):
        templates.append({'node': v, 'phase': 1, 'cell': inv['name'], 'input_signal': [v, 0]})
    for v in eligible:
        templates.extend([{'node': v, 'phase': 1, 'cell': nand2['name'], 'inputs': list(gates[v])},
                          {'node': v, 'phase': 0, 'cell': inv['name'], 'input_signal': [v, 1]}])
    if len(templates) > global_pool_limit:
        raise ValueError('global alternative-instance template pool exceeded')

    def cost(chosen):
        arrival = {(0, 0): 0., (0, 1): 0.}
        for v in range(1, inputs + 1):
            arrival[v, 0] = 0.
            if v in negative:
                arrival[v, 1] = inv['pin_delay'][inv['vars'][0]]
        area = inv['area'] * len(negative)
        for v in sorted(live):
            if remaining() <= reserve:
                raise TimeoutError('phase score deadline')
            c = nand2 if v in chosen else and2
            delay = max(arrival[x // 2, x & 1] + c['pin_delay'][pin]
                        for x, pin in zip(gates[v], c['vars']))
            phase = int(v in chosen)
            arrival[v, phase] = delay
            if v in negative:
                arrival[v, 1 - phase] = delay + inv['pin_delay'][inv['vars'][0]]
            area += c['area']
        delay = max((arrival[x // 2, x & 1] for x in roots), default=0.)
        return (area, delay) if objective == 'area-delay' else (delay, area)

    chosen = set()
    best = cost(chosen)
    trials = 0
    for v in eligible[:2000]:
        try:
            trial = chosen | {v}
            score = cost(trial)
        except TimeoutError:
            break
        trials += 1
        def less(x, y):
            return x < y - 1e-6 * max(1, abs(y))
        equal_primary = abs(score[0] - best[0]) <= 1e-6 * max(1, abs(best[0]))
        if less(score[0], best[0]) or (equal_primary and less(score[1], best[1])):
            chosen, best = trial, score
    text, stats = physical_seed(data, library_json, local_node_limit, global_node_limit,
                                global_pool_limit, remaining(), chosen)
    if remaining() < 0:
        raise ValueError('global phase recovery deadline exceeded')
    stats.update(objective=objective, phase_trials=trials, recovery_rounds=1,
                 retained_instance_templates=len(templates), candidate_templates=templates,
                 global_recovery_seconds=time.perf_counter()-start,
                 predicted_area=best[0 if objective == 'area-delay' else 1],
                 predicted_delay=best[1 if objective == 'area-delay' else 0],
                 optimality='heuristic, not a bounded or unrestricted optimum')
    return text, stats
