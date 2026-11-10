#!/usr/bin/env python3
"""Audit and summarize frozen supplemental controls without rewriting results.

Local replay is conditional on offered pools and uses estimated costs. Its
regret is NOT a circuit-level area reduction or a counterfactual optimization.
"""
import argparse
import ast
import collections
import hashlib
import json
import math
import re
import statistics
import struct
import subprocess
from pathlib import Path
from functools import lru_cache

from audit_mapped_ppa import audit
from run_epmapping_supplement import audit_abc


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quant(value, scale):
    x = value * scale
    return math.floor(x + .5) if x >= 0 else math.ceil(x - .5)


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def select(pool, policy):
    """Replay the same area-first feasibility and quantization on a fixed pool."""
    eligible = []
    for c in pool:
        # CCost stores area/delay as C++ floats before quantized ranking.
        a = f32(c[policy + '_area'])
        d = f32(c['standalone_delay' if policy == 'standalone' else 'live_delay'])
        if not c['function_valid'] or d > c['required'] + .001 or a > c['removed'] + .0001:
            continue
        if (quant(a, 1e4), quant(d, 1e3)) >= (quant(c['removed'], 1e4), quant(c['old_delay'], 1e3)):
            continue
        eligible.append((quant(a - c['removed'], 1e4), quant(d, 1e3), c['level'], c['candidate'], c))
    return min(eligible, key=lambda x: x[:4])[-1] if eligible else None


class GateFunctions(dict):
    pass


@lru_cache(maxsize=8)
def leaf_truths(leaves):
    return tuple(sum(1 << m for m in range(1 << leaves) if (m >> i) & 1) for i in range(leaves))


def functions(library):
    result = GateFunctions()
    result.pins = {}
    result.library_sha256 = sha(library)
    for block in re.split(r'(?m)^GATE\s+', library.read_text())[1:]:
        result.pins[block.split()[0]] = re.findall(r'\bPIN\s+(\w+)', block)
    for name, expression in re.findall(r'^GATE\s+(\S+)\s+\S+\s+\w+=(.*?);', library.read_text(), re.M):
        expression = expression.replace('!', '~').replace('*', '&').replace('+', '|').strip()
        tree = ast.parse(expression, mode='eval')
        assert all(isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Invert, ast.BitAnd, ast.BitOr)) for n in ast.walk(tree))
        result[name] = compile(tree, '<frozen-genlib>', 'eval')
    return result


def expression_truth(expression, leaves, cells):
    tokens = iter(re.findall(r'[^\s()]+|[()]', expression))
    mask = (1 << (1 << leaves)) - 1
    def parse():
        t = next(tokens)
        if t != '(':
            if t in ['0', '1']:
                return mask if t == '1' else 0
            assert len(t) == 1 and 0 <= ord(t) - ord('a') < leaves
            return leaf_truths(leaves)[ord(t) - ord('a')]
        name = next(tokens)
        if name == '~':
            value = parse() ^ mask
        else:
            pins = {p: parse() for p in cells.pins[name]}
            value = eval(cells[name], {'__builtins__': {}}, dict(pins, CONST0=0, CONST1=mask)) & mask
        assert next(tokens) == ')'
        return value
    value = parse()
    assert next(tokens, None) is None
    return value


def candidate_truth(c, cells):
    mask = (1 << (1 << c['leaves'])) - 1
    values = []
    for i, n in enumerate(c['nodes']):
        if n['kind'] == 0:
            assert not n['out_neg'] and n['input'] < c['leaves']
            value = leaf_truths(c['leaves'])[n['input']]
        elif n['kind'] == 1:
            assert not n['out_neg']
            value = mask if n['constant'] else 0
        else:
            assert len(n['pins']) == len(n['fanins'])
            assert all(j < i for j in n['fanins'])
            pins = {p: values[j] ^ (mask if (n['neg'] >> k) & 1 else 0) for k, (p, j) in enumerate(zip(n['pins'], n['fanins']))}
            value = eval(cells[n['cell']], {'__builtins__': {}}, dict(pins, CONST0=0, CONST1=mask)) & mask
        if n['out_neg']:
            value ^= mask
        values.append(value)
    return values[c['cell_root']]


def replay(path, cells):
    groups = collections.defaultdict(list)
    expressions = {}
    candidate_functions = {}
    lines = path.read_text().splitlines()
    truncated = 0
    for line_number, line in enumerate(lines):
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            # SIGALRM can interrupt the final streamed record, never an earlier one.
            assert line_number == len(lines) - 1 and not path.read_bytes().endswith(b'\n')
            truncated += 1
            continue
        assert c['function_valid'] == 1, (path, c)
        assert len(c['leaf_ids']) == len(c['leaf_arrivals']) == c['leaves']
        assert len(set(c['leaf_ids'])) == c['leaves']
        assert abs(c['reuse_area'] + c['retained_area'] - c['live_area']) < 1e-7
        assert 0 <= c['cell_root'] < len(c['nodes'])
        expression_key = c['expression'], c['leaves']
        if expression_key not in expressions:
            expressions[expression_key] = expression_truth(*expression_key, cells)
        assert expressions[expression_key] == int(c['truth']), (path, 'original expression', c['expression'])
        # Exact serialized DAG + ordered leaf dimension/root, never live cost or
        # expected/care truth. Every occurrence still checks its own care mask.
        parts = line.split(',"nodes":', 1)
        assert len(parts) == 2
        dag_key = c['leaves'], c['cell_root'], parts[1]
        if dag_key not in candidate_functions:
            candidate_functions[dag_key] = candidate_truth(c, cells)
        assert ((candidate_functions[dag_key] ^ int(c['truth'])) & int(c['care'])) == 0, (path, c['candidate'])
        for n in c['nodes']:
            assert all(0 <= i < len(c['nodes']) for i in n['fanins'])
        # Truth has been checked; ranking needs only costs, not millions of
        # decoded node dictionaries. Full DAG evidence stays in the raw trace.
        fields = ('candidate', 'source', 'function_valid', 'standalone_area', 'standalone_delay',
                  'reuse_area', 'retained_area', 'live_area', 'live_delay', 'required', 'removed', 'old_delay', 'level')
        groups[c['round'], c['root'], c['revision']].append({k: c[k] for k in fields})
    out = dict(trace_sha256=sha(path), library_sha256=cells.library_sha256, analyzer_sha256=sha(Path(__file__)),
               ranking_arithmetic='float32 candidate area/delay, double context baseline, implementation quantization',
               snapshots=len(groups), candidates=sum(map(len, groups.values())), original_expressions_checked=len(expressions),
               independent_candidate_functions_checked=sum(map(len, groups.values())), unique_candidate_functions_evaluated=len(candidate_functions), truncated_final_records=truncated,
               retained_candidates=sum(c['retained_area'] > .00001 for g in groups.values() for c in g),
               phase_savings_candidates=sum(c['standalone_area'] > c['live_area'] + .00001 for g in groups.values() for c in g),
               candidate_sources=collections.Counter(str(c['source']) for g in groups.values() for c in g))
    for p in ['standalone', 'reuse', 'live']:
        stats = collections.Counter()
        for pool in groups.values():
            chosen, reference = select(pool, p), select(pool, 'live')
            stats['selected'] += chosen is not None
            stats['different'] += (None if chosen is None else chosen['candidate']) != (None if reference is None else reference['candidate'])
            if chosen:
                stats['live_infeasible'] += chosen['live_area'] > chosen['removed'] + .0001 or chosen['live_delay'] > chosen['required'] + .001
                stats['live_no_improvement'] += (quant(f32(chosen['live_area']), 1e4), quant(f32(chosen['live_delay']), 1e3)) >= (quant(chosen['removed'], 1e4), quant(chosen['old_delay'], 1e3))
                # Keep the precision effect visible, separate from the pricing
                # comparison. Final whole-netlist costs/guards are double.
                stats['double_no_improvement'] += (quant(chosen['live_area'], 1e4), quant(chosen['live_delay'], 1e3)) >= (quant(chosen['removed'], 1e4), quant(chosen['old_delay'], 1e3))
            # Bounded, estimated per-snapshot difference after rejecting live-illegal choices.
            def gain(c):
                return max(0., c['removed'] - c['live_area']) if c and c['live_delay'] <= c['required'] + .001 else 0.
            stats['estimated_area_regret'] += gain(reference) - gain(chosen)
        out[p] = dict(stats)
    return out


def verify_abc_fallbacks(root):
    """Validate actual selected fallbacks after search, without changing search data."""
    for path in sorted(root.glob('common/abc/*/result.json')):
        r = json.loads(path.read_text())
        if r['status'] != 'complete' or r['guard_selected'] != 'initial':
            continue
        proof = path.parent / 'selected_initial_cec.json'
        net = root / 'common/ours' / (r['mode'] + '.' + r['design']) / 'epmapping_baseline.v'
        bench = root / 'inputs/benchmarks' / (r['design'] + '.aig')
        library, binary = root / 'inputs/asap7_6in.genlib', root / 'inputs/abc'
        hashes = dict(netlist_sha256=sha(net), benchmark_sha256=sha(bench), library_sha256=sha(library), abc_sha256=sha(binary))
        if proof.exists():
            prior = json.loads(proof.read_text())
            assert all(prior[k] == v for k, v in hashes.items())
            assert sha(path.parent / 'selected_initial_cec.log') == prior['proof_sha256']
            continue
        command = [str(binary), '-c', f'read_genlib {library}; read -m {net}; cec -n {bench}']
        out = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        log = path.parent / 'selected_initial_cec.log'
        log.write_text(out.stdout + out.stderr)
        assert out.returncode == 0 and 'Networks are equivalent' in log.read_text(), path
        proof.write_text(json.dumps(dict(hashes, cec=1, proof_sha256=sha(log), validation_outside_search_budget=True), indent=2) + '\n')


def restore_abc_output_aliases(text, library):
    """Represent simple-CO serialization buffers as wires, retaining core cells.

    ABC's abcNetlist.c calls Abc_NtkLogicMakeSimpleCos before Verilog writing;
    abcUtil.c adds BUFs for CI/CO naming and shared CO drivers. This rule is
    admitted ONLY when an independent audit reproduces the pre-write metrics.
    It never changes the input netlist or runs a synthesis pass.
    """
    buffers = set(re.findall(r'^GATE\s+(\S+)\s+\S+\s+Y=A;', library.read_text(), re.M))
    pi = {n.strip() for s in re.findall(r'\binput\s+(.*?);', text, re.S) for n in s.split(',')}
    po = {n.strip() for s in re.findall(r'\boutput\s+(.*?);', text, re.S) for n in s.split(',')}
    replaced = []
    def replace(m):
        if m.group(1) not in buffers:
            return m.group(0)
        ports = {p: s.strip() for p, s in re.findall(r'\.(\w+)\s*\(([^)]*)\)', m.group(3))}
        if set(ports) == {'A', 'Y'} and ports['Y'] in po and ports['A'] in pi | po:
            replaced.append(dict(instance=m.group(2), source=ports['A'], output=ports['Y']))
            return '\n  assign ' + ports['Y'] + ' = ' + ports['A'] + ';'
        return m.group(0)
    result = re.sub(r'^\s*(\w+)\s+(\w+)\s*\((.*?)\)\s*;', replace, text, flags=re.M | re.S)
    return result, replaced


def verify_abc_native_outputs(root):
    """Recover and validate ABC's actual pre-write output, without rerunning mfs3."""
    library, binary = root / 'inputs/asap7_6in.genlib', root / 'inputs/abc'
    for path in sorted(root.glob('common/abc/*/result.json')):
        r = json.loads(path.read_text())
        if r['status'] != 'complete':
            continue
        wd = path.parent
        original = wd / 'mapped.v'
        normalized = wd / 'mapped_native_aliases.v'
        proof = wd / 'native_alias_audit.json'
        bench = root / 'inputs/benchmarks' / (r['design'] + '.aig')
        text, replaced = restore_abc_output_aliases(original.read_text(), library)
        hashes = dict(export_sha256=sha(original), library_sha256=sha(library), abc_sha256=sha(binary), benchmark_sha256=sha(bench))
        if proof.exists():
            p = json.loads(proof.read_text())
            assert all(p[k] == v for k, v in hashes.items())
            assert normalized.read_text() == text and sha(normalized) == p['normalized_sha256']
            assert sha(wd / 'native_alias_cec.log') == p['proof_sha256']
            measured = audit_abc(normalized, library)
            single = audit_abc(normalized, library, float32=True)
            assert abs(measured['area'] - r['native_final']['area']) <= .005001
            assert abs(single['delay'] - r['native_final']['delay']) <= .005001
            assert all(abs(measured[q] - p['metrics'][q]) < (.0001 if q == 'area' else .001) for q in ['area', 'delay'])
            p.update(native_float32_metrics=single, double_minus_float32_delay=measured['delay'] - single['delay'])
            proof.write_text(json.dumps(p, indent=2) + '\n')
            continue
        normalized.write_text(text)
        measured = audit_abc(normalized, library)
        single = audit_abc(normalized, library, float32=True)
        assert abs(measured['area'] - r['native_final']['area']) <= .005001 and abs(single['delay'] - r['native_final']['delay']) <= .005001, ('alias rule or float32 timing does not reproduce native metrics', path, measured, single, r['native_final'])
        command = [str(binary), '-c', f'read_genlib {library}; read -m {normalized}; cec -n {bench}']
        out = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        log = wd / 'native_alias_cec.log'
        log.write_text(out.stdout + out.stderr)
        assert out.returncode == 0 and 'Networks are equivalent' in log.read_text(), path
        proof.write_text(json.dumps(dict(hashes, normalized_sha256=sha(normalized), proof_sha256=sha(log),
                                        cec=1, metrics=measured, native_float32_metrics=single,
                                        double_minus_float32_delay=measured['delay'] - single['delay'],
                                        aliases=replaced, validation_outside_search_budget=True), indent=2) + '\n')


def strong_full_records(root, reference):
    m = json.loads((root / 'manifest.json').read_text())
    assert m['common_rounds'] == 6 and m['reference_manifest_sha256'] == sha(reference / 'manifest.json')
    for name, h in m['input_sha256'].items():
        assert sha(root / 'inputs' / name) == h, name
    assert sha(root / 'inputs/engine') == sha(reference / 'inputs/engine')
    records = []
    for path in sorted(root.glob('common/ours/*/result.json')):
        r = json.loads(path.read_text())
        assert (r['part'], r['arm']) == ('common', 'ours') and r['status'] in ['complete', 'timeout'] and r['cec'] == 1
        cmd = json.loads((path.parent / 'command.json').read_text())
        assert cmd['env']['EPMAPPING_ROUNDS'] == '6' and cmd['env']['EPMAPPING_POST_TIMEOUT'] == '1000' and cmd['env']['EPMAPPING_WINDOW_SYN'] == '0'
        lib = root / 'inputs/asap7_6in.genlib'
        assert cmd['library_sha256'] == sha(lib) == sha(reference / 'inputs/asap7_6in.genlib')
        assert r['initial_sha256'] == sha(path.parent / 'epmapping_baseline.v')
        assert r['initial_sha256'] == sha(reference / 'common/ours' / (r['mode'] + '.' + r['design']) / 'epmapping_baseline.v')
        assert sha(root / r['selected_netlist']) == r['selected_sha256'] and r['optimizer_seconds'] <= 1000.1
        if r['status'] == 'complete':
            assert '[i] cec=1' in (path.parent / 'log').read_text()
        else:
            p = json.loads((path.parent / 'checkpoint_selection.json').read_text())
            assert p['proof_sha256'] == sha(path.parent / 'checkpoint_cec.log')
            assert 'Networks are equivalent' in (path.parent / 'checkpoint_cec.log').read_text()
        measured = audit(root / r['selected_netlist'], lib)
        initial = audit(path.parent / 'epmapping_baseline.v', lib)
        for q in ['area', 'delay']:
            tolerance = .0001 if q == 'area' else .001
            assert abs(measured[q] - r['final'][q]) < tolerance and abs(initial[q] - r['initial'][q]) < tolerance
            assert measured[q] <= initial[q] + tolerance
        r.update(arm='ours6', source_arm='ours', source_campaign=str(root))
        records.append(r)
    return records, m


def summarize(root, allow_partial=False, strong_root=None):
    manifest = json.loads((root / 'manifest.json').read_text())
    cells = functions(root / 'inputs/asap7.genlib')
    for name, expected in manifest['input_sha256'].items():
        assert sha(root / 'inputs' / name) == expected, name
    expected = {(part, arm, d, mode) for d in manifest['designs'] for mode in ['cold', 'warm']
                for part, arms in [('profile', manifest['profile_arms']), ('replay', ['vector']), ('common', ['ours', 'abc'])] for arm in arms}
    records = []
    replay_results = []
    fallback_proofs_pending = []
    native_proofs_pending = []
    for path in sorted(root.glob('*/*/*/result.json')):
        r = json.loads(path.read_text())
        key = r['part'], r['arm'], r['design'], r['mode']
        assert key in expected, key
        assert r['status'] in ['complete', 'timeout'] and r['cec'] == 1, path
        assert r['optimizer_seconds'] <= 1000.1, path
        command = json.loads((path.parent / 'command.json').read_text())
        lib = root / 'inputs' / ('asap7_6in.genlib' if r['part'] == 'common' else 'asap7.genlib')
        assert command['library_sha256'] == sha(lib)
        if r['arm'] != 'abc':
            assert command['env']['EPMAPPING_POST_TIMEOUT'] == '1000'
            assert command['env']['EPMAPPING_WINDOW_SYN'] == '0'
            netlist = root / r['selected_netlist']
            assert sha(netlist) == r['selected_sha256']
            measured = audit(netlist, lib)
            assert sha(path.parent / 'epmapping_baseline.v') == r['initial_sha256']
            initial = audit(path.parent / 'epmapping_baseline.v', lib)
            if r['status'] == 'complete':
                assert '[i] cec=1' in (path.parent / 'log').read_text()
            else:
                proof = json.loads((path.parent / 'checkpoint_selection.json').read_text())
                assert sha(path.parent / 'checkpoint_cec.log') == proof['proof_sha256']
                assert 'Networks are equivalent' in (path.parent / 'checkpoint_cec.log').read_text()
            if r['part'] != 'common':
                prior = Path(manifest['original_campaign']) / r['mode'] / 'tab_main' / ('ae.' + r['design']) / 'epmapping_baseline.v'
                assert sha(prior) == r['initial_sha256']
        else:
            assert command['timeout'] == 1000
            assert r['initial_sha256'] == sha(root / 'common/ours' / f"{r['mode']}.{r['design']}" / 'epmapping_baseline.v')
            initial = audit(root / 'common/ours' / f"{r['mode']}.{r['design']}" / 'epmapping_baseline.v', lib)
            assert 'Networks are equivalent' in (path.parent / 'cec.log').read_text()
            if r['status'] == 'complete' and r['guard_selected'] == 'initial':
                proof = path.parent / 'selected_initial_cec.json'
                if not proof.exists():
                    fallback_proofs_pending.append(str(path.parent.relative_to(root)))
                else:
                    p = json.loads(proof.read_text())
                    assert p['cec'] == 1 and p['netlist_sha256'] == r['initial_sha256']
                    assert p['proof_sha256'] == sha(path.parent / 'selected_initial_cec.log')
                    assert 'Networks are equivalent' in (path.parent / 'selected_initial_cec.log').read_text()
                    r['selected_initial_cec_audit'] = p
            if r['status'] == 'complete':
                assert sha(path.parent / 'mapped.v') == r['raw_sha256']
                raw = audit_abc(path.parent / 'mapped.v', lib)
                assert abs(raw['area'] - r['raw']['area']) < .0001 and abs(raw['delay'] - r['raw']['delay']) < .001
                native_proof = path.parent / 'native_alias_audit.json'
                if not native_proof.exists():
                    native_proofs_pending.append(str(path.parent.relative_to(root)))
                else:
                    p = json.loads(native_proof.read_text())
                    normalized = path.parent / 'mapped_native_aliases.v'
                    assert p['cec'] == 1 and sha(normalized) == p['normalized_sha256'] and p['export_sha256'] == r['raw_sha256']
                    assert sha(path.parent / 'native_alias_cec.log') == p['proof_sha256']
                    assert 'Networks are equivalent' in (path.parent / 'native_alias_cec.log').read_text()
                    assert p['library_sha256'] == sha(lib) and p['abc_sha256'] == sha(root / 'inputs/abc')
                    native_cost = audit_abc(normalized, lib)
                    assert all(abs(native_cost[q] - p['metrics'][q]) < (.0001 if q == 'area' else .001) for q in ['area', 'delay'])
                    single = audit_abc(normalized, lib, float32=True)
                    assert abs(native_cost['area'] - r['native_final']['area']) <= .005001 and abs(single['delay'] - r['native_final']['delay']) <= .005001
                    assert abs(single['delay'] - p['native_float32_metrics']['delay']) < .001
                    r['native_graph_audit'] = p
                    feasible = native_cost['area'] <= r['initial']['area'] + .0001 and native_cost['delay'] <= r['initial']['delay'] + .001
                    r['comparison_final'] = native_cost if feasible else r['initial']
                    r['comparison_guard_selected'] = 'native_graph' if feasible else 'initial'
                    r['comparison_selected_netlist'] = str((normalized if feasible else root / 'common/ours' / (r['mode'] + '.' + r['design']) / 'epmapping_baseline.v').relative_to(root))
            measured = r['raw'] if r['guard_selected'] == 'optimized' else r['initial']
        for q in ['area', 'delay']:
            assert abs(initial[q] - r['initial'][q]) < (.0001 if q == 'area' else .001)
            assert abs(measured[q] - r['final'][q]) < (.0001 if q == 'area' else .001)
            assert r['final'][q] <= r['initial'][q] + (.0001 if q == 'area' else .001)
        records.append(r)
        if r['part'] == 'replay':
            replay_results.append(dict(design=r['design'], mode=r['mode'], **replay(path.parent / 'candidates.jsonl', cells)))
    found = {(r['part'], r['arm'], r['design'], r['mode']) for r in records}
    assert len(found) == len(records)
    if strong_root is not None:
        more, strong_manifest = strong_full_records(strong_root, root)
        expected.update(('common', 'ours6', d, mode) for d in manifest['designs'] for mode in ['cold', 'warm'])
        records.extend(more)
        found.update((r['part'], r['arm'], r['design'], r['mode']) for r in more)
    assert len(found) == len(records) and found <= expected
    input_groups = collections.defaultdict(set)
    for r in records:
        input_groups[r['part'] == 'common', r['design']].add(r['initial_sha256'])
    assert all(len(hashes) == 1 for hashes in input_groups.values()), input_groups
    missing = sorted(expected - found)
    assert allow_partial or (not missing and not fallback_proofs_pending and not native_proofs_pending), (missing, fallback_proofs_pending, native_proofs_pending)
    if not allow_partial:
        for campaign, count in [(root, 96)] + ([(strong_root, 16)] if strong_root else []):
            status = json.loads((campaign / 'execution_status.json').read_text())
            assert status['complete'] and not status['errors'] and status['jobs'] == count, status
    # Search-driver selections remain unchanged in records. Controlled comparison
    # uses ABC's independently reconstructed native result under the SHARED bound.
    evaluation = [dict(r, final=r.get('comparison_final', r['final'])) for r in records]
    indexed = {(r['part'], r['arm'], r['design'], r['mode']): r for r in evaluation}
    for row in replay_results:
        key = 'profile', 'vector', row['design'], row['mode']
        if key in indexed:
            a = indexed['replay', 'vector', row['design'], row['mode']]
            b = indexed[key]
            row['vector_same_final_ppa'] = all(abs(a['final'][q] - b['final'][q]) < (.0001 if q == 'area' else .001) for q in ['area', 'delay'])
    aggregates = []
    for part, arms in [('profile', manifest['profile_arms']), ('common', ['ours', 'ours6', 'abc'] if strong_root else ['ours', 'abc'])]:
        for arm in arms:
            for mode in ['cold', 'warm']:
                rows = [r for r in evaluation if (r['part'], r['arm'], r['mode']) == (part, arm, mode)]
                if not rows:
                    continue
                a = dict(part=part, arm=arm, mode=mode, n=len(rows), timeouts=sum(r['status'] == 'timeout' for r in rows),
                         total_seconds=sum(r['optimizer_seconds'] for r in rows),
                         mean_seconds=statistics.mean(r['optimizer_seconds'] for r in rows))
                for q in ['area', 'delay']:
                    a['mean_' + q + '_percent'] = statistics.mean(100 * (r['final'][q] / r['initial'][q] - 1) for r in rows)
                    a['median_' + q + '_percent'] = statistics.median(100 * (r['final'][q] / r['initial'][q] - 1) for r in rows)
                    other = [r for r in rows if r['design'] != 'arbiter']
                    if other:
                        a['mean_' + q + '_percent_excluding_arbiter'] = statistics.mean(100 * (r['final'][q] / r['initial'][q] - 1) for r in other)
                if part == 'profile':
                    counters = collections.Counter()
                    for r in rows:
                        for line in r.get('profile_stats', []):
                            counters.update({k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)})
                    a['profile_counters'] = dict(counters)
                    pairs = [(r, indexed['profile', 'vector', r['design'], mode]) for r in rows if ('profile', 'vector', r['design'], mode) in indexed]
                    a.update(paired=len(pairs), area_delta=sum(r['final']['area'] - v['final']['area'] for r, v in pairs),
                             area_better=sum(r['final']['area'] < v['final']['area'] - .0001 for r, v in pairs),
                             area_worse=sum(r['final']['area'] > v['final']['area'] + .0001 for r, v in pairs),
                             delay_changed=sum(abs(r['final']['delay'] - v['final']['delay']) > .001 for r, v in pairs),
                             v28_same_ppa=sum(r.get('v28_same_ppa', False) for r in rows))
                else:
                    pairs = [(r, indexed['common', 'ours', r['design'], mode]) for r in rows if ('common', 'ours', r['design'], mode) in indexed]
                    a['ours_area_wins'] = sum(v['final']['area'] < r['final']['area'] - .0001 for r, v in pairs)
                    if arm == 'abc':
                        full_pairs = [(r, indexed['common', 'ours6', r['design'], mode]) for r in rows if ('common', 'ours6', r['design'], mode) in indexed]
                        if full_pairs:
                            a['ours6_area_wins'] = sum(o['final']['area'] < b['final']['area'] - .0001 for b, o in full_pairs)
                            a['ours6_area_losses'] = [b['design'] for b, o in full_pairs if o['final']['area'] > b['final']['area'] + .0001]
                            a['paired_mean_ours6_vs_abc_area_percent'] = statistics.mean(100 * (o['final']['area'] / b['final']['area'] - 1) for b, o in full_pairs)
                            other_pairs = [(b, o) for b, o in full_pairs if b['design'] != 'arbiter']
                            if other_pairs:
                                a['paired_mean_ours6_vs_abc_area_percent_excluding_arbiter'] = statistics.mean(100 * (o['final']['area'] / b['final']['area'] - 1) for b, o in other_pairs)
                        if pairs:
                            relative = [100 * (own['final']['area'] / base['final']['area'] - 1) for base, own in pairs]
                            other_relative = [100 * (own['final']['area'] / base['final']['area'] - 1) for base, own in pairs if base['design'] != 'arbiter']
                            a['paired_mean_ours_vs_abc_area_percent'] = statistics.mean(relative)
                            a['paired_median_ours_vs_abc_area_percent'] = statistics.median(relative)
                            if other_relative:
                                a['paired_mean_ours_vs_abc_area_percent_excluding_arbiter'] = statistics.mean(other_relative)
                            a['ours_area_losses'] = [base['design'] for base, own in pairs if own['final']['area'] > base['final']['area'] + .0001]
                        a['raw_area_wins_for_ours'] = sum(v['final']['area'] < r.get('native_graph_audit', {}).get('metrics', r['raw'])['area'] - .0001 for r, v in pairs if r['status'] == 'complete')
                        a['guard_model_disagreement'] = sum(r.get('raw_feasible', False) != r.get('native_feasible', False) for r in rows)
                        a['raw_delay_violations'] = sum(r.get('native_graph_audit', {}).get('metrics', r.get('raw', r['initial']))['delay'] > r['initial']['delay'] + .001 for r in rows)
                        a['guard_initial'] = sum(r.get('comparison_guard_selected', r['guard_selected']) == 'initial' for r in rows)
                        a['buffer_metric_difference'] = sum(abs(r['raw']['area'] - r['native_final']['area']) > .0001 for r in rows if r['status'] == 'complete')
                aggregates.append(a)
    return dict(schema=2 if strong_root else 1, campaign=str(root), strong_full_campaign=str(strong_root) if strong_root else None,
                strong_full_manifest_sha256=sha(strong_root / 'manifest.json') if strong_root else None,
                manifest_sha256=sha(root / 'manifest.json'), complete=not missing and not fallback_proofs_pending and not native_proofs_pending,
                expected=len(expected), observed=len(records), missing=missing, aggregates=aggregates,
                replay=replay_results, records=records, fallback_proofs_pending=fallback_proofs_pending, native_proofs_pending=native_proofs_pending)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('--allow-partial', action='store_true')
    ap.add_argument('--verify-fallback-cec', action='store_true', help='Run missing selected-initial CEC proofs after search')
    ap.add_argument('--verify-native-cec', action='store_true', help='Restore output aliases, audit native costs, and run CEC without rerunning optimization')
    ap.add_argument('--strong-full', type=Path, help='Include the normal six-round common-input campaign')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.verify_fallback_cec:
        verify_abc_fallbacks(args.root.resolve())
    if args.verify_native_cec:
        verify_abc_native_outputs(args.root.resolve())
    result = summarize(args.root.resolve(), args.allow_partial, args.strong_full.resolve() if args.strong_full else None)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['complete', 'expected', 'observed', 'aggregates']}, indent=2))
