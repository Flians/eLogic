#!/usr/bin/env python3
"""Additional independent named-port certificate; retain direct CEC failures."""
import argparse, hashlib, json, subprocess, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
from prototypes.region_stitch import symbols
from analyze_epfl_structural_budget import read_binary_aig
from bench_epoch_incumbent import record

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def validate_subject(row, subjects_path):
    assert row['preprocessing']['cec'] == 1
    subjects = json.loads(subjects_path.read_text())
    subject = next(r for r in subjects['results'] if r['bench'] == row['bench'])
    assert subject['status'] == 'ok' and subject['preprocessing']['cec'] == 1
    original = ROOT / f"tools/mockturtle/experiments/benchmarks/{row['bench']}.aig"
    prepared = subjects_path.parent / f"{row['bench']}.resyn2.aig"
    assert sha(original) == subject['input_sha256'] == row['input_sha256']
    assert sha(prepared) == subject['preprocessing']['sha256'] == row['preprocessing']['sha256']
    assert row['preprocessing'].get('source_report_sha256') == sha(subjects_path)
    interfaces = []
    for p in [original, prepared]:
        data = p.read_bytes()
        i, o, g = read_binary_aig(data)
        interfaces.append(symbols(data, i, o, g))
    assert interfaces[0] == interfaces[1], 'Original/prepared named port order differs'
    log = subjects_path.parent / f"{row['bench']}.cec.log"
    text = log.read_text()
    assert 'Networks are equivalent' in text and 'NOT EQUIVALENT' not in text.upper()
    return prepared, {
        'subject_report': str(subjects_path), 'subject_report_sha256': sha(subjects_path),
        'original_sha256': sha(original), 'prepared_sha256': sha(prepared),
        'original_to_prepared_cec_log': str(log), 'original_to_prepared_cec_log_sha256': sha(log),
        'identical_named_port_order': True,
        'interface_sha256': hashlib.sha256(json.dumps(interfaces[0]).encode()).hexdigest(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--subjects', type=Path, default=ROOT/'results_v2/epfl_resyn2_subjects_20261012/report.json')
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(exist_ok=False)
    source = json.loads(a.report.read_text())
    rows, commands = [], []
    library = a.report.parent / source['library']
    assert sha(library) == source['library_sha256']
    evaluator = ROOT / 'target/release/bench_rs_netlist_eval'
    for row in source['results']:
        if row['status'] == 'ok' or 'mapping_wall_secs' not in row:
            continue
        if row.get('partial_status') or row.get('partial_measured'):
            continue
        folder = a.report.parent / f"{row['bench']}.{row['method']}"
        mapped, net, log = folder/'mapped.v', folder/'mapped.json', folder/'evaluate.log'
        if not all(p.is_file() for p in [mapped, net, log]):
            continue
        if 'Timeout after 60s' not in log.read_text():
            continue  # An actual counterexample is not eligible for fallback proof.
        prepared, proof = validate_subject(row, a.subjects)
        new = json.loads(json.dumps(row))
        new.update(original_status=row['status'], original_error=row.get('error'),
                   original_report=str(a.report.resolve()), original_report_sha256=sha(a.report),
                   verification_policy='Direct original proof OR composed original->once-resyn2->mapped proof; direct failure retained',
                   new_mapper_run=False)
        argv = [str(evaluator), str(library.resolve()), str(prepared), str(mapped.resolve()), str(net.resolve())]
        start = time.monotonic()
        target = out / (row['bench']+'.prepared-cec.log')
        try:
            result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=60)
            target.write_text(result.stdout+result.stderr)
            assert result.returncode == 0, result.stderr
            measured = record(result.stdout)
            assert measured['cec'] == 1, 'Independent prepared-reference CEC did not pass'
            new.update(status='ok', measured=measured)
            new.pop('error', None)
        except (subprocess.TimeoutExpired, AssertionError, ValueError) as e:
            if isinstance(e, subprocess.TimeoutExpired):
                raw = e.stdout or b''
                target.write_text((raw.decode(errors='replace') if isinstance(raw, bytes) else raw)+'\n'+str(e))
            new.update(status='failed', error=str(e))
        commands.append(dict(argv=argv, verification_wall_secs=time.monotonic()-start,
                             per_process_guard_seconds=60, mapping_charged=False,
                             log=str(target), log_sha256=sha(target)))
        proof.update(evaluator_sha256=sha(evaluator), library_sha256=sha(library),
                     mapped_verilog_sha256=sha(mapped), mapped_json_sha256=sha(net),
                     prepared_to_mapped_cec=new['status']=='ok')
        new['verification_chain'] = proof
        rows.append(new)
        checkpoint=dict(library_sha256=source['library_sha256'],commands=commands,results=rows,original_report=str(a.report.resolve()),original_report_sha256=sha(a.report),mapping_budget_changed=False,baseline_algorithms_changed=False,verifier_sha256=sha(Path(__file__)),status='verification checkpoint')
        tmp=out/'report.json.tmp';tmp.write_text(json.dumps(checkpoint,indent=2)+'\n');tmp.replace(out/'report.json')
        print(row['bench'],row['method'],new['status'],flush=True)
    payload = dict(library_sha256=source['library_sha256'], commands=commands, results=rows,
                   original_report=str(a.report.resolve()), original_report_sha256=sha(a.report),
                   mapping_budget_changed=False, baseline_algorithms_changed=False,
                   verifier_sha256=sha(Path(__file__)), scope='Additional verification only; completed mapping required; partial mapping incumbents ineligible; original failures retained.')
    (out/'report.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps([dict(bench=r['bench'], status=r['status'], measured=r.get('measured'), error=r.get('error')) for r in rows], indent=2))
    raise SystemExit(0 if rows and all(r['status']=='ok' for r in rows) else 1)

if __name__ == '__main__':
    main()
