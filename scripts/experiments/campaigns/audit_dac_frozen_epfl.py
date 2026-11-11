"""Audit the complete, uniform DAC full20 checkpoint without rerunning mapping.

Run only after the finite worker releases its locks. Failed attempts remain
rows; missing designs prevent final publication. No best-repeat selection.
"""
import argparse
import csv
import fcntl
import json
import math
import pathlib
import shutil
import subprocess
import sys

import run_dac_convergence as core
from run_dac_frozen_epfl import selection


def candidate_row(name, row, subject):
    valid = row.get('status') == 'ok'
    if valid:
        assert row.get('CEC') == 1 and row.get('first_independent_CEC')
        assert row.get('qualified_clock_chain')
        assert all(isinstance(row.get(k), (int, float)) and
                   math.isfinite(row[k]) and row[k] >= 0 for k in ['A', 'D', 'T'])
        cfg = row['configuration']
        assert cfg['mapping_seconds'] == 600 and cfg['objective'] == 'delay-area'
        assert cfg['initial_cover_policy'] == 'direct'
        assert cfg['trace_write'] == 'fragmented' and cfg['evidence_arrays'] == 'copy'
    return dict(bench=name, method='egraphmap_dac_off',
                input_sha256=subject['golden_sha256'],
                status='ok' if valid else row.get('status', 'missing'),
                measured=dict(area=row['A'], delay=row['D'], cec=1,
                              mapping_seconds=row['T']) if valid else {},
                error=row.get('error'), source_case=row.get('case'),
                first_native_AD=row.get('initial_AD'),
                search_net_gain=row.get('search_net_gain'),
                search_finished=row.get('search_finished'),
                search_timeout=row.get('search_timeout'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study', type=pathlib.Path,
                    default=core.DEFAULT.parent / 'full-epfl')
    ap.add_argument('--output', type=pathlib.Path, required=True)
    a = ap.parse_args()
    study = a.study.resolve()
    assert not a.output.exists(), 'keep each completed audit immutable'
    locks = []
    parent = study.parent / 'benchmark'
    for path in [core.R / 'results_v2/independent_components_20261019/global.lock',
                 parent / 'runner.lock', study / 'runner.lock']:
        handle = path.open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    c = core.read(study / 'campaign.json')
    p = core.read(parent / 'campaign.json')
    assert c['frozen'] == p['frozen'] == core.fingerprint()
    assert c['parent_campaign_sha256'] == core.sha(parent / 'campaign.json')
    assert c['selection'] == selection(p) and c['selection']['variant'] == 'off'
    assert core.sha(study / 'executed-wrapper.py') == c['wrapper_sha256']
    assert c['expected'] == 20 and set(c['jobs']) == set(p['subjects']), \
        'full20 attempts must finish before the final audit'
    checked = 0
    invalid = []

    def verify(path, expected):
        nonlocal checked
        checked += 1
        if not path.is_file() or core.sha(path) != expected:
            invalid.append(str(path))

    for name, digest in c['frozen'].items():
        verify(parent / 'frozen' / name, digest)
    rows = []
    provenance = []
    for name, subject in p['subjects'].items():
        row = c['jobs'][name]
        assert row['design'] == name and row['rep'] == 0 and row['variant'] == 'off'
        if name in c['references']:
            key = c['references'][name]['parent_key']
            assert key == f'{name}_r0_off' and row == p['jobs'][key]
        for relative, digest in row['artifacts'].items():
            verify(pathlib.Path(row['case']) / relative, digest)
        verify(pathlib.Path(row['sample_file']), row['sample_sha256'])
        verify(pathlib.Path(subject['golden']), subject['golden_sha256'])
        verify(pathlib.Path(subject['prepared']), subject['prepared_sha256'])
        if row['status'] == 'ok':
            report = core.read(pathlib.Path(row['case']) / 'report.json')
            assert report['status'] == 'ok' and report['cec'] == 1
            assert report['evaluation']['exact_agreement']
            assert report['selected_evidence']['valid']
            assert report['selected_evidence']['netlist_binding_agreement']
            assert [report['measured']['area'], report['measured']['delay']] == [row['A'], row['D']]
            assert report['output_sha256'] == row['output_sha256']
            assert report['direct_native'] and not report.get('external_seed')
        rows.append(candidate_row(name, row, subject))
        provenance.append(dict(design=name, case=row['case'],
                               reference=c['references'].get(name),
                               artifacts=row['artifacts'],
                               sample_sha256=row['sample_sha256']))
    baseline = core.read(core.BASE)
    library = core.R / 'emapper/libraries/asap7/asap7_full.genlib'
    assert baseline['library_sha256'] == core.sha(library)
    assert not invalid, invalid[:20]
    a.output.mkdir(parents=True)
    for src, dst in [(study / 'campaign.json', 'campaign-snapshot.json'),
                     (parent / 'campaign.json', 'parent-snapshot.json'),
                     (pathlib.Path(__file__), 'audit-helper.py')]:
        shutil.copy2(src, a.output / dst)
    report = dict(library_sha256=baseline['library_sha256'],
                  metric_model=baseline['metric_model'], results=rows,
                  variant='off', per_design_selection=False,
                  initial_cover_origin='native direct mapping; no external seed',
                  baseline_runs=0, GitHub_push=False)
    core.save(a.output / 'candidate.json', report)
    core.save(a.output / 'artifact-provenance.json', provenance)
    with (a.output / 'results.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['design', 'status', 'A', 'D', 'T', 'CEC',
                         'initial_A', 'initial_D', 'search_net_gain',
                         'search_finished', 'search_timeout'])
        for row in rows:
            m = row['measured']; initial = row['first_native_AD'] or [None, None]
            writer.writerow([row['bench'], row['status'], m.get('area'), m.get('delay'),
                             m.get('mapping_seconds'), m.get('cec'), *initial,
                             row['search_net_gain'], row['search_finished'], row['search_timeout']])
    gates = {}
    for objective in ['delay-area', 'area-delay', 'pareto']:
        target = a.output / (objective + '-gate.json')
        command = [sys.executable, str(core.R / 'scripts/experiments/check_mapping_dominance.py'),
                   '--candidate', str(a.output / 'candidate.json'),
                   '--baselines', str(core.BASE), '--objective', objective,
                   '--output', str(target)]
        proc = subprocess.run(command, capture_output=True, text=True)
        (a.output / (objective + '-gate.log')).write_text(proc.stdout + proc.stderr)
        assert proc.returncode in [0, 1] and target.exists(), 'comparison failed to execute'
        gate = core.read(target)
        assert (proc.returncode == 0) == gate['all_passed']
        gates[objective] = dict(all_passed=gate['all_passed'], command=command)
    good = [r for r in rows if r['status'] == 'ok']
    validation = dict(completed=20, valid_CEC=len(good), failures=20-len(good),
                      initial_CEC=len(good), verified_artifact_references=checked,
                      invalid_artifacts=invalid, gates=gates,
                      source_campaign_sha256=core.sha(study / 'campaign.json'),
                      baseline_sha256=core.sha(core.BASE),
                      source_helper_sha256=core.sha(pathlib.Path(__file__)),
                      baseline_runs=0, GitHub_push=False,
                      search_deadline=sum(bool(r['search_timeout']) for r in good),
                      ablations_completed=False, paper_updated=False,
                      scope='uniform off, four explicit r0 references; failures retained; '
                            'deadline covers are not proof of completed search; '
                            'original baseline cohort, later supplements remain separate')
    core.save(a.output / 'validation.json', validation)
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
