#!/usr/bin/env python3
"""Three repetitions of both fixed phase fractions on max, both libraries.

This targeted stability experiment does not replace the all20 paired scope.
"""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from run_epfl_baselines import ROOT, sha


def main():
    out = ROOT / 'results_v2/epfl_phase_allocation_max_repeats_20261013'
    out.mkdir(exist_ok=False)
    handles = []
    for p in [out/'runner.lock',
              ROOT/'results_v2/epfl_original_baselines_allocation_checkpoint_20261013/runner.lock',
              ROOT/'results_v2/epfl_original_baselines_checkpoint_20261013/runner.lock',
              ROOT/'results_v2/epfl_phase_allocation_checkpoint_20261013/runner.lock',
              ROOT/'results_v2/epfl_mapping_policy_checkpoint_20261013/runner.lock',
              ROOT/'results_v2/epfl_strict_main_checkpoint_20261013/runner.lock',
              ROOT/'results_v2/epfl_baseline_checkpoint_20261013/runner.lock']:
        f = p.open('a'); fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); handles.append(f)
    helper = Path(__file__).with_name('bench_egraph_phase_allocation.py')
    wrapper = Path(__file__).with_name('run_with_mapping_budget.py')
    subjects = ROOT/'results_v2/epfl_resyn2_subjects_20261012/report.json'
    files = [Path(__file__).resolve(), helper, wrapper,
             Path(__file__).with_name('nested_mapping_clock.py'), subjects,
             ROOT/'target/release/bench_rs_egraph_map', ROOT/'target/release/bench_rs_netlist_eval',
             ROOT/'src/bin/bench_rs_egraph_map.rs',
             ROOT/'tools/mockturtle/experiments/cell_libraries/asap7.genlib',
             Path('/usr/bin/abc'), Path('/usr/local/bin/yosys'),
             *sorted((ROOT/'emapper/src').glob('*.rs')),
             *sorted((ROOT/'scripts/experiments').glob('*.py'))]
    state = {'scope': 'max only, both libraries, delay-area; three paired repetitions; all runs retained',
             'mapper_root': str(ROOT), 'fraction_candidate': .25, 'fraction_control': 1.0,
             'mapping_seconds': 300, 'global_recovery_seconds': 2,
             'expected_jobs': 12, 'source_sha256': {str(p): sha(p) for p in files}, 'jobs': []}
    def save():
        (out/'campaign.json').write_text(json.dumps(state, indent=2)+'\n')
    save()
    for k in [3, 4]:
        for repetition in range(3):
            # Alternate order; do not select or discard runs after seeing results.
            profiles = [('allocated', .25), ('full', 1.0)]
            if repetition % 2: profiles.reverse()
            for profile, fraction in profiles:
                name = f'{k}in_max_repeat{repetition}_{profile}'
                folder, audit = out/name, out/(name+'.audit.json')
                argv = [sys.executable, str(wrapper), '--mapping-seconds', '300', '--driver', str(helper),
                        '--audit', str(audit), '--', '--mapper-root', str(ROOT),
                        '--region-classes', '256', '--phase-recovery', '1',
                        '--phase-time-fraction', str(fraction), '--output', str(folder),
                        '--k', str(k), '--objective', 'delay-area', '--preprocess', 'resyn2',
                        '--prepared-report', str(subjects), '--methods', 'egraph_bounded',
                        '--suite', 'epfl', '--benches', 'max']
                state['current_job'] = {'name': name, 'argv': argv}; save()
                with (out/(name+'.log')).open('w') as f:
                    result = subprocess.run(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
                j = {'name': name, 'k': k, 'repetition': repetition, 'profile': profile,
                     'fraction': fraction, 'argv': argv, 'exit_code': result.returncode}
                for key, p in [('report', folder/'report.json'), ('audit', audit)]:
                    if p.exists(): j.update({key: str(p), key+'_sha256': sha(p)})
                state['jobs'].append(j); state.pop('current_job', None); save()
                print(name, result.returncode, flush=True)
    state['status'] = 'all_attempted'; save()


if __name__ == '__main__':
    main()
