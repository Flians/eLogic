"""Exercise normal completion and a real optimizer SIGALRM with recoverable outputs."""
import json
import os
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import shutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
from audit_mapped_ppa import audit
from recover_checkpoint import recover
from timeout_selection import prepare, adapt_row, selected_netlist, read_rows

frozen = ROOT / 'results_v20_three_parts_chain/inputs'
# Pin the checkpoint regression to the paper's admitted profile, rather than a
# pre-pivot pilot whose usb_phy area (21.38) is no longer this profile's result.
paper_campaign = ROOT / 'results_v28_three_parts_query_budget'
goal = json.loads((paper_campaign / 'cold/manifest.json').read_text())['goal']
reference = next(row for row in read_rows(paper_campaign / 'cold', 'tab_main')
                 if row['stage'] == 'ae' and row['design'] == 'usb_phy')
expected_area = float(reference['out_a'])
expected_delay = float(reference['out_d'])
env = {k: v for k, v in os.environ.items() if not k.startswith(('EPMAPPING_', 'MAP_EGG_'))}
env.update(goal)
env.update(ABC=str(frozen / 'abc'), EPMAPPING_BASELINE='high_effort',
           EPMAPPING_WRITE_BASELINE='1', EPMAPPING_AREA_MAPPING='1',
           EPMAPPING_RELAX_REQUIRED='100', EPMAPPING_CUT_LIMIT='12',
           EPMAPPING_ROUNDS='6', EPMAPPING_SUPER=str(frozen / 'database.super'),
           EPMAPPING_CHECKPOINT='1', RAYON_NUM_THREADS='1')

with tempfile.TemporaryDirectory(prefix='epm-anytime-') as folder:
    root = Path(folder)
    for name, seconds in [('complete', 1000), ('interrupted', 1)]:
        wd = root / name
        env.update(EPMAPPING_POST_TIMEOUT=str(seconds), MAP_EGG_PROFILE_CACHE_DIR=str(root / 'cache'),
                   EPMAPPING_CHECKPOINT_INTERVAL='0.01' if name == 'interrupted' else '30')
        started = time.monotonic()
        q = subprocess.run([str(ROOT / 'build/epmapping/epmapping_check'),
                            str(frozen / 'asap7.genlib'), str(frozen / 'benchmarks/usb_phy.aig'), str(wd)],
                           env=env, capture_output=True, text=True, timeout=90)
        elapsed = math.ceil(time.monotonic() - started)
        assert q.returncode == (0 if name == 'complete' else -14), (q.returncode, q.stdout, q.stderr)
        manifest = wd / 'epmapping_checkpoint.json'
        assert manifest.exists(), 'optimizer lost its last legal netlist on completion/deadline'
        record = json.loads(manifest.read_text())
        candidate = wd / record['netlist']
        assert candidate.stat().st_size > 0
        initial = audit(wd / 'epmapping_baseline.v', frozen / 'asap7.genlib')
        selected = audit(candidate, frozen / 'asap7.genlib')
        assert selected['area'] <= initial['area'] + .0001, (record, selected, initial)
        assert selected['delay'] <= initial['delay'] + .001, (record, selected, initial)
        assert abs(selected['area'] - record['area']) < .0001
        assert abs(selected['delay'] - record['delay']) < .001
        proof = subprocess.run([str(frozen / 'abc'), '-c',
            f"read_genlib '{frozen / 'asap7.genlib'}'; read -m '{candidate}'; cec -n '{frozen / 'benchmarks/usb_phy.aig'}'"],
            capture_output=True, text=True, timeout=30)
        assert proof.returncode == 0 and 'Networks are equivalent' in proof.stdout, proof.stdout + proof.stderr
        if name == 'complete':
            assert record['round'] > 0, record
            assert abs(selected['area'] - expected_area) < .0001, (record, reference)
            assert abs(selected['delay'] - expected_delay) < .001, (record, reference)
            final = audit(wd / 'epmapping_mapped.v', frozen / 'asap7.genlib')
            assert abs(selected['area'] - final['area']) < .0001
            assert abs(selected['delay'] - final['delay']) < .001
        else:
            assert record.get('partial_round') and selected['area'] < initial['area'], record
            (wd / 'log').write_text(q.stdout + q.stderr)
            recovered = recover(wd, frozen / 'asap7.genlib', frozen / 'benchmarks/usb_phy.aig',
                                frozen / 'abc', q.returncode)
            assert recovered['status'] == 'optimizer_timeout' and recovered['cec'] == 1
            try:
                recover(wd, frozen / 'asap7.genlib', frozen / 'benchmarks/usb_phy.aig', frozen / 'abc', 0)
            except ValueError:
                pass
            else:
                raise AssertionError('normal completion must not be reclassified as a timeout')
            # Feed the real one-second deadline into the same selection adapter used
            # by paper batches. Its manifest truthfully declares the test budget.
            campaign = root / 'campaign'
            run = campaign / 'tab_main/ae.usb_phy'
            shutil.copytree(wd, run)
            inputs = campaign / 'inputs'
            (inputs / 'benchmarks').mkdir(parents=True)
            for f in ['abc', 'asap7.genlib']: shutil.copy2(frozen / f, inputs / f)
            shutil.copy2(frozen / 'benchmarks/usb_phy.aig', inputs / 'benchmarks/usb_phy.aig')
            (campaign / 'manifest.json').write_text(json.dumps(dict(post_mapping_timeout=seconds, timeout_selection='checkpoint')))
            fields = ['ae', 'usb_phy', 'high_effort', '6', str(initial['area']), str(initial['delay']),
                      '-', '-', 'n/a', 'n/a', '-', '-', str(elapsed), '142']
            (run / 'row.tsv').write_text('\t'.join(fields) + '\n')
            assert prepare(campaign) == 1 and prepare(campaign) == 0
            header = 'stage design recipe rounds base_a base_d out_a out_d dA% dD% rnd cec secs rc'.split()
            adapted = adapt_row(campaign / 'tab_main', dict(zip(header, fields)))
            assert adapted['raw_status'] == 'timeout' and adapted['selection'] == 'checkpoint'
            assert adapted['rc'] == '142' and adapted['cec'] == '1'
            assert selected_netlist(run).read_bytes() == candidate.read_bytes()
        print(name, json.dumps(record), flush=True)
print('Completed and interrupted searches preserve independently checked legal checkpoints')
