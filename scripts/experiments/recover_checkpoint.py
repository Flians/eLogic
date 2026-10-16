"""Validate an anytime checkpoint after SIGALRM without relabeling search as complete."""
import hashlib
import json
from pathlib import Path
import subprocess

from audit_mapped_ppa import audit


def recover(wd, library, benchmark, abc, returncode):
    wd, library, benchmark = map(Path, (wd, library, benchmark))
    if returncode not in (-14, 142):
        raise ValueError('checkpoint recovery is restricted to optimizer SIGALRM')
    text = (wd / 'log').read_text()
    if 'post_mapping_timeout=' not in text or '[i] epmapping running (' not in text:
        raise ValueError('deadline did not occur in the optimizer')
    # Native statistics start only after run() returns and its SIGALRM is cancelled.
    # An outer timer expiring during output/CEC must not become an optimizer TO.
    if any(marker in text for marker in ('[i] epmapping synth_cache lookups=',
                                        '[i] epmapping rounds=', '[i] epmapping area ')):
        raise ValueError('deadline occurred after optimization, not in the optimizer')
    record = json.loads((wd / 'epmapping_checkpoint.json').read_text())
    name = record['netlist']
    if record['version'] != 1 or Path(name).name != name or not name.endswith('.v'):
        raise ValueError('invalid checkpoint record')
    netlist = wd / name
    initial = audit(wd / 'epmapping_baseline.v', library)
    selected = audit(netlist, library)
    if (selected['area'] > initial['area'] + .0001 or selected['delay'] > initial['delay'] + .001
            or abs(selected['area'] - record['area']) > .0001
            or abs(selected['delay'] - record['delay']) > .001):
        raise ValueError('checkpoint failed independent physical area/delay audit')
    quote = lambda p: "'" + str(p).replace("'", "'\\''") + "'"
    command = f'read_genlib {quote(library)}; read -m {quote(netlist)}; cec -n {quote(benchmark)}'
    proof = subprocess.run([str(abc), '-c', command], capture_output=True, text=True, timeout=300)
    (wd / 'checkpoint_cec.log').write_text(proof.stdout + proof.stderr)
    if proof.returncode != 0 or 'Networks are equivalent' not in proof.stdout:
        raise ValueError('checkpoint CEC failed; see checkpoint_cec.log')
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result = dict(status='optimizer_timeout', raw_returncode=returncode, cec=1,
                  selection='checkpoint' if record['round'] or record.get('partial_round') else 'initial_checkpoint',
                  checkpoint=record, area=selected['area'], delay=selected['delay'],
                  netlist_sha256=digest(netlist), initial_sha256=digest(wd / 'epmapping_baseline.v'),
                  input_sha256=digest(benchmark), library_sha256=digest(library),
                  proof_sha256=digest(wd / 'checkpoint_cec.log'))
    (wd / 'checkpoint_selection.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workdir', type=Path, required=True)
    ap.add_argument('--library', type=Path, required=True)
    ap.add_argument('--benchmark', type=Path, required=True)
    ap.add_argument('--abc', type=Path, required=True)
    ap.add_argument('--returncode', type=int, required=True)
    args = ap.parse_args()
    print(json.dumps(recover(args.workdir, args.library, args.benchmark, args.abc, args.returncode)))
