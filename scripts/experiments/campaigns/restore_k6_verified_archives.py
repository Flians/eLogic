"""Restore only SHA-verified archived /tmp artifacts; never overwrite differing bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def restore(archive, expected, destination):
    seen = set()
    decoder = subprocess.Popen(['zstd', '-q', '-dc', str(archive)], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=decoder.stdout, mode='r|') as reader:
        for member in reader:
            assert member.isfile() and member.name in expected and member.name not in seen
            path = destination / member.name
            assert path.resolve().is_relative_to(Path('/tmp')), path
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.restoring')
            assert not temporary.exists()
            with reader.extractfile(member) as source, temporary.open('xb') as target:
                shutil.copyfileobj(source, target)
            assert sha(temporary) == expected[member.name], path
            if path.exists():
                assert sha(path) == expected[member.name], 'differing existing file: ' + str(path)
                temporary.unlink()
            else:
                temporary.chmod(member.mode & 0o777)
                temporary.replace(path)
            seen.add(member.name)
    decoder.stdout.close()
    assert decoder.wait() == 0 and seen == set(expected)
    return len(seen)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    assert not a.output.exists()
    records = []
    for spec in json.loads((ROOT / 'results_v2/single_output_k6_20261020/baseline_archives.json').read_text()):
        archive, manifest = Path(spec['archive']), Path(spec['manifest'])
        assert sha(archive) == spec['sha256']
        expected = json.loads(manifest.read_text())['members']
        count = restore(archive, expected, Path('/tmp'))
        records.append(dict(archive=str(archive), sha256=sha(archive), restored_members=count))
    for label in ['k6-original-baselines-prefix36-20261020', 'k6-allocation-full40-20261020',
                  'k6-hyp-gia-prefix1-20261020', 'k6-hyp-legacy-cec-prefix3-20261020']:
        folder = ROOT / 'results_v2/archives'
        verified = json.loads((folder / (label + '.verified.json')).read_text())
        archive = folder / (label + '.tar.zst')
        manifest = json.loads((folder / (label + '.manifest.json')).read_text())
        assert sha(archive) == verified['archive_sha256']
        expected = {name: value['sha256'] for name, value in manifest['files'].items()}
        count = restore(archive, expected, Path('/'))
        records.append(dict(archive=str(archive), sha256=sha(archive), restored_members=count))
    binary = ROOT / 'results_v2/single_output_k6_20261020/e-rewriter.bfd'
    wanted = '6cd4f3cbebe43bd9d5e907f525acd51bf23b1fee0b3a711bf7c85093c4480b5c'
    assert sha(binary) == wanted
    target = Path('/tmp/emapper-esyn-baseline/e-rewriter/target/release/e-rewriter')
    archived = target.with_name('e-rewriter.restored-original-link')
    assert not archived.exists()
    target.rename(archived)
    shutil.copy2(binary, target)
    target.chmod(0o755)
    a.output.write_text(json.dumps(dict(records=records, esyn_bfd_sha256=wanted,
        original_link_preserved=str(archived), baseline_algorithms_changed=False), indent=2) + '\n')
    print('Verified and restored', sum(r['restored_members'] for r in records), 'members', flush=True)


if __name__ == '__main__':
    main()
