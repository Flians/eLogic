"""Verify archive hashes; --deep also verifies each exported member's bytes."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def digest(f):
    h = hashlib.sha256()
    for block in iter(lambda: f.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--deep', action='store_true')
    a = p.parse_args()
    catalog = json.loads((a.data / 'catalog.json').read_text())
    expected = {}
    if a.deep:
        with gzip.open(a.data / 'contents.jsonl.gz', 'rt') as f:
            for line in f:
                row = json.loads(line)
                key = (row['archive'], row['path'])
                assert key not in expected, 'duplicate archive member'
                expected[key] = row
    count = 0
    for study in catalog['studies']:
        for archive in study['archives']:
            path = a.data / archive['path']
            assert path.stat().st_size == archive['bytes']
            with path.open('rb') as f:
                assert digest(f) == archive['sha256'], str(path)
            if a.deep:
                with tarfile.open(path, 'r:gz') as tar:
                    for member in tar:
                        assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
                        row = expected.pop((archive['path'], member.name))
                        assert member.size == row['bytes']
                        assert digest(tar.extractfile(member)) == row['sha256']
                        count += 1
    for name in ['contents.jsonl.gz', 'local_inventory.jsonl.gz']:
        with (a.data / name).open('rb') as f:
            assert digest(f) == catalog[name]['sha256']
    if 'database' in catalog:
        db = catalog['database']
        with __import__('tempfile').TemporaryFile() as combined:
            for part in db['parts']:
                path = a.data / part['path']
                with path.open('rb') as f:
                    assert digest(f) == part['sha256']
                with path.open('rb') as f:
                    __import__('shutil').copyfileobj(f, combined)
            if a.deep:
                combined.seek(0)
                with gzip.GzipFile(fileobj=combined) as f:
                    assert digest(f) == db['sha256'], 'database decompression differs'
    assert not a.deep or not expected, 'exported members missing'
    print(json.dumps({'archives_verified': sum(len(s['archives']) for s in catalog['studies']),
                      'members_verified': count if a.deep else None,
                      'database_verified': 'database' in catalog,
                      'raw_data_deleted': False}))


if __name__ == '__main__':
    main()
