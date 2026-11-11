#!/usr/bin/env python3
"""Merge immutable L4 profile records without changing keys, payloads or sources."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def build(destination, sources, provenance):
    start = time.monotonic()
    assert not destination.exists(), ('library already exists', destination)
    destination.mkdir(parents=True)
    records = {}
    input_records = 0
    for label, directory in sources.items():
        for source in sorted(directory.glob('*.json')):
            record = json.loads(source.read_text())
            assert record['schema'] == 1 and source.stem == record['key']
            assert len(record['key']) == 64 and all(c in '0123456789abcdef' for c in record['key'])
            assert isinstance(record['covers'], list)
            digest = sha(source)
            entry = records.get(source.name)
            input_records += 1
            if entry:
                # A finite search must publish the same pool for an identical key.
                assert entry['sha256'] == digest, ('conflicting candidate pools', source)
                entry['sources'].append(label)
            else:
                target = destination / source.name
                target.write_bytes(source.read_bytes())
                records[source.name] = dict(sha256=digest, bytes=source.stat().st_size,
                                            candidates=len(record['covers']), sources=[label])
    elapsed = time.monotonic() - start
    manifest = dict(schema=1, mode='shared-readonly-profile-v1', provenance=provenance,
                    input_records=input_records, unique_queries=len(records),
                    deduplicated_records=input_records-len(records),
                    candidate_count=sum(r['candidates'] for r in records.values()),
                    record_bytes=sum(r['bytes'] for r in records.values()),
                    merge_seconds=elapsed, records=records)
    (destination / 'library_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for path in destination.iterdir():
        path.chmod(0o444)
    destination.chmod(0o555)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    report = build(args.output.resolve(), {str(p.resolve()):p.resolve() for p in args.sources}, {})
    print(json.dumps({k:v for k,v in report.items() if k!='records'}, indent=2))


if __name__ == '__main__':
    main()
