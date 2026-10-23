"""Merge audited, frozen data copies; retains paths and an append-only journal.

Usage: python3 merge_frozen_duplicates.py exact-duplicates.json
Only files >= 1 MiB from the audit are candidates. Git-tracked files are excluded.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CUTOFF = int(datetime.datetime(2026, 10, 15, tzinfo=ZoneInfo('Asia/Hong_Kong')).timestamp() * 10**9)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def signature(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_mode, s.st_uid, s.st_gid)


def attributes(path):
    return {name: os.getxattr(path, name) for name in os.listxattr(path)}


def main():
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    groups = json.loads(Path(sys.argv[1]).read_text())
    plan = []
    for group in groups:
        eligible = []
        for item in group['files']:
            rel = Path(item['path'])
            if rel.is_absolute() or '..' in rel.parts or item['path'] in tracked:
                continue
            if not (rel.parts[0].startswith('results') or rel.parts[0] in ('research', 'data', 'res')):
                continue
            path = ROOT / rel
            if path.resolve() != path or item['mtime_ns'] >= CUTOFF:
                continue
            try:
                s = path.lstat()
                if not stat.S_ISREG(s.st_mode) or (s.st_size, s.st_mtime_ns, s.st_dev, s.st_ino) != (item['size'], item['mtime_ns'], item['device'], item['inode']):
                    continue
                eligible.append((path, s))
            except OSError:
                continue
        by_metadata = {}
        for p, s in eligible:
            by_metadata.setdefault((s.st_dev, s.st_mode, s.st_uid, s.st_gid), []).append((p, s))
        for bucket in by_metadata.values():
            # Prefer an existing shared inode so rerunning remains idempotent.
            bucket.sort(key=lambda x: (-x[1].st_nlink, str(x[0])))
            if len({s.st_ino for _, s in bucket}) > 1:
                plan.append((group['sha256'], bucket))
    (OUT / 'merge-plan.json').write_text(json.dumps([
        {'sha256': sha, 'canonical': str(bucket[0][0].relative_to(ROOT)),
         'copies': [str(p.relative_to(ROOT)) for p, _ in bucket[1:]]}
        for sha, bucket in plan], indent=2) + '\n')
    merged = released = skipped = 0
    with (OUT / 'merge-journal.jsonl').open('a') as journal:
        def record(row):
            journal.write(json.dumps(row) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        for sha, bucket in plan:
            canonical, original = bucket[0]
            if signature(canonical.lstat()) != signature(original):
                skipped += len(bucket) - 1
                continue
            for path, expected in bucket[1:]:
                current = path.lstat()
                if current.st_ino == original.st_ino:
                    continue
                if signature(current) != signature(expected):
                    skipped += 1
                    continue
                if attributes(path) != attributes(canonical):
                    skipped += 1
                    continue
                if signature(canonical.lstat()) != signature(original) or signature(path.lstat()) != signature(current):
                    skipped += 1
                    continue
                temp = path.with_name('.' + path.name + '.dedup-' + str(os.getpid()))
                row = {'path': str(path.relative_to(ROOT)), 'canonical': str(canonical.relative_to(ROOT)),
                       'sha256': sha, 'size': current.st_size, 'mtime_ns': current.st_mtime_ns,
                       'mode': stat.S_IMODE(current.st_mode), 'old_inode': current.st_ino}
                record(dict(row, status='prepared'))
                try:
                    os.link(canonical, temp)
                    if signature(path.lstat()) != signature(current):
                        skipped += 1
                        continue
                    os.replace(temp, path)
                finally:
                    if temp.exists():
                        temp.unlink()
                result = path.lstat()
                assert (result.st_dev, result.st_ino) == (original.st_dev, original.st_ino)
                freed = current.st_blocks * 512 if current.st_nlink == 1 else 0
                released += freed
                merged += 1
                record(dict(row, status='merged', released_bytes=freed, inode=result.st_ino))
            assert digest(canonical) == sha
            print(json.dumps({'merged': merged, 'released_gib': round(released / 2**30, 3)}), flush=True)
    completed = [json.loads(line) for line in (OUT / 'merge-journal.jsonl').read_text().splitlines()
                 if json.loads(line)['status'] == 'merged']
    result = {'merged_files': len(completed), 'released_bytes': sum(row['released_bytes'] for row in completed), 'skipped_files_last_run': skipped,
              'cutoff': '2026-10-15T00:00:00+08:00', 'min_size': 1048576,
              'note': 'Merged paths share an inode. Copy a file to a new inode before editing it.'}
    (OUT / 'merge-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
