"""Resumable SHA-256 audit of frozen files, with durable per-file progress."""
import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import os
import stat
import subprocess
import time
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CUTOFF = int(datetime.datetime(2026, 10, 15, tzinfo=ZoneInfo('Asia/Hong_Kong')).timestamp() * 10**9)


def unchanged(item):
    try:
        p = ROOT / item['path']
        s = p.lstat()
        return stat.S_ISREG(s.st_mode) and not p.is_symlink() and (
            s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns
        ) == (item['device'], item['inode'], item['size'], item['mtime_ns'])
    except OSError:
        return False


def main():
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    with gzip.open(OUT / 'inventory.json.gz', 'rt') as stream:
        files = json.load(stream)
    groups = collections.defaultdict(list)
    for item in files:
        if item['size'] < 1048576 or item['mtime_ns'] >= CUTOFF or item['path'] in tracked:
            continue
        if not (item['path'].split('/')[0].startswith('results') or item['path'].split('/')[0] in ('research', 'data', 'res')):
            continue
        groups[item['size']].append(item)
    sample_cache = OUT / 'sample-candidates.json'
    if sample_cache.exists():
        samples = json.loads(sample_cache.read_text())
    else:
        samples_by_key = collections.defaultdict(list)
        count = 0
        for group in groups.values():
            if len({(x['device'], x['inode']) for x in group}) < 2:
                continue
            by_inode = {}
            for item in group:
                key = (item['device'], item['inode'])
                if not unchanged(item):
                    continue
                if key not in by_inode:
                    with (ROOT / item['path']).open('rb') as stream:
                        first = stream.read(8192)
                        stream.seek(max(0, item['size'] // 2 - 4096))
                        middle = stream.read(8192)
                        stream.seek(max(0, item['size'] - 8192))
                        last = stream.read(8192)
                    by_inode[key] = hashlib.sha256(first + middle + last).hexdigest()
                samples_by_key[(item['size'], by_inode[key])].append(item)
                count += 1
                if count % 500 == 0:
                    print(json.dumps({'phase': 'sampling', 'files': count}), flush=True)
        samples = [g for g in samples_by_key.values() if len({(x['device'], x['inode']) for x in g}) > 1]
        sample_cache.write_text(json.dumps(samples))
    hashes = {}
    log = OUT / 'sha256-progress.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line)
            hashes[(row['device'], row['inode'])] = row
    items = [item for group in samples for item in group]
    needed = {}
    for item in items:
        needed.setdefault((item['device'], item['inode']), item)
    total = sum(item['size'] for key, item in needed.items() if key not in hashes)
    print(json.dumps({'phase': 'hashing', 'unique_files': len(needed), 'remaining_gib': round(total / 2**30, 3)}), flush=True)
    done = count = 0
    last_report = time.monotonic()
    with log.open('a') as journal:
        for key, item in needed.items():
            if key in hashes or not unchanged(item):
                continue
            h = hashlib.sha256()
            with (ROOT / item['path']).open('rb') as stream:
                while chunk := stream.read(4 * 1024 * 1024):
                    h.update(chunk)
            if not unchanged(item):
                continue
            row = dict(item, sha256=h.hexdigest())
            journal.write(json.dumps(row) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
            hashes[key] = row
            done += item['size']
            count += 1
            if time.monotonic() - last_report >= 20:
                print(json.dumps({'phase': 'hashing', 'hashed_files_this_run': count, 'hashed_gib_this_run': round(done / 2**30, 3), 'remaining_gib': round((total - done) / 2**30, 3)}), flush=True)
                last_report = time.monotonic()
    exact = collections.defaultdict(list)
    for item in items:
        row = hashes.get((item['device'], item['inode']))
        if row and unchanged(item):
            exact[row['sha256']].append(item)
    result = [{'sha256': sha, 'files': group} for sha, group in exact.items()
              if len({(x['device'], x['inode']) for x in group}) > 1]
    (OUT / 'exact-duplicates.json').write_text(json.dumps(result))
    print(json.dumps({'phase': 'complete', 'duplicate_groups': len(result), 'paths': sum(len(g['files']) for g in result)}), flush=True)


if __name__ == '__main__':
    main()
