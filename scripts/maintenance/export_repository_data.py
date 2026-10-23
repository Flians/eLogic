"""Publish compact, lossless records without deleting local raw experiments.

All outcome statuses use the same selection. Large intermediate netlists,
builds, third-party copies and caches remain local, explicitly outside this export.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[2]
SKIP = {'.git', 'target', 'build', 'build-release', 'CMakeFiles', '__pycache__',
        'node_modules', 'deps', 'incremental', '.fingerprint', 'cache', 'caches',
        'stock', 'vendor', 'third_party', 'tools', 'abc', 'abc_old', 'abc-gamora'}
TEXT = {'.md', '.txt', '.csv', '.tsv', '.py', '.sh', '.rs', '.cpp', '.hpp', '.h',
        '.toml', '.lock', '.patch', '.tex', '.bib', '.cls', '.cmake', '.genlib'}
META = ('report', 'campaign', 'valid', 'audit', 'manifest', 'provenance', 'freeze',
        'sha', 'metric', 'budget', 'plan', 'command', 'config', 'selection', 'status',
        'summary', 'record', 'checkpoint', 'test', 'integration', 'library', 'receipt',
        'complete', 'allocation', 'diagnos', 'execution', 'comparison', 'adoption')
SECRET = re.compile(rb'(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----)')
MAX_TAR = 64 * 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def wanted(path, size, selected):
    if path in selected:
        return True
    if path.suffix in TEXT and size <= 4 * 1024**2:
        return True
    if path.suffix == '.json' and size <= 8 * 1024**2 and any(k in path.name.lower() for k in META):
        return True
    if path.name.endswith('.stages.jsonl') and size <= 8 * 1024**2:
        return True
    if (path.suffix == '.log' or path.name == 'log') and size <= 256 * 1024:
        return True
    return path.suffix in {'.v', '.aig'} and size <= 64 * 1024


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'data')
    p.add_argument('--include-database', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    archives = out / 'archives'
    archives.mkdir(parents=True, exist_ok=True)
    assert not (out / 'catalog.json').exists(), 'new export output required'
    selected = set()
    study = ROOT / 'results_v2/cooperative_mapping_20261022'
    for n in ['v9_control', 'v9_epfl']:
        cp = json.loads((study / n / 'campaign.json').read_text())
        for job in cp['jobs']:
            folder = Path(job['report']).parent
            selected.update(q for q in folder.rglob('*') if q.is_file() and
                            (q.name in {'raw.v', 'mapped.v', 'mapped.json'} or
                             'functional-cec' in q.parts) and 'incumbents' not in q.parts)
    roots = [q for q in (ROOT / 'results_v2').iterdir() if q.is_dir() and not q.is_symlink()]
    roots += [q for q in ROOT.glob('results_v*') if q.is_dir() and q.name != 'results_v2']
    roots += [q for q in (ROOT / 'research').iterdir() if q.is_dir() and not q.is_symlink()]
    catalog = {'schema_version': 1, 'scope': 'compact records, sources, failures and selected V9 final CEC artifacts; not all raw data',
               'raw_data_deleted': False, 'selection_depends_on_outcome': False,
               'excluded_subtrees': sorted(SKIP), 'studies': [], 'omitted_sensitive_files': []}
    with gzip.open(out / 'contents.jsonl.gz', 'wt', encoding='utf-8') as contents, \
         gzip.open(out / 'local_inventory.jsonl.gz', 'wt', encoding='utf-8') as inventory:
        for ordinal, root in enumerate(sorted(roots)):
            label = str(root.relative_to(ROOT)).replace('/', '__')
            entry = {'path': str(root.relative_to(ROOT)), 'files_published': 0, 'payload_bytes': 0,
                     'files_retained_local': 0, 'bytes_retained_local': 0, 'archives': []}
            archive = None
            current = None
            used = 0
            part = 0
            def finish():
                nonlocal archive
                if archive is not None:
                    archive.close()
                    entry['archives'].append({'path': str(current.relative_to(out)),
                                              'bytes': current.stat().st_size, 'sha256': sha(current)})
                    archive = None
            for base, dirs, files in os.walk(root, followlinks=False):
                for d in list(dirs):
                    q = Path(base) / d
                    if d in SKIP or q.is_symlink():
                        dirs.remove(d)
                        inventory.write(json.dumps({'path': str(q.relative_to(ROOT)), 'kind': 'excluded_subtree'}) + '\n')
                dirs.sort()
                for name in sorted(files):
                    path = Path(base) / name
                    if path.is_symlink() or not path.is_file():
                        continue
                    stat = path.stat()
                    relative = str(path.relative_to(ROOT))
                    publish = wanted(path, stat.st_size, selected)
                    if publish:
                        data = path.read_bytes()
                        if SECRET.search(data):
                            publish = False
                            catalog['omitted_sensitive_files'].append(relative)
                    inventory.write(json.dumps({'path': relative, 'bytes': stat.st_size,
                                                'published': publish}) + '\n')
                    if not publish:
                        entry['files_retained_local'] += 1
                        entry['bytes_retained_local'] += stat.st_size
                        continue
                    assert path.stat().st_mtime_ns == stat.st_mtime_ns, 'source changed during export: ' + relative
                    cost = len(data) + 4096
                    if archive is None or used + cost > MAX_TAR:
                        finish()
                        part += 1
                        current = archives / f'{label}.part{part:03}.tar.gz'
                        archive = tarfile.open(current, 'w:gz', compresslevel=1)
                        used = 0
                    info = tarfile.TarInfo(relative)
                    info.size = len(data)
                    info.mode = stat.st_mode & 0o777
                    archive.addfile(info, io.BytesIO(data))
                    digest = hashlib.sha256(data).hexdigest()
                    contents.write(json.dumps({'path': relative, 'archive': str(current.relative_to(out)),
                                               'bytes': len(data), 'sha256': digest}) + '\n')
                    entry['files_published'] += 1
                    entry['payload_bytes'] += len(data)
                    used += cost
            finish()
            catalog['studies'].append(entry)
            if ordinal % 40 == 0:
                print('exported', ordinal + 1, 'of', len(roots), flush=True)
    if a.include_database:
        source = ROOT / 'lib_expr2cost.json'
        assets = out / 'assets'
        assets.mkdir(exist_ok=True)
        temporary = assets / 'database.tmp.gz'
        h = hashlib.sha256()
        before = source.stat()
        with source.open('rb') as src, temporary.open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=1, mtime=0) as dst:
                for block in iter(lambda: src.read(1024 * 1024), b''):
                    h.update(block)
                    dst.write(block)
        assert source.stat().st_mtime_ns == before.st_mtime_ns
        parts = []
        with temporary.open('rb') as src:
            while block := src.read(48 * 1024**2):
                target = assets / f'lib_expr2cost.json.gz.part{len(parts)+1:03}'
                target.write_bytes(block)
                parts.append({'path': str(target.relative_to(out)), 'bytes': len(block), 'sha256': sha(target)})
        temporary.unlink()
        catalog['database'] = {'original_path': 'lib_expr2cost.json', 'bytes': before.st_size,
                               'sha256': h.hexdigest(), 'encoding': 'concatenated gzip stream parts', 'parts': parts}
    catalog['published_files'] = sum(e['files_published'] for e in catalog['studies'])
    catalog['archive_bytes'] = sum(p['bytes'] for e in catalog['studies'] for p in e['archives'])
    for name in ['contents.jsonl.gz', 'local_inventory.jsonl.gz']:
        catalog[name] = {'sha256': sha(out / name), 'bytes': (out / name).stat().st_size}
    (out / 'catalog.json').write_text(json.dumps(catalog, indent=2) + '\n')
    print('published files', catalog['published_files'], 'archive bytes', catalog['archive_bytes'],
          'sensitive omissions', len(catalog['omitted_sensitive_files']), flush=True)


if __name__ == '__main__':
    main()
