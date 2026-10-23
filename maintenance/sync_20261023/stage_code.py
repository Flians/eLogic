"""Stage only the reviewed project files; keep local nested Overleaf metadata."""
from pathlib import Path
import json
import os
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
plan = json.loads((HERE / 'stage-code-plan.json').read_text())
env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')


def git(*args, **kwargs):
    return subprocess.check_output(['git', '-c', 'gc.auto=0', *args], cwd=ROOT, env=env, **kwargs)


git('add', '-A', '--', *plan['paths'], *plan['submodules'])
git('add', '-f', '--', 'Cargo.lock')
# Explicit blobs avoid recording unregistered nested Overleaf repositories as
# unusable gitlinks. Their .git directories remain local and untouched.
rows = bytearray()
for name in plan['paper_files']:
    path = ROOT / name
    assert '.git' not in Path(name).parts
    oid = git('hash-object', '-w', '--no-filters', '--', name).decode().strip()
    mode = '100755' if path.stat().st_mode & 0o111 else '100644'
    rows.extend(f'{mode} {oid}\t{name}\0'.encode())
git('update-index', '-z', '--index-info', input=bytes(rows))
staged = git('diff', '--cached', '--name-only', '-z').decode().split('\0')
allowed = plan['paths'] + plan['submodules'] + ['emapper_paper', 'epmapping_paper']
unexpected = [p for p in staged if p and not any(p == x or p.startswith(x + '/') for x in allowed)]
assert not unexpected, unexpected
for paper in ['emapper_paper', 'epmapping_paper']:
    modes = git('ls-files', '--stage', '--', paper).decode().splitlines()
    assert modes and all(not line.startswith('160000 ') for line in modes)
assert not git('ls-files', '--stage', '--', 'tools/mockturtle-baseline')
(HERE / 'staged-code.json').write_text(json.dumps({'files': [p for p in staged if p],
    'unregistered_paper_gitlinks': False, 'local_nested_git_metadata_modified': False,
    'raw_database_staged': 'lib_expr2cost.json' in staged}, indent=2) + '\n')
print('Reviewed code/docs staged:', len([p for p in staged if p]), 'paths; paper sources are real files.')
