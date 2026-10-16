#!/usr/bin/env python3
"""DATE runs reject window synthesis and override an inherited enabling flag."""
import os
from pathlib import Path
import subprocess
import tempfile

project = Path(__file__).resolve().parents[3]
with tempfile.TemporaryDirectory(prefix='epm-window-off-') as directory:
    root = Path(directory)
    marker = root / 'launched'
    engine = root / 'engine'
    engine.write_text('#!/usr/bin/env bash\nprintf "%s" "$EPMAPPING_WINDOW_SYN" > "$EPM_TEST_MARKER"\nprintf "%s" "$EPMAPPING_SUPER" > "$EPM_TEST_MARKER.db"\n')
    engine.chmod(0o755)
    environment = dict(os.environ, EPM_BIN=str(engine), EPM_ROOT=str(root),
                       EPM_TEST_MARKER=str(marker), EPMAPPING_WINDOW_SYN='1')
    command = ['bash', '-c', 'source scripts/experiments/common.sh; run_engine "$@"',
               '_', str(root / 'run'), 'c7552', '1', 'direct']
    rejected = subprocess.run(command + ['EPMAPPING_WINDOW_SYN=1'], cwd=project,
                              env=environment, capture_output=True)
    assert rejected.returncode == 2
    assert not marker.exists()
    completed = subprocess.run(command, cwd=project, env=environment, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert marker.read_text() == '0'
    assert 'EPMAPPING_WINDOW_SYN=0' in (root / 'run/experiment_env.txt').read_text()
    completed = subprocess.run(command + ['EPMAPPING_SUPER=off'], cwd=project, env=environment, capture_output=True)
    assert completed.returncode == 0
    assert Path(str(marker)+'.db').read_text() == ''
    flags=(root/'run/experiment_env.txt').read_text().splitlines()
    assert [f for f in flags if f.startswith('EPMAPPING_SUPER=')][-1] == 'EPMAPPING_SUPER='
print('window enabling rejected; inherited enabling flag overridden; off flag recorded')
