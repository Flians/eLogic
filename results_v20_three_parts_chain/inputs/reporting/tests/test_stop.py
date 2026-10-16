#!/usr/bin/env python3
"""A stopped batch must launch no engine and must not claim stage completion."""
import os, subprocess, tempfile
from pathlib import Path
project=Path(__file__).resolve().parents[3]
with tempfile.TemporaryDirectory(prefix='epm-stop-') as tmp:
    root=Path(tmp);(root/'.stop').touch()
    env=dict(os.environ,EPM_ROOT=str(root),EPM_JOBS='1')
    result=subprocess.run(['bash','scripts/experiments/run_all.sh','--no-build','--only','funnel'],cwd=project,env=env,capture_output=True,text=True)
    assert result.returncode!=0,result.stdout+result.stderr
    assert not (root/'funnel/.done').exists()
    assert len((root/'funnel/summary.tsv').read_text().splitlines())==1
    assert not list((root/'funnel').glob('r1.*'))
print('cooperative stop launches no engine and writes no completion marker')
