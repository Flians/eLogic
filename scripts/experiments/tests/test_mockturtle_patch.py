#!/usr/bin/env python3
"""Check first application, repeat configuration and conflict preservation."""
from pathlib import Path
import shutil
import subprocess
import tempfile

project = Path(__file__).resolve().parents[3]
relative = Path('include/mockturtle/utils/tech_library.hpp')
patch = project / 'patches/mockturtle/tech-library-function-buckets.patch'
with tempfile.TemporaryDirectory(prefix='epm-patch-') as directory:
    root = Path(directory)
    (root / 'scripts').mkdir()
    (root / 'patches/mockturtle').mkdir(parents=True)
    submodule = root / 'tools/mockturtle'
    header = submodule / relative
    header.parent.mkdir(parents=True)
    shutil.copy(project / 'scripts/apply_mockturtle_patches.sh', root / 'scripts')
    shutil.copy(patch, root / 'patches/mockturtle')
    header.write_bytes((project / 'tools/mockturtle' / relative).read_bytes())
    subprocess.run(['git', 'apply', '--reverse', str(patch)], cwd=submodule, check=True)
    command = ['bash', str(root / 'scripts/apply_mockturtle_patches.sh')]
    subprocess.run(command, check=True)
    expected = header.read_bytes()
    subprocess.run(command, check=True)
    assert header.read_bytes() == expected
    header.write_text('conflicting user edit\n')
    result = subprocess.run(command, capture_output=True)
    assert result.returncode != 0
    assert header.read_text() == 'conflicting user edit\n'
print('patch application, idempotence and conflict preservation passed')
