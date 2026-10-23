#!/usr/bin/env python3
"""Check first application, repeat configuration and conflict preservation."""
from pathlib import Path
import shutil
import subprocess
import tempfile

project = Path(__file__).resolve().parents[3]
relatives = [Path('include/mockturtle') / p for p in ['utils/tech_library.hpp','algorithms/emap.hpp','views/cell_view.hpp','algorithms/cut_enumeration.hpp']]
patches = [project / 'tools/patches/mockturtle' / p for p in ['tech-library-function-buckets.patch','shared-mapped-library.patch','cut-prefix-enumeration.patch']]
with tempfile.TemporaryDirectory(prefix='epm-patch-') as directory:
    root = Path(directory)
    (root / 'scripts').mkdir()
    (root / 'tools/patches/mockturtle').mkdir(parents=True)
    submodule = root / 'tools/mockturtle'
    header = submodule / relatives[0]
    for relative in relatives:
        target=submodule/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((project/'tools/mockturtle'/relative).read_bytes())
    shutil.copy(project / 'scripts/apply_mockturtle_patches.sh', root / 'scripts')
    shutil.copy(project / 'tools/apply_mockturtle_overlay.py', root / 'tools')
    shutil.copytree(project / 'tools/mockturtle-overlay', root / 'tools/mockturtle-overlay')
    for patch in patches:
        shutil.copy(patch, root / 'tools/patches/mockturtle')
        subprocess.run(['git', 'apply', '--reverse', str(patch)], cwd=submodule, check=True)
    command = ['bash', str(root / 'scripts/apply_mockturtle_patches.sh')]
    subprocess.run(command, check=True)
    expected = {str(r):(submodule/r).read_bytes() for r in relatives}
    subprocess.run(command, check=True)
    assert all((submodule/r).read_bytes()==expected[str(r)] for r in relatives)
    header.write_text('conflicting user edit\n')
    result = subprocess.run(command, capture_output=True)
    assert result.returncode != 0
    assert header.read_text() == 'conflicting user edit\n'
    header.write_bytes(expected[str(relatives[0])])
    second=submodule/relatives[1];second.write_text('conflicting mapper edit\n')
    result=subprocess.run(command,capture_output=True)
    assert result.returncode!=0 and second.read_text()=='conflicting mapper edit\n'
    assert (submodule/relatives[2]).read_bytes()==expected[str(relatives[2])]
    second.write_bytes(expected[str(relatives[1])])
    prefix=submodule/relatives[3];prefix.write_text('conflicting cut enumeration edit\n')
    result=subprocess.run(command,capture_output=True)
    assert result.returncode!=0 and prefix.read_text()=='conflicting cut enumeration edit\n'
print('patch application, idempotence and conflict preservation passed')
