"""Build/source receipts for the canonical delivery entrypoint."""
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git_revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None

def source_files():
    paths = [ROOT/'Cargo.toml', ROOT/'Cargo.lock', ROOT/'emapper/Cargo.toml',
             ROOT/'src/bin/emapper_kernel.rs', ROOT/'src/bin/emapper_measure.rs']
    paths += list((ROOT/'emapper/src').rglob('*.rs'))
    return sorted(paths)

def hashes(paths):
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}

if __name__ == '__main__':
    import json
    binaries = [ROOT/'target/release'/n for n in ['emapper_kernel', 'emapper_measure']]
    receipt = dict(source=hashes(source_files()), binaries=hashes(binaries))
    (ROOT/'target/release/emapper-build.json').write_text(json.dumps(receipt, indent=2)+'\n')
