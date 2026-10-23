#!/usr/bin/env python3
"""Apply the four fixed project headers without overwriting unrelated edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

OVERLAY = Path(__file__).resolve().parent / 'mockturtle-overlay'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def apply(destination, check=False):
    manifest = json.loads((OVERLAY / 'manifest.json').read_text())
    pending = []
    # Validate the entire set before modifying any target.
    for entry in manifest['files']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('invalid overlay path')
        source = OVERLAY / relative
        target = destination / relative
        data = source.read_bytes()
        if sha(data) != entry['overlay_sha256']:
            raise ValueError(f'overlay checksum mismatch: {relative}')
        if not target.is_file() or target.is_symlink():
            raise ValueError(f'missing or nonregular target: {target}')
        current = sha(target.read_bytes())
        if current == entry['overlay_sha256']:
            continue
        if check or current != entry['upstream_sha256']:
            raise ValueError(f'local changes or incompatible upstream: {target}; no files replaced')
        pending.append((target, data, target.stat().st_mode & 0o777))
    for target, data, mode in pending:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.migbalance-overlay-', delete=False) as f:
                temporary = Path(f.name)
                f.write(data)
            os.chmod(temporary, mode)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return len(pending)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, default=OVERLAY.parent / 'mockturtle')
    p.add_argument('--check', action='store_true', help='verify installed project headers without writes')
    a = p.parse_args()
    try:
        changed = apply(a.destination.resolve(), a.check)
    except (OSError, ValueError, KeyError) as error:
        p.exit(1, f'[mockturtle-overlay] {error}\n')
    print(f'[mockturtle-overlay] verified four headers; replaced {changed}')


if __name__ == '__main__':
    main()
