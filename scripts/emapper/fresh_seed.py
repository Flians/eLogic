"""Unchanged emap-delay component, with a separately receipted mapping charge.

No resyn2, mapping search changes, or baseline-result writes. The component's
whole process is a conservative charge, including its own load/export overhead;
subsequent normalization and independent CEC are outside this mapping charge.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from provenance import ROOT, sha


def remaining_mapping_seconds(limit, seed_seconds):
    import math
    if (isinstance(seed_seconds, bool) or not math.isfinite(seed_seconds)
            or seed_seconds < 0 or seed_seconds >= limit):
        raise ValueError('fresh seed has exhausted or invalidated mapping budget')
    return limit - seed_seconds


def generate(binary, library, subject, folder, limit):
    binary, library, subject, folder = map(lambda p: Path(p).resolve(),
                                           [binary, library, subject, folder])
    folder.mkdir(exist_ok=False)
    raw, normalized = folder/'raw.v', folder/'seed.v'
    argv = [str(binary), str(library), str(subject), str(raw), 'delay', 'emap']
    adapter = ROOT/'scripts/experiments/port_identifiers.py'
    receipt = dict(argv=argv, binary_sha256=sha(binary), library_sha256=sha(library),
                   prepared_sha256=sha(subject), adapter_sha256=sha(adapter),
                   baseline_algorithm_changed=False, baseline_dataset_written=False,
                   CEC_in_clock=False,
                   timing_scope='whole unchanged emap-delay process; includes its own load/export; normalization and CEC excluded')
    started = time.monotonic()
    try:
        with (folder/'stdout.log').open('x') as out, (folder/'stderr.log').open('x') as err:
            proc = subprocess.run(argv, stdout=out, stderr=err, timeout=limit)
        receipt.update(exit_code=proc.returncode, mapping_seconds=time.monotonic()-started)
        proc.check_returncode()
        remaining_mapping_seconds(limit, receipt['mapping_seconds'])
        receipt['raw_sha256'] = sha(raw)
        # Necessary scalar-port escaping, same adapter as formal baseline input I/O.
        spec = importlib.util.spec_from_file_location('emapper_seed_ports', adapter)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        normalized.write_text(module.escape_indexed_scalars(raw.read_text()))
        receipt.update(status='ok', seed=str(normalized), seed_sha256=sha(normalized))
    except Exception as exc:
        receipt.update(status='failed', error=type(exc).__name__+': '+str(exc))
        receipt.setdefault('mapping_seconds', time.monotonic()-started)
        raise
    finally:
        (folder/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return normalized, receipt
