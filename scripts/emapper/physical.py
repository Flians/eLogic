"""Canonical physical measurement for supplied covers; no mapping is invoked."""
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

METRIC_MODEL = 'all physical instances; max(rise,fall) block pin delay; no load or slew'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def script_path(path):
    path = Path(path).resolve()
    if any(c in str(path) for c in '\n\r;"\\'):
        raise ValueError('unsupported physical-evaluation path')
    return path


def measure_cover(library, netlist, folder, measure_binary, yosys):
    """Pure format conversion + the same Rust evaluator as final reporting."""
    library, netlist, folder, measure_binary = map(script_path, [library, netlist, folder, measure_binary])
    yosys = Path(shutil.which(str(yosys)) or str(yosys)).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    normalized, netjson = folder/'mapped.v', folder/'mapped.json'
    command = (f'read_verilog -lib "{library.parent / "cells.v"}"; read_verilog "{netlist}"; '
               f'hierarchy -check -auto-top; rename -top mapped; '
               f'write_verilog -noattr -noexpr "{normalized}"; write_json "{netjson}"')
    commands = [[str(yosys), '-Q', '-T', '-p', command],
                [str(measure_binary), str(library), str(netjson)]]
    for argv, name in zip(commands, ['normalize.log', 'measure.log']):
        result = subprocess.run(argv, capture_output=True, text=True)
        (folder/name).write_text(result.stdout + result.stderr)
        (folder/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        result.check_returncode()
    metrics = json.loads((folder/'measure.log').read_text())
    if not all(isinstance(metrics.get(k), (int, float)) and math.isfinite(metrics[k]) and metrics[k] >= 0
               for k in ['area', 'delay', 'instances']):
        raise ValueError('non-finite physical seed metrics')
    record = dict(schema=1, metric_model=METRIC_MODEL, original=str(netlist),
                  normalized=str(normalized), netjson=str(netjson), measured=metrics,
                  library=str(library), library_sha256=sha(library), models_sha256=sha(library.parent/'cells.v'),
                  source_sha256=sha(netlist), normalized_sha256=sha(normalized), netjson_sha256=sha(netjson),
                  measure_binary=str(measure_binary), measure_binary_sha256=sha(measure_binary),
                  yosys=str(yosys), yosys_sha256=sha(yosys),
                  CEC=None, mapper_called=False, pure_conversion_and_metrics=True)
    (folder/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def supplied_seed(library, netlist, subject, folder, measure_binary, yosys, manifest=None):
    """Reuse a runner's premeasured manifest, or measure direct-frontend inputs.

    Caller-supplied EGM_SEED_AREA/DELAY never authorize a comparison.
    CEC is a separate runner/audit operation and is not inferred here.
    """
    library, netlist, subject, measure_binary = map(Path, [library, netlist, subject, measure_binary])
    yosys = Path(shutil.which(str(yosys)) or str(yosys)).resolve()
    if manifest:
        r = json.loads(Path(manifest).read_text())
        assert r['schema'] == 1 and r['metric_model'] == METRIC_MODEL
        assert Path(r['original']).resolve() == netlist.resolve()
        assert r['source_sha256'] == sha(netlist)
        assert r['library_sha256'] == sha(library) and r['models_sha256'] == sha(library.parent/'cells.v')
        assert r['measure_binary_sha256'] == sha(measure_binary) and r['yosys_sha256'] == sha(yosys)
        assert r['prepared_sha256'] == sha(subject)
        assert r['normalized_sha256'] == sha(r['normalized']) and r['netjson_sha256'] == sha(r['netjson'])
        assert r['measured'] == json.loads((Path(r['normalized']).parent/'measure.log').read_text())
        assert r['CEC'] == 1
    else:
        r = measure_cover(library, netlist, folder, measure_binary, yosys)
        r['prepared_sha256'] = sha(subject)
    return Path(r['normalized']).read_text(), r


def contribution(seed, delivered):
    def value(metrics):
        if not all(isinstance(metrics.get(k), (int, float)) and math.isfinite(metrics[k]) and metrics[k] >= 0
                   for k in ['area', 'delay']):
            raise ValueError('invalid contribution metrics')
        return metrics['delay'], metrics['area']
    a, b = value(delivered), value(seed)
    return dict(objective='delay-area', comparison='exact canonical physical metrics',
                seed_AD={k:seed[k] for k in ['area','delay']}, delivered_AD={k:delivered[k] for k in ['area','delay']},
                outcome='improved' if a < b else 'regressed' if a > b else 'equal',
                delay_reduction=seed['delay']-delivered['delay'], area_reduction=seed['area']-delivered['area'])
