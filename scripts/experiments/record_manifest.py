#!/usr/bin/env python3
"""Record the engine, source, inputs and configured concurrency for a paper reproduction."""
import argparse, datetime, hashlib, json, os, platform, subprocess
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
project=Path(__file__).resolve().parents[2]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1048576),b''):h.update(data)
    return h.hexdigest()
paths=[project/'build/epmapping/epmapping_check']
paths.extend(project/p for p in ['epmapping/epmapping.hpp','epmapping/main.cpp','scripts/experiments/common.sh','scripts/experiments/run_all.sh','scripts/experiments/abc_mfs3_one.sh','scripts/experiments/export_paper.py','tools/mockturtle/include/mockturtle/utils/tech_library.hpp','tools/mockturtle/include/mockturtle/algorithms/cut_enumeration.hpp','tools/mockturtle/experiments/cell_libraries/asap7.genlib','epmapping/db/asap7_4in_3lvl_a050.super'])
paths.extend((project/'tools/mockturtle/experiments/benchmarks').glob('*.aig'))
config={k:v for k,v in sorted(os.environ.items()) if k.startswith(('EPM_','EPMAPPING_','MAP_EGG_','RAYON_')) or k=='ABC'}
manifest={'recorded_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'timezone':'Asia/Hong_Kong','engine_sha256':sha(paths[0]),'host':platform.platform(),'config':config,'sha256':{str(p.relative_to(project)):sha(p) for p in paths}}
a.root.mkdir(parents=True,exist_ok=True);(a.root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
