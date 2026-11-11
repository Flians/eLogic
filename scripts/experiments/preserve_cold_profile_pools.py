#!/usr/bin/env python3
"""Preserve cold-only sources after paired private-warm controls finish.

The running collector uses one private path for cold then warm. Archive the
completed warm pool byte-for-byte and restore the cold subset at the original
path before library publication. No running process uses a transformed pool.
Cold result mtime is the end-of-cold boundary; candidate records are immutable
and atomically published. The saved cold count must exactly match that boundary.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

from build_readonly_profile_library import sha, snapshot
from run_three_parts import DESIGNS


def preserve(root,stage,arm,design):
    key='/'.join([stage,arm,design])
    cold_path=root/stage/arm/('cold.'+design)/'result.json'
    warm_path=root/stage/arm/('warm.'+design)/'result.json'
    cold=json.loads(cold_path.read_text());warm=json.loads(warm_path.read_text())
    pool=root/'cache_sources'/stage/arm/design
    cutoff=cold_path.stat().st_mtime_ns
    destination=root/'cold_source_snapshots'/stage/arm/design
    archive=root/'private_warm_pool_evidence'/stage/arm/design
    original=archive if pool.is_symlink() else pool
    all_files=list(original.glob('*.json'))
    cold_files=[p for p in all_files if p.stat().st_mtime_ns<=cutoff]
    assert len(cold_files)==cold['cache_files'],('cold boundary/count mismatch',key,len(cold_files),cold['cache_files'])
    assert len(all_files)==warm['cache_files'],('warm count mismatch',key)
    if not pool.is_symlink():
        destination.mkdir(parents=True);archive.parent.mkdir(parents=True,exist_ok=True)
        for source in cold_files:shutil.copy2(source,destination/source.name)
    cold_hashes=snapshot(destination);warm_hashes=snapshot(original)
    assert all(warm_hashes[name]==digest for name,digest in cold_hashes.items())
    if not pool.is_symlink():
        pool.rename(archive);pool.symlink_to(destination,target_is_directory=True)
    receipt=dict(cold_result_sha256=sha(cold_path),warm_result_sha256=sha(warm_path),
        cold_boundary_mtime_ns=cutoff,cold_record_count=len(cold_files),warm_record_count=len(all_files),
        cold_pool=str(destination.relative_to(root)),warm_pool=str(archive.relative_to(root)),
        cold_record_sha256=cold_hashes,warm_record_sha256=warm_hashes,
        warm_extra_records=len(all_files)-len(cold_files))
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();saved=root/'cold_pool_receipts.json'
    receipts=json.loads(saved.read_text())['records'] if saved.exists() else {}
    expected=[(stage,'ours',d) for stage in ['main','mfs3','strong_common','main_r1'] for d in DESIGNS]
    expected += [('ablation','full',d) for d in DESIGNS]
    while len(receipts)<len(expected):
        for stage,arm,design in expected:
            key='/'.join([stage,arm,design])
            if key in receipts:continue
            cold_path=root/stage/arm/('cold.'+design)/'result.json'
            warm_path=root/stage/arm/('warm.'+design)/'result.json'
            if not warm_path.exists():continue
            try:receipt=preserve(root,stage,arm,design)
            except json.JSONDecodeError:continue  # result publication is still in flight
            receipts[key]=receipt
            target=root/'cold_pool_receipts.json';tmp=target.with_suffix('.new')
            tmp.write_text(json.dumps(dict(complete=len(receipts)==80,expected=80,records=receipts),indent=2)+'\n');tmp.replace(target)
            print('PRESERVED',key,'cold',receipt['cold_record_count'],'warm extras',receipt['warm_extra_records'],flush=True)
        if len(receipts)<len(expected):
            assert not (root/'library_frozen_sha256.json').exists(),'Library published before all cold sources preserved'
            time.sleep(.2)
    print('COMPLETE: 80 cold-only snapshots; private warm evidence preserved',flush=True)


if __name__=='__main__':main()
