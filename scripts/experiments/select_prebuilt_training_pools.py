#!/usr/bin/env python3
"""Exclude duplicate full-library controls from candidate-bank training.

Full-library ablation queries use only the full arm's own sixteen cold sources,
just like every variant. The full bank also contains the two 40-entry comparison
flows, isolated by the technology namespace. Main/main_r1 remain cache controls.
All original pools are retained; only inactive collection aliases are redirected.
"""
import argparse,json,time
from pathlib import Path
from run_three_parts import DESIGNS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    root=a.root.resolve();done={};expected=[(s,'ours',d) for s in ['main','main_r1'] for d in DESIGNS]
    while len(done)<32:
        receipt=json.loads((root/'cold_pool_receipts.json').read_text())
        for stage,arm,design in expected:
            key='/'.join([stage,arm,design])
            if key in done or key not in receipt['records']:continue
            item=receipt['records'][key];pool=root/'cache_sources'/stage/arm/design
            assert pool.is_symlink() and pool.resolve()==(root/item['cold_pool']).resolve(),key
            empty=root/'nontraining_source_placeholders'/stage/arm/design;empty.mkdir(parents=True)
            pool.unlink();pool.symlink_to(empty,target_is_directory=True)
            done[key]=dict(cold_pool=item['cold_pool'],cold_record_count=item['cold_record_count'],
                bank_source=str(empty.relative_to(root)),reason='duplicate full-library cache-control flow; full47 training uses only ablation/full')
        target=root/'training_pool_selection.json';tmp=target.with_suffix('.new')
        tmp.write_text(json.dumps(dict(complete=len(done)==32,excluded=done,
            full_bank_training='mfs3/ours + strong_common/ours + ablation/full: 48 cold runs; full47 ablation namespace uses only own 16 cold sources',
            other_banks_training='own 16 ablation cold sources per configuration',
            all_cold_controls_retained=True),indent=2)+'\n');tmp.replace(target)
        if len(done)<32:
            assert not(root/'library_frozen_sha256.json').exists()
            time.sleep(.2)
    print('COMPLETE: 32 duplicate controls retained but excluded from bank training',flush=True)


if __name__=='__main__':main()
