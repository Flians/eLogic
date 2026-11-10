#!/usr/bin/env python3
"""Verify that current full-library recovery retains the prior mapped core DAGs."""
import argparse,json
from pathlib import Path
from run_three_parts import DESIGNS
from run_epmapping_supplement import sha,save
from exact_core_graph import exact_core_fingerprint as core_fingerprint
PROJECT=Path(__file__).resolve().parents[2]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);a=ap.parse_args();root=a.root.resolve();records=[]
    old=PROJECT/'results_v28_three_parts_query_budget';lib=root/'inputs/asap7.genlib'
    for design in DESIGNS:
        for mode in ['cold','warm']:
            r=json.loads((root/f'main/ours/{mode}.{design}/result.json').read_text());assert r['status']=='complete'
            before=old/mode/'tab_main'/f'ae.{design}/epmapping_mapped.v';after=root/r['selected_netlist']
            previous=core_fingerprint(before,lib);current=core_fingerprint(after,lib)
            records.append(dict(design=design,mode=mode,previous_netlist_sha256=sha(before),current_netlist_sha256=sha(after),previous=previous,current=current,identical=previous==current))
    save(root/'primary_core_replay.json',dict(complete=True,script_sha256=sha(Path(__file__)),pairs=32,identical=sum(r['identical'] for r in records),records=records))
    print(f"Verified {sum(r['identical'] for r in records)}/32 identical buffer-free mapped core DAGs")
if __name__=='__main__':main()
