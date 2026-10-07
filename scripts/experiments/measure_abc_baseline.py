#!/usr/bin/env python3
"""Measure ABC's parsed start; identical input files can normalize CO buffers differently."""
import argparse,csv,os,re,subprocess
from pathlib import Path
import report
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
root=a.root.resolve();lib=root/'tab_mfs3/asap7_6in.genlib'
base={r['design']:r for r in report.read_summary(str(root/'tab_mfs3/summary.tsv')) if r['stage']=='base'}
data=[]
for d in report.PAPER16:
    v=root/'tab_mfs3'/f'base.{d}'/'epmapping_mapped.v'
    result=subprocess.run([os.environ.get('ABC','abc'),'-c',f"read_genlib '{lib}'; read -m '{v}'; print_stats"],capture_output=True,text=True,check=True)
    (root/'tab_mfs3'/f'abc_baseline.{d}.log').write_text(result.stdout+result.stderr)
    values=[]
    for key in ('area','delay'):
        found=re.findall(r'\b'+key+r'\s*=\s*([0-9.]+)',result.stdout)
        if not found:raise ValueError(f'ABC has no baseline {key}: {d}')
        values.append(found[-1])
    data.append([d,*values,base[d]['base_a'],base[d]['base_d']])
with (root/'tab_mfs3/abc_baseline_metrics.tsv').open('w') as f:
    w=csv.writer(f,delimiter='\t',lineterminator='\n')
    w.writerow(['design','abc_area','abc_delay','emap_area','emap_delay']);w.writerows(data)
print('Recorded native ABC baselines for all 16 serialized inputs')
