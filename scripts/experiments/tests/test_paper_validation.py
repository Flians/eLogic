#!/usr/bin/env python3
"""Exercise paper admission against raw CEC, metrics and care configuration."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from export_paper import rows

header='stage\tdesign\trecipe\trounds\tbase_a\tbase_d\tout_a\tout_d\tdA%\tdD%\trnd\tcec\tsecs\trc\n'
row='noCARE\tsasc\thigh_effort\t6\t2\t3\t1\t3\t-50\t0\t1\t1\t9\t0\n'
log='[i] params, dc=0, cache=8192\n[i] dc_cuts=0\n[i] epmapping area 2 -> 1 delay 3 -> 3\n[i] cec=1\n'
with tempfile.TemporaryDirectory(prefix='epm-paper-validation-') as tmp:
    root=Path(tmp);folder=root/'ablation/noCARE.sasc';folder.mkdir(parents=True)
    summary=root/'ablation/summary.tsv';raw=folder/'log'
    def write(r=row,t=log):summary.write_text(header+r);raw.write_text(t)
    def reject(r=row,t=log):
        write(r,t)
        try:rows(root,'ablation')
        except ValueError:return
        raise AssertionError('invalid result admitted')
    write();assert len(rows(root,'ablation'))==1
    reject(t=log.replace('dc=0','dc=1'))
    reject(t=log.replace('dc_cuts=0','dc_cuts=10'))
    reject(t=log.replace('[i] cec=1','[i] cec=0'))
    reject(t=log.replace('area 2 -> 1','area 2 -> 1.1'))
    reject(r=row.replace('\t2\t3\t1\t3\t','\t2\t3\t3\t3\t'),t=log.replace('area 2 -> 1','area 2 -> 3'))
    reject(r=row.replace('\t2\t3\t1\t3\t','\t2\t3\t1\t4\t'),t=log.replace('delay 3 -> 3','delay 3 -> 4'))
    reject(r=row+row)
    reject(r=row.replace('\t1\t9\t0','\t0\t9\t0'))
print('raw CEC, no-care configuration, metric consistency and uniqueness checks passed')
