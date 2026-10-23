#!/usr/bin/env python3
"""Summarize actual generated supports and physical rejection reasons, not aggregate root counts."""
import csv,json,sys
from collections import Counter
from pathlib import Path
p=Path(sys.argv[1]);contexts=[];candidates=[]
for r in csv.DictReader(p.open()):
 if r['root']=='root':continue
 (contexts if r['source']=='0' else candidates).append(r)
unique={(r['root'],tuple(sorted(r['leaves'].strip(';').split(';')))) for r in contexts}
nonlocal_unique={(r['root'],tuple(sorted(r['leaves'].strip(';').split(';')))) for r in contexts if int(r['side'])}
counts=lambda key,rs:dict(sorted(Counter(r[key] for r in rs).items()))
result=dict(contexts=len(contexts),unique_supports=len(unique),nonlocal_contexts=sum(int(r['side'])>0 for r in contexts),unique_nonlocal_supports=len(nonlocal_unique),widths=dict(Counter(len(r['leaves'].strip(';').split(';')) for r in contexts)),mffc_sizes=counts('mffc',contexts),removed_sizes=counts('removed',contexts),rejections=counts('rejection',candidates),returned_covers=sum(float(r['new_area'])>=0 for r in candidates),area_saving_covers=sum(0<=float(r['new_area'])<float(r['old_area'])-1e-4 for r in candidates),area_saving_but_delay_rejected=sum(r['rejection']=='2' and 0<=float(r['new_area'])<float(r['old_area'])-1e-4 for r in candidates),feasible_covers=sum(r['rejection']=='0' for r in candidates))
print(json.dumps(result,indent=2))
