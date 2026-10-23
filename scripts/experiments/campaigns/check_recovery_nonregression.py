#!/usr/bin/env python3
"""Reject actual CEC-verified whole-cover regressions; no per-design winners."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--pairs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=json.loads(a.pairs.read_text());fail=[];checked=0;gains=[];sqrt_pairs=0
for r in rows:
 x=r['control']['row'];y=r['candidate']['row']
 if r['bench'] not in {'div','hyp','log2','multiplier','mem_ctrl'} and (x['status']!='ok' or y['status']!='ok'):fail.append({'k':r['k'],'bench':r['bench'],'reason':'required previously legal cover is not verified'});continue
 if x['status']!=y['status']:fail.append({'k':r['k'],'bench':r['bench'],'reason':'status mismatch'});continue
 if x['status']!='ok':continue
 checked+=1;sqrt_pairs+=int(r['bench']=='sqrt');assert x['measured']['cec']==y['measured']['cec']==1
 assert x['input_sha256']==y['input_sha256'] and x['preprocessing']['sha256']==y['preprocessing']['sha256']
 for label,row in [('control',x),('candidate',y)]:
  n=row.get('native',{})
  if row.get('settings',{}).get('EGM_STRICT_BUDGETS')=='1' and (n.get('enodes',float('inf'))>20000 or n.get('max_cuts_per_class',float('inf'))>16 or n.get('instances',float('inf'))>100000 or n.get('heuristic_secs',float('inf'))>2):fail.append({'k':r['k'],'bench':r['bench'],'reason':label+' actual global budget violation'})
 for metric in ('delay','area'):
  u=x['measured'][metric];v=y['measured'][metric]
  if not all(isinstance(w,(int,float)) and math.isfinite(w) and w>=0 for w in [u,v]):fail.append({'k':r['k'],'bench':r['bench'],'reason':'non-finite or negative metric'});continue
  t=1e-6*max(1,abs(u),abs(v))
  if v>u+t:fail.append({'k':r['k'],'bench':r['bench'],'metric':metric,'control':u,'candidate':v})
  if r['bench']=='sqrt' and metric=='delay' and v<u-t:gains.append(r['k'])
result={'eligible_pairs':checked,'regressions':fail,'sqrt_delay_gains':gains,'pass':checked>0 and not fail and set(gains)=={3,4} and len(gains)==sqrt_pairs};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(0 if result['pass'] else 1)
