"""Fail on same-delay area regressions in explicitly paired, hash-verified observations."""
import argparse,json
from pathlib import Path
from report_online600 import load
p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();cs,c=load(a.control);xs,x=load(a.candidate);bad=[];pairs=[]
assert cs['fingerprint']==xs['fingerprint'],'paired control must use same frozen binary/dependencies'
for b in sorted(set(c)&set(x)):
 v,w=c[b].get('measured'),x[b].get('measured')
 if not v or not w or v['cec']!=1 or w['cec']!=1:continue
 row=dict(bench=b,control_A=v['area'],candidate_A=w['area'],control_D=v['delay'],candidate_D=w['delay']);pairs.append(row)
 if abs(w['delay']-v['delay'])<1e-7 and w['area']>v['area']+1e-6:bad.append(row)
print(json.dumps(dict(pairs=pairs,regressions=bad,scope='only measured paired designs; no full-suite claim'),indent=2));raise SystemExit(1 if bad or not pairs else 0)
