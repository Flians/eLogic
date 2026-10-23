#!/usr/bin/env python3
"""Audit all EPFL region/control observations; retain missing and failed cases."""
import argparse
import csv
import json
from pathlib import Path
from run_epfl_baselines import EPFL, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    c=json.loads(a.campaign.read_text());out=a.output.resolve();out.mkdir(exist_ok=False)
    subjects={r['bench']:r for r in json.loads(Path(c['subjects']).read_text())['results']};seen={};hashes={str(a.campaign.resolve()):sha(a.campaign)}
    for j in c['jobs']:
        for key in ['report','audit']:
            if key in j:
                p=Path(j[key]);assert sha(p)==j[key+'_sha256'];hashes[str(p)]=sha(p)
        key=(j['k'],j['objective'],j['bench'],j['profile']);assert key not in seen
        if 'report' not in j:seen[key]={'status':'failed','error':'no report'};continue
        d=json.loads(Path(j['report']).read_text());assert len(d['results'])==1;r=d['results'][0];m=subjects[j['bench']]
        assert r['input_sha256']==m['input_sha256'];assert r['preprocessing']['sha256']==m['preprocessing']['sha256']
        assert r['settings']['EGM_REGION_CLASSES']==str(c['profiles'][j['profile']]['region_classes'])
        assert r['settings']['EGM_PHASE_RECOVERY']==str(c['profiles'][j['profile']]['phase_recovery'])
        r['library_sha256']=d['library_sha256'];r['observation_report']=j['report']
        if r['status']=='ok':
            n=r['native'];assert n['enodes']<=20000 and n['instances']<=100000 and n['max_cuts_per_class']<=16 and n['heuristic_secs']<=2 and r['measured']['cec']==1
        seen[key]=r
    paired=[];counts={'delay_better':0,'delay_equal':0,'delay_worse':0,'paired_cec':0,'new_cec_control_failed':0}
    full=[]
    for k in [3,4]:
      for objective in c.get('objectives',['area-delay','delay-area']):
       for bench in EPFL:
        profiles={p:seen.get((k,objective,bench,p),{'status':'missing','error':'not attempted'}) for p in c['profiles']}
        full.append({'k':k,'objective':objective,'bench':bench,'profiles':profiles})
        row={'k':k,'objective':objective,'bench':bench}
        for p,r in profiles.items():
            row[p+'_status']=r['status'];row[p+'_area']=r.get('measured',{}).get('area');row[p+'_delay']=r.get('measured',{}).get('delay');row[p+'_error']=r.get('error','')
        new,old=profiles['region_phase'],profiles['control']
        if new['status']==old['status']=='ok':
            assert new['library_sha256']==old['library_sha256'];counts['paired_cec']+=1
            d=new['measured']['delay']-old['measured']['delay'];tol=1e-6*max(1,old['measured']['delay']);counts['delay_better' if d < -tol else 'delay_worse' if d > tol else 'delay_equal']+=1
            row['delay_delta']=d;row['area_delta']=new['measured']['area']-old['measured']['area'];row['phase_accepted']=new['native']['phase_accepted']
        elif new['status']=='ok' and old['status']=='failed':counts['new_cec_control_failed']+=1
        paired.append(row)
    (out/'full_scope.json').write_text(json.dumps({'records':full,'scope':'all20 both libraries/objectives; no selection across versions/profiles'},indent=2))
    fields=['k','objective','bench','region_phase_status','region_phase_area','region_phase_delay','region_phase_error','control_status','control_area','control_delay','control_error','delay_delta','area_delta','phase_accepted']
    with (out/'paired.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(paired)
    summary={'jobs':len(c['jobs']),'expected_jobs':c['expected_jobs'],'counts':counts,'success':sum(r['status']=='ok' for r in seen.values()),'failed':sum(r['status']=='failed' for r in seen.values()),'missing':c['expected_jobs']-len(seen),'hashes':hashes,'claim':'paired same-binary experiment, not baseline dominance; missing and failures remain in full scope'}
    (out/'validation.json').write_text(json.dumps(summary,indent=2));print(summary['counts'],summary['success'],summary['failed'],summary['missing'])

if __name__=='__main__':main()
