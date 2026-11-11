#!/usr/bin/env python3
"""Exercise empty read-only pools, private miss reuse and unsafe-path rejection."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from audit_mapped_ppa import audit
from build_readonly_profile_library import build,snapshot


def main():
    engine=ROOT/'build/epmapping/epmapping_check'
    control=ROOT/'results_epmapping_joint_prebuffer_v1_20261111'
    env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','MAP_EGG_','EPM_'))}
    env.update(json.loads((control/'manifest.json').read_text())['goal'])
    env.update(ABC=str(control/'inputs/abc'),EPMAPPING_MAPPED_INPUT=str(control/'normalized/strong47/usb_phy.blif'),
               RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='prebuilt-profile-') as folder:
        root=Path(folder);shared=root/'library';shared.mkdir();shared.chmod(0o555)
        private=root/'private';before=snapshot(shared)
        env.update(MAP_EGG_PROFILE_READONLY_DIR=str(shared),MAP_EGG_PROFILE_CACHE_DIR=str(private))
        hashes=[];reports=[]
        for name in ['miss','private_reuse']:
            wd=root/name
            result=subprocess.run([str(engine),str(control/'inputs/asap7.genlib'),str(control/'inputs/benchmarks/usb_phy.aig'),str(wd)],env=env,text=True,capture_output=True,timeout=160)
            assert result.returncode==0,(result.returncode,result.stdout[-1000:],result.stderr)
            assert '[i] cec=1' in result.stdout
            stats=re.findall(r'egraph_profile (.*)',result.stdout)[-1]
            counters={k:float(v) if any(c in v for c in '.eE') else int(v) for k,v in re.findall(r'(\w+)=([0-9.eE+-]+)',stats)}
            cost=audit(wd/'epmapping_mapped.v',control/'inputs/asap7.genlib')
            assert abs(cost['area']-18.8)<.0001 and abs(cost['delay']-158.68)<.001
            assert snapshot(shared)==before
            assert counters['readonly_hits']==0 and counters['readonly_lookups']>0
            if name=='miss':assert counters['builds']>0 and counters['profile_writes']>0
            else:assert counters['builds']==0 and counters['private_disk_hits']>0
            hashes.append(hashlib.sha256((wd/'epmapping_mapped.v').read_bytes()).hexdigest())
            reports.append(dict(mode=name,cost={k:cost[k] for k in ['area','delay']},counters=counters))
        assert len(set(hashes))==1
        populated=root/'populated-library'
        build(populated,{'usb_phy':private},{})
        populated_hashes=snapshot(populated)
        trial=root/'prebuilt';trial_env=dict(env,MAP_EGG_PROFILE_READONLY_DIR=str(populated),
                                            MAP_EGG_PROFILE_CACHE_DIR=str(root/'fresh-delta'))
        result=subprocess.run([str(engine),str(control/'inputs/asap7.genlib'),str(control/'inputs/benchmarks/usb_phy.aig'),str(trial)],env=trial_env,text=True,capture_output=True,timeout=160)
        assert result.returncode==0 and '[i] cec=1' in result.stdout
        stats=re.findall(r'egraph_profile (.*)',result.stdout)[-1]
        counters={k:float(v) if any(c in v for c in '.eE') else int(v) for k,v in re.findall(r'(\w+)=([0-9.eE+-]+)',stats)}
        assert counters['readonly_hits']>0 and counters['builds']==counters['profile_writes']==0
        assert snapshot(populated)==populated_hashes
        assert hashlib.sha256((trial/'epmapping_mapped.v').read_bytes()).hexdigest()==hashes[0]
        reports.append(dict(mode='prebuilt',counters=counters))
        populated.chmod(0o755)
        for p in populated.iterdir():p.chmod(0o644)
        for overrides in [dict(MAP_EGG_PROFILE_CACHE_DIR=str(shared)),
                          dict(MAP_EGG_PROFILE_CACHE_DIR=str(shared/'nested')),
                          dict(MAP_EGG_PROFILE_READONLY_DIR=str(root/'absent')),
                          dict(MAP_EGG_PROFILE_TT_CACHE='1')]:
            result=subprocess.run([str(engine),'unused.genlib','unused.aig',str(root/'invalid')],env=dict(env,**overrides),capture_output=True,text=True,timeout=10)
            assert result.returncode==2 and '[e]' in result.stderr,(overrides,result)
        assert not(shared/'nested').exists()
        shared.chmod(0o755)
    print(json.dumps(dict(passed=True,readonly_unchanged=True,private_miss_reuse=True,unsafe_configuration_rejected=True,rows=reports),indent=2))


if __name__=='__main__':main()
