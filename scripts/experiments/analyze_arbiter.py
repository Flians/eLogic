#!/usr/bin/env python3
"""Controlled arbiter diagnosis: local search and preprocessing are separate cohorts."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

PROJECT=Path(__file__).resolve().parents[2]
PROFILES={
    'control': {},
    'hybrid': {'EPMAPPING_SINGLE_MATCHER':'hybrid'},
    'care12': {'EPMAPPING_DC_MAX_PIS':'12'},
    'care_percut12': {'EPMAPPING_DC_WINDOW':'per_cut','EPMAPPING_DC_MAX_PIS':'12'},
    'wide8': {'EPMAPPING_CUT_SIZE':'8','EPMAPPING_CUT_LIMIT':'24'},
    'exact4': {'EPMAPPING_SYN_MODE':'7','EPMAPPING_SYN_MIN_LEAVES':'2','EPMAPPING_SYN_EXACT_LEAVES':'4','EPMAPPING_SYN_EXACT_CONFLICT':'1000'},
    'egg_deeper': {'MAP_EGG_EGG_ITER_LIMIT':'5','MAP_EGG_EGG_NODE_LIMIT':'2000','MAP_EGG_FULL_BUNDLE':'1'},
    'pre_syn4': {'EPMAPPING_ABC_ROUND':'fraig; &get; &syn4; &put; compress2rs; resyn2rs'},
    'pre_sopb': {'EPMAPPING_ABC_ROUND':'fraig; &get; &sopb; &put; compress2rs; resyn2rs'},
}
HEADER='stage design recipe rounds base_a base_d out_a out_d dA% dD% rnd cec secs rc'.split()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    root=PROJECT/'results_v11_arbiter';root.mkdir(exist_ok=True);f=root/'inputs';f.mkdir(exist_ok=True)
    manifest=root/'manifest.json'
    if not manifest.exists():
        source=PROJECT/'results_v10_three_parts_1000/inputs'
        for name in ['epmapping_check','abc','asap7.genlib','database.super','main.cpp','epmapping.hpp','objective.hpp','common.sh']:
            shutil.copy2(source/name,f/name)
        shutil.copy2(source/'benchmarks/arbiter.aig',f/'arbiter.aig')
        shutil.copy2(Path(__file__),f/'analyze_arbiter.py')
        manifest.write_text(json.dumps(dict(profiles=PROFILES,post_mapping_timeout=1000,whole_window_synthesis=False,
            reference_area=557.72,reference_delay=999.87,frozen_sha256={p.name:sha(p) for p in f.iterdir()}),indent=2)+'\n')
    m=json.loads(manifest.read_text());assert m['profiles']==PROFILES
    for n,h in m['frozen_sha256'].items():assert sha(f/n)==h,n
    assert sha(PROJECT/'scripts/experiments/common.sh')==sha(f/'common.sh')
    common=dict(EPMAPPING_OBJECTIVE='area_delay_bound',EPMAPPING_AREA_UNDER_DELAY='1',EPMAPPING_WINDOW_SYN='0',
        EPMAPPING_WRITE_BASELINE='1',EPMAPPING_POST_TIMEOUT='1000',EPMAPPING_SAT_DC='0',EPMAPPING_DSE='0',
        EPMAPPING_EGG_REQ='1',EPMAPPING_SUPER=str(f/'database.super'),EPM_GENLIB=str(f/'asap7.genlib'))
    def run(name):
        wd=root/'runs'/f'{name}.arbiter';result=wd/'row.tsv'
        if result.exists():return result.read_text()
        env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','EPM_','MAP_EGG_'))}
        env.update(ABC=str(f/'abc'),EPM_BIN=str(f/'epmapping_check'),EPM_ROOT=str(root),EPM_TIMEOUT='3600',
                   EPM_RETRIES='1',EPM_CAFFEINATE='0',RAYON_NUM_THREADS='1')
        flags=[f'{k}={v}' for k,v in dict(common,**PROFILES[name]).items()]
        command=['bash','-c','source "$1"; BENCH="$2"; shift 2; run_engine "$@"','_',
                 str(PROJECT/'scripts/experiments/common.sh'),str(f),str(wd),'arbiter','6','high_effort',*flags]
        proc=subprocess.run(command,env=env,capture_output=True,text=True,cwd=PROJECT)
        if proc.returncode:raise RuntimeError(proc.stderr)
        row=proc.stdout.strip().splitlines()[-1]+'\n';assert len(row.strip().split('\t'))==14
        result.write_text(row);print(name+': '+row.strip(),flush=True);return row
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(run,PROFILES))
    (root/'summary.tsv').write_text('\t'.join(HEADER)+'\n'+''.join(rows))
    records=list(csv.DictReader((root/'summary.tsv').open(),delimiter='\t'))
    control=root/'runs/control.arbiter/epmapping_baseline.v'
    for r in records:
        wd=root/'runs'/f'{r["stage"]}.arbiter';text=(wd/'log').read_text()
        assert 'area_under_delay=1,' in text and 'path_objective=0,' in text
        assert 'window_synth windows=0 applied=0' in text or r['rc']=='142'
        if not r['stage'].startswith('pre_'):
            assert (wd/'epmapping_baseline.v').read_bytes()==control.read_bytes(),r['stage']
        r['cohort']='preprocessing' if r['stage'].startswith('pre_') else 'same_mapped_input'
        if r['rc']=='0':
            assert r['cec']=='1' and '[i] cec=1' in text
            assert float(r['out_a'])<=float(r['base_a'])+.001 and float(r['out_d'])<=float(r['base_d'])+.001
            r['reference_feasible']=float(r['out_d'])<=m['reference_delay']+.001 and float(r['out_a'])<=m['reference_area']+.001
            r['area_vs_reference']=100*(float(r['out_a'])/m['reference_area']-1)
            r['delay_vs_reference']=100*(float(r['out_d'])/m['reference_delay']-1)
        else:
            assert r['rc']=='142';r['status']='timeout; no QoR or equivalence claimed'
    (root/'verified.json').write_text(json.dumps(dict(records=records,raw_sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name in ['log','row.tsv','experiment_env.txt','epmapping_baseline.v','epmapping_mapped.v','manifest.json']}),indent=2)+'\n')
    print(json.dumps(records,indent=2),flush=True)

if __name__=='__main__':main()
