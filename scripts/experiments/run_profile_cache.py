#!/usr/bin/env python3
"""Compare legacy extraction and cold/warm profile pools on unchanged v20 inputs."""
import argparse, csv, hashlib, json, os, re, shutil, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT/'results_v20_three_parts_chain/inputs'
HEADER = 'stage design recipe rounds base_a base_d out_a out_d dA% dD% rnd cec secs rc'.split()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--designs',nargs='+',default=['sasc','simple_spi','arbiter']);p.add_argument('--root',type=Path,default=ROOT/'results_v21_profile_cache');p.add_argument('--rounds',type=int,default=6);args=p.parse_args()
    out=args.root.resolve();out.mkdir(parents=True,exist_ok=True);inputs=out/'inputs';inputs.mkdir(exist_ok=True)
    manifest_path=out/'manifest.json'
    if not manifest_path.exists():
        shutil.copy2(ROOT/'build/epmapping/epmapping_check',inputs/'epmapping_check')
        shutil.copy2(Path(__file__),inputs/Path(__file__).name)
        sources=[*sorted((ROOT/'map_egg/src').rglob('*.rs')),ROOT/'map_egg/Cargo.toml',ROOT/'map_egg/build.rs',ROOT/'epmapping/main.cpp',ROOT/'epmapping/epmapping.hpp',ROOT/'epmapping/CMakeLists.txt',ROOT/'scripts/experiments/common.sh']
        for source in sources:
            dest=inputs/'sources'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        manifest_path.write_text(json.dumps(dict(designs=args.designs,rounds=args.rounds,post_timeout=1000,whole_window=False,toolchain=subprocess.check_output(['rustc','--version'],text=True).strip(),frozen_sha256={str(p.relative_to(FROZEN)):sha(p) for p in [FROZEN/'abc',FROZEN/'asap7.genlib',FROZEN/'database.super',*[FROZEN/'benchmarks'/f'{d}.aig' for d in args.designs]]},sha256={str(p.relative_to(out)):sha(p) for p in inputs.rglob('*') if p.is_file()}),indent=2)+'\n')
    manifest=json.loads(manifest_path.read_text());assert (manifest['designs'],manifest['rounds'])==(args.designs,args.rounds)
    for name,h in manifest['sha256'].items():assert sha(out/name)==h,name
    for name,h in manifest['frozen_sha256'].items():assert sha(FROZEN/name)==h,name
    assert sha(ROOT/'scripts/experiments/common.sh')==sha(FROZEN/'common.sh')
    rows=[];verified=[]
    for design in args.designs:
        for arm in ['legacy','cold','warm']:
            wd=out/'runs'/f'{arm}.{design}';rowfile=wd/'row.tsv';cache=out/'cache'/design
            if not rowfile.exists():
                if arm=='cold': assert not cache.exists(),'cold run needs a new cache directory'
                env={k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_','EPM_','MAP_EGG_'))}
                env.update(ABC=str(FROZEN/'abc'),EPM_BIN=str(inputs/'epmapping_check'),EPM_ROOT=str(out),EPM_TIMEOUT='3600',EPM_RETRIES='1',EPM_CAFFEINATE='0',RAYON_NUM_THREADS='1')
                knobs=dict(EPMAPPING_OBJECTIVE='area_delay_bound',EPMAPPING_AREA_UNDER_DELAY='1',EPMAPPING_WINDOW_SYN='0',EPMAPPING_POST_TIMEOUT='1000',EPMAPPING_WRITE_BASELINE='1',EPMAPPING_SAT_DC='0',EPMAPPING_EGG_REQ='1',EPMAPPING_DSE='0',EPMAPPING_CHAIN_FACTOR='1',EPMAPPING_CHAIN_BLOCK='2',EPMAPPING_CHAIN_SHARE='1',EPMAPPING_FEASIBLE_COVERS='1',EPMAPPING_SUPER=str(FROZEN/'database.super'),EPM_GENLIB=str(FROZEN/'asap7.genlib'),MAP_EGG_PROFILE='0' if arm=='legacy' else '1',MAP_EGG_PROFILE_CACHE_DIR=str(cache))
                cmd=['bash','-c','source "$1"; BENCH="$2"; shift 2; run_engine "$@"','_',str(ROOT/'scripts/experiments/common.sh'),str(FROZEN/'benchmarks'),str(wd),design,str(args.rounds),'high_effort',*[k+'='+v for k,v in knobs.items()]]
                q=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
                if q.returncode:raise RuntimeError(q.stderr)
                row=q.stdout.strip().splitlines()[-1]+'\n';rowfile.write_text(row);print(row.strip(),flush=True)
            row=rowfile.read_text();rows.append(row);r=dict(zip(HEADER,row.strip().split('\t')));log=(wd/'log').read_text()
            assert r['rc']=='0' and r['cec']=='1',r
            assert 'window_synth windows=0 applied=0' in log and 'post_mapping_timeout=1000' in log
            assert float(r['out_d'])<=float(r['base_d'])+.001 and float(r['out_a'])<=float(r['base_a'])+.0001,r
            initial=ROOT/'results_v20_three_parts_chain/tab_main'/f'ae.{design}'/'epmapping_baseline.v'
            assert (wd/'epmapping_baseline.v').read_bytes()==initial.read_bytes(),f'initial mapping changed: {arm}.{design}'
            timing=re.search(r'epmapping timing .*optimizer=([\d.]+)',log);profile=re.search(r'egraph_profile (.*)',log)
            verified.append(dict(**r,optimizer_seconds=float(timing[1]),profile=profile[1],baseline_sha256=sha(initial),mapped_sha256=sha(wd/'epmapping_mapped.v'),log_sha256=sha(wd/'log')))
        cold=out/'runs'/f'cold.{design}'/'epmapping_mapped.v';warm=out/'runs'/f'warm.{design}'/'epmapping_mapped.v'
        assert cold.read_bytes()==warm.read_bytes(),f'cold/warm trajectory differs: {design}'
        (out/'summary.tsv').write_text('\t'.join(HEADER)+'\n'+''.join(rows))
        (out/'verified.json').write_text(json.dumps(verified,indent=2)+'\n')
if __name__=='__main__':main()
