#!/usr/bin/env python3
"""Freeze and rerun the three paper experiments under a common delay bound."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

PROJECT = Path(__file__).resolve().parents[2]
DESIGNS = 'aes_core arbiter div ethernet iwls05_i2c mem_ctrl RISC sasc simple_spi spi sqrt systemcaes systemcdes tv80 usb_funct usb_phy'.split()
ABLATION = DESIGNS.copy()
HEADER = 'stage design recipe rounds base_a base_d out_a out_d dA% dD% rnd cec secs rc'.split()
GOAL = dict(EPMAPPING_OBJECTIVE='area_delay_bound', EPMAPPING_AREA_UNDER_DELAY='1',
            EPMAPPING_WINDOW_SYN='0', EPMAPPING_WRITE_BASELINE='1',
            EPMAPPING_SAT_DC='0', EPMAPPING_DSE='0', EPMAPPING_EGG_REQ='1',
            EPMAPPING_POST_TIMEOUT='1000', EPMAPPING_CHAIN_FACTOR='1',
            EPMAPPING_CHAIN_BLOCK='2', EPMAPPING_CHAIN_SHARE='1',
            EPMAPPING_FEASIBLE_COVERS='1', MAP_EGG_PROFILE_NARROW='1',
            EPMAPPING_PHASE_REUSE='1', EPMAPPING_COFAC='6',
            EPMAPPING_COFAC_MIN_LEAVES='3', EPMAPPING_PHASE_COST='1',
            EPMAPPING_PHASE_FANOUT='1', MAP_EGG_PROFILE_PHASE_KEEP='0',
            EPMAPPING_OPERATOR_POLICY='3', EPMAPPING_SYN_CACHE='262144',
            EPMAPPING_CHECKPOINT='1', EPMAPPING_BUILD_REUSE='2',
            EPMAPPING_SHARED_MAPPED_LIBRARY='1', EPMAPPING_LIVE_QUERY_CACHE='9',
            EPMAPPING_COFAC_PIVOTS='2')
ARMS = {'full': {}, 'noSYN': {'EPMAPPING_SYN':'0','EPMAPPING_COFAC':'0'}, 'noEGG': {'EPMAPPING_EGG':'0'},
        'noDB': {'EPMAPPING_SUPER':'off'}, 'noCARE': {'EPMAPPING_DC':'0', 'EPMAPPING_SAT_DC':'0'},
        'noCHAIN': {'EPMAPPING_CHAIN_FACTOR':'0'},
        'noSHARE': {'EPMAPPING_CHAIN_SHARE':'0'},
        'noNARROW': {'MAP_EGG_PROFILE_NARROW':'0'},
        'noPHASE': {'EPMAPPING_PHASE_REUSE':'0'},
        'noSHANNON': {'EPMAPPING_COFAC':'4'},
        'noDAVIO': {'EPMAPPING_COFAC':'2'},
        'noREUSE': {'EPMAPPING_PHASE_FANOUT':'0'},
        'noPRICE': {'EPMAPPING_PHASE_COST':'0'},
        'cache64K': {'EPMAPPING_SYN_CACHE':'65536'},
        'noAIGREUSE': {'EPMAPPING_BUILD_REUSE':'0'},
        'noLIBSHARE': {'EPMAPPING_SHARED_MAPPED_LIBRARY':'0'},
        'noQUERY': {'EPMAPPING_LIVE_QUERY_CACHE':'8'},
        'noTRUTH': {'EPMAPPING_LIVE_QUERY_CACHE':'1'},
        'allPIVOTS': {'EPMAPPING_COFAC_PIVOTS':'0'}}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=3600)
    ap.add_argument('--abc-timeout', type=int, default=1000)
    ap.add_argument('--timeout-selection', choices=['initial','checkpoint'], default='initial',
                    help='Explicit output policy; historical campaigns retain initial fallback')
    ap.add_argument('--abc-source', type=Path, help='Reuse the freshly rerun ABC batch after proving identical inputs')
    ap.add_argument('--original-root', type=Path, default=PROJECT/'results_v10_three_parts_1000')
    ap.add_argument('--parts', default='mfs3,main,ablation')
    ap.add_argument('--profile-cache-mode', choices=['cold','warm'])
    ap.add_argument('--cache-root', type=Path, help='Dedicated shared cache root for paired cold/warm cohorts')
    args = ap.parse_args()
    assert bool(args.profile_cache_mode)==bool(args.cache_root), 'profile mode requires a dedicated cache root'
    cache_root=args.cache_root.resolve() if args.cache_root else None
    goal=dict(GOAL)
    if args.profile_cache_mode: goal['MAP_EGG_PROFILE']='1'
    original = args.original_root.resolve()
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    frozen = root/'inputs'; manifest_path = root/'manifest.json'
    if not manifest_path.exists():
        frozen.mkdir(exist_ok=True)
        sources = ['build/epmapping/epmapping_check', 'epmapping/main.cpp', 'epmapping/epmapping.hpp',
                   'epmapping/live_query_cache.hpp', 'epmapping/local_truth_table.hpp',
                   'epmapping/objective.hpp', 'epmapping/exact_aig_key.hpp', 'epmapping/bounded_cache.hpp', 'epmapping/operator_policy.hpp', 'scripts/experiments/common.sh',
                   'scripts/experiments/run_three_parts.py', 'scripts/experiments/recover_checkpoint.py',
                   'scripts/compare_with_aspdac25.sh']
        for rel in sources:
            shutil.copy2(PROJECT/rel, frozen/Path(rel).name)
        shutil.copy2(PROJECT/'tools/mockturtle/experiments/cell_libraries/asap7.genlib', frozen/'asap7.genlib')
        # Reuse the archived restricted library verbatim, including its provenance header.
        shutil.copy2(original/'inputs/asap7_6in.genlib', frozen/'asap7_6in.genlib')
        shutil.copy2(PROJECT/'epmapping/db/asap7_4in_3lvl_a050.super', frozen/'database.super')
        (frozen/'benchmarks').mkdir(exist_ok=True)
        for d in DESIGNS:
            shutil.copy2(PROJECT/f'tools/mockturtle/experiments/benchmarks/{d}.aig', frozen/'benchmarks'/f'{d}.aig')
        if args.profile_cache_mode:
            # Freeze the Rust implementation and every archived native dependency as well.
            sources=set((PROJECT/'map_egg/src').rglob('*.rs'))
            sources.update((PROJECT/'tools/patches/mockturtle').glob('*.patch'))
            sources.add(PROJECT/'scripts/apply_mockturtle_patches.sh')
            sources.update(p for p in (PROJECT/'tools/patches/mockturtle/overlay').rglob('*') if p.is_file())
            sources.update(PROJECT/p for p in ['tools/apply_mockturtle_overlay.py', 'tools/CMakeLists.txt'])
            sources.update(PROJECT/rel for rel in ['epmapping/exact_aig_key.hpp', 'epmapping/live_query_cache.hpp', 'epmapping/local_truth_table.hpp'])
            sources.update([PROJECT/'map_egg/Cargo.toml',PROJECT/'map_egg/build.rs',PROJECT/'map_egg/CMakeLists.txt',PROJECT/'epmapping/CMakeLists.txt'])
            archived=original/'inputs/compiled_dependencies'
            for item in archived.rglob('*'):
                if item.is_file() and (PROJECT/item.relative_to(archived)).is_file(): sources.add(PROJECT/item.relative_to(archived))
            for source in sorted(sources):
                dest=frozen/'build_sources'/source.relative_to(PROJECT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        abc = (original/'inputs/abc').resolve() if args.profile_cache_mode else Path(shutil.which('abc')).resolve()
        shutil.copy2(abc, frozen/'abc')
        for rel in ['abc','asap7.genlib','asap7_6in.genlib','database.super'] + [f'benchmarks/{d}.aig' for d in DESIGNS]:
            assert (frozen/rel).read_bytes()==(original/'inputs'/rel).read_bytes(), f'original input changed: {rel}'
        manifest = dict(original_root=str(original), recorded_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                        objective='minimize area subject to initial worst-output delay', whole_window_synthesis=False,
                        goal=goal, profile_cache_mode=args.profile_cache_mode, cache_root=str(cache_root) if cache_root else None, arms=ARMS, designs=DESIGNS, ablation_designs=ABLATION,
                        jobs=args.jobs, timeout=args.timeout, abc_timeout=args.abc_timeout,
                        post_mapping_timeout=1000, timeout_selection=args.timeout_selection,
                        abc_source=str(args.abc_source.resolve()) if args.abc_source else None,
                        rounds=dict(mfs3=1, main=[1,6], ablation=6), archived_engine='inputs/epmapping_check',
                        engine_sha256=sha(frozen/'epmapping_check'),
                        frozen_sha256={str(p.relative_to(root)):sha(p) for p in frozen.rglob('*') if p.is_file()})
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    manifest = json.loads(manifest_path.read_text())
    for name,h in manifest['frozen_sha256'].items():
        assert sha(root/name)==h, name
    assert (args.jobs,args.timeout,args.abc_timeout)==(manifest['jobs'],manifest['timeout'],manifest['abc_timeout']), 'different plan: use a new root'
    assert args.abc_timeout==1000, 'the declared common post-mapping budget is 1000 seconds'
    assert manifest.get('timeout_selection','initial')==args.timeout_selection, 'different timeout selection policy: use a new root'
    assert manifest.get('profile_cache_mode')==args.profile_cache_mode
    assert manifest.get('cache_root')==(str(cache_root) if cache_root else None)
    assert manifest['abc_source']==(str(args.abc_source.resolve()) if args.abc_source else None)
    assert sha(PROJECT/'scripts/experiments/common.sh')==sha(frozen/'common.sh')
    environment = {k:v for k,v in os.environ.items() if not k.startswith(('EPMAPPING_', 'EPM_', 'MAP_EGG_'))}
    environment.update(EPM_BIN=str(frozen/'epmapping_check'), EPM_ROOT=str(root),
                       EPM_TIMEOUT=str(args.timeout), EPM_RETRIES='1', EPM_CAFFEINATE='0',
                       RAYON_NUM_THREADS='1', PATH=str(frozen)+os.pathsep+os.environ['PATH'])
    def engine(stage, arm, d, rounds, recipe, extras=None):
        wd=root/stage/f'{arm}.{d}'; rowfile=wd/'row.tsv'
        if rowfile.exists():
            return rowfile.read_text()
        options=dict(manifest['goal'], **(extras or {}))
        if args.profile_cache_mode and arm!='base':
            cache=cache_root/stage/f'{arm}.{d}'
            if args.profile_cache_mode=='cold': assert not cache.exists(), f'cold cache already exists: {cache}'
            else:
                ready=root.parent/'cold'/stage/f'{arm}.{d}'/'cache_provenance.json'
                deadline=time.monotonic()+args.timeout
                while not ready.exists():
                    if time.monotonic()>deadline: raise RuntimeError(f'corresponding cold run unavailable: {stage}/{arm}.{d}')
                    time.sleep(1)
            options['MAP_EGG_PROFILE_CACHE_DIR']=str(cache)
        lib=frozen/('asap7_6in.genlib' if stage=='tab_mfs3' else 'asap7.genlib')
        options['EPMAPPING_SUPER']=str(frozen/'database.super') if options.get('EPMAPPING_SUPER')!='off' else 'off'
        flags=[f'{k}={v}' for k,v in options.items()]+[f'EPM_GENLIB={lib}']
        # Source the unchanged helper, then redirect benchmark lookup to frozen inputs.
        command=['bash','-c','source "$1"; BENCH="$2"; shift 2; run_engine "$@"','_',
                 str(PROJECT/'scripts/experiments/common.sh'),str(frozen/'benchmarks'),str(wd),d,str(rounds),recipe,*flags]
        completed=subprocess.run(command,env=environment,cwd=PROJECT,capture_output=True,text=True)
        if completed.returncode: raise RuntimeError(completed.stderr)
        row=completed.stdout.strip().splitlines()[-1]+'\n'
        assert len(row.strip().split('\t'))==14, completed.stdout
        if completed.returncode == 0 and (wd/'epmapping_baseline.v').exists():
            old_stage = 'tab_mfs3' if stage=='tab_mfs3' else 'tab_main'
            old_arm = ('ours' if arm!='base' else 'base') if stage=='tab_mfs3' else 'ae'
            assert (wd/'epmapping_baseline.v').read_bytes()==(original/old_stage/f'{old_arm}.{d}'/('epmapping_mapped.v' if old_arm=='base' else 'epmapping_baseline.v')).read_bytes(), f'original mapped input changed: {stage}/{arm}.{d}'
        rowfile.write_text(row)
        if args.profile_cache_mode and arm!='base':
            log=(wd/'log').read_text()
            outcome=dict(mode=args.profile_cache_mode,cache_directory=str(cache),cache_record_sha256={p.name:sha(p) for p in cache.glob('*.json')} if cache.exists() else {},initial_sha256=sha(wd/'epmapping_baseline.v') if (wd/'epmapping_baseline.v').exists() else None,profile=re.findall(r'egraph_profile (.*)',log),returncode=row.strip().split('\t')[-1])
            (wd/'cache_provenance.json').write_text(json.dumps(outcome,indent=2)+'\n')
        print(f'{stage}/{arm}.{d}: {row.strip()}',flush=True)
        return row
    def batch(stage, specs):
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            rows=list(pool.map(lambda s:engine(stage,*s),specs))
        (root/stage/'summary.tsv').write_text('\t'.join(HEADER)+'\n'+''.join(rows))
    def abc_run(d):
        wd=root/'tab_mfs3'/f'abc.{d}'; wd.mkdir(exist_ok=True)
        result=wd/'result.json'
        if result.exists(): return json.loads(result.read_text())
        baseline=root/'tab_mfs3'/f'base.{d}'/'epmapping_mapped.v'
        ours=root/'tab_mfs3'/f'ours.{d}'/'epmapping_baseline.v'
        assert baseline.read_bytes()==ours.read_bytes(), f'different mapped starts: {d}'
        if args.abc_source:
            source=args.abc_source.resolve(); original=source/'tab_mfs3'/f'abc.{d}'
            # The initial batch already reran ABC with the requested 1000-second cap.
            deadline=time.monotonic()+args.timeout
            while not (original/'result.json').exists():
                if time.monotonic()>deadline: raise RuntimeError(f'fresh ABC source incomplete: {d}')
                time.sleep(1)
            previous=json.loads((source/'manifest.json').read_text())
            assert previous['abc_timeout']==1000
            assert sha(source/'inputs/abc')==sha(frozen/'abc')
            cells=lambda p:[line.strip() for line in p.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
            assert cells(source/'inputs/asap7_6in.genlib')==cells(frozen/'asap7_6in.genlib')
            assert (source/'tab_mfs3'/f'base.{d}'/'epmapping_mapped.v').read_bytes()==baseline.read_bytes()
            assert (source/'inputs/benchmarks'/f'{d}.aig').read_bytes()==(frozen/'benchmarks'/f'{d}.aig').read_bytes()
            shutil.copytree(original,wd,dirs_exist_ok=True)
            metadata=dict(source_archive=str(source),source_manifest_sha256=sha(source/'manifest.json'),
                          copied_sha256={p.name:sha(p) for p in original.iterdir() if p.is_file()},
                          identical_initial_netlist=True,identical_abc_binary=True,identical_library=True)
            (wd/'reuse.json').write_text(json.dumps(metadata,indent=2)+'\n')
            return json.loads(result.read_text())
        lib=frozen/'asap7_6in.genlib'; exe=frozen/'abc'
        def call(script, log, timeout):
            import time
            start=time.monotonic()
            with log.open('w') as stream:
                try: rc=subprocess.run([str(exe),'-c',script],stdout=stream,stderr=subprocess.STDOUT,env=environment,timeout=timeout).returncode
                except subprocess.TimeoutExpired: rc=124
            return rc,time.monotonic()-start
        def metrics(path):
            text=path.read_text()
            return {k:float(re.findall(r'\b'+k+r'\s*=\s*([0-9.]+)',text)[-1]) for k in ('area','delay')}
        rc,_=call(f'read_genlib {lib}; read -m {baseline}; print_stats',wd/'baseline.log',args.abc_timeout)
        assert rc==0; native=metrics(wd/'baseline.log')
        rc,secs=call(f'read_genlib {lib}; read -m {baseline}; mfs3 -a; print_stats; write_verilog {wd / "mapped.v"}',wd/'log',args.abc_timeout)
        data=dict(design=d,rc=rc,secs=secs,cec=0,validation_secs=0,baseline=native,baseline_sha256=sha(baseline))
        if rc==0:
            data.update(metrics(wd/'log'))
            crc,csecs=call(f'read_genlib {lib}; read -m {wd / "mapped.v"}; cec -n {frozen / "benchmarks" / (d+".aig")}',wd/'cec.log',args.timeout)
            data.update(cec=int(crc==0 and 'Networks are equivalent' in (wd/'cec.log').read_text()), validation_secs=csecs)
        result.write_text(json.dumps(data,indent=2)+'\n'); print('ABC '+json.dumps(data),flush=True)
        return data
    parts=args.parts.split(','); assert set(parts)<=set(['mfs3','main','ablation'])
    if 'mfs3' in parts:
        batch('tab_mfs3', [('base',d,0,'direct',dict(EPMAPPING_SKIP='1',EPMAPPING_BASELINE_CEC='1')) for d in DESIGNS]+
              [('ours',d,1,'direct',{}) for d in DESIGNS])
        # Run independent ABC post-mapping jobs while the next experiment runs.
        # Its own 1000 s deadline is unchanged; validation remains outside it.
        abc_pool=ThreadPoolExecutor(max_workers=args.jobs)
        abc_futures=[abc_pool.submit(abc_run,d) for d in DESIGNS]
    if 'main' in parts:
        batch('tab_main',[(arm,d,r,'high_effort',{}) for arm,r in [('ae.r1',1),('ae',6)] for d in DESIGNS])
    if 'ablation' in parts:
        batch('ablation',[(arm,d,6,'high_effort',flags) for d in ABLATION for arm,flags in manifest['arms'].items()])
    if 'mfs3' in parts:
        abc_rows=[f.result() for f in abc_futures]
        abc_pool.shutdown()
        (root/'tab_mfs3'/'abc_results.json').write_text(json.dumps(abc_rows,indent=2)+'\n')

if __name__=='__main__': main()
