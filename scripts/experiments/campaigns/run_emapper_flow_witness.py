#!/usr/bin/env python3
"""Frozen full-pipeline Flow-to-witness delta ablation; fresh seed charged within600s.

One canonical source and binary. No baseline runs/results are modified. The
unchanged emap executable is a freshly timed component, not a new baseline row.
Representative/full/repeat dispatch gates are explicit; finite batches only.
"""
import argparse,csv,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'scripts/emapper'))
from provenance import source_files,hashes,sha,git_revision
from physical import measure_cover
from verify import verify
PROTOCOL=R/'results_v2/emapper_seeded_epfl_20261101/protocol.json'
SEED_COMPONENT=R/'results_v2/single_output_wide_20261021/stock_emap9_cut9_reference/build/baseline_mapper'
REPRESENTATIVES=['sin','voter','mem_ctrl']
REPEATS=['sin','voter','mem_ctrl']
ARMS=['off','on']
COMMON=['--seed-native','editable','--ablation','source-editable','--recovery-inputs','overlay',
        '--recovery-noop-prefilter','on','--recovery-schedule','legacy',
        '--area-reject-cache','off','--sdc-window-inputs','0','--observe','--cut-join','legacy','--cut-truth','legacy','--batch-frontier-cache','on','--frontier-preparation','off','--recovery-cache','legacy']

def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
    p=Path(p);q=p.with_suffix('.pending');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)

def fingerprint():
    files=source_files()+list((R/'scripts/emapper').glob('*.py'))+[Path(__file__),PROTOCOL,SEED_COMPONENT,
        R/'scripts/experiments/port_identifiers.py',R/'target/release/emapper-build.json',
        R/'target/release/emapper_kernel',R/'target/release/emapper_measure',R/'emapper/profiles/delivery.json']
    files+=list(p for p in (R/'emapper/libraries/asap7').iterdir() if p.is_file())
    return hashes(sorted(set(files)))

def plan(designs):
    result=[]
    for rep in range(3):
        names=designs if rep==0 else REPEATS
        for i,name in enumerate(names):
            stage='representative' if rep==0 and name in REPRESENTATIVES else 'full' if rep==0 else 'repeat'
            for arm in ARMS[::(-1 if (i+rep)%2 else 1)]:
                result.append(dict(key=f'r{rep}_{name}_{arm}',design=name,arm=arm,repetition=rep,stage=stage))
    return result

def initialize():
    receipt=read(R/'target/release/emapper-build.json')
    assert receipt==dict(source=hashes(source_files()),binaries=hashes([R/'target/release/emapper_kernel',R/'target/release/emapper_measure'])),'rebuild first'
    assert read(R/'emapper/profiles/delivery.json')['mapping_seconds']==600
    old=read(PROTOCOL)['inputs'];assert len(old)==20
    designs=REPRESENTATIVES+[n for n in old if n not in REPRESENTATIVES]
    inputs={n:{k:v for k,v in old[n].items() if k in ('prepared','prepared_sha256','golden','golden_sha256')} for n in designs}
    for inp in inputs.values():
        for k in ('prepared','golden'):assert sha(inp[k])==inp[k+'_sha256']
    return dict(schema=1,source_commit=git_revision(),frozen=fingerprint(),inputs=inputs,
        planned=plan(designs),expected=52,jobs={},common=COMMON,seed_component=str(SEED_COMPONENT),
        tools={k:dict(path=shutil.which(k),sha256=sha(shutil.which(k))) for k in ('abc','yosys')},
        mapping_seconds=600,CEC_in_clock=False,fresh_seed_in_clock=True,
        scope='unchanged complete candidate domain; numeric memoON/preparationOFF in BOTH arms; legacy recovery cache in BOTH arms; only legacy-vs-delta Flow witness synchronization; same candidate/proposal ordering, same full score/proof/commit,600 incl fresh unchanged seed; one binary full20 plus two extra repeats sin/voter/mem_ctrl; negatives/source-only/failures retained; no winner mixing',
        baseline_runs=0,fresh_seed_component_runs_per_case=1,baseline_changes=False,GitHub_push=False)

def summarize(out,c):
    rows=[]
    for p in c['planned']:
        j=c['jobs'].get(p['key']);row=dict(p,status='missing')
        if j:
            r=read(j['report']) if j.get('report') else {};m=r.get('measured') or {}
            row.update(status=j['status'],CEC=r.get('cec'),initial_CEC=j.get('initial',{}).get('CEC'),
                A=m.get('area'),D=m.get('delay'),T=r.get('mapping_wall_secs'),
                seed_T=r.get('fresh_seed_generation',{}).get('mapping_seconds'),
                refinement_T=r.get('refinement_wall_secs'),deadline=r.get('search_timeout'),
                finished=r.get('search_finished'),net_gain=r.get('native_search_net_gain'),error=j.get('error',r.get('error')))
        rows.append(row)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'current.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    dump(out/'summary.json',dict(completed=len(c['jobs']),expected=c['expected'],
        final_CEC=sum(x.get('CEC')==1 for x in rows),initial_CEC=sum(x.get('initial_CEC')==1 for x in rows),
        failures=[x for x in rows if x['status'] not in ('missing','ok','source_only')],
        missing=sum(x['status']=='missing' for x in rows),baseline_runs=0))

def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-jobs',type=int,default=2);ap.add_argument('--stage',choices=['representative','full','repeat'],default='representative')
    ap.add_argument('--verify-only',action='store_true');a=ap.parse_args();assert a.max_jobs>0
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);locks=[]
    for path in [R/'results_v2/independent_components_20261019/global.lock',out/'runner.lock']:
        path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    existing=(out/'campaign.json').exists()
    c=read(out/'campaign.json') if existing else initialize()
    if not existing:
        for name,digest in c['frozen'].items():
            source=R/name;target=out/'frozen'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target);assert sha(target)==digest
    assert c['frozen']==fingerprint(),'source/binary/harness drift; preserve frozen campaign'
    dump(out/'campaign.json',c)
    for j in c['jobs'].values():
        for file,h in j['artifacts'].items():assert sha(Path(j['case'])/file)==h,file
        if j.get('report'):assert sha(j['report'])==j['report_sha256']
    if a.verify_only:summarize(out,c);return
    # No implicit promotion after representatives: caller explicitly opens stage.
    if a.stage in ('full','repeat'):
        required=[p for p in c['planned'] if p['stage']=='representative']
        assert all(c['jobs'].get(p['key'],{}).get('status') in ('ok','source_only') for p in required),'complete and audit representatives first'
    if a.stage=='repeat':
        assert all(p['key'] in c['jobs'] for p in c['planned'] if p['repetition']==0),'complete full20 attempts before repeats'
    count=0
    for p in c['planned']:
        if p['stage']!=a.stage or p['key'] in c['jobs']:continue
        if count>=a.max_jobs:break
        assert c['frozen']==fingerprint()
        for tool in c['tools'].values():assert sha(tool['path'])==tool['sha256']
        inp=c['inputs'][p['design']]
        for k in ('prepared','golden'):assert sha(inp[k])==inp[k+'_sha256']
        case=out/'cases'/p['key'];assert not case.exists(),'audit and retain unrecorded partial; never overwrite'
        argv=[sys.executable,str(R/'scripts/emapper/run.py'),inp['prepared'],'--golden',inp['golden'],
              '--fresh-emap-seed',str(SEED_COMPONENT),'--output',str(case),*COMMON,'--flow-witness','delta' if p['arm']=='on' else 'legacy']
        dump(out/(p['key']+'-attempt.json'),dict(argv=argv,input=inp,frozen=c['frozen']))
        dump(out/'state.json',dict(status='running',active=p,completed=len(c['jobs']),expected=c['expected']))
        print(json.dumps(dict(event='start',job=p)),flush=True)
        started=time.monotonic()
        with (out/(p['key']+'.log')).open('x') as log:
            proc=subprocess.run(argv,cwd=R,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(f.fileno() for f in locks))
        case.mkdir(parents=True,exist_ok=True);report=case/'report.json'
        j=dict(case=str(case),runner_exit=proc.returncode,status='unmeasured_exit',argv=argv,whole_case_wall_secs=time.monotonic()-started)
        if report.exists():
            r=read(report);assert r['status']!='running','retain interrupted partial report'
            j.update(report=str(report),report_sha256=sha(report),status=r['status'])
            try:
                assert r.get('cec')==1 and r.get('evaluation',{}).get('exact_agreement'),'final CEC or metric disagreement'
                assert r.get('seed_generation_in_mapping_clock') is True,'fresh seed cost missing'
                assert r.get('external_native_import',{}).get('applied'),'native import missing'
                manifest=read(case/'incumbents/00000001/manifest.json')
                assert manifest['kind']=='complete-source-cover'
                initial=measure_cover(R/'emapper/libraries/asap7/asap7_full.genlib',case/'incumbents/00000001/mapped.v',case/'native-initial',R/'target/release/emapper_measure',c['tools']['yosys']['path'])
                os.environ['EGM_YOSYS']=c['tools']['yosys']['path'];os.environ['BENCH_RS_EMAP_ABC']=c['tools']['abc']['path']
                initial['CEC']=int(verify(R/'emapper/libraries/asap7/cells.v',Path(inp['golden']),Path(initial['netjson']),case/'native-initial/cec'))
                initial['exact_AD_agreement']=all(initial['measured'][k]==manifest[k] for k in ('area','delay'));j['initial']=initial
                assert initial['CEC']==1 and initial['exact_AD_agreement'],'invalid initial cover'
            except Exception as exc:j.update(status='invalid',error=type(exc).__name__+': '+str(exc))
        else:j['error']='No final report; raw attempt and log retained'
        assert c['frozen']==fingerprint(),'source drift during case'
        j['artifacts']={str(f.relative_to(case)):sha(f) for f in sorted(case.rglob('*')) if f.is_file()}
        j['attempt_sha256']=sha(out/(p['key']+'-attempt.json'));j['log_sha256']=sha(out/(p['key']+'.log'))
        c['jobs'][p['key']]=j;dump(out/'campaign.json',c);summarize(out,c);count+=1
        print(json.dumps(dict(event='end',key=p['key'],status=j['status'])),flush=True)
    summarize(out,c);dump(out/'state.json',dict(status='batch_complete',stage=a.stage,completed=len(c['jobs']),expected=c['expected']))

if __name__=='__main__':main()
