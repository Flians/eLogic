#!/usr/bin/env python3
"""Frozen serial EPFL selected-proof / fair matcher / accepted-cover delta study.

Only eMapper is run. Historical uniform emap-delay seeds are supplied unchanged.
Each run keeps the 600s deadline; CEC and public physical measurement are outside.
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(R / 'scripts/emapper'))
from provenance import source_files, hashes, sha, git_revision
from archive import seal, verify_files
from physical import measure_cover
from verify import verify
from selected_evidence import replay as replay_selected

PROTOCOL = R / 'results_v2/emapper_seeded_epfl_20261101/protocol.json'
ARMS = {
    'reference': ['--sdc-window-inputs', '0', '--match-schedule', 'legacy'],
    'evidence': ['--sdc-window-inputs', '0', '--selected-provenance', '--match-schedule', 'legacy'],
    'fair_match': ['--sdc-window-inputs', '0', '--selected-provenance', '--match-schedule', 'fair'],
    'cover_delta': ['--sdc-window-inputs', '0', '--selected-provenance', '--match-schedule', 'fair', '--recovery-cache', 'delta'],
}
COMMON = ['--seed-native', 'editable', '--ablation', 'source-editable',
          '--recovery-inputs', 'overlay', '--recovery-noop-prefilter', 'on',
          '--recovery-cache', 'legacy', '--recovery-schedule', 'legacy',
          '--area-reject-cache', 'off', '--observe']
TARGETS = ['sin', 'max', 'square', 'voter']

def read(p):
    return json.loads(Path(p).read_text())

def dump(p, value):
    p = Path(p); temp = p.with_suffix('.pending')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(p)

def fingerprint():
    files = source_files() + list((R/'scripts/emapper').glob('*.py')) + [
        Path(__file__), PROTOCOL, R/'target/release/emapper-build.json',
        R/'target/release/emapper_kernel', R/'target/release/emapper_measure',
        R/'emapper/profiles/delivery.json']
    files += [p for p in (R/'emapper/libraries/asap7').glob('*') if p.is_file()]
    return hashes(files)

def plan(designs):
    jobs = []
    for rep in range(3):
        names = designs if rep == 0 else TARGETS
        for i, name in enumerate(names):
            order = list(ARMS); shift = (i + rep) % len(order); order = order[shift:] + order[:shift]
            for arm in order:
                jobs.append(dict(key=f'r{rep}_{name}_{arm}', design=name, arm=arm, repetition=rep))
    return jobs

def initialize():
    inputs = read(PROTOCOL)['inputs']
    assert len(inputs) == 20 and set(TARGETS) <= set(inputs)
    designs = ['ctrl', 'i2c', 'sin', 'max', 'square', 'voter'] + [n for n in inputs if n not in ('ctrl', 'i2c', 'sin', 'max', 'square', 'voter')]
    for inp in inputs.values():
        for k in ('prepared', 'golden', 'seed'):
            assert sha(inp[k]) == inp[k+'_sha256']
    planned = plan(designs)
    return dict(schema=1, source_commit=git_revision(), frozen=fingerprint(),
                inputs=inputs, designs=designs, arms=ARMS, common_cli=COMMON,
                planned=planned, expected=len(planned), jobs={},
                mapping_seconds=600, CEC_in_clock=False, baseline_runs=0,
                AIG_rounds=5, alpha=10, wide_rounds=2, beta=2, GitHub_push=False,
                tools={k:dict(path=shutil.which(k), sha256=sha(shutil.which(k))) for k in ('yosys','abc')},
                scope='same-binary sequential increments: reference -> selected proof -> fair matching -> accepted-cover delta; all20x4 plus r1/r2 sin/max/square/voter; no per-design/repeat selection; all600s; native/public metrics, initial/finalCEC and selected proof replay; no baseline runs')

def summarize(out, c):
    rows = []
    for p in c['planned']:
        j = c['jobs'].get(p['key'])
        row = dict(p, status='missing')
        if j:
            r = read(j['report']) if j.get('report') else {}
            m = r.get('measured') or {}
            row.update(status=j['status'], CEC=r.get('cec'),
                       initial_CEC=j.get('initial',{}).get('CEC'), A=m.get('area'),
                       D=m.get('delay'), T=r.get('mapping_wall_secs'),
                       deadline=r.get('search_timeout'), finished=r.get('search_finished'),
                       native_search_net_gain=r.get('native_search_net_gain'),
                       selected_binding_proof=r.get('selected_evidence'),
                       evidence_error=j.get('evidence_error'),
                       error=j.get('error',r.get('error')))
        rows.append(row)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (out/'current.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    dump(out/'summary.json', dict(completed=len(c['jobs']), expected=c['expected'],
         final_CEC=sum(x.get('CEC')==1 for x in rows),
         initial_CEC=sum(x.get('initial_CEC')==1 for x in rows),
         failures=[x for x in rows if x['status'] not in ('missing','ok','source_only')],
         incomplete_initial_evidence=[x['key'] for x in rows if x.get('evidence_error')],
         missing=sum(x['status']=='missing' for x in rows), baseline_runs=0))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-jobs',type=int,default=4);ap.add_argument('--verify-artifacts-only',action='store_true')
    a=ap.parse_args();assert a.max_jobs>0
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);locks=[]
    for path in [R/'results_v2/independent_components_20261019/global.lock',out/'runner.lock']:
        path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    c=read(out/'campaign.json') if (out/'campaign.json').exists() else initialize()
    assert c['frozen']==fingerprint(),'source/binary/harness drift';dump(out/'campaign.json',c)
    for j in c['jobs'].values():
        if j.get('report'):assert sha(j['report'])==j['report_sha256']
        if a.verify_artifacts_only:verify_files(Path(j['case']),j['artifacts'],j['capsule'])
    if a.verify_artifacts_only:
        dump(out/'artifact-validation.json',dict(campaign_sha256=sha(out/'campaign.json'),cases=len(c['jobs']),all_hashes_match=True));summarize(out,c);return
    count=0
    for p in c['planned']:
        key=p['key']
        if key in c['jobs']:continue
        if count>=a.max_jobs:
            summarize(out,c);dump(out/'state.json',dict(status='batch_complete',completed=len(c['jobs']),expected=c['expected']));return
        assert c['frozen']==fingerprint()
        for t in c['tools'].values():assert sha(t['path'])==t['sha256']
        inp=c['inputs'][p['design']]
        for k in ('prepared','golden','seed'):assert sha(inp[k])==inp[k+'_sha256']
        case=out/'cases'/key;assert not case.exists(),'retain and audit unrecorded partial before recovery'
        argv=[sys.executable,str(R/'scripts/emapper/run.py'),inp['prepared'],'--golden',inp['golden'],'--seed-verilog',inp['seed'],'--output',str(case),*COMMON,*ARMS[p['arm']]]
        dump(out/(key+'-attempt.json'),dict(argv=argv,input=inp,frozen=c['frozen']))
        dump(out/'state.json',dict(status='running',active=key,completed=len(c['jobs']),expected=c['expected']))
        print(json.dumps(dict(event='start',key=key)),flush=True)
        with (out/(key+'.log')).open('x') as log:
            proc=subprocess.run(argv,cwd=R,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(f.fileno() for f in locks))
        case.mkdir(parents=True,exist_ok=True)
        report=case/'report.json';j=dict(case=str(case),runner_exit=proc.returncode,status='unmeasured_exit',argv=argv,
            attempt_sha256=sha(out/(key+'-attempt.json')),log_sha256=sha(out/(key+'.log')))
        if report.exists():
            r=read(report);assert r['status']!='running','retain interrupted partial report'
            j.update(report=str(report),report_sha256=sha(report),status=r['status'])
            if r.get('cec') != 1 or not r.get('evaluation',{}).get('exact_agreement'):
                j['status']='invalid';j['error']='Final equivalence or public/native metrics invalid; original report retained'
            if p['arm']!='reference' and r.get('selection',{}).get('incumbent') and not r.get('selected_evidence',{}).get('valid'):
                j['status']='invalid';j['error']='Selected proof replay missing/failed; original report retained'
            try:
                assert r.get('external_native_import',{}).get('applied'),'outer fallback is not native seed import'
                manifest=read(case/'incumbents/00000001/manifest.json');assert manifest['kind']=='complete-source-cover'
                initial=measure_cover(R/'emapper/libraries/asap7/asap7_full.genlib',case/'incumbents/00000001/mapped.v',case/'native-initial',R/'target/release/emapper_measure',shutil.which('yosys'))
                os.environ['EGM_YOSYS']=shutil.which('yosys');os.environ['BENCH_RS_EMAP_ABC']=shutil.which('abc')
                initial['CEC']=int(verify(R/'emapper/libraries/asap7/cells.v',Path(inp['golden']),Path(initial['netjson']),case/'native-initial/cec'))
                initial['exact_AD_agreement']=all(initial['measured'][k]==manifest[k] for k in ('area','delay'));j['initial']=initial
                assert initial['CEC']==1 and initial['exact_AD_agreement'],'invalid editable initial cover'
                if p['arm']!='reference':
                    j['initial_selected_evidence']=replay_selected(case/'incumbents/00000001/manifest.json', case/'selected-provenance/objects', initial['netjson'])
                assert r.get('cec')==1 and r.get('evaluation',{}).get('exact_agreement'),'final CEC/metric failure'
            except Exception as e:j.update(evidence_error=type(e).__name__+': '+str(e),initial_evidence_complete=False)
            else:j['initial_evidence_complete']=True
        else:j['error']='process exited without a report; raw attempt/log retained'
        assert c['frozen']==fingerprint(),'source drift during timed job'
        j['artifacts']={str(p.relative_to(case)):sha(p) for p in sorted(case.rglob('*')) if p.is_file()}
        j['capsule']=seal(case,out/'capsules');verify_files(case,j['artifacts'],j['capsule'])
        c['jobs'][key]=j;dump(out/'campaign.json',c);summarize(out,c);count+=1
        print(json.dumps(dict(event='end',key=key,status=j['status'])),flush=True)
    dump(out/'state.json',dict(status='complete',completed=len(c['jobs']),expected=c['expected']))

if __name__=='__main__':
    main()
