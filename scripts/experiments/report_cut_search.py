#!/usr/bin/env python3
"""Audit frozen search exploration and compare fixed configurations on matched cohorts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

PROJECT = Path(__file__).resolve().parents[2]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def mean(records):
    return {metric: statistics.mean(100*(r[out]/r[base]-1) for r in records)
            for metric,out,base in [('area','out_a','base_a'),('delay','out_d','base_d')]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args()
    root=args.root.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['whole_window_synthesis'] is False
    for n,h in manifest['frozen_sha256'].items():
        assert sha(root/'inputs'/n)==h, n
    for n,h in manifest['input_sha256'].items():
        assert sha(PROJECT/n)==h, n
    records={}
    failures=[]
    baselines={}
    for phase in ['tuning','validation','rounds','all','converged']:
        if not (root/phase/'summary.tsv').exists():
            continue
        plan=json.loads((root/phase/'plan.json').read_text())
        rows=list(csv.DictReader((root/phase/'summary.tsv').open(),delimiter='\t'))
        assert {(r['stage'],r['design']) for r in rows}=={(p,d) for p in plan['profiles'] for d in plan['designs']}
        assert len(rows)==len(plan['profiles'])*len(plan['designs'])
        for row in rows:
            profile,design=row['stage'],row['design']
            folder=root/phase/f'{profile}.{design}'
            assert (folder/'row.tsv').read_text().strip().split('\t')==list(row.values())
            if row['rc']!='0' or row['cec']!='1':
                failures.append(dict(phase=phase,profile=profile,design=design,rc=row['rc'],cec=row['cec']))
                continue
            log=(folder/'log').read_text()
            assert re.search(r'^\[i\] cec=1$',log,re.M)
            assert re.search(r'window_synth windows=0 applied=0',log)
            environment=dict(line.split('=',1) for line in (folder/'experiment_env.txt').read_text().splitlines() if '=' in line)
            assert environment['EPMAPPING_WINDOW_SYN']=='0'
            for k,v in manifest['profiles'][profile].items():
                assert environment[k]==v,(profile,k)
            assert environment['EPMAPPING_ROUNDS']==str(plan['rounds'])
            assert f'max_rounds={plan["rounds"]}' in log
            assert 1<=int(row['rnd'])<=plan['rounds']
            values=[float(row[k]) for k in ['base_a','base_d','out_a','out_d']]
            assert all(math.isfinite(v) for v in values)
            found=re.findall(r'epmapping area ([0-9.eE+-]+) -> ([0-9.eE+-]+) delay ([0-9.eE+-]+) -> ([0-9.eE+-]+)',log)[-1]
            assert values==[float(found[i]) for i in [0,2,1,3]]
            if environment.get('EPMAPPING_PATH_OBJECTIVE')=='1':
                initial=(math.floor(values[1]*1e3+.5),math.floor(values[0]*1e4+.5))
                final=(math.floor(values[3]*1e3+.5),math.floor(values[2]*1e4+.5))
                assert final<=initial,(phase,profile,design)
                assert 'path_objective=1' in log and 'objective_order=delay_area' in log
            else:
                assert values[2]<=values[0]+1e-4,(phase,profile,design)
            if environment.get('EPMAPPING_AREA_UNDER_DELAY')=='1':
                assert 'area_under_delay=1, path_objective=0' in log
                assert 'objective_order=area_delay' in log
                assert 'max_area_growth=0, max_delay_growth=0, local_area_growth=0' in log
            assert values[3]<=values[1]+1e-3,(phase,profile,design)
            baseline=(folder/'epmapping_baseline.v').read_bytes()
            assert design not in baselines or baselines[design]==baseline,design
            baselines[design]=baseline
            abc_input=PROJECT/'results_v4_cut_only/tab_mfs3'/f'base.{design}'/'epmapping_mapped.v'
            assert baseline==abc_input.read_bytes(),f'ABC input differs: {design}'
            data=dict(zip(['base_a','base_d','out_a','out_d'],values),profile=profile,design=design,
                      rounds=int(row['rnd']),wall_seconds=float(row['secs']))
            data['optimizer_seconds']=float(re.search(r'epmapping timing .*optimizer=([0-9.eE+-]+)',log)[1])
            data['rollbacks']=int(re.search(r'objective_guard rollbacks=(\d+)',log)[1])
            if environment.get('EPMAPPING_PATH_OBJECTIVE')=='1':
                counters=re.search(r'path_objective critical_roots=(\d+) noncritical_roots=(\d+) critical_replacements=(\d+) noncritical_replacements=(\d+)',log)
                assert counters
                data['path_counters']=dict(zip(['critical_roots','noncritical_roots','critical_replacements','noncritical_replacements'],map(int,counters.groups())))
                assert sum(data['path_counters'][k] for k in ['critical_roots','noncritical_roots'])==int(re.search(r'epmapping roots=(\d+)',log)[1])
                assert sum(data['path_counters'][k] for k in ['critical_replacements','noncritical_replacements'])==int(re.search(r'epmapping replacements=(\d+)',log)[1])
            records[(phase,profile,design)]=data
    report=['# Cut-only search exploration','','All configurations keep whole-window synthesis off. One fixed configuration applies to every design. All/converged phases cover all sixteen designs; tuning/validation use six declared tuning and ten validation designs, and rounds covers the six exploration designs. Every admitted result passes CEC, raw-log agreement, paired serialized input equality and objective/budget checks. Bounded-area runs minimize area under the fixed initial output-delay budget. Historical path-objective runs allow area growth for critical timing improvement and guard the (delay, area) pair.','',
            '| Cohort | Configuration | Completed | Mean area change (%) | Mean delay change (%) |',
            '|---|---|---:|---:|---:|']
    cohorts={}
    for phase in ['tuning','validation','rounds','all16','converged16']:
        phases=['tuning','validation','all'] if phase=='all16' else ['converged'] if phase=='converged16' else [phase]
        for profile in manifest['profiles']:
            rs=[r for (ph,p,d),r in records.items() if ph in phases and p==profile]
            expected=(16 if phase in ('all16','converged16') else len(manifest['validation_designs']) if phase=='validation' else 6)
            if not rs or (phase in ('all16','converged16') and len(rs)!=expected):
                continue
            m=mean(rs)
            cohorts[f'{phase}/{profile}']=dict(count=len(rs),expected=expected,**m)
            report.append(f'| {phase} | {profile} | {len(rs)}/{expected} | {m["area"]:+.3f} | {m["delay"]:+.3f} |')
    abc_rows=list(csv.DictReader((PROJECT/'results_v4_cut_only/tab_mfs3/abc.tsv').open(),delimiter='\t'))
    abc={r['design']:r for r in abc_rows if r['rc']=='0' and r['cec']=='1'}
    report+=['','Final mapped outputs versus ABC mfs3 on its 13 completed designs (same serialized start; ABC timeouts excluded):','',
             '| Configuration | Matched | Mean area difference (%) | Mean delay difference (%) | Smaller / faster / Pareto dominates |',
             '|---|---:|---:|---:|---:|']
    versus_abc={}
    for profile in manifest['profiles']:
        rs=[r for (ph,p,d),r in records.items() if ph in ['tuning','validation','all'] and p==profile and d in abc]
        if len(rs)!=len(abc):
            continue
        area=statistics.mean(100*(r['out_a']/float(abc[r['design']]['area'])-1) for r in rs)
        delay=statistics.mean(100*(r['out_d']/float(abc[r['design']]['delay'])-1) for r in rs)
        aw=dw=pw=0
        for r in rs:
            aa,ad=float(abc[r['design']]['area']),float(abc[r['design']]['delay'])
            aw+=r['out_a']<aa-1e-4
            dw+=r['out_d']<ad-1e-3
            pw+=(r['out_a']<=aa+1e-4 and r['out_d']<=ad+1e-3 and (r['out_a']<aa-1e-4 or r['out_d']<ad-1e-3))
        versus_abc[profile]=dict(count=len(rs),area=area,delay=delay,area_wins=aw,delay_wins=dw,pareto_wins=pw)
        report.append(f'| {profile} | {len(rs)} | {area:+.3f} | {delay:+.3f} | {aw} / {dw} / {pw} |')
    report+=['','Negative means smaller/faster. Means average per-design percentages. A larger area saving can consume some delay improvement even when final delay never exceeds the initial mapping. Pure lexicographic v5 runs are separate experiments. No per-design best-of selection is used.','',f'Failures/timeouts: {json.dumps(failures)}','']
    (root/'search_report.md').write_text('\n'.join(report))
    if not failures and all(cohorts.get(f'all16/{p}',{}).get('count')==16 for p in ['control','gain','required']):
        macros=['% GENERATED by report_cut_search.py from CEC-verified frozen experiments.']
        for profile,label in [('control','SearchControl'),('gain','SearchGain'),('required','SearchRequired')]:
            for cohort,suffix in [('all16',''),('validation','Validation'),('rounds','Rounds')]:
                value=cohorts.get(f'{cohort}/{profile}')
                if value:
                    for metric in ['area','delay']:
                        macros.append('\\newcommand{\\'+label+suffix+metric.title()+'}{'+f'{value[metric]:+.2f}'+'}')
        (root/'generated_search.tex').write_text('\n'.join(macros)+'\n')
    else:
        # A failed or incomplete rerun must never leave old publication macros admissible.
        (root/'generated_search.tex').unlink(missing_ok=True)
    if not failures and cohorts.get('all16/path',{}).get('count')==16:
        macros=['% GENERATED by report_cut_search.py from CEC-verified path-objective experiments.']
        for profile,label in [('path','Path'),('required','PathAreaControl')]:
            value=cohorts.get(f'all16/{profile}')
            if value:
                for metric in ['area','delay']:
                    macros.append('\\newcommand{\\'+label+metric.title()+'}{'+f'{value[metric]:+.2f}'+'}')
        path_rows=[r for (ph,p,d),r in records.items() if ph in ['all','tuning','validation'] and p=='path']
        sqrt=next(r for r in path_rows if r['design']=='sqrt')
        for metric,out,base in [('Area','out_a','base_a'),('Delay','out_d','base_d')]:
            macros.append('\\newcommand{\\PathSqrt'+metric+'}{'+f'{100*(sqrt[out]/sqrt[base]-1):+.2f}'+'}')
        macros.append('\\newcommand{\\PathBothWins}{'+str(sum(r['out_a']<r['base_a']-1e-4 and r['out_d']<r['base_d']-1e-3 for r in path_rows))+'}')
        (root/'generated_path.tex').write_text('\n'.join(macros)+'\n')
    else:
        (root/'generated_path.tex').unlink(missing_ok=True)
    if (not failures and all(cohorts.get(f'{c}/bounded',{}).get('count')==16
                             for c in ['all16','converged16']) and
        json.loads((root/'all/plan.json').read_text())['rounds']==1 and
        json.loads((root/'converged/plan.json').read_text())['rounds']==6):
        macros=['% GENERATED by report_cut_search.py from constrained-area experiments.']
        for cohort,label in [('all16','BoundOne'),('converged16','BoundSix')]:
            value=cohorts.get(f'{cohort}/bounded')
            if value:
                for metric in ['area','delay']:
                    macros.append('\\newcommand{\\'+label+metric.title()+'}{'+f'{value[metric]:+.2f}'+'}')
        (root/'generated_bound.tex').write_text('\n'.join(macros)+'\n')
    else:
        (root/'generated_bound.tex').unlink(missing_ok=True)
    artifact_hashes={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name not in ['verified_search.json']}
    output=dict(cohorts=cohorts,versus_abc=versus_abc,failures=failures,
                records=[dict(phase=ph,**r) for (ph,p,d),r in records.items()],sha256=artifact_hashes)
    (root/'verified_search.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(cohorts=cohorts,versus_abc=versus_abc,failures=failures),indent=2))

if __name__=='__main__':
    main()
