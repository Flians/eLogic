"""Read-only full20 comparison under search-inclusive600, with explicit historical adoption bounds."""
import argparse,copy,csv,hashlib,json,subprocess,sys
from pathlib import Path
from run_time_only600 import ROOT,EPFL,sha
from run_baselines600 import CATALOGUE
METRIC='all physical instances; max(rise,fall) block pin delay; no load or slew'
def methods(job):
    name=job['name']
    if '_core_'in name:return ['abc_map_area_resyn2','abc_map_delay_resyn2','mockturtle_emap_area_resyn2','mockturtle_emap_delay_resyn2']
    if '_eqmap_'in name:return ['abc_amap','lut_stage_amap','eqmap_lut_amap']
    if '_esyn_'in name:return ['esyn_frozen_model_amap_bounded_dag']
    return ['emorphic_final_sa']
def alias(method,name):
    return method+('__delay-area'if 'delay-area'in name else'__area-delay')if '_eqmap_'in name or '_esyn_'in name else method
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--proof-reuse',type=Path,nargs='*',default=[]);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(exist_ok=False,parents=True)
    candidate=json.loads(a.candidate.read_text());baseline=json.loads(a.baseline.read_text());old=json.loads(CATALOGUE.read_text())
    subjects={x['bench']:x for x in json.loads((ROOT/'results_v2/epfl_resyn2_subjects_20261012/report.json').read_text())['results']}
    hashes={str(p.resolve()):sha(p)for p in [a.candidate,a.baseline,CATALOGUE,Path(__file__),ROOT/'scripts/experiments/check_mapping_dominance.py']}
    reuses={};applied=[]
    for path in a.proof_reuse:
        proof=json.loads(path.read_text())
        assert proof['proof_hashes_verified']and proof['physical_netlist_unchanged']and proof['mapping_completed_before_verification']
        assert not proof['mapping_rerun']and not proof['original_mapping_algorithm_changed']and not proof['fresh_CEC_run']
        for p,h in proof['sha256'].items():assert sha(p)==h;hashes[p]=h
        journal=Path(proof['operator_verification_journal']);event=json.loads(journal.read_text())
        assert event['proof_sha256']==sha(path)and event['original_verification_status_retained']
        assert event['action']in ['SIGTERM_redundant_verifier_group','no_matching_verifier_already_finished','verifier_finished_before_signal']
        hashes[str(path.resolve())]=sha(path);hashes[str(journal)]=sha(journal)
        reuses[(proof['source_job'],proof['method'])]=(proof,path)
    def read_job(job):
        p=Path(job['report']);assert sha(p)==job['sha256'][str(p)];hashes[str(p)]=sha(p);return json.loads(p.read_text())
    crows=[];brows=[];library=None
    done={j['bench']:j for j in candidate['jobs']}
    for bench in EPFL:
        row=dict(bench=bench,method='egraph_time600',input_sha256=subjects[bench]['input_sha256'],status='pending600')
        if bench in done:
            r=read_job(done[bench]);native=r.get('native',{});library=native.get('library_sha256',library)
            row.update(status=r['status'],measured=r.get('measured'),mapping_wall_secs=r['mapping']['wall_secs'],
                optimizer_succeeded=native.get('optimizer_succeeded',False),selected=native.get('selected'),source_report=done[bench]['report'])
            if row['status']=='ok'and not row['optimizer_succeeded']:row['status']='source_only'
        crows.append(row)
    active={j['name']:j for j in baseline['jobs']}
    for job in old['jobs']:
        name=job['name'];entry=active.get(name);records={};times={};audit={}
        if entry:
            if entry.get('report'):
                report=read_job(entry);assert report['library_sha256']==library,'mixed library'
                records={r['method']:r for r in report.get('results',[])}
            if entry['observation_kind'].startswith('adopted'):
                times={r['method']:r for r in entry['timing_rows']}
            else:
                p=Path(entry['argv'][entry['argv'].index('--audit')+1])
                if p.exists():
                    assert sha(p)==entry['sha256'][str(p)];hashes[str(p)]=sha(p);audit=json.loads(p.read_text())
        for method in methods(job):
            row=dict(bench=job['bench'],method=alias(method,name),original_method=method,input_sha256=subjects[job['bench']]['input_sha256'],status='pending600',source_job=name)
            if entry:
                r=records.get(method)
                row.update(observation_kind=entry['observation_kind'],source_report=entry.get('report'))
                if r:
                    assert r['input_sha256']==row['input_sha256'];row.update(status=r['status'],measured=r.get('measured'),error=r.get('error'))
                    if (name,method)in reuses and row['status']!='ok':
                        proof,proof_path=reuses[name,method]
                        assert 'exit -15' in(r.get('error')or''),'do not override an actual negative/unknown verdict or a mapping failure'
                        assert r.get('mapping_wall_secs')is not None and r.get('native')is not None
                        assert str(Path(entry['report']).resolve())==proof['current_report']
                        assert proof['library_sha256']==library and proof['original_input_sha256']==row['input_sha256']
                        assert proof['measured']['cec']==1
                        row.update(original_status=r['status'],original_error=r['error'],error=None,status='ok',measured=proof['measured'],
                            verification_kind=proof['qualification'],verification_source=str(proof_path.resolve()),fresh_CEC_run=False)
                        applied.append(dict(bench=row['bench'],method=row['method'],proof=str(proof_path.resolve()),original_failure_retained=True))
                    if method in times:
                        row.update(mapping_wall_secs=float(times[method]['T']),T_kind=times[method]['T_kind'])
                    else:
                        clock=audit.get('clock',{}).get(job['bench']+'/'+method,{})
                        if clock:
                            row.update(mapping_wall_secs=clock['used'],T_kind=('search_inclusive_active_interval_union_wall' if audit.get('search_included') else 'mapping_only_active_interval_wall'))
                            if clock.get('exhausted')or clock.get('incomplete')or clock.get('active'):assert row['status']!='ok'
                            if (name,method)in reuses:assert abs(clock['used']-reuses[name,method][0]['mapping_wall_secs'])<1e-9
                    if row['status']=='ok':assert row.get('measured',{}).get('cec')==1 and row.get('mapping_wall_secs',601)<=600
                else:row.update(status='failed',error='driver exited without method report; original argv/log/audit retained')
            brows.append(row)
    assert len(crows)==20 and len(brows)==260
    for label,rows in [('candidate',crows),('baselines',brows)]:
        (out/(label+'.json')).write_text(json.dumps(dict(library_sha256=library,metric_model=METRIC,results=rows),indent=2)+'\n')
    fields=['bench','method','original_method','status','A','D','T','T_kind','cec','optimizer_succeeded','selected','observation_kind','error','original_status','original_error','verification_kind','verification_source','source_job','source_report']
    with (out/'A_D_T.csv').open('w')as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in crows+brows:
            v={k:r.get(k,'')for k in fields};m=r.get('measured')or{}
            v.update(A=m.get('area',''),D=m.get('delay',''),T=r.get('mapping_wall_secs',''),cec=m.get('cec',''));writer.writerow(v)
    required=sorted({r['method']for r in brows});gates={}
    for objective in ['delay-area','area-delay','pareto']:
        cmd=[sys.executable,str(ROOT/'scripts/experiments/check_mapping_dominance.py'),'--candidate',str(out/'candidate.json'),'--baselines',str(out/'baselines.json'),'--required-baselines',*required,'--objective',objective,'--output',str(out/(objective+'.json'))]
        result=subprocess.run(cmd,capture_output=True,text=True);(out/(objective+'.log')).write_text(result.stdout+result.stderr)
        assert result.returncode in [0,1]and(out/(objective+'.json')).exists(),result.stderr
        gates[objective]=json.loads((out/(objective+'.json')).read_text())['all_passed']
    summary=dict(candidate_completed=len(done),candidate_expected=20,candidate_optimized=sum(r.get('optimizer_succeeded',False)for r in crows),
        candidate_verified_outputs=sum((r.get('measured')or{}).get('cec')==1 for r in crows),candidate_statuses={s:sum(r['status']==s for r in crows)for s in sorted({r['status']for r in crows})},
        baseline_completed_groups=len(active),baseline_expected_groups=120,baseline_qualified_observations=sum(r['status']=='ok'for r in brows),baseline_statuses={s:sum(r['status']==s for r in brows)for s in sorted({r['status']for r in brows})},
        gates=gates,unconnected_tools=['MtM'],all_failures_retained=True,verification_proof_reuses=applied,hashes=hashes,
        interpretation='600s is a changed resource protocol. Adopted completed observations retain their original stricter policy and time/upper-bound labels. Source-only CEC is not optimized coverage. Lexicographic wins are not Pareto dominance.')
    (out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['#600秒完整范围检查','',json.dumps({k:v for k,v in summary.items()if k!='hashes'},ensure_ascii=False,indent=2),'','|设计|状态|优化覆盖返回|面积|延迟|映射秒|CEC|','|---|---|---|---:|---:|---:|---:|']
    for r in crows:
        m=r.get('measured')or{};lines.append('|'+ '|'.join(str(v)for v in [r['bench'],r['status'],r.get('optimizer_succeeded',''),m.get('area',''),m.get('delay',''),r.get('mapping_wall_secs',''),m.get('cec','')])+'|')
    (out/'report.md').write_text('\n'.join(lines)+'\n');print({k:v for k,v in summary.items()if k not in ['hashes','interpretation']})
if __name__=='__main__':main()
