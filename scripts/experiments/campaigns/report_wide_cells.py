"""Read-only complete-scope A/D/T and directed coverage; missing and failed cases remain visible."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from run_epfl_baselines import EPFL


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    state = json.loads(a.campaign.read_text())
    rows, lookup, libraries = [], {}, set()
    for job in state['jobs']:
        assert all(sha(p) == h for p, h in job['sha256'].items()), job['name']
        report = json.loads(Path(job['report']).read_text()) if job.get('report') else None
        if report:
            libraries.add(report['library_sha256'])
        value = report['results'][0] if report else {}
        measured = value.get('measured', {})
        policy, search = value.get('resource_policy', {}), value.get('scalability', {})
        if search:
            assert value['mapping_wall_secs'] <= 300
            assert search['cumulative_accounted_enode_upper'] <= policy['node_limit']
            assert search['source_plus_optimizer_retained_upper'] <= policy['global_retained_candidate_limit']
            assert not policy['refund_on_rebuild']
        folder = Path(job['report']).parent if report else a.campaign.parent / job['name']
        coverage = [json.loads(p.read_text()) for p in folder.rglob('*.wide-generation.json')]
        wide_names={'AO322x2_ASAP7_75t_R','OA331x2_ASAP7_75t_R','AO331x2_ASAP7_75t_R','AO332x2_ASAP7_75t_R','OA332x2_ASAP7_75t_R','AO333x1_ASAP7_75t_R','OA333x2_ASAP7_75t_R'}
        selected_wide=None
        netlists=list(folder.glob('*.egraph_staged_control_resyn2/mapped.json'))
        if len(netlists)==1:
            data=json.loads(netlists[0].read_text())
            selected_wide=sum(c.get('type') in wide_names for m in data['modules'].values() for c in m.get('cells',{}).values())
        repetition=int(job['name'].split('_',1)[0][1:])
        value = dict(bench=job['bench'], repetition=repetition, profile=job['profile'], status=value.get('status','failed'),
                     A=measured.get('area'), D=measured.get('delay'), T=value.get('mapping_wall_secs'),
                     CEC=measured.get('cec'), optimized=search.get('optimizer_succeeded', False),
                     selected=search.get('selected'), wide_candidates=sum(v['global_new_candidates'] for v in coverage),
                     wide_work_used=sum(v['work_used'] for v in coverage),
                     work_exhausted_cells=sum(sum(c['work_exhausted'] for c in v['cells']) for v in coverage),
                     source_job=job['name'], selected_wide=selected_wide)
        key=(job['bench'],job['profile'],repetition)
        assert key not in lookup, 'duplicate design/profile/repetition'
        lookup[key] = value
    assert len(libraries) <= 1, 'mixed libraries'
    repetitions=range(state['configuration'].get('rounds',1))
    for repetition in repetitions:
        for bench in EPFL:
            for profile in [0,1]:
                rows.append(lookup.get((bench,profile,repetition),dict(bench=bench,profile=profile,repetition=repetition,status='missing')))
    paired = []
    for repetition in repetitions:
        for bench in EPFL:
            b, c = lookup.get((bench,0,repetition)), lookup.get((bench,1,repetition))
            if not b or not c or any(v['status']!='ok' for v in [b,c]):
                continue
            paired.append(dict(bench=bench,repetition=repetition,delta_A=c['A']-b['A'],delta_D=c['D']-b['D'],delta_T=c['T']-b['T'],
                               both_optimized=b['optimized'] and c['optimized']))
    a.output.mkdir(exist_ok=False,parents=True)
    columns = ['bench','repetition','profile','status','A','D','T','CEC','optimized','selected','wide_candidates','wide_work_used','work_exhausted_cells','source_job','selected_wide']
    with (a.output/'A_D_T.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    (a.output/'validation.json').write_text(json.dumps(dict(completed_jobs=len(state['jobs']),expected_jobs=state['expected_jobs'],
        library_hashes=sorted(libraries),profile_parameter=state['configuration'].get('profile_parameter','EGM_WIDE_MATCH'),
        profile0=state['configuration'].get('profiles',['wide matching off','wide matching on'])[0],
        profile1=state['configuration'].get('profiles',['wide matching off','wide matching on'])[1],
        source_only_is_not_optimization_success=True,baseline_comparison_complete=False,main_promoted=False,
        paired=paired,rows=rows,campaign=str(a.campaign.resolve()),campaign_sha256=sha(a.campaign),reporter_sha256=sha(__file__)),indent=2)+'\n')

if __name__ == '__main__':
    main()
