"""Audit completed600 matrix: actual outputs, source distinction, clock and hashes."""
import argparse,json,math
from pathlib import Path
from run_time_only600 import sha
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    state=json.loads(a.campaign.read_text());assert len(state['jobs'])==state['expected_jobs']==20 and state['status']=='attempted'
    files=dict(state['fingerprint'])
    for j in state['jobs']:files.update(j['sha256'])
    for p,h in files.items():assert sha(p)==h,'artifact/source drift: '+p
    records=[]
    for j in state['jobs']:
        p=Path(j['report']);r=json.loads(p.read_text());n=r['native'];clock=r['mapping'];folder=p.parent
        assert abs(clock['deadline_monotonic']-clock['started_monotonic']-600)<1e-6
        assert r['measured']['cec']==1 and all(math.isfinite(r['measured'][x])and r['measured'][x]>=0 for x in ['area','delay'])
        check=json.loads((folder/'functional-cec/verification.json').read_text())
        assert check['cec']==1 and check['original_gate_bytes_unchanged']and check['functional_models_from_verilog_not_mapper_truth_tables']and not check['mapper_called']
        for k in ['search_ratio','global_cumulative_node_allowance','optimizer_pool_limit','global_retained_candidate_limit','count_limits','per_stage_time_limits','optimizer_address_space_limit_bytes']:assert n[k]is None
        assert n['time_only']and n['source_construction_is_inside600s']and n['source_sha256']==sha(n['source_verilog'])
        assert sha(folder/'raw.v')==n['selected_verilog_sha256']
        imp=json.loads((folder/'attempt0.pool.import.json').read_text())
        assert imp['all_original_helpers_retained']and imp['import_complete']
        item=dict(bench=r['bench'],status=r['status'],mapping_secs=clock['wall_secs'],CEC=1,
            optimizer_succeeded=n['optimizer_succeeded'],selected=n['selected'],actual_imported_enodes=imp['actual_imported_enodes'],partial_native_file_retained=(folder/'attempt0.v').exists())
        if n['optimizer_succeeded']:
            assert r['status']=='ok'and clock['exit_code']==0 and not clock['timeout']and clock['wall_secs']<600 and n['mapping_complete']
            actual=n['optimizer_native'];cap=json.loads((folder/'attempt0.pool.retention-policy.json').read_text())
            assert cap['retained_count_ceiling']is None and cap['pruned_by_count']==0
            assert not actual['instance_limit_hit']and not actual['cut_limit_hit']
            item.update(actual_enodes=actual['enodes'],retained_instances=actual['instances'],actual_max_cuts_per_class=actual['max_cuts_per_class'],recovery_secs=actual['heuristic_secs'])
        else:
            assert r['status']=='mapping_timeout'and clock['timeout']and not n['mapping_complete']and n['selected']=='source_witness'
            assert sha(folder/'raw.v')==n['source_sha256']
        records.append(item)
    out=dict(cases=20,independent_named_port_CEC=20,complete_optimized_covers=sum(x['optimizer_succeeded']for x in records),
        source_only_timeouts=sum(not x['optimizer_succeeded']for x in records),all_configured_global_count_ceilings_removed=True,
        no_count_cap_truncation_in_completed_covers=True,mapping_limit_secs=600,timeout_drain_overhead_is_not_success=True,
        source_and_artifact_files_hashed=len(files),records=records,source_campaign_sha256=sha(a.campaign),audit_source_sha256=sha(Path(__file__)))
    a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items()if k!='records'})
if __name__=='__main__':main()
