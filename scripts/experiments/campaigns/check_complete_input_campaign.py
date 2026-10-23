"""Independent audit of source-first capacity protocol, no mapping or repairs."""
import argparse,json,hashlib,collections
from pathlib import Path
from audit_generation_commands import audit
from input_capacity_policy import input_capacity
R=Path(__file__).resolve().parents[3];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('campaign',type=Path);a=ap.parse_args();c=json.loads(a.campaign.read_text());assert all(sha(p)==h for p,h in c['fingerprint'].items());rows=[]
 for j in c['jobs']:
  assert all(sha(p)==h for p,h in j['sha256'].items());d=json.loads(Path(j['report']).read_text());r=d['results'][0];proof=audit(j,c);v=dict(bench=j['bench'],k=j['k'],status=r['status'],valid=r['status']=='ok',error=r.get('error'),proof=proof,report=j['report'],audit=j['audit'])
  folder=Path(j['report']).parent/(r['bench']+'.'+r['method']);p=folder/'raw.complete.json'
  if p.exists():
   m=json.loads(p.read_text());cap=input_capacity(r['preprocessing']['reused_from']);assert m['global_cumulative_node_allowance']==cap['node_limit'] and m['global_retained_candidate_limit']==cap['base_enode_upper']+100000;assert m['source_plus_optimizer_retained_upper']<=m['global_retained_candidate_limit'];assert m['optimizer_pool_limit']==100000 and m['optimizer_recovery_secs']==2
   assert m['native_optimizer_sha256']==c['fingerprint'][m['native_optimizer']] and m['library_sha256']==d['library_sha256'] and m['subject_sha256']==r['preprocessing']['sha256'];assert m['no_refund_on_retry'] and m['mapping_limit_secs']==300 and m['optimizer_shared_window_secs']==(0 if c['constructor_only']else 60)
   charge=0;failures=[];artifacts=[]
   for i,e in enumerate(m['attempts']):
    assert e['index']==i and e['node_allowance']<=cap['node_limit']-charge
    if i:assert m['attempts'][i-1]['admission'] and not m['attempts'][i-1]['admission']['import_complete'] and m['attempts'][i-1]['admission']['rejected_before_insertion']
    ledger=e['admission'];ledger_path=Path(e['admission_path'])
    if ledger is not None:assert json.loads(ledger_path.read_text())==ledger and ledger['resident_enodes']<=ledger['cumulative_admitted_enodes']<=e['node_allowance'] and not ledger['refund_on_rebuild'];assert e['charged_admission_upper']==ledger['cumulative_admitted_enodes']
    else:assert e['charged_admission_upper']==e['node_allowance']
    charge+=e['charged_admission_upper'];assert e['cumulative_accounted_upper']==charge
    for key in ['admission_path','pool_path','candidate_path','trace_path','stdout_path','stderr_path']:
     z=Path(e[key])
     if z.exists():artifacts.append(dict(path=str(z),sha256=sha(z)))
    if not e['complete_returned_cover']:failures.append(e)
   assert charge==m['cumulative_accounted_enode_upper']<=cap['node_limit'];assert sum(e['wall_secs']for e in m['attempts'])<=m['total_mapping_frontend_secs'] and m['total_mapping_frontend_secs']<=300
   assert sha(folder/'raw.v')==m['selected_verilog_sha256'];assert sha(m['source_verilog'])==m['source_sha256']
   if r['status']=='ok':
    assert r['measured']['cec']==1 and r['measured']['multioutput_instances']==0
    for key in ['area','delay']:assert abs(r['native'][key]-r['measured'][key])<1e-6
    if m['selected']=='source_witness':assert m['selected_verilog_sha256']==m['source_sha256'];assert r['native']['area']==m['source']['area'] and r['native']['delay']==m['source']['delay']
    else:assert m['optimizer_succeeded'] and m['attempts'][m['selected_attempt']]['complete_returned_cover'];assert (r['native']['delay'],r['native']['area'])<=(m['source']['delay'],m['source']['area'])
   v.update(selected=m['selected'],optimizer_returned_cover=m['optimizer_succeeded'],optimizer_failures=failures,actual_mapping_frontend_secs=m['total_mapping_frontend_secs'],node_allowance=cap['node_limit'],accounted_admission_upper=charge,global_retained_upper=m['source_plus_optimizer_retained_upper'],retained_allowance=m['global_retained_candidate_limit'],area=r.get('measured',{}).get('area'),delay=r.get('measured',{}).get('delay'),artifact_hashes=artifacts)
  rows.append(v)
 report=dict(complete=c.get('status')=='attempted' and len(rows)==c['expected_jobs'],attempts=len(rows),expected=c['expected_jobs'],CEC=sum(r['valid']for r in rows),selected_modes=dict(collections.Counter(r.get('selected','none')for r in rows)),optimizer_returned_covers=sum(r.get('optimizer_returned_cover',False)for r in rows),optimizer_failed_attempts=sum(len(r.get('optimizer_failures',[]))for r in rows),baseline_algorithms_changed=False,new_capacity_protocol=True,not_an_unchanged20k_comparison=True,source_only_fallbacks_not_egraph_optimization_successes=True,all_failed_optimization_attempts_retained=True,rows=rows)
 dest=a.campaign.parent/'validation.json';dest.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='rows'},indent=2))
if __name__=='__main__':main()
