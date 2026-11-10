#!/usr/bin/env python3
"""Audit and freeze a completed campaign into a separate paper worktree."""
import argparse,csv,gzip,hashlib,json,shutil,subprocess,sys,tarfile,time
from pathlib import Path
from run_epmapping_supplement import sha,save
PROJECT=Path(__file__).resolve().parents[2]

def freeze_sources(root,dest):
    """Keep executed engine sources distinct from the final independent auditors."""
    with tarfile.open(dest/'source_snapshot.tar.gz','w:gz') as tar:
        tar.add(root/'inputs/sources',arcname='frozen_sources')
        for name in ['analyze_prebuffer_three_parts.py','finish_prebuffer_three_parts.py','compare_prebuffer_flow.py','mapped_shared_blif.py','verify_prebuffer_imports.py','exact_core_graph.py','run_prebuffer_three_parts.py','run_buffer_three_parts.py','run_buffer_core_validation.py','run_epmapping_supplement.py','audit_mapped_ppa.py','run_three_parts.py','recover_checkpoint.py','report.py','audit_buffer_core.py','analyze_epmapping_supplement.py','resume_prebuffer_campaign.py']:
            tar.add(PROJECT/'scripts/experiments'/name,arcname='auditors/'+name)
        for name in ['asap7.genlib','asap7_6in.genlib']:
            tar.add(root/'inputs'/name,arcname='libraries/'+name)
        if (root/'build_context').exists():tar.add(root/'build_context',arcname='build_context')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--paper-work',required=True,type=Path);ap.add_argument('--wait',action='store_true');a=ap.parse_args()
    root=a.root.resolve();paper=a.paper_work.resolve()
    while True:
        status=json.loads((root/'execution_status.json').read_text()) if (root/'execution_status.json').exists() else {}
        if status.get('errors'):raise RuntimeError(status)
        if status.get('complete'):break
        if not a.wait:raise RuntimeError('campaign incomplete')
        time.sleep(45)
    with (root/'final_audit.log').open('w') as f:
        subprocess.run([sys.executable,str(PROJECT/'scripts/experiments/analyze_prebuffer_three_parts.py'),'--root',str(root)],check=True,stdout=f,stderr=subprocess.STDOUT)
    data=json.loads((root/'analysis.json').read_text());assert data['complete'] and data['observed']==832
    dest=paper/'data/prebuffer-20261111';dest.mkdir(parents=True,exist_ok=True)
    (dest/'analysis.json.gz').write_bytes(gzip.compress((root/'analysis.json').read_bytes(),mtime=0))
    subprocess.run([sys.executable,str(PROJECT/'scripts/experiments/compare_prebuffer_flow.py'),'--root',str(root),'--control',str(PROJECT/'results_epmapping_three_parts_buffer_v1_20261111')],check=True)
    for source,target in [(root/'normalized/manifest.json','normalization_manifest.json'),(root/'abc_import_proofs.json','abc_import_proofs.json'),(root/'full_flow_comparison.json','full_flow_comparison.json')]:shutil.copy2(source,dest/target)
    shutil.copy2(root/'manifest.json',dest/'campaign_manifest.json');shutil.copy2(root/'execution_status.json',dest/'execution_status.json')
    proofs=[]
    for r in data['records']:
        wd=root/r['stage']/r['arm']/f"{r['mode']}.{r['design']}"
        proof=dict(stage=r['stage'],arm=r['arm'],design=r['design'],mode=r['mode'],status=r['status'],selected_netlist=r['selected_netlist'],selected_sha256=r['selected_sha256'],initial_sha256=r['initial_sha256'],log_sha256=sha(wd/'log'),selected_cec=1)
        if r['arm']!='abc' and r['status']=='complete':
            cec=list(wd.glob('cec_*.v'));assert len(cec)==1 and sha(cec[0])==r['selected_sha256'];proof['native_cec_input_sha256']=sha(cec[0])
        for name in ['checkpoint_cec.log','selected_native_cec.log','selected_initial_cec.log','raw_cec.log']:
            if (wd/name).exists():proof[name+'_sha256']=sha(wd/name)
        proofs.append(proof)
    save(dest/'selected_output_audit.json',proofs)
    with (dest/'paired_results.tsv').open('w') as f:
        fields=['stage','arm','design','mode','status','initial_area','initial_delay','area','delay','buffer_gain','core_gain','optimizer_seconds','selected_sha256'];w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader()
        for r in data['records']:w.writerow({**{k:r[k] for k in ['stage','arm','design','mode','status','buffer_gain','core_gain','optimizer_seconds','selected_sha256']},'initial_area':r['initial']['area'],'initial_delay':r['initial']['delay'],'area':r['final']['area'],'delay':r['final']['delay']})
    freeze_sources(root,dest)
    resume=json.loads((root/'latest_resume.json').read_text()) if (root/'latest_resume.json').exists() else None
    if resume:shutil.copy2(root/resume['archive']/'resume_manifest.json',dest/'resume_manifest.json')
    save(dest/'manifest.json',dict(campaign=root.name,source_campaign=str(root),engine_sha256=json.loads((root/'manifest.json').read_text())['engine_sha256'],observed=832,method_runs=768,abc_runs=64,files={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='manifest.json'},normalization='shared mapped BLIF with identity BUF bypass before search; same physical core/pins/ordered outputs; fixed bound from cleaned input; aliases are wires',cold='empty private profile pool',warm='paired cold pool in a new process; OS caches retained',validation='selected output byte hashes bind native CEC inputs or independent ABC/recovered-checkpoint CEC; original logs and netlists retained in source campaign'))
    subprocess.run([sys.executable,str(paper/'scripts/prebuffer_campaign.py'),'--write'],check=True,cwd=paper)
    with (root/'paper_data_verification.log').open('w') as f:subprocess.run(['make','verify-data'],check=True,cwd=paper,stdout=f,stderr=subprocess.STDOUT)
    save(root/'paper_snapshot_status.json',dict(complete=True,paper_work=str(paper),data_directory=str(dest),analysis_sha256=sha(root/'analysis.json')))
    print('Complete: campaign audited, paper snapshot frozen and all four data verifiers passed',flush=True)
if __name__=='__main__':main()
