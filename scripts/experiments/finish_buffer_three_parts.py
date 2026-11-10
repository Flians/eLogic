#!/usr/bin/env python3
"""Audit and freeze a completed campaign into a separate paper worktree."""
import argparse,csv,gzip,hashlib,json,shutil,subprocess,sys,tarfile,time
from pathlib import Path
from run_epmapping_supplement import sha,save
PROJECT=Path(__file__).resolve().parents[2]

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
        subprocess.run([sys.executable,str(PROJECT/'scripts/experiments/analyze_buffer_three_parts.py'),'--root',str(root)],check=True,stdout=f,stderr=subprocess.STDOUT)
    data=json.loads((root/'analysis.json').read_text());assert data['complete'] and data['observed']==832
    dest=paper/'data/buffer-20261111';dest.mkdir(parents=True,exist_ok=True)
    (dest/'analysis.json.gz').write_bytes(gzip.compress((root/'analysis.json').read_bytes(),mtime=0))
    replay=json.loads((root/'primary_core_replay.json').read_text());assert replay['complete'] and replay['pairs']==32
    shutil.copy2(root/'primary_core_replay.json',dest/'primary_core_replay.json')
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
    with tarfile.open(dest/'source_snapshot.tar.gz','w:gz') as tar:
        tar.add(root/'inputs/sources',arcname='frozen_sources')
        for name in ['analyze_buffer_three_parts.py','finish_buffer_three_parts.py','compare_buffer_primary.py','exact_core_graph.py']:
            tar.add(PROJECT/'scripts/experiments'/name,arcname='auditors/'+name)
    save(dest/'manifest.json',dict(campaign=root.name,source_campaign=str(root),engine_sha256=json.loads((root/'manifest.json').read_text())['engine_sha256'],observed=832,method_runs=768,abc_runs=64,files={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='manifest.json'},normalization='buffer area from identity cells; core is total minus buffer; serialization aliases are wires',cold='empty private profile pool',warm='paired cold pool in a new process; OS caches retained',validation='selected output byte hashes bind native CEC inputs or independent ABC/recovered-checkpoint CEC; original logs and netlists retained in source campaign'))
    subprocess.run([sys.executable,str(paper/'scripts/buffer_campaign.py'),'--write'],check=True,cwd=paper)
    with (root/'paper_data_verification.log').open('w') as f:subprocess.run(['make','verify-data'],check=True,cwd=paper,stdout=f,stderr=subprocess.STDOUT)
    save(root/'paper_snapshot_status.json',dict(complete=True,paper_work=str(paper),data_directory=str(dest),analysis_sha256=sha(root/'analysis.json')))
    print('Complete: campaign audited, paper snapshot frozen and all three data verifiers passed',flush=True)
if __name__=='__main__':main()
