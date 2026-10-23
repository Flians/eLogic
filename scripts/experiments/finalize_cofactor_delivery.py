#!/usr/bin/env python3
"""Snapshot an audited, visually approved 6+1 paper; never approve QA itself."""
import argparse, datetime, hashlib, json, shutil, subprocess, sys
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve();project=Path(__file__).resolve().parents[2]
    assert json.loads((root/'execution_complete.json').read_text())['complete']
    q=json.loads((root/'build_ready.json').read_text());pdf=Path(q['pdf'])
    assert q['pages']==7 and q['body_pages']==6 and q['reference_pages']==1
    assert q['visual_QA_pending'] is False and q['visually_checked_pages']==list(range(1,8))
    assert q['sha256']==sha(pdf),'PDF changed since visual inspection'
    assert q['overfull_boxes'] is False and q['unresolved_references'] is False
    verified=json.loads((root/'verified_profiles.json').read_text())
    for mode,h in verified['cohort_certificate_sha256'].items():
        cert=root/mode/'verified_metrics.json';assert sha(cert)==h
        for rel,hh in json.loads(cert.read_text())['raw_sha256'].items():assert sha(root/mode/rel)==hh,(mode,rel)
    for rel,h in verified['raw_sha256'].items():assert sha(root/rel)==h,rel
    subprocess.run([sys.executable,str(project/'scripts/experiments/summarize_cofactor_campaign.py'),'--root',str(root)],check=True)
    source=pdf.parent;snapshot=root/'paper_snapshot';snapshot.mkdir(exist_ok=True)
    for pattern in ['*.tex','*.bbl','*.cls','Makefile','doc/*.tex','figs/*.pdf','figs/*.png','figs/results/*','ref/*.bib','scripts/*.py']:
        for p in source.glob(pattern):
            if not p.is_file():continue
            target=snapshot/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    shutil.copy2(pdf,snapshot/pdf.name)
    scripts=['common.sh','run_three_parts.py','report.py','report_three_parts.py','report_three_parts_profiles.py','timeout_selection.py','audit_mapped_ppa.py','export_paper.py','plot_paper.py','plot_profile_cache.py','plot_cofactor_ablation.py','summarize_cofactor_campaign.py','plot_build_ablation.py','recover_checkpoint.py','run_three_parts_paired.py','reproduce_paper.sh','finalize_build_campaign.py','finalize_cofactor_delivery.py']
    publication=root/'publication_sources';publication.mkdir(exist_ok=True)
    for name in scripts:shutil.copy2(project/'scripts/experiments'/name,publication/name)
    for name in ['run_campaign.py','watch_validation.py','archive_history.py','deduplicate_storage.py','finish_when_ready.py']:
        if (root/name).exists():shutil.copy2(root/name,publication/name)
    pilot=project/'research/build_reuse_20261016' if 'noAIGREUSE' in json.loads((root/'cold/manifest.json').read_text())['arms'] else project/'research/usb_candidate_generation_20261015'
    validation={str(p.relative_to(project)):sha(p) for p in pilot.rglob('*') if p.is_file() and p.name in ['verified_pilot.json','completion_manifest.json','RESULTS.txt','validated_env.json','verified_results.json','delivery_manifest.json','RESULTS.md','ANALYSIS.md','pilot_env.json','final_validation.json']}
    (root/'validation_sources.json').write_text(json.dumps(dict(pilot_root=str(pilot),files=validation,comment='Pre-campaign validation only; campaign results use the frozen binary in each cohort/inputs.'),indent=2)+'\n')
    delivered=project/'output/pdf/DATE27-ePMapping.pdf';delivered.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(pdf,delivered)
    q.update(delivered_pdf=str(delivered));(root/'build_ready.json').write_text(json.dumps(q,indent=2)+'\n')
    artifact_names=['README.txt','RESULTS.txt','execution_complete.json','execution_plan.json','setup_audit.json','reference_rounding_audit.json','verified_profiles.json','cold/verified_metrics.json','warm/verified_metrics.json','paired_results.tsv','construction_counters.tsv','construction_comparisons.json','generated_profiles.tex','generated_mfs_profiles.tex','build_ready.json','validation_sources.json']
    if (root/'storage_deduplication.json').exists():artifact_names.append('storage_deduplication.json')
    artifacts={name:sha(root/name) for name in artifact_names}
    for folder in ['paper_snapshot','publication_sources','reference_sources','initial_validation']:
        for p in (root/folder).rglob('*'):
            if p.is_file():artifacts[str(p.relative_to(root))]=sha(p)
    for name in ['paper_build.log','paper_build_retry.log','paper_build_final.log','final_layout.log']:
        if (root/name).exists():artifacts[name]=sha(root/name)
    manifest=dict(created_at=datetime.datetime.now().astimezone().isoformat(),delivered_pdf=str(delivered),delivered_pdf_sha256=sha(delivered),snapshot_sha256=artifacts,quality_checks=q,raw_data='verified_profiles.json -> cohort certificates -> immutable per-run raw evidence; separate timeout-selection proof hashes are validated by the cohort audits')
    (root/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Delivered',delivered,sha(delivered))
if __name__=='__main__':main()
