"""Freeze an audited, visually checked delivery and enumerate its local Git scope."""
import hashlib,json,shutil
from pathlib import Path
project=Path(__file__).resolve().parents[2]
root=project/'results_v28_three_parts_query_budget'
research=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ready=json.loads((root/'build_ready.json').read_text())
assert ready['pages']==7 and ready['body_pages']==6 and ready['reference_pages']==1
assert not any(ready[k] for k in ['visual_QA_pending','overfull_boxes','unresolved_references'])
assert ready['visually_checked_pages']==list(range(1,8))
pdf=Path(ready['pdf']);assert sha(pdf)==ready['sha256']
complete=json.loads((root/'execution_complete.json').read_text());assert complete['complete'] and complete['method_runs']==704
assert complete['verified_profiles_sha256']==sha(root/'verified_profiles.json')
analysis=json.loads((root/'query_budget_analysis.json').read_text())
assert all(m['runs']==64 and m['completed']==64 and m['timeouts']==0 for m in analysis['primary_completion'].values())
verified=json.loads((root/'verified_profiles.json').read_text())
metrics=verified['cohort_metrics']
readme=project/'epmapping/README.md'
text=readme.read_text();prefix=text[:text.index('The separate validation covers 16 circuits')]
pc,pw=analysis['primary_completion']['cold'],analysis['primary_completion']['warm']
paired=analysis['paired']['cold']
text=prefix+f'''The current full rerun is `results_v28_three_parts_query_budget`: sixteen circuits,
all three experiment parts, nineteen ablation arms, and separate cold/warm pools.
Its 128 complete-method runs (direct, cap 1/cap 6, and independent ablation full)
all finish normally; maximum optimizer time is {pc['max_seconds']:.2f} seconds cold
and {pw['max_seconds']:.2f} seconds warm. All reported outputs pass CEC and
independent serialized-netlist area/delay checks. The complete-method main
experiment has mean area changes of {metrics['cold']['main_own_area']:+.2f}% cold
and {metrics['warm']['main_own_area']:+.2f}% warm relative to its original mapping.

ABC completes {analysis['mfs3_completed']} matched-start cases and times out on
{analysis['mfs3_timeout']}. On the completed subset, our mean raw-output area ratio
to mfs3 is {paired['mean_area_ratio_to_raw_mfs3']:.6f}
({paired['area_wins']} area wins, {paired['area_losses']} losses). This is an
aggregate area advantage, not per-circuit dominance or a universal speedup.

The controls are `noQUERY` (inverse-query memo and level shortcut off), `noTRUTH`
(decoded simulation/dynamic-table reuse off; common scalar cut memo retained),
and `allPIVOTS` (all pivots). Weakening the configuration can cause timeouts;
their selected outputs remain marked incomplete. See
`results_v28_three_parts_query_budget/QUERY_BUDGET_ANALYSIS.txt` and
`paired_results.tsv` for separate cold/warm PPA, time and timeout counts.
The older 64-run runtime pilot and v27 campaign remain historical measurements;
they do not supply numbers to the updated paper.
'''
readme.write_text(text)
original=project/'results_v20_three_parts_chain'
designs=json.loads((root/'execution_plan.json').read_text())['circuits']
assert designs==16
names=json.loads((root/'cold/manifest.json').read_text())['designs']
reference=['results_v20_three_parts_chain/inputs']
for d in names:
 for part,arm,file in [('tab_mfs3','base','epmapping_mapped.v'),('tab_mfs3','ours','epmapping_baseline.v'),('tab_main','ae','epmapping_baseline.v')]:
  reference.append(str((original/part/f'{arm}.{d}'/file).relative_to(project)))
refhash={str(p.relative_to(project)):sha(p) for rel in reference for p in ([project/rel] if (project/rel).is_file() else (project/rel).rglob('*')) if p.is_file()}
(research/'original_inputs_verified.json').write_text(json.dumps(refhash,indent=2)+'\n')
out=project/'output/pdf/DATE27-ePMapping.pdf';out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(pdf,out)
scripts='audit_mapped_ppa check_mapping_dominance recover_checkpoint timeout_selection run_three_parts run_three_parts_paired report_three_parts report_three_parts_profiles plot_paper plot_cofactor_ablation plot_profile_cache plot_build_ablation summarize_cofactor_campaign finalize_build_campaign'.split()
paths=['epmapping','map_egg','patches/mockturtle','scripts/apply_mockturtle_patches.sh','scripts/compare_with_aspdac25.sh','tools/mockturtle',
       'scripts/experiments/common.sh','scripts/experiments/export_paper.py','scripts/experiments/record_manifest.py','scripts/experiments/report.py','scripts/experiments/reproduce_paper.sh','scripts/experiments/run_all.sh','scripts/experiments/selftest.sh',
       *[f'scripts/experiments/{s}.py' for s in scripts],
       'scripts/experiments/report_objectives.py',
       'scripts/experiments/tests',
       'results_v28_three_parts_query_budget','research/paper_v28_20261016','output/pdf/DATE27-ePMapping.pdf',*reference]
# Tests outside this campaign are small, but keep unrelated experiment drivers out.
for rel in paths:assert (project/rel).exists(),rel
(research/'commit_paths.json').write_text(json.dumps(paths,indent=2)+'\n')
proofs=['execution_plan.json','execution_complete.json','verified_profiles.json','paired_results.tsv','query_budget_analysis.json','QUERY_BUDGET_ANALYSIS.txt','RESULTS.txt','construction_comparisons.json','build_ready.json']
manifest=dict(campaign=str(root.relative_to(project)),paper_repository='epmapping_paper',paper_commit=None,pdf=str(out.relative_to(project)),pdf_sha256=sha(out),
 constraints=dict(original_inputs_unchanged=True,delay_bound='fixed initial mapping, 0.001 ps native comparison tolerance',whole_window_synthesis=False,post_mapping_seconds=1000),
 result_sha256={r:sha(root/r) for r in proofs},original_reference_manifest_sha256=sha(research/'original_inputs_verified.json'),
 native_source_certificate_sha256=sha(research/'native_sources_verified.json'),primary_completion=analysis['primary_completion'],
 baseline_completion=dict(completed=analysis['mfs3_completed'],timeouts=analysis['mfs3_timeout']),
 local_commit_only=True,pushed=False)
(root/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Delivery artifacts prepared; record paper commit before committing root.')
