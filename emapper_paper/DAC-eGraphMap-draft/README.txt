Timeout-stage audit: results_v2/time_only_600_20261022/timeout_analysis_replay/report.md (7cuts-unfinished,2recovery-unfinished; no fix/PPA claim).
CURRENT TOTAL-MAPPING600 PROTOCOL (2026-10-22)
Mapping-related rewriting, candidate generation, extraction, SA and LUT/cell mapping share600seconds; nested callbacks use interval union. Excludes once-resyn2, pure conversion, independentCEC and metrics. CEC is a verdict, without a runtime column.
Baseline algorithms, rules, models and objectives are unchanged. Extra adapter2s/20k/16 limits forEqMap/E-Syn are removed through existing resource interfaces. Author-coded method schedules remain intact; this is budget instrumentation, not baseline optimization.
Canonical baseline runner: scripts/experiments/campaigns/run_baseline_total600.py
Canonical checkpoint: results_v2/time_only_600_20261022/total600_original_baselines_epfl/campaign.json
Published interim data: total600_progress36_scope (36/120 groups,102qualified slots,2failures; incomplete).
Candidate20/20:11native optimized covers,9source-only timeouts,20CEC; log2/hyp not solved. No main-kernel merge or full-suite win.
Regenerate: write_total600_diagnosis_paper.py --report REPORT_DIR. Published PDF5body+1references, visualQA in paper_timeout_analysis36_publish/qa.json.
All previous policies, observations, partials and failures remain archived below and in raw study directories.

ARCHIVED PREVIOUS NOTES FOLLOW:
CURRENT600-SECOND STUDY (2026-10-22; BASELINE RETESTS IN PROGRESS)
Primary table:47-record inventory(45physical+2constants), uniform DA600-second candidate. All20 attempted;11 complete optimized covers,9 source-only timeouts;20 independent final-output CEC. Baseline69/120 completed groups explicitly reused,182 qualified observations;51 incomplete groups are being retested without algorithm/model/native-parameter changes.
Current data:all_results_time600.csv; results_v2/time_only_600_20261022/candidate_full20_baseline_preflight/.
Candidate hashes/clock/caps/CEC audit:full20_candidate_audit.json. Real EqMap nested-ABC and70-call native-SA clock/CEC probes:real_baseline_meter_probes/validation.json; probes do not replace primary observations.
Baseline canonical runner:run_baselines600.py, original_baselines_epfl/campaign.json. The two older initialization-only directories contain no new mappings and must not be counted again.
Regenerate with:scripts/experiments/campaigns/write_time600_paper.py --report REPORT_DIR; old refresh scripts/all_results.csv remain historical.
Reviewed current PDF:5body+1references, within6+1maximum. Full baseline comparison/repeats are incomplete; no research-kernel merge or full-suite dominance claim.

ARCHIVED NOTES FOLLOW:
2026-10-20 wider-inventory follow-up (not yet complete):
The current primary tables remain the earlier proportional-capacity K3/K4
study. A new K6 experiment admits40/47 single-output cells; seven7--9 input
cells are unsupported by the current cut representation. K3/K4 same-final-
binary sensitivity experiments, full K6 comparison and repeats are pending.
The discussion now records only verified initial sqrt and fixed-topology
pin-symmetry diagnostics. It does not relabel old tables as K6, change author
baseline algorithms, promote a native default, or claim full-suite dominance.
Evidence: results_v2/single_output_k6_20261020/paper_progress/qa.json.

CURRENT EPFL STUDY (2026-10-20)
The current tables use the audited proportional-capacity study in
results_v2/proportional_sweep_20261019/qualification. One global rho=1/2
is used for all20 circuits and both libraries. All120 primary cases are
verified:38 optimized covers and2 basic source covers per ratio. The72
targeted repeats are separate and never replace first-sweep points.

Refresh current tables/body only after restoring audited raw archives:
  python3 results_v2/proportional_sweep_20261019/refresh_paper.py
Compile from the repository root:
  latexmk -cd -pdf -interaction=nonstopmode -halt-on-error epmapping_paper/DAC-eGraphMap-draft/main.tex
The PDF must pass6+1 page and visual QA before copying to output/pdf.
all_results.csv now contains the240 displayed EPFL records, with exact
metric values, mapping-time qualifications and basic-source labels.
The520-slot full original-baseline inventory remains in qualification/
baseline_scope. Historical removals use the old20k resource protocol.
No baseline algorithm or main native default is changed by this study.

ARCHIVED NOTES BELOW: prior refresh_tables.py and old suite instructions
refer to historical ISCAS studies; they do not refresh current EPFL tables.

DAC eGraphMap research working draft (2026-10-07)

main.tex is a standalone, anonymous English manuscript. It follows the
repository's DATE27-ePMapping-sub organization but describes the separate
direct e-graph mapper. No DATE paper experimental numbers are reused.
output/pdf/DAC-eGraphMap-draft.pdf is the compiled review copy.

Refresh tables from saved, CEC-passing reports at repository root:
  python3 epmapping_paper/DAC-eGraphMap-draft/refresh_tables.py

Compile the standalone source twice (use an existing writable build dir):
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory /tmp main.tex
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory /tmp main.tex

Run these compilation commands from this directory. The final PDF is
copied into output/pdf at repository root. The built-in editor compiler
was attempted but its sandbox executable was unavailable in this session;
system pdflatex produced the review copy instead.

table_provenance.json pins the report hashes. all_results.csv includes
all saved comparison methods, not only the fixed columns printed in the
paper. Three- and four-input physical library results remain separate.
The historical standard-cell reports predate the disjoint-support pairing
fix, but those libraries contain no grouped multi-output cells; the fix
does not execute for them. Synthetic tests exercise the updated code.

EqMap reproduction (requires ABC, Yosys, Rust and the common evaluator):
  git clone https://github.com/cornell-zhang/eqmap /tmp/eqmap-baseline
  git -C /tmp/eqmap-baseline checkout d3144c45d3441530cd8d240c5fc6ea6645b8a2fb
  cargo build --release --locked --manifest-path /tmp/eqmap-baseline/Cargo.toml --bin eqmap_fpga
  python3 scripts/experiments/bench_eqmap_baseline.py --eqmap-source /tmp/eqmap-baseline --output /tmp/eqmap-repeat --k 3 --lut-k 6 --benches c17 c432 c499 c880 c1355 c1908 c2670

The pinned dependencies failed to build on Rust 1.88 (experimental if-let
guard in safety-net). Rust 1.99.0 stable successfully built them. The
project's own updated binaries/tests were built with Rust 1.88.0.

EqMap native 6-LUT count/depth is measured independently of standard-cell
area/delay. The main flow uses abc -lut 6, EqMap -k 6, then ABC amap into
the SAME three-input ASAP7 subset. A no-EqMap LUT staging control and a
direct ABC control accompany every design. Both LUT-stage and final
standard-cell equivalence checks must pass. Current EqMap -k below 6
is a finite cost preference, not an enforced upper bound: diagnostic
3-LUT reports emit wider LUTs and are NOT homogeneous 3-LUT results.

Still required before submission:
* Realistic multi-output libraries/benchmarks and a direct emap comparison.
* E-Syn and E-morphic paper-branch baselines with the same final evaluator.
* Larger EPFL/arithmetic suites, timing-target Pareto sweeps and ablations.
* Repeated runtimes, hardware/memory reporting, load-aware timing/physical runs.
* Full novelty review and verification of the intended DAC template/rules.

SDC is implemented; ODC, sequential reachability and external environmental
assumptions are not. Synthetic full-adder, dual-AND and SDC results establish
mechanism correctness, not production multi-output PPA benefits.

Follow-up on 2026-10-12: cumulative boundary unions cover three disjoint
output supports; pair-only and cumulative heuristic/CBC synthetic fixtures
are measured with the common evaluator and independent CEC. The synthetic
area gain comes with a delay increase. Structural fanout seeding remains
optional: only c17 changes to a lower-area/slower point, other designs do
not improve. A bounded c17 timing-constrained integer run is feasible but
not proven optimal. These reports are additional evidence in provenance;
primary comparison tables remain frozen and synthetic libraries are not
aggregated with ASAP7.

User-specified DAC page budget: at most 6 body pages plus 1 separate
references page. References now start after clearpage; current compiled
copy has 6 body pages and 1 reference page. This is a user constraint, not
a claim that the future venue's complete formatting rules were verified.
Two explicit lexicographic objectives and separately generated heuristic
profile results are included. Lexicographic wins are not Pareto dominance.

2026-10-12 fixed-resyn2 completed suite:
Primary tables now use fixed_all_area_20261012 and fixed_4in_area_20261012.
All designs within the defined 7-case/3-case scope have every connected
baseline measured; E-morphic uses final SA, E-Syn remains a bounded adapter,
and EqMap remains a cross-target composition. The complete report inventory
and strict objective/Pareto gates are in results_v2/fixed_complete_20261012.
1,200 successful netlist records include internal candidates and controls.
Failed adapter attempts and native-output-hash-checked repairs are retained.
The current manuscript has six body pages and one separate reference page.
Industrial multi-output libraries, ODC and repeated-run statistics remain open.

2026-10-12 primary baseline update:
Tables I/II compare ABC map -a/map, emap area/delay and both eGraphMap
objectives. The selected-primary gates are in abc_map_primary_20261012;
the original all-configuration gates and all_results.csv remain available.
EqMap and E-Syn compositions retain their actual ABC amap backend.
The manuscript separates experimental protocol, results and limitations;
adapter debugging and submission-planning commentary stays in the audit.
