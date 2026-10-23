# Current verified evidence — 2026-10-23

## eGraphMap technology mapping

- EPFL arithmetic/control20; one shared, CEC-verified `resyn2` input per design.
- One600-second total mapping-related search window; independent CEC, shared
  preparation, pure conversion and reporting excluded. Old300s/20knode/2s
  cohorts remain historical.
- V9 isolated complete-score/provenance prefilter: one binary, period1 in both
  profiles; each profile has3/20 designs,6CEC outputs across3 designs in total.
  Full20 evaluation and repetitions are incomplete. This is not merged into
  the main mapper and does not implement general lazy physical materialization.

| Design | Control A | Control D | V9 A | V9 D |
| --- | ---: | ---: | ---: | ---: |
| adder | 87.10 | 2569.15 | 87.10 | 2569.15 |
| div | 9598.12 | 43058.72 | 9598.32 | 43058.72 |
| log2 | 6349.19 | 4009.14 | 6324.35 | 3989.77 |

Div has an equal-delay area regression. Log2 improves in this single pair but
remains behind the stronger separate historical D3852.82 reference and same-library
emap D3864.80. Raw600s timeouts and predeadline validated optimized incumbents
are retained; a CEC-passing incumbent does not mean the search completed.
All full-scope `(D,A)`, `(A,D)` and Pareto gates are false.

Records: `results_v2/cooperative_mapping_20261022/v9_progress/`,
`v9_control/campaign.json`, `v9_epfl/campaign.json` and the explicit target
adoption records. Do not select different versions' winners per design.

Baseline snapshot:43/120 groups,106qualified observations,11failed and143pending
method slots. ABC map and independent emap cover all20; EqMap/E-Syn/E-morphic
are not complete, and MtM is unconnected. New E-Syn/div failed in the input
adapter's1MB expression guard before author search; it is not a600s native
timeout. Historical E-Syn native sources also contain disclosed budget/model-path
patches. They are not relabelled as clean upstream runs. Sources are now
registered under [`tools/`](../tools/BASELINES.md), named E-morphic locally.

## ePMapping post-mapping optimization

This is a separate project and objective: minimize area under the initial mapped
delay bound. Its current paper evidence is
`results_v28_three_parts_query_budget/`, with a1000-second post-mapping budget.
`RESULTS.txt`, `paired_results.tsv`, `verified_profiles.json` and
`delivery_manifest.json` preserve outcome and timeout-selection distinctions.
Cold/warm352method attempts each have348/349completed optimizer runs; all352
selected outputs in each profile have CEC and initial-bound evidence. These
figures must not be used as eGraphMap technology-mapping results.

Paper sources and released PDFs are under `emapper_paper/`, `epmapping_paper/` and `output/pdf/`.
Publication organization does not change their algorithms or experimental claims.
