# Shared incumbent recovery

The typed cone/scored main profile has completed fresh full EPFL and targeted alternating-order validation. Defaults remain unchanged. Full80:64 direct CEC/16 node failures; candidate34/40 covers versus previous protected control30/40. Thirty old-legal pairs have D2better/28equal/0worse and A2better/28equal/0worse. Targeted view36legal pairs has six gains/thirty ties for both metrics, no regressions. Original baselines remain unchanged; all six full-scope baseline gates remain false. See `results_v2/scored_main_integration_20261015/report.md`.

The tested delay-area configuration keeps one imported e-graph and one retained physical pool. A physically ready frontier produces a valid delay seed. The original cold seeds, tie decisions, repair witness and structural-reference setup remain a separate working trajectory, with a soft seed deadline at 25% of the remaining window. A required-time DAG area-flow cover is saved rather than immediately replacing that trajectory. Phase/local/cone and actual-area recovery continue, then complete validated incumbents are compared lexicographically. Invalid or expired alternatives do not replace a valid incumbent. This does not guarantee the result of an independent two-second legacy run under arbitrary system load.

Arrival labels use per-cover epochs and reusable buffers. Every new cover invalidates old labels; physical areas retain ascending instance-ID summation and single-instance charging. This changes evaluation work, not candidate functions, arithmetic, ranking or configured trial limits. Diagnostic cut-name memoization and producer indexing run after extraction and preserve the old diagnostic content/order.

## Configuration

The new options are typed fields in `MapParams`; the CLI exposes matching environment variables:

| Field | Environment | Default |
|---|---|---|
| `fast_delay_seed` | `EGM_FAST_DELAY_SEED` | false |
| `required_area_dag` | `EGM_REQUIRED_AREA_DAG` | false |
| `defer_area_dag` | `EGM_DEFER_AREA_DAG` | false |
| `preserve_legacy_seeds` | `EGM_PRESERVE_LEGACY_SEEDS` | false |
| `cone_import` | `EGM_CONE_IMPORT` | false |
| `scored_incumbent` | `EGM_SCORED_INCUMBENT` | false |
| `metric_epochs` | `EGM_METRIC_EPOCHS` | false |
| `bounded_proposals` | `EGM_BOUNDED_PROPOSALS` | false |
| `seed_window_fraction` | `EGM_SEED_WINDOW_FRACTION` | 1.0 |

The current candidate enables fast_delay_seed, required_area_dag, defer_area_dag, preserve_legacy_seeds, metric_epochs, cone_import and scored_incumbent; bounded_proposals stays false and seed_window_fraction stays1.0. The campaign runner also explicitly sets strict budgets, dependency worklists, the existing cut/cap/canonicalization options, structural references, required-time recovery and phase-quarter scheduling. Enabling only one flag is not the full tested configuration. Existing `legacy` and `queue-delay` presets are unchanged.

Reproduce the exact full configuration and its legacy control with:

```sh
python3 scripts/experiments/campaigns/run_scored_main.py \
  --output results_v2/cone_scored_fresh_main_run --max-jobs 80
```

Use a fresh output directory after changing source/binary/configuration. Recorded jobs are hash-checked and skipped; unrecorded partial directories require an explicit audit. The saved argv, environment and source hashes are the authoritative configuration.

## Budgets and verification

The study uses all twenty EPFL arithmetic/control designs, both restricted ASAP7 inventories, and the same once-resyn2 inputs. Mapping-only allowance is 300 seconds, with the existing 60-second process guard; conversion, CEC and reporting are excluded from the intended mapping allowance. Native process time including diagnostics is retained as a conservative upper bound.

Actual global limits remain 20,000 e-nodes, 16 cuts/class, 100,000 retained physical candidates and 16 saturation iterations. One two-second recovery window includes mandatory setup, all alternatives and final validation. Return reserve is normally 0.05 seconds, increased to 0.15 for compact import under the same deadline. Stage checks are not preemptive; actual overruns fail. Transient generated candidates are recorded separately: retained-pool size is not a RAM cap. Timing/area caches contain labels and instance references, not exempted Boolean nodes.

Independent common physical metrics use all emitted instances and unloaded dependent-pin delay. Direct named-port CEC is retained. A completed map with a resource-only direct-verification timeout may receive an additional original-to-prepared-to-mapped certificate: original/prepared hashes, the saved preparation proof and exact named-port order are checked, then the unchanged public verifier checks prepared versus mapped under its original 60-second guard. Original failures remain immutable. Counterexamples, partial mappings and mapping-budget failures are never converted into success. The same supplementary policy was offered to eligible original baselines; no baseline algorithm/model/parameter is changed.

Remaining div/hyp/mem_ctrl import rejections are still failures; log2 and multiplier now pass under both libraries. No resident/cumulative budget substitution, per-design source/configuration selection, global optimality or all-baseline dominance is claimed.

## Tests

```sh
cargo test --offline --locked -p emapper egraph -- --test-threads=1
cargo test --offline --locked --release -p emapper egraph -- --test-threads=1
```

The existing 2,200-layer overflow stress fixture uses its own enlarged test-thread stack so an unoptimized test build can execute its original graph and assertions. This does not change a CLI mapping stack or any node/time/trial limit. Reports retain the initial debug stack failure and the configuration-wiring red/green test evidence.


## Historical isolated cone-import prototype

The archived version in `results_v2/cone_guard_implementation_20261015` stores exact XOR/three-input truth-table operators and actual helper functions in counted global nodes. It uses one fixed90%initial-capacity rule, helper quota2048, existing node/cut/pool/SAT/recovery limits and the current protected recovery. That prototype used `EGM_XOR_IMPORT`; it was not wired into the then-current main API/CLI. The current main API uses the typed option described below. Defaults remain unchanged. Counter occurrence labels in old prototype logs are not unique-enode counts; actual imported/post-SAT totals are authoritative. This is a changed internal representation, not an unchanged expanded-AND work-unit claim.

The full20/two-library study and seven-target alternating repetitions improve both sqrt metrics and admit log2/multiplier in both libraries, but retain one4in adder area regression. Both strict self-nonregression gates remain false; therefore no production promotion or table replacement. Detailed commands, all outcomes, source hashes, original baseline audit and remaining three prototype import failures are in `results_v2/cone_guard_implementation_20261015/report.md`. At that historical checkpoint main still had five import failures. Further experiments must preserve global accounting and baseline author algorithms, and may not overwrite negative runs or choose winners per design.


## Typed cone import and scored-cover retention (2026-10-15)

`MapParams::cone_import` is now a main API option, exposed as `EGM_CONE_IMPORT`; the old experimental `EGM_XOR_IMPORT` is not the main switch. The fixed campaign activates exact XOR/three-input-table import only above the uniform 90% initial-capacity threshold. Each operator carries its actual children and truth table, all actual helpers are charged (fixed quota 2048), and source-exact physical cuts use the existing cut banks. Schema2 separates source occurrences from unique actual operator/helper classes. This changes the e-node representation unit relative to expanded ANDs; it is not an unchanged expanded-AND work-unit comparison. Original source AND counts remain recorded. Node/cut/pool/SAT/recovery limits are unchanged; there is no per-region reset or hidden Boolean graph.

`MapParams::scored_incumbent` / `EGM_SCORED_INCUMBENT` retains one already fully scored, validated cover from actual-area recovery. The original proposal order and working trajectory continue unchanged; at final comparison the saved cover competes lexicographically, followed by the original deferred incumbent. There are no extra scoring calls or extra physical candidates. This closes the measured score/application deadline gap; it does not promise a delay gain on its own.

The new fixed delay-area profile is reproduced by `python3 scripts/experiments/campaigns/run_scored_main.py --output FRESH_DIRECTORY --max-jobs 80`. It compares the previously protected main profile (cone0/scored0) to exactly the same configuration with cone1/scored1. Full main80: 64 direct original-input CEC successes and16 retained node failures;30 old-legal pairs have D2better/28equal/0worse and A2better/28equal/0worse. Candidate maps34/40, adding log2/multiplier under both inventories; div/hyp/mem_ctrl still fail. See `results_v2/scored_main_full_report_20261015`. Do not count failures as valid PPA or merge prototype observations into main results. Defaults remain off. Repeated validation is complete:64 status slots comprise56 fresh successful native mappings and8 explicitly adopted unchanged control failures. Including32 matching full observations, the96-observation target view has84 direct CEC/12failure slots,36old-legalpairs D6better/30equal/0worse and A6better/30equal/0worse. It is not a repeat of all20 designs. Unique main-native totals are136attempts:120CEC/16node failures; overlaps are not added. See `results_v2/scored_main_repeats_report_20261015`.


## Isolated source-planning experiment (2026-10-16)

The V6 library-aware source planner is **not a main API or preset**. Its provisional `EGM_SHARED_MAJORITY` option exists only in the archived isolated mapper. The main configuration above, its defaults, source and binary are unchanged. V6 retains actual-library-matched scalar TT3/TT4 functions and all operands; global source-reference planning happens before any e-graph construction. No e-graph state is retired to reset cumulative accounting. Incomplete prefixes stop before node20001 and fail. Scalar truth-table e-nodes are not expanded-AND work units.

The complete80 fresh self-comparison has68 direct CEC successes and12 node failures. Both profiles use main cone/scored protection; all34 common legal pairs tie in delay, and only4in multiplier area improves by0.80. Sqrt retained cuts, pool fingerprints and selected physical instances are identical despite extra source expressions. Div/hyp/mem_ctrl still fail; a source-plan estimate is neither a mapped cover nor an impossibility bound. The generic any-metric nonregression gate does not establish repeated qualification, a delay gain or production promotion. Full/repeated evidence and the explicit decision belong in `results_v2/library_source_plan_evidence_20261016`; no per-circuit winner mixture is allowed.


V6 final decision: two eight-target alternating repetitions64fresh (63CEC,one control2.004620s recovery failure), no delay gain. The canonical-main comparison additionally regresses4in priority area52.15->52.42. Thus isolated same-binary nonregression is insufficient and V6 is not promoted. Qualification/report and complete failure records are in library_source_plan_evidence_20261016; current main API/configuration above remains authoritative.


## Isolated joint-witness retention experiment (2026-10-16)

Main APIs, defaults and binary remain unchanged. The isolated `/tmp/emapper-joint-batch-20261016` prototype adds typed `timing_witness_pool`, `online_area`, `activate_deferred` and `joint_cones`, all default false. These are **not production environment switches**. Formal V4 enables the first three and leaves `joint_cones` false; both profiles retain cone/scored incumbent options. Failed atomic cone-repair probes and earlier pilot versions remain separate artifacts.

Before the existing optional 100000-instance cap, the prototype reserves the union of an area-ranked finite witness and a fast all-signal physical witness, using existing instance IDs. It preserves a complete old incumbent; preserving the old candidate witness alone does not guarantee final area nonregression when a strictly faster cover wins the delay-first objective. Bounded scan batches apply already validated scored covers, and later required-time flow uses 30% of its remaining late window, leaving the rest for actual-area recovery. The saved legacy shared-area cover **within the current retained pool** is retained for final comparison; pool changes do not preserve every binding of the previous pipeline's actual final cover. a faster but larger cover is still lexicographically preferred. Therefore this experiment does not satisfy an additional no-area-regression requirement.

Fast pre-cap witness ranking produces no returned incumbent and is candidate-generation work inside the original native60s/mapping300s guards. The extractor recomputes its own witnesses inside the same global2s inclusive recovery window. No second recovery allowance, larger pool, increased cut count or equal executed-work claim is made. Frontier label audits use local inequality tolerance1e-6; a conservative path error is2*eclasses*1e-6. Their scope is the supplied frozen physical pool and unloaded dependent-pin delay, never global library/cut, area or lexicographic optimality.

The bounded-cone importer now rejects before adding a new node beyond20000, counting every input and demanded output phase. Incomplete prefixes are explicit failures and not legal mapped coverage. Div, hyp and mem_ctrl remain unresolved in both inventories. Source/binary/tests, pilot failures and full matrix are independently archived under `results_v2/joint_batch_implementation_20261016` and `joint_batch_full_20261016`. Main promotion remains blocked by observed area regressions, including priority.
