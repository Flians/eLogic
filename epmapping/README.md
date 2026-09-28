# epmapping

`epmapping` is the eLogic-style (`emig/main.cpp`) post-mapping rewrite driver for a
**gate-level** netlist. It takes the mapped `cell_view<block_network>` produced by
mockturtle `emap`, walks every gate and every cut of that gate, asks the cell library /
`map_egg` for a better sub-graph, and substitutes the winner back into the netlist.

```
AIGER ──ABC (EPMAPPING_BASELINE)──▶ AIG ──mockturtle emap──▶ cell_view<block_network>
         direct = none                       (area-oriented by default)
         high_effort = {fraig; resyn2rs; fraig; compress2rs} to a fixpoint (default)
         custom = one pass of EPMAPPING_ABC_SCRIPT
                                                                    │
                                          epmapping::optimizer (this directory)
                                                                    │
                    for every gate (reverse topological order), for every cut of that gate:
                              ┌─────────────────────────────────────┴─────────────────────────┐
                              │ 1. don't cares    window simulation (`per_node` by default)    │
                              │ 2. single cell    library N/P supergate index                  │
                              │ 3. multi cell     pre-mapped 4-input sub-networks              │
                              │ 4. structural synthesis  SOP factoring / DSD (+ exact, MFFC)   │
                              │ 5. e-graph rewrite  MFFC cut, only if 2-4 found nothing        │
                              └─────────────────────────────────────┬─────────────────────────┘
                                                                    │  best cover that wins the
                                                                    │  area/delay acceptance gate
                                                                    ▼
                        substitute_node ──▶ MFFW window fallback (roots it could not improve)
                                                                    │
                                              write_verilog_with_cell + ABC CEC vs the AIGER
```

> **Note.** This engine now has a single flow: the per-cut **staged** decision below
> (SAT don't cares, then single-cell / four-input multi-cell match, then the optional structural
> synthesis, then the MFFC-gated e-graph stage, with the MFFW whole-window synthesis as a fallback).
> Older tables in this document that compare against a `legacy` candidate set, dependency cuts, the
> cone mapper, ODC/SDC probes or MFFW rewriting are **historical measurements** — that code has been
> removed, and so have the corresponding `EPMAPPING_*` knobs.

## Driver (`epmapping.hpp`, `main.cpp`)

* Cut enumeration is a **live** database (`dynamic_network_cuts` plus its lazy manager, the pair
  `mockturtle::rewrite` uses) over `cut_size`/`cut_limit`, together with a hand-maintained fanout
  list (`block_network` has no fanout iteration).
* Gates are processed in **reverse topological order**, and every substitution keeps the round
  consistent: the fanout lists are updated, `levels_`/`arrivals_` are re-propagated over the
  replacement's fanout cone (`rewrite.hpp`'s `update_levels`), and the cuts of the nodes this round
  has not handled yet in that cone are cleared so that they are recomputed from the current fanins
  (`clear_cuts_fanout_rec`). If a cut's cone no longer terminates at its leaves, that root's whole
  fanin cone is rebuilt and the root retried once, so a committed window can no longer invalidate
  the remaining cuts of the round.
* For each cut a compact **gate-level expression** is built from the cone, using standard-cell
  names as operators, cut leaves as `a`..`f`, constants as `0`/`1` and complemented edges as
  `(~ ...)`.
* The competing covers (single cell, pre-mapped multi-cell, structural synthesis, e-graph, MFFW
  window) are rebuilt in the network (`create_node` + `add_cell`, inserting explicit inverter cells
  for `neg_mask`/`out_neg`) and installed with `substitute_node`.
* A candidate is accepted only when it improves the current objective (area-first by default) and
  does not exceed the original gate's structural level (`preserve_depth`) or library-timing delay
  by more than `delay_slack` (`preserve_delay`).
* `main.cpp` sets `MAP_EGG_TECHLIB_PATH` to the genlib passed on the command line, writes the
  optimized netlist with `write_verilog_with_cell`, and verifies it with
  `experiments::abc_cec_mapped_cell_impl` against the original input AIGER.

### Consistency fixes (2026-09-22)

* The e-graph cover's delay was computed with the library gate's **pin index** while the signal is
  wired to `pin_names[k]`. A cover from `emap_egg` carries a permuted `CellVariant` (which the
  extraction picks exactly when the pin delays differ), so the model charged the delay of another
  pin: measured on `c7552`, 40 of 107 626 cover pins mismatched by up to ~2 library time units,
  above the `EPMAPPING_NO_DELAY_REGRESSION` tolerance. `finalize_local_cover` and `cover_delay` now
  resolve the pin by name.
* A committed MFFW window rewrites nodes that cuts of not-yet-processed gates still depend on, which
  used to leave those cuts unmappable for the rest of the round (`c7552`, one round:
  `EPMAPPING_WINDOW_SYN=1` reported `rejected=300`, and only 2 of 430 windows could be applied; with
  the window off `rejected=0`). The round now keeps its cut database, fanout lists and timing live,
  so `rejected=0` and 33 windows commit on the same run (`86.87` area instead of `87.66`).
* `build_window_cover` failing mid-build left its fresh nodes live but unreachable, which
  `compute_area()` then charged for the rest of the round; the failure path rolls them back.
* `cover_matches_tt` returned `true` for cuts wider than six leaves, so `EPMAPPING_CUT_SIZE=7..9`
  skipped the functional validation entirely; wide cuts are now compared against their exact dynamic
  truth table.
* `EPMAPPING_DC_MAX_PIS` is clamped to 12 (it is shifted into a `uint32_t`), `EPMAPPING_SINGLE_ENTRIES`
  is honoured on completely specified cuts too, and the `dc_cuts` counter and the `dc_window` timer
  are no longer dead under the default `per_node` window.
* `mffc_area` is an O(V+E) peel instead of an O(|cone|^2) fixpoint, and a cut's cone truth table is
  computed once per cut rather than by every source.

## map_egg

`map_egg` is the engine's Rust half and the project's only `#[cxx::bridge]`. It pre-loads the
standard-cell `.genlib` (`TechLib`) and answers the C++ cut loop with costed covers:

1. pre-load the standard-cell `.genlib` (`TechLib`),
2. build one-way **AIG → gate-level-expression** rules for every logic gate,
3. turn the input **gate-level sub-graph** into AIG,
4. run AIG rewriting, and
5. extract the gate-level expression.

The algorithms live in this crate. They were a separate `emapper` crate until they were folded in:

| module | contents |
| --- | --- |
| `emap_cover.rs` | `map_cut_to_cover_cover_only`, `map_cut_to_cover_alternatives`, `map_tt_to_library`, `MappedCover{Choice,Node,Graph}` |
| `emap_egg.rs` | `gate_expr_to_aig_sexp`, `sexp_to_tt9`, `EggRewriteBundle`, `GateEGraph`, `AIG` |
| `techlib.rs` | `TechLib`, `Cell`/`CellVariant`, `MappingObjective`, `MappingAreaCost` |
| `emap_cut.rs`, `tt9.rs`, `aig_graph.rs` | cut structure, truth table, AIG graph |
| `binding.rs`, `params.rs` | `NodeBinding`/`NodeBindingKind`, `EmapParams` |

The manifest declares exactly the eight crates those modules use (`cxx`, `egg`, `serde`,
`serde_json`, `aiger`, `itertools`, `petgraph`, `thiserror`). The upstream crate's ILP extractors,
Python bindings, serialization crates, logging helpers and its own mapper are gone.

How a cut is answered:

* `emap_egg::gate_expr_to_aig_sexp` expands a cell-name gate-level s-expression to AIG using each
  cell's `Expr` (genlib pre-load), so a cut sub-graph never has to be expanded as a minterm SOP.
* `EggRewriteBundle::new_cover_only` builds the **one-way** rule set (AIG → gate cover rules plus the
  AIG structural rules). `map_cut_to_cover_cover_only` saturates that bundle and `map_cut_to_library`
  does the direct library match.
* `cut_mapping_with_objective` uses the cover-only egg path; `cut_mapping_library` uses the direct
  library path. Both are exposed over the cxx FFI together with the structured cell cover
  (`cell_nodes`, `pin_names`, `neg_mask`, `out_neg`, unrounded `area`/`delay`).
* The multi-cell sub-network database is not a Rust-side cache: the C++ side loads it from a `.super`
  library (`EPMAPPING_SUPER`) as a separate technology-library instance and offers those covers as
  another candidate source. `map_egg paths library_hits=/egg_calls=/egg_cache_hits=` still reports how
  many cuts the library matcher answered and how many e-graph calls the memo cache served.

Low equality-saturation limits are used for cut mapping (`MAP_EGG_EGG_ITER_LIMIT`,
`MAP_EGG_EGG_NODE_LIMIT`, defaults `2`/`500`): a cut AIG is tiny, while the mapper-wide defaults
(30 iterations / 10k nodes) spend seconds per cut without changing the extracted cover.

## Files

| file | role |
| --- | --- |
| `epmapping.hpp` | `epmapping::optimizer` — cuts, dispatch, candidate rebuild, substitution |
| `main.cpp` | CLI (`genlib`, `input.aig`, `workdir`), Verilog output, ABC CEC |
| `../map_egg/src/lib.rs` | FFI: `cut_mapping_library`, `cut_mapping_with_objective`; owns the cover-generation modules (the former `emapper` crate) |
| `../map_egg/src/emap_egg.rs` | genlib pre-load, gate→AIG expansion, one-way cover bundle |
| `../map_egg/src/emap_cover.rs` | `map_cut_to_library`, `map_cut_to_cover_cover_only` |
| `../map_egg/src/{emap_cut,techlib,tt9,aig_graph}.rs` | cut structure, technology library model, truth-table type, AIG graph |
| `../map_egg/src/{binding,params}.rs` | `NodeBinding`/`NodeBindingKind` and `EmapParams` |

The standalone IWLS'24 `emapper` tool (its own mapper, ABC wrappers, supergate cache and C++
executable) and `map_egg`'s unused upstream extractors were removed from this project; the
development tree `~/workplace/MIGBalance` still has both, and the removals are in this
repository's history (commits `f83fd5b`, `453ca93`).

## Usage

```sh
build/epmapping/epmapping_check <genlib> <input.aig> [workdir]
```

Environment knobs:

| variable | default | meaning |
| --- | --- | --- |
| `EPMAPPING_OBJECTIVE` | `1` | `1` = area-first (the ASPDAC'25 objective), `0` = delay-first |
| `EPMAPPING_CUT_SIZE` | `6` | max cut leaves (hard ceiling `9`: `map_egg` truth tables are 9-input) |
| `EPMAPPING_CUT_LIMIT` | `15` | cuts kept per node, **15** being the most the live cut database holds. `merge_cuts` only truncates *after* enumerating the whole tuple space, so the per-round cost grows like `cut_limit^k` for a k-input cell; 12 is the runtime knee (measured on `systemcdes`: `cut_enum` 214.8 s at 20, 17.3 s at 12, 2.0 s at 8, with identical mapped area at 12). The live cut database (`dynamic_network_cuts`) has a compile-time limit of 16 cuts per node, so values above 15 are clamped with a warning |
| `EPMAPPING_PRESERVE_DEPTH` | `1` | reject candidates deeper than the original gate |
| `EPMAPPING_PRESERVE_DELAY` | `1` | reject candidates slower than the original gate |
| `EPMAPPING_DELAY_SLACK` | `0.05` | relative local delay slack for area wins |
| `EPMAPPING_EGG` | `1` | run the e-graph (egg) stage. The staged flow used to run it unconditionally while this flag only fed the banner; it now gates the stage, and the default is the behaviour that was really in effect |
| `EPMAPPING_VALIDATE` | `1` | `1` = exhaustively check each accepted candidate against the cone |
| `EPMAPPING_MAP_MULTIOUTPUT` / `EPMAPPING_MAP_CUT_LIMIT` | `0` / `16` | baseline `emap` knobs (both measured neutral) |
| `EPMAPPING_SUPER` | — | supergate library (multi-cell database) for the *optimizer*; `tech_library` otherwise holds single gates only |
| `EPMAPPING_ROUNDS` | `6` | max reverse-topological passes (**6** = the reported tables' `rN`); a pass that replaces nothing or does not improve the objective ends the run early (`run()` in `epmapping.hpp`) |
| `EPMAPPING_BASELINE` | `high_effort` | initial (subject) netlist: `direct` = `read_aiger` + area-oriented `emap`, no ABC optimization (the paper's `A_i`); `high_effort` = `read_aiger` + `{fraig; resyn2rs; fraig; compress2rs}` iterated until the AIG stops shrinking + area-oriented `emap` (the paper's `A_e`); `custom` = one pass of `EPMAPPING_ABC_SCRIPT` |
| `EPMAPPING_ABC_ROUND` | `fraig; resyn2rs; fraig; compress2rs` | per-round script of the `high_effort` baseline |
| `EPMAPPING_ABC_ITERS` | `20` | iteration cap of the `high_effort` baseline; reaching it prints a warning, because a cap hit is not a fixpoint |
| `EPMAPPING_ABC_SCRIPT` | — | single-pass subject-graph preprocessing; setting it implies `EPMAPPING_BASELINE=custom` (the empty string implies `direct`). IWLS'24 subject graph: `fraig; compress2rs; resyn2rs; compress2rs`; one-pass ASPDAC'25 §4.1 flow: `fraig; rw; rs; rf; resyn2rs; compress2rs` |
| `EPMAPPING_AREA_MAPPING` | `1` | `1` = area-oriented `emap` (the ASPDAC'25 baseline `A_e`); `0` = delay-oriented |
| `EPMAPPING_RELAX_REQUIRED` | `100` with area mapping, else `0` | required-time relaxation (%) for `emap`; area-oriented mapping ignores `required_time`, so 100% is the natural value |
| `EPMAPPING_VERBOSE` | `0` | per-cut/progress logging |
| `MAP_EGG_EGG_LIBRARY_FIRST` | `1` | try a single library cell before egg on large cuts |
| `MAP_EGG_EGG_ITER_LIMIT` / `MAP_EGG_EGG_NODE_LIMIT` | `2` / `500` | egg limits for cut mapping |
| `MAP_EGG_FULL_BUNDLE` | `0` | `1` = saturate with `cell -> AIG` expansion too (measured neutral) |
| `EPMAPPING_EGG_REQ` | `1` | `1` = give the e-graph stage the leaf arrivals and the cone's original delay as an upper bound (min area subject to no delay regression); `0` = price with zero arrivals |
| `EPMAPPING_EGG_MIN_MFFC` | `0` | skip the staged e-graph stage when the MFFC area is below this (runtime-for-area trade) |
| `EPMAPPING_DC_MAX_PIS` | `8` | cap on the don't-care window inputs; the simulation is `2^n_pis`. A wider window is *discarded* (it reports no don't cares), so this caps the window path's cost; with `EPMAPPING_SAT_DC=1` the discarded cuts are still refined by SAT, which this cap does not affect |
| `EPMAPPING_SAT_DC` | `0` | `0` (**default** since 2026-09) = window simulation only, i.e. mockturtle's `rewrite`/`satisfiability_dont_cares` behaviour: cheap, deterministic, bounded; `1` = additionally refine the window's care mask with SAT over the fanin cone of the cut's leaves (SAT only *clears* minterms the cone cannot produce). It is the only don't-care source for the cuts whose window exceeds `EPMAPPING_DC_MAX_PIS` |
| `EPMAPPING_SINGLE_MATCHER` | `index` | matcher of the staged **single-cell** stage: `index` = the library's precomputed N/P supergate index (`get_supergates`, one hash probe per output phase and DC completion — `tech_library::generate_library()` already seeds every genlib cell with all its input permutations and polarities, so it explores the same configurations as the brute force); `brute` = the original brute force (all cells × `pins!` permutations × `2^pins` polarities × 2 output phases × every care minterm), which handles **any** DC count and was 60–70% of a pass; `hybrid` = `index`, then the brute force on the cuts whose DC count is above `EPMAPPING_SINGLE_FREE` |
| `EPMAPPING_SINGLE_ENTRIES` | `128` | supergate entries tried per output phase and DC completion by that stage (the library list is ordered by ascending area, so a small cap keeps the cheapest entries but can drop a better delay/level one) |
| `EPMAPPING_SINGLE_FREE` | `4` | free minterms the completion enumeration of that stage may enumerate (2^this completions). Above it the `index` matcher answers nothing for the cut; `hybrid` falls back to the brute force there |
| `EPMAPPING_DC_TIME_BUDGET` | `0` | wall-clock budget (seconds) for the whole don't-care computation of a run; cuts processed after it is spent are matched exactly (no DCs) — sound, and it bounds the runtime on designs with large windows (`sqrt`, `div`, `mem_ctrl`) |
| `EPMAPPING_DC_WINDOW` | `per_node` | which window the don't cares are computed on. `per_node` (**default** since 2026-09) = one reconvergence window per **root**, simulated once and shared by every cut of that root, with `max_leaves = EPMAPPING_DC_MAX_PIS` so no window is discarded (mockturtle's `rewrite`); a cut whose leaves are not all inside the window gets **no** don't cares. `per_cut` = one window per **cut**, grown from the cut's own leaves (mockturtle's `satisfiability_dont_cares(ntk, leaves)`), so every cut gets a care set and the mask is a pure function of the leaf set (which the per-round memo caches). Measured on the sibling engine (nine designs, A\_e, one pass, `cec=1`): `per_node` builds 2.2–5.9x fewer windows and spends 1.6–2.9x less don't-care time, with the same area on 7 of 9 designs and 0.01% worse on the other two |
| `EPMAPPING_SAT_DC_CALLS` / `EPMAPPING_SAT_DC_MAX_NODES` | `128` / `4000` | per-cut SAT-call budget (`0` = unlimited) and fanin-cone cap; exceeding either falls back to the window mask, which is sound. Both only matter with `EPMAPPING_SAT_DC=1`; at most 127 calls are possible per cut, so the default can never bind |
| `EPMAPPING_SYN` | `1` | structural-synthesis candidate source, **on by default**, with a **single gate**: `0` = off, non-zero = offer it on every cut at least `EPMAPPING_SYN_MIN_LEAVES` wide |
| `EPMAPPING_SYN_MODE` | `1` | synthesis modes (bitmask): `1` = SOP factoring, `2` = disjoint-support decomposition with factoring on the prime nodes, `3` = both |
| `EPMAPPING_SYN_MIN_LEAVES` | `5` | minimum cut width (leaves) for the synthesis stage |
| `EPMAPPING_SYN_NO_DC` | `0` | `1` = synthesize without the cut's don't cares (they are passed to the synthesizer by default) |
| `EPMAPPING_WINDOW_SYN` | `1` (engine) / `0` in the paper's stages | whole-window structural synthesis over the **MFFW** of a root's cut leaves: every window output is re-synthesized from the window inputs, the shared multi-output AIG is mapped jointly, and all outputs are substituted at once. The cover is materialized **unreachable** first and the commit is decided on the *measured* area change (plus a global critical-path gate), so a window that does not really get smaller is abandoned instead of applied. Uses `EPMAPPING_SYN_MODE` for the engines |
| `EPMAPPING_WINDOW_SYN_MIN_GAIN` | `0` | measured window area gain needed before a window is committed (`θ`) |
| `EPMAPPING_WINDOW_SYN_LEAVES` / `_GATES` | `6` / `64` | window input and gate caps |
| `EPMAPPING_WINDOW_FORCE` | — | **debug**: `1` = commit a window even when the measured change is not a win and even when the predicted critical path grows (compares the estimate against the post-commit measurement) |
| `EPMAPPING_CARE_CACHE` | `set` | care-mask memo: `off` or `set` (leaf set, minterm-permuted) |
| `EPMAPPING_CARE_VERIFY` | `0` | `1` = recompute on every cache hit and report mismatches (exactness check) |
| `MAP_EGG_TECHLIB_PATH` | — | set automatically by `main.cpp` |

`scripts/run_epmapping.sh [workdir] [benchmark ...]` runs a benchmark subset and prints a summary.

## Reproducing the experiments

Everything the DATE'27 paper reports is produced by one driver: the engine is built first, each stage is
resumable, and a final report aggregates the tables.

```bash
scripts/build.sh                                  # cmake + cargo (Release) -> build/
scripts/experiments/run_all.sh                    # every stage, then results/tables.{md,tex}
scripts/experiments/run_all.sh --list             # show the stages
scripts/experiments/run_all.sh --only tab_main,tab_mfs3,report
scripts/experiments/run_all.sh --skip runtime     # skip the timing-sensitive stages
scripts/experiments/run_all.sh --force tab_mfs3   # redo a stage that already has a .done marker
```

A full run takes hours (the three 27-design sweeps dominate), so start it in the background or under
`nohup`. Re-running without `--force` continues where it stopped: every stage drops a `.done` marker.

The paper's tables are all built on the same **16 circuits** (ASPDAC'25's Table 1); the 27-design
sweeps are the extended tables of this README and are labelled `ext27_*`.

| stage | experiment | output |
| --- | --- | --- |
| `tab_main` | the paper's comparison with ASPDAC'25 over its 16 circuits: one pass and convergence on A_e, convergence on A_i | `results/tab_main/summary.tsv` |
| `tab_mfs3` | the engine side of the paper's comparison with ABC's `mfs3 -a` over the same 16 circuits, on the 40-cell ASAP7 subset (cells with at most six inputs; both start from the area-oriented `emap` netlist, one pass, 1000 s): the 16 baselines plus the 16 one-pass runs | `results/tab_mfs3/summary.tsv` |
| `tab_mfs3_abc` | ABC's side of the same comparison: `mfs3 -a` on each baseline netlist, same 1000 s cap, area/delay/runtime per design | `results/tab_mfs3/abc.tsv`, `results/tab_mfs3/abc.<design>.log` |
| `iscas_ae` / `iscas_iwls` | the ISCAS subset on both graphs | `results/<stage>/results.csv` |
| `ext27_ae` / `ext27_iwls` / `ext27_ai` | the extended 27-design set on each subject graph (README only) | same |
| `oracle` | cut stages alone / with synthesis / with the window fallback, both graphs | `results/oracle/summary.tsv` |
| `db_build` | one-off: produces the area-capped supergate libraries (`a045`/`a055`/`a060`) with ABC `super` into `epmapping/db/`, only when missing | `epmapping/db/*.super`, `results/db_build/libraries.tsv` |
| `db` | `c7552` against six database variants: single cells only (`none`), the two-level database, and the three-level one at four area caps | `results/db/summary.tsv` |
| `funnel` | fig:share(a): the layer funnel in one pass (L1 wins / L4 admitted / L4 wins), with whole-run wall clock | `results/funnel/summary.tsv` |
| `lever1` | SAT refinement on/off, reconvergence-window cap 0/2/8 | `results/lever1/summary.tsv` |
| `lever1_micro` | the Lever-1 micro-benchmarks: window cap 8 vs 10 on `sin`, `DC_FREE` 4/12 on `sasc`, and the per-cut vs per-node window scheme over the eleven designs | `results/lever1_micro/summary.tsv` |
| `lever3` | saturation budget and node limit (search harder), plus the full configuration | `results/lever3/summary.tsv` |
| `runtime` | one pass with and without the e-graph, and convergence | `results/runtime/summary.tsv` |
| `ablation` | the eleven-design, six-round ablation behind fig:share(b,c): `full`, `noSYN`, `noWINDOW`, `noEGG`, `noCARE` (`EPMAPPING_SAT_DC=0`) and `noDB` (`EPMAPPING_SUPER=off`). This stage runs the window **on** (`EPMAPPING_WINDOW_SYN=1`, unlike the paper's stages) because the window is one of the stages it removes | `results/ablation/summary.tsv` |
| `report` | aggregation into tables | `results/tables.md`, `results/tables.tex` |

`report.py` turns those logs into the figure data as well: the fig:share(a) `\\addplot`
coordinates, the oracle-time and removed-stage area shares of fig:share(b,c), the e-graph memo-cache
hit rate, and the database coverage of §Lever 1.

Every run also keeps its raw engine log at `results/<stage>/<config>.<design>/log`, so any number can be
traced to the line that printed it.

`tab:mfs3` is the one comparison whose second column comes from another tool: `tab_mfs3` writes each
`mfs3` baseline netlist, `tab_mfs3_abc` runs `mfs3 -a` on it under the same 1000 s cap, and `report.py`
reads both sides back. Re-measured on 2026-09-23 (engine `4e94a70e...`), 13 of the 14 rows `mfs3 -a`
completes reproduce the paper's values exactly. The exception is `sqrt`: the paper listed it as
`TO@1000`, but it does finish (1270.98 um2, -6.21 %, in 779 s), so the paper's `tab:mfs3` now carries
that row and a mean over 14 designs. `div` and `mem_ctrl` exceed the cap on both sides, and every
`ours` number in the stage is in `results/tab_mfs3/summary.tsv`.

Runtimes are wall clock of an *awake* machine. macOS suspends an idle laptop, and a suspended run
keeps its `alarm` timer frozen while a stopwatch keeps counting: on 2026-09-23 that recorded a 780 s
`mfs3` run as 6571 s and made the 1000 s cap look ineffective. Every measured run is therefore wrapped
in `caffeinate`; set `EPM_CAFFEINATE=0` to disable that.

### Which stage produces which paper artifact

| paper artifact | stage |
| --- | --- |
| `tab:mfs3` (16 circuits, 40-cell library, `mfs3 -a` vs ours) | `tab_mfs3` + `tab_mfs3_abc`, compared by `report.py` |
| `tab:main` (16 circuits vs ASPDAC'25, both recipes) | `tab_main` |
| `fig:share(a)`, the layer funnel | `funnel` + `report.py` (counter parsing) |
| `fig:share(b,c)`, oracle-time and removed-stage area shares | `ablation` (`full`/`noSYN`/`noWINDOW`/`noEGG`/`noCARE`/`noDB`) + `report.py` |
| §Lever 1: care-mask cost, window cap 8 vs 10, per-cut vs per-node, exact refinement | `lever1`, `lever1_micro` |
| §Lever 1: stage ablation (+12.25 synthesis, +3.44 care, +1.11 e-graph, +0.10 database) | `ablation` |
| §Lever 1: database coverage (no entry for 3/4 of the four-input space, 45 %, `c7552`) | `db` (`EPMAPPING_DB_STATS=1`) |
| §Lever 2: "searching harder does not pay" (saturation budget, node limit) | `lever3` |
| §Lever 2: memo-cache hit rate | `funnel`/`ablation` + `report.py` |
| ISCAS numbers in the text | `iscas_ae`, `iscas_iwls` |
| runtime claims | `runtime`, `funnel` |
| the dependency-cut numbers (`80.33 -> 78.91`) | not scripted on purpose: the engine does not form dependency cuts, and the paper says so |
| `tab:layers` (method.tex) | conceptual, no data |
| the extended 27-design tables (README only) | `ext27_ae`, `ext27_iwls`, `ext27_ai` |
| `tab:oracle` (README only) | `oracle` |

### Checking the scripts without running the engine

```bash
scripts/experiments/selftest.sh                  # syntax + spec + report-parsing checks
scripts/experiments/run_all.sh --dry-run         # print every stage's run list, run nothing
```

`selftest.sh` fails if a shell script does not parse, if a stage generates a malformed run (bad
working directory, missing benchmark, non-numeric round count, unknown recipe, non-`KEY=VALUE`
argument), if two stages would write the same work directory, or if `report.py` stops finding a
section in synthetic logs.

### The supergate libraries are built once

Every number except `tab:mfs3` runs with `EPMAPPING_SUPER` pointing at a pre-mapped multi-cell
sub-network database, so the libraries are the one expensive artifact that is not re-derived per run:

* `epmapping/db/asap7_4in_2lvl.super` and `asap7_4in_3lvl_a050.super` (the one the paper's tables use)
  ship with the repository;
* `db_build` produces the rest (`a045`, `a055`, `a060`) with ABC's `super -I 4 -L 3 -A <cap>` into the
  same directory, **only when the file is missing**, and records sizes and SHA-256 in
  `results/db_build/libraries.tsv`;
* the `db` measurement stage never generates anything: it fails loudly and points at `db_build` if a
  library is absent. Point `EPM_DB_DIR` elsewhere to keep the libraries in a shared location.

The `db` stage compares the single-cell case (`none`, genlib cells only) with the multi-cell
databases, which is what the `45 %` coverage and the "the database is worth `+0.10`" numbers come
from.

### Knobs

The defaults are the paper's setting; all of them can be overridden in the environment.

| variable | default | meaning |
| --- | --- | --- |
| `EPM_ROOT` | `<project>/results` | output root for all stage data |
| `EPM_JOBS` | 3 | designs in flight |
| `RAYON_NUM_THREADS` | 3 | threads per engine process |
| `EPM_TIMEOUT` | 1800 | seconds per engine run; a timeout *is* a measurement (recorded, never retried) |
| `EPM_CAFFEINATE` | 1 | wrap every measured run in `caffeinate -i -s` so an idle-sleeping laptop cannot inflate runtimes (macOS) |
| `EPM_DC_BUDGET` | 60 | `EPMAPPING_DC_TIME_BUDGET`; without it `sin`, `log2`, `multiplier` and `sqrt` cannot converge in 1800 s |
| `EPM_ROUNDS` | 6 | rounds of the converged (`rN`) configuration (`EPMAPPING_RN_ROUNDS` for `compare_with_paper.sh`) |
| `EPM_BIN` | newest build | engine binary to use |

An intermittent crash in the library setup (`rc=139`) is retried up to three times; a timeout is not.

### Provenance

The driver prints the engine path and its SHA-256 before running anything, and each stage keeps a
timestamped `.done` marker. Note that a C++ rebuild is **not** bit-reproducible: the same sources can
move a few results (e.g. `max` on the `A_i` graph came out 139.38 with one build and 139.45 with the
next, both far from the 140.91 of the pre-fix engine). Quote the hash printed at the start of a run,
not only the source revision, when a number matters.

## Example results (asap7, area-first, `delay_slack=0.05`, iterated)

Baseline = `emap` on the default ABC preprocessing (the `high_effort` `A_e` subject graph:
`{fraig; resyn2rs; fraig; compress2rs}` iterated to its node-count fixpoint) with the default
area-oriented mapping, i.e. the ASPDAC'25 `A_e` point. The tables below were measured on the *one-pass*
§4.1 subject graph (`EPMAPPING_ABC_SCRIPT="fraig; rw; rs; rf; resyn2rs; compress2rs"`), so their
baseline column is a little larger than a bare run's now; those measured on the **IWLS'24** subject
graph say so and need `EPMAPPING_ABC_SCRIPT="fraig; compress2rs; resyn2rs; compress2rs"`.
Set `EPMAPPING_ABC_SCRIPT="resyn2rs; compress2rs"` for the lighter subject graph.

| design | base area | opt area | ΔA | base delay | opt delay | ΔD | rounds | CEC | ΔA with `EGG=1` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c432 | 10.81 | 10.81 | 0.00% | 295.71 | 295.71 | 0.00% | 1 | 1 | −1.39% |
| c499 | 46.78 | 46.78 | 0.00% | 243.37 | 243.37 | 0.00% | 1 | 1 | −0.04% |
| c880 | 20.40 | 20.27 | −0.64% | 276.22 | 281.50 | +1.91% | 3 | 1 | −0.64% |
| c1355 | 38.93 | 38.93 | 0.00% | 248.14 | 248.14 | 0.00% | 1 | 1 | −0.26% |
| c1908 | 30.10 | 30.03 | −0.23% | 266.94 | 268.11 | +0.44% | 2 | 1 | −0.23% |
| c2670 | 37.88 | 37.84 | −0.11% | 202.92 | 202.92 | 0.00% | 2 | 1 | −0.05% |
| c3540 | 61.93 | 61.72 | −0.34% | 387.44 | 394.82 | +1.90% | 2 | 1 | −1.02% |
| c5315 | 73.50 | 73.47 | −0.04% | 352.27 | 352.74 | +0.13% | 2 | 1 | −0.04% |
| c7552 | 84.00 | 83.96 | −0.05% | 548.41 | 549.72 | +0.24% | 2 | 1 | −0.11% |
| max | 180.24 | 180.10 | −0.08% | 1895.20 | 1904.40 | +0.49% | 4 | 1 | −0.45% |
| sin | 410.61 | 409.16 | −0.35% | 1790.45 | 1801.25 | +0.60% | 3 | 1 | −0.63% |

The last column is the same run with `EPMAPPING_EGG=1` (egg on 5–6-leaf cuts): egg still wins on most
ISCAS designs (−1.39% vs 0.00% on c432, −1.02% vs −0.34% on c3540) at 3–20× the runtime, while the
native cover wins on c2670. Both are ABC-CEC equivalent.

All designs are ABC-CEC equivalent to the original AIGER. Local delay is bounded by
`delay_slack`; set `EPMAPPING_PRESERVE_DELAY=0` for unrestricted area-first, or
`EPMAPPING_OBJECTIVE=0` for delay-first.

Note how much the `fraig` preprocessing matters for *apparent* gain: on the lighter
`resyn2rs; compress2rs` subject graph the same engine reported c499 **−5.20%**, but that subject
graph still contained the redundancies `fraig` removes, so the win was mostly re-doing ABC's job.
On the paper's graph c499 is −0.04%. c432 goes the other way (−1.06% → −1.39%) because its optimized
graph is smaller and the relative effect of each replacement is larger.

### Paper baseline (ASPDAC'25 initial netlist, the default)

The initial netlist is generated by one of the paper's two recipes (`EPMAPPING_BASELINE`), and the
mapping is area-oriented `emap` (`EPMAPPING_AREA_MAPPING=1`, `EPMAPPING_RELAX_REQUIRED=100`):

```sh
# A_e, the paper's optimized subject graph (the default): iterate
#   fraig; resyn2rs; fraig; compress2rs
# until the AIG cannot be reduced any further, then map.
  build/epmapping/epmapping_check <genlib> <aig> <workdir>

# A_i, the unoptimized subject graph: read_aiger + area-oriented emap, no ABC optimization at all.
EPMAPPING_BASELINE=direct build/epmapping/epmapping_check <genlib> <aig> <workdir>

# One-pass readings (this is how the tables measured before the presets existed are reproduced):
EPMAPPING_ABC_SCRIPT="fraig; rw; rs; rf; resyn2rs; compress2rs" \
  build/epmapping/epmapping_check <genlib> <aig> <workdir>      # ASPDAC'25 §4.1, one pass
EPMAPPING_ABC_SCRIPT="fraig; compress2rs; resyn2rs; compress2rs" \
  build/epmapping/epmapping_check <genlib> <aig> <workdir>      # IWLS'24 subject graph
```

The `high_effort` loop stops when a round no longer reduces the AND count and reports every round, so
the subject graph a comparison starts from is always identifiable; if the `EPMAPPING_ABC_ITERS` cap is
reached instead, it prints a warning, because a cap hit is not a fixpoint.

Where the subject graphs agree, a bare run reproduces the paper's `A_e`/`D_e`: `arbiter` is exactly at
its fixpoint (our `A_e` 557.72 / 999.87 against their 557.84 / 999.95). `sqrt` is close (1157.38
against 1171.15). On most other designs our ABC loop ends up *smaller* than their published `A_e`
(Table 1 starts from `A_e`; see the deviation table below), which is the honest bound on how
comparable the percentage gains are: a smaller start leaves less headroom, which is the paper's own
thesis.

On the *one-pass §4.1* graph (the previous default) the current engine finds **0%** on c432/c499/c1355:
an area-oriented mapped netlist is already on the local MFFC frontier for structural cuts with
completely specified functions. Reproducing the paper's additional ~2.5% requires the two engines it
adds — don't-care matching and non-local (dependency) cuts — not more iterations.

`block_network` copies share storage and `cell_view::clone()` drops cell bindings, so the optimizer
takes a deep topological rebuild of its input (`optimizer::ntk()` is the optimized network).

### Fair comparison with IWLS'24 (Table III setup)

Reproduce with
`EPM_OUT=/tmp/epmap_fair EPM_JOBS=5 EPMAPPING_ABC_SCRIPT="fraig; compress2rs; resyn2rs; compress2rs" ./scripts/compare_with_paper.sh 1800 <designs...>`
(the override is needed now: the default subject graph is the ASPDAC'25 `high_effort` one)
(the helper defaults to the shipped sub-network database and to the `plain`+`dc2` design-space
portfolio; `EPMAPPING_DSE=0`, `EPMAPPING_DSE_PIPES=`, `EPMAPPING_SUPER=` override them). The script
runs the area-oriented baseline (`EPMAPPING_OBJECTIVE=1`; the mapping defaults are already
area-oriented) plus
one pass (`ROUNDS=1`) and up to six passes (`ROUNDS=6`), and CECs all three with ABC.
`EPMAPPING_BASELINE_CEC=1` checks the raw `emap` netlist as well.

| design | base area | δA1e | δA∞e | paper δA1e | paper δA∞e | t1 (s) | t∞ (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adder | 57.40 | -0.03% | -0.03% | — | — | 63 | 70 |
| arbiter | 557.72 | -1.11% | -1.11% | −14.08% | **−43.85%** | 68 | 81 |
| bar | 133.63 | -6.56% | -6.56% | — | — | 65 | 65 |
| c6288 | 132.60 | -3.48% | -3.48% | −0.25% | 1/1/1 | 67 | 58 |
| cavlc | 32.70 | -4.01% | -4.13% | −1.20% | −0.96% | 55 | 67 |
| ctrl | 5.80 | -1.55% | -1.55% | — | — | 58 | 59 |
| dec | 27.06 | +0.00% | +0.00% | — | — | 60 | 66 |
| div | 1184.83 | -0.16% | -0.30% | −9.53% | — | 164 | 282 |
| i2c | 57.12 | -6.37% | -6.48% | — | — | 59 | 60 |
| int2float | 10.90 | -4.77% | -4.77% | — | — | 66 | 57 |
| iwls05_i2c | 49.19 | -4.51% | -4.51% | — | −1.26% | 65 | 64 |
| log2 | 1455.42 | -0.12% | -0.15% | — | — | 264 | 768 |
| max | 137.61 | -1.55% | -1.55% | — | — | 65 | 73 |
| mem_ctrl | 2025.00 | -16.63% | -16.63% | — | −12.25% | 247 | 700 |
| multiplier | 1270.30 | -0.09% | -0.09% | — | — | 108 | 127 |
| priority | 27.63 | -3.91% | -3.98% | — | — | 65 | 66 |
| router | 7.69 | -5.46% | -5.59% | — | — | 58 | 61 |
| sasc | 27.61 | -0.36% | -0.36% | — | −1.39% | 58 | 57 |
| simple_spi | 36.62 | -0.82% | -0.82% | — | −1.88% | 62 | 59 |
| sin | 274.08 | -1.70% | -2.78% | −0.24% | −1.41% | 113 | 216 |
| spi | 147.65 | -3.20% | -3.20% | — | −1.69% | 256 | 197 |
| sqrt | 1025.43 | -0.14% | -0.14% | — | −5.24% | 127 | 179 |
| square | 1064.58 | -0.22% | -0.38% | — | — | 109 | 170 |
| systemcaes | 476.89 | -5.60% | -7.56% | — | −3.60% | 81 | 92 |
| systemcdes | 130.67 | -5.95% | -5.95% | — | −1.58% | 167 | 290 |
| usb_phy | 22.67 | -0.93% | -0.93% | — | −1.96% | 61 | 68 |
| voter | 522.97 | -0.38% | -0.38% | — | — | 76 | 79 |
| **avg (27)** | | **-2.95%** | **-3.09%** | −1.06% (39) | −2.50% (39) | 100 | 153 |

Every completed run is ABC-CEC equivalent (base, r1 and rN); the baseline netlists also CEC on their
own, so the `cec=0` failures seen earlier on `ctrl`/`sasc`/`simple_spi`/`i2c`/`usb_phy`/`systemcaes`
were a writer/build bug in this optimizer, not a mapping bug — see the primary-output pitfall below.

Reading of the table:

* **Baselines.** Our baseline is the same area-oriented `emap` point as the paper's on the designs
  where both are published (`arbiter` 557.72 vs 557.84, `cavlc` 32.70 vs 34.27, `mem_ctrl` 2025.00 vs
  2063.01), so the comparison is like-for-like there. On `div`/`sqrt` our baseline is *smaller* than
  the paper's `A_e` (1184.83 vs 1296.90 and 1025.43 vs 1171.15), i.e. our headroom is correspondingly
  smaller — the paper's percentages on those two start from a worse point. Reproducing their `A_e`
  exactly was attempted: `EPMAPPING_REQUIRED_FROM_DELAY=1` maps once for delay and then recovers area
  with that critical path as the required time, but mockturtle's `area_oriented_mapping` mode ignores
  `required_time` (the result is bit-identical to pure area mapping), while the delay-oriented flow
  with its area-recovery rounds overshoots (div 1595.65). So the shipped baseline is the honest,
  if not identical, area-oriented control; it is documented rather than tuned to inflate δA.
* **The engine.** These numbers are the recommended configuration (`scripts/_epm_one.sh` exports it):
  the paper-style baseline, then the pre-mapped multi-cell sub-network database
  (`EPMAPPING_SUPER=epmapping/db/asap7_4in_3lvl_a050.super`) and one design-space-exploration cycle
  per configuration
  (`EPMAPPING_DSE_PIPES="plain dc2"`). Average **δA1 = -2.95%, δA∞ = -3.09%**, i.e. both of the paper's
  thresholds (−1.06% and −2.50%) are met and passed. Without the DSE cycle the same sweep was
  **δA1 = −0.10%, δA∞ = −0.11%**, so the design-space-exploration loop is what produces almost all of
  the gain (paper: −1.06% / −2.50% over 39 designs). Largest per-design wins: `mem_ctrl` −17.3%,
  `bar` −6.6%, `i2c` −6.4%, `router` −5.1%, `int2float` −4.8%, `cavlc` −4.1%, `c6288` −3.5%,
  `sin` −2.8%; the engine also *reduces* delay on most of them. Without any database the same sweep
  was −0.03%/−0.04%, with the 2-level database −0.03%/−0.04%, with the shipped 3-level one
  −0.10%/−0.11%.
* **Every run is ABC-CEC equivalent** (`cec=1/1/1` for base/r1/rN on all 27 designs), with no timeouts
  and no crashes. Each design now runs in its own work directory — the Verilog hand-off has a fixed
  file name, so the older shared-directory runs had a latent race between parallel designs.
* **Runtime.** The 16 MB database costs 34 s *per invocation* (one library build per run) and the
  richer match set lengthens the search: the first pass averages 96 s against the paper's 8.58 s, and
  convergence 146 s against 38.67 s (`log2` 787 s, `mem_ctrl` 600 s, `div` 284 s, `systemcdes`
  241 s). Every run still finishes inside the sweep's 1500 s budget, with no crash and no timeout. `EPMAPPING_EGG=1` is far worse (194 s average,
  five timeouts) and buys nothing.
* Conclusion: the remaining ~2.4% is **not** reachable from this cut space, with or without a mapped
  sub-network database. The MFFC measurement above shows why the reward of a *local* replacement is
  bounded by a dependency cut that almost always has a one-gate MFFC; the paper's remaining area comes
  from non-local divisors, global/observability don't cares and SAT-verified candidates, which are
  listed as the next step below.

### Fair comparison with ASPDAC'25 (matched initial netlist)

The follow-up paper *Area-Oriented Optimization After Standard-Cell Mapping* (Costamagna, Tempia
Calvino, Mishchenko, De Micheli, ASPDAC'25, `10.1145/3658617.3697722`) is the paper this engine is
now measured against. It also uses asap7 + Mockturtle `emap`, but generates its subject graph
differently: **high-effort area-oriented AIG optimization** — `fraig` first, then the area commands
`rw`, `rs`, `rf`, `resyn2rs`, `compress2rs`, iterated until the subject graph stops shrinking — and
then *area-oriented* mapping with the best achievable delay as the (nominal) constraint. Its Table 1
reports the mapped area `A_e`, delay `D_e` and its own post-mapping resubstitution improvement after
one pass (`t_e1`) and to convergence (`t_e∞`); Table 2 adds a mapped design-space-exploration cycle
(`δA5×5`).

To compare like-for-like the initial netlist is generated the same way:

```
# both are the engine's defaults; shown for explicitness
EPMAPPING_BASELINE=high_effort            # their A_e subject graph: {fraig; resyn2rs; fraig; compress2rs}
                                          # iterated until the AIG cannot be reduced any further
EPMAPPING_AREA_MAPPING=1 EPMAPPING_RELAX_REQUIRED=100            # their area-oriented mapping
```

The alternative initial netlist is `EPMAPPING_BASELINE=direct` (the paper's `A_i`): `read_aiger` plus
area-oriented `emap`, no ABC optimization at all.

Mockturtle's area-oriented mode ignores `required_time` (see the `REQUIRED_FROM_DELAY` note above),
so "best possible delay as constraint" is nominal there — the same behaviour the paper's own mapper
has. `scripts/compare_with_aspdac25.sh <results.csv>` prints every table below.

**Which ABC recipe?** The paper gives *three* different subject-graph recipes, one per experiment:

* **§4.1 (aggressive flow, Table 1):** "an aggressive optimization flow running the following
  area-oriented commands and scripts in ABC: `rw`, `rs`, `rf`, `resyn2rs`, and `compress2rs`" …
  "Before applying any optimization, the AIG is functionally reduced using command `fraig`" … "We
  start with one-pass commands `rw`, `rs`, `rf` to slow down convergence" … "We iterate this
  procedure as long as there are changes in the subject graph."
* **§4.3 (high-effort/delay-constrained, Table 2):** "High-effort optimization is achieved by
  iteratively applying `resyn2rs` and `compress2rs` until the subject graph cannot be reduced, and
  merging equivalent nodes using command `fraig` before applying any script."
* **§4.4 (design-space exploration, Table 2):** "The subject graphs are optimized using **two
  iterations of the script `fraig; compress2rs; resyn2rs`**."

The engine's two initial-netlist presets are the paper's own two readings: `direct` for the
unoptimized `A_i` graph, and the `high_effort` fixpoint loop (`EPMAPPING_ABC_ROUND`, default
`fraig; resyn2rs; fraig; compress2rs`, i.e. `fraig` before *each* script as §4.3 requires) for `A_e`.
Both are exact in the sense that the recipe is fixed and the stopping rule (a round that no longer
reduces the AND count) is reported, so a run is never compared against an arbitrarily truncated
subject graph. The one-pass §4.1 reading that the older tables in this file were measured on is still
available as `EPMAPPING_ABC_SCRIPT="fraig; rw; rs; rf; resyn2rs; compress2rs"` (`custom`).

`scripts/abc_recipe_probe.sh` runs each candidate through ABC + area-oriented `emap` and scores the
mean deviation from the published `A_e` over the twelve designs whose value is published (lower is
better; the `HF∞` row is the engine's default):

| recipe | mean dev | max dev | `mem_ctrl` ours/paper | `sqrt` ours/paper | `div` ours/paper |
| --- | --- | --- | --- | --- | --- |
| **`P1` = `fraig; rw; rs; rf; resyn2rs; compress2rs`** | **5.11%** | 13.6% | 2061.1 / 2063.0 | 1164.3 / 1171.2 | 1192.7 / 1296.9 |
| `Q1` = `fraig; resyn2rs; compress2rs` | 5.37% | 14.0% | 2066.1 / 2063.0 | 1164.8 / 1171.2 | 1192.3 / 1296.9 |
| `compress2rs` alone (no `fraig`) | 5.89% | 12.8% | 2146.3 / 2063.0 | 1021.1 / 1171.2 | 1197.5 / 1296.9 |
| `P2` = `P1` twice | 6.09% | 13.6% | 2012.1 / 2063.0 | 1162.1 / 1171.2 | 1191.4 / 1296.9 |
| `Q1b`…`Q1d` = §4.3 recipe iterated 2–4× | 6.4–6.9% | 14.0% | 1985.5→1930.1 / 2063.0 | 1151.7→1148.1 / 1171.2 | ≈1188 / 1296.9 |
| `S44` = §4.4 recipe | 7.26% | 13.2% | 1975.9 / 2063.0 | 1025.6 / 1171.2 | 1185.0 / 1296.9 |
| `HF1` = `fraig; resyn2rs; fraig; compress2rs` (§4.3 literal, one round) | 5.47% | 14.0% | 2053.6 / 2063.0 | 1157.4 / 1171.2 | 1190.6 / 1296.9 |
| `HF2`…`HF5` = that loop run 2–5× (our node-count fixpoint) | 6.5–7.1% | 14.0% | 1978.4→1899.5 / 2063.0 | 1148.9→1146.4 / 1171.2 | ≈1187 / 1296.9 |
| `fraig` only | 13.06% | 39.9% | 2268.8 / 2063.0 | 1349.5 / 1171.2 | 1814.3 / 1296.9 |

`fraig` **between** the two scripts (the §4.3 wording "merging equivalent nodes using command
`fraig` before applying any script") matters: `HF1` reaches `mem_ctrl` 2053.6 against `Q1`'s 2066.1.
But running that loop to our node-count fixpoint *overshoots* the published table (`mem_ctrl`
1978→1900 for 2–5 rounds against 2063), and `HF1` is statistically tied with the engine's default
`P1` over the twelve designs (mean 5.47% vs 5.11%; `P1` closer on `sqrt`/`spi`/`tv80`/`div`, `HF1` on
`aes_core`/`mem_ctrl`/`sasc`/`systemcdes`). Both are therefore defensible readings of Table 1's
*one-pass* numbers — `P1` can still be selected with
`EPMAPPING_ABC_SCRIPT="fraig; rw; rs; rf; resyn2rs; compress2rs"` — but the engine's default initial
netlist is now the `high_effort` fixpoint loop (`HFI` below), because that is the recipe both `A_i` and
`A_e` are defined against (§4.2/§4.3: what the paper maps and measures its gains on).

**The two initial netlists, measured.** Reproduce with
`EPMAPPING_BASELINE=direct` (`A_i`) and `EPMAPPING_BASELINE=high_effort` (`A_e`, the default); `rnd` is
the number of `{fraig; resyn2rs; fraig; compress2rs}` rounds (the round that no longer reduced the AND
count included), i.e. how far the loop had to run to reach its node-count fixpoint.

| design | `A_i` ours | `A_i` paper | Δ | `A_e` ours | `A_e` paper | Δ | rnd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arbiter | 557.72 | 557.84 | −0.0% | 557.72 | 557.84 | −0.0% | 1 |
| mem_ctrl | 2278.85 | 2547.32 | −10.5% | 1895.40 | 2063.01 | −8.1% | 11 |
| sqrt | 1354.92 | 1372.25 | −1.3% | 1157.38 | 1171.15 | −1.2% | 2 |
| div | 3609.30 | 3914.60 | −7.8% | 1184.50 | 1296.90 | −8.7% | 6 |
| aes_core | 1103.53 | 1198.55 | −7.9% | 1029.98 | 1106.60 | −6.9% | 16 |
| iwls05_i2c | 58.77 | 66.32 | −11.4% | 47.66 | 49.96 | −4.6% | 5 |
| sasc | 35.49 | 40.46 | −12.3% | 27.28 | 31.72 | −14.0% | 2 |
| simple_spi | 49.18 | 54.77 | −10.2% | 36.96 | 41.58 | −11.1% | 3 |
| spi | 182.17 | 205.45 | −11.3% | 149.43 | 167.59 | −10.8% | 3 |
| systemcdes | 150.27 | 172.17 | −12.7% | 131.06 | 142.22 | −7.8% | 6 |
| tv80 | 460.20 | 512.39 | −10.2% | 340.61 | 356.38 | −4.4% | 4 |
| usb_phy | 27.15 | 28.82 | −5.8% | 22.54 | 23.97 | −6.0% | 3 |
| ethernet | 3735.35 | — | — | 2541.41 | 3123.35 | −18.6% | 4 |
| RISC | 3564.94 | — | — | 2927.19 | 3172.54 | −7.7% | 6 |
| systemcaes | 558.96 | — | — | 483.75 | 530.89 | −8.9% | 10 |
| usb_funct | 792.68 | — | — | 646.23 | 702.45 | −8.0% | 5 |
| c7552 | 88.21 | — | — | 81.87 | — | — | 4 |

Averaged over the twelve designs whose `A_i`/`A_e` the paper publishes: `A_i` mean |Δ| 8.45% (max
12.7%), `A_e` mean |Δ| 6.97% (max 14.0%). Every deviation is negative (our ABC ends up stronger), so
post-mapping gains here are measured on a smaller netlist and are therefore *conservative*.

The recipe is nonetheless the paper's, and that is testable on the paper's own effect size — the gain
that subject-graph optimization alone produces, `A_e/A_i − 1`:

| design | ours | paper | | design | ours | paper |
| --- | --- | --- | --- | --- | --- | --- |
| sqrt | −14.6% | −14.7% | | mem_ctrl | −16.8% | −19.0% |
| div | −67.2% | −66.9% | | iwls05_i2c | −18.9% | −24.7% |
| spi | −18.0% | −18.4% | | tv80 | −26.0% | −30.4% |
| simple_spi | −24.8% | −24.1% | | systemcdes | −12.8% | −17.4% |
| usb_phy | −17.0% | −16.8% | | aes_core | −6.7% | −7.7% |
| sasc | −23.1% | −21.6% | | arbiter | +0.0% | +0.0% |

Our two recipes reproduce the paper's `A_e/A_i` ratio on the tightly-matched designs (`sqrt`, `div`,
`spi`, `simple_spi`, `usb_phy`, `sasc`, all within ~1.5 points) and are slightly weaker on the
`fraig`-dominated ones, which is consistent with a `fraig`/resubstitution difference rather than a
different recipe — the absolute areas stay systematically below the published ones, and the remainder
is the mapper (the paper constrains area-oriented mapping with the delay of the delay-oriented
mapping; mockturtle's area-oriented mode ignores `required_time`, so that constraint is nominal for us
— `EPMAPPING_REQUIRED_FROM_DELAY` re-enables it). `A_i` is area-oriented: mapping it delay-oriented
instead lands 14.93% off the published `A_i` (37% on `arbiter`) against 8.45%.
Reproduce the whole table with `EPM_RECIPES='HFI|AEd' scripts/abc_recipe_probe.sh`.

**The one-pass baseline reproduces where the subject graphs agree.** `emap` area/delay on the *one-pass*
§4.1 subject graph (the engine's default before the `A_e` loop was adopted, selectable with
`EPMAPPING_ABC_SCRIPT="fraig; rw; rs; rf; resyn2rs; compress2rs"`) versus their published `A_e`/`D_e`
(16 designs they list, sorted by agreement):

| design | `A_e` paper | `A_e` ours | Δ | `D_e` paper | `D_e` ours | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| arbiter | 557.84 | 557.72 | −0.02% | 999.95 | 999.87 | −0.01% |
| mem_ctrl | 2063.01 | 2061.12 | −0.09% | 1649.46 | 1640.58 | −0.54% |
| sqrt | 1171.15 | 1164.26 | −0.59% | 78957.63 | 87244.90 | +10.5% |
| tv80 | 356.38 | 350.29 | −1.71% | 907.86 | 825.85 | −9.0% |
| aes_core | 1106.60 | 1072.14 | −3.11% | 434.52 | 393.79 | −9.4% |
| iwls05_i2c | 49.96 | 48.12 | −3.68% | 288.07 | 302.09 | +4.9% |
| usb_phy | 23.97 | 22.91 | −4.42% | 179.06 | 163.29 | −8.8% |
| usb_funct | 702.45 | 655.43 | −6.69% | 722.00 | 583.42 | −19.2% |
| RISC | 3172.54 | 2941.43 | −7.28% | 1304.74 | 1178.78 | −9.7% |
| div | 1296.90 | 1192.73 | −8.03% | 60248.23 | 53451.40 | −11.3% |
| systemcaes | 530.89 | 487.24 | −8.22% | 784.00 | 721.60 | −8.0% |
| systemcdes | 142.22 | 130.19 | −8.46% | 530.29 | 463.80 | −12.5% |
| simple_spi | 41.58 | 37.24 | −10.44% | 287.00 | 245.13 | −14.6% |
| sasc | 31.72 | 27.42 | −13.56% | 191.00 | 191.20 | +0.10% |
| spi | 167.59 | 155.59 | −7.16% | 489.49 | 529.67 | +8.2% |
| ethernet | 3123.35 | 2543.72 | −18.56% | 588.34 | 644.11 | +9.5% |

On `arbiter` and `mem_ctrl` the baseline matches almost exactly (0.02% / 0.09% in area), which
confirms the same library, mapper and flow; on the rest our `rw; rs; rf; resyn2rs; compress2rs`
reproduction ends up **smaller** than their published `A_e` (up to 18.6% on `ethernet`), i.e. our AIG
optimizer is stronger than the one behind their table. The comparison is therefore only exact on
`arbiter`/`mem_ctrl`/`sqrt`; elsewhere it starts from a smaller netlist, so the percentage gains are
not strictly comparable (a smaller start leaves less headroom, which is the paper's own thesis).

**Post-mapping comparison.** Area/delay/runtime of our engine on that same initial netlist, no DSE
(their Table 1 setting), against their published resubstitution result. The table below is the
historical measurement on the **one-pass §4.1** subject graph
(`EPMAPPING_ABC_SCRIPT="fraig; rw; rs; rf; resyn2rs; compress2rs"`); the runs on the current default
(`high_effort`) subject graph are in the following table.

| configuration (16 designs) | δA1 | δA∞ | δD∞ | t1 | t∞ | notes |
| --- | --- | --- | --- | --- | --- | --- |
| paper, Table 1 (their engine) | **−2.70%** | **−6.39%** | −4.19% | 28.3 s | 66.2 s | paper's own 39-design average: −1.21% / −2.81%, 17.6 s / 47.2 s |
| ours, `legacy` + NDR | −0.15% | −0.16% | −0.57% | 112 s | 225 s | no delay regression anywhere |
| ours, `legacy` + slack 0.05 | −0.19% | −0.21% | −0.58% | 115 s | 226 s | 0 timeouts |
| ours, `staged` + slack 0.05 | −0.23% | −0.28% (12 designs) | −1.08% | 304 s | 692 s | **4 timeouts** (`sqrt`, `mem_ctrl`, `ethernet`, `RISC`) |
| ours, `legacy` + DSE | **−4.96%** | **−4.97%** | −1.32% | 111 s | 243 s | 7/16 designs slower |
| ours, `legacy` + DSE + NDR | −3.23% | **−3.25%** | **−6.87%** | 109 s | 228 s | no design slower, 0 timeouts |

(Runtimes are the sweep's wall clock and include the ~45 s database parse per invocation — see
"Runtime" below; the sweep scripts are the ones described above and every completed run is `cec=1/1/1`.)

**Our post-mapping resubstitution is an order of magnitude behind theirs on this netlist.** Per
design, with the staged flow and the paper's own initial netlist:

| design | paper δA1 | paper δA∞ | ours δA1 | ours δA∞ | paper δD∞ | ours δD∞ |
| --- | --- | --- | --- | --- | --- | --- |
| arbiter | −12.01% | **−51.81%** | +0.00% | **+0.00%** | −39.57% | +0.00% |
| mem_ctrl | −5.86% | −12.79% | −0.31% | −0.37% | −13.00% | −1.32% |
| div | −7.81% | −9.30% | −0.01% | −0.03% | +0.01% | +0.02% |
| sqrt | −3.16% | −5.22% | −0.05% | −0.12% | −1.21% | −0.22% |
| ethernet | −0.96% | −3.68% | −0.02% | −0.03% | +0.00% | −0.14% |
| systemcaes | −3.25% | −3.62% | −1.10% | −1.10% | +0.00% | −0.61% |
| systemcdes | −1.21% | −2.38% | −0.16% | −0.16% | +0.00% | −0.85% |
| spi | −1.52% | −2.08% | −0.01% | −0.01% | +0.00% | −2.52% |
| iwls05_i2c | −1.46% | −2.00% | −0.06% | −0.06% | +0.00% | +0.00% |
| aes_core | −1.07% | −1.81% | −0.31% | −0.36% | −2.05% | +0.00% |
| tv80 | −0.70% | −1.48% | −0.17% | −0.22% | +0.00% | +0.00% |
| usb_phy | −0.83% | −1.42% | +0.00% | +0.00% | −8.51% | +0.00% |
| simple_spi | −0.53% | −1.39% | −0.16% | −0.16% | +0.00% | +0.00% |
| sasc | −1.07% | −1.32% | −0.40% | −0.40% | +0.00% | +0.00% |
| RISC | −1.03% | −1.03% | −0.14% | −0.17% | +0.00% | −4.26% |
| usb_funct | −0.70% | −0.98% | −0.10% | −0.22% | −2.65% | +0.55% |

The single largest discrepancy is `arbiter`: their engine reports a **51.81%** area reduction (and
39.57% delay reduction) on a circuit whose baseline we reproduce to 0.02% in area and 0.01% in delay,
while our engine makes **zero** improving substitutions there. On `arbiter`, the single-cell and
four-input multi-cell stages answer 24 897 cuts, the MFFC-cut e-graph stage is reached 905 times and
the dependency-cut stage produces 137 candidates — none accepted — and even
`EPMAPPING_PRESERVE_DEPTH=0 PRESERVE_DELAY=0` (pure area) leaves the netlist at 557.72. Their 51.81%
therefore comes from candidate sub-networks our sources do not generate at all.

**Post-mapping comparison on the `A_e` initial netlist (the current default).** The same sweep re-run
with `EPMAPPING_BASELINE=high_effort` (the paper's optimized subject graph; the presets above), no
DSE, `EPMAPPING_NO_DELAY_REGRESSION=1`, `EPMAPPING_ABC_ITERS=20`, 16 designs, `EPM_JOBS=2
RAYON_NUM_THREADS=4` for both flows. `t_opt` subtracts the initial-netlist generation time measured
separately (`EPMAPPING_SKIP=1`), because each sweep invocation regenerates the subject graph:

| configuration | δA1 | δA∞ | δD1 | δD∞ | t1 | t∞ | t_opt1 | t_opt∞ | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper, Table 1 (their engine) | **−2.70%** | **−6.39%** | — | −4.19% | 28.3 s | 66.2 s | — | — | their own 39-design average: −1.21% / −2.81% |
| ours, `legacy` (16 designs) | −0.13% | −0.15% | −0.66% | −0.67% | 160 s | 320 s | 67 s | 228 s | `mem_ctrl` rN measured in a separate isolated run (1332 s) |
| ours, `staged` (9-design subset) | −0.14% | −0.15% | −0.88% | −0.88% | 155 s | 520 s | 100 s | 465 s | `aes_core`/`systemcdes`/`tv80` measured in isolated re-runs |
| ours, `legacy`, same 9 designs | −0.08% | −0.08% | −0.45% | −0.45% | 111 s | 150 s | 40 s | 79 s | paired against the `staged` row |

This removes the caveat that made the earlier comparison inexact: on **their own** `A_e` subject graph
— the netlist whose smaller size left our one-pass comparison measuring a different starting point —
our engine still finds only ~0.1% where theirs finds 2.7% after one pass and 6.4% at convergence, i.e.
a 20× area gap after one pass and >40× at convergence, at 2–5× the runtime (paper t∞ 66 s here, 47 s
on their 39-design average). Paired on the same nine designs, `staged` is ~1.9× better than `legacy`
(−0.15% vs −0.08%) but ~3.5× slower (t∞ 520 s vs 150 s), matching the ordering measured on the
one-pass graph.

Per design (`legacy` unless a `staged` line is shown):

| design | δA1 paper | δA∞ paper | ours δA1 | ours δA∞ | ours δD∞ | t1 | t∞ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arbiter | −12.01% | −51.81% | +0.00% | +0.00% | +0.00% | 73 s | 74 s |
| mem_ctrl | −5.86% | −12.79% | −0.22% | −0.37% | +0.00% | 403 s | 1332 s |
| systemcaes | −3.25% | −3.62% | −1.09% | −1.09% | −3.87% | 80 s | 86 s |
| sasc | −1.07% | −1.32% | −0.37% | −0.37% | +0.00% | 42 s | 45 s |
| RISC | −1.03% | −1.03% | −0.13% | −0.14% | −2.47% | 331 s | 716 s |
| tv80 | −0.70% | −1.48% | −0.13% | −0.15% | +0.00% | 58 s | 86 s |
| sqrt | −3.16% | −5.22% | −0.02% | −0.06% | −0.28% | 160 s | 736 s |
| iwls05_i2c | −1.46% | −2.00% | −0.06% | −0.06% | +0.00% | 71 s | 78 s |
| aes_core | −1.07% | −1.81% | −0.04% | −0.05% | +0.00% | 305 s | 369 s |
| simple_spi | −0.53% | −1.39% | −0.05% | −0.05% | +0.00% | 48 s | 65 s |
| div | −7.81% | −9.30% | −0.02% | −0.02% | −0.08% | 121 s | 267 s |
| systemcdes | −1.21% | −2.38% | −0.02% | −0.02% | +0.00% | 250 s | 419 s |
| spi | −1.52% | −2.08% | −0.01% | −0.01% | −4.04% | 110 s | 175 s |
| ethernet | −0.96% | −3.68% | −0.00% | −0.00% | +0.00% | 413 s | 584 s |
| usb_funct | −0.70% | −0.98% | +0.00% | +0.00% | +0.00% | 60 s | 56 s |
| usb_phy | −0.83% | −1.42% | +0.00% | +0.00% | +0.00% | 38 s | 40 s |
| *staged* aes_core | | | −0.16% | −0.16% | +0.00% | 397 s | 1551 s |
| *staged* tv80 | | | −0.31% | −0.41% | +0.00% | 108 s | 427 s |
| *staged* spi | | | −0.18% | −0.18% | −4.04% | 175 s | 886 s |
| *staged* iwls05_i2c | | | −0.08% | −0.08% | +0.00% | 61 s | 120 s |
| *staged* sasc | | | −0.37% | −0.37% | +0.00% | 47 s | 54 s |
| *staged* usb_phy | | | −0.04% | −0.09% | −3.89% | 53 s | 85 s |
| *staged* simple_spi | | | −0.05% | −0.05% | +0.00% | 48 s | 73 s |
| *staged* systemcdes | | | −0.03% | −0.03% | +0.00% | 224 s | 1118 s |
| *staged* arbiter | | | +0.00% | +0.00% | +0.00% | 278 s | 367 s |

Every run in the two tables is `cec=1`. Note the runtime asymmetry the subject graph introduces:
`mem_ctrl` needs 190 s for one `legacy` pass on the one-pass §4.1 graph and **403 s** alone on the
`A_e` graph — the `fraig`-collapsed structure has more reconvergent cuts, and therefore more
dependency-cut window simulation, for the same netlist size. The paper's `t∞` (66 s over 16 designs) is
its resubstitution time on that same `A_e` graph, so the runtime column is comparable.

The runtime columns above were measured with **two sweeps running concurrently** (each `EPM_JOBS=2`
with `RAYON_NUM_THREADS=4`), so they sit roughly 2× above a lone-sweep measurement: `mem_ctrl` `legacy`
r1 took 881 s inside the concurrent sweep and 403 s when the same configuration was re-run alone. The
isolated number is the one reported for that design, and the same is true for the `staged`
`aes_core`/`systemcdes`/`tv80` rows (which also needed a 1800 s budget to converge: 1551/1118/427 s).
The area/delay columns are deterministic and unaffected, and both flows were measured under
the same concurrency, so the comparison between them and the 3–7× ratio to the paper hold; treat the
absolute seconds as an upper bound unless the row says it was isolated.

**The `syn4` DSE pipeline changes the conclusion on the large designs.** The same matched netlist with
the design-space portfolio extended by structural synthesis
(`EPMAPPING_DSE_PIPES="plain syn4"`, `EPMAPPING_DSE=1`, `EPMAPPING_NO_DELAY_REGRESSION=1`, `legacy`
flow, 1800 s budget):

| design | `A_e` | r1 (one pass) | rN (converged) | paper δA1 / δA∞ | CEC |
| --- | --- | --- | --- | --- | --- |
| arbiter | 557.72 | **272.37 (−51.16%)** | **272.30 (−51.20%)** | −12.01% / −51.81% | 1/1/1 |
| mem_ctrl | 1895.40 | **1409.23 (−25.65%)** | timeout (1800 s) | −5.86% / −12.79% | 1/1 |
| systemcaes | 483.75 | **379.35 (−21.59%)** | **378.96 (−21.66%)** | −3.25% / −3.62% | 1/1/1 |
| sasc | 27.28 | 27.18 (−0.37%) | 27.18 (−0.37%) | −1.07% / −1.32% | 1/1/1 |
| spi | 149.43 | 149.41 (−0.01%) | 149.41 (−0.01%) | −1.52% / −2.08% | 1/1/1 |

So on these five designs the portfolio averages **−19.8%** after one pass: `arbiter` matches the paper's
convergence on area with a far better delay (999.87 → 355.17, −64.5% against their −39.6%), `mem_ctrl`
beats their convergence gain in a *single* pass (and their Table 2 averages are −4.23%/−5.47%), and
`systemcaes` is 6× their gain. `sasc` and `spi` stay behind, which is where the local candidate set (and
hence the native synthesis source above) is the limit — the portfolio cannot invent structure that the
unmapped netlist does not offer. Reproduce with
`EPM_BIN=build-stgrel/epmapping/epmapping_check EPM_OUT=/tmp/dse EPMAPPING_DSE=1
EPMAPPING_NO_DELAY_REGRESSION=1 EPMAPPING_DSE_PIPES="plain syn4" ./scripts/_epm_one.sh <design> /tmp/dse 1800`.

**What does not close the gap** (measured on the matched netlist, `legacy` flow unless noted):
`EPMAPPING_NO_DELAY_REGRESSION` vs the default 5% local slack (−0.16% vs −0.21% — the strict rule
costs only 0.05 pp here, unlike on our own subject graph); `EPMAPPING_PRESERVE_DEPTH=0` (identical:
`iwls05_i2c` 48.09, `sasc` 27.31, `systemcaes` 481.90, `usb_phy` 22.91); and a **bigger database**. The
shipped 3-level database covers 29 256/65 536 four-input functions (44.6%, `avg_best_area=0.390`); a
4-level one generated with `abc -c "super -I 4 -L 4 -A 0.50 -T 900 -o asap7.genlib"` has 642 814
supergates and covers 31 034 (47.4%) with the *same* average best area (0.393), and on
`iwls05_i2c`/`systemcaes`/`sasc`/`usb_phy` its mapping results are **bit-identical** to the shipped
database. Their advantage is in the **candidate set** — an area-optimal mapped sub-network per 4-input
function, dependency cuts with window simulation and SAT-verified acceptance — not in the acceptance
rule, the cost model or the database size.

**The DSE loop is what makes our engine competitive.** With the paper's own initial netlist and one
design-space-exploration cycle, our engine reaches **δA1 = −4.96%, δA∞ = −4.97%** over the same 16
designs — comparable to their Table 2 average (−4.23% to convergence, −5.47% for their 5×5 cycle) and
clearly better than their Table 1 average of −2.81%. Some per-design results are strong: `mem_ctrl`
2061.12 → 1738.53 (−15.7%, delay −23.9%), `sqrt` 1164.26 → 1032.83 (−11.3%, −27.8%), `systemcaes`
−8.7%, `spi` −5.7%, `ethernet` −4.6%, `iwls05_i2c` −4.2%. But that variant picks the DSE netlist on
area alone, so **7 of 16 designs end up slower** (up to `RISC` +23.5%); adding
`EPMAPPING_NO_DELAY_REGRESSION=1` makes the harness keep the DSE result only when it is not slower,
which gives the honest no-regression number: **δA1 = −3.23%, δA∞ = −3.25%, δD∞ = −6.87%, no design
slower, no timeout** (the big wins survive — `mem_ctrl` −15.7%, `sqrt` −11.3%, `ethernet` −4.6%,
`systemcaes` −8.7%, `spi` −5.0% — while `arbiter`, `aes_core`, `simple_spi`, `tv80`, `usb_funct`,
`systemcdes` and `RISC` fall back to their direct result).

So the two engines have complementary profiles on this netlist: the paper's engine wins on **pure
post-mapping resubstitution** (−6.39% against our −0.21% without DSE), while our DSE loop brings the
total back into the same band (−3.25% with delay guaranteed non-increasing, −4.97% area-first) — but
the DSE is a different mechanism (unmap → re-optimize the AIG → remap), not a better resubstitution
engine, and 10 of the 16 designs have a matched-subject-graph baseline within 0.1–8% only.

**Runtime.** The 16 MB supergate database costs **up to ~45 s to parse per invocation when it is not in
the page cache** (45.0 s / 48.1 s on the first two `EPMAPPING_SKIP=1` runs, 0.35 s without it, and
0.27 s once cached), and that is included in our wall-clock numbers above. Net of the parse our
convergence pass averages ~181 s against their 66.2 s on the same designs, with a much larger spread
(`mem_ctrl` 619 s, `sqrt` 695 s, `RISC` 581 s against their 183 s / 51 s / 351 s). Main.cpp prints the
parse time separately (`super library load = ... s`) so the two can be compared; the `staged` flow
additionally times out at 1 800 s on `sqrt`, `mem_ctrl`, `ethernet` and `RISC` with this subject graph.

### Why the `staged` flow times out: the don't-care computation, not the e-graph

The staged flow is much slower per pass than `legacy` on the same netlist (`sqrt` 651 s vs 217 s,
`mem_ctrl` 936 s vs 221 s, `RISC` 1149 s vs 264 s), and with six rounds it blows the 1 800 s budget on
four designs. That is **not** the e-graph rewriting. `epmapping.hpp` now times every per-root source
and `main.cpp` prints it:

```
[i] epmapping staged_secs care=… stage1=… stage2=… egg=… dep_cut=… mffc=… care_cache=hits/lookups care_mismatch=… dc_cut=… dc_collect=… dc_sim=…
```

| one round, paper's initial netlist | wall | care | dep_cut | egg | stage1 | stage2 |
| --- | --- | --- | --- | --- | --- | --- |
| `sqrt` (6 884 e-graph calls) | 743 s | **445.7 s (60%)** | 78.9 s | **33.6 s (4.5%)** | 5.8 s | 5.4 s |
| `div` (3 786) | 498 s | **293.1 s (59%)** | 71.1 s | **5.4 s (1.1%)** | 3.6 s | 3.3 s |
| `arbiter` (905) | 84 s | **41.4 s (49%)** | 2.5 s | **0.58 s (0.7%)** | 0.70 s | 0.64 s |

`care` is `compute_care_mask`, which for **every cut** runs a reconvergence-driven window enumeration
and then simulates all `2^n_pis` assignments of the window inputs (up to 4 096) through the window. A
single e-graph saturation costs only 0.6–4.9 ms by contrast. Blocking the stage entirely
(`EPMAPPING_EGG_MIN_MFFC=1000000000`, verified `egg_calls=0`) removes 10 s of the 743 s `sqrt` pass and
**still times out at 1 800 s on `sqrt`, `mem_ctrl` and `RISC`**. The staged flow always computes the
cut don't cares (the window simulation; `EPMAPPING_SAT_DC=1` optionally sharpens them with SAT), which is what the remaining cost is.

**The lever is the window size, not a cache.** The window cap used to be hard-coded at 12 window
inputs; `EPMAPPING_DC_MAX_PIS` now exposes it (`sqrt`, one round, staged, paper's netlist):

| `EPMAPPING_DC_MAX_PIS` | care | wall | area | delay | cec |
| --- | --- | --- | --- | --- | --- |
| 12 (previous hard-coded value) | 408.4 s | — | 1164.26 → 1163.11 | 87244.9 → 87115.7 | 1 |
| 10 | 320.7 s | 530 s | 1164.26 → 1163.11 | 87244.9 → 87115.7 | 1 |
| **8** | **93.8 s** | **314 s** | 1164.26 → 1163.11 | 87244.9 → 87115.7 | 1 |

i.e. **4.4× less don't-care time with bit-identical results** on `sqrt`; on `ethernet` the same change
takes `care` from 453 s to 250 s and the area from 2542.86 to 2542.84 with `delay` improving to 631.19.
Giving up on a larger window is conservative — it reports *no* don't cares for that cut, which is
sound — so `EPMAPPING_DC_MAX_PIS=8` became the default when the staged flow did (pass `12` to
reproduce the numbers measured with the old hard-coded value).

**A care-mask cache by cut signature is exact but useless for this wall.** `compute_care_mask` is now
memoised per round, keyed by the leaf **set** (leaf indices sorted, with the minterm index permuted
back into the caller's leaf order): `EPMAPPING_CARE_CACHE=off|set` (default `set`) and
`EPMAPPING_CARE_VERIFY=1`, which recomputes on every hit and compares. The re-indexing has to permute
the **minterm index bits**, not the mask's own bits — an earlier version did the latter and produced
non-equivalent netlists (`cec=0`), which the verification mode and the ABC CEC both catch. With the
fix:

| design (one round) | cache | care | hits | verify mismatches | area |
| --- | --- | --- | --- | --- | --- |
| `div` | off | 168.5 s | — | — | 1192.73 → 1192.36 |
| `div` | `set` | 169.5 s | 30 219/132 744 (22.8%) | **0** | 1192.73 → 1192.36 |
| `sqrt` | off | 408.4 s | — | — | 1164.26 → 1163.11 |
| `sqrt` | `set` | 408.6 s | 47 931/198 582 (24.1%) | **0** | 1164.26 → 1163.11 |
| `ethernet` | `set` | 453.3 s | 22 358/105 808 (21.1%) | **0** | 2543.72 → 2542.86 |

The cache is exact — 100 508 hits, zero mismatches, bit-identical area/delay/`cec` to the uncached
run — but it saves **no** time: `care` is unchanged, so the repeated leaf sets are the *cheap* cuts,
while the expensive large-window computations have unique signatures and never hit. It stays on
because it is exact and effectively free, but the runtime fix is `EPMAPPING_DC_MAX_PIS`.

**Caching the simulation itself does not work, and the reason is fundamental.** The redundancy is
real: one round, `EPMAPPING_CARE_CACHE=off`:

| design | DC calls | same window PI set as any earlier call | same as the previous call | sim time, first time | sim time, repeat window |
| --- | --- | --- | --- | --- | --- |
| `div` | 132 744 | 103 755 (78%) | 87 511 (66%) | 9.6 s | **30.5 s** |
| `arbiter` | 72 201 | 54 949 (76%) | 39 401 (55%) | 4.5 s | **15.6 s** |

so 76–78% of the simulation time re-simulates an already-simulated window (window enumeration and
`collect_nodes` are negligible: 0.07 s and 0.03 s on `div`). Two implementations were measured and
both produce **non-equivalent netlists**:

* keyed by the window PI set, storing the simulation matrix of the call that filled it
  (0.2% hits — a different leaf set with the same PI set needs nodes outside that closure);
* stored as **per-node tables and extended on demand** (each missing node computed from its cached
  fanin tables down to the window's free inputs): 53–55% hits, but `div` ends at `cec=0` with
  **87 778 mask mismatches out of 72 973 hits** against a fresh recomputation.

Disabling the on-demand computation and reusing only tables a full simulation recorded still gives
**49 364 mismatches out of 45 151 hits** and `cec=0`, so the fault is the reuse, not the extension:
a node's *value* for a given free-input assignment is not preserved by this engine's substitutions.
A replacement is only required to match its cut on the cut's **care** minterms, so it may differ on
assignments that the window that produced the cached table can exercise. The per-cut care **mask** is
exact because it is the *reachable set* for a fixed leaf set (verified: 100 508 hits, 0 mismatches),
but simulated node values are not. The simulation cache was therefore removed; the diagnostics that
localised the cost (`dc_cut`/`dc_collect`/`dc_sim`, plus the `staged_secs` timers and the exact
care-mask cache) are kept; the recurrence counters themselves were removed with it.

**Reproduce** (current default initial netlist, `EPMAPPING_BASELINE=high_effort`):

```
EPM_OUT=/tmp/epmap_ae EPM_JOBS=2 RAYON_NUM_THREADS=4 EPMAPPING_DSE=0 \
  EPMAPPING_NO_DELAY_REGRESSION=1 \
  ./scripts/compare_with_paper.sh 1200 div sqrt arbiter mem_ctrl aes_core ethernet iwls05_i2c \
      RISC sasc simple_spi spi systemcaes systemcdes tv80 usb_funct usb_phy
scripts/compare_with_aspdac25.sh /tmp/epmap_ae/results.csv

# the historical table above was measured on the one-pass §4.1 subject graph; add
#   EPMAPPING_ABC_SCRIPT="fraig; rw; rs; rf; resyn2rs; compress2rs"
# (and the same options) to reproduce it.
```

Two incidental findings from these runs: the `staged library_cuts=` diagnostic used to be a **dead
counter** (`stats::staged_library_hits` was never incremented, so it always printed 0 — it now counts
the cuts where the single-cell or four-input multi-cell stage produced a cover at all, e.g. 24 897 on
`arbiter` and 12 157 on `iwls05_i2c`), and one legacy-flow `arbiter` run died with `SIGSEGV`
(`rc=139`) while the identical configuration succeeded in another run — noted here, not yet reproduced
on its own.

### Don't cares: window simulation by default, SAT as an opt-in sharpener

The don't-care computation runs **two** sources, in this order (see `fresh_care_mask`); only the first
runs by default (`EPMAPPING_SAT_DC=0`):

1. **window simulation** (`window_dont_cares`): the reconvergence window of the cut's leaves is
   simulated over all `2^n_pis` assignments of its free inputs (`EPMAPPING_DC_MAX_PIS=8`). It replaces
   the logic above the window with free variables, so it is **sound but approximate** -- a minterm it
   reports as *care* may be unreachable in the real circuit -- and it gives up entirely (reporting *no*
   don't cares) when the window is wider than the cap.
2. **SAT refinement** (`sat_care_refine`, opt-in via `EPMAPPING_SAT_DC=1`): the fanin cone of the cut's leaves is
   encoded as a CNF (constants fixed, primary inputs free, every gate a Tseitin relation over its pin
   function, built with `bill`/ABC's `bsat2`) and the refinement clears every minterm of the window's
   care mask that the cone cannot produce. An UNSAT partial assignment clears a whole sub-cube per
   call; only bits are ever *cleared* and every cleared bit is proved unreachable, so hitting
   `EPMAPPING_SAT_DC_CALLS` (or a cone larger than `EPMAPPING_SAT_DC_MAX_NODES`) just leaves the
   window mask in place -- still sound, only less sharp.

**How the window is simulated (mockturtle's method).** `satisfiability_dont_cares` / `rewrite`
compute the care set with `window_view` + `simulate_nodes`: every node of the window gets a
`kitty` truth table over the window's free inputs, i.e. all `2^n_pis` patterns are evaluated **64 at a
time in one word**, and the care set is then built by enumerating the patterns once and reading the cut
leaves' tables. This implementation does the same without `window_view` (which maps only the *false*
constant of `block_network`, so mockturtle's `simulate_nodes` cannot be used directly on it): window
inputs become `kitty::create_nth_var` tables, constants become all-0/all-1, and each gate's table is
the OR over the minterms of its pin function of the AND of the (possibly complemented) pin tables --
pin `k` of the genlib function is variable `k` (LSB-first), matching the netlist. The care set is the
set of leaf-value combinations that occur; the returned don't-care table is its complement.

Two fixes were measured on the way, both `cec=1` and area/delay-identical on the six designs below:

| design | `window_dont_cares` (with the old per-cut `fanout_view` + per-pattern loop) | after removing the `fanout_view` | after bit-parallel simulation |
| --- | --- | --- | --- |
| `aes_core` | 97.4 s (of a 100.4 s `care`) | 3.7 s | **1.0 s** |
| `div` | (untimed; `care` 133 s) | 8.1 s | **1.1 s** |
| `arbiter` | 10.3 s | 1.5 s | **0.3 s** |
| `tv80` | -- | 1.7 s | **0.2 s** |
| `spi` | -- | 0.7 s | **0.12 s** |
| `sasc` | -- | 0.04 s | **0.010 s** |

(the same runs: `dc_sim` 1.20 -> 0.07 s on `arbiter`, 3.26 -> 0.40 s on `aes_core`, 7.11 -> 0.46 s on
`div`). The window **sizing** was also aligned to mockturtle's `max_tfi_inputs` idea for a test
(cutting the reconvergence window at `EPMAPPING_DC_MAX_PIS` instead of letting it grow to `|C| + 4` and
then discarding over-wide windows): that gives more cuts a don't-care set (`spi` 3 657 -> 4 640,
`tv80` 11 636 -> 12 726, `aes_core` 37 970 -> 38 365) but costs `spi` **0.17% area** (149.10 -> 149.35)
and changes nothing elsewhere, so the growing window + discard stayed.

Measured (asap7, area-first, staged flow, `.super` database):

| configuration | area | don't-care work | cuts with DC | CEC |
| --- | --- | --- | --- | --- |
| `EPMAPPING_SAT_DC=0` (window only, `DC_MAX_PIS=8`) | 27.17 (`sasc`) | 0.35 s care (0.18 s simulation) | 901 | 1 |
| `EPMAPPING_SAT_DC=1` (SAT on; opt-in since 2026-09) | 27.17 (`sasc`) | 0.41 s care (0.20 s SAT: 3092 cuts, 190 540 calls, 7205 minterms cleared) | 937 | 1 |
| `EPMAPPING_SAT_DC=1` on `arbiter` (`A_e`) | 557.72 | 69.9 s SAT vs 83.4 s window (72 200 cuts, 6.66 M calls, 597 k cleared) | -- | 1 |
| `EPMAPPING_DC_MAX_PIS=0/2/8` with `SAT_DC=1` | identical on `sasc` | same 3092 cuts / 190 540 calls / 7205 cleared | -- | 1 |

The last row is the point: **the window mask is only the starting point of the refinement, so as long
as SAT completes the cap does not change the result** -- the SAT path finds *more* cuts with don't
cares (937 vs 901 on `sasc`) at comparable cost, and on the design where the cap used to bind it is
*cheaper* (70 s vs 83 s on `arbiter`). The `2^n_pis` wall that caused the `sqrt`/`mem_ctrl`/`ethernet`
timeouts is therefore capped (`DC_MAX_PIS=8`), made cheap (bit-parallel simulation) *and* sharpened by
SAT. A wrong CNF polarity in the first implementation was caught by the end-to-end CEC gate (`cec=0`,
exit 3) instead of being shipped, which is why every run here asserts `cec=1`.

**Which one is faster: the window, by one to two orders of magnitude.** Measured on `A_e`, one round,
idle machine, current defaults (`SAT_DC_CALLS=128`), `RAYON_NUM_THREADS=4` -- the engine's own timers in
seconds (`dc_window` = the whole window call, `dc_sim` = the bit-parallel simulation inside it,
`dc_cone` = the per-cut fanin-cone construction inside the SAT refinement, which runs before its
solver timer):

| design | `dc_window` | of which `dc_sim` | `dc_cone` | SAT solver | `care` total |
| --- | --- | --- | --- | --- | --- |
| `arbiter` | **0.29** | 0.07 | 1.16 | 64.7 | ~75-85 |
| `div` | **1.12** | 0.46 | 20.0 | 5.62 | ~58 |
| `aes_core` | **1.00** | 0.40 | 0.39 | 4.37 | ~11 |
| `tv80` | **0.20** | 0.07 | 0.27 | 13.9 | ~16 |
| `spi` | **0.12** | 0.04 | 0.12 | 3.20 | ~5 |
| `sasc` | **0.010** | 0.003 | 0.001 | 0.04 | 0.10 |

So the two are **not alternatives**: the window is the cheap first pass (its `2^n_pis` simulation over a
handful of gates), SAT is the expensive sharpener whose *cone construction* can itself dominate (`div`:
20 s of cone against 5.6 s of solving). Consequences for the knobs:

* `EPMAPPING_SAT_DC_CALLS=128` (the default; `0` = unlimited): caps the SAT work per cut. **No cut of
  the six benchmarks above needs more** (`budget_hits=0`, call counts identical to 256), while a cap of
  64 binds on 43 602 `arbiter` cuts and 6 557 `tv80` cuts and costs `tv80` **0.015% area**
  (338.63 -> 338.68). Use 64 only when that trade is acceptable.
* `EPMAPPING_SAT_DC=0` is the window-only path and **the default since 2026-09**, for the reasons above:
  on `sasc` it gives the same area and delay as the full pipeline (901 vs 937 cuts with don't cares,
  `care` 0.35 s vs 0.41 s -- see the table above). It is *not* free in general: on `arbiter` it leaves 512
  of 72 330 cuts with don't cares (against 42 752 with SAT) and on `div`/`aes_core` it costs
  +0.003-0.004% area. Set `EPMAPPING_SAT_DC=1` to buy those back at the runtime cost above.
* `EPMAPPING_DC_TIME_BUDGET` bounds the *whole* don't-care computation (window + cone + SAT) in
  wall-clock seconds; cuts after the budget are matched exactly. That is the only knob that also
  bounds the cone construction.

### Whole-window synthesis over the MFFW (`EPMAPPING_WINDOW_SYN`, on by default)


The single-output window re-coverage measured above (0–1 wins) fails for a structural reason: a
reconvergence window's internal nodes keep external fanouts, so re-implementing only the root does not
remove them. `window_synthesize` fixes that by working on the **MFFW** (`build_mffw`: fanout-free, so
every internal node is removable) and rebuilding the window **as a whole**:

1. every window output is re-synthesized from the window inputs (SOP factoring / DSD, `EPMAPPING_SYN_MODE`),
   all of them into **one AIG** — `aig_network::create_and` hash-conses, so structures shared by several
   outputs exist once;
2. that shared multi-output AIG is mapped **jointly** with the engine's technology library (`emap`);
3. the cover is materialized first (unreachable) and the window is committed only if the *measured*
   whole-window area change is a win and the global critical path does not grow, while every output keeps
   its level and arrival (window outputs are observable, so they must be exactly equivalent — no don't
   cares are used here); all outputs are then substituted at once (`build_window_cover` materializes the
   shared cover once and `window_synthesize` rewires every output), which is what makes the internal
   nodes die. A window that does not get smaller is abandoned (its fresh nodes are marked dead).

Because a commit rewires **every** window output (not just the root), those substitutions are counted
into `replacements` -- the counter `run()` uses to decide whether a round made *any* progress -- and
reported as `window=` in the `replacements=... (..., window=...)` breakdown. Before that, a round whose
only wins came from this fallback looked empty and stopped the iteration one round early.

The stage is a **fallback**: it is tried only for a root whose cut stages produced nothing. A committed
window changes the network the same pass's later roots see, so applying it first takes away improvements
the cut stages would have found (measured on `c5315`: window first −0.44%, cut stages alone −0.78%,
window as fallback −1.14%); as a fallback it can only add. Measured (`staged`, `.super` database,
`EPMAPPING_SYN=1 EPMAPPING_SYN_MIN_LEAVES=4 EPMAPPING_SYN_MODE=3`, all `cec=1`):

| design | base | cut synthesis only | *window first (removed)* | window as **fallback** (the flow) |
| --- | --- | --- | --- | --- |
| c432 | 6.61 (0.00%) | 6.57 (−0.61%) | — | **6.57 (−0.61%)** (0 windows) |
| c3540 | 41.62 (−0.62%) | 41.25 (−1.50%) | 41.21 (−1.60%) | **41.21 (−1.60%)** (1 window + 21 cut wins) |
| c5315 | 64.11 (−0.16%) | 63.71 (−0.78%) | 63.93 (**−0.44%**) | **63.48 (−1.14%)** (10 windows + 21 cut wins) |

> **Pre-fix numbers.** This table was measured before the measured-commit fix
> (§The window fallback commits on the *measured* area change): the old commit also applied windows that
> did not really get smaller, so the fallback's contribution here is larger than it is in the fixed
> engine. On the paired check in that section (A_e, one pass, strict NDR, default `SYN_MIN_LEAVES`) the
> fallback is worth 0.02–0.08 µm² on three of eight designs and is neutral on the rest, and the
> "window first" column came from a code path that no longer exists.

So, pre-fix, the window was worth **0.1–0.4 pp** on top of the cut stages as a fallback
(`c5315` −0.78% → −1.14%), and cost ~0.2–0.7 s per design (all windows are built and priced, only the
profitable ones are committed). Window-first was measurably worse on `c5315` because the committed
window rewires the network before the later roots are processed.

### Structural synthesis as a candidate source (`EPMAPPING_SYN`, on by default)

> **Run-to-run determinism (fixed).** `mockturtle::sop_factoring` stores its parameters *by
> reference* (`sop_factoring_params const& _ps;`) while its constructor's default argument is `{}`,
> so a default-constructed `sop_factoring` leaves `_ps` bound to a temporary that died at the end of
> the constructor. Every read of `_ps.try_both_polarities` was therefore a read of stack garbage
> (UBSan: `load of value 64, which is not a valid value for type 'bool'`), which made the
> synthesized covers — and with them the staged reject counters, and occasionally the final area —
> differ between runs of the *same* binary. `build_support_function` now passes a
> static-storage-duration `sop_factoring_params` object, so the reference stays valid; the same
> inputs now give bit-identical results in every run. Upstream fix would be to make
> `_ps` a value member.

The library stages answer a cut with a cover *picked from the library* (single cell, four-input
sub-network, DB, e-graph). The synthesis stage instead **builds a new multi-level network for the cut
function** — SOP factoring (ABC `&sopb`-style) or disjoint-support decomposition with factoring on the
prime nodes (`&dsd`-style) — with the cut's don't cares as flexibility (`sop_factoring`'s DC overload
and the DSD primes), maps it with the engine's own technology library (`emap`, area-oriented) and hands
the resulting cover to the normal acceptance gate as source 10. A synthesis bug can therefore only
degrade to "no candidate": `cover_from_mapped` re-costs the cover through `finalize_local_cover` and
re-checks it against the cone truth table on the care minterms before it is considered.

This is the candidate class the engine was missing (the ABC `syn4`/`sopb` DSE pipelines in
`scripts/_epm_one.sh` are the same mechanism at whole-design scope). Measured (staged flow, `.super`
database, `EPMAPPING_SYN=1 EPMAPPING_SYN_MIN_LEAVES=4`, `EPMAPPING_SYN_MODE=3` so both synthesis modes
are offered and the best is kept):

| design | default | + synthesis | wins | synthesis cost | CEC |
| --- | --- | --- | --- | --- | --- |
| c432 | 6.61 (0.00%) | **6.57 (−0.61%)** | 2 | 1143 roots, 0.22 s | 1 |
| c3540 | 41.62 (−0.62%) | **41.25 (−1.50%)** | 24 (21 SOP + 3 DSD) | 8126 roots, 2.7 s | 1 |
| c499 | 22.83 | 22.83 | 0 | 1696 roots, 0.92 s | 1 |
| c1355 | 22.80 (−0.09%) | 22.80 | 0 | 2869 roots, 1.68 s | 1 |
| c2670 | 26.65 (−0.30%) | **26.64 (−0.34%)** | 1 (SOP) / 2 (DSD) | 2615 roots, 0.58 s | 1 |
| c5315 | 64.11 (−0.16%) | **63.64 (−0.89%)** | 6 SOP / **19 DSD** | 11 596 roots, 3.9 s | 1 |
| sasc | 27.17 (−0.40%) | 27.17 | 0 | 2401 roots, 0.42 s | 1 |

So the stage is cheap (a few seconds of covers even on the largest of these) and it wins where the local
candidate set was weak: `c3540` more than doubles the flow's gain (−0.62% → −1.50%, 24 wins), `c5315`
goes −0.16% → −0.89% (19 DSD wins) and `c432` goes from **0%** to −0.61%. Which mode wins is design
dependent (`c3540`/`c432` prefer SOP factoring, `c5315`/`c2670` DSD), which is why `EPMAPPING_SYN_MODE=3`
offers both and lets the acceptance gate choose. On the ASPDAC'25 `arbiter` (`A_e`, 66 057 synthesis roots, 132 114 covers, 17.4 s) it
wins nothing: the per-cone scope is too small to reach the paper's `arbiter`/`mem_ctrl` numbers — those need the *global* restructuring, which is
what the `syn4` pipeline in the DSE portfolio provides (`arbiter` 557.72 → 272.37, see the matched
comparison above). Extending the same synthesizer to a reconvergence/MFFC **window** (multi-output, with
the window's internal nodes rebuilt) is the natural next step; the machinery for window rebuilding
already exists (`EPMAPPING_WINDOW_SYN`).

### Fixed: the `cut_mapping_library_dc` fast path had become a full cover search

Re-measuring the legacy flow on the `A_e` initial netlist exposed an **8× runtime regression** with no
area benefit, introduced when the `map_egg` FFI layer was reduced to a single entry point:

* `cut_mapping_library_dc` — called from C++ as the *"single-cell matcher"* (`call_small`, and the
  the legacy flow's `call_small`) — had been rewired to the full cover extraction
  (`map_cut_to_cover_cover_only`). A full cover exists for almost every cut, so `call_small` started
  "succeeding" on **every** cut, including the ~100 k cuts with more leaves than the library's largest
  cell: `div` went from `small_successes=11 876`/`large_successes=327` (the old semantics — large cuts
  almost never match a single cell) to `small_successes=32 233`/`large_successes=100 702`, i.e. the cut
  counts themselves.
* It also made `library_hits` dead (the counter was never incremented) and
  `supergate_db_entries()` a hard-coded `0` stub, so the diagnostics *looked* alive while reporting
  nothing. `db_hits`/`supergate_db entries` have been removed from the output; `library_hits` and
  `egg_calls` now count real calls.

The fix routes `cut_mapping_library_dc` back to `map_tt_to_library` (library only, no cover
search), which restores the old candidate set. Measured on `div`, one legacy round on the ASPDAC'25
§4.1 subject graph, `RAYON_NUM_THREADS=4`: **909 s → 111 s**, `small_successes=11 830`,
`large_successes=327`, `library_hits=12 157` (the "before" run shared the machine with three other
diagnostic runs and the "after" run did not, but the mechanism is not a scheduling artefact: the
number of cuts for which a candidate is built dropped by 2.7× on small cuts and 300× on large ones).
The `staged` flow never calls this entry point (`library_hits=0`, unchanged results), which is why the
staged tables are unaffected.

### The legacy DC entry points honour the `care_mask` now (and keep their stage contracts)

The four bridge functions that carry a `care_mask` used to drop it. They now share one implementation,
`run_extract_library_dc` (`../map_egg/src/lib.rs`):

* **The mask only adds candidates.** A completion copies the care minterms from the cut function and
  picks arbitrary values on the free ones, so every returned cell agrees with the original on all care
  minterms. The engine's masks are an *under*-approximation of the true don't-care set (an exhaustive
  window simulation, see `compute_care_mask`), so no returned mapping can become invalid.
* **Bounded enumeration.** At most `MAX_COMPLETIONS = 12` free minterms are enumerated (`2^12` library
  probes worst case); beyond that the exact function -- itself a legal completion -- answers the cut.
  The engine's own single-cell stage enumerates at most `dc_max_free = 4`, so a wider mask can only
  come from a direct FFI caller.
* **Only 64 minterms are addressable.** `compute_care_mask` never asks for don't cares on a cut wider
  than six leaves (`n < 2 || n > 6` returns the all-ones mask). Minterms a `u64` cannot address are
  now treated as *care* instead of being read through a wrapped shift: the old `care >> bit` panicked
  for `bit >= 64` in a debug build and aliased to `bit & 63` in a release one.
* **The fully specified case keeps each entry point's contract** (`ExactSearch`):
  `cut_mapping_library_dc` stays *library only* (exactly one cell, never a cover -- the "single-cell
  matcher" contract restored above), while `cut_mapping_with_objective_dc` keeps the cover extraction
  it always ran **and** adds the don't-care aware single cells, comparing both with the same
  `(area, delay)` / `(delay, area)` key.
* `cut_mapping_multi_dc` now returns the e-graph cover of the exact function *plus* the best DC-aware
  single cell, and finally honours `limit` (0 = unlimited); its unread `use_egg` argument is gone.
  `cut_mapping_db_dc` (the old `supergates.jsonl` database bridge, a hard-coded
  `failure_cost("database_unavailable")` since `map_egg` was split out) and `cut_mapping_egg_multi`
  were dropped from the bridge: the precomputed multi-cell database now lives in
  `epmapping/db/*.super`, loaded by the C++ side into `tech_library<9>`, so only that side can answer
  it -- a stub that carries a `care_mask` but can never return a cover only invites callers to
  misread "no database" as "no don't-care match".

None of this is on the staged flow's path (`cut_mapping_egg_arr` is the only bridge call the engine
makes; every staged run reports `library_hits=0`), so the staged tables are unaffected. Unit tests
cover the routing, the mask benefit (`c & !(a&b)`: 0.16 -> 0.09 once minterms 3 and 7 are free), the
never-worse property and the wide-cut shift guard.

### `map_egg` indexes the full technology library again (nine-input cells)

`techlib_store()` loaded the genlib with `max_inputs = 6`, and `finalize_cell` *silently* drops every
cell with more variables. `asap7.genlib` defines 47 gates, seven of which have seven or more pins
(`AO322x2`, `OA331x2`, `AO331x2`, `AO332x2`, `OA332x2`, `AO333x1`, `OA333x2`) -- exactly the cells that
turn a wide cut into a single instantiation, and exactly the cells the C++ side could already use,
because it reads the same file with `tech_library<9>`. The bound is now nine, matching
`EmapParams::max_inputs` (9), `EMAPPER_MAX_INPUTS` (9) and `finalize_cell`'s hard ceiling.

Measured (one round, `EPMAPPING_CUT_SIZE=9`, on the subject graph `(a&b&c)|(d&e&f)|(g&h&i)`): one
nine-input `AO333x1` (area 0.17, delay 37.11) instead of three cells (area 0.33, delay 51.07). On
`c1355` and `c2670` the staged flow is **unchanged** (`c2670`: `26.73 -> 26.52`, delay
`307.96 -> 313.95`, 35 replacements), because the C++ `tech_library<9>` already supplied those cells:
with `EPMAPPING_EGG=0` the wide cone still maps to the single nine-input cell. The fix removes the
asymmetry between the two mappers; it does not add a capability the engine lacked.

### ISCAS subset (the paper's Table II designs)

`scripts/compare_with_paper.sh` is run over the ISCAS designs the paper's Table II reports, with the same
engine configuration and the **IWLS'24** subject graph
(`EPMAPPING_ABC_SCRIPT="fraig; compress2rs; resyn2rs; compress2rs"`). `paper ∆Amatch` is the paper's
database-matching result for that design.

| design | base area | δA1e | δA∞e | paper ∆Amatch | cec | t∞ (s) |
| --- | --- | --- | --- | --- | --- | --- |
| c1355 | 22.76 | +0.00% | +0.00% | — | 1/1/1 | 72 |
| c499 | 22.79 | +0.00% | +0.00% | — | 1/1/1 | 71 |
| c432 | 6.61 | +0.00% | +0.00% | −2.24% | 1/1/1 | 74 |
| c1908 | 16.03 | -0.12% | -0.12% | −3.60% | 1/1/1 | 72 |
| c880 | 15.75 | -0.44% | -0.44% | — | 1/1/1 | 79 |
| c3540 | 43.28 | -0.30% | -0.30% | −1.75% | 1/1/1 | 63 |
| c2670 | 26.86 | -0.07% | -0.30% | — | 1/1/1 | 61 |
| c5315 | 64.64 | -0.09% | -0.09% | −0.48% | 1/1/1 | 66 |
| c7552 | 80.37 | -0.77% | -1.88% | −0.76% | 1/1/1 | 62 |
| c6288 | 132.60 | +0.00% | +0.00% | −0.25% | 1/1/1 | 60 |
| **avg (10)** | | **-0.18%** | **-0.31%** | — | | |

## DATE'27 re-measurement (engine of 2026-09-21, binary `dc6f3436…`)

The tables in this section were measured with the engine built on 2026-09-21 (SHA-256
`dc6f3436…`). The engine has since gained incremental fan-out/cut/timing maintenance, so re-run
`scripts/experiments/run_all.sh` before quoting these numbers for a newer revision; the raw logs of
that measurement were kept under `/tmp` and are no longer available.

Every number below was re-measured on this tree following the arrangement of the DATE'27 paper that
ships in `ePMapping_paper/` (`DATE27-ePMapping-sub/doc/exp.tex`):

* subject graph `EPMAPPING_BASELINE=high_effort` (A_e) unless a section says otherwise;
* area-first objective with the **strict** rule "reduce area without worsening any arrival"
  (`EPMAPPING_NO_DELAY_REGRESSION=1`, i.e. `delay_slack=0`);
* driver DSE **off** (`EPMAPPING_DSE=0`), shipped three-level `a <= 0.50` database
  (`epmapping/db/asap7_4in_3lvl_a050.super`), `RAYON_NUM_THREADS=4`, 1800 s per configuration;
* `EPMAPPING_DC_TIME_BUDGET=60` for the *complete* tables: without it the don't-care stage alone
  exceeds the budget on `sin` (879 s of `care`, 876 s of it SAT) and `log2`, so those designs cannot
  converge (see "The don't-care stage" below). Where the budget does not bind it is bit-preserving:
  `adder` (57.38) and `bar` (127.14) are identical with and without it.
* the whole-window (MFFW) rewrite is **off** in the paper's stages: `EPM_BASE_ENV` sets
  `EPMAPPING_WINDOW_SYN=0`, because the DATE'27 paper does not study that stage. The stages that do
  study it set it back to `1` explicitly (`oracle`'s window arm and the `ablation` arms).
  **The tables in this section were measured before that change, i.e. with the window on**, so their
  engine-side numbers are optimistic and must be re-measured before they are quoted: on the `tab:mfs3`
  `A_i` start the window turns `aes_core` -0.56 % into **-0.28 %** and `mem_ctrl` -1.63 % into
  **-0.88 %** when it is switched off.

The `tab:main` subsection also measures the third starting point of §Setup, `EPMAPPING_BASELINE=direct`
(A_i, no ABC preprocessing). Each table states its own subject graph; the deltas of different graphs
are not comparable, because the A_e and single-pass baselines already contain the ABC work.

### 口径 differences to watch

1. **`tab:main` is the authors' window-off run; the `A_e` table of this section is window-on.** The
   paper's `Ours` column was measured with `EPMAPPING_WINDOW_SYN=0` (the paper口径) and, per the older
   note here, with one DSE cycle and a twelve-round cap, so it is *not* the table below. The two agree
   once the window's contribution is added back to the column here: `arbiter`, `div`, `sasc`, `spi` and
   `usb_phy` match exactly (e.g. `spi` 149.24 + 0.11 = 149.35, `usb_phy` 22.45 + 0.09 = 22.54), and
   every other design is within 0.3 um^2 except the four whose runs differ (`mem_ctrl` +19.1,
   `RISC` +11.3, `ethernet` +1.1, `usb_funct` +3.7). Quote the paper's own numbers for `tab:main`;
   re-measure this section's tables with `EPMAPPING_WINDOW_SYN=0` only when they are to be compared
   with it directly.
2. **`tab:db`'s c7552 column.** Its 80.33 -> 78.91 was measured *with dependency cuts* (the older
   README labels that column that way); this engine deliberately does not form them. Without them the
   database does not move `c7552` at all in the A_e口径 (81.68-81.71 over the whole coverage range
   0-77 %), and only -0.14 pp on the IWLS graph. The coverage column itself reproduces exactly
   (24 % / 34 % / 45 % for 2-level / a<=0.45 / a<=0.50).

### The don't-care stage is the runtime wall (one pass, A_e)

| design | `care` (s) | of which SAT | SAT calls | window calls | `cut_enum` | `egg` | `SYN` | engine total | wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `adder` | 0.31 | 0.27 | 31 667 | 256 | 0.000 | 0.51 | 0.00 | 0.8 | 38 |
| `bar` | 0.40 | 0.24 | — | 48 | 0.000 | 0.44 | 0.58* | 1.0 | 42 |
| `cavlc` | 0.34 | 0.19 | — | 107 | 0.000 | 1.16 | 0.00 | 1.7 | 82 |
| `max` | 0.87 | 0.09 | — | 616 | 0.000 | 0.68 | 1.72 | 1.7 | 44 |
| `div` | **91.4** | 7.0 | — | 3 686 | 2.48 | 4.72 | 2.24 | 105.7 | 373 |
| `sin` | **879.5** | **876.0** | 779 139 | 1 519 | 2.48 | 3.24 | 45 roots | 887.0 | 963 |

\* `bar`'s synthesis time is in `secs_synth`; the staged `staged_secs` line reports it separately.

* Only `sin` is SAT-bound (876 of 879 s, 779 139 calls, `budget_hits=0`); `div`'s 91 s is mostly the
  window/cone construction (SAT is 7 s). This is the same asymmetry the paper reports, but the absolute
  cost is what makes the *no-budget* sweep time out: with `EPMAPPING_DC_TIME_BUDGET=60`, `sin` converges
  in 307 s (4 rounds, 274.76, -0.57 %) against -0.64 % after one unbounded pass that then cannot finish.
* `EPMAPPING_SAT_DC_CALLS=128` never binds here (`budget_hits=0` everywhere measured).

### Reproducibility and the intermittent crash

* Same configuration, two runs: `c3540` (A_e, SYN on, EGG/WINDOW off, one pass) gives identical
  area/delay/`cec` **and** identical counters (`replacements`, `staged library_cuts`, `synth roots`,
  `window_synth windows`, `rounds`) once timing fields are stripped - the quantized-key argument holds.
* The engine segfaults intermittently (~5 % of runs), always immediately after the `super library:`
  line and before any further output, i.e. inside `read_aiger`/the library setup; it is not a race in
  `tech_library`/`super_utils` (both are single-threaded) and does not reproduce on demand. A crashed
  run records no area, so `scripts/_epm_one.sh` now retries `rc=139` up to three attempts (a timeout,
  `rc=142`, is never retried: it is the measurement). The failure is unrelated to the crash reports in
  `~/Library/Logs/DiagnosticReports` from 2026-09-18/19 (those are `emap` arrival/required assertions
  and `cut<16>` overruns from earlier experiments).

### The window fallback commits on the *measured* area change (fixed 2026-09)

`window_synthesize` (L5, the MFFW fallback) used to decide from `removable - cover_area` and only then
commit, where `removable` is the sum of the MFFW's internal cell areas -- i.e. the estimate *assumed*
that every window node dies. It does not, for two reasons, and the commit is now two-phase: the cover
is materialized first (still unreachable), the **realized** area change is computed exactly, and the
window is rewired only when it is a win; otherwise the fresh nodes are abandoned (marked dead, and
`compute_area()` skips dead nodes, so they cost nothing).

* `build_mffw` expands through the cached `fanouts_` vector, which is rebuilt only once per pass, so
  after the first commit of a pass the window can contain nodes that are no longer in any output cone
  and survive the commit. The exact dying set is now computed with the `take_out_node` cascade fixpoint
  over a fanout adjacency inverted from the window's own fanin lists (`block_network` has no
  `foreach_fanout`).
* Structural hashing is **not** a factor, contrary to what this section claimed before the fix:
  `block_network` has no structural hashing, and the materialized area matched the charge on every
  measured commit (`dead=0.45 new=0.44 measured=0.01 actual=0.01` on `max`).

The commit is additionally gated on the **global critical path**. `arrivals_` is refreshed only once
per pass, so it is stale for the fanout cone of every commit and a committed window cannot be undone;
the gate re-propagates the whole network with the substituted outputs driven by the new cover
(`predict_worst_delay`). The per-output level/arrival rules inherited from the cut stages stay as the
local check.

Reproducer (`max`, A_i = `EPMAPPING_BASELINE=direct`, `EPMAPPING_ROUNDS=1`,
`EPMAPPING_OBJECTIVE=1`, NDR off, shipped defaults otherwise):

```
$ ... epmapping_check asap7.genlib max.aig out
# before the fix:  windows=403 applied=9 gain=0.25 actual=-1.69   area 139.76 -> 140.76 (+0.72 %)
# after the fix:   windows=353 applied=2 gain=0.02 actual=0.02   area 139.76 -> 139.23 (-0.38 %)
$ ... EPMAPPING_WINDOW_SYN=0 ...
                   area 139.76 -> 139.23 (-0.38 %)
```

(The previous version of this section quoted 140.91 / 139.38 for the same pair; that run also set
`EPMAPPING_NO_DELAY_REGRESSION=1`, where the fixed engine lands on 139.38 with or without the window.)

Paired validation (A_e = `high_effort`, one pass, strict NDR, engine defaults), `cec=1` everywhere:

| design | `WINDOW_SYN=1` | `WINDOW_SYN=0` | commits (gain = actual) |
| --- | --- | --- | --- |
| `max` | 134.91 | 134.91 | 0 |
| `c5315` | 64.03 | 64.05 | 1 (0.02) |
| `sasc` | 27.18 | 27.18 | 0 |
| `simple_spi` | 36.85 | 36.93 | 2 (0.08) |
| `tv80` | 339.25 | 339.22 | 1 (0.06) |
| `usb_phy` | 22.50 | 22.54 | 2 (0.04) |
| `systemcdes` | 130.70 | 130.70 | 0 |
| `iwls05_i2c` | 47.57 | 47.57 | 0 |

The fallback is a win on three designs and neutral on four. On `tv80` it is locally a 0.06 win but the
pass ends 0.03 higher -- path-dependence, i.e. a locally winning commit changes which cuts the remaining
roots see, not a forecasting failure. No window was abandoned (`abandoned=0`) or critical-path-rejected
(`delay_rejected=0`) in this sample: with the corrected `removable` the estimate equals the measurement
on every commit, so the measured gate is a safeguard (it fires for a window node that survives the
commit, e.g. a dangling node), while the active fix is that the bogus candidates no longer reach the
commit at all (on `max`, 9 commits become 2, and the 1.53 um^2 loss disappears).

**The sweep tables in this README have been re-measured with the fix** (same口径, binary
`dc6f3436…`), and the change is confined to the designs where the fallback had been committing bad
windows or where the corrected `removable` changes which window is chosen:

| design | sweep | before | after |
| --- | --- | --- | --- |
| `max` | A_e r1 / rN | 135.15 / 135.07 | **134.91 / 134.91** |
| `max` | single-pass r1 / rN | 137.14 / 137.06 | **136.90 / 136.90** |
| `max` | A_i r1 = rN | 140.91 | **139.38** |
| `mem_ctrl` | single-pass r1 / rN | 2004.81 / 1988.54 | **2001.69 / 1974.41** |
| `mem_ctrl` | A_e rN | 1854.39 | **1853.68** |
| `sin` | A_e r1 / rN | 275.41 / 274.96 | **275.22 / 274.77** |
| `div` | A_e r1 / rN | 1184.21 / 1184.07 | **1184.16 / 1184.02** |
| `square` | A_e r1 / rN | 1062.10 / 1059.63 | **1062.04 / 1059.59** |
| `ethernet` | A_e r1 | 2540.82 | **2540.50** |
| `i2c` | single-pass rN | 56.63 | **56.49** |
| `c5315` | ISCAS, single-pass r1 | 64.60 | **64.53** |
| `c6288` | A_e r1 / rN | 132.42 / 132.07 | 132.44 / 132.09 (the only design that moves the other way) |

The means move by at most 0.04 pp (A_e 27 designs: δA₁ -0.78 -> -0.79 %, δA∞ -1.24 -> -1.25 %;
single-pass 27: -0.50/-0.76 -> -0.52/-0.80 %; ISCAS single-pass: -0.51/-0.63 -> -0.52/-0.62 %). Two
口径 bugs in the old tables were fixed at the same time: the A_i table had been measured *without* the
60 s don't-care budget (hence its five timeouts; with the budget only `div` remains), and `div`/`sqrt`
were shown as `--` there although the budget lets them run.

### tab:main (27 designs)

**A_e = `high_effort` (§Setup graph), strict NDR, DSE off, DC budget 60 s — the complete table (27/27 converge)**

| design | base (µm²) | r1 | δA₁ | rN | δA∞ | δD∞ | rounds | CEC b/1/N | t1 (s) | t∞ (s) | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adder | 57.40 | 57.38 | -0.03 | 57.38 | -0.03 | -0.30 | 2 | 1/1/1 | 69 | 69 | OK |
| bar | 131.20 | 129.46 | -1.33 | 127.14 | -3.09 | -2.99 | 3 | 1/1/1 | 71 | 71 | OK |
| max | 135.53 | 134.91 | -0.46 | 134.91 | -0.46 | -0.43 | 2 | 1/1/1 | 67 | 75 | OK |
| sin | 276.16 | 275.22 | -0.34 | 274.77 | -0.50 | -3.07 | 4 | 1/1/1 | 164 | 264 | OK |
| log2 | 1462.00 | 1459.59 | -0.16 | 1459.34 | -0.18 | -2.13 | 5 | 1/1/1 | 475 | 1030 | OK |
| div | 1184.50 | 1184.16 | -0.03 | 1184.02 | -0.04 | -0.39 | 3 | 1/1/1 | 255 | 428 | OK |
| cavlc | 33.07 | 33.00 | -0.21 | 33.00 | -0.21 | +0.00 | 2 | 1/1/1 | 64 | 64 | OK |
| multiplier | 1270.83 | 1269.04 | -0.14 | 1268.70 | -0.17 | -0.26 | 3 | 1/1/1 | 248 | 363 | OK |
| sqrt | 1157.38 | 1121.40 | -3.11 | 1105.72 | -4.46 | -5.92 | 6 | 1/1/1 | 282 | 747 | OK |
| ctrl | 5.86 | 5.84 | -0.34 | 5.72 | -2.39 | +0.00 | 5 | 1/1/1 | 58 | 60 | OK |
| dec | 27.06 | 27.06 | +0.00 | 27.06 | +0.00 | +0.00 | 1 | 1/1/1 | 58 | 55 | OK |
| arbiter | 557.72 | 557.72 | +0.00 | 557.72 | +0.00 | +0.00 | 1 | 1/1/1 | 148 | 148 | OK |
| int2float | 10.89 | 10.87 | -0.18 | 10.84 | -0.46 | +0.00 | 3 | 1/1/1 | 64 | 52 | OK |
| i2c | 55.77 | 55.50 | -0.48 | 55.22 | -0.99 | +0.00 | 6 | 1/1/1 | 66 | 73 | OK |
| priority | 27.51 | 27.36 | -0.55 | 27.36 | -0.55 | +0.00 | 2 | 1/1/1 | 68 | 69 | OK |
| router | 7.72 | 6.99 | -9.46 | 6.68 | -13.47 | -4.22 | 5 | 1/1/1 | 64 | 71 | OK |
| sasc | 27.28 | 27.18 | -0.37 | 27.18 | -0.37 | +0.00 | 6 | 1/1/1 | 69 | 76 | OK |
| iwls05_i2c | 47.66 | 47.57 | -0.19 | 47.51 | -0.31 | +0.00 | 6 | 1/1/1 | 68 | 82 | OK |
| simple_spi | 36.96 | 36.85 | -0.30 | 36.83 | -0.35 | -5.32 | 6 | 1/1/1 | 66 | 78 | OK |
| voter | 522.54 | 522.25 | -0.06 | 522.22 | -0.06 | -1.94 | 3 | 1/1/1 | 186 | 265 | OK |
| usb_phy | 22.54 | 22.50 | -0.18 | 22.45 | -0.40 | -2.82 | 6 | 1/1/1 | 74 | 78 | OK |
| c6288 | 132.65 | 132.44 | -0.16 | 132.09 | -0.42 | -0.35 | 4 | 1/1/1 | 103 | 178 | OK |
| systemcaes | 483.75 | 476.73 | -1.45 | 475.43 | -1.72 | -3.87 | 6 | 1/1/1 | 163 | 304 | OK |
| square | 1064.49 | 1062.04 | -0.23 | 1059.59 | -0.46 | +0.00 | 6 | 1/1/1 | 210 | 474 | OK |
| systemcdes | 131.06 | 130.70 | -0.27 | 130.60 | -0.35 | -0.21 | 6 | 1/1/1 | 112 | 291 | OK |
| spi | 149.43 | 149.35 | -0.05 | 149.24 | -0.13 | -6.83 | 6 | 1/1/1 | 90 | 171 | OK |
| mem_ctrl | 1895.40 | 1872.26 | -1.22 | 1853.68 | -2.20 | -2.16 | 6 | 1/1/1 | 281 | 627 | OK |

**Averages**: δA₁ = -0.79% (27 designs), δA∞ = -1.25% (27), δD∞ = -1.60% (27); TIMEOUT (0): none.


**IWLS'24 single-pass graph (the graph the paper's tab:main was actually measured on), same口径**

| design | base (µm²) | r1 | δA₁ | rN | δA∞ | δD∞ | rounds | CEC b/1/N | t1 (s) | t∞ (s) | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adder | 57.40 | 57.38 | -0.03 | 57.38 | -0.03 | -0.30 | 2 | 1/1/1 | 66 | 65 | OK |
| bar | 133.63 | 131.83 | -1.35 | 129.43 | -3.14 | -2.99 | 3 | 1/1/1 | 71 | 71 | OK |
| max | 137.61 | 136.90 | -0.52 | 136.90 | -0.52 | -0.69 | 2 | 1/1/1 | 72 | 71 | OK |
| sin | 274.08 | 273.44 | -0.23 | 273.16 | -0.34 | -3.67 | 4 | 1/1/1 | 146 | 223 | OK |
| log2 | 1455.42 | 1453.34 | -0.14 | 1453.12 | -0.16 | -3.53 | 4 | 1/1/1 | 292 | 812 | OK |
| div | 1184.83 | 1184.46 | -0.03 | 1184.41 | -0.04 | -0.13 | 3 | 1/1/1 | 176 | 431 | OK |
| cavlc | 32.70 | 32.60 | -0.31 | 32.60 | -0.31 | +0.00 | 2 | 1/1/1 | 60 | 68 | OK |
| multiplier | 1270.30 | 1269.08 | -0.10 | 1268.78 | -0.12 | -0.25 | 3 | 1/1/1 | 222 | 400 | OK |
| sqrt | 1025.43 | 1025.41 | -0.00 | 1025.41 | -0.00 | -0.05 | 2 | 1/1/1 | 300 | 472 | OK |
| ctrl | 5.80 | 5.74 | -1.03 | 5.74 | -1.03 | +0.00 | 2 | 1/1/1 | 56 | 62 | OK |
| dec | 27.06 | 27.06 | +0.00 | 27.06 | +0.00 | +0.00 | 1 | 1/1/1 | 73 | 70 | OK |
| arbiter | 557.72 | 557.72 | +0.00 | 557.72 | +0.00 | +0.00 | 1 | 1/1/1 | 150 | 157 | OK |
| int2float | 10.90 | 10.88 | -0.18 | 10.85 | -0.46 | -0.72 | 3 | 1/1/1 | 69 | 69 | OK |
| i2c | 57.12 | 56.83 | -0.51 | 56.49 | -1.10 | +0.00 | 6 | 1/1/1 | 65 | 81 | OK |
| priority | 27.63 | 27.55 | -0.29 | 27.43 | -0.72 | -1.03 | 3 | 1/1/1 | 66 | 62 | OK |
| router | 7.69 | 7.32 | -4.81 | 7.17 | -6.76 | -4.77 | 4 | 1/1/1 | 62 | 65 | OK |
| sasc | 27.61 | 27.47 | -0.51 | 27.47 | -0.51 | +0.00 | 6 | 1/1/1 | 59 | 63 | OK |
| iwls05_i2c | 49.19 | 49.08 | -0.22 | 49.08 | -0.22 | +0.00 | 6 | 1/1/1 | 64 | 79 | OK |
| simple_spi | 36.62 | 36.56 | -0.16 | 36.52 | -0.27 | +0.00 | 6 | 1/1/1 | 61 | 73 | OK |
| voter | 522.97 | 522.72 | -0.05 | 522.69 | -0.05 | -3.18 | 3 | 1/1/1 | 165 | 224 | OK |
| usb_phy | 22.67 | 22.63 | -0.18 | 22.56 | -0.49 | -2.82 | 6 | 1/1/1 | 72 | 59 | OK |
| c6288 | 132.60 | 132.39 | -0.16 | 132.07 | -0.40 | -0.42 | 3 | 1/1/1 | 92 | 157 | OK |
| systemcaes | 476.89 | 470.41 | -1.36 | 470.09 | -1.43 | -5.11 | 6 | 1/1/1 | 104 | 246 | OK |
| square | 1064.58 | 1062.11 | -0.23 | 1059.66 | -0.46 | +0.00 | 6 | 1/1/1 | 191 | 399 | OK |
| systemcdes | 130.67 | 130.33 | -0.26 | 130.24 | -0.33 | +0.00 | 6 | 1/1/1 | 85 | 156 | OK |
| spi | 147.65 | 147.43 | -0.15 | 147.35 | -0.20 | -6.11 | 6 | 1/1/1 | 73 | 172 | OK |
| mem_ctrl | 2025.00 | 2001.69 | -1.15 | 1974.41 | -2.50 | -5.76 | 6 | 1/1/1 | 223 | 353 | OK |

**Averages**: δA₁ = -0.52% (27 designs), δA∞ = -0.80% (27), δD∞ = -1.54% (27); TIMEOUT (0): none.

**A_i = `direct` (raw AIGER + area-oriented `emap`, no ABC at all): the optimizer runs on that netlist too**

`EPMAPPING_BASELINE=direct` skips the ABC preprocessing entirely (see `epmapping/main.cpp`: the
`subject_path` *is* the AIGER), so the flow measured here is raw AIGER -> `emap` -> ePMapping. These
deltas therefore cover the whole pipeline and are **not** comparable with the two tables above: the
A_e and single-pass baselines already contain the ABC work.

| design | base (µm²) | r1 | δA₁ | rN | δA∞ | δD∞ | rounds | CEC b/1/N | t1 (s) | t∞ (s) | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adder | 57.40 | 57.40 | +0.00 | 57.40 | +0.00 | +0.00 | 1 | 1/1/1 | 83 | 72 | OK |
| bar | 128.74 | 128.74 | +0.00 | 128.74 | +0.00 | +0.00 | 1 | 1/1/1 | 76 | 71 | OK |
| max | 139.76 | 139.38 | -0.27 | 139.38 | -0.27 | -2.02 | 3 | 1/1/1 | 103 | 125 | OK |
| sin | 269.37 | 267.48 | -0.70 | 266.97 | -0.89 | -11.51 | 5 | 1/1/1 | 158 | 260 | OK |
| log2 | 1488.88 | 1485.72 | -0.21 | 1482.82 | -0.41 | -6.88 | 6 | 1/1/1 | 328 | 1279 | OK |
| div | 3609.30 | 3550.77 | -1.62 | — | — | — | — | 1/1/— | 530 | 1800 | TIMEOUT |
| cavlc | 34.20 | 34.12 | -0.23 | 34.06 | -0.41 | +0.00 | 3 | 1/1/1 | 71 | 73 | OK |
| multiplier | 1297.82 | 1296.03 | -0.14 | 1295.14 | -0.21 | -2.19 | 6 | 1/1/1 | 221 | 670 | OK |
| sqrt | 1354.92 | 1241.22 | -8.39 | 1212.16 | -10.54 | -0.17 | 6 | 1/1/1 | 1159 | 1539 | OK |
| ctrl | 7.23 | 7.00 | -3.18 | 6.72 | -7.05 | -12.51 | 5 | 1/1/1 | 60 | 58 | OK |
| dec | 27.06 | 27.06 | +0.00 | 27.06 | +0.00 | +0.00 | 1 | 1/1/1 | 56 | 69 | OK |
| arbiter | 557.72 | 557.72 | +0.00 | 557.72 | +0.00 | +0.00 | 1 | 1/1/1 | 145 | 135 | OK |
| int2float | 11.36 | 11.26 | -0.88 | 11.14 | -1.94 | +0.00 | 6 | 1/1/1 | 61 | 75 | OK |
| i2c | 67.28 | 67.16 | -0.18 | 67.04 | -0.36 | +0.00 | 6 | 1/1/1 | 75 | 87 | OK |
| priority | 52.76 | 44.99 | -14.73 | 39.15 | -25.80 | -12.84 | 6 | 1/1/1 | 76 | 97 | OK |
| router | 13.29 | 13.07 | -1.66 | 12.40 | -6.70 | -11.79 | 4 | 1/1/1 | 69 | 79 | OK |
| sasc | 35.49 | 35.05 | -1.24 | 35.04 | -1.27 | +0.00 | 6 | 1/1/1 | 75 | 70 | OK |
| iwls05_i2c | 58.77 | 57.07 | -2.89 | 55.65 | -5.31 | +0.00 | 6 | 1/1/1 | 65 | 81 | OK |
| simple_spi | 49.18 | 48.68 | -1.02 | 48.44 | -1.50 | -6.41 | 6 | 1/1/1 | 69 | 74 | OK |
| voter | 869.59 | 819.51 | -5.76 | 786.05 | -9.61 | -6.14 | 6 | 1/1/1 | 192 | 493 | OK |
| usb_phy | 27.15 | 26.47 | -2.50 | 25.97 | -4.35 | -10.00 | 6 | 1/1/1 | 56 | 74 | OK |
| c6288 | 127.12 | 126.94 | -0.14 | 126.85 | -0.21 | -17.50 | 5 | 1/1/1 | 79 | 172 | OK |
| systemcaes | 558.96 | 547.01 | -2.14 | 538.91 | -3.59 | -21.59 | 6 | 1/1/1 | 124 | 290 | OK |
| square | 1054.43 | 1050.95 | -0.33 | 1043.90 | -1.00 | -0.22 | 6 | 1/1/1 | 172 | 355 | OK |
| systemcdes | 150.27 | 148.74 | -1.02 | 148.04 | -1.48 | -1.30 | 6 | 1/1/1 | 70 | 196 | OK |
| spi | 182.17 | 175.38 | -3.73 | 171.23 | -6.01 | -7.42 | 6 | 1/1/1 | 64 | 104 | OK |
| mem_ctrl | 2278.85 | 2239.71 | -1.72 | 2194.91 | -3.68 | -3.85 | 6 | 1/1/1 | 133 | 318 | OK |

**Averages**: δA₁ = -2.03% (27 designs), δA∞ = -3.56% (26), δD∞ = -5.17% (26); TIMEOUT (1): div.

The three starting points disagree by a lot, and not in the direction one would expect:

* **The percentages are a property of the baseline, not of the engine.** Against A_i the same engine
  reports δA₁ -2.03 % / δA∞ -3.56 %, against A_e -0.79 % / -1.25 %. Most of the A_i figure is the ABC
  preprocessing that A_e already had: `priority`'s direct mapping is 52.76 µm² against A_e's 27.51, so
  the engine gets -25.80 % there, while on A_e the ABC flow has already removed that area and the
  engine gets -0.55 %. Quoting the A_i number as "the gain of ePMapping" would count ABC's work twice.
* **What matters is the absolute final area, and A_e wins it on 15/27 designs** outright (the IWLS
  single-pass graph wins 7, A_i 3; `dec` and `arbiter` are exact ties at 27.06/557.72, where every flow
  is neutral). The three A_i wins are the designs whose direct mapping is already the smallest:
  `sin` (266.97 against 274.77 on A_e), `c6288` (126.85 against 132.09) and `square` (1043.90 against
  1059.59).
* **The ABC flow is not uniformly an improvement.** On `bar`, `sin`, `c6288` and `square` the direct
  mapping is already smaller than A_e's (128.74 < 131.20, 269.37 < 276.16, 127.12 < 132.65,
  1054.43 < 1064.49). On `c6288` it *reduces* the AIG from 2 337 to 1 870 ANDs, yet the mapped netlist
  grows from 966 to 1 162 gates (-20 % ANDs, +11 % gates, +4.4 % area): `resyn2rs`/`compress2rs` break
  up the shared adder-array structure that `emap` turns into multi-input cells. That is a property of
  the setup flow, not of ePMapping, but it is why the same optimizer can look much better or worse
  depending on which graph it is handed.
* **Some of the A_e headroom is manufactured by ABC.** On `adder`, `bar`, `dec` and `arbiter` the
  optimizer makes *no* replacement at all on the direct netlist (`rounds=1`, +0.00 %), so `bar`'s
  -3.09 % in the A_e table comes from a structure the ABC graph exposes, not from one the direct
  mapping has.
* **Only `div` exceeds the 1 800 s per-configuration budget on A_i** once the 60 s don't-care budget is
  actually applied: its direct mapping is 3.0x A_e's (3 609.30 against 1 184.50), and no pass finishes.
  The first version of this table was measured *without* the budget by mistake and reported five
  timeouts; with the budget `sin`, `log2`, `multiplier` and `sqrt` all converge (and `sqrt` reaches
  1 212.16 from 1 354.92).
* `max` was the window-fallback accounting defect documented above; with that fix the A_i row goes from
  +0.82 % to -0.33 % (139.76 -> 139.38, identical to `EPMAPPING_WINDOW_SYN=0`).



**IWLS'24 single-pass graph, **no** DC budget (engine defaults) — 4 designs exceed 1800 s**

| design | base (µm²) | r1 | δA₁ | rN | δA∞ | δD∞ | rounds | CEC b/1/N | t1 (s) | t∞ (s) | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adder | 57.40 | 57.38 | -0.03 | 57.38 | -0.03 | -0.30 | 2 | 1/1/1 | 88 | 91 | OK |
| bar | 133.63 | 131.83 | -1.35 | 129.43 | -3.14 | -2.99 | 3 | 1/1/1 | 83 | 107 | OK |
| max | 137.61 | 137.14 | -0.34 | 137.06 | -0.40 | -0.75 | 4 | 1/1/1 | 82 | 116 | OK |
| sin | 274.08 | — | — | — | — | — |  | 1// | 1800 | 1800 | TIMEOUT |
| log2 | 1455.42 | — | — | — | — | — |  | 1// | 1801 | 1800 | TIMEOUT |
| cavlc | 32.70 | 32.60 | -0.31 | 32.60 | -0.31 | 0.00 | 2 | 1/1/1 | 133 | 115 | OK |
| div | 1184.83 | 1184.46 | -0.03 | 1184.41 | -0.04 | -0.13 | 3 | 1/1/1 | 490 | 738 | OK |
| multiplier | 1270.30 | — | — | — | — | — |  | 1// | 1800 | 1800 | TIMEOUT |
| ctrl | 5.80 | 5.74 | -1.03 | 5.74 | -1.03 | 0.00 | 2 | 1/1/1 | 73 | 72 | OK |
| dec | 27.06 | 27.06 | 0.00 | 27.06 | 0.00 | 0.00 | 1 | 1/1/1 | 76 | 85 | OK |
| sqrt | 1025.43 | — | — | — | — | — |  | 1// | 1800 | 1800 | TIMEOUT |
| int2float | 10.90 | 10.88 | -0.18 | 10.85 | -0.46 | -0.72 | 3 | 1/1/1 | 68 | 67 | OK |
| arbiter | 557.72 | 557.72 | 0.00 | 557.72 | 0.00 | 0.00 | 1 | 1/1/1 | 195 | 204 | OK |
| i2c | 57.12 | 56.83 | -0.51 | 56.63 | -0.86 | 0.00 | 6 | 1/1/1 | 75 | 93 | OK |
| priority | 27.63 | 27.55 | -0.29 | 27.43 | -0.72 | -1.03 | 3 | 1/1/1 | 75 | 70 | OK |
| router | 7.69 | 7.32 | -4.81 | 7.17 | -6.76 | -5.00 | 4 | 1/1/1 | 67 | 58 | OK |
| sasc | 27.61 | 27.47 | -0.51 | 27.47 | -0.51 | 0.00 | 6 | 1/1/1 | 67 | 69 | OK |
| iwls05_i2c | 49.19 | 49.08 | -0.22 | 49.08 | -0.22 | 0.00 | 6 | 1/1/1 | 72 | 86 | OK |
| simple_spi | 36.62 | 36.56 | -0.16 | 36.52 | -0.27 | 0.00 | 6 | 1/1/1 | 68 | 82 | OK |
| usb_phy | 22.67 | 22.63 | -0.18 | 22.56 | -0.49 | -2.82 | 6 | 1/1/1 | 61 | 53 | OK |
| c6288 | 132.60 | 132.37 | -0.17 | 132.05 | -0.41 | -0.05 | 3 | 1/1/1 | 86 | 145 | OK |
| voter | 522.97 | 522.72 | -0.05 | 522.68 | -0.06 | -3.18 | 3 | 1/1/1 | 196 | 415 | OK |
| systemcaes | 476.89 | 470.41 | -1.36 | 470.07 | -1.43 | -5.11 | 6 | 1/1/1 | 90 | 294 | OK |
| systemcdes | 130.67 | 130.33 | -0.26 | 130.24 | -0.33 | 0.00 | 6 | 1/1/1 | 61 | 159 | OK |
| spi | 147.65 | 147.43 | -0.15 | 147.35 | -0.20 | -6.11 | 6 | 1/1/1 | 57 | 138 | OK |
| square | 1064.58 | 1061.74 | -0.27 | 1059.12 | -0.51 | 0.00 | 6 | 1/1/1 | 297 | 1244 | OK |
| mem_ctrl | 2025.00 | 1994.51 | -1.51 | 1950.12 | -3.70 | -4.99 | 6 | 1/1/1 | 183 | 830 | OK |

**Averages**: δA₁ = -0.60% (23 designs), δA∞ = -0.95% (23 converging), δD∞ = -1.44%; TIMEOUT (4): sin, log2, multiplier, sqrt.


**A_e, **no** DC budget (engine defaults) — 5 designs exceed 1800 s**

| design | base (µm²) | r1 | δA₁ | rN | δA∞ | δD∞ | rounds | CEC b/1/N | t1 (s) | t∞ (s) | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adder | 57.40 | 57.38 | -0.03 | 57.38 | -0.03 | -0.30 | 2 | 1/1/1 | 38 | 39 | OK |
| bar | 131.20 | 129.46 | -1.33 | 127.14 | -3.09 | -2.99 | 3 | 1/1/1 | 42 | 46 | OK |
| max | 135.53 | 135.15 | -0.28 | 135.07 | -0.34 | -0.64 | 4 | 1/1/1 | 44 | 59 | OK |
| sin | 276.16 | 274.38 | -0.64 | — | — | — |  | 1/1/ | 963 | 1800 | TIMEOUT |
| log2 | 1462.00 | — | — | — | — | — |  | 1// | 1800 | 1801 | TIMEOUT |
| cavlc | 33.07 | 33.00 | -0.21 | 33.00 | -0.21 | 0.00 | 2 | 1/1/1 | 82 | 79 | OK |
| div | 1184.50 | 1184.17 | -0.03 | 1183.98 | -0.04 | -0.44 | 4 | 1/1/1 | 373 | 897 | OK |
| multiplier | 1270.83 | — | — | — | — | — |  | 1// | 1800 | 1800 | TIMEOUT |
| sqrt | 1157.38 | — | — | — | — | — |  | 1// | 1800 | 1800 | TIMEOUT |
| ctrl | 5.86 | 5.84 | -0.34 | 5.72 | -2.39 | 0.00 | 5 | 1/1/1 | 73 | 76 | OK |
| dec | 27.06 | 27.06 | 0.00 | 27.06 | 0.00 | 0.00 | 1 | 1/1/1 | 71 | 75 | OK |
| int2float | 10.89 | 10.87 | -0.18 | 10.84 | -0.46 | 0.00 | 3 | 1/1/1 | 74 | 74 | OK |
| arbiter | 557.72 | 557.72 | 0.00 | 557.72 | 0.00 | 0.00 | 1 | 1/1/1 | 207 | 212 | OK |
| i2c | 55.77 | 55.50 | -0.48 | 55.22 | -0.99 | 0.00 | 6 | 1/1/1 | 76 | 87 | OK |
| priority | 27.51 | 27.36 | -0.55 | 27.36 | -0.55 | 0.00 | 2 | 1/1/1 | 73 | 77 | OK |
| router | 7.72 | 6.99 | -9.46 | 6.68 | -13.47 | -4.22 | 5 | 1/1/1 | 70 | 65 | OK |
| sasc | 27.28 | 27.18 | -0.37 | 27.18 | -0.37 | 0.00 | 6 | 1/1/1 | 78 | 83 | OK |
| iwls05_i2c | 47.66 | 47.57 | -0.19 | 47.51 | -0.31 | 0.00 | 6 | 1/1/1 | 80 | 89 | OK |
| simple_spi | 36.96 | 36.85 | -0.30 | 36.83 | -0.35 | -5.32 | 6 | 1/1/1 | 79 | 86 | OK |
| usb_phy | 22.54 | 22.50 | -0.18 | 22.45 | -0.40 | -2.82 | 6 | 1/1/1 | 71 | 81 | OK |
| c6288 | 132.65 | 132.42 | -0.17 | 132.07 | -0.44 | -0.35 | 4 | 1/1/1 | 115 | 220 | OK |
| voter | 522.54 | 522.25 | -0.06 | 522.22 | -0.06 | -1.94 | 3 | 1/1/1 | 240 | 530 | OK |
| systemcaes | 483.75 | 476.73 | -1.45 | 474.21 | -1.97 | -3.87 | 6 | 1/1/1 | 183 | 422 | OK |
| systemcdes | 131.06 | 130.70 | -0.27 | 130.60 | -0.35 | -0.21 | 6 | 1/1/1 | 126 | 351 | OK |
| spi | 149.43 | 149.35 | -0.05 | 149.24 | -0.13 | -6.83 | 6 | 1/1/1 | 90 | 200 | OK |
| square | 1064.49 | 1061.69 | -0.26 | — | — | — |  | 1/1/ | 400 | 1801 | TIMEOUT |
| mem_ctrl | 1895.40 | 1872.26 | -1.22 | 1848.06 | -2.50 | -2.13 | 6 | 1/1/1 | 184 | 500 | OK |

**Averages**: δA₁ = -0.75% (24 designs), δA∞ = -1.29% (22 converging), δD∞ = -1.46%; TIMEOUT (5): sin, log2, multiplier, sqrt, square.


### ISCAS subset (10 designs, §Overall's second paragraph)

**A_e**

| design | base (µm²) | r1 | δA₁ | rN | δA∞ | δD∞ | rounds | CEC b/1/N | t1 (s) | t∞ (s) | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c432 | 6.61 | 6.60 | -0.15 | 6.60 | -0.15 | -4.06 | 2 | 1/1/1 | 67 | 72 | OK |
| c499 | 22.83 | 22.83 | +0.00 | 22.83 | +0.00 | -12.54 | 1 | 1/1/1 | 69 | 67 | OK |
| c880 | 15.78 | 15.63 | -0.95 | 15.63 | -0.95 | -2.29 | 2 | 1/1/1 | 74 | 70 | OK |
| c1355 | 22.82 | 22.80 | -0.09 | 22.80 | -0.09 | -12.66 | 2 | 1/1/1 | 69 | 75 | OK |
| c1908 | 15.95 | 15.71 | -1.50 | 15.71 | -1.50 | -6.82 | 2 | 1/1/1 | 66 | 73 | OK |
| c2670 | 26.73 | 26.67 | -0.22 | 26.67 | -0.22 | -0.14 | 6 | 1/1/1 | 70 | 74 | OK |
| c3540 | 41.88 | 41.52 | -0.86 | 41.38 | -1.19 | -7.73 | 3 | 1/1/1 | 75 | 76 | OK |
| c5315 | 64.21 | 64.03 | -0.28 | 63.76 | -0.70 | -5.52 | 6 | 1/1/1 | 76 | 107 | OK |
| c6288 | 132.65 | 132.44 | -0.16 | 132.09 | -0.42 | -0.35 | 4 | 1/1/1 | 103 | 203 | OK |
| c7552 | 81.87 | 81.74 | -0.16 | 81.68 | -0.23 | -5.49 | 6 | 1/1/1 | 94 | 135 | OK |

**Averages**: δA₁ = -0.44% (10 designs), δA∞ = -0.55% (10), δD∞ = -5.76% (10); TIMEOUT (0): none.


**IWLS'24 single-pass graph (the paper's口径)**

| design | base (µm²) | r1 | δA₁ | rN | δA∞ | δD∞ | rounds | CEC b/1/N | t1 (s) | t∞ (s) | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c432 | 6.61 | 6.61 | +0.00 | 6.61 | +0.00 | +0.00 | 1 | 1/1/1 | 120 | 125 | OK |
| c499 | 22.79 | 22.79 | +0.00 | 22.79 | +0.00 | -12.54 | 1 | 1/1/1 | 130 | 127 | OK |
| c880 | 15.75 | 15.59 | -1.02 | 15.59 | -1.02 | -2.95 | 2 | 1/1/1 | 127 | 126 | OK |
| c1355 | 22.76 | 22.76 | +0.00 | 22.76 | +0.00 | -7.87 | 1 | 1/1/1 | 102 | 112 | OK |
| c1908 | 16.03 | 15.79 | -1.50 | 15.79 | -1.50 | -1.34 | 2 | 1/1/1 | 110 | 109 | OK |
| c2670 | 26.86 | 26.80 | -0.22 | 26.80 | -0.22 | -0.41 | 6 | 1/1/1 | 110 | 126 | OK |
| c3540 | 43.28 | 42.50 | -1.80 | 42.41 | -2.01 | -12.85 | 3 | 1/1/1 | 92 | 85 | OK |
| c5315 | 64.64 | 64.53 | -0.17 | 64.16 | -0.74 | -3.03 | 6 | 1/1/1 | 88 | 106 | OK |
| c6288 | 132.60 | 132.39 | -0.16 | 132.07 | -0.40 | -0.42 | 3 | 1/1/1 | 129 | 164 | OK |
| c7552 | 80.37 | 80.10 | -0.34 | 80.10 | -0.34 | +0.00 | 6 | 1/1/1 | 60 | 76 | OK |

**Averages**: δA₁ = -0.52% (10 designs), δA∞ = -0.62% (10), δD∞ = -4.14% (10); TIMEOUT (0): none.

### tab:oracle (c432 / c3540 / c5315 / c2670, one pass, strict NDR, shipped database)

"cut stages" is L1+L2 only; "cut stages +egg" leaves the reward-gated e-graph on (the engine default);
"+SYN" adds structural synthesis; "+window" adds the MFFW fallback.

| design | graph | cut stages | cut stages +egg | +SYN | +window | paper (+SYN / +window) |
| --- | --- | --- | --- | --- | --- | --- |
| c432 | A_e | 6.61 (+0.00%) | 6.61 (+0.00%) | −0.15% | −0.15% | −0.61% / −0.61% |
| c3540 | A_e | 41.75 (−0.31%) | 41.73 (−0.36%) | −0.81% | −0.81% | −1.50% / −1.60% |
| c5315 | A_e | 64.21 (+0.00%) | 64.21 (+0.00%) | −0.25% | −0.20% | −0.78% / −1.14% |
| c2670 | A_e | 26.73 (+0.00%) | 26.73 (+0.00%) | +0.00% | −0.22% | −0.34% / — |
| c432 | IWLS | 6.61 (+0.00%) | 6.61 (+0.00%) | +0.00% | +0.00% | — |
| c3540 | IWLS | 43.22 (−0.14%) | 43.18 (−0.23%) | −1.64% | −1.71% | — |
| c5315 | IWLS | 64.58 (−0.09%) | 64.58 (−0.09%) | −0.11% | −0.06% | — |
| c2670 | IWLS | 26.86 (+0.00%) | 26.86 (+0.00%) | −0.22% | −0.22% | — |

The paper's magnitudes do not reproduce on either graph with the current engine (its `c3540` cut-stages
column is −0.62% where both graphs measure −0.14…−0.36%, and `c5315` +window −1.14% against −0.06…−0.20%).

### tab:db (c7552, six passes, strict NDR)

| database | ABC command | structures | 4-input coverage | A_e | IWLS |
| --- | --- | --- | --- | --- | --- |
| single gates only | — | 194 pins | 0% | 81.68 | 80.10 |
| 2-level | `super -I 4 -L 2 -T 60` | 99 510 | 24% | 81.69 | 79.99 |
| 3-level, a ≤ 0.45 | `super -I 4 -L 3 -A 0.45 -T 240` | 228 163 | 34% | 81.71 | 80.03 |
| 3-level, a ≤ 0.50 (shipped) | `… -A 0.50 …` | 505 598 | 45% | 81.68 | 80.10 |
| 3-level, a ≤ 0.55 | `… -A 0.55 …` | 977 855 | 64% | 81.68 | 79.90 |
| 3-level, a ≤ 0.60 | `… -A 0.60 …` | 1 112 219 | 77% | 81.68 | 80.10 |

The coverage column reproduces the paper exactly; the **area lever does not**. Without dependency cuts the
database *is* consulted and its covers *are* accepted (`staged library_cuts` 6 297 → 16 806, `multi` wins
0 → 33, `reject_objective_by_source[5]` 3 387, accepted replacements 304 → 359), yet the final area and
delay are identical to two decimals (81.68 / 621.36 on A_e). The paper's 80.33 → 78.91 was measured
**with dependency cuts**, which this engine does not form. The shipped file also holds 505 598 structures
where the paper lists 504 368 (it was regenerated).

### tab:funnel (one pass, A_e; `(s)` is the whole run wall on an otherwise idle machine)

| circuit | L1 wins | L2 wins | L3 (SYN) wins | L4 cuts | L4 wins | L5 (window) | wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c7552 | 5 | 18 | 9 | 449 | 48 | 0 | 49 |
| sin | 148 | 16 | 45 | 1 519 | 6 | 11 | 752 |
| div | 923 | 2 | 18 | 3 686 | 2 | 10 | 113 |
| multiplier | 1 643 | 13 | 1 | 3 783 | 8 | 0 | 850 |
| log2 | 1 137 | 70 | 77 | 6 132 | 12 | 27 | >1 800 (TIMEOUT) |

`log2`'s counts are from the DC-budget run (its unbounded one-pass exceeds the 1 800 s budget). The funnel
shape survives: L4 is admitted thousands of times for single-digit wins, and the library layers answer the
bulk of the substitutions.

### Lever 1 — sat/don't cares (one pass, A_e + DC budget 60 s)

| design | `SAT_DC=1` care | of which SAT | SAT calls | cuts with DC | `SAT_DC=0` (window only) care | area (`=1` / `=0`) |
| --- | --- | --- | --- | --- | --- | --- |
| sasc | 0.050 s | 0.022 s | 44 676 | 748 | 0.005 s | 27.18 / 27.18 |
| arbiter | 35.1 s | 32.2 s | 5 274 156 | 61 320 | 0.134 s | 557.72 / 557.72 |

`EPMAPPING_DC_MAX_PIS=0/2/8` gives identical output (27.18). With the window now bit-parallel the trade is
reversed from the paper's measurement: the window costs 0.13 s on `arbiter` where SAT costs 32.2 s and buys
no area at all (the paper measured SAT 69.9 s against an 83.4 s window).

### Lever 3 — searching harder does not pay (one pass, A_e, strict NDR)

`c7552` 81.74, `c3540` 41.52, `c5315` 64.03, `c2670` 26.67 — **identical area *and* delay** for the base
configuration and for `MAP_EGG_EGG_ITER_LIMIT=5/10`, `MAP_EGG_EGG_NODE_LIMIT=2000/5000` and
`MAP_EGG_FULL_BUNDLE=1`, each within 31–37 s. The paper's two remaining Lever-3 rows
(`EPMAPPING_EGG_MULTI` on every large cut and `EPMAPPING_EGG_COST=lib|depth|both`) need knobs this tree
does not have; they were re-run in `MIGBalance` instead.


The two rows that need knobs this tree no longer has were re-run in `MIGBalance` (same subject graph and
objective, one pass): `EPMAPPING_EGG_MULTI=4` with `EPMAPPING_STAGED_EGG_ALL=1` (the multi-solution source
on every large cut) moves `c7552` 81.74 → 81.73 and takes 76 s against 36 s, and changes nothing on
`c3540` (41.52), `c5315` (64.03) or `c2670` (26.67); `EPMAPPING_EGG_COST=depth` and `=both` are inert
(identical area and delay on all four, 32–35 s). That is the paper's Lever-3 claim: effort inside L4 buys
more candidates of the same kind, not a different result.
### Lever 2 — the stage ablation (eleven designs, A_e, strict NDR, DSE off, DC budget 60 s, six rounds)

Absolute area increase when one stage is removed from the full flow, so a larger number means the stage
contributes more. `−SYN` is structural synthesis off, `−window` the MFFW fallback off, `−egg` the e-graph off.

| design | full | −SYN | −window | −egg | ΔSYN | Δwindow | Δegg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aes_core | 1027.00 | 1029.18 | 1027.65 | 1027.06 | +2.18 | +0.65 | +0.06 |
| arbiter | 557.72 | 557.72 | 557.72 | 557.72 | +0.00 | +0.00 | +0.00 |
| div | 1183.98 | 1184.11 | 1184.05 | 1184.19 | +0.13 | +0.07 | +0.21 |
| iwls05_i2c | 47.51 | 47.62 | 47.51 | 47.51 | +0.11 | +0.00 | +0.00 |
| mem_ctrl | 1854.57 | 1883.86 | 1867.44 | 1855.76 | +29.29 | +12.87 | +1.19 |
| sasc | 27.18 | 27.17 | 27.18 | 27.18 | −0.01 | +0.00 | +0.00 |
| simple_spi | 36.83 | 36.86 | 36.91 | 36.83 | +0.03 | +0.08 | +0.00 |
| spi | 149.24 | 149.41 | 149.35 | 149.24 | +0.17 | +0.11 | +0.00 |
| systemcdes | 130.60 | 130.98 | 130.60 | 130.62 | +0.38 | +0.00 | +0.02 |
| tv80 | 338.23 | 339.55 | 338.76 | 338.81 | +1.32 | +0.53 | +0.58 |
| usb_phy | 22.45 | 22.45 | 22.54 | 22.45 | +0.00 | +0.09 | +0.00 |
| **cumulative** | | | | | **+33.60** | **+14.40** | **+2.06** |

The ordering is structural synthesis, then the window, then the e-graph: the candidate-set lever dominates
the scope lever, which dominates the search lever. The magnitudes are larger than the +5.31 / +0.49 the
paper recorded for the window and the e-graph, mostly because the window stage is now counted as round
progress (`st.replacements` includes the gates a committed window rewires), so it gets the rounds it
was previously denied.

### The e-graph (L4) share of a pass

`full` against `EPMAPPING_EGG=0`, one pass, A_e + DC budget:

| design | area (on / off) | egg seconds | wall (on / off) |
| --- | --- | --- | --- |
| sqrt | 1121.40 / 1121.40 | 26.2 | 182 / 147 |
| div | 1184.17 / 1184.24 | 1.5 | 110 / 108 |
| mem_ctrl | 1872.26 / 1872.37 | 5.8 | 186 / 179 |

### Relation to prior work, matched subject graph (A_e)

Over the 16 designs of the ASPDAC'25 comparison, with the same iterated area-oriented ABC subject graph
and the same area-oriented `emap`: **δA₁ = −0.51%, δA∞ = −0.76%** (against the −2.70% / −6.39% that
\cite{aspdac25} reports for its post-mapping resubstitution). The gap is now ~8×, not the ~42× implied by
the −0.15% the paper currently quotes. The largest contributors are `sqrt` (−4.46%), `mem_ctrl` (−2.20%)
and `systemcaes` (−1.72%). This comparison deliberately puts both engines on *our* (smaller) subject
graph; the absolute comparison against the netlist ASPDAC'25 actually obtains is the next section.


### Relation to prior work: our netlist vs the area/delay ASPDAC'25 *obtains*

\cite{aspdac25} (Table 1) gives, for the 16 benchmarks where its per-design results can be matched, its
own initial mapped netlist (`A_e`, `D_e`) and the gain its post-mapping resubstitution reports after one
pass and at convergence; what it *obtains* is `A_e*(1+dA_einf)` in area and `D_e*(1+dD_einf)` in delay.
This table uses that as the reference, because the current toolchain already builds a **smaller initial
netlist than theirs on all 16 designs** (mean -7.93 %, up to -18.67 % on `ethernet`). Both of our
starting points are shown: `A_e` (the same iterated area-oriented ABC flow, built by the current
toolchain) and `A_i` (`EPMAPPING_BASELINE=direct`: no ABC preprocessing, so the optimizer runs on the
raw mapped netlist). Otherwise the DATE'27口径: shipped database, DSE off, strict NDR, 60 s don't-care
budget, up to six passes.

| design | ASPDAC'25 obtained (µm²) | ours, `A_e` (µm²) | Δ (`A_e`) | ours, `A_i` (µm²) | Δ (`A_i`) | ASPDAC'25 obtained delay (ps) | ours, `A_e` (ps) | Δ (`A_e`) | ours, `A_i` (ps) | Δ (`A_i`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `aes_core` | 1086.57 | 1026.72 | -5.51 | 1094.03 | +0.69 | 425.61 | 410.72 | -3.50 | 390.28 | -8.30 |
| `arbiter` | 268.82 | 557.72 | +107.47 | 557.72 | +107.47 | 604.27 | 999.87 | +65.47 | 999.87 | +65.47 |
| `div` | 1176.29 | 1184.02 | +0.66 | — | — | 60254.25 | 52899.30 | -12.21 | — | — |
| `ethernet` | 3008.41 | 2539.90 | -15.57 | 3701.49 | +23.04 | 588.34 | 518.31 | -11.90 | 508.94 | -13.50 |
| `iwls05_i2c` | 48.96 | 47.51 | -2.96 | 55.65 | +13.66 | 288.07 | 303.61 | +5.39 | 267.23 | -7.23 |
| `mem_ctrl` | 1799.15 | 1853.68 | +3.03 | 2194.91 | +22.00 | 1435.03 | 1486.89 | +3.61 | 1640.59 | +14.32 |
| `RISC` | 3139.86 | 2917.77 | -7.07 | 3489.72 | +11.14 | 1304.74 | 1417.86 | +8.67 | 629.09 | -51.78 |
| `sasc` | 31.30 | 27.18 | -13.17 | 35.04 | +11.94 | 191.00 | 191.20 | +0.10 | 140.61 | -26.38 |
| `simple_spi` | 41.00 | 36.83 | -10.18 | 48.44 | +18.14 | 287.00 | 291.82 | +1.68 | 200.75 | -30.05 |
| `spi` | 164.10 | 149.24 | -9.06 | 171.23 | +4.34 | 489.49 | 466.75 | -4.65 | 439.94 | -10.12 |
| `sqrt` | 1110.02 | 1105.72 | -0.39 | 1212.16 | +9.20 | 78002.24 | 82802.70 | +6.15 | 101780.00 | +30.48 |
| `systemcaes` | 511.67 | 475.43 | -7.08 | 538.91 | +5.32 | 784.00 | 681.61 | -13.06 | 607.50 | -22.51 |
| `systemcdes` | 138.84 | 130.60 | -5.93 | 148.04 | +6.63 | 530.29 | 487.83 | -8.01 | 425.30 | -19.80 |
| `tv80` | 351.11 | 338.23 | -3.67 | 431.69 | +22.95 | 907.86 | 846.12 | -6.80 | 777.17 | -14.40 |
| `usb_funct` | 695.57 | 642.94 | -7.57 | 763.22 | +9.73 | 702.87 | 663.36 | -5.62 | 394.24 | -43.91 |
| `usb_phy` | 23.63 | 22.45 | -4.99 | 25.97 | +9.90 | 163.82 | 158.68 | -3.14 | 139.65 | -14.76 |

**Means**: against what ASPDAC'25 obtains, the `A_e` flow is smaller on 13/16 designs (Δarea -5.96 %
over the 15 designs other than `arbiter`, +1.13 % including it; Δdelay +1.39 %), while the `A_i` flow is
*larger* on all 15 designs that converge (Δarea +18.41 %) but much faster (Δdelay -10.16 %). `div` is
the only design with no `A_i` result: its one-pass netlist is 3 550.77 µm² against the 1 176.29 µm² that
ASPDAC'25 obtains, and no second pass finishes inside 1 800 s.

* **The two starts answer different questions.** `A_e` is the whole pipeline; `A_i` is the engine alone
  on a raw mapping. The +18.41 % of `A_i` says the engine cannot recover the ABC preprocessing
  (`ethernet` +23.04 %, `tv80` +22.95 %, `mem_ctrl` +21.70 %, `simple_spi` +18.14 %), and its -10.16 %
  delay says the area-oriented ABC flow is what costs the delay in the first place.
* **The `A_i` delay win is inherited from the start, not created by the engine.** The `A_i` baselines are
  already faster than what ASPDAC'25 obtains (`RISC` 639.47 against 1304.74 ps, `usb_funct` 421.38
  against 702.87, `simple_spi` 214.50 against 287.00). The engine's own delay change on `A_i` is -5.04 %
  on average (`systemcaes` -24.63 %, `usb_phy` -10.00 %, `spi` -7.42 %).
* **`arbiter` is the one design their flow wins**: 557.84 -> 268.82 (-51.81 %), while both of our starts
  leave it untouched (557.72 -> 557.72). Our `A_e` baseline matches theirs to 0.02 %, so this is a
  candidate-set difference, not a setup effect; our DSE wrapper with `&syn4` reaches 272.6 there, and
  beats their 1 799 on `mem_ctrl` with 1 640 (`scripts/_epm_one.sh`).
* **`mem_ctrl` (+3.03 %) and `div` (+0.66 %) are the only other `A_e` losses**, and on both our initial
  netlist is already 8-10 % smaller than theirs, i.e. the loss is in the post-mapping step.
* Averages are arithmetic means; `arbiter` alone moves the `A_e` area mean from -5.96 % to +1.13 %.

## Notes

* Iteration is a safety net, not the main lever. The pass is greedy and rebuilds the network in
  topological order between passes (`params::max_rounds`); a pass ends the run as soon as it replaces
  nothing or its objective metric does not improve (`run()`: `if (st.replacements == before) break;`,
  then `if (objective_metric() >= metric_before - 1e-6) break;`), which is also what keeps equal-cost
  area/delay ties from churning. On the paper's subject graph one pass already reaches the converged
  value almost everywhere (c499: −0.04% after 1 pass and after the pass cap), and most designs stop
  after 2 passes.
* Candidate comparisons use **quantized** integer keys (area at 1e-4, delay at 1e-3). Raw
  floating-point near-ties flip with codegen changes, and a single flip cascades through the greedy
  pass into a different final netlist; quantization makes repeated runs reproducible.
* Runtime used to be dominated by the egg path. It is now off by default and 5–6-leaf cuts go to the
  native library cover instead; `EPMAPPING_EGG=1` re-enables it for comparison (see the native
  cover and egg measurements above).
* `block_network` has no structural hashing, so a replacement always materializes fresh nodes; the
  gain model compares the candidate cover area against the cut's exact MFFC area.
* `write_verilog_with_cell` drops fanin complementation, so every inversion is materialized with an
  explicit inverter cell.
* **Primary-output driver pitfall (fixed):** the writer names a primary output net after its driver
  node and emits a driver instance only when that node has a bound cell. A PO driven by a *cell-less*
  constant or PI therefore ends up **undriven** (ABC reports `Constant-0 drivers added to N
  non-driven nets` and the output is tied to 0 → CEC=0). `emap` itself binds `_const1_`/`_const0_`
  (area 0.00) to constant POs, so `rebuild_topological` must copy cell bindings for constants and
  PIs, and additionally wraps any remaining cell-less constant/PI PO in a buffer cell.

## Next steps (measured priorities)

Ten mechanisms have been implemented and measured. Every local one is at a fixed point on an `emap`
mapping of the same library (structural cuts, native multi-cell cover, egg, the mapped sub-network
database, dependency cuts with non-local divisors, MFFW, observation windows, the mapper's own knobs,
unmap/rebalance/remap, and AIG-level don't-care/SAT resubstitution), and both global don't-care
constructions are now measured end to end:

| mechanism | measured outcome |
| --- | --- |
| local cut/MFFC rewriting | −0.11% average (the shipped engine, `cec=1` everywhere) |
| sampled satisfiability don't cares | provably spurious when the exact local set says "all tuples occur" |
| observability don't cares (sampled + CEC-verified) | 9% of candidates accepted, worth **−0.008%** on `div` |
| ABC `dc2` / `&resub` | nothing that survives remapping (`div` +6.6% mapped area) |
| unoptimized subject graph | −0.63% on `mem_ctrl`, still far worse than from the optimized graph |

Two things remain, with their expected value stated honestly:

1. **Exact (exhaustive) observability for few-input designs.** With ≤8 primary inputs the sampled
   observability becomes exact (`ctrl` 7, `dec` 8), so the don't-care sets are sound without CEC
   proposals and the candidates are correct by construction. This is bounded, implementable and
   provably sound, but the designs it applies to are small, so its effect on a 27-design average is
   below 0.01%.
2. **A SAT engine with counter-example refinement** (the paper's `B_cex` loop). This is the only route
   to the paper's remaining numbers, and the measurements bound it from both sides: sampling alone
   gives large but false potential, the accepted fraction is ~9%, and ABC's own SAT/DC resubstitution
   finds *nothing* on `arbiter` (the design that contributes −1.12 points to the paper's 39-design
   average) or on `mem_ctrl`/`div`/`sqrt`. Implementing it would be a research-scale effort with an
   expected gain that the evidence above does not support.

For the record, the paper's headline numbers are not reproducible in this environment: `arbiter`
(−43.85%) has no satisfiability *or* observability don't cares (0 cuts with ODCs, 0 structural cuts
with SDCs) and no tool here — including ABC's SAT resubstitution — reduces it; `mem_ctrl` (−12.25%)
and `div` (−9.53%) likewise gain ≈0 while our *absolute* areas already beat the paper's optimized
results on `sqrt`, `spi`, `sasc`, `simple_spi`, `systemcdes`, `systemcaes`, `sin`, `usb_phy`, `cavlc`
and `iwls05_i2c`.

### Measured: optimizing the unoptimized subject graph

The paper notes that `mem_ctrl`, `log2` and `sin` are best optimized from the *unoptimized* subject
graph. Measured with `EPMAPPING_BASELINE=direct` (no ABC preprocessing at all):

| design | emap of the raw AIG | after epmapping | paper `A_i` | paper gain from it |
| --- | --- | --- | --- | --- |
| mem_ctrl | 2278.85 | 2264.59 (−0.63%) | 2547.32 | −7.63% (δA1i) |

Our raw-AIG mapping is already smaller than the paper's baseline, and the engine adds only 0.63%
there — and 2264.59 is still far worse than the 2018.07 reached from the optimized subject graph.

### Measured: AIG-level don't-care/SAT resubstitution does not transfer to mapped area

Since the paper's gains are don't-care restructurings, the natural question is whether ABC's mature
implementations reproduce them — they do not, and their AIG reductions do not survive remapping:

| tool (on the preprocessed AIG) | arbiter | mem_ctrl | div | sqrt |
| --- | --- | --- | --- | --- |
| `dc2` (don't-care simplification), ANDs | 11839 → 11839 | 46836 → 42797 (−8.6%) | 57247 → 40520 (−29%) | 24618 → 18453 (−25%) |
| `&resub` (SAT resubstitution), ANDs | 11839 → 11839 | no change | no change | no change |

and unmapping our own optimized mapped netlist, running `dc2`, and remapping (`emap + epmapping`)
makes `div` *worse*: 1184.83 → **1262.77** (−17% AIG nodes, +6.6% mapped area, delay 52695 → 65141).
The same loop on `sin` is also slightly worse (273.48 → 273.76). So the AIG-node metric that `dc2`
optimizes is not the mapped-area metric this objective is about.

Also measured, all neutral: `emap` with `map_multioutput=1` reproduces the baseline area exactly on
`sin`/`arbiter`/`mem_ctrl`/`systemcaes`/`c7552`, and a larger cut limit only costs runtime.

### Per-cut decision: don't cares first, then the library and synthesis sources

The per-cut candidate set is:

1. **compute the cut's satisfiability don't cares** first (`compute_care_mask(ctx, true)`);
2. search **a single library cell and the four-input multi-cell library** together, all of them
   DC-aware, and keep the better implementation:
   * single cell — the library's precomputed **N/P supergate index**
     (`match_partial_truth_table(..., multi_cell=false, max_vars=6, single_entries)`): one hash probe
     per output phase and DC completion, replacing the brute force that was 60–70% of a pass.
     `EPMAPPING_SINGLE_MATCHER=brute` restores that brute force (any DC count),
     `hybrid` runs the index and falls back to it only on the cuts whose DC count exceeds
     `EPMAPPING_SINGLE_FREE`;
   * four-input multi-cell — `multi_cell_cover_dc` (`multi_cell=true, max_vars=4`);
3. only if neither library stage improved **and the cut is the MFFC cut** — the cut whose cone is
   exactly the target's MFFC (`is_mffc_cut`: `cone_nodes == mffc`), i.e. the cut whose replacement
   removes the whole MFFC — run **e-graph (egg) rewriting** on the cut function
   (`cut_mapping_egg_dc`, pure equality saturation, no database pre-check, no arrival input).
4. **structural synthesis** (source 10, `EPMAPPING_SYN`, on by default): synthesize the cut function
   algebraically (SOP factoring / DSD) instead of picking a cover, map it with the technology library
   and offer it to the same acceptance gate — see the section above.

**Objective: reduce area without worsening delay** (`EPMAPPING_NO_DELAY_REGRESSION=1` forces
`preserve_delay` on with a zero local slack, so no candidate may increase its node's arrival time; the
network's critical path is reported before and after every run and never grows).

Measured with `EPMAPPING_NO_DELAY_REGRESSION=1` (4 rounds):

| design | area | delay | substitutions | MFFC cuts reaching egg | cec |
| --- | --- | --- | --- | --- | --- |
| c7552 | 80.37 → **79.55** (−1.02%) | 871.71 → **871.71** (unchanged) | 288: single 68, multi 40, egg 176, dep cut 4 | 1 835 | 1 |
| sin | 274.08 → **273.67** (−0.15%) | 2554.02 → **2507.37** (improved) | 336: single 298, multi 25, egg 8, dep cut 5 | 4 890 | 1 |

Two ablations on `c7552` explain the numbers: gating egg **to the MFFC cut** makes it a small but real
win (78.87 → **78.80** at 5% local slack, 178 egg substitutions — unrestricted egg had measured
neutral-to-negative before), while the zero-slack rule costs 0.75 area (78.80 → 79.55) because the
winning substitutions on that design need the slack. The delay is preserved exactly in the zero-slack
configuration.

Input and output **phase** is considered by every source and paid for: both output polarities are
looked up (`get_supergates(fe)` and `get_supergates(~fe)`), each entry carries its input-negation mask,
the cone mapper loops over both phases as well, and `finalize_local_cover` charges `inv_cell_area_` once
per negated input plus once for an inverted output — those inverters are materialized, because the
emitted Verilog drops fanin complementation. Counters report it per run:
`epmapping phase: replacements with input negation=… with inverted output=…`.

Measured (paper baseline, shipped database, dependency cuts, 4 rounds):

| design | legacy sources | staged (DC first) | staged replacement mix |
| --- | --- | --- | --- |
| c7552 | 80.37 → **78.91** (36 s) | 80.37 → 79.35 (43 s) | single 51, multi 33, cone 31, dep cut 5 |
| sin | 274.08 → 273.48 (103 s) | 274.08 → **273.46** (157 s) | single 299, multi 25, cone 34, dep cut 5 |

> **Note.** The Rust `supergates.jsonl` cover cache referred to below was removed together with
> the legacy crate; the engine now takes multi-cell covers from the `.super` libraries
> (`EPMAPPING_SUPER`). These measurements are kept as history.

`supergates.jsonl` is queried on every cut but **is inert in every configuration measured** — its
candidates never win a substitution, and ablating it (pointing `EMAPPER_SUPERGATE_DB` at a missing
file so the database loads empty) leaves the result bit-identical:

| experiment (paper baseline, dependency cuts, `staged`, cone off) | results with `supergates.jsonl` | results with an empty database |
| --- | --- | --- |
| `c7552`, 4 rounds | 78.87, 130 substitutions, 1 091 candidate hits | **78.87**, 130, 0 hits |
| `c3540`, 4 rounds | 43.05, 28 substitutions, 208 hits | **43.05**, 28, 0 hits |
| `legacy`, `c7552`, 4 rounds | 78.91, 194 substitutions | **78.91**, 194 |
| `staged_rust` (removed flow) `EGG=1`, `c7552`, 1 round | 79.92, 3 002 egg calls, 345 hits | **79.92**, 3 005 egg calls, 0 hits |
| `EPMAPPING_SUPER=` (no native supergate library), `c7552` | 80.16, 864 hits, 0 wins | **80.16**, 0 hits |

So it is *queried* (36–1 091 candidate covers per design on `c432`/`c1908`/`c3540`/`c5315`/`c7552`/`bar`)
but never *chosen*: the native three-level ABC library dominates its 3 007 delay-oriented records, and
even with that library switched off its covers do not pass the caller's acceptance (area against the
MFFC plus the delay rule). It is therefore kept only as an option; the productive route to a useful
Rust-side database would be to regenerate it against this engine's cost model (area with materialized
inverters) instead of a delay objective.

**Stage 3 is what costs `staged` its area, and switching it off makes `staged` beat `legacy`:**

| c7552, 4 rounds | area | replacements | cone wins | covers found | time |
| --- | --- | --- | --- | --- | --- |
| `staged`, cone on (spec) | 79.35 | 120 | 31 | 8 610 | 43 s |
| `staged`, cone only when *no cover at all* | 79.35 | 120 | 31 | 8 610 | 43 s |
| **`staged`, cone off** | **78.87** | 130 | 0 | 8 817 | 43 s |
| `legacy` (reference) | 78.91 | 194 | — | — | 36 s |

So the cone mapper's covers are locally improving (each passes the area/delay test and the exhaustive
functional check) but they displace *better* library substitutions later: with stage 3 off, `staged`
performs **more** library substitutions (130 vs 120: single 79 + multi 45 + dependency cut 6) and ends
up 0.48 area better, on `c7552` beating `legacy` by 0.04 while staying equal on `sin` (273.46 vs
273.48). The wider reading of "若无" (only when no *cover* exists) also did not help (79.35), which is
what settled it: **the stage was removed**, because the structural-synthesis stage added later produces
the same candidate class with an explicit cost model. The row above is the historical measurement that
motivated the removal; the staged flow now measures (6 rounds)

| design | `legacy` (6 rounds) | `staged` + cone off (6 rounds) |
| --- | --- | --- |
| c7552 | 80.37 → 78.86 (37 s) | 80.37 → **78.82** (44 s) |
| sin | 274.08 → 273.48 | 274.08 → **273.46** (155 s) |

i.e. equal or slightly better than `legacy` on both, all `cec=1`.

**The full 27-design sweep has now run.** It used the recommended configuration (shipped database,
one DSE cycle) plus `EPMAPPING_NO_DELAY_REGRESSION=1`, i.e. the strict version of the objective
"reduce area without worsening delay":

| metric | `legacy` + DSE (default, `delay_slack=0.05`) | `staged` + NDR (`delay_slack=0`) |
| --- | --- | --- |
| δA1 (27 designs) | −2.95% | −1.67% |
| δA∞ (converged designs) | −3.09% (27) | −1.12% (25) |
| δD∞ | −1.06% | −3.01% (max +0.00%) |
| timeouts | 0 | **2** (`log2`, `mem_ctrl`) |
| `cec` | 1/1/1 everywhere | 1/1/1 everywhere it converged |

So the staged flow with the strict no-regression rule (a) buys **less area** than the verified default
and (b) is slow enough to blow the 1 800 s budget on `log2` and `mem_ctrl`, where the e-graph stage is
called thousands of times for a handful of wins (see the e-graph section below). Two things explain the
area gap: the local delay slack (`delay_slack=0` forbids *any* per-node arrival increase, while the
paper-style default allows 5%) and the staged flow's narrower source set on the `i2c`/`cavlc`/`router`/
`spi`/`systemcdes` designs, where the DSE cycle does most of the work. The per-design sweep table:

| design | base | r1 | δA1 | rN | δA∞ | δD∞ |
| --- | --- | --- | --- | --- | --- | --- |
| adder | 57.40 | 57.38 | −0.03% | 57.38 | −0.03% | −0.30% |
| bar | 133.63 | 124.86 | −6.56% | 124.86 | −6.56% | −10.53% |
| max | 137.61 | 135.48 | −1.55% | 135.48 | −1.55% | −7.81% |
| sin | 274.08 | 273.69 | −0.14% | 273.67 | −0.15% | −1.83% |
| log2 | 1455.42 | 1454.04 | −0.09% | — | **TIMEOUT** | — |
| div | 1184.83 | 1184.58 | −0.02% | 1184.58 | −0.02% | −0.03% |
| cavlc | 32.70 | 32.67 | −0.09% | 32.67 | −0.09% | +0.00% |
| multiplier | 1270.30 | 1269.12 | −0.09% | 1269.12 | −0.09% | −0.25% |
| sqrt | 1025.43 | 1024.58 | −0.08% | 1021.53 | −0.38% | −0.82% |
| ctrl | 5.80 | 5.71 | −1.55% | 5.71 | −1.55% | −8.82% |
| dec | 27.06 | 27.06 | +0.00% | 27.06 | +0.00% | +0.00% |
| arbiter | 557.72 | 557.72 | +0.00% | 557.72 | +0.00% | +0.00% |
| int2float | 10.90 | 10.38 | −4.77% | 10.38 | −4.77% | −20.95% |
| i2c | 57.12 | 57.06 | −0.11% | 57.06 | −0.11% | +0.00% |
| priority | 27.63 | 26.55 | −3.91% | 26.54 | −3.94% | −0.76% |
| router | 7.69 | 7.66 | −0.39% | 7.66 | −0.39% | −1.65% |
| sasc | 27.61 | 27.51 | −0.36% | 27.51 | −0.36% | +0.00% |
| iwls05_i2c | 49.19 | 46.93 | −4.59% | 46.93 | −4.59% | −5.35% |
| simple_spi | 36.62 | 36.26 | −0.98% | 36.26 | −0.98% | +0.00% |
| voter | 522.97 | 521.03 | −0.37% | 521.10 | −0.36% | −3.87% |
| usb_phy | 22.67 | 22.49 | −0.79% | 22.48 | −0.84% | −3.89% |
| c6288 | 132.60 | 132.60 | +0.00% | 132.60 | +0.00% | −0.03% |
| systemcaes | 476.89 | 473.93 | −0.62% | 473.74 | −0.66% | −5.04% |
| square | 1064.58 | 1062.56 | −0.19% | 1061.15 | −0.32% | +0.00% |
| systemcdes | 130.67 | 130.48 | −0.15% | 130.43 | −0.18% | +0.00% |
| spi | 147.65 | 147.64 | −0.01% | 147.62 | −0.02% | −3.31% |
| mem_ctrl | 2025.00 | 1669.55 | −17.55% | — | **TIMEOUT** | — |

Every completed run is ABC-CEC equivalent; delay never regresses (`max δD∞ = +0.00%`) on any design.
Reproduce with
`EPM_OUT=/tmp/epmap_staged EPM_JOBS=5 EPMAPPING_NO_DELAY_REGRESSION=1 ./scripts/compare_with_paper.sh 1800 <designs>`.

**Attributing the gap: it is the local delay slack, not the flow.** The verified default and this
sweep differ in two things — the flow (`legacy` vs `staged`) and `delay_slack`
(`EPMAPPING_NO_DELAY_REGRESSION=1` forces it from 0.05 to 0, forbidding *any* per-node arrival
increase). Running all four combinations on the six designs with the largest gap (single DSE cycle,
shipped database, dependency cuts) isolates them:

| design | `legacy` + slack 0.05 (verified default) | `staged` + slack 0.05 | `legacy` + NDR | `staged` + NDR |
| --- | --- | --- | --- | --- |
| i2c | 57.12 → 53.40 (−6.48%) | 57.12 → 53.40 (**−6.51%**) | 57.12 → 57.06 (−0.11%) | 57.12 → 57.06 (−0.11%) |
| cavlc | 32.70 → 31.35 (−4.13%) | 32.70 → 31.50 (−3.67%) | 32.70 → 32.69 (−0.03%) | 32.70 → 32.67 (−0.09%) |
| router | 7.69 → 7.26 (−5.59%) | 7.69 → 7.25 (**−5.72%**) | 7.69 → 7.66 (−0.39%) | 7.69 → 7.66 (−0.39%) |
| spi | 147.65 → 142.93 (−3.20%) | 147.65 → 144.00 (−2.47%) | 147.65 → 147.65 (+0.00%) | 147.65 → 147.62 (−0.02%) |
| systemcdes | 130.67 → 122.89 (−5.95%) | 130.67 → 122.86 (**−5.98%**) | 130.67 → 130.46 (−0.16%) | 130.67 → 130.43 (−0.18%) |
| sin | 274.08 → 266.46 (−2.78%) | 274.08 → 269.71 (−1.59%) | 274.08 → 273.71 (−0.13%) | 274.08 → 273.67 (−0.15%) |

The two NDR columns are indistinguishable, and so are the two slack columns except on `sin`/`spi`
(where `legacy` is better) and `i2c`/`router`/`systemcdes` (where `staged` is equal or better). The
5–6 percentage points on `i2c`/`cavlc`/`router`/`systemcdes` come **entirely from allowing each node's
arrival to grow by up to 5%**; with that slack the staged flow matches `legacy`, and without it both
flows collapse to ≈0.1%. So the staged flow is a legitimate alternative whose own contribution is
small and design-dependent — not the cause of the sweep's lower δA — and a "no delay regression"
objective is inherently worth ~2 percentage points of area on this suite. All 24 runs are `cec=1/1/1`;
the slack columns do increase delay (`i2c` 287.68 → 306.69, `sin` 2554.02 → 2473.69).

### Egg (e-graph) rewriting

The stage flattens the cut's **gate-level cone to an AIG** (`gate_expr_to_aig_sexp`, using each cell's
genlib `Expr`), saturates it with the **one-way `AIG -> cell` cover rules**
(`EggRewriteBundle::new_cover_only` adds no `gate -> AIG` expansion), and extracts the cheapest cover
per e-class. The library cells and the AIG nodes therefore live in the *same* e-graph, and the
extraction itself picks the cell cover — this is **not** a second technology-mapping pass over an
extracted AIG.

Pricing is `(area, real library delay)`, the only model wired through `cut_mapping_egg_arr`: the
`depth` variant that `EPMAPPING_EGG_COST` used to advertise was never reachable from the staged entry
point (the cost-model argument was ignored, and `both` simply ran the same saturation twice), so the
knob was removed. `EPMAPPING_EGG_REQ` (default on) hands the e-graph the cut leaves' arrivals *and*
the cone's original delay as an output bound, turning "minimum area" into "minimum area subject to no
delay regression". Neither model decides acceptance: `consider` recomputes every candidate's delay
from the netlist's own arrivals (`cover_delay`) before comparing.

**The extraction is memoised.** `cut_mapping_egg_arr` / `cut_mapping_egg_dc` now go through
`cached_cut_mapping`, keyed on `(expression, objective, required_output_delay, arrivals)`; the staged
flow asks for the same cut function once per root, and the hit rate is high — 84% on `tv80`
(9 502 / 11 292) and 88% on `aes_core` (36 873 / 41 989) — which cuts the stage's wall clock from
16.95 s to 3.52 s and from 81.2 s to 14.2 s respectively (`map_egg paths ... egg_cache_hits=` reports
it). The cache also makes the stage reproducible: a repeated expression returns the cover that was
computed first instead of a fresh extraction.
That is why switching the staged stage-3 model is invisible in the result — single round, stage 3
active, `EPMAPPING_NO_DELAY_REGRESSION=1`:

| design | `lib` (default) | `depth` | `both` | `lib` + arrivals + delay bound |
| --- | --- | --- | --- | --- |
| c7552 | 80.34 | 80.34 | 80.34 | 80.34 |
| c3540 | 43.18 | 43.18 | 43.18 | 43.18 |
| c5315 | 64.61 | 64.61 | 64.61 | 64.61 |
| c2670 | 26.86 | 26.86 | 26.86 | 26.86 |

The same holds for the *search budget*, which is decided per cut and never reaches saturation
truncation on these cuts: `MAP_EGG_EGG_ITER_LIMIT` 2 → 5/10, `MAP_EGG_EGG_NODE_LIMIT` 500 → 2 000/5 000,
and the full bidirectional bundle (`MAP_EGG_FULL_BUNDLE=1`, `cell -> AIG` expansion on top of
`AIG -> cell`) all leave `c7552` at 80.34 and `c3540` at 43.18. The extractor's alternatives are only
root e-class **ties** under the generator's cost (`choice_optimal_tie` in
`extract_mapped_cover_covers_vec`) — and a stage-3 cut already runs only when the library stages found
no cover at all.

What *does* move the needle is the candidate **source set**. Running the multi-solution e-graph source
on every large cut (not only on the gated MFFC cut) is where the e-graph rewriting pays:

| design | staged (default) | `EGG_MULTI=2 STAGED_EGG_ALL=1`, depth model | same, `EGG_MULTI_MODEL=lib` |
| --- | --- | --- | --- |
| c7552 | 80.34, delay 871.71, 7 s | **80.32**, 871.71, 77 s | **80.32**, 871.71, 76 s |
| c3540 | 43.18, delay 612.23, 4 s | **43.14**, 612.23, 28 s | **43.14**, **604.47**, 27 s |
| c5315 | 64.61, 552.90, 3 s | 64.61, 552.90, 36 s | 64.61, 552.90, 36 s |
| c2670 | 26.86, 299.59, 2 s | 26.86, 299.59, 9 s | 26.86, 299.59, 8 s |

Three things are worth reading off this table. First, the extra source reproduces its gains with
`limit=2` (`=4` is identical) and is stable over repeats at the 0.01 area level. Second, the **library
model is the better one here**: it matches the depth model on area everywhere and additionally cuts
`c3540`'s critical delay from 612.23 to 604.47 (−1.3%), because an area-first DP over real pin delays
finds a same-area, faster cover while a depth-blind DP cannot see the difference. Third, the price is
the e-graph runtime — roughly 10× the staged flow's — which is why the configuration stays **opt-in**.


Where that runtime goes is visible in the per-design counters of the staged flow (single round, paper
baseline with the shipped database):

| design | MFFC cuts that reached e-graph stage 3 | e-graph wins | single-cell wins | multi-cell wins | r1 time |
| --- | --- | --- | --- | --- | --- |
| c7552 | 464 | 46 | 7 | 0 | 9 s |
| sin | 1 527 | 7 | 187 | 16 | 70 s |
| div | 3 673 | 14 | 943 | 0 | 731 s |
| multiplier | 3 805 | 9 | 1 628 | 13 | 396 s |
| log2 | 5 999 | **1** | 884 | 114 | 142 s (rN times out at 1 800 s) |

thousands of saturations for single-digit wins on the arithmetic designs. The reward is bounded by the
area of the MFFC being replaced, so the stage can be gated on it: `EPMAPPING_EGG_MIN_MFFC=<area>` skips
the stage when `context.original_area` is below the threshold. A threshold of `1.0` blocks **every**
stage-3 call on `c7552` and `c3540` (all their e-graph-reaching MFFCs are below one area unit, i.e. at
most ~10 cells), leaving `c7552` unchanged at 80.34 and costing `c3540` 0.04 (43.18 → 43.22). So the
gate is a runtime-for-area trade, not a free win, and it defaults to `0` (always run).

One implementation fix came out of this exploration:

* `materialize_mapped_cover_from_best` no longer clones the whole DP map for every root alternative;
  the alternatives path is a plain lookup for non-root classes.

A caveat on the counters: the e-graph path breaks equal-cost ties in the order it happens to visit the
cover DAG's classes, and `egg` itself uses hash maps internally, so the per-source counts are *not*
exactly reproducible — five identical `c7552` runs give `egg=45…48` while the area stays at 80.34 and
the delay at 871.71. Sorting the cover-DAG class/worklist order removed part of it, and the reported
area/delay (0.01 resolution) has been stable in every repeat measured, but treat single-substitution
counters as indicative rather than exact.

### Design-space exploration (the paper's Section IV-D) is the largest single lever

The sweep scripts run **design-space-exploration cycles**, exactly as the paper describes in
Section IV-D ("after 5 iterations ... we unmap the network resulting in an AIG, remove redundancies,
and balance it. Finally, we map back to technology and restart area-oriented optimization"): take the
optimized mapped netlist, unmap it to an AIG, re-optimize it, map and optimize again, and keep the
smaller result. `EPMAPPING_DSE=0` disables it, `EPMAPPING_DSE_CYCLES=n` iterates it, and
`EPMAPPING_DSE_PIPES` selects the unmap pipelines (default `plain dc2`, i.e. both are tried and the
better is kept). The chosen netlist is CEC-checked against the **original** benchmark, never against an
intermediate AIG, so `cec=1` still means "equivalent to the input design".

The pipeline matters, and it is not monotone, so the cycle is a portfolio:

| unmap pipeline (after `strash`) | bar | mem_ctrl | square | spi | log2 |
| --- | --- | --- | --- | --- | --- |
| `fraig; compress2rs; resyn2rs` (`plain`) | 126.70 | 1784.91 | **1064.29** | 148.68 | 1490.10 |
| **`fraig; dc2; compress2rs; resyn2rs` (`dc2`)** | **124.86** | **1674.47** | 1089.51 | **142.94** | **1464.56** |
| `fraig; balance; …` / six scripts | 126.70 | 1786.10 | 1064.29 | 148.56 | — |

`dc2` (ABC's don't-care-based simplification) is worth another 1.5–6.6% on most designs but costs 2.4%
on `square`, which is why both pipelines are tried. Verified effects of one cycle on the direct result:

| design | direct (`r1`) | one `plain` cycle | one portfolio cycle | paper δA1e |
| --- | --- | --- | --- | --- |
| mem_ctrl | 2019.99 (−0.25%) | 1791.56 (−11.5%) | **1674.47 (−17.3%)** | −5.56% |
| bar | 133.63 (0%) | 126.62 (−5.25%) | **124.86 (−6.56%)** | — |
| sin | 273.52 (−0.20%) | **269.42 (−1.70%)** | — | −0.24% |

This is a *mapped-design* transformation (it starts from our own optimized netlist), it is the paper's
own mechanism, and it is not merely more AIG preprocessing: four iterations of `resyn2rs`/`compress2rs`
on the **original** benchmark AIG only reach 1941.75 for `mem_ctrl`, while the unmapped *mapped*
netlist reaches 1795.36 before the post-mapping pass even runs, and 1674.47 with `dc2` in the loop.

The loop converges after one cycle for a fixed pipeline (a second `plain` cycle reproduces bar's
126.62 and sin's 269.42 exactly), which is why the one-pass and converged columns of the table below
are close.

### The sub-network database is the strongest lever, and it was incomplete

`tech_library<9>::get_supergates(tt)` only knows single gates unless a supergate library is loaded.
The database's *coverage* decides which cut functions can be matched at all, and `EPMAPPING_DB_STATS=1`
measures it over all 65,536 four-input functions:

| database | ABC command | structures | file | 4-input coverage | load | c7552 (with dependency cuts) |
| --- | --- | --- | --- | --- | --- | --- |
| single gates only | — | 194 pins | — | 0 for multi-cell functions | 0 s | 80.33 (−0.05%) |
| 2-level (previous default) | `super -I 4 -L 2 -T 60` | 99,510 | 3.2 MB | 15,664 / 65,536 (24%) | 5 s | 80.04 (−0.41%) |
| 3-level, area ≤ 0.45 | `super -I 4 -L 3 -A 0.45 -T 240` | 227,119 | 7.1 MB | 22,378 (34%) | 5 s | 79.72 (−0.81%) |
| **3-level, area ≤ 0.50 (shipped)** | `super -I 4 -L 3 -A 0.50 -T 240` | 504,368 | 16 MB | 29,256 (45%) | 34 s | **78.91 (−1.82%)** |
| 3-level, area ≤ 0.55 | `super -I 4 -L 3 -A 0.55 -T 240` | 976,556 | 32 MB | — | 102 s | 79.19 (−1.47%) |
| 3-level, area ≤ 0.60 | `super -I 4 -L 3 -A 0.60 -T 240` | 1,441,668 | 48 MB | — | 199 s | 79.17 (−1.49%) |
| 3-level, no area cap | `super -I 4 -L 3 -T 240` | 1,201,636 | 43 MB | 65,388 (99.8%) | 240 s | — |

Two things stand out. Three quarters of all four-input functions had **no entry at all** in the
previous default, so their cuts could not be matched no matter how good the rest of the engine was.
And the effect of a richer database is *not* monotone (0.50 beats 0.55 and 0.60): the database decides
which candidates exist and in which order they are tried, so it steers the search, not just the
matching. The shipped database is the one that measured best, and it makes `c7552` beat the paper
(−1.82% here vs −1.08% in their Table III and −0.76% in their Table II).

Its cost is the load time (34 s per invocation, one library build per run) and a longer search; the
sweep scripts therefore run with a 1500 s budget per configuration.

### Measured: the pre-mapped sub-network database (`EPMAPPING_SUPER`) plus don't cares

`tech_library<9>::get_supergates(tt)` only knows single gates unless a **supergate library** is
loaded: with an empty spec, `_super_lib` holds just the library's pins (that is why the egg path used
to be the only source of multi-cell covers). ABC can precompute the database the paper describes
("a database of mapped sub-networks for completely-specified functions"):

```sh
abc -c "super -I 4 -L 2 -T 60 -o asap7.genlib"     # writes asap7.super next to the genlib
EPMAPPING_SUPER=$PWD/epmapping/db/asap7_4in_2lvl.super build/epmapping/epmapping_check ...
```

`epmapping/db/asap7_4in_2lvl.super` (3.2 MB, 99 510 supergates, ≤4 inputs, ≤2 levels; `-I 4 -L 3`
gives 346 860 supergates and the same results at 5× the load time) is the shipped database. It is
loaded into a **second** `tech_library<9>` used only by the optimizer, so the `emap` baseline is
byte-identical to the run without it. This is the sweep scripts' recommended configuration and the
database turns the plateau into small but real gains, all `cec=1` (paper baseline, default 4 passes):

| design | base area | opt area | δA | base delay | opt delay | δD | runtime |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c3540 | 43.28 | **43.17** | −0.25% | 612.23 | **576.32** | −5.87% | 2.5 s |
| c7552 | 80.37 | **80.24** | −0.16% | 871.71 | 871.71 | 0.00% | 2.2 s |
| sin | 274.08 | **273.45** | −0.23% | 2554.02 | **2532.12** | −0.86% | 59 s |

The cut don't cares come from the window simulation by default (`EPMAPPING_SAT_DC=1` sharpens them with
SAT) and are handed to the
DC-aware multi-cell match and to the synthesis stage.

### Measured: the egg path is a runtime wall with almost no area payoff

The dispatch is by cut size: cuts with `leaves ≤ small_cut_size` (default 4) go to
`cut_mapping_library` and larger cuts (5–6 leaves) go to `map_egg`. That split is expensive because
the library path is **not** a multi-cell mapper: it is a *single-cell* match
(`map_tt_to_library` → `best_for_tt_phase_arrivals`) plus a lookup in the precomputed
`supergates.jsonl` database, which only covers ~938 distinct `(n, tt)` keys. Any 5–6-input cut whose
cover needs more than one cell (or is not in that small cache) therefore falls through to egg.

On the paper's area-oriented baseline (the default ABC preprocessing, `EPMAPPING_AREA_MAPPING=1`,
`EPMAPPING_RELAX_REQUIRED=100`, `EPMAPPING_OBJECTIVE=1`), single round:

| design | library only (`SMALL_CUT=6`) | default (egg for 5–6 leaves) | egg calls |
| --- | --- | --- | --- |
| div | 11 s, 1184.83 → 1184.83 (0%) | 814 s, 1184.83 → **1184.74** (−0.01%) | 87 369 |
| log2 | 22 s, 1455.42 → 1455.42 (0%) | 732 s, 1455.42 → **1454.12** (−0.09%) | 74 723 |
| multiplier | — | 394 s, 1270.30 → **1269.12** (−0.09%) | 36 348 |
| sin | — | 199 s, 274.08 → **273.77** (−0.11%) | 21 654 |
| max | — | 8 s, 137.61 → 137.61 (0%) | 477 |
| c499 | 0.3 s, 22.79 → 22.79 (0%) | 5 s, 22.79 → 22.79 (0%) | 852 |
| c7552 | 2 s, 80.37 → 80.33 (−0.05%) | 63 s, 80.37 → **80.32** (−0.06%) | 3 843 |

So on the paper's baseline the egg path buys **at most 0.01 percentage points** (c7552), and 0% on
c499, while costing 6–14 minutes per pass on the arithmetic designs and blowing the 900 s budget for
a second pass. The apparent "egg is essential" result came from the *lighter* subject graph
(`EPMAPPING_ABC_SCRIPT="resyn2rs; compress2rs"`): there c499 goes 43.11 → 40.87 (−5.20%) with egg and
0% without it, and `MAP_EGG_EGG_NODE_LIMIT` 100 / 150 / 200 / 300 / 500 gives
0% / −0.07% / −5.20% / −5.20% / −5.20% — i.e. that gain was the engine re-discovering redundancies
`fraig` removes for free. On the paper's baseline the whole NODE_LIMIT range gives 0% on c499.

That measurement is why the default changed: egg is now behind `EPMAPPING_EGG=1` and every cut is
answered by the native library cover (see the section above), which is ~10× faster at −0.02% instead
of −0.06%.

### Measured: raising the leaf limit (7–9 leaves) buys nothing

The leaf count is capped at 6 in three independent places: `params::cut_size`, the `Tt9` truth table
in `map_egg` (`[u64; 8]` = 512 bits = 9 inputs, and cut leaves are named `a..i`), and `care_mask`
(`u64` = 64 minterms = 6 inputs, so DC is silently disabled above 6). `mockturtle::cut_enumeration`
itself allows up to `max_cut_size = 16`, and `block_network` up to 32 fanins, so the binding limit is
the 9-input truth table. `EPMAPPING_CUT_SIZE` may be raised up to 9 (larger values are clamped with a
warning); defaults are unchanged.

Area-oriented paper baseline, one pass, `CUT_SIZE=6` vs `9`:

| design | cuts | egg calls | area | time |
| --- | --- | --- | --- | --- |
| c432 | 382 → 1017 | 150 → 568 | 6.61 → 6.61 / **6.61 → 6.61** | 1 s → 2 s |
| c499 | 2059 → 2159 | 852 → 932 | 22.79 → 22.79 / **22.79 → 22.79** | 8 s → 8 s |
| c7552 | 11403 → 13543 | 3843 → 5165 | 80.37 → 80.32 / **80.37 → 80.32** | 39 s → 44 s |
| sin | 34292 → 43347 | 21654 → 30690 | 274.08 → 273.77 / **274.08 → 273.77** | 136 s → 200 s |

(the two columns after "area" are `CUT_SIZE=6` and `CUT_SIZE=9`.) `library_hits` is **identical** in
both runs while `egg_calls` grows by exactly the number of new 7–9-leaf cuts: asap7's widest cell has
5 inputs and `supergates.jsonl` only has ≤6-input classes, so every extra cut can only be answered by
egg — which already buys ≈0 at 5–6 leaves. On the designs where egg is the wall (div/sqrt/log2/
multiplier, >900 s already at `CUT_SIZE=6`) it would only make the wall taller. The structural
frontier is exhausted; the paper's remaining area needs different cuts (non-local/dependency), not
bigger ones.
