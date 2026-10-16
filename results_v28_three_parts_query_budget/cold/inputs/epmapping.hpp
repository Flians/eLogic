#pragma once

#include "objective.hpp"
#include "bounded_cache.hpp"
#include "exact_aig_key.hpp"
#include "live_query_cache.hpp"
#include "local_truth_table.hpp"
#include "operator_policy.hpp"
#include "map_egg/src/lib.rs.h"
#include "rust/cxx.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <fstream>
#include <limits>
#include <memory>
#include <array>
#include <numeric>
#include <deque>
#include <deque>
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <chrono>
#include <cstring>
#include <vector>

#include <kitty/dynamic_truth_table.hpp>
#include <kitty/operations.hpp>
#include <mockturtle/algorithms/cut_enumeration.hpp>
#include <mockturtle/algorithms/dont_cares.hpp>
#include <mockturtle/algorithms/emap.hpp>
#include <mockturtle/algorithms/node_resynthesis/dsd.hpp>
#include <mockturtle/algorithms/node_resynthesis/exact.hpp>
#include <mockturtle/algorithms/node_resynthesis/sop_factoring.hpp>
#include <mockturtle/algorithms/reconv_cut.hpp>
#include <mockturtle/networks/aig.hpp>
#include <mockturtle/networks/block.hpp>
#include <mockturtle/utils/standard_cell.hpp>
#include <mockturtle/utils/tech_library.hpp>
#include <mockturtle/views/cell_view.hpp>
#include <mockturtle/views/topo_view.hpp>

/* ABC's `bsat2` through `bill`: the don't-care refinement below is a SAT query, not a simulation. */
#include <bill/sat/interface/abc_bsat2.hpp>

namespace epmapping
{

/// The mapped gate-level network the optimizer rewrites.
using ep_network = mockturtle::cell_view<mockturtle::block_network>;

struct params
{
  uint32_t cut_size{ 6u };
  /// Cuts kept per node (`EPMAPPING_CUT_LIMIT`). **12**: the cost of a larger limit is that the
  /// generic `merge_cuts` of `cut_enumeration` only calls `rcuts.limit()` *after* enumerating the
  /// whole mixed-radix tuple space, so the intermediate list grows like the product of the children's
  /// cut-list sizes (`cut_limit^k` for a k-input cell) and the insert/dominance checks are linear in
  /// it -- on 4-input cells the per-round cost explodes combinatorially with the limit. Measured on
  /// `systemcdes` (1 round): `cut_enum` 214.8 s at 20, 17.3 s at 12 and 2.0 s at 8, with identical
  /// mapped area at 12 (130.87) and +0.015% at 8. 12 is the runtime knee.
  ///
  /// The cap is also a hard constraint now: the live cut database is mockturtle's
  /// `dynamic_network_cuts`, whose compile-time `max_cut_num` is 16, so `EPMAPPING_CUT_LIMIT > 15`
  /// is clamped (with a warning) rather than silently overflowing the cut set. The default
  /// is that maximum, **15**; pass `EPMAPPING_CUT_LIMIT=12` when runtime matters and the extra cuts
  /// buy nothing.
  uint32_t cut_limit{ 15u };
  /// 0 = delay first, 1 = area first (matches `map_egg`'s objective encoding).
  /// Area-first is the default because the engine's reference flow is ASPDAC'25's area-oriented
  /// optimization; `EPMAPPING_OBJECTIVE=0` restores the delay-first acceptance.
  uint8_t objective{ 1u };
  /// Transactional round guard for the measured whole-network lexicographic objective.
  bool global_objective_guard{ false };
  /// Experimental cut ranking by net area change, rather than replacement area alone.
  bool rank_area_gain{ false };
  /// Use live output required times to spend slack on noncritical cones.
  bool use_required_times{ false };
  /// Minimize area under the fixed initial output delay; takes precedence over path priorities.
  bool area_under_delay{ false };
  /// Reuse physical inverter inputs when materializing opposite phases.
  bool phase_reuse{ true };
  /// Experimental live-context pricing of physical phase reuse, without mutating the network.
  bool phase_cost{ false };
  /// Also reuse a live inverter fanout, including shared gates inside the root's fanin.
  bool phase_fanout{ false };
  /// Experimental cut-only alternatives: 1=output-phase factoring, 2=Shannon, 4=Davio.
  uint32_t cofactor_modes{ 0u };
  uint32_t cofactor_min_leaves{ 3u };
  bool synth_cache_replace{ true };
  bool cofactor_dedup{ true };
  bool cofactor_proof_cache{ true };
  /// Zero preserves exhaustive pivots. Nonzero is an experimental per-family budget.
  uint32_t cofactor_pivots{ 2u };
  /// 0=off, 1=measure operators, 2=experimental adaptive new-build scheduling.
  uint32_t operator_policy_mode{ 0u };
  /// 0=off, 1=measure exact AIG aliases, 2=reuse their immutable mapped covers.
  uint32_t build_reuse{ 0u };
  bool shared_mapped_library{ false };
  /// Exact live-query memoization: 1=inverse search, 2=removable set, 4=dynamic TT reuse, 8=predecoded local simulation.
  uint32_t live_query_memo{9u};
  /// Seconds between checked in-round snapshots; zero disables in-round snapshots.
  double checkpoint_interval{ 30.0 };
  /// Canonical, globally checked snapshot, completed rounds, incomplete-round flag.
  std::function<void(ep_network const&, uint32_t, bool)> checkpoint;
  /// Critical cones optimize delay; other cones recover area under the live delay target.
  bool path_objective{ false };
  double critical_slack{ 1e-3 };
  /// Relative whole-network growth budgets from the initial mapping; negative disables.
  double max_area_growth{ -1.0 };
  double max_delay_growth{ -1.0 };
  /// Optional relative area-growth limit on each replaced MFFC.
  double local_area_growth{ -1.0 };
  /// When true, a candidate is rejected if its structural depth exceeds the original gate's.
  bool preserve_depth{ true };
  /// When true, a candidate is rejected if its library-timing delay exceeds the original gate's
  /// by more than [`params::delay_slack`] (relative).
  bool preserve_delay{ true };
  /// Relative local-delay slack allowed for a replacement when [`params::preserve_delay`] is set.
  /// `0` forbids any local delay regression; larger values trade delay for area.
  double delay_slack{ 0.05 };
  /// Reverse-topological passes (**6**, the `rN` configuration of the reported tables; each pass
  /// rebuilds the network in topological order first, because new nodes are appended after their
  /// parents, which would otherwise break `cut_enumeration`). Iteration stops early on a pass that
  /// makes no replacement or no objective progress, so the cap only binds on designs that keep
  /// improving.
  uint32_t max_rounds{ 6u };
  /// Run the e-graph (egg) stage (`EPMAPPING_EGG`, **on by default**).
  ///
  /// The staged flow used to run this stage unconditionally while the flag only fed the banner, so
  /// `EPMAPPING_EGG=0` did not actually disable it; the flag now gates the stage and defaults to the
  /// behaviour that was really in effect. Measured over 11 designs, removing the stage is worse on
  /// every design where it matters (`tv80` +0.58, `aes_core` +0.21, `div` +0.21, `systemcdes` +0.02,
  /// `sasc` +0.01) and better only on `mem_ctrl` (-0.54): cumulative +0.49, the smallest of the three
  /// optional stages.
  bool use_egg{ true };
  /// Largest number of don't-care minterms enumerated per dependency cut (2^this completions).
  uint32_t dc_max_free{ 4u };
  /// Which matcher answers the staged single-cell stage (`EPMAPPING_SINGLE_MATCHER`):
  /// * `index` (default) -- the library's precomputed N/P supergate index (`get_supergates`): one
  ///   hash probe per output phase plus one per DC completion. `tech_library::generate_library()`
  ///   seeds every genlib cell with all its input permutations and polarities, so this explores the
  ///   same configurations the brute force does, but it can only match cuts whose DC count is at most
  ///   `params::single_free` (it enumerates completions, `nullopt` beyond that).
  /// * `brute` -- the original brute force (all cells x `pins!` permutations x `2^pins` polarities x
  ///   2 output phases over every care minterm). Handles any DC count. This was 60-70% of a pass
  ///   (`arbiter` 213.7 s, `div` 187.6 s, `mem_ctrl` 391.1 s, one round) and is kept as the reference.
  /// * `hybrid` -- `index`, then the brute force on the cuts the index cannot enumerate (DC count
  ///   above `params::single_free`): the candidate set is a superset of `brute`'s at a fraction of its
  ///   cost.
  enum class single_matcher : uint8_t
  {
    index = 0u,
    brute = 1u,
    hybrid = 2u
  };
  single_matcher single{ single_matcher::index };
  /// Supergate entries tried per output phase and DC completion by the staged single-cell stage
  /// (`EPMAPPING_SINGLE_ENTRIES`). The library list for a key is ordered by ascending area, so the
  /// historic dependency-cut budget (`params::dc_entry_limit`, 4) kept the cheapest entries but could
  /// drop a better delay/level one; `c_exact_entry_limit` (128) is what the exact (DC-free) branch
  /// already uses, which makes the DC-bearing cuts see the same candidate list.
  uint32_t single_entries{ 128u };
  /// Free minterms the staged **single-cell** stage may enumerate (`EPMAPPING_SINGLE_FREE`), kept
  /// separate from `params::dc_max_free` (4) because the stages fail differently: the multi-cell
  /// database stage walks every entry of a completion, while the single-cell index lookup is a hash
  /// probe. Measured neutral on `spi`/`tv80` (6 and 8 give the same area as 4), because the cuts above
  /// the limit carry >=8 free minterms and no completion enumeration reaches them -- that is what
  /// `single_matcher::hybrid` is for.
  uint32_t single_free{ 4u };
  /// Supergate entries tried per completion and phase for a partial function.
  uint32_t dc_entry_limit{ 4u };
  /// Wall-clock budget (seconds) for the whole don't-care computation of a run
  /// (`EPMAPPING_DC_TIME_BUDGET`, 0 = unbounded): once spent, the remaining cuts are matched exactly
  /// (no DCs). Sound -- it only gives up don't cares -- and it bounds the staged flow's runtime on
  /// designs with large windows (`sqrt`, `div`, `mem_ctrl`), where `compute_care_mask` otherwise
  /// spends 50-60% of a pass.
  double dc_time_budget{ 0.0 };
  /// Which window the satisfiability don't cares are computed on (`EPMAPPING_DC_WINDOW`):
  /// * `per_cut` -- one reconvergence window per **cut**, grown from the cut's own leaves;
  ///   this is mockturtle's `satisfiability_dont_cares(ntk, leaves)` (the variant `cut_rewriting`,
  ///   `refactoring` and `resubstitution` call). Every cut gets a care set, the mask is a pure function
  ///   of the leaf set (so the per-round `care_key` memo applies and is shared across roots), and the
  ///   free boundary is the reconvergence cut of exactly those leaves; the price is one window plus one
  ///   simulation per cut (~7 400 windows against ~2 400 roots on `sin`).
  /// * `per_node` (**default** since 2026-09) -- one reconvergence window per **root**, simulated once
  ///   and shared by all of that
  ///   root's cuts; this is mockturtle's `rewrite` (`rps.max_leaves = ps.window_size = cap`, so the
  ///   creation limit is the simulation cap and no window is discarded). Cheaper in window work, but a
  ///   cut whose leaves are not all inside the window gets **no** don't cares (upstream's containment
  ///   fallback, 5-36% of the cuts measured) and the leaf-set memo is bypassed because the mask then
  ///   depends on the root.
  ///
  /// Measured over nine designs (A_e, one pass, strict NDR, SAT off, `cec=1`, identical delays): the two
  /// modes give the **same area on 7 of 9** and `per_node` is 0.01% *worse* on `aes_core` and `sin`, while
  /// being 2.2-5.9x cheaper in window work and 1.6-2.9x cheaper in don't-care time (1.08 -> 0.40 s on
  /// `aes_core`, 1.66 -> 0.57 s on `div`). `per_node` is the default because that DC time is the only
  /// cost the mode moves; the 0.01% area difference is the price a sweep accepts for it. See
  /// results/dc_window_ab.txt for the full table.
  enum class dc_window_mode : uint8_t
  {
    per_cut = 0u,
    per_node = 1u
  };
  dc_window_mode dc_window{ dc_window_mode::per_node };
  /*! Hand the egg stage the cut leaves' real arrivals **and** the cut cone's original delay as an
      output-delay bound (`EPMAPPING_EGG_REQ=1`), turning it from "minimum area" into "minimum area
      subject to no delay regression". Worthless without arrivals, so it implies `lib`. */
  bool egg_required{ true };
  /*! Only run the e-graph stage when the MFFC it would replace covers at least this much area
      (`EPMAPPING_EGG_MIN_MFFC`, default `0` = always). A saturation costs the same for every cut and
      the reward is bounded by the MFFC area, yet on the arithmetic EPFL designs thousands of
      saturations are spent on MFFCs of one or two cells and buy nothing at all. */
  double egg_min_mffc{ 0.0 };

  /*! "Reduce area without worsening delay": forces the local delay slack to zero, so no candidate may
     increase its node's arrival time (`EPMAPPING_NO_DELAY_REGRESSION=1`). */
  bool no_delay_regression{ false };
  /// Exhaustively compare the winning candidate's function against the replaced cone before
  /// substituting (at most 64 minterms). Cheap, and it turns any cover-conversion bug into a
  /// rejected candidate instead of a wrong netlist.
  bool validate_candidates{ true };
  /// Disable all cut don't cares for a genuine fully-specified-function ablation.
  bool use_dont_cares{ true };
  /// Memoize context-independent mapped SOP/DSD covers; costs are recomputed on every use.
  uint32_t synth_cache_entries{ 8192u };
  /// Structural-synthesis candidate stage (`EPMAPPING_SYN`; single gate: `0` = off, any non-zero
  /// value = on, which is the default).
  ///
  /// Offered on every cut at least `synth_min_leaves` wide. The stage re-synthesizes the cut function
  /// algebraically (SOP factoring, optionally with the cut's don't cares), maps the resulting network
  /// with the same technology library, and offers the cover to the normal acceptance gate. This is
  /// a candidate class the library stages cannot produce. Its scope is still one cut: on `arbiter`,
  /// even wider cuts and exact local synthesis give no gain from the original mapping. Global
  /// AIG `syn4`/`sopb` preprocessing changes the mapped starting structure and is a separate stage.
  uint32_t synth{ 1u };
  /// Minimum cut width (leaves) for the synthesis stage (`EPMAPPING_SYN_MIN_LEAVES`).
  uint32_t synth_min_leaves{ 4u };
  /// Enabled synthesis modes (`EPMAPPING_SYN_MODE`): bit 0 = SOP factoring, bit 1 = disjoint-support
  /// decomposition with factoring on the prime nodes, bit 2 = exact AIG synthesis (percy/SAT,
  /// minimum AND count, don't cares honoured). Both cheap engines are on by default and the
  /// acceptance gate keeps whichever cover is better: which one wins is design dependent
  /// (`c3540`/`c432` prefer SOP factoring, `c5315`/`c2670` DSD), and `SYN_MIN_LEAVES=4 SYN_MODE=3`
  /// measured `c432` 0% -> -0.61%, `c3540` -0.62% -> -1.50% and `c5315` -0.16% -> -0.89%.
  uint32_t synth_modes{ 3u };
  /// Fanin limit of the exact mode (`EPMAPPING_SYN_EXACT_LEAVES`): percy is practical up to four.
  uint32_t synth_exact_leaves{ 4u };
  /// SAT conflict limit of the exact mode (`EPMAPPING_SYN_EXACT_CONFLICT`); a timeout is not a
  /// candidate (the stage simply contributes nothing).
  uint32_t synth_exact_conflict{ 200000u };
  /// Restrict the exact mode to the MFFC cut (`EPMAPPING_SYN_EXACT_MFFC`, default on).
  bool synth_exact_mffc{ true };
  /// Whole-window structural synthesis over the MFFW of a root's cut leaves (`EPMAPPING_WINDOW_SYN`).
  ///
  /// 1 = on (**the default**), 0 = off. The window is rebuilt as a whole (every output re-synthesized
  /// from the window inputs, mapped jointly) and all outputs are substituted at once, so the
  /// window's internal nodes really die — a single-output re-cover measured 0-1 wins because the
  /// surviving fanouts kept them alive. Window outputs are *observable*, so they must be exactly
  /// equivalent: no don't cares are used here (unlike the cut stages). Only tried for a root whose
  /// cut stages produced no replacement.
  ///
  /// **On by default.** An 11-design ablation (`EPMAPPING_ROUNDS=6`, `EPMAPPING_SUPER`, NDR, DC budget
  /// 60) that removes one stage at a time leaves it *better* than the full flow on every design where
  /// it makes a difference: `mem_ctrl` +4.44, `tv80` +0.42, `div` +0.26, `usb_phy` +0.09, `spi` +0.06,
  /// `aes_core` +0.03, `sasc` +0.01 (cumulative +5.31), 0.00 elsewhere. It costs 5-63 s per design.
  uint32_t window_synth{ 1u };
  /// Removable window area (library units) needed before a window is synthesized (`θ`).
  double window_synth_min_gain{ 0.0 };
  /// Window input / gate caps (`EPMAPPING_WINDOW_SYN_LEAVES` defaults to `window_max_leaves`).
  uint32_t window_synth_max_leaves{ 6u };
  uint32_t window_synth_max_gates{ 64u };
  /* The window stage is a **fallback**: it is tried only for a root whose cut stages produced
     nothing. Measured on `c5315`: applying the window first (even without skipping the cut synthesis)
     ends at −0.44% while the cut stages alone reach −0.78%, because a committed window changes the
     network the later cut stages see. As a fallback it can only add where they found nothing
     (`c5315` −0.78% → −1.14%, `c3540` −1.50% → −1.60%). */
  /// Give the synthesizer the cut's satisfiability don't cares (`EPMAPPING_SYN_NO_DC=1` disables).
  bool synth_use_dc{ true };
  /// SAT-based don't-care refinement (`EPMAPPING_SAT_DC`), **off by default**: a minterm that the
  /// window simulation reports as *care* is a satisfiability don't care exactly when the fanin cone of
  /// the cut's leaves cannot produce it; SAT proves that, which lifts the `2^n_pis` window cap (it is
  /// the only DC source for the 54% of `sin`'s cuts whose window exceeds the cap). It is opt-in
  /// because it is the expensive half of the don't-care computation -- 60 s of SAT buys 0.08% on `sin`
  /// and 0.003-0.004% on `div`/`aes_core`, and nothing at all on `sasc`/`arbiter` -- and because with
  /// `sat_dc_calls` unable to bind (<=127 calls/cut) its only limit is the wall-clock
  /// `dc_time_budget`, which makes results machine-load dependent. The window simulation alone is
  /// mockturtle's `rewrite`/`satisfiability_dont_cares` behaviour: cheap, deterministic and bounded.
  bool sat_dc{ false };
  /// SAT calls per cut before the refinement gives up (the window mask, which is sound, is kept).
  /// 0 = unlimited. The SAT refinement is the expensive half of the don't-care computation (after
  /// the window fix: `arbiter` 1.5 s of window against 45-65 s of SAT; `tv80` 1.4 s against 13 s).
  /// 128 is the measured default: no cut of `arbiter`, `div`, `aes_core`, `spi`, `sasc` or `tv80`
  /// needs more (budget hits = 0, call counts identical to 256 everywhere), while a cap of 64 binds
  /// on 43 602 `arbiter` cuts and 6 557 `tv80` cuts and costs `tv80` 0.015% area (338.63 -> 338.68).
  uint32_t sat_dc_calls{ 128u };
  /// Largest fanin cone (nodes) the refinement encodes; bigger cones keep the window mask.
  uint32_t sat_dc_max_nodes{ 4000u };
  bool dependency_cuts{ false };
  bool dependency_internal_divisors{ false };
  bool feasible_covers{ true };
  uint32_t dependency_min_removed{1u};
  bool global_divisors{false};
  bool observability_care{false};
  bool chain_factor{true}, chain_share{true};
  uint32_t chain_min_length{6u},chain_block{2u};
  double observability_budget{200.0};
  uint32_t observability_min_depth{0u};
  uint32_t global_sat_conflicts{2000u};
  uint32_t dependency_window_pis{10u};
  bool verbose{ false };
};

struct stats
{
  uint64_t cofactor_cuts{0},cofactor_builds{0},cofactor_hits{0},cofactor_covers{0},cofactor_replacements{0};
  uint64_t cofactor_duplicates{0};
  uint64_t cofactor_proofs{0};
  uint64_t mapping_calls{0}, aig_alias_hits{0}, aig_reused{0};
  double secs_factor{0},secs_mapping{0},secs_proof{0},secs_aig_key{0};
  double secs_cofactor{0.0};
  uint64_t dependency_cuts{0}, dependency_replacements{0}, divisor_replacements{0};
  double secs_dependency{0};
  uint64_t global_sampled{0},global_sat_queries{0},global_proved{0},global_unknown{0},global_replacements{0};
  double secs_global{0};
  uint64_t odc_cuts{0},odc_queries{0},odc_cleared{0},odc_projected{0},odc_wins{0},odc_unknown{0};
  double secs_odc{0};
  uint64_t chain_attempts{0},chain_roots{0},chain_rejected{0},chain_product_hits{0},chain_segment_hits{0};
  double secs_chain{0};
  uint64_t roots{ 0u };
  uint64_t cuts{ 0u };
  /// Replacements whose winning candidate came from the native cover.
  uint64_t local_replacements{ 0u };
  /// Replacements won by the native multi-cell sub-network source (single cells use `local_`).
  uint64_t multi_cell_replacements{ 0u };
  /// Staged flow: cuts where the library sources produced at least one cover at all.
  uint64_t staged_library_hits{ 0u };
  /// Staged flow: roots whose cut was the MFFC cut and therefore reached the e-graph stage.
  uint64_t mffc_cut_egg_roots{ 0u };
  /// Staged flow: covers the e-graph stage returned (one per mffc-cut egg call that produced one).
  uint64_t egg_covers{ 0u };
  /// Staged candidates rejected by the final acceptance gate.
  uint64_t staged_depth_rejections{ 0u };
  uint64_t staged_delay_rejections{ 0u };
  uint64_t critical_roots{ 0u };
  uint64_t noncritical_roots{ 0u };
  uint64_t critical_replacements{ 0u };
  uint64_t noncritical_replacements{ 0u };
  uint64_t staged_objective_rejections{ 0u };
  uint64_t staged_objective_by_source[11]{ 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
  /// Wall-clock seconds spent in each per-root source (diagnostic: which stage owns the runtime).
  /// Per-round cut enumeration over the whole network (one call per round, not per root).
  double secs_cut_enum{ 0.0 };
  double secs_care{ 0.0 };
  double secs_stage1{ 0.0 };
  double secs_stage2{ 0.0 };
  double secs_egg{ 0.0 };
  double secs_mffc{ 0.0 };
  /// Care-mask cache effectiveness (the DC computation is the staged flow's runtime wall).
  uint64_t care_lookups{ 0u };
  uint64_t care_hits{ 0u };
  uint64_t care_mismatches{ 0u };
  double secs_dc_cut{ 0.0 };
  double secs_dc_collect{ 0.0 };
  double secs_dc_sim{ 0.0 };
  /// Whole `window_dont_cares` call (cut + collect + bit-parallel simulation + mask assembly) and
  /// the per-cut fanin-cone construction inside `sat_care_refine` (which runs before its solver).
  double secs_dc_window{ 0.0 };
  double secs_dc_cone{ 0.0 };
  /// Window calls, total/maximum gate count of the collected windows (diagnostic for the prep cost).
  uint64_t dc_window_calls{ 0u };
  uint64_t dc_window_nodes{ 0u };
  uint64_t dc_window_nodes_max{ 0u };
  /// `EPMAPPING_DC_WINDOW=per_node`: windows built (one per root that reaches a DC-eligible cut), cuts
  /// served out of them, cuts whose leaves were not inside the window (no don't cares) and the largest
  /// window in free inputs.
  uint64_t dc_node_windows{ 0u };
  uint64_t dc_node_window_cuts{ 0u };
  uint64_t dc_node_window_no_dc{ 0u };
  uint64_t dc_node_window_pis_max{ 0u };
  /* Whole-window structural synthesis (`EPMAPPING_WINDOW_SYN`). */
  uint64_t window_synth_windows{ 0u };
  uint64_t window_synth_applied{ 0u };
  /// Gate substitutions made by committed windows: one commit rewires **every** window output, not
  /// just the root, and they are counted into `replacements` as well (see `window_synthesize`).
  uint64_t window_synth_replacements{ 0u };
  /// Window attempts whose *measured* area change was not a win (the fresh nodes were abandoned).
  uint64_t window_synth_abandoned{ 0u };
  /// Window attempts rejected because the commit would have lengthened the global critical path.
  uint64_t window_synth_delay_rejected{ 0u };
  double window_synth_gain{ 0.0 };
  double window_synth_actual{ 0.0 };
  double secs_window_synth{ 0.0 };
  /* Structural-synthesis stage (`EPMAPPING_SYN`): roots that reached it, covers produced, wins. */
  uint64_t synth_roots{ 0u };
  uint64_t synth_covers{ 0u };
  uint64_t synth_replacements{ 0u };
  double secs_synth{ 0.0 };
  uint64_t synth_cache_lookups{ 0u };
  uint64_t synth_cache_hits{ 0u };
  /* SAT don't-care refinement: cuts refined, SAT calls issued, care minterms cleared to DCs, and
     cuts whose refinement stopped at the call budget (the window mask stays, so it is sound). */
  uint64_t dc_sat_cuts{ 0u };
  uint64_t dc_sat_calls{ 0u };
  uint64_t dc_sat_cleared{ 0u };
  uint64_t dc_sat_budget_hits{ 0u };
  double secs_dc_sat{ 0.0 };
  /// Replacements whose winning cover used a negated input / an inverted output (input-output phase).
  uint64_t replacements_with_input_negation{ 0u };
  uint64_t replacements_with_output_inversion{ 0u };
  uint64_t large_replacements{ 0u };
  /// Candidates rejected because their function did not match the replaced cone.
  uint64_t functional_rejections{ 0u };
  /// Winning candidates that could not be built / would have created a cycle.
  uint64_t build_rejections{ 0u };
  uint64_t cycle_rejections{ 0u };
  uint64_t rejected_expressions{ 0u };
  /// Gate substitutions applied: winning cut candidates, committed analytic chain roots, and every gate
  /// rewired by a committed MFFW window (`window_synth_replacements`). `run()` uses it to detect a
  /// round that made no progress at all.
  uint64_t replacements{ 0u };
  uint32_t objective_round_rollbacks{ 0u };
  uint64_t objective_discarded_replacements{ 0u };
  double area_before{ 0.0 };
  double area_after{ 0.0 };
  double delay_before{ 0.0 };
  double delay_after{ 0.0 };
  /// Passes actually executed (stops early when a pass changes nothing).
  uint32_t rounds{ 0u };
  /// Replacements per executed pass, indexed from pass 1.
  std::vector<uint64_t> replacements_per_round;
  /// Cuts that had at least one satisfiability don't care.
  uint64_t dc_cuts{ 0u };
  /// DC-bearing cuts whose free-minterm count exceeds `params::dc_max_free` (the index matcher
  /// answers nothing for those; `single_matcher::hybrid` falls back to the brute force).
  uint64_t dc_cuts_over_free{ 0u };
  /// Free-minterm histogram of the DC-bearing cuts: `dc_free_hist[k]` counts cuts with exactly `k`
  /// free minterms, `[8]` the ones with 8 or more. Used to size `params::single_free`.
  std::array<uint64_t, 9> dc_free_hist{};
  /// Cuts matched exactly because `EPMAPPING_DC_TIME_BUDGET` was already spent (diagnostic).
  uint64_t dc_budget_stops{ 0u };
  std::unordered_map<std::string, uint64_t> failure_reasons;
  std::vector<std::string> failure_examples;
};

/*!
 * \brief eLogic-style gate-level rewriting over a mapped `cell_view<block_network>`.
 *
 * For every gate (processed in reverse topological order) all cuts are enumerated. A cut with at
 * most `params::small_cut_size` leaves is mapped with the emap cut-mapping library
 * (`cut_mapping_library`, direct standard-cell match); a larger cut is mapped with `map_egg`
 * (equality saturation, `cut_mapping_with_objective`). Each candidate is expressed as a
 * gate-level subgraph using standard-cell names, and the best candidate that improves the
 * objective is substituted back into the network.
 *
 * Reverse topological order matters: when a gate is processed, its whole fanin cone is still
 * untouched, so the cuts computed on the original network remain structurally valid and all cut
 * leaves are live.
 */
template<uint32_t MaxCutSize = 6u>
class optimizer
{
public:
  using network = mockturtle::cell_view<mockturtle::block_network>;
  using node = typename network::node;
  using signal = typename network::signal;
  using tech_library_t = mockturtle::tech_library<9>;

  /* Dynamic cut database, the way `mockturtle::rewrite` uses it (`dynamic_network_cuts` plus
     `detail::dynamic_cut_enumeration_impl`): the cut set of a node can be invalidated and is
     recomputed lazily from its fanins. That is what keeps the round's cut database valid after a
     substitution rewires nodes that cuts computed earlier still depend on. The dynamic container has
     a compile-time limit of 16 cuts per node, so `params::cut_limit` is capped below it. */
  static constexpr uint32_t max_dynamic_cut_limit{ 15u };
  using cuts_t = mockturtle::dynamic_network_cuts<network, MaxCutSize, false, mockturtle::empty_cut_data>;
  using cut_manager_t = mockturtle::detail::dynamic_cut_enumeration_impl<network, MaxCutSize, false, mockturtle::empty_cut_data>;

  struct cut_context
  {
    uint32_t root_index{ 0u };
    std::vector<uint32_t> leaves;
    std::vector<uint32_t> leaf_levels;
    std::vector<uint32_t> leaf_arrivals;
    std::string expression;
    std::vector<uint32_t> cone_nodes;
    /// `cone_nodes` in topological (children-first) order, for local cone simulation.
    std::vector<uint32_t> cone_order;
    double original_area{ 0.0 };
    uint32_t original_level{ 0u };
    double original_delay{ 0.0 };
    bool dependency{ false };
    bool observability{false};
    bool critical{ false };
    /// Bit `i` is 1 when minterm `i` of the cut function is a **care** minterm (must match); the
    /// remaining minterms are satisfiability don't cares. All ones means "completely specified".
    uint64_t care_mask{ ~uint64_t{ 0 } };
    bool has_dont_cares{ false };
    /// Exact function of the cone over `leaves` (bit `m` = value at minterm `m`), computed on demand
    /// and cached: the cut's truth table used to be recomputed by every source (single cell,
    /// multi-cell, each synthesis mode and the final validation). Only meaningful for at most six
    /// leaves.
    mutable uint64_t tt_cache{ 0u };
    mutable bool tt_ready{ false };
    mutable std::optional<uint32_t> policy_features;
    mutable uint64_t removed_revision{~uint64_t{0}}, removed_slots{0};
    mutable std::optional<std::unordered_set<uint32_t>> removed_cache;
  };

  explicit optimizer(network const& ntk, params ps = {}, tech_library_t const* tech = nullptr)
      : ntk_(rebuild_topological( ntk )), ps_(ps), tech_(tech), operator_policy_(ps.operator_policy_mode)
  {
    not_tt_ = kitty::dynamic_truth_table( 1u );
    not_tt_._bits[0] = 1u;
    index_library();
    if(ps_.shared_mapped_library && tech_)
      mapped_library_=std::make_shared<std::vector<mockturtle::standard_cell> const>(tech_->get_cells());
  }

  /// The optimized network (a deep, topologically ordered copy of the input).
  network const& ntk() const { return ntk_; }
  auto const& operator_records() const { return operator_policy_.records(); }
  void print_live_queries() const { std::cout << "[i] epmapping live_queries inverse=" << inverse_queries_ << " hits=" << inverse_hits_ << " level_skips=" << inverse_level_skips_ << " walks=" << inverse_walks_ << " removed_builds=" << removed_builds_ << " removed_hits=" << removed_hits_ << " tt_hits=" << dynamic_tt_hits_ << " local_tt=" << local_truth_simulations_ << std::endl; }

  stats run()
  {
    stats st;
    st.area_before = ntk_.compute_area();
    st.delay_before = ntk_.compute_worst_delay();
    if ( ps_.area_under_delay )
    {
      ps_.path_objective = false;
      ps_.objective = 1u;
      ps_.global_objective_guard = true;
      ps_.use_required_times = true;
      ps_.rank_area_gain = true;
      ps_.preserve_depth = false;
      ps_.preserve_delay = false;
      ps_.max_area_growth = 0.0;
      ps_.max_delay_growth = 0.0;
      ps_.local_area_growth = 0.0;
      ps_.window_synth = 0u;
    }
    if ( ps_.path_objective )
    {
      ps_.objective = 0u;
      ps_.global_objective_guard = true;
      ps_.use_required_times = true;
      ps_.rank_area_gain = true;
      ps_.preserve_depth = false;
      ps_.preserve_delay = false;
      ps_.max_delay_growth = 0.0;
      ps_.window_synth = 0u;
    }
    area_bound_ = ps_.max_area_growth >= 0.0
                      ? st.area_before * ( 1.0 + ps_.max_area_growth )
                      : std::numeric_limits<double>::infinity();
    delay_bound_ = ps_.max_delay_growth >= 0.0
                       ? st.delay_before * ( 1.0 + ps_.max_delay_growth )
                       : std::numeric_limits<double>::infinity();
    if ( ps_.use_required_times && !std::isfinite( delay_bound_ ) )
      delay_bound_ = st.delay_before;

    capture_checkpoint(st,false);

    auto const max_rounds = std::max<uint32_t>( 1u, ps_.max_rounds );
    for ( uint32_t round = 0u; round < max_rounds; ++round )
    {
      /* After a pass the newly created nodes have higher indices than their parents, so the
         storage order is no longer topological and `cut_enumeration` cannot run again. Rebuild
         the network in topological order before every pass after the first. */
      if ( round > 0u )
        ntk_ = rebuild_topological( ntk_ );

      auto const metric_before = objective_metric();
      double const policy_area_before=operator_policy_.enabled()?ntk_.compute_area():0.0;
      auto const before = st.replacements;
      auto const accepted_before = st;
      auto checkpoint = ( ps_.global_objective_guard || std::isfinite( area_bound_ ) || std::isfinite( delay_bound_ ) )
                            ? std::optional<network>{ rebuild_topological( ntk_ ) }
                            : std::nullopt;
      run_round( st );
      ++st.rounds;
      st.replacements_per_round.push_back( st.replacements - before );
      if ( checkpoint )
      {
        // Measure the same canonical output representation used for the delivered netlist.
        ntk_ = rebuild_topological( ntk_ );
        if ( ( ps_.global_objective_guard && objective_metric() > metric_before ) ||
             ntk_.compute_area() > area_bound_ + 1e-4 ||
             ntk_.compute_worst_delay() > delay_bound_ + 1e-3 )
        {
          ntk_ = std::move( *checkpoint );
          ++st.objective_round_rollbacks;
          st.objective_discarded_replacements += st.replacements - before;
          st.replacements = before;
          st.replacements_per_round.back() = 0u;
          st.local_replacements = accepted_before.local_replacements;
          st.large_replacements = accepted_before.large_replacements;
          st.multi_cell_replacements = accepted_before.multi_cell_replacements;
          st.synth_replacements = accepted_before.synth_replacements;
          st.cofactor_replacements = accepted_before.cofactor_replacements;
          st.dependency_replacements = accepted_before.dependency_replacements;
          st.divisor_replacements = accepted_before.divisor_replacements;
          st.global_replacements = accepted_before.global_replacements;
          st.odc_wins=accepted_before.odc_wins;
          st.chain_roots=accepted_before.chain_roots;
          st.critical_replacements = accepted_before.critical_replacements;
          st.noncritical_replacements = accepted_before.noncritical_replacements;
          st.window_synth_replacements = accepted_before.window_synth_replacements;
          st.replacements_with_input_negation = accepted_before.replacements_with_input_negation;
          st.replacements_with_output_inversion = accepted_before.replacements_with_output_inversion;
          operator_policy_.finish_round(0.0,false);
          break;
        }
      }
      if(operator_policy_.enabled())operator_policy_.finish_round(policy_area_before-ntk_.compute_area(),true);
      capture_checkpoint(st,false);
      if ( st.replacements == before )
        break;
      /* Continue on secondary-only improvements; stop when the full objective has no progress. */
      if ( objective_metric() >= metric_before )
        break;
    }

    /* Final rebuild: it drops dead nodes, restores a topological order and materializes primary
       outputs that a substitution left driving a cell-less constant or PI (otherwise the Verilog
       writer emits them as undriven nets and CEC fails). */
    if ( ps_.verbose )
    {
      std::cout << "[d] before final rebuild: area=" << ntk_.compute_area()
                << " nodes=" << ntk_.size() << " gates=" << ntk_.num_gates()
                << " cell_nodes=" << count_cell_nodes() << "\n";
    }
    ntk_ = rebuild_topological( ntk_ );
    // An incomplete round can have a better legal snapshot than its eventual end.
    // Export the best checked state on normal completion as well as on deadlines.
    if ( ps_.area_under_delay && best_checkpoint_ &&
         objective_key(ps_.objective,best_checkpoint_->compute_area(),best_checkpoint_->compute_worst_delay()) <
         objective_key(ps_.objective,ntk_.compute_area(),ntk_.compute_worst_delay()) )
      ntk_ = std::move(*best_checkpoint_);
    if ( ps_.verbose )
    {
      std::cout << "[d] after final rebuild: area=" << ntk_.compute_area()
                << " nodes=" << ntk_.size() << " gates=" << ntk_.num_gates()
                << " cell_nodes=" << count_cell_nodes() << "\n";
    }

    st.area_after = ntk_.compute_area();
    st.delay_after = ntk_.compute_worst_delay();
    st.functional_rejections = functional_rejections_;
    st.build_rejections = build_rejections_;
    st.cycle_rejections = cycle_rejections_;
    st.staged_depth_rejections = staged_depth_rejections_;
    st.staged_delay_rejections = staged_delay_rejections_;
    st.staged_objective_rejections = staged_objective_rejections_;
    st.dc_budget_stops = dc_budget_stops_;
    st.care_lookups = st_care_lookups_;
    st.care_hits = st_care_hits_;
    st.care_mismatches = st_care_mismatches_;
    st.secs_dc_cut = st_dc_cut_;
    st.secs_dc_collect = st_dc_collect_;
    st.secs_dc_sim = st_dc_sim_;
    st.secs_dc_window = st_dc_window_;
    st.secs_dc_cone = st_dc_cone_;
    st.dc_window_calls = dc_window_calls_;
    st.dc_window_nodes = dc_window_nodes_;
    st.dc_window_nodes_max = dc_window_nodes_max_;
    st.dc_node_windows = st_dc_node_windows_;
    st.dc_node_window_cuts = st_dc_node_window_cuts_;
    st.dc_node_window_no_dc = st_dc_node_window_no_dc_;
    st.dc_node_window_pis_max = st_dc_node_window_pis_max_;
    st.dc_sat_cuts = st_sat_cuts_;
    st.dc_sat_calls = st_sat_calls_;
    st.dc_sat_cleared = st_sat_cleared_;
    st.dc_sat_budget_hits = st_sat_budget_;
    st.secs_dc_sat = st_dc_sat_;
    return st;
  }

private:
  /// Full lexicographic objective; secondary-only progress permits another round.
  std::pair<int64_t, int64_t> objective_metric() const
  {
    return objective_key( ps_.objective, ntk_.compute_area(), ntk_.compute_worst_delay() );
  }

  /// Number of live nodes with a bound cell (debug).
  uint64_t count_cell_nodes() const
  {
    uint64_t c = 0u;
    ntk_.foreach_node( [&]( auto const& n ) {
      if ( ntk_.has_cell( n ) )
        ++c;
    } );
    return c;
  }

  /// Index of the cheapest single-input buffer cell in `library`, or `UINT32_MAX` if there is none.
  static uint32_t buffer_cell_index( std::vector<mockturtle::standard_cell> const& library )
  {
    uint32_t best = UINT32_MAX;
    double best_area = std::numeric_limits<double>::max();
    for ( uint32_t i = 0u; i < library.size(); ++i )
    {
      auto const& gates = library[i].gates;
      if ( gates.size() != 1u || gates.front().num_vars != 1u )
        continue;
      if ( gates.front().function._bits[0] != 0x2u )
        continue;

      if ( best == UINT32_MAX || gates.front().area < best_area )
      {
        best = i;
        best_area = gates.front().area;
      }
    }
    return best;
  }

  /// Deep, topological rebuild of a mapped network.
  ///
  /// `block_network` copy is shallow (shared storage) and `cell_view::clone()` returns a plain
  /// `block_network` (losing cell bindings), so this walks the source in topological order,
  /// recreating PIs/POs and every gate with its bound standard cell while preserving fanin
  /// complementation. The result's node indices are a topological order, which
  /// `mockturtle::cut_enumeration` requires.
  static network rebuild_topological( network const& src )
  {
    network dst{ src.get_library() };
    std::unordered_map<uint32_t, signal> old2new;
    old2new.reserve( src.size() );

    /* Cheapest single-input buffer cell, used to materialize degenerate primary outputs. */
    auto const buf_cell = buffer_cell_index( src.get_library() );

    /* Constant and PI nodes may carry a bound cell in the source (emap binds e.g. `_const1_` to
       the true constant when it drives a primary output). Copying that binding is mandatory: the
       Verilog writer only emits a driver for a primary output when its node has a cell, so
       dropping it turns the output into an undriven net (`Constant-0 drivers added ...`). */
    auto const copy_cell = [&]( node const& src_node, signal const& dst_signal ) {
      if ( src.has_cell( src_node ) )
        dst.add_cell( dst.get_node( dst_signal ), src.get_cell_index( src_node ) );
    };

    auto const false_node = src.get_node( src.get_constant( false ) );
    old2new[src.node_to_index( false_node )] = dst.get_constant( false );
    copy_cell( false_node, dst.get_constant( false ) );
    auto const true_node = src.get_node( src.get_constant( true ) );
    if ( true_node != false_node )
    {
      old2new[src.node_to_index( true_node )] = dst.get_constant( true );
      copy_cell( true_node, dst.get_constant( true ) );
    }

    src.foreach_pi( [&]( auto const& pi, auto ) {
      auto const idx = src.node_to_index( pi );
      if ( old2new.find( idx ) == old2new.end() )
      {
        old2new[idx] = dst.create_pi();
        copy_cell( pi, old2new[idx] );
      }
    } );

    mockturtle::topo_view<network> topo{ src };
    topo.foreach_node( [&]( auto const& n, auto ) {
      auto const idx = src.node_to_index( n );
      if ( src.is_constant( n ) || src.is_pi( n ) )
        return;
      std::vector<signal> children;
      children.reserve( src.fanin_size( n ) );
      src.foreach_fanin( n, [&]( auto const& f ) {
        auto const it = old2new.find( src.node_to_index( src.get_node( f ) ) );
        if ( it != old2new.end() )
          children.push_back( f.complement ? !it->second : it->second );
      } );
      if ( children.size() != src.fanin_size( n ) )
        return;

      signal out;
      if ( src.has_cell( n ) && !src.get_cell( n ).gates.empty() )
      {
        out = dst.create_node( children, src.get_cell( n ).gates.front().function );
        dst.add_cell( dst.get_node( out ), src.get_cell_index( n ) );
      }
      else
      {
        out = dst.create_node( children, src.node_function( n ) );
      }
      old2new[idx] = out;
    } );

    src.foreach_po( [&]( auto const& f, auto ) {
      auto const it = old2new.find( src.node_to_index( src.get_node( f ) ) );
      if ( it == old2new.end() )
        return;
      auto out = f.complement ? !it->second : it->second;
      /* A primary output driven by a cell-less constant or PI would be emitted as an undriven net
         (the writer names the net after the driver but has no instance to assign it). Wrap such
         outputs in a buffer cell so every PO has an explicit driver. */
      auto const out_node = dst.get_node( out );
      if ( buf_cell != UINT32_MAX && !dst.has_cell( out_node ) &&
           ( dst.is_constant( out_node ) || dst.is_pi( out_node ) ) )
      {
        out = dst.create_node( { out }, dst.get_library()[buf_cell].gates.front().function );
        dst.add_cell( dst.get_node( out ), buf_cell );
      }
      dst.create_po( out );
    } );

    return dst;
  }

  /// Fanout lists of the current network. Rebuilt once per pass; within a pass the network only
  /// grows (nodes are appended), so the lists stay a superset of the true fanouts and the window
  /// construction below re-verifies every claim it makes through `fanout_size`.
  void rebuild_fanouts()
  {
    ++live_revision_;
    required_cache_.clear();
    fanouts_.assign( ntk_.size(), {} );
    for ( uint32_t i = 0u; i < ntk_.size(); ++i )
    {
      auto const n = ntk_.index_to_node( i );
      if ( ntk_.is_dead( n ) )
        continue;
      ntk_.foreach_fanin( n, [&]( auto const& f ) {
        auto const fi = ntk_.node_to_index( ntk_.get_node( f ) );
        if ( fi < fanouts_.size() )
          fanouts_[fi].push_back( i );
      } );
    }
  }

  /* -------------------------------------------------------------------------------------------
     Live fanouts, live cuts and live timing.
     `block_network` has no fanout iteration, so `fanouts_` is maintained here: rebuilt once per
     round and kept exact afterwards. On top of it, the three things `mockturtle::rewrite` maintains
     after a substitution are maintained too: the fanout lists, the cut database (a cut set can be
     cleared and is recomputed lazily) and the levels/arrivals of the affected fanout cone.
     ------------------------------------------------------------------------------------------- */

  void fanout_add( uint32_t child, uint32_t parent )
  {
    if ( child >= fanouts_.size() )
      fanouts_.resize( child + 1u );
    fanouts_[child].push_back( parent );
  }

  void fanout_remove( uint32_t child, uint32_t parent )
  {
    if ( child >= fanouts_.size() )
      return;
    auto& v = fanouts_[child];
    v.erase( std::remove( v.begin(), v.end(), parent ), v.end() );
  }

  /// Register a freshly created node in the fanout lists.
  void note_created( signal const& f )
  {
    auto const parent = ntk_.node_to_index( ntk_.get_node( f ) );
    ntk_.foreach_fanin( ntk_.get_node( f ), [&]( auto const& c ) {
      fanout_add( ntk_.node_to_index( ntk_.get_node( c ) ), parent );
    } );
    ensure_level_capacity();
  }

  /// The round's cut database: a full enumeration plus the lazy manager that recomputes an
  /// invalidated cut set from the current fanins.
  void rebuild_cuts( stats& st )
  {
    cut_params_ = mockturtle::cut_enumeration_params{};
    cut_params_.cut_size = std::min<uint32_t>( ps_.cut_size, MaxCutSize );
    cut_params_.cut_limit = std::min<uint32_t>( ps_.cut_limit, max_dynamic_cut_limit );
    cut_params_.minimize_truth_table = false;
    if ( ps_.cut_limit > max_dynamic_cut_limit && !cut_limit_warned_ )
    {
      std::cerr << "[w] EPMAPPING_CUT_LIMIT=" << ps_.cut_limit
                << " exceeds the live cut database's compile-time limit (16); using "
                << max_dynamic_cut_limit << "\n";
      cut_limit_warned_ = true;
    }
    auto const t_cuts = std::chrono::steady_clock::now();
    cuts_ = std::make_unique<cuts_t>( ntk_.size() + ( ntk_.size() >> 1 ) );
    cut_mgr_ = std::make_unique<cut_manager_t>( ntk_, cut_params_, cut_stats_, *cuts_ );
    cut_mgr_->init_cuts();
    cut_mgr_->run();
    st.secs_cut_enum += secs_since( t_cuts );
    processed_.assign( ntk_.size(), 0u );
  }

  /// Re-propagate `levels_`/`arrivals_` from a changed node over its fanout cone, stopping as soon as
  /// the values stop changing -- the region a single replacement can actually move
  /// (`rewrite.hpp`'s `update_levels`).
  void refresh_levels_arrivals( uint32_t seed )
  {
    std::vector<uint32_t> work{ seed };
    std::unordered_set<uint32_t> queued{ seed };
    while ( !work.empty() )
    {
      auto const v = work.back();
      work.pop_back();
      queued.erase( v );
      if ( v >= fanouts_.size() )
        continue;
      for ( auto const p : fanouts_[v] )
      {
        if ( p >= ntk_.size() || ntk_.is_dead( ntk_.index_to_node( p ) ) )
          continue;
        if ( recompute_timing( p ) && queued.insert( p ).second )
          work.push_back( p );
      }
    }
  }

  /// Recompute `levels_[idx]`/`arrivals_[idx]` from the current fanins; true when a value changed.
  bool recompute_timing( uint32_t idx )
  {
    if ( idx >= ntk_.size() )
      return false;
    auto const n = ntk_.index_to_node( idx );
    if ( ntk_.is_constant( n ) || ntk_.is_pi( n ) )
      return false;
    ensure_level_capacity();
    auto const& cell = ntk_.get_cell( n );
    mockturtle::gate const* gate = cell.gates.empty() ? nullptr : &cell.gates.front();
    uint32_t lv = 0u;
    double arr = 0.0;
    uint32_t pin = 0u;
    ntk_.foreach_fanin( n, [&]( auto const& f ) {
      lv = std::max( lv, level_of_signal( f ) + 1u );
      auto const pd = gate != nullptr ? pin_delay_of( *gate, pin ) : 0.0;
      arr = std::max( arr, arrival_of_signal( f ) + pd );
      ++pin;
    } );
    bool const changed = ( idx >= levels_.size() ) || ( levels_[idx] != lv ) ||
                         ( std::abs( arrivals_[idx] - arr ) > 1e-9 );
    if ( idx < levels_.size() )
    {
      levels_[idx] = lv;
      arrivals_[idx] = arr;
    }
    return changed;
  }

  /// `substitute_node` that keeps `fanouts_` exact and the affected timing fresh. Callers that need
  /// the replaced node's parents (the cut invalidation walk) get them through `parents_out`.
  void substitute_keep_fanouts( uint32_t old_index, signal const& new_signal,
                                std::vector<uint32_t>* parents_out = nullptr )
  {
    std::vector<uint32_t> parents;
    if ( old_index < fanouts_.size() )
      parents = fanouts_[old_index];
    ntk_.substitute_node( ntk_.index_to_node( old_index ), new_signal );
    global_valid_=false;global_solver_.reset();
    if ( old_index < fanouts_.size() )
      fanouts_[old_index].clear();
    auto const new_index = ntk_.node_to_index( ntk_.get_node( new_signal ) );
    for ( auto const p : parents )
      fanout_add( new_index, p );
    refresh_levels_arrivals( new_index );
    if ( ps_.path_objective )
      update_live_delay_target();
    ++live_revision_;
    required_cache_.clear();
    if ( parents_out != nullptr )
      parents_out->insert( parents_out->end(), parents.begin(), parents.end() );
  }

  /// Invalidate the cuts of every node this round has **not** handled yet in the fanout cone of the
  /// nodes that changed (`rewrite.hpp`'s `clear_cuts_fanout_rec`). A cleared cut set is recomputed
  /// lazily by `cut_mgr_->compute_cuts` when the node is reached.
  ///
  /// A handled (or dead) round-start node is skipped: nothing above it can still be pending, so the
  /// walk stays proportional to the affected region instead of scanning the whole TFO cone.
  void invalidate_cuts_fanout_from( std::vector<uint32_t> const& seeds )
  {
    if ( !cut_mgr_ || !cuts_ )
      return;
    std::vector<uint32_t> work{ seeds };
    std::unordered_set<uint32_t> seen;
    while ( !work.empty() )
    {
      auto const v = work.back();
      work.pop_back();
      if ( !seen.insert( v ).second || v >= fanouts_.size() )
        continue;
      for ( auto const p : fanouts_[v] )
      {
        if ( p >= ntk_.size() || ntk_.is_dead( ntk_.index_to_node( p ) ) )
          continue;
        if ( p < processed_.size() && processed_[p] != 0u )
        {
          // Critical-first order can leave an unprocessed node beyond a processed fanout.
          if ( ps_.path_objective )
            work.push_back( p );
          continue;
        }
        if ( p < cuts_->nodes_size() && cuts_->cuts( p ).size() > 0u )
          cut_mgr_->clear_cuts( ntk_.index_to_node( p ) );
        work.push_back( p );
      }
    }
  }

  void invalidate_cuts_fanout( uint32_t changed_index )
  {
    std::vector<uint32_t> seeds{ changed_index };
    invalidate_cuts_fanout_from( seeds );
  }

  /// Clear the cut sets of the whole **fanin** cone of `root_index`, so that
  /// `cut_mgr_->compute_cuts(root)` rebuilds them from the primary inputs up and is guaranteed to
  /// produce valid cuts even if an earlier substitution left a cut set in the cone that the
  /// fanout-directed invalidation missed. Only non-constant, non-PI nodes are cleared: the unit and
  /// zero cuts of the boundary are the base case of the merge. Used as the self-healing path when a
  /// cut's cone no longer terminates at its leaves.
  void clear_cuts_fanin_rec( uint32_t root_index )
  {
    if ( !cut_mgr_ || !cuts_ )
      return;
    std::vector<uint32_t> work{ root_index };
    std::unordered_set<uint32_t> seen;
    while ( !work.empty() )
    {
      auto const v = work.back();
      work.pop_back();
      if ( !seen.insert( v ).second || v >= ntk_.size() )
        continue;
      auto const n = ntk_.index_to_node( v );
      if ( !ntk_.is_constant( n ) && !ntk_.is_pi( n ) &&
           v < cuts_->nodes_size() && cuts_->cuts( v ).size() > 0u )
        cut_mgr_->clear_cuts( n );
      ntk_.foreach_fanin( n, [&]( auto const& f ) {
        work.push_back( ntk_.node_to_index( ntk_.get_node( f ) ) );
      } );
    }
  }

  /*!
   * \brief Maximum fanout-free window of an input set.
   *
   * Repeatedly adds any node all of whose fanins are already in the window, until no more can be
   * added ("fully expanded", DAC'23 §IV-B / DATE'25 Algorithm 1). The result is the maximal
   * multi-output fanout-free sub-structure with `inputs` as its inputs: every node in it that is not
   * an input is *above* the inputs, and its outputs are exactly the nodes with a fanout outside.
   */
  std::vector<uint32_t> build_mffw( std::vector<uint32_t> const& inputs ) const
  {
    if ( inputs.empty() || fanouts_.empty() )
      return {};
    std::vector<uint8_t> in_window( fanouts_.size(), 0u );
    std::vector<uint32_t> window;
    std::deque<uint32_t> work;
    for ( auto const i : inputs )
    {
      if ( i >= in_window.size() || in_window[i] != 0u )
        continue;
      in_window[i] = 1u;
      window.push_back( i );
      work.push_back( i );
    }
    while ( !work.empty() )
    {
      auto const v = work.front();
      work.pop_front();
      if ( v >= fanouts_.size() )
        continue;
      for ( auto const u : fanouts_[v] )
      {
        if ( u >= in_window.size() || in_window[u] != 0u )
          continue;
        auto const u_node = ntk_.index_to_node( u );
        if ( ntk_.is_dead( u_node ) )
          continue;
        bool all_inside = true;
        ntk_.foreach_fanin( u_node, [&]( auto const& f ) {
          if ( !all_inside )
            return;
          auto const fi = ntk_.node_to_index( ntk_.get_node( f ) );
          if ( fi >= in_window.size() || in_window[fi] == 0u )
            all_inside = false;
        } );
        if ( !all_inside )
          continue;
        in_window[u] = 1u;
        window.push_back( u );
        work.push_back( u );
      }
    }
    return window;
  }




  /// MFFC of `root_index`: the nodes that become dead when `root_index` is replaced. A node joins
  /// the set once every one of its fanouts is already in it (the classic `abcRefs` fixpoint).
  std::vector<uint32_t> mffc_of( uint32_t root_index ) const
  {
    std::unordered_set<uint32_t> dead{ root_index };
    std::vector<uint32_t> work{ root_index };
    while ( !work.empty() )
    {
      auto const v = work.back();
      work.pop_back();
      auto const n = ntk_.index_to_node( v );
      ntk_.foreach_fanin( n, [&]( auto const& f ) {
        auto const fn = ntk_.get_node( f );
        if ( ntk_.is_constant( fn ) || ntk_.is_pi( fn ) )
          return;
        auto const fi = ntk_.node_to_index( fn );
        if ( dead.find( fi ) != dead.end() )
          return;
        uint32_t inside = 0u;
        if ( fi < fanouts_.size() )
          for ( auto const fo : fanouts_[fi] )
            if ( dead.find( fo ) != dead.end() )
              ++inside;
        if ( inside == ntk_.fanout_size( ntk_.index_to_node( fi ) ) )
        {
          dead.insert( fi );
          work.push_back( fi );
        }
      } );
    }
    return std::vector<uint32_t>( dead.begin(), dead.end() );
  }







  /*!
   * \brief SPFD-style support selection for a local dependency cut (IWLS'24 §III-B).
   *
   * The paper's covering condition `Υ_x ⊆ ⋃_{y∈L} Υ_y` says: every pair of input patterns that the
   * target must distinguish is already distinguished by some divisor in `L`. Equivalently, the
   * partition of the window patterns induced by `L` must refine the partition induced by the
   * target's value, i.e. the target must be a function of `L`. Greedy selection repeatedly adds the
   * divisor that resolves the most still-mixed classes, and stops as soon as the partition is a
   * refinement (or `dep_max_support` divisors are used). Divisors are restricted to window nodes
   * outside the target's MFFC, as the paper does, because MFFC nodes are already covered by the
   * structural exploration.
   *
   * The cut's function is only specified on the support tuples that actually occur, which is exactly
   * where the don't cares come from.
   */
  /// Why a dependency cut could not be selected (diagnostics for the probe).
  enum dep_status
  {
    dep_ok = 0u,
    dep_root_missing = 1u,
    dep_no_candidates = 2u,
    dep_support_limit = 3u,
    dep_no_progress = 4u,
    dep_too_many_inputs = 5u
  };

  // Enumerate a common free-input domain, then prove functional dependency by
  // partition refinement. No outputs of the window are rebuilt: only this root.
  // A reused internal divisor and its fanins remain live. Do not charge their
  // area as removed: this is a partial MFFC replacement, not whole-window synthesis.
  std::vector<uint32_t> dependency_removable_cone( std::vector<uint32_t> const &mffc,
                                                   std::vector<uint32_t> const &support ) const
  {
    std::unordered_set<uint32_t> inside( mffc.begin(), mffc.end() ), keep;
    std::vector<uint32_t> stack( support.begin(), support.end() );
    while ( !stack.empty() )
    {
      auto i = stack.back();
      stack.pop_back();
      if ( !inside.count( i ) || !keep.insert( i ).second )
        continue;
      ntk_.foreach_fanin( ntk_.index_to_node( i ), [&]( auto const &f )
                          { stack.push_back( ntk_.node_to_index( ntk_.get_node( f ) ) ); } );
    }
    std::vector<uint32_t> removed;
    for ( auto i : mffc )
      if ( !keep.count( i ) )
        removed.push_back( i );
    return removed;
  }

  std::vector<cut_context> dependency_contexts( uint32_t ri,
                                                std::vector<uint32_t> const &mffc ) const
  {
    std::vector<cut_context> result;
    mockturtle::reconvergence_driven_cut_parameters rp;
    rp.max_leaves = std::clamp( ps_.dependency_window_pis, 2u, 12u );
    mockturtle::reconvergence_driven_cut_statistics rs;
    mockturtle::detail::reconvergence_driven_cut_impl<network, false, false> cutter( ntk_, rp, rs );
    auto boundary = cutter.run( std::vector<node>{ ntk_.index_to_node( ri ) } ).first;
    if ( boundary.empty() || boundary.size() > rp.max_leaves )
      return result;
    std::vector<uint32_t> inputs;
    for ( auto n : boundary )
      inputs.push_back( ntk_.node_to_index( n ) );
    auto pool = build_mffw( inputs );
    std::unordered_set<uint32_t> dead( mffc.begin(), mffc.end() );
    if ( ps_.dependency_min_removed > 1u )
      pool.erase( std::remove_if( pool.begin(), pool.end(),
                                  [&]( auto i ) {
                                    return dependency_removable_cone( mffc, { i } ).size() <
                                           ps_.dependency_min_removed;
                                  } ),
                  pool.end() );
    pool.erase( std::remove_if( pool.begin(), pool.end(),
                                [&]( auto i )
                                {
                                  return ( dead.count( i ) &&
                                           ( !ps_.dependency_internal_divisors || i == ri ) ) ||
                                         ntk_.is_dead( ntk_.index_to_node( i ) ) ||
                                         reaches_node( ntk_.index_to_node( i ),
                                                       ntk_.index_to_node( ri ) );
                                } ),
                pool.end() );
    std::stable_sort( pool.begin(), pool.end(),
                      [&]( auto a, auto b )
                      {
                        if ( fanouts_[a].size() != fanouts_[b].size() )
                          return fanouts_[a].size() > fanouts_[b].size();
                        return arrivals_[a] < arrivals_[b];
                      } );
    if ( pool.size() > 128u )
      pool.resize( 128u );
    if ( pool.empty() )
      return result;
    std::vector<node> roots{ ntk_.index_to_node( ri ) };
    for ( auto i : pool )
      roots.push_back( ntk_.index_to_node( i ) );
    auto sim = simulate_window( boundary, roots );
    if ( !sim || !sim->count( ri ) )
      return result;
    auto const &target = sim->at( ri );
    uint32_t count = 1u << boundary.size();
    // Each seed gives a different greedy support, avoiding a single tie-dependent path.
    for ( uint32_t seed = 0; seed < std::min<size_t>( 16, pool.size() + 1 ); ++seed )
    {
      std::vector<uint32_t> support, groups( count, 0u );
      uint64_t remaining = uint64_t( count / 2 ) * uint64_t( count / 2 );
      auto score = [&]( uint32_t idx, bool commit )
      {
        uint32_t bins = 1u << ( support.size() + 1 );
        std::vector<uint32_t> zero( bins, 0 ), one( bins, 0 );
        auto const &tt = sim->at( idx );
        for ( uint32_t m = 0; m < count; ++m )
        {
          auto g = groups[m] | ( uint32_t( kitty::get_bit( tt, m ) ) << support.size() );
          ( kitty::get_bit( target, m ) ? one : zero )[g]++;
          if ( commit )
            groups[m] = g;
        }
        uint64_t mixed = 0;
        for ( uint32_t g = 0; g < bins; ++g )
          mixed += uint64_t( zero[g] ) * one[g];
        return mixed;
      };
      // Start with the exact initial mixed-pair count, including unbalanced functions.
      uint64_t ones = 0;
      for ( uint32_t m = 0; m < count; ++m )
        ones += kitty::get_bit( target, m );
      remaining = ones * ( count - ones );
      if ( seed )
      {
        auto i = pool[seed - 1];
        remaining = score( i, true );
        support.push_back( i );
      }
      while ( remaining && support.size() < 6u )
      {
        uint32_t best = UINT32_MAX;
        uint64_t lowest = remaining;
        for ( auto i : pool )
        {
          if ( std::find( support.begin(), support.end(), i ) != support.end() )
            continue;
          auto v = score( i, false );
          if ( v < lowest )
          {
            lowest = v;
            best = i;
          }
        }
        if ( best == UINT32_MAX )
          break;
        remaining = score( best, true );
        support.push_back( best );
      }
      if ( remaining || support.empty() )
        continue;
      cut_context ctx;
      ctx.dependency = true;
      ctx.root_index = ri;
      ctx.leaves = support;
      ctx.cone_nodes = dependency_removable_cone( mffc, support );
      if ( ctx.cone_nodes.size() < ps_.dependency_min_removed )
        continue;
      ctx.original_area = mffc_area( ctx.cone_nodes, ri );
      ctx.original_level = levels_[ri];
      ctx.original_delay = arrivals_[ri];
      ctx.tt_ready = true;
      ctx.care_mask = 0;
      for ( auto i : support )
      {
        ctx.leaf_levels.push_back( levels_[i] );
        ctx.leaf_arrivals.push_back( uint32_t( std::lround( arrivals_[i] ) ) );
      }
      for ( uint32_t m = 0; m < count; ++m )
      {
        ctx.care_mask |= uint64_t( 1 ) << groups[m];
        if ( kitty::get_bit( target, m ) )
          ctx.tt_cache |= uint64_t( 1 ) << groups[m];
      }
      ctx.has_dont_cares =
          ctx.care_mask != ( support.size() == 6u
                                 ? ~uint64_t( 0 )
                                 : ( ( uint64_t( 1 ) << ( 1u << support.size() ) ) - 1 ) );
      result.push_back( std::move( ctx ) );
    }
    return result;
  }

  // Full-network sampled signatures are only a filter. Every offered relation
  // is proved by a bounded SAT miter before it enters the physical cost gate.
  static constexpr size_t global_words = 16u;
  using global_signature = std::array<uint64_t, global_words>;
  using global_solver = bill::solver<bill::solvers::bsat2>;

  bool prepare_global_divisors()
  {
    if ( global_valid_ )
      return true;
    odc_root_ = UINT32_MAX;
    odc_assumptions_.clear();
    global_signatures_.assign( ntk_.size(), {} );
    global_seen_.assign( ntk_.size(), false );
    global_solver_ = std::make_unique<global_solver>();
    for ( uint32_t i = 0; i < ntk_.size(); ++i )
      global_solver_->add_variable();
    auto lit = []( uint32_t i, bool value )
    {
      return bill::lit_type( i, value ? bill::lit_type::polarities::positive
                                      : bill::lit_type::polarities::negative );
    };
    uint32_t pi = 0;
    bool valid = true;
    mockturtle::topo_view<network> topo{ ntk_ };
    topo.foreach_node(
        [&]( auto n )
        {
          auto i = ntk_.node_to_index( n );
          global_seen_[i] = true;
          if ( ntk_.is_constant( n ) )
          {
            global_signatures_[i].fill( ntk_.constant_value( n ) ? ~uint64_t( 0 ) : 0 );
            global_solver_->add_clause( lit( i, ntk_.constant_value( n ) ) );
            return;
          }
          if ( ntk_.is_pi( n ) )
          {
            for ( uint32_t m = 0; m < global_words * 64u; ++m )
            {
              bool value = false;
              if ( m == 1 )
                value = true;
              else if ( m >= 2 && m < 2 + ntk_.num_pis() )
                value = ( m - 2 == pi );
              else if ( m >= 2 + ntk_.num_pis() && m < 2 + 2 * ntk_.num_pis() )
                value = ( m - 2 - ntk_.num_pis() != pi );
              else if ( m >= 2 + 2 * ntk_.num_pis() )
              {
                uint64_t x = ( uint64_t( m ) + 1 ) * 0x9e3779b97f4a7c15ULL ^
                             ( uint64_t( pi ) + 1 ) * 0xbf58476d1ce4e5b9ULL;
                x = ( x ^ ( x >> 30 ) ) * 0xbf58476d1ce4e5b9ULL;
                x = ( x ^ ( x >> 27 ) ) * 0x94d049bb133111ebULL;
                x ^= x >> 31;
                uint32_t const thresholds[] = { 1024, 4096, 16384, 32768, 49152, 61440, 64512 };
                value = ( x & 65535u ) < thresholds[m % 7];
              }
              if ( value )
                global_signatures_[i][m / 64] |= uint64_t( 1 ) << ( m % 64 );
            }
            ++pi;
            return;
          }
          std::vector<uint32_t> fanins;
          std::vector<bool> phases;
          ntk_.foreach_fanin( n,
                              [&]( auto f )
                              {
                                fanins.push_back( ntk_.node_to_index( ntk_.get_node( f ) ) );
                                phases.push_back( f.complement );
                              } );
          if ( fanins.empty() || fanins.size() > 6u )
          {
            valid = false;
            return;
          }
          auto fn = ntk_.node_function( n );
          for ( uint32_t pattern = 0; pattern < ( 1u << fanins.size() ); ++pattern )
          {
            uint32_t index = 0;
            std::vector<bill::lit_type> clause;
            for ( uint32_t k = 0; k < fanins.size(); ++k )
            {
              bool value = ( pattern >> k ) & 1u;
              index |= uint32_t( value ^ phases[k] ) << k;
              clause.push_back( lit( fanins[k], !value ) );
            }
            bool output = kitty::get_bit( fn, index );
            clause.push_back( lit( i, output ) );
            global_solver_->add_clause( clause );
            if ( output )
              for ( size_t word = 0; word < global_words; ++word )
              {
                uint64_t cube = ~uint64_t( 0 );
                for ( uint32_t k = 0; k < fanins.size(); ++k )
                  cube &= ( ( pattern >> k ) & 1u ) ? global_signatures_[fanins[k]][word]
                                                    : ~global_signatures_[fanins[k]][word];
                global_signatures_[i][word] |= cube;
              }
          }
        } );
    global_valid_ = valid;
    return valid;
  }

  bool prove_global_dependency( cut_context const &ctx, stats &st )
  {
    auto lit = []( uint32_t i, bool value )
    {
      return bill::lit_type( i, value ? bill::lit_type::polarities::positive
                                      : bill::lit_type::polarities::negative );
    };
    for ( uint32_t m = 0; m < ( 1u << ctx.leaves.size() ); ++m )
    {
      std::vector<bill::lit_type> assumptions;
      for ( uint32_t k = 0; k < ctx.leaves.size(); ++k )
        assumptions.push_back( lit( ctx.leaves[k], ( m >> k ) & 1u ) );
      assumptions.push_back( lit( ctx.root_index, !( ( ctx.tt_cache >> m ) & 1u ) ) );
      ++st.global_sat_queries;
      auto state = global_solver_->solve( assumptions, ps_.global_sat_conflicts );
      if ( state != bill::result::states::unsatisfiable )
      {
        if ( state == bill::result::states::undefined )
          ++st.global_unknown;
        return false;
      }
    }
    ++st.global_proved;
    return true;
  }

  std::vector<std::pair<cut_context, CCost>>
  global_divisor_candidates( uint32_t ri, std::vector<uint32_t> const &mffc, stats &st )
  {
    std::vector<std::pair<cut_context, CCost>> result;
    if ( !prepare_global_divisors() || !global_seen_[ri] )
      return result;
    auto const &target = global_signatures_[ri];
    uint32_t ones = 0;
    for ( auto word : target )
      ones += __builtin_popcountll( word );
    // Unsensitized signatures cannot guide bounded enumeration reliably.
    if ( ones == 0 || ones == global_words * 64u )
      return result;
    struct literal
    {
      uint32_t index;
      bool neg;
      uint32_t distance;
    };
    std::vector<literal> and_pool, or_pool, equal;
    for ( uint32_t i = 0; i < global_seen_.size(); ++i )
    {
      if ( !global_seen_[i] || i == ri || ntk_.is_dead( ntk_.index_to_node( i ) ) ||
           ntk_.is_constant( ntk_.index_to_node( i ) ) )
        continue;
      for ( uint32_t neg = 0; neg < 2; ++neg )
      {
        bool implies = true, implied = true;
        uint32_t distance = 0;
        for ( size_t w = 0; w < global_words; ++w )
        {
          auto v = neg ? ~global_signatures_[i][w] : global_signatures_[i][w];
          implies &= ( v & ~target[w] ) == 0;
          implied &= ( target[w] & ~v ) == 0;
          distance += __builtin_popcountll( v ^ target[w] );
          if ( !implies && !implied )
            break;
        }
        literal l{ i, neg != 0, distance };
        if ( distance == 0 && implies && implied )
          equal.push_back( l );
        else
        {
          if ( implied )
            and_pool.push_back( l );
          if ( implies )
            or_pool.push_back( l );
        }
      }
    }
    auto trim = [&]( auto &pool )
    {
      std::stable_sort( pool.begin(), pool.end(),
                        [&]( auto a, auto b )
                        {
                          return std::make_tuple( a.distance, arrivals_[a.index], a.index, a.neg ) <
                                 std::make_tuple( b.distance, arrivals_[b.index], b.index, b.neg );
                        } );
      std::vector<literal> filtered;
      for ( auto l : pool )
      {
        if ( !reaches_node( ntk_.index_to_node( l.index ), ntk_.index_to_node( ri ) ) )
          filtered.push_back( l );
        if ( filtered.size() == 32u )
          break;
      }
      pool = std::move( filtered );
    };
    trim( equal );
    trim( and_pool );
    trim( or_pool );
    uint32_t proofs = 0;
    auto offer = [&]( std::vector<uint32_t> leaves, uint64_t truth )
    {
      ++st.global_sampled;
      cut_context ctx;
      ctx.dependency = true;
      ctx.root_index = ri;
      ctx.leaves = std::move( leaves );
      ctx.cone_nodes = dependency_removable_cone( mffc, ctx.leaves );
      ctx.original_area = mffc_area( ctx.cone_nodes, ri );
      ctx.original_delay = arrivals_[ri];
      ctx.original_level = levels_[ri];
      ctx.tt_ready = true;
      ctx.tt_cache = truth;
      ctx.care_mask = ( uint64_t( 1 ) << ( 1u << ctx.leaves.size() ) ) - 1;
      for ( auto i : ctx.leaves )
      {
        ctx.leaf_levels.push_back( levels_[i] );
        ctx.leaf_arrivals.push_back( uint32_t( std::lround( arrivals_[i] ) ) );
      }
      std::optional<CCost> best;
      if ( ctx.leaves.size() == 1u && truth == 2u )
      {
        CCost c;
        MappedNode leaf;
        leaf.kind = 0;
        leaf.input_idx = 0;
        c.cell_nodes.push_back( std::move( leaf ) );
        c.cell_root = 0;
        c.area = 0;
        c.delay = float( arrivals_[ctx.leaves[0]] );
        if ( candidate_rejection_kind( c, ctx ) == 0 )
          best = std::move( c );
      }
      for ( auto &cover : supergate_candidates( ctx, truth, 32u, false, 3u ) )
        if ( candidate_rejection_kind( cover, ctx ) == 0 &&
             ( !best || candidate_is_better( cover, ctx, *best, ctx ) ) )
          best = std::move( cover );
      if ( !best || proofs >= 16u )
        return;
      ++proofs;
      if ( prove_global_dependency( ctx, st ) )
        result.emplace_back( std::move( ctx ), std::move( *best ) );
    };
    for ( auto l : equal )
      offer( { l.index }, l.neg ? 1u : 2u );
    auto pairs = [&]( std::vector<literal> const &pool, bool is_and )
    {
      for ( size_t a = 0; a < pool.size() && proofs < 16u; ++a )
        for ( size_t b = a + 1; b < pool.size() && proofs < 16u; ++b )
        {
          auto x = pool[a], y = pool[b];
          if ( x.index == y.index )
            continue;
          bool match = true;
          for ( size_t w = 0; w < global_words; ++w )
          {
            auto vx = x.neg ? ~global_signatures_[x.index][w] : global_signatures_[x.index][w];
            auto vy = y.neg ? ~global_signatures_[y.index][w] : global_signatures_[y.index][w];
            if ( ( is_and ? ( vx & vy ) : ( vx | vy ) ) != target[w] )
            {
              match = false;
              break;
            }
          }
          if ( !match )
            continue;
          uint64_t truth = 0;
          for ( uint32_t m = 0; m < 4; ++m )
          {
            bool vx = bool( m & 1u ) ^ x.neg, vy = bool( m & 2u ) ^ y.neg;
            if ( is_and ? ( vx && vy ) : ( vx || vy ) )
              truth |= uint64_t( 1 ) << m;
          }
          offer( { x.index, y.index }, truth );
        }
    };
    pairs( and_pool, true );
    pairs( or_pool, false );
    // Conditional divisor reuse: replace one pin of the existing cell, while
    // proving the output relation under the other pins' observability mask.
    auto root = ntk_.index_to_node( ri );
    std::vector<uint32_t> pins;
    std::vector<bool> pin_neg;
    ntk_.foreach_fanin( root,
                        [&]( auto f )
                        {
                          pins.push_back( ntk_.node_to_index( ntk_.get_node( f ) ) );
                          pin_neg.push_back( f.complement );
                        } );
    if ( pins.size() <= 3u )
      for ( uint32_t pin = 0; pin < pins.size() && proofs < 16u; ++pin )
      {
        global_signature zero{}, one{}, care{};
        auto fn = ntk_.node_function( root );
        for ( uint32_t pattern = 0; pattern < ( 1u << pins.size() ); ++pattern )
        {
          uint32_t index = 0;
          for ( uint32_t k = 0; k < pins.size(); ++k )
            index |= uint32_t( bool( ( pattern >> k ) & 1u ) ^ pin_neg[k] ) << k;
          if ( !kitty::get_bit( fn, index ) )
            continue;
          for ( size_t w = 0; w < global_words; ++w )
          {
            uint64_t cube = ~uint64_t( 0 );
            for ( uint32_t k = 0; k < pins.size(); ++k )
              if ( k != pin )
                cube &= ( ( pattern >> k ) & 1u ) ? global_signatures_[pins[k]][w]
                                                  : ~global_signatures_[pins[k]][w];
            ( ( ( pattern >> pin ) & 1u ) ? one : zero )[w] |= cube;
          }
        }
        for ( size_t w = 0; w < global_words; ++w )
          care[w] = zero[w] ^ one[w];
        std::vector<literal> choices;
        for ( uint32_t i = 0; i < global_seen_.size(); ++i )
        {
          if ( !global_seen_[i] || i == ri || i == pins[pin] ||
               ntk_.is_dead( ntk_.index_to_node( i ) ) )
            continue;
          for ( uint32_t neg = 0; neg < 2u; ++neg )
          {
            bool match = true;
            uint32_t distance = 0;
            for ( size_t w = 0; w < global_words; ++w )
            {
              auto v = neg ? ~global_signatures_[i][w] : global_signatures_[i][w];
              auto delta = v ^ global_signatures_[pins[pin]][w];
              distance += __builtin_popcountll( delta );
              if ( delta & care[w] )
              {
                match = false;
                break;
              }
            }
            if ( match )
              choices.push_back( { i, neg != 0, distance } );
          }
        }
        // Earlier signals first: physical delay feasibility is still checked for every cover.
        std::stable_sort(
            choices.begin(), choices.end(),
            [&]( auto a, auto b )
            {
              return std::make_tuple( arrivals_[a.index], a.distance, a.index, a.neg ) <
                     std::make_tuple( arrivals_[b.index], b.distance, b.index, b.neg );
            } );
        uint32_t tried = 0;
        for ( auto l : choices )
        {
          if ( proofs >= 16u || tried >= 16u )
            break;
          if ( reaches_node( ntk_.index_to_node( l.index ), root ) )
            continue;
          ++tried;
          auto support = pins;
          support[pin] = l.index;
          uint64_t truth = 0;
          for ( uint32_t m = 0; m < ( 1u << pins.size() ); ++m )
          {
            uint32_t index = 0;
            for ( uint32_t k = 0; k < pins.size(); ++k )
              index |= uint32_t( bool( ( m >> k ) & 1u ) ^ pin_neg[k] ^ ( k == pin && l.neg ) )
                       << k;
            if ( kitty::get_bit( fn, index ) )
              truth |= uint64_t( 1 ) << m;
          }
          offer( std::move( support ), truth );
        }
      }
    return result;
  }

  // Stop at a PO or the first branch. Along a unique fanout path every side
  // input is unaffected by changing this root, so the conjunction of gate
  // Boolean derivatives is a sufficient (conservative) observability mask.
  bool prepare_observability( uint32_t ri )
  {
    if ( !prepare_global_divisors() )
      return false;
    if ( odc_root_ == ri )
      return !odc_assumptions_.empty() && odc_assumptions_.size() >= ps_.observability_min_depth;
    odc_root_ = ri;
    odc_assumptions_.clear();
    std::unordered_set<uint32_t> outputs;
    ntk_.foreach_po( [&]( auto f )
                     { outputs.insert( ntk_.node_to_index( ntk_.get_node( f ) ) ); } );
    auto lit = []( uint32_t i, bool value )
    {
      return bill::lit_type( i, value ? bill::lit_type::polarities::positive
                                      : bill::lit_type::polarities::negative );
    };
    uint32_t current = ri;
    for ( uint32_t depth = 0; depth < 64u; ++depth )
    {
      if ( outputs.count( current ) )
        break;
      std::vector<uint32_t> parents;
      for ( auto p : fanouts_[current] )
        if ( !ntk_.is_dead( ntk_.index_to_node( p ) ) )
          parents.push_back( p );
      if ( parents.size() != 1u )
        break;
      auto parent = parents[0];
      auto n = ntk_.index_to_node( parent );
      std::vector<uint32_t> side;
      std::vector<bool> neg;
      uint32_t position = UINT32_MAX, index = 0;
      bool child_neg = false;
      ntk_.foreach_fanin( n,
                          [&]( auto f )
                          {
                            auto i = ntk_.node_to_index( ntk_.get_node( f ) );
                            if ( i == current )
                            {
                              position = index;
                              child_neg = f.complement;
                            }
                            else
                            {
                              side.push_back( i );
                              neg.push_back( f.complement );
                            }
                            ++index;
                          } );
      if ( position == UINT32_MAX || side.size() > 5u )
        break;
      auto function = ntk_.node_function( n );
      auto variable = global_solver_->add_variable();
      for ( uint32_t m = 0; m < ( 1u << side.size() ); ++m )
      {
        uint32_t fixed = 0, k = 0;
        std::vector<bill::lit_type> clause;
        for ( uint32_t pin = 0; pin < index; ++pin )
          if ( pin != position )
          {
            bool value = ( m >> k ) & 1u;
            fixed |= uint32_t( value ^ neg[k] ) << pin;
            clause.push_back( lit( side[k], !value ) );
            ++k;
          }
        bool derivative = kitty::get_bit( function, fixed ) ^
                          kitty::get_bit( function, fixed | ( 1u << position ) );
        (void)child_neg; // complementing the toggled pin does not change its Boolean derivative
        clause.push_back( lit( variable, derivative ) );
        global_solver_->add_clause( clause );
      }
      odc_assumptions_.push_back( lit( variable, true ) );
      current = parent;
    }
    return !odc_assumptions_.empty() && odc_assumptions_.size() >= ps_.observability_min_depth;
  }

  void refine_observable_context( cut_context &ctx, stats &st )
  {
    if ( !prepare_observability( ctx.root_index ) )
      return;
    auto n = uint32_t( ctx.leaves.size() );
    if ( n == 0 || n > 6u )
      return;
    uint64_t cleared = 0;
    auto assumptions = odc_assumptions_;
    auto lit = []( uint32_t i, bool value )
    {
      return bill::lit_type( i, value ? bill::lit_type::polarities::positive
                                      : bill::lit_type::polarities::negative );
    };
    std::function<void( uint32_t, uint32_t )> visit = [&]( uint32_t depth, uint32_t prefix )
    {
      uint64_t cube = 0;
      for ( uint32_t m = prefix; m < ( 1u << n ); m += ( 1u << depth ) )
        cube |= uint64_t( 1 ) << m;
      if ( !( cube & ctx.care_mask ) )
        return;
      ++st.odc_queries;
      auto state = global_solver_->solve( assumptions, ps_.global_sat_conflicts );
      if ( state == bill::result::states::unsatisfiable )
      {
        cleared |= cube;
        return;
      }
      if ( state == bill::result::states::undefined )
      {
        ++st.odc_unknown;
        return;
      }
      if ( depth == n )
        return;
      assumptions.push_back( lit( ctx.leaves[depth], false ) );
      visit( depth + 1, prefix );
      assumptions.pop_back();
      assumptions.push_back( lit( ctx.leaves[depth], true ) );
      visit( depth + 1, prefix | ( 1u << depth ) );
      assumptions.pop_back();
    };
    visit( 0, 0 );
    cleared &= ctx.care_mask;
    if ( !cleared )
      return;
    ++st.odc_cuts;
    st.odc_cleared += __builtin_popcountll( cleared );
    ctx.care_mask &= ~cleared;
    ctx.has_dont_cares = true;
    ctx.observability = true;
    auto original = truth_of( ctx );
    // Project away leaves that are redundant on observable care assignments.
    // Enumerate support subsets, rather than all don't-care completions.
    std::vector<uint32_t> subsets;
    for ( uint32_t sub = 0; sub < ( 1u << n ); ++sub )
      subsets.push_back( sub );
    std::stable_sort( subsets.begin(), subsets.end(),
                      []( auto a, auto b )
                      { return __builtin_popcount( a ) < __builtin_popcount( b ); } );
    for ( auto sub : subsets )
    {
      std::vector<uint32_t> positions;
      for ( uint32_t k = 0; k < n; ++k )
        if ( ( sub >> k ) & 1u )
          positions.push_back( k );
      if ( positions.empty() )
        positions.push_back( 0 ); // constants keep one leaf; the cover may ignore it
      uint64_t care = 0, truth = 0;
      bool consistent = true;
      for ( uint32_t m = 0; m < ( 1u << n ); ++m )
      {
        if ( !( ( ctx.care_mask >> m ) & 1u ) )
          continue;
        uint32_t tuple = 0;
        for ( uint32_t k = 0; k < positions.size(); ++k )
          tuple |= ( ( m >> positions[k] ) & 1u ) << k;
        auto bit = uint64_t( 1 ) << tuple;
        bool value = ( original >> m ) & 1u;
        if ( ( care & bit ) && bool( truth & bit ) != value )
        {
          consistent = false;
          break;
        }
        care |= bit;
        if ( value )
          truth |= bit;
      }
      if ( !consistent )
        continue;
      std::vector<uint32_t> leaves, levels, arrivals;
      for ( auto k : positions )
      {
        leaves.push_back( ctx.leaves[k] );
        levels.push_back( ctx.leaf_levels[k] );
        arrivals.push_back( ctx.leaf_arrivals[k] );
      }
      if ( leaves.size() < ctx.leaves.size() )
        ++st.odc_projected;
      ctx.leaves = std::move( leaves );
      ctx.leaf_levels = std::move( levels );
      ctx.leaf_arrivals = std::move( arrivals );
      ctx.dependency = true;
      ctx.tt_ready = true;
      ctx.tt_cache = truth;
      ctx.care_mask = care;
      ctx.has_dont_cares = ( care != full_care_mask( ctx.leaves.size() ) );
      break;
    }
  }

  // Compose affine Boolean chain steps T(x)=p*x+q. Only AO21 chains are
  // transformed; segment sharing is structural and algebraically exact.
  void factor_ao_chains( stats &st )
  {
    auto start = std::chrono::steady_clock::now();
    auto checkpoint = rebuild_topological( ntk_ );
    double before_area = checkpoint.compute_area();
    auto is_ao = [&]( uint32_t i )
    {
      auto n = ntk_.index_to_node( i );
      return !ntk_.is_dead( n ) && !ntk_.is_pi( n ) && !ntk_.is_constant( n ) &&
             ntk_.node_function( n ).num_vars() == 3u &&
             ( ntk_.node_function( n )._bits[0] & 255u ) == 248u;
    };
    std::vector<uint32_t> and_cells( 6, UINT32_MAX );
    uint32_t ao_cell = UINT32_MAX;
    auto const &library = ntk_.get_library();
    for ( uint32_t i = 0; i < library.size(); ++i )
    {
      if ( library[i].gates.size() != 1u )
        continue;
      auto const &g = library[i].gates.front();
      auto n = g.function.num_vars();
      if ( n >= 2u && n <= 5u &&
           ( g.function._bits[0] & full_care_mask( n ) ) ==
               ( uint64_t( 1 ) << ( ( 1u << n ) - 1 ) ) )
        if ( and_cells[n] == UINT32_MAX || g.area < library[and_cells[n]].gates.front().area )
          and_cells[n] = i;
      if ( n == 3u && ( g.function._bits[0] & 255u ) == 248u )
        if ( ao_cell == UINT32_MAX || g.area < library[ao_cell].gates.front().area )
          ao_cell = i;
    }
    if ( ao_cell == UINT32_MAX || std::any_of( and_cells.begin() + 2, and_cells.end(),
                                               []( auto i ) { return i == UINT32_MAX; } ) )
      return;
    struct step
    {
      signal p, q;
    };
    struct target
    {
      uint32_t root;
      std::vector<step> steps;
    };
    std::unordered_set<uint32_t> outputs;
    ntk_.foreach_po( [&]( auto f )
                     { outputs.insert( ntk_.node_to_index( ntk_.get_node( f ) ) ); } );
    std::vector<target> targets;
    auto old_size = ntk_.size();
    for ( uint32_t i = 0; i < old_size; ++i )
    {
      if ( !is_ao( i ) )
        continue;
      bool endpoint = outputs.count( i ) || fanouts_[i].size() != 1u;
      if ( !endpoint )
      {
        auto parent = fanouts_[i][0];
        if ( !is_ao( parent ) )
          endpoint = true;
        else
        {
          uint32_t pin = 0;
          ntk_.foreach_fanin( ntk_.index_to_node( parent ),
                              [&]( auto f )
                              {
                                if ( ntk_.node_to_index( ntk_.get_node( f ) ) == i && pin == 2u )
                                  endpoint = true;
                                ++pin;
                              } );
        }
      }
      if ( !endpoint )
        continue;
      target t{ i, {} };
      signal current = ntk_.make_signal( ntk_.index_to_node( i ) );
      bool valid = true;
      while ( is_ao( ntk_.node_to_index( ntk_.get_node( current ) ) ) && t.steps.size() < 64u )
      {
        std::vector<signal> f;
        ntk_.foreach_fanin( ntk_.get_node( current ),
                            [&]( auto s )
                            {
                              f.push_back( s );
                              if ( s.complement )
                                valid = false;
                            } );
        if ( !valid )
          break;
        uint32_t chain = arrivals_[ntk_.node_to_index( ntk_.get_node( f[0] ) )] >=
                                 arrivals_[ntk_.node_to_index( ntk_.get_node( f[1] ) )]
                             ? 0u
                             : 1u;
        // Choose a chain input when present; otherwise terminate at one product input.
        auto eligible = [&]( uint32_t k )
        {
          auto n = ntk_.node_to_index( ntk_.get_node( f[k] ) );
          return is_ao( n ) && fanouts_[n].size() == 1u && !outputs.count( n );
        };
        if ( eligible( 1u - chain ) && !eligible( chain ) )
          chain = 1u - chain;
        t.steps.push_back( { f[1u - chain], f[2] } );
        current = f[chain];
        if ( !eligible( chain ) )
          break;
      }
      if ( valid && t.steps.size() >= ps_.chain_min_length )
      {
        t.steps.push_back( { ntk_.get_constant( false ), current } );
        targets.push_back( std::move( t ) );
      }
    }
    if ( targets.empty() )
      return;
    ++st.chain_attempts;
    std::unordered_map<std::string, signal> products, segments;
    auto key_signal = [&]( signal f )
    { return ntk_.node_to_index( ntk_.get_node( f ) ) * 2u + uint32_t( f.complement ); };
    auto make_cell = [&]( uint32_t cell, std::vector<signal> children )
    {
      auto const &g = library[cell].gates.front();
      // AND pins are symmetric; AO21 has two symmetric product pins.
      size_t symmetric = cell == ao_cell ? 2u : children.size();
      if ( symmetric == 2u && cell == ao_cell )
      {
        if ( arrival_of_signal( children[0] ) < arrival_of_signal( children[1] ) )
          std::swap( children[0], children[1] );
        if ( pin_delay_of( g, 0 ) > pin_delay_of( g, 1 ) )
          std::swap( children[0], children[1] );
      }
      else
      {
        std::vector<size_t> pins( symmetric );
        std::iota( pins.begin(), pins.end(), 0u );
        std::sort( pins.begin(), pins.end(),
                   [&]( auto a, auto b ) { return pin_delay_of( g, a ) < pin_delay_of( g, b ); } );
        std::sort( children.begin(), children.end(),
                   [&]( auto a, auto b )
                   { return arrival_of_signal( a ) > arrival_of_signal( b ); } );
        auto sorted = children;
        for ( size_t k = 0; k < symmetric; ++k )
          children[pins[k]] = sorted[k];
      }
      auto f = ntk_.create_node( children, g.function );
      ntk_.add_cell( ntk_.get_node( f ), cell );
      note_created( f );
      uint32_t level = 0;
      double arrival = 0;
      for ( size_t k = 0; k < children.size(); ++k )
      {
        level = std::max( level, levels_[ntk_.node_to_index( ntk_.get_node( children[k] ) )] + 1u );
        arrival = std::max( arrival, arrival_of_signal( children[k] ) + pin_delay_of( g, k ) );
      }
      auto idx = ntk_.node_to_index( ntk_.get_node( f ) );
      levels_[idx] = level;
      arrivals_[idx] = arrival;
      return f;
    };
    auto product = [&]( std::vector<signal> factors )
    {
      std::sort( factors.begin(), factors.end(),
                 [&]( auto a, auto b ) { return key_signal( a ) < key_signal( b ); } );
      factors.erase( std::unique( factors.begin(), factors.end(),
                                  [&]( auto a, auto b )
                                  { return key_signal( a ) == key_signal( b ); } ),
                     factors.end() );
      std::string key;
      for ( auto f : factors )
        key += std::to_string( key_signal( f ) ) + ",";
      if ( auto it = products.find( key ); it != products.end() )
      {
        ++st.chain_product_hits;
        return it->second;
      }
      auto nodes = factors;
      while ( nodes.size() > 1u )
      {
        std::vector<signal> next;
        for ( size_t a = 0; a < nodes.size(); a += 5u )
        {
          auto count = std::min<size_t>( 5, nodes.size() - a );
          if ( count == 1u )
            next.push_back( nodes[a] );
          else
            next.push_back(
                make_cell( and_cells[count],
                           std::vector<signal>( nodes.begin() + a, nodes.begin() + a + count ) ) );
        }
        nodes = std::move( next );
      }
      products.emplace( key, nodes.front() );
      return nodes.front();
    };
    std::vector<std::pair<uint32_t, signal>> replacements;
    for ( auto const &t : targets )
    {
      // The noSHARE ablation retains within-root memoization but isolates roots.
      if ( !ps_.chain_share )
      {
        products.clear();
        segments.clear();
      }
      std::function<signal( size_t, size_t )> q = [&]( size_t begin, size_t end ) -> signal
      {
        if ( end - begin == 1u )
          return t.steps[begin].q;
        std::string key;
        for ( size_t j = begin; j < end; ++j )
          key += std::to_string( key_signal( t.steps[j].p ) ) + ":" +
                 std::to_string( key_signal( t.steps[j].q ) ) + ",";
        if ( auto it = segments.find( key ); it != segments.end() )
        {
          ++st.chain_segment_hits;
          return it->second;
        }
        auto length = end - begin;
        auto split =
            begin + ( ps_.chain_block && length > ps_.chain_block ? ps_.chain_block : length / 2u );
        auto left = q( begin, split ), right = q( split, end );
        std::vector<signal> factors;
        for ( size_t j = begin; j < split; ++j )
          factors.push_back( t.steps[j].p );
        auto p = product( std::move( factors ) );
        auto out = make_cell( ao_cell, { p, right, left } );
        segments.emplace( std::move( key ), out );
        return out;
      };
      replacements.emplace_back( t.root, q( 0, t.steps.size() ) );
    }
    for ( auto [root, out] : replacements )
      substitute_keep_fanouts( root, out );
    auto candidate = rebuild_topological( ntk_ );
    auto area = candidate.compute_area(), delay = candidate.compute_worst_delay();
    bool accept = area < before_area - 1e-4 && delay <= delay_bound_ + 1e-3;
    std::cout << "[i] chain_factor proposed_area=" << area << " proposed_delay=" << delay
              << " roots=" << targets.size() << " segments=" << segments.size()
              << " products=" << products.size() << " accepted=" << accept << "\n";
    if ( accept )
    {
      ntk_ = std::move( candidate );
      st.chain_roots += replacements.size();
      st.replacements += replacements.size();
    }
    else
    {
      ntk_ = std::move( checkpoint );
      ++st.chain_rejected;
    }
    st.secs_chain += secs_since( start );
    compute_levels();
    rebuild_fanouts();
    global_valid_ = false;
    global_solver_.reset();
  }

  void run_round ( stats &st )
  {
    /* Node indices are only meaningful within a round (`rebuild_topological` renumbers the network
       between rounds), so the care-mask cache lives exactly one round. */
    global_valid_=false;global_solver_.reset();
    care_mask_cache_.clear();
    node_window_.reset();
    std::ofstream dep_trace;
    if(auto path=std::getenv("EPMAPPING_DEPENDENCY_TRACE"))
    {
      dep_trace.open(path,std::ios::app);
      dep_trace << "root,leaves,side,mffc,removed,old_area,old_delay,required,truth,care,source,new_area,new_delay,rejection\n";
    }
    compute_levels();
    /* `mffc_of` and the MFFW window builder both need fanout counts. */
    rebuild_fanouts();
    if ( ps_.path_objective )
      update_live_delay_target();
    if(ps_.chain_factor)factor_ao_chains(st);
    rebuild_cuts( st );

    /* Gate nodes in index (== topological) order; iterate in reverse. */
    std::vector<node> gates;
    for ( uint32_t i = 0u; i < ntk_.size(); ++i )
    {
      auto const n = ntk_.index_to_node( i );
      if ( ntk_.is_constant( n ) || ntk_.is_pi( n ) || ntk_.is_dead( n ) || !ntk_.has_cell( n ) )
        continue;
      gates.push_back( n );
    }
    if ( ps_.path_objective )
      std::stable_partition( gates.begin(), gates.end(), [&]( auto const n ) {
        auto const index = ntk_.node_to_index( n );
        return !is_critical( index, arrivals_[index] );
      } ); // Reverse iteration handles initially critical gates first.

    for ( auto it = gates.rbegin(); it != gates.rend(); ++it )
    {
      auto const root = *it;
      auto const root_index = ntk_.node_to_index( root );
      /* The invalidation walk below uses `processed_` to find the nodes this round has not handled
         yet; a dead root is marked processed so the walk stops there. */
      if ( root_index < processed_.size() )
        processed_[root_index] = 1u;
      if ( ntk_.is_dead( root ) )
        continue;
      ++st.roots;
      if ( ps_.path_objective )
      {
        if ( is_critical( root_index, arrivals_[root_index] ) )
          ++st.critical_roots;
        else
          ++st.noncritical_roots;
      }
      /* An earlier substitution this round may have invalidated this node's cuts (a window commit
         rewires nodes that pending cuts still depend on): recompute them from the current fanins,
         the way `rewrite.hpp` recomputes a cut set it cleared. */
      cut_mgr_->compute_cuts( root );
      std::vector<uint32_t> mffc_of_root;
      if ( !fanouts_.empty() )
      {
        auto const t_mffc = std::chrono::steady_clock::now();
        mffc_of_root = mffc_of( root_index );
        st.secs_mffc += secs_since( t_mffc );
      }
      if ( ps_.verbose && ( st.roots % 200u == 0u ) )
        std::cerr << "[epmapping] roots=" << st.roots << " cuts=" << st.cuts
                  << " replacements=" << st.replacements << std::endl;

      /* Whole-window structural synthesis (`EPMAPPING_WINDOW_SYN`). It is invoked only as a
         **fallback**, after the cut loop below, and only when this root got no replacement at all
         (see the call site): a committed window rebuilds all of its outputs at once, so it can never
         take a better single-cut candidate away. `try_window` picks this root's most promising cut by
         the removable area of its MFFW and hands the leaf set to `window_synthesize`. */
      auto try_window = [&]() -> bool {
        if ( ps_.window_synth == 0u || ( ps_.synth_modes & 3u ) == 0u || fanouts_.empty() )
          return false;
        auto const t_win = std::chrono::steady_clock::now();
        double best_removable = -1.0;
        std::vector<uint32_t> best_leaves;
        for ( auto const& cut : cuts_->cuts( root_index ) )
        {
          auto const k = cut->size();
          if ( k < 2u || k > ps_.window_synth_max_leaves )
            continue;
          auto ctx = make_context( root, root_index, *cut );
          if ( !ctx )
            continue;
          auto const win = build_mffw( ctx->leaves );
          if ( win.size() <= ctx->leaves.size() + 1u ||
               win.size() > ctx->leaves.size() + ps_.window_synth_max_gates )
            continue;
          double removable = 0.0;
          for ( auto const v : win )
          {
            auto const n = ntk_.index_to_node( v );
            if ( ntk_.has_cell( n ) && !ntk_.get_cell( n ).gates.empty() )
              removable += ntk_.get_cell( n ).gates.front().area;
          }
          if ( removable > best_removable )
          {
            best_removable = removable;
            best_leaves = ctx->leaves;
          }
        }
        /* only synthesize the most promising window of this root (the others cost the same but the
           substitution is greedy, so one attempt per root keeps the runtime bounded) */
        bool applied = false;
        if ( best_removable >= ps_.window_synth_min_gain && !best_leaves.empty() )
          applied = window_synthesize( root_index, best_leaves, st );
        st.secs_window_synth += secs_since( t_win );
        return applied;
      };
      auto const replacements_before_cuts = st.replacements;

      std::optional<CCost> best_cost;
      std::optional<cut_context> best_ctx;
      /* 0 = library/precomputed DB, 1 = egg, 2 = native multi-cell cover. */
      int best_source = 0;

      /* A substitution earlier in the round can invalidate a cut set that was merged from an
         invalidated fanin before the fanout-directed walk reached that fanin. When a cut's cone no
         longer terminates at its leaves, rebuild the root's whole fanin cone once and retry, so the
         cuts a round uses are never silently dropped. */
      for ( uint32_t cut_attempt = 0u; cut_attempt < 2u; ++cut_attempt )
      {
      bool cuts_stale = false;
      for ( auto const& cut : cuts_->cuts( root_index ) )
      {
        if ( cut->size() == 0u || cut->size() > cut_params_.cut_size )
          continue;
        /* Skip the trivial cut whose only leaf is the root: substituting it would make the
           replacement depend on the node being replaced (a combinational cycle). */
        if ( cut->size() == 1u && static_cast<uint32_t>( *cut->begin() ) == root_index )
          continue;
        ++st.cuts;

        auto context = make_context( root, root_index, *cut );
        if ( !context )
        {
          if ( cut_attempt == 0u )
          {
            cuts_stale = true;
            break; /* rebuild this root's fanin cone and retry once */
          }
          ++st.rejected_expressions;
          continue;
        }
        /* Candidate sources of the staged flow, in order (see `synthesize_covers`). */
        std::optional<CCost> cost;
        int cost_source = 0;

        auto consider = [&]( std::optional<CCost>&& candidate, int source, bool already_priced=false ) {
          auto const evaluation_start=operator_policy_.enabled()?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
          auto evaluate=[&]() {
          if ( !candidate )
            return;
          /* Unified delay model across sources: the native cover computes it itself, while
             `map_egg` may have produced it with different (or no) leaf arrivals. */
          if ( already_priced ) {}
          else if ( ps_.phase_cost && ps_.phase_reuse )
          {
            auto const priced=phase_cover_cost(*context,*candidate);
            if(!priced) return;
            candidate->area=static_cast<float>(priced->first);
            candidate->delay=static_cast<float>(priced->second);
            candidate->aft_dep=static_cast<uint32_t>(std::lround(priced->second));
          }
          else if ( auto const delay = cover_delay( *context, *candidate ) )
          {
            candidate->delay = static_cast<float>( *delay );
            candidate->aft_dep = static_cast<uint32_t>( std::lround( *delay ) );
          }
          else
          {
            return;
          }
          auto const rejection = candidate_rejection_kind( *candidate, *context );
          if ( rejection != 0u )
          {
            if(auto op=policy_source(source))operator_policy_.reject(*op,policy_bucket(*context),rejection);
            if ( rejection == 3u && source >= 0 && source < 11 )
              ++st.staged_objective_by_source[source];
            return;
          }
          if ( !cost || candidate_is_better( *candidate, *context, *cost, *context ) )
          {
            cost = std::move( *candidate );
            cost_source = source;
          }
          };evaluate();
          if(operator_policy_.enabled())if(auto op=policy_source(source))
            operator_policy_.elapsed(*op,policy_bucket(*context),secs_since(evaluation_start));
        };

        {
            /* Per-cut staged decision, **don't cares first**:
             (1) compute the cut's don't cares (window simulation, then SAT refinement);
             (2) single library cell (the N/P supergate index by default);
             (3) native four-input multi-cell library (`EPMAPPING_SUPER`);
             (4) structural synthesis (source 10, `EPMAPPING_SYN`);
             (5) e-graph rewriting, only for the MFFC cut and only when (2)-(4) found nothing.
             The MFFW window fallback runs only if the whole root produced no replacement.
             Input and output phase is considered by every source: both output polarities are looked
             up, each entry carries its input-negation mask, and `finalize_local_cover` charges one
             inverter per negated input plus one for an inverted output (materialized in the netlist). */
          {
            auto const t_care = std::chrono::steady_clock::now();
            compute_care_mask( *context, true );
          if(ps_.observability_care && st.secs_odc<ps_.observability_budget && context->leaves.size()<=6u)
          {
            auto start=std::chrono::steady_clock::now();
            refine_observable_context(*context,st);
            st.secs_odc+=secs_since(start);
          }
            st.secs_care += secs_since( t_care );
          }
          if ( context->has_dont_cares )
            ++st.dc_cuts;

          uint32_t free_count = 0u;
          if ( context->has_dont_cares )
          {
            /* Diagnosis for the indexed single-cell stage: how many cuts have more free minterms than
               the DC-completion enumeration can afford (those get no single-cell candidate there). */
            auto const n_leaves = context->leaves.size();
            auto const n_minterms = ( n_leaves >= 6u ) ? 64u : ( uint32_t{ 1u } << n_leaves );
            for ( uint32_t m = 0u; m < n_minterms; ++m )
              if ( ( ( context->care_mask >> m ) & 1u ) == 0u )
                ++free_count;
            if ( free_count > ps_.dc_max_free )
              ++st.dc_cuts_over_free;
            ++st.dc_free_hist[std::min<uint32_t>( free_count, 8u )];
          }

          if(context->observability)
          {
            auto truth=truth_of(*context);
            if((truth&context->care_mask)==0u || (truth&context->care_mask)==context->care_mask)
            {
              CCost c;MappedNode constant;constant.kind=1u;
              constant.const_val=(truth&context->care_mask)!=0u;c.cell_nodes.push_back(std::move(constant));
              c.cell_root=0;c.area=0;c.delay=0;consider(std::optional<CCost>{std::move(c)},2);
            }
          }
          {
            auto const t_s1 = std::chrono::steady_clock::now();
            /* Single library cell: the library's N/P supergate index holds every genlib cell with all
               its permutations and input polarities, so it answers the same question as the brute force
               by hash probe -- except on cuts with more don't cares than the completion enumeration
               affords, where `hybrid` falls back to the brute force (any DC count, slower). */
            std::optional<CCost> single;
            if ( ps_.single != params::single_matcher::brute )
              single = match_partial_truth_table( *context, truth_of( *context ),
                                                  context->care_mask, ps_.single_free,
                                                  /*multi_cell=*/false, /*max_vars=*/6u,
                                                  ps_.single_entries );
            if ( !single &&
                 ( ps_.single == params::single_matcher::brute ||
                   ( ps_.single == params::single_matcher::hybrid && free_count > ps_.single_free ) ) )
              single = match_single_cell( *context );
            st.secs_stage1 += secs_since( t_s1 );
            if ( single )
              ++st.staged_library_hits;
            consider( std::move( single ), 2 ); /* single-cell library */
          }
          {
            auto const t_s2 = std::chrono::steady_clock::now();
            auto dc_cover = multi_cell_cover_dc( *context );
            st.secs_stage2 += secs_since( t_s2 );
            if ( dc_cover )
            {
              ++st.staged_library_hits;
              consider( std::optional<CCost>{ std::move( *dc_cover ) }, 5 ); /* native 4-input multi-cell */
            }
          }
          /* Structural synthesis (source 10): a cover *built* for the cut function (SOP factoring /
             DSD, optionally with the cut's don't cares) instead of one picked from the library.
             Enabled by `EPMAPPING_SYN`, offered on every cut at least `EPMAPPING_SYN_MIN_LEAVES`
             wide. See [`synthesize_cone`]. */
          if ( ps_.synth != 0u && context->leaves.size() >= ps_.synth_min_leaves )
          {
            bool const mffc_root = is_mffc_cut( *context, mffc_of_root );
            for ( auto& cover : synthesize_covers( *context, st, mffc_root ) )
              consider( std::optional<CCost>{ std::move( cover ) }, 10, ps_.operator_policy_mode==3 && ps_.phase_cost && ps_.phase_reuse );
          }
          if ( ps_.cofactor_modes && context->leaves.size()>=ps_.cofactor_min_leaves )
            for (auto& cover:cofactor_covers(*context,st,cost ? std::optional<double>(cost->area) : std::nullopt))
              consider(std::optional<CCost>{std::move(cover.first)},11+cover.second,ps_.operator_policy_mode==3 && ps_.phase_cost && ps_.phase_reuse);

          /* Stage 3: e-graph rewriting (`EPMAPPING_EGG`, on by default), only for the cut whose cone
             is exactly the MFFC (where the whole MFFC is replaced) and only when neither library
             stage improved. The cut's gate-level cone is flattened to an AIG, saturated with the
             one-way `AIG -> cell` cover rules, and the cheapest cover is extracted in e-graph space
             (cell nodes and AIG nodes live in the same e-graph; this is not a second mapping pass).
             `EPMAPPING_EGG_REQ` (default on) hands the e-graph the leaf arrivals and the cone's
             original delay as an output bound, so the extraction returns the minimum-area cover that
             cannot regress the delay. */
          if ( ps_.use_egg && !cost && !context->dependency && is_mffc_cut( *context, mffc_of_root ) &&
               context->original_area >= ps_.egg_min_mffc )
          {
            ++st.mffc_cut_egg_roots;
            auto const t_egg = std::chrono::steady_clock::now();
            operator_policy_.allow(5,policy_bucket(*context),false,true);
            auto covers=call_egg_covers( *context, st );
            if(operator_policy_.enabled())operator_policy_.elapsed(5,policy_bucket(*context),secs_since(t_egg));
            for ( auto& c : covers )
              consider( std::optional<CCost>{ std::move( c ) }, 1 );
            st.secs_egg += secs_since( t_egg );
          }

        }
        if ( !cost )
        {
          ++st.failure_reasons[last_failure_reason_];
          if ( st.failure_examples.size() < 3u )
            st.failure_examples.push_back( context->expression );
          continue;
        }

        if ( !best_cost || candidate_is_better( *cost, *context, *best_cost, *best_ctx ) )
        {
          best_cost = std::move( cost );
          best_ctx = std::move( context );
          best_source = cost_source;
        }
      }

      if ( !cuts_stale )
        break;
      clear_cuts_fanin_rec( root_index );
      cut_mgr_->compute_cuts( root );
      best_cost.reset();
      best_ctx.reset();
      best_source = 0;
      }

      if ( ps_.dependency_cuts )
      {
        auto const start = std::chrono::steady_clock::now ();
        for ( auto &ctx : dependency_contexts ( root_index, mffc_of_root ) )
        {
          ++st.dependency_cuts;
          uint32_t side=0;std::string leaves;
          if(dep_trace)
            for(auto i:ctx.leaves){
              if(!reaches_node(root,ntk_.index_to_node(i))) ++side;
              leaves+=std::to_string(i)+";";
            }
          auto trace=[&](int source,double area,double delay,int rejection){
            if(dep_trace) dep_trace << root_index << ',' << leaves << ',' << side << ','
              << mffc_of_root.size() << ',' << ctx.cone_nodes.size() << ',' << ctx.original_area << ','
              << ctx.original_delay << ',' << required_time(root_index) << ',' << ctx.tt_cache << ','
              << ctx.care_mask << ',' << source << ',' << area << ',' << delay << ',' << rejection << '\n';
          };
          trace(0,-1,-1,-1);
          auto offer = [&] ( std::optional<CCost> candidate, int source )
          {
            if ( !candidate ){trace(source,-1,-1,4); return;}
            auto delay = cover_delay ( ctx, *candidate );
            if ( !delay )
              return;
            candidate->delay = static_cast<float> ( *delay );
            auto rejection=candidate_rejection_kind(*candidate,ctx);
            trace(source,candidate->area,candidate->delay,rejection);
            if ( rejection != 0 ) return;
            if ( !best_cost || candidate_is_better ( *candidate, ctx, *best_cost, *best_ctx ) )
            {
              best_cost = std::move ( candidate );
              best_ctx = ctx;
              best_source = source;
            }
          };
          if ( ctx.leaves.size () == 1u &&
               ( ctx.tt_cache & ctx.care_mask ) == ( 2u & ctx.care_mask ) )
          {
            CCost reuse;
            MappedNode leaf;
            leaf.kind = 0u;
            leaf.input_idx = 0u;
            reuse.cell_nodes.push_back ( std::move ( leaf ) );
            reuse.cell_root = 0u;
            reuse.area = 0;
            offer ( std::optional<CCost>{ std::move ( reuse ) }, 7 );
          }
          offer ( match_partial_truth_table ( ctx, truth_of ( ctx ), ctx.care_mask, ps_.single_free,
                                              false, 6u, ps_.single_entries ),
                  6 );
          offer ( multi_cell_cover_dc ( ctx ), 6 );
          if ( ps_.synth )
            for ( auto &cover : synthesize_covers ( ctx, st ) )
              offer ( std::optional<CCost>{ std::move ( cover ) }, 6 );
        }
        st.secs_dependency += secs_since ( start );
      }
      if(ps_.global_divisors)
      {
        auto start=std::chrono::steady_clock::now();
        for(auto& candidate:global_divisor_candidates(root_index,mffc_of_root,st))
          if(!best_cost || candidate_is_better(candidate.second,candidate.first,*best_cost,*best_ctx))
          {best_ctx=std::move(candidate.first);best_cost=std::move(candidate.second);best_source=8;}
        st.secs_global+=secs_since(start);
      }
      if ( best_cost && best_ctx )
      {
        auto const replacement_start=operator_policy_.enabled()?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        bool const applied=try_replace ( root, *best_ctx, *best_cost );
        if(operator_policy_.enabled())if(auto op=policy_source(best_source))
          operator_policy_.elapsed(*op,policy_bucket(*best_ctx),secs_since(replacement_start));
        if ( applied )
        {
          if ( std::getenv("EPMAPPING_REPLACEMENT_TRACE") )
          {
            std::cerr << "[replacement] root=" << root_index << " source=" << best_source
                      << " original_area=" << best_ctx->original_area << " area=" << best_cost->area
                      << " delay=" << best_cost->delay << " expr=" << best_ctx->expression << " cells=";
            for ( auto const& nd : best_cost->cell_nodes )
              if ( nd.kind == 2u ) std::cerr << static_cast<std::string>(nd.cell_name) << ',';
            std::cerr << '\n';
          }
          ++st.replacements;
          if(auto op=policy_source(best_source))
            operator_policy_.accepted(*op,policy_bucket(*best_ctx),best_ctx->original_area-best_cost->area);
          if(best_ctx->observability)++st.odc_wins;
          if ( ps_.path_objective )
          {
            if ( best_ctx->critical )
              ++st.critical_replacements;
            else
              ++st.noncritical_replacements;
          }
          {
            bool input_neg = false, out_neg = false;
            for ( auto const& nd : best_cost->cell_nodes )
            {
              if ( nd.kind != 2u )
                continue;
              if ( nd.neg_mask != 0u )
                input_neg = true;
              if ( nd.out_neg != 0u )
                out_neg = true;
            }
            if ( input_neg )
              ++st.replacements_with_input_negation;
            if ( out_neg )
              ++st.replacements_with_output_inversion;
          }
          if ( best_source == 8 ) ++st.global_replacements;
          if ( best_source == 6 ) ++st.dependency_replacements;
          if ( best_source == 7 ) ++st.divisor_replacements;
          if ( best_source == 2 )
            ++st.local_replacements;
          else if ( best_source == 1 )
            ++st.large_replacements;
          else if ( best_source == 5 )
            ++st.multi_cell_replacements;
          else if ( best_source == 10 )
            ++st.synth_replacements;
          else if ( best_source >= 11 && best_source <= 14 )
            ++st.cofactor_replacements;
        }
      }

      /* Window-synthesis fallback: only when this root's cut stages produced nothing does the window
         get a chance, so a committed window can never take away an improvement the cut stages would
         have found. */
      if ( ps_.window_synth != 0u && st.replacements == replacements_before_cuts )
        (void)try_window();
      if (st.replacements != replacements_before_cuts) capture_checkpoint(st,true);
    }
  }

  void capture_checkpoint(stats const& st,bool partial)
  {
    if(!ps_.checkpoint) return;
    auto const now=std::chrono::steady_clock::now();
    if(partial && (!ps_.area_under_delay || ps_.checkpoint_interval<=0.0 ||
       std::chrono::duration<double>(now-last_checkpoint_time_).count()<ps_.checkpoint_interval)) return;
    last_checkpoint_time_=now;
    // Deep rebuild: cell_view copies share underlying storage, and the live search
    // continues mutating it. A saved checkpoint must own an independent network.
    auto snapshot=rebuild_topological(ntk_);
    auto const area=snapshot.compute_area(),delay=snapshot.compute_worst_delay();
    auto const metric=objective_key(ps_.objective,area,delay);
    if(area>area_bound_+1e-4 || delay>delay_bound_+1e-3 ||
       (best_checkpoint_metric_ && metric>=*best_checkpoint_metric_)) return;
    best_checkpoint_metric_=metric;
    best_checkpoint_=std::move(snapshot);
    ps_.checkpoint(*best_checkpoint_,st.rounds,partial);
  }

  uint32_t policy_bucket(cut_context const& context) const
  {
    auto const base=std::min<uint32_t>(7,context.leaves.size())*4+(context.cone_nodes.size()>=3?2:0)+(context.has_dont_cares?1:0);
    if(ps_.operator_policy_mode!=3) return base;
    if(context.policy_features) return *context.policy_features;
    bool phase=false;
    for(auto leaf:context.leaves)
      if(external_inverse_fanout(leaf,context.root_index)){phase=true;break;}
    auto const tt=cone_truth_table_dynamic(context);
    uint32_t support=0;bool affine=true,unate=true;
    for(uint32_t i=0;i<context.leaves.size();++i) {
      if(!kitty::has_var(tt,i)) continue;
      ++support;
      auto lo=kitty::cofactor0(tt,i),hi=kitty::cofactor1(tt,i),delta=lo^hi;
      affine &= kitty::is_const0(delta) || kitty::is_const0(~delta);
      unate &= kitty::is_const0(lo & ~hi) || kitty::is_const0(hi & ~lo);
    }
    uint32_t shape=support<context.leaves.size()?0u:affine?1u:unate?2u:3u;
    // Slack, external opposite phase and Boolean structure distinguish regions
    // that the old leaf-count/cone-size/DC groups incorrectly pooled together.
    bool const critical=is_critical(context.root_index,context.original_delay);
    context.policy_features=base+32u*critical+64u*phase+128u*shape;
    return *context.policy_features;
  }
  static std::optional<uint32_t> policy_source(int source)
  {if(source==10)return 0;if(source>=11 && source<=14)return source-10;if(source==1)return 5;return std::nullopt;}


  struct cell_info
  {
    uint32_t index{ 0u };
    std::unordered_map<std::string, uint32_t> pin_pos;
  };

  void index_library()
  {
    auto const& lib = ntk_.get_library();
    for ( uint32_t i = 0u; i < lib.size(); ++i )
    {
      auto const& cell = lib[i];
      if ( cells_by_name_.find( cell.name ) != cells_by_name_.end() )
        continue;

      cell_info info;
      info.index = i;
      if ( !cell.gates.empty() )
      {
        auto const& gate = cell.gates.front();
        for ( uint32_t p = 0u; p < gate.pins.size(); ++p )
          info.pin_pos.emplace( gate.pins[p].name, p );
      }
      cells_by_name_.emplace( cell.name, std::move( info ) );

      /* Smallest inverter (single 1-input gate with the NOT truth table 0x1). */
      if ( cell.gates.size() == 1u && cell.gates.front().num_vars == 1u &&
           cell.gates.front().function._bits[0] == 1u )
      {
        if ( inv_cell_index_ == UINT32_MAX || cell.area < inv_cell_area_ )
        {
          inv_cell_index_ = i;
          inv_cell_area_ = cell.area;
          if ( !cell.gates.front().pins.empty() )
          {
            auto const& pin = cell.gates.front().pins.front();
            inv_delay_ = std::max( pin.rise_block_delay, pin.fall_block_delay );
          }
        }
      }
    }
  }

  static double pin_delay_of( mockturtle::gate const& gate, size_t i )
  {
    if ( i >= gate.pins.size() )
      return 0.0;
    return std::max( gate.pins[i].rise_block_delay, gate.pins[i].fall_block_delay );
  }

  /// Pin delay of the **physical** pin a cover node's `k`-th fanin is connected to. A cover produced
  /// by `map_egg` carries a `CellVariant`, whose pin order can be a permutation of the library
  /// gate's, so indexing the delay with the cover position charged the delay of a different pin.
  static double pin_delay_of_gate_pin( cell_info const& info, mockturtle::gate const& gate,
                                       std::string const& pin_name )
  {
    auto const it = info.pin_pos.find( pin_name );
    if ( it == info.pin_pos.end() )
      return 0.0;
    return pin_delay_of( gate, it->second );
  }

  /// Fresh arrival propagation over the whole network, ignoring the incremental `arrivals_` cache
  /// (which goes stale for the fanout cone of every commit). `overrides` replaces the arrival of
  /// specific nodes with a value supplied by the caller -- used to predict the critical path a window
  /// commit would produce, where the substituted outputs are driven by the new cover instead of their
  /// old fanins.
  double predict_worst_delay( std::unordered_map<uint32_t, double> const& overrides = {} ) const
  {
    mockturtle::topo_view<network> topo{ ntk_ };
    std::vector<double> arr( ntk_.size(), 0.0 );
    topo.foreach_node( [&]( auto const& n, auto ) {
      auto const idx = ntk_.node_to_index( n );
      if ( idx >= arr.size() || ntk_.is_dead( n ) )
        return;
      if ( ntk_.is_constant( n ) || ntk_.is_pi( n ) )
      {
        arr[idx] = 0.0;
        return;
      }
      if ( auto const it = overrides.find( idx ); it != overrides.end() )
      {
        arr[idx] = it->second;
        return;
      }
      auto const& cell = ntk_.get_cell( n );
      mockturtle::gate const* gate = cell.gates.empty() ? nullptr : &cell.gates.front();
      double worst = 0.0;
      uint32_t pin = 0u;
      ntk_.foreach_fanin( n, [&]( auto const& f ) {
        auto const fi = ntk_.node_to_index( ntk_.get_node( f ) );
        double edge = gate != nullptr ? pin_delay_of( *gate, pin ) : 0.0;
        if ( f.complement )
          edge += inv_delay_;
        worst = std::max( worst, ( fi < arr.size() ? arr[fi] : 0.0 ) + edge );
        ++pin;
      } );
      arr[idx] = worst;
    } );

    double worst = 0.0;
    ntk_.foreach_po( [&]( auto const& f ) {
      auto const fi = ntk_.node_to_index( ntk_.get_node( f ) );
      worst = std::max( worst, ( fi < arr.size() ? arr[fi] : 0.0 ) + ( f.complement ? inv_delay_ : 0.0 ) );
    } );
    return worst;
  }

  void compute_levels()
  {
    levels_.assign( ntk_.size(), 0u );
    arrivals_.assign( ntk_.size(), 0.0 );
    mockturtle::topo_view<network> topo{ ntk_ };
    topo.foreach_node( [&]( auto const& n, auto ) {
      auto const idx = ntk_.node_to_index( n );
      if ( idx >= levels_.size() )
      {
        levels_.resize( idx + 1u, 0u );
        arrivals_.resize( idx + 1u, 0.0 );
      }
      if ( ntk_.is_constant( n ) || ntk_.is_pi( n ) )
      {
        levels_[idx] = 0u;
        arrivals_[idx] = 0.0;
        return;
      }
      uint32_t lv = 0u;
      double arr = 0.0;
      auto const& cell = ntk_.get_cell( n );
      mockturtle::gate const* gate = cell.gates.empty() ? nullptr : &cell.gates.front();
      uint32_t pin = 0u;
      ntk_.foreach_fanin( n, [&]( auto const& f ) {
        auto const fi = ntk_.node_to_index( ntk_.get_node( f ) );
        if ( fi < levels_.size() )
        {
          lv = std::max( lv, levels_[fi] + 1u );
          auto const edge = f.complement ? inv_delay_ : 0.0;
          auto const pd = gate != nullptr ? pin_delay_of( *gate, pin ) : 0.0;
          arr = std::max( arr, arrivals_[fi] + pd + edge );
        }
        ++pin;
      } );
      levels_[idx] = lv;
      arrivals_[idx] = arr;
    } );
  }

  void ensure_level_capacity()
  {
    if ( levels_.size() < ntk_.size() )
      levels_.resize( ntk_.size(), 0u );
    if ( arrivals_.size() < ntk_.size() )
      arrivals_.resize( ntk_.size(), 0.0 );
  }

  uint32_t level_of_signal( signal const& s ) const
  {
    auto const idx = ntk_.node_to_index( ntk_.get_node( s ) );
    return idx < levels_.size() ? levels_[idx] : 0u;
  }

  double arrival_of_signal( signal const& s ) const
  {
    auto const idx = ntk_.node_to_index( ntk_.get_node( s ) );
    auto const base = idx < arrivals_.size() ? arrivals_[idx] : 0.0;
    return base + ( s.complement ? inv_delay_ : 0.0 );
  }

  std::optional<signal> external_inverse_fanout( uint32_t index, uint32_t root_index ) const
  {
    if ( !ps_.phase_fanout || root_index==UINT32_MAX || index>=fanouts_.size() ) return std::nullopt;
    ++inverse_queries_;
    if(ps_.live_query_memo & 1u)
      if(auto cached=inverse_memo_.find(live_revision_,ntk_.size(),root_index,index)){++inverse_hits_;return *cached;}
    std::optional<signal> best;
    for (auto parent:fanouts_[index]) {
      auto n=ntk_.index_to_node(parent);
      if(ntk_.is_dead(n) || ntk_.fanout_size(n)==0u || !ntk_.has_cell(n) || ntk_.num_outputs(n)!=1u ||
         ntk_.fanin_size(n)!=1u || ntk_.node_function(n)!=not_tt_) continue;
      // An inverter may be shared inside this root's fanin. Reusing it is legal unless it
      // depends on the root. Require a live path to a PO, excluding speculative candidate debris.
      if(!std::isfinite(required_time(parent))) continue;
      bool depends;
      if((ps_.live_query_memo & 1u) && parent<levels_.size() && root_index<levels_.size() && levels_[parent]<levels_[root_index]) {
        ++inverse_level_skips_;depends=false; // a live fanin ancestor has strictly lower structural level
      } else {++inverse_walks_;depends=reaches_node(n,ntk_.index_to_node(root_index));}
      if(depends) continue;
      bool valid=false;
      ntk_.foreach_fanin(n,[&](auto const& f){valid=!f.complement && ntk_.node_to_index(ntk_.get_node(f))==index;});
      if(!valid) continue;
      auto s=ntk_.make_signal(n);
      if(!best || std::make_pair(arrival_of_signal(s),parent)<
                   std::make_pair(arrival_of_signal(*best),ntk_.node_to_index(ntk_.get_node(*best)))) best=s;
    }
    if(ps_.live_query_memo & 1u) inverse_memo_.insert(index,best);
    return best;
  }

  signal make_inverter( signal const& x, uint32_t root_index=UINT32_MAX )
  {
    /* `~( ~n ) == n`: a complemented signal needs no physical inverter and would not be
       representable by `write_verilog_with_cell` (which drops fanin complementation). */
    if ( x.complement )
      return ntk_.make_signal( ntk_.get_node( x ) );
    // A cut leaf can itself be a physical inverter. Its opposite phase is
    // already available at its input, even when that input is outside the cut.
    // Reuse that signal rather than creating a second physical inverter.
    auto const xn = ntk_.get_node( x );
    if ( ps_.phase_reuse && !ntk_.is_pi( xn ) && !ntk_.is_constant( xn ) && ntk_.has_cell( xn ) &&
         ntk_.num_outputs( xn ) == 1u && ntk_.fanin_size( xn ) == 1u &&
         ntk_.node_function( xn ) == not_tt_ )
    {
      std::optional<signal> input;
      ntk_.foreach_fanin( xn, [&]( auto const& f ) { input = f; } );
      if ( input && !input->complement )
      {
        if ( std::getenv("EPMAPPING_REPLACEMENT_TRACE") )
          std::cerr << "[phase_reuse] bypass_inverter=" << ntk_.node_to_index(xn)
                    << " reuse=" << ntk_.node_to_index(ntk_.get_node(*input)) << '\n';
        return *input;
      }
    }
    if ( ps_.phase_reuse )
      if (auto reused=external_inverse_fanout(ntk_.node_to_index(xn),root_index)) {
        if(std::getenv("EPMAPPING_REPLACEMENT_TRACE"))
          std::cerr << "[phase_reuse] fanout=" << ntk_.node_to_index(ntk_.get_node(*reused))
                    << " input=" << ntk_.node_to_index(xn) << '\n';
        return *reused;
      }
    if ( inv_cell_index_ == UINT32_MAX )
      return x;
    std::vector<signal> children{ x };
    auto const f = ntk_.create_node( children, not_tt_ );
    note_created( f );
    ensure_level_capacity();
    auto const idx = ntk_.node_to_index( ntk_.get_node( f ) );
    if ( idx < levels_.size() )
    {
      levels_[idx] = level_of_signal( x ) + 1u;
      arrivals_[idx] = arrival_of_signal( x ) + inv_delay_;
    }
    ntk_.add_cell( ntk_.get_node( f ), inv_cell_index_ );
    return f;
  }

  template<typename Cut>
  std::optional<cut_context> make_context( node root, uint32_t root_index, Cut const& cut ) const
  {
    cut_context context;
    context.root_index = root_index;

    std::unordered_map<uint32_t, uint32_t> leaf_pos;
    for ( auto const leaf : cut )
    {
      auto const index = static_cast<uint32_t>( leaf );
      if ( leaf_pos.find( index ) != leaf_pos.end() )
        continue;
      if ( context.leaves.size() >= MaxCutSize )
        return std::nullopt;
      leaf_pos.emplace( index, static_cast<uint32_t>( context.leaves.size() ) );
      context.leaves.push_back( index );
      context.leaf_levels.push_back( index < levels_.size() ? levels_[index] : 0u );
      context.leaf_arrivals.push_back(
          index < arrivals_.size()
              ? static_cast<uint32_t>( std::lround( std::max( 0.0, arrivals_[index] ) ) )
              : 0u );
    }
    if ( context.leaves.empty() )
      return std::nullopt;

    std::unordered_map<uint32_t, std::string> memo;
    std::unordered_set<uint32_t> cone;
    bool failed = false;

    std::function<std::optional<std::string>( node )> expand =
        [&]( node current ) -> std::optional<std::string> {
      auto const index = ntk_.node_to_index( current );
      if ( auto const it = leaf_pos.find( index ); it != leaf_pos.end() )
        return std::string( 1, static_cast<char>( 'a' + it->second ) );
      if ( auto const it = memo.find( index ); it != memo.end() )
        return it->second;
      if ( ntk_.is_constant( current ) )
        return std::string( kitty::get_bit( ntk_.node_function( current ), 0u ) ? "1" : "0" );
      if ( ntk_.is_pi( current ) || !ntk_.has_cell( current ) )
        return std::nullopt;

      std::vector<std::string> inputs;
      ntk_.foreach_fanin( current, [&]( auto const& f ) {
        if ( failed )
          return;
        auto child = expand( ntk_.get_node( f ) );
        if ( !child )
        {
          failed = true;
          return;
        }
        inputs.push_back( f.complement ? ( "(~ " + *child + ")" ) : *child );
      } );
      if ( failed )
        return std::nullopt;

      auto const& cell = ntk_.get_cell( current );
      if ( cell.gates.empty() || inputs.size() != cell.gates.front().num_vars )
      {
        failed = true;
        return std::nullopt;
      }

      std::string expression = "(" + cell.name;
      for ( auto const& input : inputs )
      {
        expression += " ";
        expression += input;
      }
      expression += ")";
      memo.emplace( index, expression );
      cone.insert( index );
      context.cone_order.push_back( index );
      return expression;
    };

    auto expression = expand( root );
    if ( !expression )
      return std::nullopt;

    context.expression = std::move( *expression );
    context.cone_nodes.assign( cone.begin(), cone.end() );
    context.original_area = mffc_area( context.cone_nodes, root_index );
    context.original_level = root_index < levels_.size() ? levels_[root_index] : 0u;
    context.original_delay = root_index < arrivals_.size() ? arrivals_[root_index] : 0.0;
    context.critical = ps_.path_objective && is_critical( root_index, context.original_delay );
    /* The care mask starts out "completely specified"; `run_round` computes the real don't cares once
       per cut, after the context exists. (The old call here passed `force = false` and therefore
       always returned immediately.) */
    context.care_mask = full_care_mask( context.leaves.size() );
    context.has_dont_cares = false;
    return context;
  }

  static uint32_t dc_max_pis()
  {
    static uint32_t const cap = [] {
      char const* v = std::getenv( "EPMAPPING_DC_MAX_PIS" );
      /* 8, not the original hard-coded 12: the don't-care window simulation is `2^n_pis`, and on the
         ASPDAC'25 subject graph the staged flow times out at the 1800 s budget on `sqrt`, `mem_ctrl`,
         `ethernet` and `RISC` with 12 while 8 gives *bit-identical* area/delay/cec 4.4x cheaper. Set
         `EPMAPPING_DC_MAX_PIS=12` for the old behaviour. */
      /* 1..12: the simulation enumerates `2^n_pis` patterns, so a larger cap is neither sound to
         shift (`1u << n_pis`) nor affordable. 12 was the original hard-coded value. */
      return v == nullptr ? 8u
                          : static_cast<uint32_t>( std::max( 1, std::min( 12, std::atoi( v ) ) ) );
    }();
    return cap;
  }

  static bool care_cache_verify()
  {
    static bool const on = std::getenv( "EPMAPPING_CARE_VERIFY" ) != nullptr;
    return on;
  }

  /// `EPMAPPING_CARE_CACHE=off|set` (default `set`): memoise the per-cut care mask by leaf set.
  static bool care_cache_on()
  {
    static bool const on = [] {
      char const* v = std::getenv( "EPMAPPING_CARE_CACHE" );
      return v == nullptr || std::string{ v } != "off";
    }();
    return on;
  }

  /// Cache key of a cut for [`compute_care_mask`]: the leaf **set**, i.e. the leaf indices sorted,
  /// plus the permutation that maps the caller's leaf order onto that sorted order
  /// (`order[k]` is the position in `leaves` of the k-th smallest leaf index). Keying on the set
  /// rather than the exact vector is what makes sibling cuts that share leaves hit.
  static std::pair<std::string, std::array<uint8_t, MaxCutSize>> care_key(
      std::vector<uint32_t> const& leaves, bool sort_leaves )
  {
    std::array<uint8_t, MaxCutSize> order{};
    for ( size_t i = 0u; i < leaves.size() && i < MaxCutSize; ++i )
      order[i] = static_cast<uint8_t>( i );
    if ( sort_leaves )
      std::sort( order.begin(), order.begin() + std::min<size_t>( leaves.size(), MaxCutSize ),
                 [&]( uint8_t a, uint8_t b ) { return leaves[a] < leaves[b]; } );
    std::string key;
    key.resize( leaves.size() * sizeof( uint32_t ) );
    for ( size_t k = 0u; k < leaves.size(); ++k )
      std::memcpy( key.data() + k * sizeof( uint32_t ), &leaves[order[k]], sizeof( uint32_t ) );
    return { std::move( key ), order };
  }

  /// A care mask is a truth table over `2^n` minterms, so re-indexing it is a permutation of the
  /// **minterm index bits**, not of the mask's own bits: minterm `m` in the caller's leaf order
  /// (`bit j` of `m` = value of `leaves[j]`) becomes the minterm whose `bit k` is the value of
  /// `sorted_leaves[k] = leaves[order[k]]`.
  static uint32_t minterm_to_sorted( uint32_t m, std::array<uint8_t, MaxCutSize> const& order,
                                    size_t n )
  {
    uint32_t out = 0u;
    for ( size_t k = 0u; k < n; ++k )
      if ( ( m >> order[k] ) & 1u )
        out |= ( uint32_t{ 1u } << k );
    return out;
  }

  /// The inverse: sorted order -> caller order (`bit order[k]` of the result is `bit k` of `m`).
  static uint32_t minterm_from_sorted( uint32_t m, std::array<uint8_t, MaxCutSize> const& order,
                                      size_t n )
  {
    uint32_t out = 0u;
    for ( size_t k = 0u; k < n; ++k )
      if ( ( m >> k ) & 1u )
        out |= ( uint32_t{ 1u } << order[k] );
    return out;
  }

  /// Caller leaf order -> sorted leaf order, minterm-wise.
  static uint64_t mask_to_sorted( uint64_t mask, std::array<uint8_t, MaxCutSize> const& order,
                                 size_t n )
  {
    uint64_t out = 0u;
    for ( uint32_t m = 0u; m < ( uint32_t{ 1u } << n ); ++m )
      if ( ( mask >> m ) & 1u )
        out |= ( uint64_t{ 1u } << minterm_to_sorted( m, order, n ) );
    return out;
  }

  /// Sorted leaf order -> caller leaf order, minterm-wise.
  static uint64_t mask_from_sorted( uint64_t mask, std::array<uint8_t, MaxCutSize> const& order,
                                   size_t n )
  {
    uint64_t out = 0u;
    for ( uint32_t m = 0u; m < ( uint32_t{ 1u } << n ); ++m )
      if ( ( mask >> m ) & 1u )
        out |= ( uint64_t{ 1u } << minterm_from_sorted( m, order, n ) );
    return out;
  }

  /// All-ones care mask over the `2^n` minterms of an `n`-leaf cut (`n` is clamped to the six
  /// minterm bits a `uint64_t` holds).
  static uint64_t full_care_mask( size_t n )
  {
    return ( n >= 6u ) ? ~uint64_t{ 0u }
                       : ( ( uint64_t{ 1u } << ( uint64_t{ 1u } << n ) ) - 1u );
  }

  /// Satisfiability don't cares of the cut leaves, as a care mask (bit `i` = 1 means minterm `i`
  /// must be matched). `satisfiability_dont_cares` computes an **under-approximation** of the true
  /// don't-care set from an exhaustive window simulation, so any completion is functionally safe.
  ///
  /// The computation is by far the staged flow's largest per-cut cost (window enumeration plus an
  /// exhaustive simulation of up to `2^12` window-input assignments), and the same ordered leaf list
  /// recurs across the cuts of different roots, so the mask is memoised per round. A cached mask
  /// stays *sound* after substitutions because `try_replace` only accepts covers that match the cone
  /// on the care minterms, so the functions driving any leaf — and therefore the satisfiability
  /// don't-care set itself — are preserved.
  void compute_care_mask( cut_context& context, bool force = false ) const
  {
    auto const n = context.leaves.size();
    auto const minterms = ( n >= 6u ) ? 64u : ( uint64_t{ 1u } << n );
    auto const full = ( minterms == 64u ) ? ~uint64_t{ 0u } : ( ( uint64_t{ 1u } << minterms ) - 1u );
    context.care_mask = full;
    context.has_dont_cares = false;
    if ( !ps_.use_dont_cares || !force || n < 2u || n > 6u )
      return;

    /* `EPMAPPING_DC_WINDOW=per_node`: one window plus one simulation per **root**, shared by all of its
       cuts (mockturtle `rewrite`). The window is built lazily on the first DC-eligible cut of the root
       and rebuilt whenever the root changes; because the mask then depends on the root and not only on
       the leaf set, the leaf-set memo is bypassed. A cut whose leaves are not all inside the window gets
       no don't cares (upstream's containment fallback). */
    if ( ps_.dc_window == params::dc_window_mode::per_node )
    {
      auto const t_win = std::chrono::steady_clock::now();
      if ( !node_window_ || node_window_->root_index != context.root_index )
        node_window_ = build_node_window( context.root_index );
      if ( node_window_ && node_window_->valid )
      {
        ++st_dc_node_window_cuts_;
        if ( auto const care = project_care( node_window_->tts, node_window_->n_pis, context.leaves ) )
          context.care_mask = *care;
        else
          ++st_dc_node_window_no_dc_;
      }
      /* count the per-node window path into the same timer as the per-cut one, so `dc_window` is not
         reported as zero when `per_node` (the default) is active. */
      st_dc_window_ += secs_since( t_win );
      context.has_dont_cares = ( context.care_mask != full );
      return;
    }


    bool const cached_call = care_cache_on();
    if ( cached_call )
      ++st_care_lookups_;
    auto const [key, order] = care_key( context.leaves, true );
    if ( cached_call )
    if ( auto const it = care_mask_cache_.find( key ); it != care_mask_cache_.end() )
    {
      ++st_care_hits_;
      context.care_mask = mask_from_sorted( it->second, order, n );
      context.has_dont_cares = ( context.care_mask != full );
      if ( care_cache_verify() )
      {
        auto const fresh = fresh_care_mask( context.leaves );
        if ( fresh != context.care_mask )
        {
          ++st_care_mismatches_;
          if ( st_care_mismatches_ <= 3u )
            std::cerr << "[epmapping] care-cache mismatch n=" << n
                      << " cached=" << context.care_mask << " fresh=" << fresh << std::endl;
        }
      }
      return;
    }

    auto const care = fresh_care_mask( context.leaves );
    context.care_mask = care;
    context.has_dont_cares = ( care != full );
    if ( cached_call )
      care_mask_cache_.emplace( std::move( key ), mask_to_sorted( care, order, n ) );
  }

  /// The uncached satisfiability-don't-care mask of `leaves`, in the caller's leaf order.
  uint64_t fresh_care_mask( std::vector<uint32_t> const& leaves ) const
  {
    auto const n = leaves.size();
    auto const minterms = ( n >= 6u ) ? 64u : ( uint64_t{ 1u } << n );
    auto const full = ( n >= 6u ) ? ~uint64_t{ 0u } : ( ( uint64_t{ 1u } << minterms ) - 1u );

    /* Don't cares are an optimization, not a correctness requirement: once the measured don't-care
       time exceeds `EPMAPPING_DC_TIME_BUDGET` the remaining cuts are matched exactly (no DCs). This
       is what keeps designs whose windows are large (`sqrt`, `div`, `mem_ctrl`) inside a bounded
       runtime instead of spending 50-60% of the pass in `compute_care_mask`. */
    if ( ps_.dc_time_budget > 0.0 && dc_time_spent_ >= ps_.dc_time_budget )
    {
      ++dc_budget_stops_;
      return full;
    }
    auto const t_dc = std::chrono::steady_clock::now();

    std::vector<node> leaf_nodes;
    leaf_nodes.reserve( n );
    for ( auto const idx : leaves )
      leaf_nodes.push_back( ntk_.index_to_node( idx ) );

    auto const t_window = std::chrono::steady_clock::now();
    auto const dc = window_dont_cares( leaf_nodes );
    st_dc_window_ += secs_since( t_window );
    uint64_t care = 0u;
    for ( uint64_t i = 0u; i < minterms; ++i )
      if ( !kitty::get_bit( dc, i ) )
        care |= ( uint64_t{ 1u } << i );
    /* The window simulation over-approximates reachability (it replaces the logic above the window
       with free variables) and gives up entirely when the window is wider than
       `EPMAPPING_DC_MAX_PIS`; the SAT refinement clears the minterms the real fanin cone cannot
       produce, and is itself skipped once the budget is gone (still sound: it only clears bits). */
    if ( ps_.sat_dc )
      if ( auto const refined = sat_care_refine( leaves, care ) )
        care = *refined;
    dc_time_spent_ += secs_since( t_dc );
    return care;
  }

  /// SAT refinement of a care mask: a care minterm that the **fanin cone of the cut's leaves** cannot
  /// produce is a satisfiability don't care, and an UNSAT query proves it.
  ///
  /// The cone is encoded as a CNF (constants fixed, primary inputs free, every gate a Tseitin
  /// relation over its pin function), and the minterm space is explored with recursive pruning: an
  /// unsatisfiable partial assignment clears the whole sub-cube with one call. Only *clearing* bits
  /// is possible, and every cleared bit is proved unreachable, so running out of budget (see
  /// `EPMAPPING_SAT_DC_CALLS`) leaves the window mask in place — still sound, just less sharp.
  std::optional<uint64_t> sat_care_refine( std::vector<uint32_t> const& leaves, uint64_t care ) const
  {
    auto const n = leaves.size();
    if ( n < 1u || n > 6u )
      return std::nullopt;
    auto const minterms = uint32_t{ 1u } << n;
    auto const full = ( n == 6u ) ? ~uint64_t{ 0u } : ( ( uint64_t{ 1u } << minterms ) - 1u );
    care &= full;
    if ( care == 0u )
      return uint64_t{ 0u };

    auto const t_cone = std::chrono::steady_clock::now();

    /* Topological fanin cone of the leaves, constants and PIs included (the leaves are interior
       nodes of that cone and become assumption variables). Iterative DFS: the cone can be deep. */
    std::unordered_set<uint32_t> in_cone;
    std::vector<uint32_t> order;
    {
      std::vector<std::pair<uint32_t, bool>> stack;
      for ( auto const l : leaves )
        stack.emplace_back( l, false );
      while ( !stack.empty() )
      {
        auto const [idx, expanded] = stack.back();
        stack.pop_back();
        if ( expanded )
        {
          order.push_back( idx );
          continue;
        }
        if ( !in_cone.insert( idx ).second )
          continue;
        stack.emplace_back( idx, true );
        auto const nd = ntk_.index_to_node( idx );
        if ( ntk_.is_constant( nd ) || ntk_.is_pi( nd ) )
          continue;
        ntk_.foreach_fanin( nd, [&]( auto const& f ) {
          auto const fi = ntk_.node_to_index( ntk_.get_node( f ) );
          if ( in_cone.find( fi ) == in_cone.end() )
            stack.emplace_back( fi, false );
        } );
      }
    }
    st_dc_cone_ += secs_since( t_cone );
    if ( order.size() > ps_.sat_dc_max_nodes )
      return std::nullopt; /* cone too large to encode per cut: fall back to the window mask */
    auto const t_sat = std::chrono::steady_clock::now();

    std::unordered_map<uint32_t, uint32_t> var_of;
    var_of.reserve( order.size() * 2u );
    bill::solver<bill::solvers::bsat2> solver;
    for ( auto const idx : order )
      var_of.emplace( idx, solver.add_variable() );

    auto const lit = []( uint32_t var, bool value ) {
      return bill::lit_type( var, value ? bill::lit_type::polarities::positive
                                        : bill::lit_type::polarities::negative );
    };

    for ( auto const idx : order )
    {
      auto const nd = ntk_.index_to_node( idx );
      auto const out_var = var_of.at( idx );
      if ( ntk_.is_constant( nd ) )
      {
        solver.add_clause( lit( out_var, ntk_.constant_value( nd ) ) );
        continue;
      }
      if ( ntk_.is_pi( nd ) )
        continue;

      auto const function = ntk_.node_function( nd );
      uint32_t fanin_count = 0u;
      std::vector<uint32_t> fanin_var;
      std::vector<uint8_t> fanin_neg;
      ntk_.foreach_fanin( nd, [&]( auto const& f ) {
        fanin_var.push_back( var_of.at( ntk_.node_to_index( ntk_.get_node( f ) ) ) );
        fanin_neg.push_back( f.complement ? 1u : 0u );
        ++fanin_count;
      } );
      if ( fanin_count == 0u || fanin_count > 6u )
        return std::nullopt; /* unexpected gate shape: fall back to the window mask */
      /* Forbid every input pattern that the gate's pin function maps to the other output value. */
      for ( uint32_t pattern = 0u; pattern < ( uint32_t{ 1u } << fanin_count ); ++pattern )
      {
        uint32_t function_index = 0u;
        for ( uint32_t k = 0u; k < fanin_count; ++k )
        {
          auto const bit = static_cast<uint32_t>( ( pattern >> k ) & 1u ) ^ fanin_neg[k];
          function_index |= bit << k;
        }
        auto const out = kitty::get_bit( function, function_index ) ? 1u : 0u;
        std::vector<bill::lit_type> clause;
        clause.reserve( fanin_count + 1u );
        /* This clause is false exactly for `inputs == pattern && out != function(pattern)`: each
           input literal is false when its variable equals the pattern bit, and the output literal is
           false when the output differs from the function value. */
        for ( uint32_t k = 0u; k < fanin_count; ++k )
          clause.push_back( lit( fanin_var[k], ( ( pattern >> k ) & 1u ) == 0u ) );
        clause.push_back( lit( out_var, out != 0u ) );
        solver.add_clause( clause );
      }
    }

    std::vector<uint32_t> leaf_var( n );
    for ( size_t i = 0u; i < n; ++i )
    {
      auto const it = var_of.find( leaves[i] );
      if ( it == var_of.end() )
        return std::nullopt;
      leaf_var[i] = it->second;
    }

    std::vector<bill::lit_type> assumptions;
    assumptions.reserve( n );
    uint64_t cleared = 0u;
    uint64_t calls = 0u;
    uint64_t assigned_low = 0u;
    bool budget_hit = false;
    std::function<void( uint32_t )> rec = [&]( uint32_t depth ) {
      if ( budget_hit )
        return;
      if ( ps_.sat_dc_calls != 0u && calls >= ps_.sat_dc_calls )
      {
        budget_hit = true;
        return;
      }
      ++calls;
      auto const state = solver.solve( assumptions );
      if ( state == bill::result::states::unsatisfiable )
      {
        /* The whole sub-cube is unreachable: every completion is a satisfiability don't care. */
        auto const low_bits = ( uint32_t{ 1u } << depth ) - 1u;
        for ( uint32_t m = 0u; m < minterms; ++m )
        {
          if ( ( m & low_bits ) == assigned_low )
            cleared |= ( uint64_t{ 1u } << m );
        }
        return;
      }
      if ( depth == n )
        return; /* reachable: keep the care bit */
      for ( uint32_t value = 0u; value < 2u; ++value )
      {
        assumptions.push_back( lit( leaf_var[depth], value != 0u ) );
        assigned_low |= static_cast<uint64_t>( value ) << depth;
        rec( depth + 1u );
        assigned_low &= ~( uint64_t{ 1u } << depth );
        assumptions.pop_back();
      }
    };
    rec( 0u );

    st_dc_sat_ += secs_since( t_sat );
    ++st_sat_cuts_;
    st_sat_calls_ += calls;
    st_sat_cleared_ += static_cast<uint64_t>( __builtin_popcountll( cleared ) );
    if ( budget_hit )
      ++st_sat_budget_;
    return care & ~cleared;
  }

  /// Per-node don't-care window (`EPMAPPING_DC_WINDOW=per_node`): the reconvergence window of one root,
  /// built and simulated **once**, plus the per-node truth tables every cut of that root is projected
  /// from. Mirrors mockturtle's `rewrite::perform_rewriting_dc()` (`rps.max_leaves = ps.window_size`),
  /// where the creation limit *is* the simulation cap, so no window is discarded.
  struct dc_node_window
  {
    uint32_t root_index{ 0u };
    uint32_t n_pis{ 0u };
    std::unordered_map<uint32_t, kitty::dynamic_truth_table> tts;
    bool valid{ false };
  };

  /// Collect and simulate a reconvergence window: `extended_leaves` are its free inputs and `roots` the
  /// nodes it was built for. Returns one `dynamic_truth_table` per window node, over the window's free
  /// inputs, or `nullopt` when the window cannot be simulated reliably.
  ///
  /// `collect_nodes` walks *fanins* from the roots down to the window inputs, so it needs only a color
  /// view -- building a `fanout_view` of the whole network here (as the original code did) costs
  /// O(|network|) *per window* and dominated the whole don't-care computation (measured on `aes_core`:
  /// 97 s of a 100 s `care`, against 3.8 s for the actual `2^n_pis` simulation).
  ///
  /// Bit-parallel simulation, the way mockturtle's `satisfiability_dont_cares` / `rewrite` do it with
  /// `simulate_nodes`: every node gets a `dynamic_truth_table` over the window's `n_pis` free inputs, so
  /// all `2^n_pis` patterns are evaluated 64 at a time (one word) instead of one hash-map lookup per
  /// pattern and per gate. Window inputs become variables, constants become all-0/all-1 tables, and a
  /// gate's table is the OR over the minterms of its pin function of the AND of the (possibly
  /// complemented) pin tables -- pin `k` of the genlib function is variable `k` (LSB-first), matching
  /// the netlist connection order.
  std::optional<std::unordered_map<uint32_t, kitty::dynamic_truth_table>>
  simulate_window( std::vector<node> const& extended_leaves, std::vector<node> const& roots ) const
  {
    auto const n_pis = static_cast<uint32_t>( extended_leaves.size() );
    mockturtle::color_view<network> color_ntk{ ntk_ };
    /* `collect_nodes` returns nodes in topological (children-first) order and includes `roots`. */
    auto const t_collect = std::chrono::steady_clock::now();
    std::vector<node> const gates{ mockturtle::collect_nodes( color_ntk, extended_leaves, roots ) };
    st_dc_collect_ += secs_since( t_collect );

    ++dc_window_calls_;
    dc_window_nodes_ += gates.size();
    dc_window_nodes_max_ = std::max<uint64_t>( dc_window_nodes_max_, gates.size() );

    auto const t_sim = std::chrono::steady_clock::now();
    std::unordered_map<uint32_t, kitty::dynamic_truth_table> tts;
    tts.reserve( gates.size() + n_pis );
    kitty::dynamic_truth_table ones( n_pis );
    ones = ~ones;
    kitty::dynamic_truth_table const zeros( n_pis );
    for ( uint32_t i = 0u; i < n_pis; ++i )
    {
      kitty::dynamic_truth_table var( n_pis );
      kitty::create_nth_var( var, static_cast<uint8_t>( i ) );
      tts[ntk_.node_to_index( extended_leaves[i] )] = std::move( var );
    }

    for ( auto const& n : gates )
    {
      auto const idx = ntk_.node_to_index( n );
      if ( tts.find( idx ) != tts.end() )
        continue; /* free window input */
      auto const function = ntk_.node_function( n );
      auto const pins = static_cast<uint32_t>( function.num_vars() );
      if ( pins == 0u )
      {
        tts[idx] = ntk_.constant_value( n ) ? ones : zeros;
        continue;
      }
      std::vector<kitty::dynamic_truth_table> pin_tt;
      pin_tt.reserve( pins );
      bool valid = true;
      ntk_.foreach_fanin( n, [&]( auto const& f ) {
        auto const fn = ntk_.get_node( f );
        if ( ntk_.is_constant( fn ) )
          pin_tt.push_back( ntk_.constant_value( fn ) ? ones : zeros );
        else if ( auto const it = tts.find( ntk_.node_to_index( fn ) ); it != tts.end() )
          pin_tt.push_back( f.complement ? ~it->second : it->second );
        else
        {
          valid = false;
          pin_tt.push_back( zeros );
        }
      } );
      if ( !valid || pin_tt.size() != pins )
      {
        st_dc_sim_ += secs_since( t_sim );
        return std::nullopt; /* cannot simulate this window reliably */
      }

      kitty::dynamic_truth_table out( n_pis );
      for ( uint32_t m = 0u; m < ( uint32_t{ 1u } << pins ); ++m )
      {
        if ( !kitty::get_bit( function, m ) )
          continue;
        kitty::dynamic_truth_table term = ones;
        for ( uint32_t p = 0u; p < pins; ++p )
          term &= ( ( m >> p ) & 1u ) ? pin_tt[p] : ~pin_tt[p];
        out |= term;
      }
      tts[idx] = std::move(out);
    }
    st_dc_sim_ += secs_since( t_sim );
    return tts;
  }

  /// Care minterms of `leaves` (in the caller's leaf order) projected out of a simulated window: the set
  /// of leaf-value combinations that some assignment of the window's free inputs produces. `nullopt`
  /// when a leaf has no table, i.e. it is not inside the window -- the containment fallback mockturtle's
  /// `rewrite` applies (that cut then gets no don't cares).
  std::optional<uint64_t> project_care( std::unordered_map<uint32_t, kitty::dynamic_truth_table> const& tts,
                                        uint32_t n_pis, std::vector<uint32_t> const& leaves ) const
  {
    auto const assignments = uint32_t{ 1u } << n_pis;
    uint64_t care = 0u;
    for ( uint32_t i = 0u; i < assignments; ++i )
    {
      uint32_t entry = 0u;
      for ( size_t j = 0u; j < leaves.size(); ++j )
      {
        auto const nd = ntk_.index_to_node( leaves[j] );
        if ( ntk_.is_constant( nd ) )
        {
          entry |= static_cast<uint32_t>( ntk_.constant_value( nd ) ? 1u : 0u ) << j;
          continue;
        }
        auto const it = tts.find( leaves[j] );
        if ( it == tts.end() )
          return std::nullopt;
        entry |= static_cast<uint32_t>( kitty::get_bit( it->second, i ) ) << j;
      }
      care |= uint64_t{ 1u } << entry;
    }
    return care;
  }

  /// Local re-implementation of `mockturtle::satisfiability_dont_cares`, the **per-cut** window
  /// (`EPMAPPING_DC_WINDOW=per_cut`): the window is the reconvergence cut of the *cut's own leaves*, so
  /// every cut gets a care set and the mask is a pure function of the leaf set (which is what the
  /// per-round memo caches).
  ///
  /// mockturtle's version builds a `window_view`, which only maps the **false** constant node;
  /// `block_network` has a separate true-constant node, so its `simulate_nodes` throws. Here the
  /// window's gates are simulated directly (constants handled explicitly) with no `window_view`.
  kitty::dynamic_truth_table window_dont_cares( std::vector<node> const& leaves ) const
  {
    /* The window is allowed to grow to `|C| + 4` (at least 8) and is then **discarded** when it has
       more than `EPMAPPING_DC_MAX_PIS` free inputs, because the simulation enumerates `2^n_pis`
       patterns. Sizing the window *to* the cap instead (the way `mockturtle::satisfiability_dont_cares`
       does with `max_tfi_inputs`) was measured too: it does give more cuts a don't-care set
       (`spi` 3 657 -> 4 640, `tv80` 11 636 -> 12 726, `aes_core` 37 970 -> 38 365 cuts) but costs
       `spi` 0.17% area (149.10 -> 149.35, `cec=1`) and changes nothing on the other designs, so the
       window keeps growing and the over-wide ones report no don't cares. */
    auto const n_leaves = static_cast<uint32_t>( leaves.size() );
    auto const no_dc = [n_leaves] { return kitty::dynamic_truth_table( n_leaves ); };
    auto const cap = dc_max_pis();
    mockturtle::reconvergence_driven_cut_parameters rcps;
    rcps.max_leaves = std::max<uint64_t>( 8u, static_cast<uint64_t>( leaves.size() ) + 4u );
    mockturtle::reconvergence_driven_cut_statistics rcst;
    auto const t_cut = std::chrono::steady_clock::now();
    mockturtle::detail::reconvergence_driven_cut_impl<network, false, false> cuts( ntk_, rcps, rcst );
    auto const extended_leaves = cuts.run( leaves ).first;
    st_dc_cut_ += secs_since( t_cut );

    auto const n_pis = static_cast<uint32_t>( extended_leaves.size() );
    /* The window simulation enumerates `2^n_pis` assignments, so the window size is the single biggest
       factor in this function's cost; `EPMAPPING_DC_MAX_PIS` caps it (default 8; 12 was the original
       hard-coded value). Giving up on a large window reports **no** don't cares (an all-zero don't-care
       mask), which is conservative and keeps the result sound. */
    if ( n_pis == 0u || n_pis > cap )
      return no_dc();

    auto const tts = simulate_window( extended_leaves, leaves );
    if ( !tts )
      return no_dc();

    std::vector<uint32_t> leaf_indices;
    leaf_indices.reserve( leaves.size() );
    for ( auto const l : leaves )
      leaf_indices.push_back( ntk_.node_to_index( l ) );
    auto const care = project_care( *tts, n_pis, leaf_indices );
    if ( !care )
      return no_dc();

    /* The returned table is the **don't-care** mask (bit set = don't care), matching the callers. */
    kitty::dynamic_truth_table care_tt( n_leaves );
    for ( uint32_t m = 0u; m < ( uint32_t{ 1u } << n_leaves ); ++m )
      if ( ( *care >> m ) & 1u )
        kitty::set_bit( care_tt, m );
    return ~care_tt;
  }

  /// `EPMAPPING_DC_WINDOW=per_node`: the reconvergence window of `root_index`, built once and simulated
  /// once for all of that root's cuts (mockturtle `rewrite`: `rps.max_leaves = window_size`). The
  /// creation limit *is* the simulation cap here, so the window is never discarded.
  std::optional<dc_node_window> build_node_window( uint32_t root_index ) const
  {
    auto const cap = dc_max_pis();
    if ( root_index >= ntk_.size() )
      return std::nullopt;
    auto const root_node = ntk_.index_to_node( root_index );
    if ( ntk_.is_constant( root_node ) || ntk_.is_pi( root_node ) || ntk_.is_dead( root_node ) )
      return std::nullopt;

    mockturtle::reconvergence_driven_cut_parameters rcps;
    rcps.max_leaves = cap;
    mockturtle::reconvergence_driven_cut_statistics rcst;
    auto const t_cut = std::chrono::steady_clock::now();
    mockturtle::detail::reconvergence_driven_cut_impl<network, false, false> cuts( ntk_, rcps, rcst );
    std::vector<node> const roots{ root_node };
    auto const extended_leaves = cuts.run( roots ).first;
    st_dc_cut_ += secs_since( t_cut );

    auto const n_pis = static_cast<uint32_t>( extended_leaves.size() );
    ++st_dc_node_windows_;
    st_dc_node_window_pis_max_ = std::max<uint64_t>( st_dc_node_window_pis_max_, n_pis );
    if ( n_pis == 0u || n_pis > cap )
      return std::nullopt;

    auto tts = simulate_window( extended_leaves, roots );
    if ( !tts )
      return std::nullopt;

    dc_node_window win;
    win.root_index = root_index;
    win.n_pis = n_pis;
    win.tts = std::move( *tts );
    win.valid = true;
    return win;
  }


  /// Area actually removed when the cone's root is replaced.
  ///
  /// This is the fanout-free cone rooted at `root_index`, **not** merely the nodes whose fanouts
  /// stay inside the cut cone: a node only dies once *every* one of its fanouts dies, and that
  /// chain is transitive. Compute the fixpoint by starting from all cone nodes and repeatedly
  /// dropping any node that still has a fanout outside the surviving set (the root always dies
  /// because `substitute_node` rewires all of its parents).
  std::unordered_set<uint32_t> removable_nodes( std::vector<uint32_t> const& cone, uint32_t root_index ) const
  {
    /* Single peelfixpoint instead of the previous O(|cone|^2) scan: `remaining[v]` is how many of
       `v`'s fanouts are still alive, the root dies first (it is replaced) and a node dies once the
       last of its fanouts does. */
    std::unordered_map<uint32_t, uint32_t> remaining;
    remaining.reserve( cone.size() * 2u );
    for ( auto const v : cone )
      remaining.emplace( v, ntk_.fanout_size( ntk_.index_to_node( v ) ) );
    std::unordered_set<uint32_t> dead;
    std::vector<uint32_t> peel{ root_index };
    while ( !peel.empty() )
    {
      auto const v = peel.back();
      peel.pop_back();
      if ( !dead.insert( v ).second )
        continue;
      ntk_.foreach_fanin( ntk_.index_to_node( v ), [&]( auto const& f ) {
        auto const it = remaining.find( ntk_.node_to_index( ntk_.get_node( f ) ) );
        if ( it == remaining.end() || it->second == 0u )
          return;
        if ( --it->second == 0u )
          peel.push_back( it->first );
      } );
    }

    return dead;
  }

  double mffc_area( std::vector<uint32_t> const& cone, uint32_t root_index ) const
  {
    auto const dead=removable_nodes(cone,root_index);
    double area = 0.0;
    for ( auto const idx : dead )
    {
      auto const n = ntk_.index_to_node( idx );
      if ( ntk_.has_cell( n ) )
        area += ntk_.get_cell( n ).area;
    }
    return area;
  }

  uint32_t candidate_level( CCost const& cost, cut_context const& context ) const
  {
    auto const& nodes = cost.cell_nodes;
    std::vector<uint32_t> lv( nodes.size(), 0u );
    for ( size_t i = 0u; i < nodes.size(); ++i )
    {
      auto const& nd = nodes[i];
      if ( nd.kind == 0u )
      {
        lv[i] = nd.input_idx < context.leaf_levels.size() ? context.leaf_levels[nd.input_idx] : 0u;
      }
      else if ( nd.kind == 1u )
      {
        lv[i] = 0u;
      }
      else
      {
        uint32_t m = 0u;
        for ( size_t k = 0u; k < nd.fanins.size(); ++k )
        {
          auto const fi = nd.fanins[k];
          if ( fi >= lv.size() )
            return 0u;
          auto const d = lv[fi] + ( ( ( nd.neg_mask >> k ) & 1u ) ? 1u : 0u );
          m = std::max( m, d );
        }
        lv[i] = m + 1u + ( nd.out_neg != 0u ? 1u : 0u );
      }
    }
    return cost.cell_root < lv.size() ? lv[cost.cell_root] : 0u;
  }

  /* Comparisons are done on quantized keys: raw floating-point near-ties can flip with codegen
     changes, and because the pass is greedy and sequential a single flip cascades into a different
     final netlist. Integer keys make the result reproducible. */
  static int64_t quantum( double value, double scale )
  {
    return static_cast<int64_t>( std::llround( value * scale ) );
  }
  static int64_t area_key( double area ) { return quantum( area, 1e4 ); }   /* 1e-4 */
  static int64_t delay_key( double delay ) { return quantum( delay, 1e3 ); } /* 1e-3 */

  uint32_t candidate_rejection_kind( CCost const& cost, cut_context const& context, bool count=true ) const
  {
    auto const cand_level = candidate_level( cost, context );
    auto const cand_delay = static_cast<double>( cost.delay );
    if ( ps_.preserve_depth && cand_level > context.original_level )
    {
      if(count) ++staged_depth_rejections_;
      return 1u;
    }
    if ( ps_.preserve_delay || ps_.use_required_times )
    {
      auto const limit = ( ps_.use_required_times ? required_time( context.root_index )
                                                 : context.original_delay * ( 1.0 + ps_.delay_slack ) ) + 1e-3;
      if ( cand_delay > limit )
      {
        if(count) ++staged_delay_rejections_;
        return 2u;
      }
    }
    if ( ps_.local_area_growth >= 0.0 &&
         cost.area > context.original_area * ( 1.0 + ps_.local_area_growth ) + 1e-4 )
    {
      if(count) ++staged_objective_rejections_;
      return 3u;
    }

    if ( objective_key( context_objective( context ), cost.area, cand_delay ) <
         objective_key( context_objective( context ), context.original_area, context.original_delay ) )
      return 0u;
    if(count) ++staged_objective_rejections_;
    return 3u;
  }


  bool candidate_is_better( CCost const& a, cut_context const& actx,
                            CCost const& b, cut_context const& bctx ) const
  {
    auto const al = candidate_level( a, actx );
    auto const bl = candidate_level( b, bctx );
    auto const ak = objective_key( context_objective( actx ), a.area - ( ps_.rank_area_gain ? actx.original_area : 0.0 ), a.delay );
    auto const bk = objective_key( context_objective( bctx ), b.area - ( ps_.rank_area_gain ? bctx.original_area : 0.0 ), b.delay );
    // Depth is only a deterministic tie-break after BOTH requested objectives.
    return std::make_tuple( ak.first, ak.second, al ) < std::make_tuple( bk.first, bk.second, bl );
  }

  uint8_t context_objective( cut_context const& context ) const
  {
    return ps_.path_objective ? ( context.critical ? 0u : 1u ) : ps_.objective;
  }

  bool is_critical( uint32_t index, double arrival ) const
  {
    return required_time( index ) - arrival <= ps_.critical_slack;
  }

  void update_live_delay_target()
  {
    double worst = 0.0;
    ntk_.foreach_po( [&]( auto const& f ) { worst = std::max( worst, arrival_of_signal( f ) ); } );
    live_delay_target_ = std::min( delay_bound_, worst );
  }

  /// Required arrival at a node under the fixed output budget. Only live physical fanin edges
  /// contribute; memoization is invalidated after every substitution and round rebuild.
  double required_time( uint32_t index ) const
  {
    if ( auto const it = required_cache_.find( index ); it != required_cache_.end() )
      return it->second;
    // An unexpected cycle is rejected conservatively instead of recursing indefinitely.
    required_cache_[index] = -std::numeric_limits<double>::infinity();
    double required = std::numeric_limits<double>::infinity();
    ntk_.foreach_po( [&]( auto const& f ) {
      if ( ntk_.node_to_index( ntk_.get_node( f ) ) == index )
        required = std::min( required, ( ps_.path_objective ? live_delay_target_ : delay_bound_ ) -
                                       ( f.complement ? inv_delay_ : 0.0 ) );
    } );
    if ( index < fanouts_.size() )
      for ( auto const parent : fanouts_[index] )
      {
        auto const n = ntk_.index_to_node( parent );
        if ( ntk_.is_dead( n ) || !ntk_.has_cell( n ) )
          continue;
        auto const& cell = ntk_.get_cell( n );
        if ( cell.gates.empty() )
          continue;
        uint32_t pin = 0u;
        ntk_.foreach_fanin( n, [&]( auto const& f ) {
          if ( ntk_.node_to_index( ntk_.get_node( f ) ) == index )
            required = std::min( required, required_time( parent ) -
                                          pin_delay_of( cell.gates.front(), pin ) -
                                          ( f.complement ? inv_delay_ : 0.0 ) );
          ++pin;
        } );
      }
    required_cache_[index] = required;
    return required;
  }

  /// Unrounded arrival time of the `input_idx`-th cut leaf. The vector handed to `map_egg` is
  /// quantized to integer library time units (`std::lround`), which is coarse for a library whose
  /// values have 0.01 granularity (asap7: 14.77 … 37.11), so the decision path uses this instead.
  double leaf_arrival( cut_context const& context, uint8_t input_idx ) const
  {
    if ( input_idx >= context.leaves.size() )
      return 0.0;
    auto const index = context.leaves[input_idx];
    return index < arrivals_.size() ? std::max( 0.0, arrivals_[index] ) : 0.0;
  }

  /// Arrival time at the cover's root, computed from the cut leaves' arrivals with exactly the
  /// model [`build_candidate`] materializes: per-fanin pin delay plus an inverter for every input
  /// negation and for a negated output. Used for **every** candidate source so their delays are
  /// comparable regardless of what the source reported.
  std::optional<double> cover_delay( cut_context const& context, CCost const& cost ) const
  {
    if ( ps_.phase_cost && ps_.phase_reuse )
    {
      auto const priced = phase_cover_cost( context, cost );
      return priced ? std::optional<double>{ priced->second } : std::nullopt;
    }
    auto const& nodes = cost.cell_nodes;
    if ( cost.cell_root >= nodes.size() )
      return std::nullopt;
    auto const& lib = ntk_.get_library();
    std::vector<double> arrival( nodes.size(), 0.0 );
    for ( size_t i = 0u; i < nodes.size(); ++i )
    {
      auto const& nd = nodes[i];
      if ( nd.kind == 0u )
      {
        arrival[i] = leaf_arrival( context, nd.input_idx );
        continue;
      }
      if ( nd.kind == 1u )
        continue;

      auto const it = cells_by_name_.find( static_cast<std::string>( nd.cell_name ) );
      if ( it == cells_by_name_.end() || it->second.index >= lib.size() )
        return std::nullopt;
      auto const& gate = lib[it->second.index].gates.front();
      double worst = 0.0;
      for ( size_t k = 0u; k < nd.fanins.size(); ++k )
      {
        if ( nd.fanins[k] >= i )
          return std::nullopt; /* not topological */
        double const edge = ( ( nd.neg_mask >> k ) & 1u ) ? inv_delay_ : 0.0;
        /* the signal reaches the cell at the **physical** pin `pin_names[k]`, which is not
           necessarily pin `k` of the library gate (the e-graph cover carries permuted variants). */
        worst = std::max( worst, arrival[nd.fanins[k]] + pin_delay_of_gate_pin( it->second, gate, static_cast<std::string>( nd.pin_names[k] ) ) + edge );
      }
      if ( nd.out_neg != 0u )
        worst += inv_delay_;
      arrival[i] = worst;
    }
    return arrival[cost.cell_root];
  }

  /// Symbolic materialization counts new reachable cells and retained removable physical gates;
  /// already shared signals have zero incremental cost. Cancellation follows make_inverter, including cut
  /// leaves. No netlist nodes are created here, and general external gate sharing is not credited.
  std::optional<std::pair<double, double>> phase_cover_cost( cut_context const& context,
                                                           CCost const& cost ) const
  {
    struct item { double area, arrival; std::vector<size_t> fanins; std::optional<size_t> inverse; };
    std::vector<item> graph;
    std::unordered_map<uint32_t, size_t> existing;
    std::unordered_set<uint32_t> uncached_removed;
    if(ps_.live_query_memo & 2u) {
      if(!context.removed_cache || context.removed_revision!=live_revision_ || context.removed_slots!=ntk_.size()) {
        context.removed_cache=removable_nodes(context.cone_nodes,context.root_index);
        context.removed_revision=live_revision_;context.removed_slots=ntk_.size();++removed_builds_;
      } else ++removed_hits_;
    } else {uncached_removed=removable_nodes(context.cone_nodes,context.root_index);++removed_builds_;}
    auto const& removed=(ps_.live_query_memo & 2u) ? *context.removed_cache : uncached_removed;
    std::function<size_t(uint32_t)> external = [&]( uint32_t index ) -> size_t {
      if ( auto it = existing.find(index); it != existing.end() ) return it->second;
      auto n = ntk_.index_to_node(index);
      auto id = graph.size(); existing[index] = id;
      // Retaining a removable physical gate consumes its area; a shared gate would survive
      // anyway and is free incremental area. Boundary signals stay outside `removed`.
      auto area=removed.count(index) && ntk_.has_cell(n) ? ntk_.get_cell(n).area : 0.0;
      graph.push_back({area, index < arrivals_.size() ? arrivals_[index] : 0.0, {}, std::nullopt});
      if ( !ntk_.is_pi(n) && !ntk_.is_constant(n) && ntk_.has_cell(n) &&
           ntk_.num_outputs(n)==1u && ntk_.fanin_size(n)==1u && ntk_.node_function(n)==not_tt_ )
        ntk_.foreach_fanin(n, [&](auto const& f) {
          if (!f.complement) { auto child=external(ntk_.node_to_index(ntk_.get_node(f))); graph[id].inverse=child; }
        });
      if(!graph[id].inverse)
        if(auto opposite=external_inverse_fanout(index,context.root_index)) {
          auto child=external(ntk_.node_to_index(ntk_.get_node(*opposite))); graph[id].inverse=child;
        }
      return id;
    };
    auto invert = [&]( size_t id ) -> size_t {
      if ( graph[id].inverse ) return *graph[id].inverse;
      auto out=graph.size();
      graph.push_back({inv_cell_area_,graph[id].arrival+inv_delay_,{id},id}); return out;
    };
    auto const& nodes=cost.cell_nodes;
    if(cost.cell_root>=nodes.size()) return std::nullopt;
    std::vector<size_t> ids;
    for(size_t i=0;i<nodes.size();++i)
    {
      auto const& nd=nodes[i];
      if(nd.kind==0u) {
        if(nd.input_idx>=context.leaves.size()) return std::nullopt;
        ids.push_back(external(context.leaves[nd.input_idx])); continue;
      }
      if(nd.kind==1u) {
        auto id=graph.size(); graph.push_back({0.0,0.0,{},std::nullopt});
        if(nd.const_val) id=invert(id);
        ids.push_back(id); continue;
      }
      auto it=cells_by_name_.find(static_cast<std::string>(nd.cell_name));
      if(it==cells_by_name_.end() || nd.pin_names.size()!=nd.fanins.size()) return std::nullopt;
      auto const& gate=ntk_.get_library()[it->second.index].gates.front();
      std::vector<size_t> children; double arrival=0.0;
      for(size_t k=0;k<nd.fanins.size();++k) {
        if(nd.fanins[k]>=i) return std::nullopt;
        auto child=ids[nd.fanins[k]];
        if((nd.neg_mask>>k)&1u) child=invert(child);
        children.push_back(child);
        arrival=std::max(arrival,graph[child].arrival+pin_delay_of_gate_pin(it->second,gate,static_cast<std::string>(nd.pin_names[k])));
      }
      size_t id;
      if(children.size()==1u && gate.function==not_tt_) id=invert(children.front());
      else { id=graph.size(); graph.push_back({gate.area,arrival,std::move(children),std::nullopt}); }
      if(nd.out_neg) id=invert(id);
      ids.push_back(id);
    }
    double area=0.0; std::unordered_set<size_t> seen;
    std::function<void(size_t)> visit=[&](size_t id) {
      if(!seen.insert(id).second) return;
      area+=graph[id].area;
      for(auto child:graph[id].fanins) visit(child);
    };
    visit(ids[cost.cell_root]);
    if(std::getenv("EPMAPPING_PHASE_COST_TRACE") && area+1e-4<cost.area)
      std::cerr << "[phase_cost] root=" << context.root_index << " standalone=" << cost.area
                << " live=" << area << " delay=" << graph[ids[cost.cell_root]].arrival << '\n';
    return std::pair<double,double>{area,graph[ids[cost.cell_root]].arrival};
  }



  /// Pure e-graph (egg) rewriting of the cut function (no DB/library pre-check, no arrival input).
  /// The cover(s) the e-graph stage offers for `context`.
  ///
  /// `ps_.egg_required` (default on) passes the cut leaves' arrivals and the cut cone's original
  /// delay as an output-delay bound, so heuristic extraction filters its retained covers by an
  /// estimated timing limit rather than simply taking an unconstrained area winner.
  /// The bound carries a small margin because the arrivals handed to `map_egg` are integers while
  /// the acceptance test is exact: a slightly loose bound only wastes a candidate, since `consider`
  /// re-derives the real delay from the netlist's own arrivals before any comparison.
  ///
  /// Only one cost model exists: `map_egg` prices the extraction with the real library
  /// (`(area, library delay)`); the `depth` variant the old `EPMAPPING_EGG_COST` advertised was never
  /// wired through the staged entry point, and requesting `both` only ran the same saturation twice.
  std::vector<CCost> call_egg_covers( cut_context const& context, stats& st )
  {
    std::vector<CCost> out;
    // Arrival information is needed for both objectives even without an output bound.
    auto const* arrivals = context.leaf_arrivals.data();
    float const required = !(ps_.phase_cost && ps_.phase_reuse) && ( ps_.egg_required || ps_.use_required_times )
                               ? static_cast<float>( ( ps_.use_required_times ? required_time( context.root_index ) : context.original_delay ) + 1.0 )
                               : std::numeric_limits<float>::infinity();
    auto const covers = cut_mapping_egg_arr( context.expression,
                                             arrivals,
                                             context.leaf_arrivals.size(),
                                             context_objective( context ),
                                             required,
                                             1u,
                                             0u );
    for ( auto const& c : covers )
      if ( auto const v = check_result( c ) )
      {
        ++st.egg_covers;
        out.push_back( *v );
      }
    return out;
  }

  static double secs_since( std::chrono::steady_clock::time_point t0 )
  {
    return std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
  }

  /// Is `context` the cut whose cone is exactly the target's MFFC (the "MFFC cut")? Only those cuts
  /// replace the whole MFFC, which is where the staged flow runs e-graph rewriting.
  static bool is_mffc_cut( cut_context const& context, std::vector<uint32_t> const& mffc )
  {
    if ( mffc.empty() || context.cone_nodes.size() != mffc.size() )
      return false;
    std::unordered_set<uint32_t> const in_cone( context.cone_nodes.begin(), context.cone_nodes.end() );
    for ( auto const v : mffc )
      if ( in_cone.find( v ) == in_cone.end() )
        return false;
    return true;
  }





  std::optional<CCost> check_result( CCost const& result )
  {
    if ( result.cell_nodes.empty() )
    {
      last_failure_reason_ = result.failure_reason.empty()
                                 ? std::string( "unknown" )
                                 : static_cast<std::string>( result.failure_reason );
      return std::nullopt;
    }
    return result;
  }



  /// Staged flow stage 2: don't-care aware match against precomputed **multi-cell** sub-networks of
  /// at most `max_vars` inputs (the four-input database).
  std::optional<CCost> multi_cell_cover_dc( cut_context const& context, uint32_t max_vars = 4u ) const
  {
    auto const n = context.leaves.size();
    if ( n == 0u || n > 6u || (context.cone_order.empty() && !context.dependency) )
      return std::nullopt;
    return match_partial_truth_table( context, truth_of( context ), context.care_mask,
                                      ps_.dc_max_free, /*multi_cell=*/true, max_vars );
  }




  /// Supergate entries tried per phase by the exact (completely specified) matcher.
  static constexpr uint32_t c_exact_entry_limit{ 128u };

  /// Structural-synthesis candidate (`EPMAPPING_SYN`).
  ///
  /// The library stages answer a cut with a cover *picked from the library*; this stage builds a new
  /// multi-level network for the cut function instead — SOP factoring over the cut's truth table (the
  /// same algebraic restructuring ABC's `&sopb`/`&syn4` performs, here per cone) — with the cut's
  /// satisfiability don't cares as flexibility, maps it with the engine's own technology library, and
  /// hands the cover to the normal acceptance gate. Constant or trivial functions are left to the
  /// library stages.
  /// Structural-synthesis candidates (`EPMAPPING_SYN`): one cover per enabled synthesis mode.
  ///
  /// `EPMAPPING_SYN_MODE` bit 0 selects SOP factoring (`&sopb`-style), bit 1 disjoint-support
  /// decomposition with SOP factoring on the prime nodes (`&dsd`-style). Both build a *new*
  /// multi-level network for the cut function and map it with the engine's own technology library,
  /// so the acceptance gate compares them with the library covers on equal terms.
  std::vector<CCost> synthesize_covers( cut_context const& context, stats& st,
                                        bool is_mffc_cut_root = false )
  {
    std::vector<CCost> out;
    if ( ps_.synth == 0u || tech_ == nullptr )
      return out;
    auto const n = context.leaves.size();
    if ( n < (context.dependency ? 2u : ps_.synth_min_leaves) || n > 6u || (context.cone_order.empty() && !context.dependency) )
      return out;

    auto const tt = cone_truth_table_dynamic( context );
    if ( kitty::is_const0( tt ) || kitty::is_const0( ~tt ) )
      return out;

    ++st.synth_roots;
    auto const t_syn = std::chrono::steady_clock::now();
    bool const dbg = std::getenv( "EPMAPPING_SYN_DEBUG" ) != nullptr;
    if ( dbg )
      std::cerr << "[syn] cut n=" << n << " care=" << context.care_mask << "\n";

    /* Don't cares give the synthesizer room: minterms the circuit cannot produce need not be
       implemented, which is exactly how a smaller cover becomes legal. */
    std::optional<kitty::dynamic_truth_table> dc;
    if ( ps_.synth_use_dc && context.has_dont_cares )
    {
      kitty::dynamic_truth_table dc_tt( static_cast<uint32_t>( n ) );
      auto const minterms = uint32_t{ 1u } << n;
      for ( uint32_t m = 0u; m < minterms; ++m )
        if ( ( ( context.care_mask >> m ) & 1u ) == 0u )
          kitty::set_bit( dc_tt, m );
      if ( !kitty::is_const0( dc_tt ) )
        dc = dc_tt;
    }

    auto const run = [&]( uint32_t mode ) -> std::optional<CCost> {
      /* Mapping here has zero-arrival PIs and a fixed library. Cache the structure per objective,
         never the final cost: the same truth table can occur at different live cut arrivals.
         Exact synthesis is excluded because its conflict-limited search has its own cache. */
      std::string const key = std::to_string( n ) + ":" + std::to_string( truth_of( context ) ) +
                              ":" + std::to_string( dc ? dc->_bits[0] : 0u ) + ":" +
                              std::to_string( mode ) + ":" + std::to_string( context_objective( context ) );
      bool const cacheable = mode < 2u && ps_.synth_cache_entries != 0u;
      std::optional<std::pair<rust::Vec<MappedNode>, uint32_t>> built;
      bool hit = false;
      if ( cacheable )
      {
        ++st.synth_cache_lookups;
        if ( auto const cached = synth_cache_.find( key ) )
        {
          built = *cached;
          hit = true;
          ++st.synth_cache_hits;
        }
      }
      if ( !hit )
      {
        if(!operator_policy_.allow(0,policy_bucket(context),false))return std::nullopt;
        auto const build_start=operator_policy_.enabled()?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        built = synthesize_support_nodes( tt, dc, mode, context_objective( context ) );
        if(operator_policy_.enabled())operator_policy_.build_elapsed(0,policy_bucket(context),secs_since(build_start));
        if ( cacheable )
          synth_cache_.insert( key, built, ps_.synth_cache_entries, ps_.synth_cache_replace );
      }
      else operator_policy_.allow(0,policy_bucket(context),true);
      if ( !built )
        return std::nullopt;
      auto cover = finalize_local_cover( context, std::move( built->first ), built->second );
      if ( !cover )
        return std::nullopt;
      /* A synthesis bug must degrade to "no candidate", never to a wrong netlist. */
      if ( !cover_matches_tt( *cover, context, truth_of( context ), context.care_mask ) )
        return std::nullopt;
      ++st.synth_covers;
      return cover;
    };

    for ( uint32_t mode = 0u; mode < 3u; ++mode )
    {
      if ( ( ( ps_.synth_modes >> mode ) & 1u ) == 0u )
        continue;
      /* The exact mode is the most expensive one and is practical only for small functions, so it is
         gated to `EPMAPPING_SYN_EXACT_LEAVES` leaves and (by default) to the MFFC cut. */
      if ( mode == 2u && ps_.synth_exact_mffc && !is_mffc_cut_root )
        continue;
      if ( auto cover = run( mode ) )
        out.push_back( std::move( *cover ) );
    }
    st.secs_synth += secs_since( t_syn );
    if(operator_policy_.enabled())operator_policy_.elapsed(0,policy_bucket(context),secs_since(t_syn));
    return out;
  }

  /// Alternate cut decompositions, mapped independently instead of selecting by AIG gate count.
  /// Exact cofactors are used even when the cut has don't cares. Thus all candidates implement
  /// the full cut function; care masks are only an additional caller-side equivalence check.
  std::vector<std::pair<CCost,uint32_t>> cofactor_covers(cut_context const& context,stats& st,std::optional<double> incumbent=std::nullopt)
  {
    std::vector<std::pair<CCost,uint32_t>> out;
    auto const n=static_cast<uint32_t>(context.leaves.size());
    if(!tech_ || n<ps_.cofactor_min_leaves || n>6u) return out;
    auto const tt=cone_truth_table_dynamic(context);
    if(kitty::is_const0(tt) || kitty::is_const0(~tt)) return out;
    ++st.cofactor_cuts; auto start=std::chrono::steady_clock::now();
    std::unordered_set<std::string> seen;
    if(!incumbent) incumbent=context.original_area;
    for(uint32_t kind=0;kind<4;++kind) {
      if(!(ps_.cofactor_modes & (kind==0 ? 1u : kind==1 ? 2u : 4u))) continue;
      auto const family_start=operator_policy_.enabled()?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
      auto count=kind==0 ? 2u : n;
      std::vector<uint32_t> pivots(count);std::iota(pivots.begin(),pivots.end(),0u);
      if(kind!=0 && ps_.cofactor_pivots && ps_.cofactor_pivots<count) {
        // Favor decompositions with small cofactor/difference supports. This is only
        // a search-budget heuristic; the normal exact timing/area guards still apply.
        auto score=[&](uint32_t p) {
          auto lo=kitty::cofactor0(tt,p),hi=kitty::cofactor1(tt,p),delta=lo^hi;
          uint32_t l=0,h=0,d=0;
          for(uint32_t i=0;i<n;++i) {l+=kitty::has_var(lo,i);h+=kitty::has_var(hi,i);d+=kitty::has_var(delta,i);}
          return kind==1 ? l+h : d+(kind==2 ? l:h);
        };
        std::stable_sort(pivots.begin(),pivots.end(),[&](auto a,auto b){return score(a)<score(b);});
        pivots.resize(ps_.cofactor_pivots);
        std::sort(pivots.begin(),pivots.end()); // Preserve original order among retained candidates.
      }
      for(auto pivot:pivots) {
        auto key=std::to_string(n)+":"+std::to_string(truth_of(context))+":"+
                 std::to_string(kind)+":"+std::to_string(pivot)+":"+std::to_string(context_objective(context));
        std::shared_ptr<cached_cofactor const> built;
        auto found=cofactor_cache_.find(key);
        if(found) {built=*found;++st.cofactor_hits;operator_policy_.allow(kind+1,policy_bucket(context),true);}
        else {
          bool promising=false;
          if(operator_policy_.enabled() && kind!=0) {
            auto lo=kitty::cofactor0(tt,pivot),hi=kitty::cofactor1(tt,pivot),delta=lo^hi;
            uint32_t l=0,h=0,d=0;
            for(uint32_t i=0;i<n;++i){l+=kitty::has_var(lo,i);h+=kitty::has_var(hi,i);d+=kitty::has_var(delta,i);}
            promising=lo!=hi && (kind==1 ? std::max(l,h)+2<=n : d<=1);
          }
          if(!operator_policy_.allow(kind+1,policy_bucket(context),false,promising))continue;
          auto const build_start=operator_policy_.enabled()?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
          ++st.cofactor_builds;
          auto const factor_start=std::chrono::steady_clock::now();
          mockturtle::aig_network aig; std::vector<mockturtle::aig_network::signal> pis;
          for(uint32_t i=0;i<n;++i) pis.push_back(aig.create_pi());
          // Avoid choosing between output phases by AIG count before technology mapping.
          mockturtle::sop_factoring_params params;params.try_both_polarities=false;
          mockturtle::sop_factoring<mockturtle::aig_network> factor(params);
          auto synth=[&](kitty::dynamic_truth_table const& f) {
            auto root=aig.get_constant(false);
            factor(aig,f,pis.begin(),pis.end(),[&](auto const& signal){root=signal;});return root;
          };
          auto root=aig.get_constant(false);
          if(kind==0) {root=synth(pivot ? ~tt : tt);if(pivot) root=!root;}
          else {
            auto lo=kitty::cofactor0(tt,pivot),hi=kitty::cofactor1(tt,pivot);
            if(kind==1) {auto l=synth(lo),h=synth(hi);root=aig.create_ite(pis[pivot],h,l);}
            else {
              auto delta=synth(lo^hi);
              if(kind==2) {auto l=synth(lo);root=aig.create_xor(l,aig.create_and(pis[pivot],delta));}
              else {auto h=synth(hi);root=aig.create_xor(h,aig.create_and(!pis[pivot],delta));}
            }
          }
          aig.create_po(root);
          st.secs_factor+=secs_since(factor_start);
          std::string aig_key;
          if(ps_.build_reuse) {
            auto const key_start=std::chrono::steady_clock::now();
            aig_key=exact_aig_key(aig,truth_of(context),context_objective(context));
            if(auto alias=aig_mapping_cache_.find(aig_key)) {
              ++st.aig_alias_hits;
              if(ps_.build_reuse==2){built=*alias;++st.aig_reused;}
            }
            st.secs_aig_key+=secs_since(key_start);
          }
          if(!built) {
          ++st.mapping_calls;
          auto const mapping_start=std::chrono::steady_clock::now();
          mockturtle::emap_params mp;mp.cell_library=mapped_library_;mp.area_oriented_mapping=context_objective(context)==1u;
          if(mp.area_oriented_mapping) mp.relax_required=100.0;
          auto mapped=mockturtle::emap<9>(aig,*tech_,mp);
          auto value=build_mapped_nodes(mapped,n);
          st.secs_mapping+=secs_since(mapping_start);
          auto const proof_start=std::chrono::steady_clock::now();
          auto structure=value && ps_.cofactor_dedup ? mapped_structure_key(value->first,value->second) : std::string{};
          bool valid=false;
          if(value && ps_.cofactor_proof_cache) {
            // This proof depends on the ordered full function, immutable DAG and
            // fixed library, never on arrivals or the current removable cone.
            CCost proof;proof.cell_nodes=value->first;proof.cell_root=value->second;
            valid=cover_matches_tt(proof,context,truth_of(context),~uint64_t{0});
            ++st.cofactor_proofs;
          }
          double const lower=value && ps_.operator_policy_mode==3 ? mapped_noninverter_area(value->first,value->second) : 0.0;
          built=std::make_shared<cached_cofactor>(cached_cofactor{std::move(value),std::move(structure),valid,lower});
          st.secs_proof+=secs_since(proof_start);
          if(ps_.build_reuse) aig_mapping_cache_.insert(aig_key,built,ps_.synth_cache_entries,ps_.synth_cache_replace);
          }
          cofactor_cache_.insert(key,built,ps_.synth_cache_entries,ps_.synth_cache_replace);
          if(operator_policy_.enabled())operator_policy_.build_elapsed(kind+1,policy_bucket(context),secs_since(build_start));
        }
        if(!built->mapped) continue;
        if(ps_.cofactor_proof_cache && !built->valid) continue;
        if(ps_.cofactor_dedup && !seen.insert(built->structure).second) {
          ++st.cofactor_duplicates;continue;
        }
        // Only strict area dominance is safe: ties may improve delay or depth.
        // Inverters are omitted because live phase reuse/cancellation may be free.
        if(ps_.operator_policy_mode==3 && context_objective(context)==1u &&
           built->noninverter_area>*incumbent+0.001) {
          operator_policy_.dominated(kind+1,policy_bucket(context));continue;
        }
        auto cover=finalize_local_cover(context,built->mapped->first,built->mapped->second);
        if(!cover) continue;
        if(!ps_.cofactor_proof_cache) {
          ++st.cofactor_proofs;
          if(!cover_matches_tt(*cover,context,truth_of(context),~uint64_t{0})) continue;
        }
        ++st.cofactor_covers;
        if(ps_.operator_policy_mode==3 && candidate_rejection_kind(*cover,context,false)==0u)
          incumbent=std::min(*incumbent,static_cast<double>(cover->area));
        if(std::getenv("EPMAPPING_COFAC_TRACE") && candidate_rejection_kind(*cover,context,false)==0u)
          std::cerr << "[cofactor] root=" << context.root_index << " kind=" << kind << " pivot=" << pivot
                    << " original=" << context.original_area << " area=" << cover->area << " delay=" << cover->delay
                    << " expr=" << context.expression << '\n';
        out.emplace_back(std::move(*cover),kind);
      }
      if(operator_policy_.enabled())operator_policy_.elapsed(kind+1,policy_bucket(context),secs_since(family_start));
    }
    st.secs_cofactor+=secs_since(start);return out;
  }

  /// Reachable non-inverter cells are always newly materialized (no general
  /// external gate hashing). Omitting every inverter and retained external gate
  /// gives a lower bound even with boundary phase cancellation and fanout reuse.
  double mapped_noninverter_area(rust::Vec<MappedNode> const& nodes,uint32_t root) const
  {
    if(root>=nodes.size()) return 0.0;
    std::vector<bool> seen(nodes.size(),false);double area=0.0;
    std::function<void(uint32_t)> visit=[&](uint32_t i) {
      if(i>=nodes.size() || seen[i])return;
      seen[i]=true;auto const& nd=nodes[i];
      if(nd.kind!=2u)return;
      auto it=cells_by_name_.find(static_cast<std::string>(nd.cell_name));
      if(it==cells_by_name_.end())return;
      auto const& gate=ntk_.get_library()[it->second.index].gates.front();
      if(!(nd.fanins.size()==1u && gate.function==not_tt_))area+=gate.area;
      for(auto f:nd.fanins)visit(f);
    };
    visit(root);return area;
  }

  /// Exact encoding (not a hash): distinct pin assignments and physical phases stay distinct.
  static std::string mapped_structure_key(rust::Vec<MappedNode> const& nodes,uint32_t root)
  {
    std::string key=std::to_string(root)+";";
    auto field=[&](std::string const& value){key+=std::to_string(value.size())+":"+value;};
    for(auto const& nd:nodes) {
      key+=std::to_string(nd.kind)+","+std::to_string(nd.input_idx)+","+
           std::to_string(nd.const_val)+","+std::to_string(nd.neg_mask)+","+std::to_string(nd.out_neg)+";";
      field(static_cast<std::string>(nd.cell_name));field(static_cast<std::string>(nd.out_pin));
      key+=std::to_string(nd.fanins.size())+";";
      for(auto f:nd.fanins)key+=std::to_string(f)+",";
      key+=std::to_string(nd.pin_names.size())+";";
      for(auto const& pin:nd.pin_names)field(static_cast<std::string>(pin));
    }
    return key;
  }

  /// Build a mapped cover for `tt` (over `n` support variables): algebraic synthesis (SOP factoring or
  /// DSD + factoring, optionally with don't cares) followed by technology mapping, returned as
  /// `(nodes, root)` in the engine's cover format. Input `k` of the returned cover is support variable
  /// `k`, so the caller decides what that means: a cut leaf, or a position in a cone cut.
  ///
  /// `mode` 0 = SOP factoring (`&sopb`-style), 1 = DSD with SOP factoring on the prime nodes
  /// (`&dsd`-style).
  /// Synthesize one function into an existing AIG (`pis[i]` is variable `i`) and return its root
  /// signal. `mode` 0 = SOP factoring, 1 = DSD + factoring, 2 = exact AIG synthesis (percy/SAT).
  std::optional<mockturtle::aig_network::signal>
  build_support_function( mockturtle::aig_network& aig,
                          std::vector<mockturtle::aig_network::signal> const& pis,
                          kitty::dynamic_truth_table const& tt,
                          std::optional<kitty::dynamic_truth_table> const& dc,
                          uint32_t mode ) const
  {
    auto const n = tt.num_vars();
    if ( n == 0u || n > 6u || pis.size() != n )
      return std::nullopt;
    bool have_root = false;
    mockturtle::aig_network::signal root_signal{};
    auto const store_root = [&]( auto const& f ) {
      root_signal = f;
      have_root = true;
    };
    /* `mockturtle::sop_factoring` stores its parameters **by reference**
       (`sop_factoring_params const& _ps;`) while its constructor's default argument is `{}`:
       a default-constructed `sop_factoring` therefore leaves `_ps` bound to a temporary that died
       at the end of the constructor, and whether `_ps.try_both_polarities` reads as true or false
       depends on whatever the stack holds at that moment (UBSan: "load of value 64, which is not a
       valid value for type 'bool'"). That made the synthesized covers — and with them the staged
       reject counters, occasionally the final area — differ from run to run. Pass a
       static-storage-duration object so the reference stays valid. */
    static mockturtle::sop_factoring_params const sop_ps{};
    if ( mode == 0u )
    {
      mockturtle::sop_factoring<mockturtle::aig_network> resyn( sop_ps );
      if ( dc )
        resyn( aig, tt, *dc, pis.begin(), pis.end(), store_root );
      else
        resyn( aig, tt, pis.begin(), pis.end(), store_root );
    }
    else if ( mode == 1u )
    {
      mockturtle::sop_factoring<mockturtle::aig_network> prime( sop_ps );
      mockturtle::dsd_resynthesis<mockturtle::aig_network,
                                   mockturtle::sop_factoring<mockturtle::aig_network>> resyn( prime );
      resyn( aig, tt, pis.begin(), pis.end(), store_root );
    }
    else
    {
      /* Exact AIG synthesis (percy, SAT): minimum AND count over AIG primitives, don't cares
         honoured, and a function-keyed cache so the SAT cost is paid once per distinct function.
         This is the only mode that is *optimal* rather than heuristic — optimal in AND count, not in
         library area, which is why its cover still goes through the normal area/delay gate. */
      if ( n > ps_.synth_exact_leaves )
        return std::nullopt;
      mockturtle::exact_resynthesis_params eps;
      eps.cache = exact_cache_;
      eps.blacklist_cache = exact_blacklist_;
      eps.conflict_limit = static_cast<int>( ps_.synth_exact_conflict );
      mockturtle::exact_aig_resynthesis<mockturtle::aig_network> resyn( false, eps );
      if ( dc )
        resyn( aig, tt, *dc, pis.begin(), pis.end(), store_root );
      else
        resyn( aig, tt, pis.begin(), pis.end(), store_root );
    }
    if ( !have_root )
      return std::nullopt;
    if ( std::getenv( "EPMAPPING_SYN_DEBUG" ) != nullptr )
      std::cerr << "[syn] mode=" << mode << " n=" << n << " factored gates=" << aig.num_gates()
                << " root_complemented=" << ( root_signal.complement ? 1 : 0 ) << "\n";
    return root_signal;
  }

  std::optional<std::pair<rust::Vec<MappedNode>, uint32_t>>
  synthesize_support_nodes( kitty::dynamic_truth_table const& tt,
                            std::optional<kitty::dynamic_truth_table> const& dc,
                            uint32_t mode, uint8_t objective ) const
  {
    auto const n = tt.num_vars();
    if ( tech_ == nullptr || n == 0u || n > 6u )
      return std::nullopt;

    mockturtle::aig_network aig;
    std::vector<mockturtle::aig_network::signal> pis;
    pis.reserve( n );
    for ( uint32_t i = 0u; i < n; ++i )
      pis.push_back( aig.create_pi() );

    auto const root_signal = build_support_function( aig, pis, tt, dc, mode );
    if ( !root_signal )
      return std::nullopt;

    aig.create_po( *root_signal );
    mockturtle::emap_params map_ps;
    map_ps.cell_library=mapped_library_;
    map_ps.area_oriented_mapping = ( objective == 1u );
    map_ps.relax_required = map_ps.area_oriented_mapping ? 100.0 : map_ps.relax_required;
    /* Keep `cut_limit` at the mapper's default: the `fast_network_cuts` instantiation behind `emap`
       has a compile-time `max_cut_num` of 16 and raising the limit writes out of bounds (SIGBUS). */
    auto const mapped = mockturtle::emap<9>( aig, *tech_, map_ps );
    if ( std::getenv( "EPMAPPING_SYN_DEBUG" ) != nullptr )
      std::cerr << "[syn] mapped nodes=" << mapped.size() << " gates=" << mapped.num_gates() << "\n";
    return build_mapped_nodes( mapped, n );
  }

  /// Convert a mapped `cell_view<block_network>` (single PO, `n_leaves` PIs) into the engine's cover
  /// format: complemented edges become `neg_mask` bits and a complemented root becomes `out_neg`,
  /// exactly as the library matcher emits them, so `finalize_local_cover` (and the acceptance gate
  /// after it) costs a synthesized cover the same way as a library one.
  std::optional<std::pair<rust::Vec<MappedNode>, uint32_t>>
  build_mapped_nodes( network const& mapped, uint32_t n_leaves ) const
  {
    auto multi = build_mapped_nodes_multi( mapped, n_leaves );
    if ( !multi || multi->second.empty() )
      return std::nullopt;
    return std::make_pair( std::move( multi->first ), multi->second.front() );
  }

  /// Multi-output variant of [`build_mapped_nodes`]: returns the shared node list and the cover root
  /// of every primary output, in PO order.
  std::optional<std::pair<rust::Vec<MappedNode>, std::vector<uint32_t>>>
  build_mapped_nodes_multi( network const& mapped, uint32_t n_leaves ) const
  {
    rust::Vec<MappedNode> nodes;
    std::unordered_map<uint32_t, uint32_t> cover_of;
    std::unordered_map<uint32_t, uint32_t> pi_of;
    auto const& lib = mapped.get_library();
    mapped.foreach_pi( [&]( auto const& n, auto index ) {
      pi_of.emplace( mapped.node_to_index( n ), static_cast<uint32_t>( index ) );
    } );

    std::function<std::optional<uint32_t>( uint32_t )> materialize =
        [&]( uint32_t idx ) -> std::optional<uint32_t> {
      if ( auto const it = cover_of.find( idx ); it != cover_of.end() )
        return it->second;
      auto const nd = mapped.index_to_node( idx );
      MappedNode out;
      if ( mapped.is_constant( nd ) )
      {
        out.kind = 1u;
        out.const_val = mapped.constant_value( nd ) ? 1u : 0u;
      }
      else if ( mapped.is_pi( nd ) )
      {
        out.kind = 0u;
        auto const it = pi_of.find( idx );
        if ( it == pi_of.end() || it->second >= n_leaves )
          return std::nullopt;
        out.input_idx = static_cast<uint8_t>( it->second );
      }
      else
      {
        if ( !mapped.has_cell( nd ) )
          return std::nullopt;
        auto const cell = mapped.get_cell_index( nd );
        if ( cell >= lib.size() || lib[cell].gates.empty() )
          return std::nullopt;
        auto const& gate = lib[cell].gates.front();
        out.kind = 2u;
        out.cell_name = gate.name;
        for ( auto const& pin : gate.pins )
          out.pin_names.push_back( rust::String( pin.name ) );
        bool ok = true;
        uint32_t k = 0u;
        std::vector<uint32_t> fanins;
        std::vector<uint8_t> negs;
        mapped.foreach_fanin( nd, [&]( auto const& f ) {
          auto const child = materialize( mapped.node_to_index( mapped.get_node( f ) ) );
          if ( !child )
          {
            ok = false;
            return;
          }
          fanins.push_back( *child );
          negs.push_back( mapped.is_complemented( f ) ? 1u : 0u );
          ++k;
        } );
        if ( !ok || k == 0u || k != gate.pins.size() )
          return std::nullopt;
        for ( uint32_t i = 0u; i < k; ++i )
        {
          out.fanins.push_back( fanins[i] );
          if ( negs[i] != 0u )
            out.neg_mask |= static_cast<uint16_t>( uint16_t{ 1u } << i );
        }
      }
      auto const index = static_cast<uint32_t>( nodes.size() );
      nodes.push_back( std::move( out ) );
      cover_of.emplace( idx, index );
      return index;
    };

    std::vector<uint32_t> roots;
    mapped.foreach_po( [&]( auto const& f ) {
      auto const index = materialize( mapped.node_to_index( mapped.get_node( f ) ) );
      if ( !index )
        return;
      if ( mapped.is_complemented( f ) )
        nodes[*index].out_neg = 1u;
      roots.push_back( *index );
    } );
    if ( roots.empty() )
      return std::nullopt;
    for ( auto const r : roots )
      if ( nodes[r].kind != 2u )
        return std::nullopt;
    return std::make_pair( std::move( nodes ), std::move( roots ) );
  }


  /// Whole-window structural synthesis (`EPMAPPING_WINDOW_SYN`).
  ///
  /// Takes the MFFW of `leaves` (`build_mffw`: fanout-free, so every internal node is removable),
  /// synthesizes **all** of the window's outputs from the window inputs (SOP factoring / DSD, sharing
  /// through the AIG's structural hashing), maps the shared multi-output AIG jointly with the engine's
  /// technology library, and substitutes every output at once. A window is committed only when the
  /// **measured** area change is a win (the cover is materialized unreachable first) and the commit
  /// does not lengthen the global critical path; every output keeps its level/arrival along the way.
  /// Window outputs are observable, hence exactly equivalent: no don't cares here.
  bool window_synthesize( uint32_t root_index, std::vector<uint32_t> const& leaves, stats& st )
  {
    if ( ps_.window_synth == 0u || tech_ == nullptr || leaves.size() < 2u ||
         leaves.size() > ps_.window_synth_max_leaves )
      return false;
    auto const window = build_mffw( leaves );
    if ( window.size() <= leaves.size() + 1u ||
         window.size() > leaves.size() + ps_.window_synth_max_gates )
      return false;

    std::vector<uint8_t> in_window( ntk_.size(), 0u ), is_leaf( ntk_.size(), 0u ), po_driver( ntk_.size(), 0u );
    for ( auto const v : window )
      if ( v < in_window.size() )
        in_window[v] = 1u;
    for ( auto const l : leaves )
      if ( l < is_leaf.size() )
        is_leaf[l] = 1u;
    ntk_.foreach_po( [&]( auto const& f ) {
      auto const fi = ntk_.node_to_index( ntk_.get_node( f ) );
      if ( fi < po_driver.size() )
        po_driver[fi] = 1u;
    } );

    /* `inside[v]` = how many of `v`'s fanouts stay inside the window (used to find the outputs),
       and `local_fanouts[v]` = which window nodes they are. `block_network` has no fanout
       iteration and the `fanouts_` cache is stale after a commit, so the *only* way to get a
       fresh fanout set is to invert the fanin lists of the window. */
    std::vector<uint32_t> inside( ntk_.size(), 0u );
    std::unordered_map<uint32_t, std::vector<uint32_t>> local_fanouts;
    local_fanouts.reserve( window.size() );
    for ( auto const v : window )
    {
      if ( v >= in_window.size() || is_leaf[v] != 0u )
        continue;
      ntk_.foreach_fanin( ntk_.index_to_node( v ), [&]( auto const& f ) {
        auto const fi = ntk_.node_to_index( ntk_.get_node( f ) );
        if ( fi < inside.size() && in_window[fi] != 0u )
        {
          ++inside[fi];
          local_fanouts[fi].push_back( v );
        }
      } );
    }

    std::vector<uint32_t> outputs;
    double removable = 0.0;
    for ( auto const v : window )
    {
      if ( v >= in_window.size() || is_leaf[v] != 0u )
        continue;
      auto const n = ntk_.index_to_node( v );
      if ( !ntk_.has_cell( n ) || ntk_.get_cell( n ).gates.empty() )
        return false;
      if ( ntk_.is_dead( n ) )
        continue; /* already unreachable: it is not part of `compute_area()` either */
      removable += ntk_.get_cell( n ).area;
      auto const total = ntk_.fanout_size( n );
      auto const in_count = v < inside.size() ? inside[v] : 0u;
      if ( total > in_count || po_driver[v] != 0u )
        outputs.push_back( v );
      else if ( total != in_count )
        return false; /* inconsistent fanout information: not a clean window */
    }
    if ( outputs.empty() || removable < ps_.window_synth_min_gain )
      return false;

    /* A window input must not depend on a replaced output, or the substitution closes a cycle. */
    for ( auto const i : leaves )
      for ( auto const o : outputs )
        if ( reaches_node( ntk_.index_to_node( i ), ntk_.index_to_node( o ) ) )
          return false;

    ++st.window_synth_windows;

    /* Per-output contexts over the window inputs (functionally exact: no don't cares). */
    std::vector<cut_context> contexts;
    std::vector<uint64_t> funcs;
    contexts.reserve( outputs.size() );
    funcs.reserve( outputs.size() );
    for ( auto const o : outputs )
    {
      auto ctx = make_context( ntk_.index_to_node( o ), o, leaves );
      if ( !ctx )
        return false;
      funcs.push_back( truth_of( *ctx ) );
      contexts.push_back( std::move( *ctx ) );
    }

    struct window_candidate
    {
      rust::Vec<MappedNode> nodes;
      std::vector<uint32_t> roots;
      double area{ 0.0 };
      bool valid{ false };
    };
    std::optional<window_candidate> best;
    for ( uint32_t mode = 0u; mode < 2u; ++mode )
    {
      if ( ( ( ps_.synth_modes >> mode ) & 1u ) == 0u )
        continue;
      mockturtle::aig_network aig;
      std::vector<mockturtle::aig_network::signal> pis;
      pis.reserve( leaves.size() );
      for ( size_t i = 0u; i < leaves.size(); ++i )
        pis.push_back( aig.create_pi() );

      bool ok = true;
      for ( auto const& ctx : contexts )
      {
        auto const tt = cone_truth_table_dynamic( ctx );
        if ( kitty::is_const0( tt ) || kitty::is_const0( ~tt ) )
        {
          ok = false; /* degenerate window output: leave it to the cut stages */
          break;
        }
        auto const sig = build_support_function( aig, pis, tt, std::nullopt, mode );
        if ( !sig )
        {
          ok = false;
          break;
        }
        aig.create_po( *sig );
      }
      if ( !ok )
        continue;

      mockturtle::emap_params map_ps;
      map_ps.area_oriented_mapping = ( ps_.objective == 1u );
      map_ps.relax_required = map_ps.area_oriented_mapping ? 100.0 : map_ps.relax_required;
      /* `cut_limit` stays at the mapper's default (16): larger values write out of bounds (SIGBUS). */
      auto const mapped = mockturtle::emap<9>( aig, *tech_, map_ps );
      auto built = build_mapped_nodes_multi( mapped, static_cast<uint32_t>( leaves.size() ) );
      if ( !built || built->second.size() != contexts.size() )
        continue;
      window_candidate cand;
      cand.nodes = std::move( built->first );
      cand.roots = std::move( built->second );
      cand.valid = true;
      for ( size_t k = 0u; k < contexts.size() && cand.valid; ++k )
      {
        auto cover = finalize_local_cover( contexts[k], cand.nodes, cand.roots[k] );
        if ( !cover )
        {
          cand.valid = false;
          break;
        }
        cand.area = cover->area;
        /* window outputs are observable: the synthesized network must compute them exactly */
        if ( !cover_matches_tt( *cover, contexts[k], funcs[k], ~uint64_t{ 0u } ) )
          cand.valid = false;
        /* no output may get slower (same slack rule as the cut candidates) */
        auto const allowed = contexts[k].original_delay * ( 1.0 + std::max( 0.0, ps_.delay_slack ) ) + 1e-3;
        if ( static_cast<double>( cover->delay ) > allowed )
          cand.valid = false;
        if ( ps_.preserve_depth && candidate_level( *cover, contexts[k] ) > contexts[k].original_level )
          cand.valid = false;
      }
      if ( !cand.valid )
        continue;
      if ( cand.area >= removable - std::max( ps_.window_synth_min_gain, 1e-3 ) )
        continue; /* the whole window does not get cheaper */
      if ( !best || cand.area < best->area )
        best = std::move( cand );
    }
    if ( !best )
      return false;

    /* Two-phase commit: materialize the shared cover first (it is still unreachable), then decide on
       the **measured** area change and only rewire when it really drops.
       The old single-phase commit accepted on `removable - best->area` and only measured afterwards,
       where `removable` is the sum of the MFFW's internal cells -- i.e. it *assumed* that every window
       node dies. That assumption does not hold: `build_mffw` expands through the cached `fanouts_`
       vector, which is rebuilt once per pass, so after the first commit of a pass the window can
       contain nodes that are no longer in any output's cone and keep each other alive. Measured on
       `max` (`A_i`, one round): 9 commits claimed `gain=0.25` against `actual=-1.69`, turning a
       -0.38% win (139.76 -> 139.23 with `EPMAPPING_WINDOW_SYN=0`) into a +0.72% loss (139.76 ->
       140.76). The materialized area itself is exact: there is no structural hashing in
       `block_network`, so `new_area` below matches the charge on every measured case. */
    auto const size_before = ntk_.size();
    auto const area_before = ntk_.compute_area();
    std::vector<signal> out_signals;
    if ( !build_window_cover( best->nodes, best->roots, outputs, leaves, out_signals ) )
    {
      /* `build_window_cover` can fail after creating some nodes (unknown cell, pin-name mismatch):
         those are unreachable but not dead, so `compute_area()` would still charge them. */
      abandon_nodes( size_before );
      return false;
    }

    double new_area = 0.0;
    for ( auto i = size_before; i < ntk_.size(); ++i )
    {
      auto const n = ntk_.index_to_node( i );
      if ( !ntk_.is_dead( n ) && ntk_.has_cell( n ) )
        new_area += ntk_.get_cell( n ).area;
    }

    /* Exactly which window nodes really die. Substituting an output takes it out; every other window
       node dies once **all** of its fanouts are gone (`take_out_node` cascades through the fanout
       count). This is *not* the sum of the window's cells for the reason above, and `fanouts_` is the
       stale cache (`block_network` has no `foreach_fanout`), so the live fanout adjacency is inverted
       from the window's own fanin lists (`local_fanouts`). */
    double dead_area = 0.0;
    size_t dead_nodes = 0u;
    {
      std::vector<uint8_t> gone( ntk_.size(), 0u );
      for ( auto const o : outputs )
      {
        if ( o >= gone.size() )
          continue;
        gone[o] = 1u;
        /* the substituted node itself dies too (`substitute_node` takes it out) */
        auto const n = ntk_.index_to_node( o );
        if ( !ntk_.is_dead( n ) && ntk_.has_cell( n ) )
        {
          dead_area += ntk_.get_cell( n ).area;
          ++dead_nodes;
        }
      }
      bool changed = true;
      while ( changed )
      {
        changed = false;
        for ( auto const v : window )
        {
          if ( v >= gone.size() || gone[v] != 0u || is_leaf[v] != 0u )
            continue;
          auto const n = ntk_.index_to_node( v );
          if ( !ntk_.has_cell( n ) || ntk_.is_dead( n ) )
            continue;
          /* `v` is not an output, so all its fanouts are inside the window; it dies once every one of
             them is gone. A window node with no fanout at all is a dangling node that the commit does
             not touch, so it survives (and its area must not be counted as removed). */
          auto const it = local_fanouts.find( v );
          if ( it == local_fanouts.end() || it->second.empty() )
            continue;
          bool all_gone = true;
          for ( auto const u : it->second )
            if ( gone[u] == 0u )
            {
              all_gone = false;
              break;
            }
          if ( all_gone )
          {
            gone[v] = 1u;
            dead_area += ntk_.get_cell( n ).area;
            ++dead_nodes;
            changed = true;
          }
        }
      }
    }
    double const realized = dead_area - new_area;

    /* The commit must not lengthen the **global critical path**. `arrivals_` is only refreshed once
       per pass, so it is stale for the fanout cone of every commit and a committed window cannot be
       undone; predict the critical path the commit would produce with a fresh propagation in which
       the substituted outputs are driven by the new cover. */
    std::unordered_map<uint32_t, double> overrides;
    overrides.reserve( outputs.size() );
    for ( size_t k = 0u; k < outputs.size() && k < out_signals.size(); ++k )
      overrides[outputs[k]] = arrival_of_signal( out_signals[k] );
    double const worst_before = predict_worst_delay();
    double const worst_after = predict_worst_delay( overrides );

    /* Debug: `EPMAPPING_WINDOW_FORCE=1` commits even when the measured change is not a win (and even
       when the predicted critical path grows), so the fixpoint estimate can be compared against the
       post-commit measurement. */
    bool const force = std::getenv( "EPMAPPING_WINDOW_FORCE" ) != nullptr;
    if ( !force && worst_after > worst_before + 1e-6 )
    {
      abandon_nodes( size_before );
      ++st.window_synth_delay_rejected;
      if ( ps_.verbose )
        std::cerr << "[epmapping] window_synth root=" << root_index << " ABANDON(delay) before="
                  << worst_before << " after=" << worst_after << "\n";
      return false;
    }
    if ( !force && realized <= std::max( ps_.window_synth_min_gain, 1e-6 ) )
    {
      /* The measured change is not a win: leave the fresh nodes unreachable and mark them dead so
         they cost nothing (`compute_area()` skips dead nodes, and the next pass drops them). */
      abandon_nodes( size_before );
      ++st.window_synth_abandoned;
      if ( ps_.verbose )
        std::cerr << "[epmapping] window_synth root=" << root_index << " ABANDON predicted="
                  << ( removable - best->area ) << " measured=" << realized << "\n";
      return false;
    }

    /* Rewire every output. Each output is a boundary of the rewritten window, so everything above
       it can now see a different fanin cone: keep the fanout lists exact, re-propagate the affected
       levels/arrivals (`substitute_keep_fanouts`) and invalidate the cuts of the nodes this round has
       not handled yet (`invalidate_cuts_fanout_from`, the fanout-limited form of `rewrite.hpp`'s
       `clear_cuts_fanout_rec`). Without this, a committed window left ~5% of the remaining cuts of
       the round unmappable (`EPMAPPING_WINDOW_SYN=1` reported `rejected=300` on `c7552` against 0
       with the window off) and the rest evaluated on pre-commit arrivals. */
    std::vector<uint32_t> changed_parents;
    changed_parents.reserve( outputs.size() );
    for ( size_t k = 0u; k < outputs.size(); ++k )
      substitute_keep_fanouts( outputs[k], out_signals[k], &changed_parents );
    invalidate_cuts_fanout_from( changed_parents );
    auto const area_after = ntk_.compute_area();
    if ( ps_.verbose )
      std::cerr << "[epmapping] window_synth root=" << root_index << " leaves=" << leaves.size()
                << " nodes=" << ( window.size() - leaves.size() ) << " outputs=" << outputs.size()
                << " before=" << area_before << " dead=" << dead_area << " new=" << new_area
                << " after=" << area_after << " dead_nodes=" << dead_nodes
                << " removable=" << removable << " -> " << best->area << " measured=" << realized
                << " actual=" << ( area_before - area_after )
                << " cp=" << worst_before << "->" << worst_after << ( force ? " FORCED" : "" )
                << "\n";
    st.window_synth_actual += area_before - area_after;
    st.window_synth_gain += removable - best->area;
    ++st.window_synth_applied;
    /* One commit rewires **every** window output, and `st.replacements` is the progress counter the
       round loop uses (`run()`): counting only the cut path let a round whose only wins came from this
       fallback look empty, which stopped the iteration one round too early. */
    st.window_synth_replacements += outputs.size();
    st.replacements += outputs.size();
    return true;
  }

  /// Materialize a shared multi-output cover into the network **without rewiring anything**: the fresh
  /// nodes are unreachable, so `window_synthesize` can measure the realized area change and decide.
  /// Cells are created in the cover's topological order with a memo, so structures shared by several
  /// outputs exist only once. `out_signals[k]` is the new driver for `outputs[k]`.
  bool build_window_cover( rust::Vec<MappedNode> const& nodes, std::vector<uint32_t> const& roots,
                           std::vector<uint32_t> const& outputs, std::vector<uint32_t> const& leaves,
                           std::vector<signal>& out_signals )
  {
    auto const& lib = ntk_.get_library();
    auto const const_false = ntk_.get_constant( false );
    std::vector<signal> signals( nodes.size(), const_false );
    for ( size_t i = 0u; i < nodes.size(); ++i )
    {
      auto const& nd = nodes[i];
      if ( nd.kind == 0u )
      {
        if ( nd.input_idx >= leaves.size() )
          return false;
        signals[i] = ntk_.make_signal( ntk_.index_to_node( leaves[nd.input_idx] ) );
        continue;
      }
      if ( nd.kind == 1u )
      {
        signals[i] = ntk_.get_constant( nd.const_val != 0u );
        continue;
      }
      auto const it = cells_by_name_.find( static_cast<std::string>( nd.cell_name ) );
      if ( it == cells_by_name_.end() || it->second.index >= lib.size() )
        return false;
      auto const& info = it->second;
      auto const& gate = lib[info.index].gates.front();
      if ( gate.pins.size() != nd.pin_names.size() || nd.fanins.size() != nd.pin_names.size() )
        return false;

      std::vector<signal> children( gate.pins.size(), const_false );
      for ( size_t k = 0u; k < nd.fanins.size(); ++k )
      {
        signal sig = signals[nd.fanins[k]];
        if ( ( nd.neg_mask >> k ) & 1u )
          sig = make_inverter( sig );
        auto const pit = info.pin_pos.find( static_cast<std::string>( nd.pin_names[k] ) );
        if ( pit == info.pin_pos.end() )
          return false;
        children[pit->second] = sig;
      }
      auto const f = ntk_.create_node( children, gate.function );
      note_created( f );
      ensure_level_capacity();
      auto const new_index = ntk_.node_to_index( ntk_.get_node( f ) );
      uint32_t lv = 0u;
      double arr = 0.0;
      for ( size_t p = 0u; p < children.size(); ++p )
      {
        lv = std::max( lv, level_of_signal( children[p] ) + 1u );
        arr = std::max( arr, arrival_of_signal( children[p] ) + pin_delay_of( gate, p ) );
      }
      if ( new_index < levels_.size() )
      {
        levels_[new_index] = lv;
        arrivals_[new_index] = arr;
      }
      ntk_.add_cell( ntk_.get_node( f ), info.index );
      signal out = f;
      if ( nd.out_neg != 0u )
        out = make_inverter( out );
      signals[i] = out;
    }

    out_signals.clear();
    out_signals.reserve( outputs.size() );
    for ( size_t k = 0u; k < outputs.size(); ++k )
    {
      if ( k >= roots.size() || roots[k] >= signals.size() )
        return false;
      signal out = signals[roots[k]];
      if ( out.complement )
        out = make_inverter( out ); /* materialised here, so `new_area` sees it */
      out_signals.push_back( out );
    }
    return true;
  }

  /// Mark the nodes built by an abandoned window attempt dead. They were never reachable, so
  /// `take_out_node` (which also cascades into their fanins) is the exact inverse of the build.
  void abandon_nodes( uint32_t size_before )
  {
    for ( auto i = ntk_.size(); i-- > size_before; )
    {
      auto const n = ntk_.index_to_node( i );
      if ( !ntk_.is_dead( n ) && ntk_.fanout_size( n ) == 0u )
      {
        ntk_.foreach_fanin( n, [&]( auto const& f ) {
          fanout_remove( ntk_.node_to_index( ntk_.get_node( f ) ), i );
        } );
        ntk_.take_out_node( n );
      }
    }
  }


  /// Number of library cells in a materialized cover (leaves and constants are not cells).
  static uint32_t cover_cell_count( CCost const& cost )
  {
    uint32_t cells = 0u;
    for ( auto const& nd : cost.cell_nodes )
      if ( nd.kind == 2u )
        ++cells;
    return cells;
  }

  std::optional<CCost> match_single_cell( cut_context const& context ) const
  {
    if ( tech_ == nullptr || context.leaves.empty() || context.leaves.size() > MaxCutSize )
      return std::nullopt;

    auto const expected = cone_truth_table_dynamic( context );
    std::vector<uint32_t> support;
    auto const n = context.leaves.size();
    auto const minterms = uint32_t{ 1u } << n;
    for ( uint32_t leaf = 0u; leaf < n; ++leaf )
    {
      bool depends = false;
      for ( uint32_t m = 0u; m < minterms; ++m )
      {
        if ( ( ( m >> leaf ) & 1u ) != 0u )
          continue;
        if ( kitty::get_bit( expected, m ) != kitty::get_bit( expected, m | ( uint32_t{ 1u } << leaf ) ) )
        {
          depends = true;
          break;
        }
      }
      if ( depends )
        support.push_back( leaf );
    }
    if ( support.empty() )
      return std::nullopt;

    std::optional<CCost> best;
    auto const& cells = tech_->get_cells();
    for ( auto const& cell : cells )
    {
      if ( cell.gates.empty() )
        continue;
      auto const& gate = cell.gates.front();
      auto const pins = gate.pins.size();
      if ( pins == 0u || pins != support.size() || gate.function.num_vars() != pins )
        continue;

      std::vector<uint32_t> permutation( support.size() );
      std::iota( permutation.begin(), permutation.end(), 0u );
      do
      {
        for ( uint32_t polarity = 0u; polarity < ( uint32_t{ 1u } << pins ); ++polarity )
        {
          for ( uint32_t output_neg = 0u; output_neg < 2u; ++output_neg )
          {
            bool matches = true;
            for ( uint32_t m = 0u; m < minterms; ++m )
            {
              if ( ( ( context.care_mask >> m ) & 1u ) == 0u )
                continue;
              uint32_t gate_input = 0u;
              for ( size_t pin = 0u; pin < pins; ++pin )
              {
                auto const leaf = support[permutation[pin]];
                auto const value = static_cast<uint32_t>( ( m >> leaf ) & 1u ) ^
                                    ( ( polarity >> pin ) & 1u );
                gate_input |= value << pin;
              }
              auto value = kitty::get_bit( gate.function, gate_input ) ? 1u : 0u;
              value ^= output_neg;
              if ( value != ( kitty::get_bit( expected, m ) ? 1u : 0u ) )
              {
                matches = false;
                break;
              }
            }
            if ( !matches )
              continue;

            rust::Vec<MappedNode> nodes;
            for ( auto const leaf : support )
            {
              MappedNode input;
              input.kind = 0u;
              input.input_idx = static_cast<uint8_t>( leaf );
              nodes.push_back( std::move( input ) );
            }
            MappedNode mapped;
            mapped.kind = 2u;
            mapped.cell_name = cell.name;
            mapped.out_pin = gate.output_name;
            for ( auto const& pin : gate.pins )
              mapped.pin_names.push_back( rust::String( pin.name ) );
            mapped.neg_mask = static_cast<uint16_t>( polarity );
            mapped.out_neg = static_cast<uint8_t>( output_neg );
            for ( auto const pin : permutation )
              mapped.fanins.push_back( pin < support.size() ? pin : UINT32_MAX );
            nodes.push_back( std::move( mapped ) );
            auto cover = finalize_local_cover( context, std::move( nodes ),
                                               static_cast<uint32_t>( support.size() ) );
            if ( cover && ( !best || candidate_is_better( *cover, context, *best, context ) ) )
              best = std::move( *cover );
          }
        }
      } while ( std::next_permutation( permutation.begin(), permutation.end() ) );
    }
    return best;
  }

  /// Every library entry for `tt` over `context.leaves`, materialized (both output phases).
  std::vector<CCost> supergate_candidates( cut_context const& context, uint64_t tt,
                                           uint32_t entries,
                                           std::optional<bool> multi_cell = std::nullopt,
                                           uint32_t max_vars = 6u ) const
  {
    std::vector<CCost> out;
    if ( tech_ == nullptr || context.leaves.empty() || context.leaves.size() > 6u )
      return out;
    auto const order = cut_support_order( context.leaves.size(), tt );
    if ( order.empty() )
      return out;
    auto const compacted = compact_truth_table( context.leaves.size(), tt, order );
    kitty::dynamic_truth_table dyn( static_cast<uint32_t>( order.size() ) );
    kitty::create_from_words( dyn, &compacted, &compacted + 1 );
    auto const fe = kitty::extend_to<6>( dyn );
    for ( uint32_t phase = 0u; phase < 2u; ++phase )
    {
      auto const* list = tech_->get_supergates( phase == 0u ? fe : ~fe );
      if ( list == nullptr )
        continue;
      size_t const limit = std::min<size_t>( list->size(), entries );
      for ( size_t i = 0u; i < limit; ++i )
      {
        auto const& sg = (*list)[i];
        if ( sg.root != nullptr && sg.root->num_vars > max_vars )
          continue;
        auto cover = instantiate_supergate( context, order, sg, phase != 0u );
        if ( !cover )
          continue;
        /* Single cell vs precomputed multi-cell sub-network, decided on the materialized cover: one
           library cell is a "single-cell implementation", more than one is multi-cell. */
        if ( multi_cell.has_value() && ( cover_cell_count( *cover ) > 1u ) != *multi_cell )
          continue;
        out.push_back( std::move( *cover ) );
      }
    }
    return out;
  }

  std::optional<CCost> match_truth_table( cut_context const& context, uint64_t tt,
                                          uint32_t entries,
                                          std::optional<bool> multi_cell = std::nullopt,
                                          uint32_t max_vars = 6u ) const
  {
    std::optional<CCost> best;
    for ( auto& cover : supergate_candidates( context, tt, entries, multi_cell, max_vars ) )
      if ((!ps_.feasible_covers || candidate_rejection_kind(cover,context)==0u) &&
          (!best || candidate_is_better( cover, context, *best, context )))
        best = std::move( cover );
    return best;
  }


  /// Library cover for an **incompletely specified** function: enumerate the completions of
  /// `care_mask` (at most `params::dc_max_free` free minterms) and keep the cheapest cover.
  std::optional<CCost> match_partial_truth_table( cut_context const& context, uint64_t tt,
                                                  uint64_t care_mask,
                                                  uint32_t max_free = std::numeric_limits<uint32_t>::max(),
                                                  std::optional<bool> multi_cell = std::nullopt,
                                                  uint32_t max_vars = 6u,
                                                  uint32_t entry_limit = 0u ) const
  {
    if ( max_free == std::numeric_limits<uint32_t>::max() )
      max_free = ps_.dc_max_free;
    auto const n = context.leaves.size();
    auto const minterms = uint32_t{ 1u } << n;
    auto const full = ( n == 6u ) ? ~uint64_t{ 0u } : ( ( uint64_t{ 1u } << minterms ) - 1u );
    auto const dc = ( ~care_mask ) & full;
    if ( dc == 0u )
      return match_truth_table( context, tt,
                                entry_limit != 0u ? entry_limit : c_exact_entry_limit,
                                multi_cell, max_vars );

    std::vector<uint32_t> free_bits;
    for ( uint32_t m = 0u; m < minterms; ++m )
      if ( ( dc >> m ) & 1u )
        free_bits.push_back( m );
    if ( free_bits.empty() || free_bits.size() > max_free )
      return std::nullopt;

    std::optional<CCost> best;
    auto const per_completion = ( entry_limit != 0u ) ? entry_limit : ps_.dc_entry_limit;
    for ( uint64_t subset = 0u; subset < ( uint64_t{ 1u } << free_bits.size() ); ++subset )
    {
      auto completion = tt;
      for ( size_t k = 0u; k < free_bits.size(); ++k )
      {
        if ( ( subset >> k ) & 1u )
          completion |= ( uint64_t{ 1u } << free_bits[k] );
        else
          completion &= ~( uint64_t{ 1u } << free_bits[k] );
      }
      auto cover = match_truth_table( context, completion, per_completion, multi_cell, max_vars );
      if ( cover && ( !best || candidate_is_better( *cover, context, *best, context ) ) )
        best = std::move( cover );
    }
    return best;
  }


  /// Leaf positions (into `context.leaves`) the function actually depends on, in increasing order.
  std::vector<uint8_t> cut_support_order( size_t n, uint64_t bits ) const
  {
    auto const minterms = uint32_t{ 1u } << n;
    std::vector<uint8_t> order;
    for ( uint8_t j = 0u; j < n; ++j )
    {
      for ( uint32_t m = 0u; m < minterms; ++m )
      {
        if ( ( ( m >> j ) & 1u ) != 0u )
          continue;
        if ( ( ( bits >> m ) & 1u ) != ( ( bits >> ( m | ( uint32_t{ 1u } << j ) ) ) & 1u ) )
        {
          order.push_back( j );
          break;
        }
      }
    }
    return order;
  }

  /// The same function expressed over only the `order` leaves.
  uint64_t compact_truth_table( size_t n, uint64_t bits, std::vector<uint8_t> const& order ) const
  {
    auto const minterms = uint32_t{ 1u } << n;
    uint64_t out = 0u;
    for ( uint32_t m = 0u; m < minterms; ++m )
    {
      uint32_t compact = 0u;
      for ( size_t t = 0u; t < order.size(); ++t )
        compact |= ( ( m >> order[t] ) & 1u ) << t;
      if ( ( ( bits >> m ) & 1u ) != 0u )
        out |= ( uint64_t{ 1u } << compact );
    }
    return out;
  }

  struct resolved_child
  {
    uint32_t index{ 0u };
    uint8_t neg{ 0u };
  };

  /// Materialize one library supergate as a cover graph over the cut's leaves.
  /// Materialize one library supergate into `nodes`, taking its inputs from `input_cover`.
  ///
  /// `input_cover(ctr)` returns the index (inside `nodes`) of the cover node that drives supergate
  /// input `sg.permutation[ctr]`, in the **positive** phase; the polarity bit becomes the consuming
  /// cell's `neg_mask`. Returns the index of the sub-cover's root, or nullopt on a malformed
  /// supergate.
  template<typename InputCover>
  std::optional<uint32_t> materialize_supergate( rust::Vec<MappedNode>& nodes,
                                                 mockturtle::supergate<9> const& sg,
                                                 InputCover&& input_cover ) const
  {
    if ( sg.root == nullptr )
      return std::nullopt;
    auto const support = static_cast<size_t>( sg.root->num_vars );
    if ( support == 0u || sg.permutation.size() < support )
      return std::nullopt;

    std::vector<uint32_t> cover_of_input( support, UINT32_MAX );
    std::vector<uint8_t> neg_of_input( support, 0u );
    for ( size_t ctr = 0u; ctr < support; ++ctr )
    {
      auto const pin = static_cast<size_t>( sg.permutation[ctr] );
      if ( pin >= support || cover_of_input[pin] != UINT32_MAX )
        return std::nullopt; /* not a permutation */
      auto const index = input_cover( ctr );
      if ( index == UINT32_MAX )
        return std::nullopt;
      cover_of_input[pin] = index;
      neg_of_input[pin] = static_cast<uint8_t>( ( sg.polarity >> ctr ) & 1u );
    }

    std::function<std::optional<resolved_child>( mockturtle::composed_gate<9> const& )> build =
        [&]( mockturtle::composed_gate<9> const& g ) -> std::optional<resolved_child> {
      if ( g.root == nullptr )
      {
        if ( g.id >= support || cover_of_input[g.id] == UINT32_MAX )
          return std::nullopt;
        return resolved_child{ cover_of_input[g.id], neg_of_input[g.id] };
      }

      auto const& gate = *g.root;
      MappedNode nd;
      nd.kind = 2u;
      nd.cell_name = gate.name;
      for ( auto const& pin : gate.pins )
        nd.pin_names.push_back( rust::String( pin.name ) );

      size_t k = 0u;
      if ( g.fanin.empty() )
      {
        /* simple (single-level) gate: the supergate inputs feed its pins directly */
        for ( ; k < gate.pins.size(); ++k )
        {
          auto const input = static_cast<uint32_t>( k );
          if ( input >= support || cover_of_input[input] == UINT32_MAX )
            return std::nullopt;
          if ( neg_of_input[input] != 0u )
            nd.neg_mask |= static_cast<uint16_t>( 1u << k );
          nd.fanins.push_back( cover_of_input[input] );
        }
      }
      else
      {
        for ( auto const* child : g.fanin )
        {
          auto const r = build( *child );
          if ( !r )
            return std::nullopt;
          if ( r->neg != 0u )
            nd.neg_mask |= static_cast<uint16_t>( 1u << k );
          nd.fanins.push_back( r->index );
          ++k;
        }
      }
      if ( nd.fanins.size() != gate.pins.size() )
        return std::nullopt;

      auto const index = static_cast<uint32_t>( nodes.size() );
      nodes.push_back( std::move( nd ) );
      return resolved_child{ index, 0u };
    };

    auto const root = build( *sg.root );
    if ( !root || root->neg != 0u )
      return std::nullopt;
    return root->index;
  }

  /// Single-cell / supergate cover of the whole cut function (no cut enumeration inside the cone).
  std::optional<CCost> instantiate_supergate( cut_context const& context,
                                              std::vector<uint8_t> const& order,
                                              mockturtle::supergate<9> const& sg,
                                              bool negate_output ) const
  {
    rust::Vec<MappedNode> nodes;
    auto const root = materialize_supergate( nodes, sg, [&]( size_t ctr ) -> uint32_t {
      if ( ctr >= order.size() )
        return UINT32_MAX;
      MappedNode leaf;
      leaf.kind = 0u;
      leaf.input_idx = order[ctr];
      auto const index = static_cast<uint32_t>( nodes.size() );
      nodes.push_back( std::move( leaf ) );
      return index;
    } );
    if ( !root )
      return std::nullopt;
    if ( negate_output )
      nodes[*root].out_neg = 1u;
    return finalize_local_cover( context, std::move( nodes ), *root );
  }

  /// One candidate mapping for a local cone node.
  struct cone_choice
  {
    std::vector<uint32_t> cut;                  /* local ids, ascending */
    std::vector<uint8_t> order;                 /* support positions inside `cut` */
    mockturtle::supergate<9> const* sg{ nullptr };
    bool negate_output{ false };
    /* When non-empty, the node is implemented by this *synthesized* sub-cover rather than by a
       library supergate: input `k` of the sub-cover is support position `k`, i.e. local id
       `cut[order[k]]`. See [`synthesize_support_nodes`]. */
    rust::Vec<MappedNode> syn{};
    uint32_t syn_root{ 0u };
  };



  /// Turn a freshly built cover graph into a `CCost`: area and library-timing delay.
  std::optional<CCost> finalize_local_cover( cut_context const& context,
                                             rust::Vec<MappedNode> nodes,
                                             uint32_t root ) const
  {
    if ( root >= nodes.size() || nodes[root].kind != 2u )
      return std::nullopt;

    auto const& lib = ntk_.get_library();
    double area = 0.0;
    std::vector<double> arrival( nodes.size(), 0.0 );
    for ( size_t i = 0u; i < nodes.size(); ++i )
    {
      auto const& nd = nodes[i];
      if ( nd.kind == 0u )
      {
        arrival[i] = leaf_arrival( context, nd.input_idx );
        continue;
      }
      if ( nd.kind == 1u )
        continue;

      auto const it = cells_by_name_.find( static_cast<std::string>( nd.cell_name ) );
      if ( it == cells_by_name_.end() || it->second.index >= lib.size() )
        return std::nullopt;
      auto const& gate = lib[it->second.index].gates.front();
      area += gate.area;
      double worst = 0.0;
      for ( size_t k = 0u; k < nd.fanins.size(); ++k )
      {
        double edge = 0.0;
        if ( ( nd.neg_mask >> k ) & 1u )
        {
          edge = inv_delay_;
          area += inv_cell_area_;
        }
        worst = std::max( worst, arrival[nd.fanins[k]] + pin_delay_of_gate_pin( it->second, gate, static_cast<std::string>( nd.pin_names[k] ) ) + edge );
      }
      if ( nd.out_neg != 0u )
      {
        worst += inv_delay_;
        area += inv_cell_area_;
      }
      arrival[i] = worst;
    }

    CCost cost;
    cost.cell_nodes = std::move( nodes );
    cost.cell_root = root;
    cost.delay = static_cast<float>( arrival[root] );
    cost.area = static_cast<float>( area );
    cost.aft_dep = static_cast<uint32_t>( std::lround( arrival[root] ) );
    cost.aft_size = static_cast<uint32_t>( std::lround( area ) );
    if ( ps_.phase_cost && ps_.phase_reuse ) {
      if ( auto const priced = phase_cover_cost( context, cost ) )
      {
        cost.area=static_cast<float>(priced->first); cost.delay=static_cast<float>(priced->second);
        cost.aft_size=static_cast<uint32_t>(std::lround(priced->first));
        cost.aft_dep=static_cast<uint32_t>(std::lround(priced->second));
      }
      else if(ps_.operator_policy_mode==3) return std::nullopt; }
    return cost;
  }

  kitty::dynamic_truth_table cone_truth_table_dynamic( cut_context const& context ) const
  {
    auto const n = static_cast<uint32_t>( context.leaves.size() );
    kitty::dynamic_truth_table table( n );
    if ( n == 0u || n > MaxCutSize )
      return table;

    if ( context.dependency )
    {
      for ( uint32_t m=0; m<(1u<<n); ++m )
        if ((context.tt_cache>>m)&1u) kitty::set_bit(table,m);
      return table;
    }
    if((ps_.live_query_memo & 12u) && n<=6u) {
      table._bits[0]=truth_of(context);++dynamic_tt_hits_;return table;
    }
    auto const minterms = uint32_t{ 1u } << n;
    std::vector<uint8_t> values( ntk_.size(), 0u );
    for ( uint32_t m = 0u; m < minterms; ++m )
    {
      for ( size_t i = 0u; i < context.leaves.size(); ++i )
        values[context.leaves[i]] = static_cast<uint8_t>( ( m >> i ) & 1u );
      for ( auto const idx : context.cone_order )
      {
        auto const cone_node = ntk_.index_to_node( idx );
        auto const function = ntk_.node_function( cone_node );
        uint32_t function_index = 0u;
        uint32_t fanin_pos = 0u;
        ntk_.foreach_fanin( cone_node, [&]( auto const& f ) {
          auto const fanin_node = ntk_.get_node( f );
          uint8_t value = ntk_.is_constant( fanin_node )
                              ? static_cast<uint8_t>( ntk_.constant_value( fanin_node ) ? 1u : 0u )
                              : values[ntk_.node_to_index( fanin_node )];
          function_index |= static_cast<uint32_t>( value ^ ( f.complement ? 1u : 0u ) ) << fanin_pos;
          ++fanin_pos;
        } );
        values[idx] = kitty::get_bit( function, function_index ) ? 1u : 0u;
      }
      if ( values[context.root_index] != 0u )
        kitty::set_bit( table, m );
    }
    return table;
  }

  /// Truth table of the cut cone over its leaves, LSB-first over the fanin (= pin) order, as a bit
  /// mask over the `2^n` minterms. This legacy representation is limited to six leaves.
  uint64_t cone_truth_table( cut_context const& context ) const
  {
    auto const n = context.leaves.size();
    if((ps_.live_query_memo & 8u) && n<=6u) {
      std::vector<local_truth_node> nodes;nodes.reserve(context.cone_order.size());
      for(auto index:context.cone_order) {
        auto node=ntk_.index_to_node(index);auto function=ntk_.node_function(node);
        local_truth_node decoded;decoded.index=index;decoded.function=std::move(function._bits);
        ntk_.foreach_fanin(node,[&](auto const& f){auto n=ntk_.get_node(f);bool constant=ntk_.is_constant(n);decoded.fanins.push_back({ntk_.node_to_index(n),bool(f.complement),constant,constant && bool(ntk_.constant_value(n))});});
        nodes.push_back(std::move(decoded));
      }
      ++local_truth_simulations_;
      return simulate_local_truth(context.leaves,nodes,context.root_index,ntk_.size(),truth_scratch_);
    }
    uint64_t table = 0u;
    std::vector<uint8_t> values( ntk_.size(), 0u );
    for ( uint32_t m = 0u; m < ( uint32_t{ 1u } << n ); ++m )
    {
      for ( size_t i = 0u; i < n; ++i )
        values[context.leaves[i]] = static_cast<uint8_t>( ( m >> i ) & 1u );
      for ( auto const idx : context.cone_order )
      {
        auto const cone_node = ntk_.index_to_node( idx );
        auto const function = ntk_.node_function( cone_node );
        uint32_t function_index = 0u;
        uint32_t fanin_pos = 0u;
        ntk_.foreach_fanin( cone_node, [&]( auto const& f ) {
          auto const fanin_node = ntk_.get_node( f );
          uint8_t value = 0u;
          if ( ntk_.is_constant( fanin_node ) )
            value = ntk_.constant_value( fanin_node ) ? 1u : 0u;
          else
            value = values[ntk_.node_to_index( fanin_node )];
          function_index |= static_cast<uint32_t>( value ^ ( f.complement ? 1u : 0u ) ) << fanin_pos;
          ++fanin_pos;
        } );
        values[idx] = kitty::get_bit( function, function_index ) ? 1u : 0u;
      }
      if ( values[context.root_index] != 0u )
        table |= ( uint64_t{ 1u } << m );
    }
    return table;
  }

  /// Cached [`cone_truth_table`] of a cut (see `cut_context::tt_cache`).
  uint64_t truth_of( cut_context const& context ) const
  {
    if ( !context.tt_ready )
    {
      context.tt_cache = cone_truth_table( context );
      context.tt_ready = true;
    }
    return context.tt_cache;
  }

  std::optional<signal> build_candidate( CCost const& cost, cut_context const& context )
  {
    auto const& nodes = cost.cell_nodes;
    auto const& lib = ntk_.get_library();
    if ( cost.cell_root >= nodes.size() )
      return std::nullopt;

    /* Validate the whole cover first: creating nodes before a late failure would leak
       dangling nodes into the network. */
    for ( auto const& nd : nodes )
    {
      if ( nd.kind == 0u )
      {
        if ( nd.input_idx >= context.leaves.size() )
          return std::nullopt;
        continue;
      }
      if ( nd.kind != 2u )
        continue;

      auto const it = cells_by_name_.find( static_cast<std::string>( nd.cell_name ) );
      if ( it == cells_by_name_.end() || it->second.index >= lib.size() )
        return std::nullopt;
      auto const& cell = lib[it->second.index];
      if ( cell.gates.empty() )
        return std::nullopt;
      auto const& gate = cell.gates.front();
      if ( gate.pins.size() != gate.function.num_vars() || nd.fanins.size() != nd.pin_names.size() )
        return std::nullopt;
      for ( auto const fi : nd.fanins )
        if ( fi >= nodes.size() )
          return std::nullopt;
      std::unordered_set<std::string> seen_pins;
      for ( auto const& pin : nd.pin_names )
      {
        auto const pname = static_cast<std::string>( pin );
        if ( it->second.pin_pos.find( pname ) == it->second.pin_pos.end() )
          return std::nullopt;
        seen_pins.insert( pname );
      }
      if ( seen_pins.size() != gate.pins.size() )
        return std::nullopt;
    }

    auto const const_false = ntk_.get_constant( false );
    std::vector<signal> signals( nodes.size(), const_false );
    for ( size_t i = 0u; i < nodes.size(); ++i )
    {
      auto const& nd = nodes[i];
      if ( nd.kind == 0u )
      {
        signals[i] = ntk_.make_signal( ntk_.index_to_node( context.leaves[nd.input_idx] ) );
        continue;
      }
      if ( nd.kind == 1u )
      {
        auto c = ntk_.get_constant( nd.const_val != 0u );
        if ( c.complement )
          c = make_inverter( const_false, context.root_index );
        signals[i] = c;
        continue;
      }

      auto const& info = cells_by_name_.at( static_cast<std::string>( nd.cell_name ) );
      auto const& cell = lib[info.index];
      auto const& gate = cell.gates.front();

      std::vector<signal> children( gate.pins.size(), const_false );
      for ( size_t k = 0u; k < nd.fanins.size(); ++k )
      {
        signal s = signals[nd.fanins[k]];
        if ( ( nd.neg_mask >> k ) & 1u )
          s = make_inverter( s, context.root_index );
        auto const pit = info.pin_pos.find( static_cast<std::string>( nd.pin_names[k] ) );
        children[pit->second] = s;
      }

      // Explicit INV nodes in Rust/native covers need the same physical-phase
      // handling as neg_mask/out_neg; bypassing an existing inverter must not
      // rebind its input (possibly a PI) to the inverter cell.
      if ( ps_.phase_reuse && children.size() == 1u && gate.function == not_tt_ )
      {
        auto out = make_inverter( children.front(), context.root_index );
        if ( nd.out_neg != 0u ) out = make_inverter( out, context.root_index );
        signals[i] = out;
        continue;
      }

      auto const f = ntk_.create_node( children, gate.function );
      note_created( f );
      ensure_level_capacity();
      auto const new_index = ntk_.node_to_index( ntk_.get_node( f ) );
      uint32_t lv = 0u;
      double arr = 0.0;
      for ( size_t p = 0u; p < children.size(); ++p )
      {
        lv = std::max( lv, level_of_signal( children[p] ) + 1u );
        arr = std::max( arr, arrival_of_signal( children[p] ) + pin_delay_of( gate, p ) );
      }
      if ( new_index < levels_.size() )
      {
        levels_[new_index] = lv;
        arrivals_[new_index] = arr;
      }
      ntk_.add_cell( ntk_.get_node( f ), info.index );

      signal out = f;
      if ( nd.out_neg != 0u )
        out = make_inverter( out, context.root_index );
      signals[i] = out;
    }

    signal root_signal = signals[cost.cell_root];
    if ( root_signal.complement )
      root_signal = make_inverter( root_signal, context.root_index );
    return root_signal;
  }

  bool try_replace( node root, cut_context const& context, CCost const& cost,
                    std::optional<uint64_t> expected_tt = std::nullopt )
  {
    auto const new_signal = build_candidate( cost, context );
    if ( !new_signal )
    {
      ++build_rejections_;
      return false;
    }
    auto const new_node = ntk_.get_node( *new_signal );
    if ( new_node == root )
      return false;
    /* A cover whose root is one of the cut's leaves (or any other node below the root) would make
       the root depend on its own descendant: reject it. */
    if ( reaches_node( new_node, root ) )
    {
      ++cycle_rejections_;
      return false;
    }
    /* `cone_truth_table` and the care mask are 64-bit, i.e. six leaves; a wider cut (reachable with
       `EPMAPPING_CUT_SIZE > 6`) is validated against its exact dynamic truth table instead of
       silently skipping the check the way `cover_matches_tt` used to. */
    bool matches = true;
    if ( ps_.validate_candidates )
      matches = ( context.leaves.size() > 6u )
                    ? cover_matches_tt_wide( cost, context, cone_truth_table_dynamic( context ) )
                    : cover_matches_tt( cost, context,
                                        expected_tt ? *expected_tt : truth_of( context ),
                                        context.care_mask );
    if ( !matches )
    {
      ++functional_rejections_;
      if ( ps_.verbose && functional_rejections_ <= 4u )
      {
        std::cerr << "[epmapping] functional_rejection expr=" << context.expression
                  << " leaves=" << context.leaves.size() << " root=" << cost.cell_root << "\n";
        for ( size_t i = 0u; i < cost.cell_nodes.size(); ++i )
        {
          auto const& nd = cost.cell_nodes[i];
          std::cerr << "  [" << i << "] kind=" << static_cast<unsigned>( nd.kind );
          if ( nd.kind == 0u )
            std::cerr << " leaf=" << static_cast<unsigned>( nd.input_idx );
          if ( nd.kind == 1u )
            std::cerr << " const=" << static_cast<unsigned>( nd.const_val );
          if ( nd.kind == 2u )
          {
            std::cerr << " cell=" << static_cast<std::string>( nd.cell_name )
                      << " neg=" << nd.neg_mask << " out_neg=" << static_cast<unsigned>( nd.out_neg )
                      << " fanins=";
            for ( auto const fi : nd.fanins )
              std::cerr << fi << ",";
            std::cerr << " pins=";
            for ( auto const& p : nd.pin_names )
              std::cerr << static_cast<std::string>( p ) << ",";
          }
          std::cerr << "\n";
        }
      }
      return false;
    }
    substitute_keep_fanouts( ntk_.node_to_index( root ), *new_signal );
    /* The fanout cone of a cut-substituted node has already been processed this round, so this walk
       stops immediately; it is kept because the invariant must hold for every substitution. */
    invalidate_cuts_fanout( ntk_.node_to_index( ntk_.get_node( *new_signal ) ) );
    return true;
  }

  /// Does the cover compute `expected` on every care minterm? `expected` is the function over
  /// `context.leaves` (LSB-first over that leaf order) — for dependency cuts it comes from the
  /// window simulation rather than from a structural cone.
  bool cover_matches_tt( CCost const& cost, cut_context const& context, uint64_t expected,
                         uint64_t care_mask ) const
  {
    auto const n = context.leaves.size();
    if ( n == 0u || n > 6u )
      return true;
    auto const minterms = uint32_t{ 1u } << n;
    auto const& nodes = cost.cell_nodes;
    if ( cost.cell_root >= nodes.size() )
      return false;
    auto const& lib = ntk_.get_library();

    std::vector<uint8_t> cover_values( nodes.size(), 0u );
    for ( uint32_t m = 0u; m < minterms; ++m )
    {
      for ( size_t i = 0u; i < nodes.size(); ++i )
      {
        auto const& nd = nodes[i];
        if ( nd.kind == 0u )
        {
          cover_values[i] = nd.input_idx < n ? static_cast<uint8_t>( ( m >> nd.input_idx ) & 1u ) : 0u;
          continue;
        }
        if ( nd.kind == 1u )
        {
          cover_values[i] = nd.const_val != 0u ? 1u : 0u;
          continue;
        }
        auto const it = cells_by_name_.find( static_cast<std::string>( nd.cell_name ) );
        if ( it == cells_by_name_.end() || it->second.index >= lib.size() )
          return false;
        auto const& gate = lib[it->second.index].gates.front();
        uint32_t function_index = 0u;
        for ( size_t k = 0u; k < nd.fanins.size(); ++k )
        {
          if ( nd.fanins[k] >= i )
            return false; /* not topological */
          auto const pit = it->second.pin_pos.find( static_cast<std::string>( nd.pin_names[k] ) );
          if ( pit == it->second.pin_pos.end() )
            return false;
          auto const value = static_cast<uint8_t>( cover_values[nd.fanins[k]] ^ ( ( nd.neg_mask >> k ) & 1u ) );
          function_index |= static_cast<uint32_t>( value ) << pit->second;
        }
        auto value = kitty::get_bit( gate.function, function_index ) ? 1u : 0u;
        if ( nd.out_neg != 0u )
          value ^= 1u;
        cover_values[i] = value;
      }
      if ( ( ( care_mask >> m ) & 1u ) == 0u )
        continue; /* don't-care minterm: any completion is acceptable */
      auto const bit = ( expected >> m ) & 1u;
      if ( ( cover_values[cost.cell_root] != 0u ? 1u : 0u ) != bit )
        return false;
    }
    return true;
  }

  /// [`cover_matches_tt`] for cuts wider than six leaves: the expected function is a dynamic truth
  /// table and **every** minterm is a care minterm (the don't-care computation does not run past six
  /// leaves), so this is an exact function comparison. Used to be skipped entirely, which left cuts
  /// of 7-9 leaves (`EPMAPPING_CUT_SIZE > 6`) without any functional validation.
  bool cover_matches_tt_wide( CCost const& cost, cut_context const& context,
                              kitty::dynamic_truth_table const& expected ) const
  {
    auto const n = static_cast<uint32_t>( context.leaves.size() );
    if ( n == 0u )
      return true;
    if ( n > MaxCutSize || expected.num_vars() != n )
      return false;
    auto const minterms = uint32_t{ 1u } << n;
    auto const& nodes = cost.cell_nodes;
    if ( cost.cell_root >= nodes.size() )
      return false;
    auto const& lib = ntk_.get_library();

    std::vector<uint8_t> cover_values( nodes.size(), 0u );
    for ( uint32_t m = 0u; m < minterms; ++m )
    {
      for ( size_t i = 0u; i < nodes.size(); ++i )
      {
        auto const& nd = nodes[i];
        if ( nd.kind == 0u )
        {
          cover_values[i] = nd.input_idx < n ? static_cast<uint8_t>( ( m >> nd.input_idx ) & 1u ) : 0u;
          continue;
        }
        if ( nd.kind == 1u )
        {
          cover_values[i] = nd.const_val != 0u ? 1u : 0u;
          continue;
        }
        auto const it = cells_by_name_.find( static_cast<std::string>( nd.cell_name ) );
        if ( it == cells_by_name_.end() || it->second.index >= lib.size() )
          return false;
        auto const& gate = lib[it->second.index].gates.front();
        uint32_t function_index = 0u;
        for ( size_t k = 0u; k < nd.fanins.size(); ++k )
        {
          if ( nd.fanins[k] >= i )
            return false; /* not topological */
          auto const pit = it->second.pin_pos.find( static_cast<std::string>( nd.pin_names[k] ) );
          if ( pit == it->second.pin_pos.end() )
            return false;
          auto const value = static_cast<uint8_t>( cover_values[nd.fanins[k]] ^ ( ( nd.neg_mask >> k ) & 1u ) );
          function_index |= static_cast<uint32_t>( value ) << pit->second;
        }
        auto value = kitty::get_bit( gate.function, function_index ) ? 1u : 0u;
        if ( nd.out_neg != 0u )
          value ^= 1u;
        cover_values[i] = value;
      }
      if ( ( cover_values[cost.cell_root] != 0u ? 1u : 0u ) !=
           static_cast<uint8_t>( kitty::get_bit( expected, m ) ) )
        return false;
    }
    return true;
  }


  /// True if `root` is reachable from `start` through fanins (`start == root` returns true).
  bool reaches_node( node start, node root ) const
  {
    std::vector<node> stack{ start };
    std::unordered_set<uint32_t> seen;
    while ( !stack.empty() )
    {
      auto const n = stack.back();
      stack.pop_back();
      if ( n == root )
        return true;
      if ( !seen.insert( ntk_.node_to_index( n ) ).second )
        continue;
      ntk_.foreach_fanin( n, [&]( auto const& f ) { stack.push_back( ntk_.get_node( f ) ); } );
    }
    return false;
  }

private:
  mutable std::vector<uint8_t> truth_scratch_;
  mutable uint64_t local_truth_simulations_{0};
  uint64_t live_revision_{0};
  mutable live_query_cache<std::optional<signal>> inverse_memo_;
  mutable uint64_t inverse_queries_{0},inverse_hits_{0},inverse_level_skips_{0},inverse_walks_{0};
  mutable uint64_t removed_builds_{0},removed_hits_{0},dynamic_tt_hits_{0};
  uint32_t odc_root_{UINT32_MAX};
  std::vector<bill::lit_type> odc_assumptions_;
  bool global_valid_{false};
  std::vector<global_signature> global_signatures_;
  std::vector<bool> global_seen_;
  std::unique_ptr<global_solver> global_solver_;
  network ntk_;
  std::optional<network> best_checkpoint_;
  std::optional<std::pair<int64_t,int64_t>> best_checkpoint_metric_;
  std::chrono::steady_clock::time_point last_checkpoint_time_;
  params ps_;
  tech_library_t const* tech_{ nullptr };
  operator_policy operator_policy_;
  std::shared_ptr<std::vector<mockturtle::standard_cell> const> mapped_library_;
  // Lifetime is one optimizer with a fixed library and objective; bounded across rounds.
  bounded_cache<std::optional<std::pair<rust::Vec<MappedNode>, uint32_t>>> synth_cache_;
  struct cached_cofactor {
    std::optional<std::pair<rust::Vec<MappedNode>, uint32_t>> mapped;
    std::string structure;
    bool valid;
    double noninverter_area;
  };
  bounded_cache<std::shared_ptr<cached_cofactor const>> cofactor_cache_;
  bounded_cache<std::shared_ptr<cached_cofactor const>> aig_mapping_cache_;
  std::vector<std::vector<uint32_t>> fanouts_;
  std::unordered_map<std::string, cell_info> cells_by_name_;
  uint32_t inv_cell_index_{ UINT32_MAX };
  /// Per-round care-mask memo (see `compute_care_mask`); `mutable` because the DC computation is
  /// `const` but its cache is a pure memoisation.
  mutable std::unordered_map<std::string, uint64_t> care_mask_cache_;
  mutable uint64_t st_care_lookups_{ 0u };
  mutable uint64_t st_care_hits_{ 0u };
  mutable uint64_t st_care_mismatches_{ 0u };
  /* Don't-care time budget (`EPMAPPING_DC_TIME_BUDGET`) and its stop counter (mutable: both are
     touched from `fresh_care_mask`, which is `const`). */
  mutable double dc_time_spent_{ 0.0 };
  mutable uint64_t dc_budget_stops_{ 0u };
  mutable double st_dc_cut_{ 0.0 };
  mutable double st_dc_collect_{ 0.0 };
  mutable double st_dc_sim_{ 0.0 };
  mutable double st_dc_window_{ 0.0 };
  mutable double st_dc_cone_{ 0.0 };
  mutable uint64_t dc_window_calls_{ 0u };
  mutable uint64_t dc_window_nodes_{ 0u };
  mutable uint64_t dc_window_nodes_max_{ 0u };
  /* Per-node don't-care window (`EPMAPPING_DC_WINDOW=per_node`): built lazily for the root being
     processed, reset every round. */
  mutable std::optional<dc_node_window> node_window_;
  mutable uint64_t st_dc_node_windows_{ 0u };
  mutable uint64_t st_dc_node_window_cuts_{ 0u };
  mutable uint64_t st_dc_node_window_no_dc_{ 0u };
  mutable uint64_t st_dc_node_window_pis_max_{ 0u };
  /* Function-keyed caches for the exact (SAT) synthesis mode: the SAT search is paid once per
     distinct (function, don't-care) pair, and failures are remembered so they are not retried. */
  mutable std::shared_ptr<mockturtle::exact_resynthesis_params::cache_map_t> exact_cache_{
      std::make_shared<mockturtle::exact_resynthesis_params::cache_map_t>() };
  mutable std::shared_ptr<mockturtle::exact_resynthesis_params::blacklist_cache_map_t> exact_blacklist_{
      std::make_shared<mockturtle::exact_resynthesis_params::blacklist_cache_map_t>() };
  /* SAT don't-care refinement counters (mutable: the refinement runs inside `compute_care_mask`,
     which is `const`). */
  mutable uint64_t st_sat_cuts_{ 0u };
  mutable uint64_t st_sat_calls_{ 0u };
  mutable uint64_t st_sat_cleared_{ 0u };
  mutable uint64_t st_sat_budget_{ 0u };
  mutable double st_dc_sat_{ 0.0 };
  /* Live cut database of the current round (`run_round`) and the nodes it has already handled. */
  std::unique_ptr<cuts_t> cuts_;
  std::unique_ptr<cut_manager_t> cut_mgr_;
  mockturtle::cut_enumeration_params cut_params_;
  mockturtle::cut_enumeration_stats cut_stats_;
  std::vector<uint8_t> processed_;
  bool cut_limit_warned_{ false };
  double inv_cell_area_{ 0.0 };
  double inv_delay_{ 0.0 };
  kitty::dynamic_truth_table not_tt_;
  std::vector<uint32_t> levels_;
  std::vector<double> arrivals_;
  mutable std::unordered_map<uint32_t, double> required_cache_;
  double area_bound_{ std::numeric_limits<double>::infinity() };
  double delay_bound_{ std::numeric_limits<double>::infinity() };
  double live_delay_target_{ std::numeric_limits<double>::infinity() };
  std::string last_failure_reason_;
  uint64_t functional_rejections_{ 0u };
  mutable uint64_t staged_depth_rejections_{ 0u };
  mutable uint64_t staged_delay_rejections_{ 0u };
  mutable uint64_t staged_objective_rejections_{ 0u };
  uint64_t build_rejections_{ 0u };
  uint64_t cycle_rejections_{ 0u };
};

} // namespace epmapping
