//! **Joint DAG-aware ILP extraction** — emapper v2, stage **S5**.
//!
//! v1 (`emap.rs`) prices every cut independently and every e-class independently. This module
//! instead solves one global covering problem over an explicit technology-match set:
//!
//! ```text
//!   min   Σ_m area(cell(m)) · y_m        (+ w · T)
//!   s.t.  Σ_{m: out(m)=c} y_m ≥ 1                     ∀ required class c
//!         y_m ≤ Σ_{m': out(m')=i} y_{m'}               ∀ input class i of match m   (support)
//!         l_out ≥ l_i + 1 − BIG·(1−y_m)                ∀ input i                    (acyclicity)
//!         a_out ≥ a_i + pin_delay(m,i) − BIGA·(1−y_m)  ∀ input i                    (timing)
//!         T ≥ a_c                                      ∀ required class c
//!         y_m ∈ {0,1},  l_c ≥ 0,  a_c ≥ 0
//! ```
//!
//! A *class* is `(subject-graph node, phase)`. The key difference from tree-cost extraction
//! (`egg`'s `CostFunction`, and EqMap's ASIC path) is that `y_m` denotes **a cell instance**, so a
//! shared sub-graph is paid for exactly once — the objective is DAG cost, not tree cost.
//!
//! Inverters are modelled as **matches of their own** (`(n,p) ← (n,¬p)`), never folded into a
//! neighbour's area. That is what makes inverter *sharing* part of the optimisation instead of an
//! accounting convention.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use coin_cbc::{Col, Model, Sense};
use petgraph::graph::NodeIndex;
use petgraph::visit::{EdgeRef, NodeIndexable, Topo};

use crate::aig_graph::{Aig, AigNode};
use crate::techlib::{CellVariant, TechLib};
use crate::tt9::{tt9_bit_get, tt9_bit_set, tt9_from_u64, tt9_not_masked, tt9_zero, Tt9};

/// Interned string shared by every match that uses the same physical pin name.
type Istr = Arc<str>;

/// One **additional** output pin of a multi-output cell instance.
///
/// The instance's [`Match::area`] is charged once and covers every output listed here.
#[derive(Clone, Debug)]
pub struct MatchOut {
    pub node: usize,
    /// `true` when this pin computes the *complement* of the node's function.
    pub neg: bool,
    /// physical output pin name
    pub pin: Istr,
    /// this pin needs its own output inverter, already included in [`Match::area`]
    pub inv: bool,
}

/// One library cell instance placed on the subject graph.
#[derive(Clone, Debug)]
pub struct Match {
    /// Dense subject-graph id this match computes (see [`Dense`]).
    pub out_node: usize,
    /// `true` when the match computes the *complement* of the node's function.
    pub out_neg: bool,
    /// the primary output pin needs its own output inverter (already in [`Match::area`])
    pub out_inv: bool,
    /// further output pins of the same instance; empty for an ordinary single-output cell
    pub extra_outs: Vec<MatchOut>,
    /// Dense input ids, aligned with the physical pin order of `pin_names`.
    pub leaves: Vec<usize>,
    /// Whether input `i` must be fed the complemented net of `leaves[i]`.
    pub leaf_neg: Vec<bool>,
    /// Pin-to-output delay for input `i` (excludes any input inverter — that is its own match).
    pub pin_delay: Vec<f32>,
    /// Area of the bare cell; input/output inverters are charged by their own matches.
    pub area: f32,
    pub cell_name: Istr,
    pub out_pin: Istr,
    pub pin_names: Vec<Istr>,
}

#[derive(Clone, Debug)]
pub struct IlpParams {
    /// Maximum number of cut leaves (`k`). Bounded by 6: the simulated truth table is one `u64`.
    pub max_cut_leaves: usize,
    /// Keep at most this many cuts per node after dedup/ranking.
    pub max_cuts_per_node: usize,
    /// Keep at most this many matches per `(node, phase)` after Pareto pruning.
    pub max_matches_per_class: usize,
    /// Weight of the worst-PO arrival in the objective; `0.0` = pure area, large = pure delay.
    pub arrival_weight: f64,
    /// Emit explicit level columns to forbid cycles. Unnecessary (and expensive) for a cut-derived
    /// match set, where every match consumes strictly earlier nodes; required once matches can point
    /// either way, as in the e-graph version.
    pub use_levels: bool,
    /// Coordinate-descent rounds in the stage-S4 cover (see [`greedy_cover`]).
    pub greedy_rounds: usize,
    /// When set, run the **timing-aware** stage-S4 cover with an arrival target of
    /// `(1 + slack) * <delay-optimal arrival>`. `None` = pure-area cover.
    pub timed_slack: Option<f32>,
    /// Per-match weight `w(m) = area(m) + delay_weight * max_pin_delay(m)` used by the stage-S4
    /// cover. Sweeping it traces an area–delay trade-off family; `0.0` is pure area.
    pub delay_weight: f64,
    /// Rounds of exact-area reverse remapping after the area-flow cover (emap's `ela_rounds`).
    pub exact_rounds: usize,
    /// Emit matches for multi-output cells: one instance covering two classes from a shared pin set.
    pub use_multioutput: bool,
    /// Cap on candidate leaf sets in the multi-output pairing search (it is quadratic in this).
    pub mo_max_sets: usize,
    /// Partner cuts kept per seed cut in the multi-output pair index.
    pub mo_max_partners: usize,
    /// Cap on generated multi-output *candidate pairs* (before cell matching).
    pub mo_max_cands: usize,
    pub time_limit_secs: f64,
    pub ratio_gap: f64,
}

impl Default for IlpParams {
    fn default() -> Self {
        Self {
            max_cut_leaves: 6,
            max_cuts_per_node: 64,
            max_matches_per_class: 24,
            arrival_weight: 0.0,
            use_levels: false,
            greedy_rounds: 12,
            timed_slack: None,
            delay_weight: 0.0,
            exact_rounds: 2,
            use_multioutput: false,
            mo_max_sets: 16384,
            mo_max_partners: 8,
            mo_max_cands: 200_000,
            time_limit_secs: 60.0,
            ratio_gap: 0.001,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct IlpStats {
    pub classes: usize,
    pub cuts: usize,
    pub matches_generated: usize,
    pub matches_kept: usize,
    pub inverters: usize,
    pub build_secs: f64,
    pub solve_secs: f64,
    pub status: String,
    pub objective: f64,
    /// Area of the stage-S4 greedy cover used as the MIP start / fallback.
    pub greedy_area: f32,
    /// `true` when CBC's answer was unusable and the greedy cover was emitted instead.
    pub used_greedy: bool,
    /// `(delay-optimal arrival, arrival target)` when the timing-aware cover ran.
    pub timed_target: Option<(f32, f32)>,
    /// Multi-output generation/selection accounting. Zeroed when the feature is off.
    pub mo: MoStats,
    /// Multi-output instances actually present in the final cover.
    pub mo_selected: usize,
}

/// A solved (or attempted) covering.
#[derive(Clone, Debug)]
pub struct IlpCover {
    pub matches: Vec<Match>,
    /// Index into `matches` for each class, or `None` when the class is not materialised.
    pub chosen: Vec<Option<usize>>,
    /// inverter cell used to materialise per-output inverters of multi-output instances
    pub inv_cell: Istr,
    pub inv_in: Istr,
    pub inv_out: Istr,
    pub area_model: f32,
    pub arrival_model: f32,
    pub depth_model: u32,
    pub stats: IlpStats,
}

#[derive(Debug)]
pub enum IlpError {
    NoInverter,
    Unreachable(String),
    Infeasible(String),
}

impl std::fmt::Display for IlpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoInverter => {
                write!(
                    f,
                    "library has no inverter; (n, neg) phases are unrealisable"
                )
            }
            Self::Unreachable(c) => write!(f, "no match realises required class {c}"),
            Self::Infeasible(s) => write!(f, "ILP infeasible or unfinished: {s}"),
        }
    }
}

impl std::error::Error for IlpError {}

// ---------------------------------------------------------------------------------------------
// dense node ids
// ---------------------------------------------------------------------------------------------

/// `StableGraph` node indices are sparse once `cleanup_graph` has removed nodes, so everything
/// downstream works on a dense `0..n` id and converts back through [`Dense::to_node`].
pub struct Dense {
    to_dense: Vec<u32>,
    to_node: Vec<NodeIndex>,
    /// Dense ids in topological order (primary outputs excluded).
    pub order: Vec<usize>,
    pub is_pi: Vec<bool>,
    pub is_zero: Vec<bool>,
}

impl Dense {
    #[must_use]
    pub fn build(aig: &Aig) -> Self {
        let bound = aig.graph().node_bound();
        let mut to_dense = vec![u32::MAX; bound];
        let mut to_node: Vec<NodeIndex> = Vec::new();
        for n in aig.graph().node_indices() {
            to_dense[n.index()] = to_node.len() as u32;
            to_node.push(n);
        }
        let mut order = Vec::new();
        let mut topo = Topo::new(aig.graph());
        while let Some(n) = topo.next(aig.graph()) {
            if !matches!(aig.node_type(n), AigNode::Output(_)) {
                order.push(to_dense[n.index()] as usize);
            }
        }
        let is_pi = to_node
            .iter()
            .map(|&n| matches!(aig.node_type(n), AigNode::Input(_)))
            .collect();
        let is_zero = to_node
            .iter()
            .map(|&n| matches!(aig.node_type(n), AigNode::Zero))
            .collect();
        Self {
            to_dense,
            to_node,
            order,
            is_pi,
            is_zero,
        }
    }

    #[inline]
    #[must_use]
    pub fn dense(&self, n: NodeIndex) -> usize {
        self.to_dense[n.index()] as usize
    }

    #[inline]
    #[must_use]
    pub fn node(&self, d: usize) -> NodeIndex {
        self.to_node[d]
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.to_node.len()
    }
}

#[inline]
const fn class_of(node: usize, neg: bool) -> usize {
    node * 2 + neg as usize
}

#[inline]
const fn class_node(c: usize) -> usize {
    c / 2
}

#[inline]
const fn class_neg(c: usize) -> bool {
    c % 2 == 1
}

// ---------------------------------------------------------------------------------------------
// cut enumeration + truth-table simulation
// ---------------------------------------------------------------------------------------------

/// Bit `j` of the pattern for leaf `i` is `(j >> i) & 1`. Only valid for `i < 6`.
fn leaf_pattern(i: usize) -> u64 {
    debug_assert!(i < 6);
    let mut p = 0u64;
    let mut j = 0usize;
    while j < 64 {
        if (j >> i) & 1 == 1 {
            p |= 1u64 << j;
        }
        j += 1;
    }
    p
}

fn mask_for(k: usize) -> u64 {
    if k >= 6 {
        u64::MAX
    } else {
        (1u64 << (1usize << k)) - 1
    }
}

/// Evaluate the function of `root` over `leaves`, bit-parallel in one `u64`.
///
/// `None` when the leaf set does not block every path from `root` to a primary input, which cannot
/// happen for cuts produced by [`enumerate_cuts`].
fn cut_tt(aig: &Aig, root: NodeIndex, leaves: &[NodeIndex], k: usize) -> Option<u64> {
    fn eval(n: NodeIndex, aig: &Aig, memo: &mut HashMap<NodeIndex, u64>) -> Option<u64> {
        if let Some(&v) = memo.get(&n) {
            return Some(v);
        }
        let (e1, e2) = aig.try_unwrap_and(n)?;
        let a = eval(aig.edge_source(e1), aig, memo)?;
        let b = eval(aig.edge_source(e2), aig, memo)?;
        let a = if aig.is_edge_inverted(e1) { !a } else { a };
        let b = if aig.is_edge_inverted(e2) { !b } else { b };
        let v = a & b;
        memo.insert(n, v);
        Some(v)
    }

    let mut memo: HashMap<NodeIndex, u64> = HashMap::new();
    for (i, l) in leaves.iter().enumerate() {
        memo.insert(*l, leaf_pattern(i));
    }
    eval(root, aig, &mut memo).map(|t| t & mask_for(k))
}

/// Classic priority-merge `k`-cut enumeration over dense ids.
///
/// Every node carries the trivial cut `{node}` so consumers can treat it as a leaf; matches are
/// *not* generated for that trivial cut (it would be a self-loop).
fn enumerate_cuts(aig: &Aig, dense: &Dense, k: usize, cap: usize) -> Vec<Vec<Vec<usize>>> {
    let mut cuts: Vec<Vec<Vec<usize>>> = vec![Vec::new(); dense.len()];
    for &d in &dense.order {
        let n = dense.node(d);
        match aig.node_type(n) {
            AigNode::Input(_) | AigNode::Zero => cuts[d] = vec![vec![d]],
            AigNode::And => {
                let (e1, e2) = aig.try_unwrap_and(n).expect("AND node has two fanins");
                let a = dense.dense(aig.edge_source(e1));
                let b = dense.dense(aig.edge_source(e2));
                let mut set: HashSet<Vec<usize>> = HashSet::new();
                set.insert(vec![d]);
                for ca in &cuts[a] {
                    for cb in &cuts[b] {
                        /* No size pre-filter here. `|A| + |B|` over-estimates the union whenever the
                        two cuts share leaves, so skipping on it discards valid cuts — two 4-leaf
                        cuts sharing two leaves still fit in k = 6, and those are exactly the deep
                        cuts a delay-oriented cover needs. */
                        let mut u: Vec<usize> = ca.iter().chain(cb.iter()).copied().collect();
                        u.sort_unstable();
                        u.dedup();
                        if u.len() <= k {
                            set.insert(u);
                        }
                    }
                }
                let mut v: Vec<Vec<usize>> = set.into_iter().collect();
                /* The cap used to keep the `cap` **smallest** cuts, on the grounds that a
                delay-oriented cover wants them. For area mapping that is backwards: it discards
                exactly the deep compound cuts that let one library cell cover several AIG nodes,
                and it does so on the high-reconvergence nodes where the cut set actually exceeds
                the cap — which is where the cover's fragmentation comes from. Stratify the quota
                across cut sizes instead, so every size is represented. */
                {
                    v.sort_by(|x, y| x.len().cmp(&y.len()).then_with(|| x.cmp(y)));
                    let per = (cap / (k + 1)).max(1);
                    let mut keep: Vec<Vec<usize>> = Vec::new();
                    let mut rest: Vec<Vec<usize>> = Vec::new();
                    let mut quota = vec![per; k + 1];
                    for u in v {
                        let s = u.len();
                        if s <= k && quota[s] > 0 {
                            quota[s] -= 1;
                            keep.push(u);
                        } else {
                            rest.push(u);
                        }
                    }
                    keep.extend(rest.into_iter().take(cap.saturating_sub(keep.len())));
                    keep.truncate(cap);
                    cuts[d] = keep;
                }
            }
            AigNode::Output(_) => unreachable!("output nodes are filtered out of `order`"),
        }
    }
    cuts
}

// ---------------------------------------------------------------------------------------------
// match generation
// ---------------------------------------------------------------------------------------------

struct Interner {
    map: HashMap<String, Istr>,
}

impl Interner {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn get(&mut self, s: &str) -> Istr {
        if let Some(v) = self.map.get(s) {
            return v.clone();
        }
        let v: Istr = Arc::from(s);
        self.map.insert(s.to_string(), v.clone());
        v
    }
}

fn variant_to_match(
    interner: &mut Interner,
    out_node: usize,
    out_neg: bool,
    leaves: &[usize],
    v: &CellVariant,
) -> Match {
    let k = v.n_inputs as usize;
    Match {
        out_node,
        out_neg,
        out_inv: false,
        extra_outs: Vec::new(),
        leaves: leaves.to_vec(),
        leaf_neg: (0..k).map(|i| (v.neg_mask >> i) & 1 == 1).collect(),
        pin_delay: v.pin_delay.clone(),
        area: v.area,
        cell_name: interner.get(&v.cell_name),
        out_pin: interner.get(&v.out_pin),
        pin_names: v.pin_names.iter().map(|p| interner.get(p)).collect(),
    }
}

/// Variants of `(cell, pin)` whose physical output realises `tt` at arity `k`, as
/// `(perm, neg_mask, out_neg, pin_delay, pin_names)`.
#[allow(clippy::type_complexity)]
fn mo_variants(
    tech: &TechLib,
    name: &str,
    pin: &str,
    k: u8,
    tt: Tt9,
) -> Vec<(Vec<u8>, u16, bool, Vec<f32>, Vec<String>)> {
    tech.candidates(k, tt)
        .map(|c| {
            c.iter()
                .filter(|v| v.cell_name == name && v.out_pin == pin)
                .map(|v| {
                    (
                        v.perm.clone(),
                        v.neg_mask,
                        v.out_neg,
                        v.pin_delay.clone(),
                        v.pin_names.clone(),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Statistics from [`multioutput_matches`]. Surfaced so that "multi-output was requested but
/// produced nothing usable" is observable instead of silently degrading to a single-output cover.
#[derive(Default, Clone, Copy, Debug)]
pub struct MoStats {
    pub cuts_indexed: usize,
    pub seeds: usize,
    pub partner_edges: usize,
    /// Pairs produced by the identical-leaf-set stage (the FA/HA pattern).
    pub identical_pairs: usize,
    /// Pairs produced by the subset stage (second output uses part of the shared pin set).
    pub subset_pairs: usize,
    pub generated: usize,
    pub hit_total_cap: bool,
    pub bucket_overflows: usize,
}

/// Matches for **multi-output** cells: one instance covering two classes from a shared pin set.
///
/// The two outputs of a multi-output cell are functions of the *same* pins, so a shared instance is
/// legal only when one leaf set is a cut for both subject-graph nodes; each output is then reduced to
/// its own essential support inside that shared set. Each output may still take its own output
/// inverter, which is charged into the instance area — it cannot be folded into the cell, and it
/// cannot be shared either.
///
/// Candidates come from a **pair-of-leaves index**: two cuts can share a pin set only if they agree
/// on at least two leaves, so every such pair collides in the bucket keyed by that leaf pair. This
/// replaces the previous "all node pairs in a bounded neighbourhood" search, which had two defects:
/// it re-derived the neighbourhood with a linear scan, and — because the seed list was a
/// *lexicographic sort of leaf sets* truncated to `max_sets` — it only ever visited the input-most
/// corner of the design. On `multiplier` that yielded 24 candidates for the whole circuit, none of
/// them ever live, so multi-output silently contributed nothing.
fn multioutput_matches(
    aig: &Aig,
    dense: &Dense,
    tech: &TechLib,
    cuts: &[Vec<Vec<usize>>],
    interner: &mut Interner,
    max_k: usize,
    max_partners: usize,
    max_total: usize,
    max_sets: usize,
) -> (Vec<Match>, MoStats) {
    let mut stats = MoStats::default();
    let mut groups: HashMap<String, (f32, Vec<String>)> = HashMap::new();
    for c in &tech.cells {
        let e = groups.entry(c.name.clone()).or_insert((c.area, Vec::new()));
        if !e.1.contains(&c.out_pin) {
            e.1.push(c.out_pin.clone());
        }
    }
    groups.retain(|_, (_, pins)| pins.len() >= 2);
    if groups.is_empty() {
        return (Vec::new(), stats);
    }

    /* ---- flatten every usable cut into an id space -------------------------------------- */
    /* A bucket is capped: the index only needs to supply a *neighbourhood*, and an unbounded bucket
    on a popular leaf pair (a primary input, say) would cost far more than it yields. */
    const BUCKET_CAP: usize = 96;
    const MAX_NODES_PER_SET: usize = 8;
    let mut cut_node: Vec<usize> = Vec::new();
    let mut cut_leaves: Vec<Vec<usize>> = Vec::new();
    let mut by_pair: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for &d in &dense.order {
        if dense.is_pi[d] || dense.is_zero[d] {
            continue;
        }
        for cut in &cuts[d] {
            if cut.len() < 2 || cut.len() > max_k {
                continue;
            }
            let id = cut_leaves.len() as u32;
            for i in 0..cut.len() {
                for j in (i + 1)..cut.len() {
                    let key = (cut[i].min(cut[j]) as u32, cut[i].max(cut[j]) as u32);
                    let b = by_pair.entry(key).or_default();
                    if b.len() >= BUCKET_CAP {
                        stats.bucket_overflows += 1;
                    } else {
                        b.push(id);
                    }
                }
            }
            cut_node.push(d);
            cut_leaves.push(cut.clone());
        }
    }
    stats.cuts_indexed = cut_leaves.len();
    if cut_leaves.is_empty() {
        return (Vec::new(), stats);
    }

    /* ---- candidate pairs, from two generators -------------------------------------------- */
    let stage_mask: u32 = std::env::var("ILP_MO_STAGES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0b111);
    let mut cands: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    let mut seen: std::collections::HashSet<(usize, usize, Vec<usize>)> =
        std::collections::HashSet::new();
    let mut push = |n1: usize,
                    n2: usize,
                    union: Vec<usize>,
                    cands: &mut Vec<(usize, usize, Vec<usize>)>,
                    seen: &mut std::collections::HashSet<(usize, usize, Vec<usize>)>|
     -> bool {
        if n1 == n2 || union.is_empty() || union.len() > max_k {
            return false;
        }
        if !seen.insert((n1, n2, union.clone())) {
            return false;
        }
        cands.push((n1, n2, union));
        true
    };

    /* Stage 1 — nodes sharing an **identical** cut leaf set. This is the FA/HA pattern: `CON` and
    `SN` of a full adder are both functions of the very same three inputs, so their cuts coincide.
    It is by far the highest-yield pairing, and the previous generator explicitly dropped it
    (`if l2 == *l1 { continue; }`), which is why multi-output never fired on `multiplier`.
    Grouping is done by sorting an index instead of hashing cloned leaf sets: `cut_leaves` holds
    over a million entries on multiplier. */
    let mut order: Vec<u32> = (0..cut_leaves.len() as u32).collect();
    order.sort_unstable_by(|&a, &b| cut_leaves[a as usize].cmp(&cut_leaves[b as usize]));
    if stage_mask & 1 != 0 {
        let mut i = 0usize;
        while i < order.len() {
            let mut j = i + 1;
            while j < order.len() && cut_leaves[order[j] as usize] == cut_leaves[order[i] as usize]
            {
                j += 1;
            }
            if j - i >= 2 {
                let set = cut_leaves[order[i] as usize].clone();
                let ids = &order[i..j];
                let lim = ids.len().min(MAX_NODES_PER_SET);
                'outer: for a in 0..lim {
                    for b in (a + 1)..lim {
                        if cands.len() >= max_total {
                            stats.hit_total_cap = true;
                            break 'outer;
                        }
                        let (n1, n2) = (cut_node[ids[a] as usize], cut_node[ids[b] as usize]);
                        if push(n1, n2, set.clone(), &mut cands, &mut seen) {
                            stats.identical_pairs += 1;
                        }
                    }
                }
            }
            i = j;
        }
    }

    /* Stage 2 — the general neighbourhood, via the pair-of-leaves index: two cuts can share a pin
    set only if they agree on at least two leaves, so every such pair collides in the bucket keyed
    by that leaf pair. Seeds are a **stride** sample of the cut id space, which follows
    `dense.order` and therefore spans inputs to outputs. The old lexicographic truncation of leaf
    sets was in effect "the first `max_sets` leaf sets", i.e. the input cone only. */
    let n = cut_leaves.len();
    let take = max_sets.min(n).max(1);
    let stride = (n / take).max(1);
    let seeds: Vec<u32> = (0..n)
        .step_by(stride)
        .take(take)
        .map(|i| i as u32)
        .collect();
    stats.seeds = seeds.len();

    for &id1 in if stage_mask & 2 != 0 {
        &seeds[..]
    } else {
        &[][..]
    } {
        if cands.len() >= max_total {
            stats.hit_total_cap = true;
            break;
        }
        let l1 = &cut_leaves[id1 as usize];
        let n1 = cut_node[id1 as usize];
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for i in 0..l1.len() {
            for j in (i + 1)..l1.len() {
                let key = (l1[i].min(l1[j]) as u32, l1[i].max(l1[j]) as u32);
                if let Some(b) = by_pair.get(&key) {
                    for &id2 in b {
                        if id2 != id1 {
                            *counts.entry(id2).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
        let mut partners: Vec<(u32, u32)> = counts
            .into_iter()
            .filter(|&(_, shared)| shared >= 2)
            .collect();
        partners.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        partners.truncate(max_partners);
        stats.partner_edges += partners.len();
        for (id2, _) in partners {
            if cands.len() >= max_total {
                stats.hit_total_cap = true;
                break;
            }
            let l2 = &cut_leaves[id2 as usize];
            let n2 = cut_node[id2 as usize];
            let mut union: Vec<usize> = l1.iter().chain(l2.iter()).copied().collect();
            union.sort_unstable();
            union.dedup();
            push(n1, n2, union, &mut cands, &mut seen);
        }
    }

    /* Stage 3 — the seed's own **subsets**. A cell's second output need not use all of the shared
    pin set, so a cut `l2 ⊊ l1` is a legitimate partner; those pairs share only `|l2|` leaves and
    are therefore pushed to the back by Stage 2's densest-overlap ordering and truncated away.
    Looking each subset up directly in the sorted index restores them. `order` is sorted by
    `cut_leaves`, so a binary search suffices and no cut->nodes hash map is needed (there are over
    a million cuts on multiplier). */
    for &id1 in if stage_mask & 4 != 0 {
        &seeds[..]
    } else {
        &[][..]
    } {
        if cands.len() >= max_total {
            stats.hit_total_cap = true;
            break;
        }
        let l1 = &cut_leaves[id1 as usize];
        let n1 = cut_node[id1 as usize];
        if l1.len() < 3 {
            continue;
        }
        for mask in 1u32..(1u32 << l1.len()) {
            let sub: Vec<usize> = (0..l1.len())
                .filter(|q| mask & (1 << q) != 0)
                .map(|q| l1[q])
                .collect();
            if sub.len() < 2 || sub.len() >= l1.len() {
                continue;
            }
            let lo = order.partition_point(|&x| cut_leaves[x as usize] < sub);
            let mut hit = lo;
            while hit < order.len() && cut_leaves[order[hit] as usize] == sub {
                let n2 = cut_node[order[hit] as usize];
                hit += 1;
                if cands.len() >= max_total {
                    stats.hit_total_cap = true;
                    break;
                }
                if push(n1, n2, l1.clone(), &mut cands, &mut seen) {
                    stats.subset_pairs += 1;
                }
            }
        }
    }

    /* ---- emit matches ------------------------------------------------------------------- */
    let mut out: Vec<Match> = Vec::new();
    for (n1, n2, union) in cands {
        if out.len() >= max_total {
            stats.hit_total_cap = true;
            break;
        }
        let unodes: Vec<NodeIndex> = union.iter().map(|&x| dense.node(x)).collect();
        let (Some(r1), Some(r2)) = (
            cut_tt(aig, dense.node(n1), &unodes, union.len()),
            cut_tt(aig, dense.node(n2), &unodes, union.len()),
        ) else {
            continue;
        };
        let t1u = tt9_from_u64(r1);
        let t2u = tt9_from_u64(r2);
        let mut sup = crate::emap::tt_support(&t1u, union.len() as u8);
        for q in crate::emap::tt_support(&t2u, union.len() as u8) {
            if !sup.contains(&q) {
                sup.push(q);
            }
        }
        sup.sort_unstable();
        if sup.is_empty() || sup.len() > max_k {
            continue;
        }
        let leaves: Vec<usize> = sup.iter().map(|&q| union[q]).collect();
        let k = leaves.len();
        let t1 = crate::emap::tt_reduce(&t1u, &sup);
        let t2 = crate::emap::tt_reduce(&t2u, &sup);

        for (name, (carea, pins)) in &groups {
            let (pa, pb) = (&pins[0], &pins[1]);
            for phase1 in [false, true] {
                for phase2 in [false, true] {
                    let ta = if phase1 {
                        tt9_not_masked(&t1, k as u8)
                    } else {
                        t1
                    };
                    let tb = if phase2 {
                        tt9_not_masked(&t2, k as u8)
                    } else {
                        t2
                    };
                    let ka = mo_variants(tech, name, pa, k as u8, ta);
                    if ka.is_empty() {
                        continue;
                    }
                    let kb = mo_variants(tech, name, pb, k as u8, tb);
                    if kb.is_empty() {
                        continue;
                    }
                    for (perm, mask, inva, pda, pna) in &ka {
                        for (perm2, mask2, invb, _, _) in &kb {
                            if perm != perm2 || mask != mask2 {
                                continue;
                            }
                            let area = carea
                                + if *inva { tech.inv_area } else { 0.0 }
                                + if *invb { tech.inv_area } else { 0.0 };
                            out.push(Match {
                                out_node: n1,
                                out_neg: phase1,
                                out_inv: *inva,
                                extra_outs: vec![MatchOut {
                                    node: n2,
                                    neg: phase2,
                                    pin: interner.get(pb),
                                    inv: *invb,
                                }],
                                leaves: leaves.clone(),
                                leaf_neg: (0..k).map(|q| (mask >> q) & 1 == 1).collect(),
                                pin_delay: pda.clone(),
                                area,
                                cell_name: interner.get(name),
                                out_pin: interner.get(pa),
                                pin_names: pna.iter().map(|x| interner.get(x)).collect(),
                            });
                        }
                    }
                }
            }
        }
    }
    stats.generated = out.len();
    if std::env::var_os("EMAPPER_DEBUG_MO").is_some() {
        let mut by_cell: HashMap<&str, usize> = HashMap::new();
        for m in &out {
            *by_cell.entry(&m.cell_name).or_insert(0) += 1;
        }
        let mut v: Vec<_> = by_cell.into_iter().collect();
        v.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        let mut by_k: HashMap<usize, usize> = HashMap::new();
        for m in &out {
            *by_k.entry(m.leaves.len()).or_insert(0) += 1;
        }
        let mut kv: Vec<_> = by_k.into_iter().collect();
        kv.sort();
        eprintln!("mo-gen: pool={} by_cell={:?} by_k={:?}", out.len(), v, kv);
    }
    (out, stats)
}

/// Heuristic per-class arrival: one topological sweep over the candidate matches. Used only to rank
/// matches before the ILP — stage S4 replaces this with the full flow/ref-count pass.
fn heuristic_arrivals(
    dense: &Dense,
    matches: &[Match],
    is_free: &[bool],
    inv_delay: f32,
    n_classes: usize,
) -> Vec<f32> {
    let mut by_out: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, m) in matches.iter().enumerate() {
        by_out[class_of(m.out_node, m.out_neg)].push(i);
        for o in &m.extra_outs {
            by_out[class_of(o.node, o.neg)].push(i);
        }
    }
    let mut arr = vec![f32::INFINITY; n_classes];
    for &d in &dense.order {
        if dense.is_pi[d] || dense.is_zero[d] {
            arr[class_of(d, false)] = 0.0;
            arr[class_of(d, true)] = inv_delay;
        }
    }
    for &d in &dense.order {
        for phase in [false, true] {
            let cls = class_of(d, phase);
            if is_free[cls] {
                continue;
            }
            let mut best = f32::INFINITY;
            for &mi in &by_out[cls] {
                let m = &matches[mi];
                let mut a = 0.0f32;
                for (i, &leaf) in m.leaves.iter().enumerate() {
                    let li = class_of(leaf, m.leaf_neg[i]);
                    if li == cls {
                        a = f32::INFINITY;
                        break;
                    }
                    let la = if is_free[li] { 0.0 } else { arr[li] };
                    a = a.max(la + m.pin_delay[i]);
                }
                if a < best {
                    best = a;
                }
            }
            arr[cls] = best;
        }
    }
    arr
}

/// Keep, per `(node, phase)`, only matches that are Pareto-optimal in `(area, predicted arrival)`;
/// within one wiring signature only the Pareto set survives.
fn prune_matches(
    matches: Vec<Match>,
    pred: &[f32],
    n_classes: usize,
    cap: usize,
    area_only: bool,
) -> (Vec<Match>, Vec<f32>) {
    /* A multi-output instance must be kept or dropped as a unit: its two outputs share one area and
    one pin set, so running the per-class filter over them independently would leave a class whose
    only realisation is an instance that was pruned away under the other output. They therefore
    bypass the filter entirely. */
    let (multi, single): (Vec<usize>, Vec<usize>) =
        (0..matches.len()).partition(|&i| !matches[i].extra_outs.is_empty());

    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for &i in &single {
        let m = &matches[i];
        groups[class_of(m.out_node, m.out_neg)].push(i);
    }

    let order_by = |a: &usize, b: &usize| {
        matches[*a]
            .area
            .partial_cmp(&matches[*b].area)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                pred[*a]
                    .partial_cmp(&pred[*b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    };

    let mut keep: Vec<usize> = Vec::new();
    for g in &groups {
        if g.is_empty() {
            continue;
        }
        let mut by_sig: BTreeMap<(Vec<usize>, Vec<bool>), Vec<usize>> = BTreeMap::new();
        for &i in g {
            let m = &matches[i];
            by_sig
                .entry((m.leaves.clone(), m.leaf_neg.clone()))
                .or_default()
                .push(i);
        }
        let mut candidates: Vec<usize> = Vec::new();
        /* Within one leaf signature every variant realises the same function from the same leaves, so
        keeping the whole (area, arrival) Pareto front mainly gives the flow more ways to make a
        bad early choice. `ILP_MATCHES_LEAN` keeps only the cheapest one. */
        let lean = std::env::var_os("ILP_MATCHES_LEAN").is_some();
        for (_, members) in by_sig {
            let mut sorted = members;
            sorted.sort_by(order_by);
            if lean {
                if let Some(&i) = sorted.first() {
                    candidates.push(i);
                }
                continue;
            }
            let mut best_arr = f32::INFINITY;
            for i in sorted {
                if pred[i] < best_arr {
                    best_arr = pred[i];
                    candidates.push(i);
                }
            }
        }
        candidates.sort_by(order_by);
        let mut picked: Vec<usize> = Vec::new();
        if area_only {
            /* On a pure-area run the arrival dimension of the Pareto test is not a cost the cover
            is being judged on, and dropping a late-but-cheap realisation can remove the only match
            that makes a good DAG cover possible (the filter's own comment notes that tree-cost
            dominance is unsafe for a DAG). Keep the best match of every leaf signature and skip
            the cross-signature dominance test; the cap below still bounds the result. */
            picked = candidates;
        } else {
            let mut best_arr = f32::INFINITY;
            for i in candidates {
                if pred[i] < best_arr {
                    best_arr = pred[i];
                    picked.push(i);
                }
            }
        }
        if picked.len() > cap && cap >= 2 {
            let n = picked.len();
            let mut thinned: Vec<usize> =
                (0..cap).map(|j| picked[j * (n - 1) / (cap - 1)]).collect();
            thinned.dedup();
            picked = thinned;
        }
        keep.extend(picked);
    }

    /* Per-class Pareto pruning is **not cone-aware**: it can leave a class whose only surviving
    matches all depend on a deeper class that was itself pruned away, and then the greedy cover
    cannot be completed at all (observed on `multiplier`, `log2` and `sin` with the default cap of
    24 — the run ends in "neither CBC nor the greedy pass produced a complete cover"). Restore, for
    any class that is not realisable, its cheapest discarded match whose leaves are all realisable,
    and repeat to a fixpoint so the repair propagates from the primary inputs upward. */
    let mut kept = vec![false; matches.len()];
    for &i in &keep {
        kept[i] = true;
    }
    let mut multi_of: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    let mut has_match = vec![false; n_classes];
    for &i in &multi {
        let m = &matches[i];
        let c = class_of(m.out_node, m.out_neg);
        multi_of[c].push(i);
        has_match[c] = true;
        for o in &m.extra_outs {
            let oc = class_of(o.node, o.neg);
            multi_of[oc].push(i);
            has_match[oc] = true;
        }
    }
    for &i in &single {
        has_match[class_of(matches[i].out_node, matches[i].out_neg)] = true;
    }
    let leaves_ok = |i: usize, real: &[bool]| -> bool {
        let m = &matches[i];
        m.leaves
            .iter()
            .enumerate()
            .all(|(p, &l)| real[class_of(l, m.leaf_neg[p])])
    };
    // A class with no match at all is a primary input or a constant: realisable by definition.
    let mut real: Vec<bool> = (0..n_classes).map(|c| !has_match[c]).collect();
    loop {
        let mut changed = false;
        for c in 0..n_classes {
            if real[c] {
                continue;
            }
            let already = groups[c].iter().any(|&i| kept[i] && leaves_ok(i, &real))
                || multi_of[c].iter().any(|&i| leaves_ok(i, &real));
            if already {
                real[c] = true;
                changed = true;
                continue;
            }
            if let Some(&i) = groups[c]
                .iter()
                .filter(|&&i| !kept[i] && leaves_ok(i, &real))
                .min_by(|&&a, &&b| {
                    matches[a]
                        .area
                        .partial_cmp(&matches[b].area)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                kept[i] = true;
                keep.push(i);
                real[c] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    keep.sort_unstable();
    keep.dedup();
    keep.extend(multi.iter().copied());
    let mut out = Vec::with_capacity(keep.len());
    let mut out_pred = Vec::with_capacity(keep.len());
    for &i in &keep {
        out.push(matches[i].clone());
        out_pred.push(pred[i]);
    }
    (out, out_pred)
}

// ---------------------------------------------------------------------------------------------
// solve
// ---------------------------------------------------------------------------------------------

/// Solve the joint covering problem for the primary outputs of `aig`.
pub fn solve(
    aig: &Aig,
    tech: &TechLib,
    required: &[(usize, bool)],
    ps: &IlpParams,
) -> Result<IlpCover, IlpError> {
    let t_build = Instant::now();
    let dense = Dense::build(aig);
    let n_classes = dense.len() * 2;

    let inv_area = tech.inv_area;
    let inv_delay = tech.inv_delay;
    let Some(inv) = tech.inv_variant.clone() else {
        return Err(IlpError::NoInverter);
    };

    // ---- free classes: primary inputs (positive) and the constant node (both phases) ----------
    let mut is_free = vec![false; n_classes];
    for d in 0..dense.len() {
        if dense.is_pi[d] {
            is_free[class_of(d, false)] = true;
        }
        if dense.is_zero[d] {
            is_free[class_of(d, false)] = true;
            is_free[class_of(d, true)] = true;
        }
    }

    // ---- cuts + matches ----------------------------------------------------------------------
    let cuts = enumerate_cuts(aig, &dense, ps.max_cut_leaves, ps.max_cuts_per_node);
    if std::env::var_os("EMAPPER_DEBUG_CUTS").is_some() {
        let mut hist = vec![0usize; ps.max_cut_leaves + 2];
        let (mut at_cap, mut nodes, mut maxc) = (0usize, 0usize, 0usize);
        for c in &cuts {
            if c.is_empty() {
                continue;
            }
            nodes += 1;
            maxc = maxc.max(c.len());
            if c.len() >= ps.max_cuts_per_node {
                at_cap += 1;
            }
            for x in c {
                hist[x.len().min(ps.max_cut_leaves + 1)] += 1;
            }
        }
        eprintln!(
            "cuts: nodes={nodes} total={} at_cap={at_cap} max_per_node={maxc} size_hist={:?}",
            hist.iter().sum::<usize>(),
            hist
        );
    }
    let mut interner = Interner::new();
    let mut matches: Vec<Match> = Vec::new();
    let mut n_cuts = 0usize;

    for &d in &dense.order {
        if dense.is_pi[d] || dense.is_zero[d] {
            continue;
        }
        for cut in &cuts[d] {
            if cut.len() == 1 && cut[0] == d {
                continue; // trivial cut == self-loop
            }
            if cut.is_empty() || cut.len() > ps.max_cut_leaves {
                continue;
            }
            n_cuts += 1;
            let k = cut.len();
            let nodes: Vec<NodeIndex> = cut.iter().map(|&x| dense.node(x)).collect();
            let Some(raw) = cut_tt(aig, dense.node(d), &nodes, k) else {
                continue;
            };
            let tt = tt9_from_u64(raw);
            /* Reduce to the **essential support** before matching, exactly as mockturtle's
            `cut_enumeration_ps.minimize_truth_table` does for emap. A deep cut often carries
            leaves that do not influence its function (reconvergent structure); without this the
            lookup insists on a cell with the cut's arity and finds nothing, so every such cut —
            and every wide cell it could have reached — is lost. */
            let support = crate::emap::tt_support(&tt, k as u8);
            let (kk, ttr, leaf_map): (u8, Tt9, Vec<usize>) = if support.len() < k {
                let rl: Vec<usize> = support.iter().map(|&i| cut[i]).collect();
                (
                    support.len() as u8,
                    crate::emap::tt_reduce(&tt, &support),
                    rl,
                )
            } else {
                (k as u8, tt, cut.clone())
            };
            for out_neg in [false, true] {
                let target = if out_neg {
                    tt9_not_masked(&ttr, kk)
                } else {
                    ttr
                };
                let Some(cands) = tech.candidates(kk, target) else {
                    continue;
                };
                for v in cands {
                    // `out_neg == true` variants are exactly "cell + output inverter"; the explicit
                    // inverter match below covers that and can be *shared*. Skipping them here is
                    // lossless and keeps the model smaller.
                    if v.out_neg {
                        continue;
                    }
                    matches.push(variant_to_match(&mut interner, d, out_neg, &leaf_map, v));
                }
            }
        }
    }

    // ---- multi-output instances --------------------------------------------------------------
    let mut mo_stats = MoStats::default();
    if ps.use_multioutput {
        let (mo, st) = multioutput_matches(
            aig,
            &dense,
            tech,
            &cuts,
            &mut interner,
            ps.max_cut_leaves,
            ps.mo_max_partners,
            ps.mo_max_cands,
            ps.mo_max_sets,
        );
        mo_stats = st;
        matches.extend(mo);
    }

    // ---- explicit inverter matches -----------------------------------------------------------
    let mut n_inv = 0usize;
    for &d in &dense.order {
        if dense.is_zero[d] {
            continue;
        }
        for (from, to) in [(false, true), (true, false)] {
            if is_free[class_of(d, to)] {
                continue;
            }
            n_inv += 1;
            matches.push(Match {
                out_node: d,
                out_neg: to,
                out_inv: false,
                extra_outs: Vec::new(),
                leaves: vec![d],
                leaf_neg: vec![from],
                pin_delay: vec![inv_delay],
                area: inv_area,
                cell_name: interner.get(&inv.cell_name),
                out_pin: interner.get(&inv.out_pin),
                pin_names: vec![interner.get(inv.pin_names.first().map_or("A", String::as_str))],
            });
        }
    }

    let generated = matches.len();
    let class_arr = heuristic_arrivals(&dense, &matches, &is_free, inv_delay, n_classes);
    // Per-match predicted arrival, used only for ranking (classes are not match indices).
    let pred: Vec<f32> = matches
        .iter()
        .map(|m| {
            let mut a = 0.0f32;
            for (i, &leaf) in m.leaves.iter().enumerate() {
                let li = class_of(leaf, m.leaf_neg[i]);
                if li == class_of(m.out_node, m.out_neg) {
                    return f32::INFINITY;
                }
                let la = if is_free[li] { 0.0 } else { class_arr[li] };
                a = a.max(la + m.pin_delay[i]);
            }
            a
        })
        .collect();
    /* `max_matches_per_class == 0` disables the Pareto filter entirely. That filter is a tree-cost
    dominance test, and tree-cost dominance is NOT safe for a DAG cover: a match that is worse in
    both area and arrival as a tree can still be the right choice once its inputs are shared. The
    filter exists to make the MIP tractable, so it has to be measured, not assumed. */
    let (matches, _) = if ps.max_matches_per_class == 0 {
        (matches, pred)
    } else {
        let area_only = ps.timed_slack.is_none() && ps.delay_weight == 0.0;
        prune_matches(
            matches,
            &pred,
            n_classes,
            ps.max_matches_per_class,
            area_only,
        )
    };

    /* Every edge of a cut match goes from a node to one of its transitive fan-in nodes, and an
    inverter match keeps the node. So every edge weakly decreases the topological rank, and the
    only cycle the cut space can produce is the two-cycle between the two inverter matches of a
    single node — which pruning *can* create, by discarding every cut match of a phase and
    leaving its inverter as the sole producer. Forbid that pair explicitly (the general level
    constraints stay available behind `use_levels` for the e-graph match set). */
    let mut inv_by_node: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, m) in matches.iter().enumerate() {
        if m.leaves.len() == 1 && m.leaves[0] == m.out_node {
            inv_by_node.entry(m.out_node).or_default().push(i);
        }
    }
    let inverter_pairs: Vec<(usize, usize)> = inv_by_node
        .into_values()
        .filter_map(|v| {
            let mut s = v;
            s.sort_unstable();
            if s.len() >= 2 {
                Some((s[0], s[1]))
            } else {
                None
            }
        })
        .collect();

    for &(node, neg) in required {
        if is_free[class_of(node, neg)] {
            continue;
        }
        if !matches
            .iter()
            .any(|m| m.out_node == node && m.out_neg == neg)
        {
            return Err(IlpError::Unreachable(format!("({node},{neg})")));
        }
    }

    // ---- model -------------------------------------------------------------------------------
    let mut model = Model::default();
    model.set_parameter("log", "0");
    model.set_parameter(
        "seconds",
        &format!("{}", ps.time_limit_secs.max(1.0) as u64),
    );
    if ps.ratio_gap > 0.0 {
        model.set_parameter("ratioGap", &format!("{}", ps.ratio_gap));
    }
    model.set_obj_sense(Sense::Minimize);

    let big_level = (dense.len() + 2) as f64;
    /* Keep the timing BIG-M numerically close to the objective. A `1e6` BIG-M next to `area ≈ 0.04`
    objective coefficients wrecked CBC's tolerances: it burned the whole budget without closing
    the gap. `max_pin_delay * levels` is a valid upper bound on any arrival. */
    let max_pin_delay = matches
        .iter()
        .flat_map(|m| m.pin_delay.iter().copied())
        .fold(0.0f32, f32::max);
    let big_arr = f64::from(max_pin_delay) * big_level + f64::from(inv_delay) + 1.0;

    /* A cut match only consumes nodes strictly inside its own transitive fan-in, and an inverter
    match keeps the node; a cover built from those can never contain a cycle. The level rows are
    therefore pure overhead here — they stay behind a flag for the e-graph match set, where
    matches may point either way and cycles are real. */
    let use_levels = ps.use_levels;
    let use_timing = ps.arrival_weight > 0.0;

    let y: Vec<Col> = matches
        .iter()
        .map(|m| {
            let c = model.add_binary();
            model.set_obj_coeff(c, f64::from(m.area));
            c
        })
        .collect();

    let lvl: Vec<Col> = if use_levels {
        (0..n_classes)
            .map(|_| {
                let c = model.add_col();
                model.set_col_lower(c, 0.0);
                model.set_col_upper(c, big_level);
                c
            })
            .collect()
    } else {
        Vec::new()
    };
    let arr: Vec<Col> = if use_timing {
        (0..n_classes)
            .map(|_| {
                let c = model.add_col();
                model.set_col_lower(c, 0.0);
                model.set_col_upper(c, big_arr);
                c
            })
            .collect()
    } else {
        Vec::new()
    };
    if use_levels || use_timing {
        for c in 0..n_classes {
            if is_free[c] {
                if use_timing {
                    model.set_col_upper(arr[c], 0.0);
                }
                if use_levels {
                    model.set_col_upper(lvl[c], 0.0);
                }
            }
        }
    }
    let tmax: Option<Col> = if use_timing {
        let c = model.add_col();
        model.set_col_lower(c, 0.0);
        model.set_col_upper(c, big_arr);
        if ps.arrival_weight > 0.0 {
            model.set_obj_coeff(c, ps.arrival_weight);
        }
        Some(c)
    } else {
        None
    };

    let mut producers: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, m) in matches.iter().enumerate() {
        producers[class_of(m.out_node, m.out_neg)].push(i);
        for o in &m.extra_outs {
            producers[class_of(o.node, o.neg)].push(i);
        }
    }

    for (i, m) in matches.iter().enumerate() {
        let oc = class_of(m.out_node, m.out_neg);
        for (p, &leaf) in m.leaves.iter().enumerate() {
            let ic = class_of(leaf, m.leaf_neg[p]);
            if ic == oc {
                let row = model.add_row();
                model.set_row_equal(row, 0.0);
                model.set_weight(row, y[i], 1.0);
                continue;
            }
            // support: y_m ≤ Σ producers(ic). A free input (primary input / constant) always has a
            // driver, so it must NOT get this row — with an empty producer set the row would read
            // `y_m ≤ 0` and forbid every match that touches a PI.
            if !is_free[ic] {
                let row = model.add_row();
                model.set_row_upper(row, 0.0);
                model.set_weight(row, y[i], 1.0);
                for &j in &producers[ic] {
                    model.set_weight(row, y[j], -1.0);
                }
                if producers[ic].is_empty() {
                    model.set_row_upper(row, -1.0); // nothing can drive this input
                }
            }

            if use_levels {
                // acyclicity: l_out - l_ic - BIG·y_m ≥ 1 - BIG
                let row = model.add_row();
                model.set_row_lower(row, 1.0 - big_level);
                model.set_weight(row, lvl[oc], 1.0);
                model.set_weight(row, lvl[ic], -1.0);
                model.set_weight(row, y[i], -big_level);
            }

            if use_timing {
                /* Every output pin of the instance gets its own arrival constraint; an inverted
                output adds the inverter delay. */
                let mut outs: Vec<(usize, f32)> =
                    vec![(oc, if m.out_inv { inv_delay } else { 0.0 })];
                for o in &m.extra_outs {
                    outs.push((class_of(o.node, o.neg), if o.inv { inv_delay } else { 0.0 }));
                }
                for (oc2, extra) in outs {
                    let row = model.add_row();
                    model.set_row_lower(row, f64::from(m.pin_delay[p] + extra) - big_arr);
                    model.set_weight(row, arr[oc2], 1.0);
                    model.set_weight(row, arr[ic], -1.0);
                    model.set_weight(row, y[i], -big_arr);
                }
            }
        }
    }

    for &(node, neg) in required {
        let c = class_of(node, neg);
        if is_free[c] {
            continue;
        }
        let row = model.add_row();
        model.set_row_lower(row, 1.0);
        for &j in &producers[c] {
            model.set_weight(row, y[j], 1.0);
        }
    }

    // at most one of the two inverter directions of a node (see `inverter_pairs`)
    for &(a, b) in &inverter_pairs {
        let row = model.add_row();
        model.set_row_upper(row, 1.0);
        model.set_weight(row, y[a], 1.0);
        model.set_weight(row, y[b], 1.0);
    }

    /* At most one realisation per class. Cheap, and it is what gives the LP relaxation any meaning:
    without it the relaxation can "share" one fractional cell across arbitrarily many parents and
    the lower bound collapses. */
    for (c, prod) in producers.iter().enumerate() {
        if is_free[c] || prod.is_empty() {
            continue;
        }
        let row = model.add_row();
        model.set_row_upper(row, 1.0);
        for &j in prod {
            model.set_weight(row, y[j], 1.0);
        }
    }

    if let Some(t) = tmax {
        for &(node, neg) in required {
            let c = class_of(node, neg);
            if is_free[c] {
                continue;
            }
            let row = model.add_row();
            model.set_row_lower(row, 0.0);
            model.set_weight(row, t, 1.0);
            model.set_weight(row, arr[c], -1.0);
        }
    }

    /* Stage S4 supplies the MIP start. Without an incumbent, CBC on this (deliberately weak)
    covering formulation can burn the whole budget without ever finding an integral point. */
    /* When an arrival target is given, stage S4 runs the timing-aware variant; the sweep over
    `timed_slack` is what traces the area–delay Pareto front. */
    /* `w(m) = area(m) + delay_weight * max_pin_delay(m)`. Sweeping `delay_weight` walks an
    area–delay trade-off family through the same sharing-aware coordinate descent. */
    let w: Vec<f64> = matches
        .iter()
        .map(|m| {
            let d = m.pin_delay.iter().copied().fold(0.0f32, f32::max);
            f64::from(m.area) + ps.delay_weight * f64::from(d)
        })
        .collect();

    /* Estimated number of consumers per class, used to amortise the area flow. mockturtle emap
    seeds this with the subject graph's fanout count and refines it with the realised reference
    counts each iteration; here the subject-graph fanout is used throughout. */
    let mut est_refs: Vec<f32> = (0..n_classes)
        .map(|c| {
            let d = class_node(c);
            let f = aig.output_degree(dense.node(d));
            (f.max(1)) as f32
        })
        .collect();

    let mut timed_target = None;
    let greedy = match ps.timed_slack {
        Some(slack) => {
            let arr_min = arrival_flow(&dense, &matches, &producers, &is_free, n_classes);
            let t_min = required
                .iter()
                .map(|&(n, g)| {
                    let c = class_of(n, g);
                    if is_free[c] {
                        0.0
                    } else {
                        arr_min[c]
                    }
                })
                .fold(0.0f32, f32::max);
            let t = t_min * (1.0 + slack.max(0.0));
            timed_target = Some((t_min, t));
            timed_cover(
                &dense,
                &matches,
                &producers,
                required,
                &is_free,
                n_classes,
                t,
                ps.greedy_rounds,
                &w,
                &est_refs,
                inv_delay,
            )
        }
        None => greedy_cover(
            &dense,
            &matches,
            required,
            &is_free,
            n_classes,
            ps.greedy_rounds,
            &w,
            ps.exact_rounds,
            &mut est_refs,
        ),
    };
    let greedy_area: f32 = greedy.iter().map(|&i| matches[i].area).sum();
    /* `time_limit_secs <= 0.0` runs the stage-S4 cover only. CBC's `seconds` parameter does not
    reliably interrupt the LP phase, so a Pareto sweep would otherwise pay a full MIP timeout per
    point. */
    let skip_cbc = ps.time_limit_secs <= 0.0;
    if !skip_cbc {
        for &i in &greedy {
            model.set_col_initial_solution(y[i], 1.0);
        }
    }

    let build_secs = t_build.elapsed().as_secs_f64();
    let t_solve = Instant::now();
    let sol = if skip_cbc { None } else { Some(model.solve()) };
    let solve_secs = t_solve.elapsed().as_secs_f64();
    let status = sol.as_ref().map_or_else(
        || "greedy-only".to_string(),
        |s| format!("{:?}", s.raw().status()),
    );

    /* CBC only produced a usable answer if the binary block is integral: on `Stopped` without an
    incumbent it hands back the LP relaxation, which is not a cover and would emit combinational
    loops. Otherwise fall back to the stage-S4 greedy cover. */
    let integral = sol.as_ref().is_some_and(|s| {
        matches.iter().enumerate().all(|(i, _)| {
            let v = s.col(y[i]);
            (v - 0.0).abs() <= 1e-6 || (v - 1.0).abs() <= 1e-6
        })
    });

    let mut chosen: Vec<Option<usize>> = vec![None; n_classes];
    let mut area_model = 0.0f32;
    if integral && (status == "Finished" || status == "Stopped") {
        let s = sol.as_ref().expect("integral implies a solution");
        for (i, m) in matches.iter().enumerate() {
            if s.col(y[i]) > 0.5 {
                chosen[class_of(m.out_node, m.out_neg)] = Some(i);
                for o in &m.extra_outs {
                    chosen[class_of(o.node, o.neg)] = Some(i);
                }
                area_model += m.area;
            }
        }
    }
    let mut used_greedy = false;
    if !cover_complete(&matches, &chosen, required, &is_free) {
        used_greedy = true;
        chosen = vec![None; n_classes];
        area_model = 0.0;
        for &i in &greedy {
            let m = &matches[i];
            chosen[class_of(m.out_node, m.out_neg)] = Some(i);
            for o in &m.extra_outs {
                chosen[class_of(o.node, o.neg)] = Some(i);
            }
            area_model += m.area;
        }
    }
    if !cover_complete(&matches, &chosen, required, &is_free) {
        /* Name the first class reachable from a primary output that has no realisation, plus how
        many matches that class ever had — an empty `by_out` entry means the cut enumeration
        missed it, a non-empty one means every candidate was filtered away. */
        let missing = {
            let mut seen = vec![false; n_classes];
            let mut stack: Vec<usize> = required
                .iter()
                .map(|&(n, g)| class_of(n, g))
                .filter(|&c| !is_free[c])
                .collect();
            let mut found = None;
            while let Some(c) = stack.pop() {
                if seen[c] {
                    continue;
                }
                seen[c] = true;
                let Some(mi) = chosen[c] else {
                    found = Some(c);
                    break;
                };
                let m = &matches[mi];
                for (p2, &l) in m.leaves.iter().enumerate() {
                    let ic = class_of(l, m.leaf_neg[p2]);
                    if !is_free[ic] {
                        stack.push(ic);
                    }
                }
            }
            found
        };
        let detail = match missing {
            Some(c) => {
                let nm = matches
                    .iter()
                    .filter(|m| {
                        class_of(m.out_node, m.out_neg) == c
                            || m.extra_outs.iter().any(|o| class_of(o.node, o.neg) == c)
                    })
                    .count();
                format!(
                    "first unrealised class {c}, matches_for_it={nm}, node={}, neg={} ",
                    c / 2,
                    c % 2 == 1
                )
            }
            None => "no unreachable class found ".to_string(),
        };
        return Err(IlpError::Infeasible(format!(
            "status={status}: neither CBC nor the greedy pass produced a complete cover ({detail})"
        )));
    }
    if let Some(c) = first_cycle(&matches, &chosen) {
        return Err(IlpError::Infeasible(format!(
            "status={status}: chosen cover contains a cycle through class {c}"
        )));
    }

    let (arr_post, lvl_post) =
        evaluate_cover(&dense, &matches, &chosen, &is_free, n_classes, inv_delay);
    let mut arrival_model = 0.0f32;
    let mut depth_model = 0u32;
    for &(node, neg) in required {
        let c = class_of(node, neg);
        arrival_model = arrival_model.max(arr_post[c]);
        depth_model = depth_model.max(lvl_post[c]);
    }

    Ok(IlpCover {
        stats: IlpStats {
            classes: n_classes,
            cuts: n_cuts,
            matches_generated: generated,
            matches_kept: matches.len(),
            inverters: n_inv,
            build_secs,
            solve_secs,
            status,
            objective: sol
                .as_ref()
                .map_or(f64::from(greedy_area), |s| s.raw().obj_value()),
            greedy_area,
            used_greedy,
            timed_target,
            mo: mo_stats,
            mo_selected: chosen
                .iter()
                .flatten()
                .filter(|&&i| !matches[i].extra_outs.is_empty())
                .collect::<std::collections::HashSet<_>>()
                .len(),
        },
        matches,
        chosen,
        inv_cell: Istr::from(tech.inv_cell.as_deref().unwrap_or("INV")),
        inv_in: Istr::from(tech.inv_in_pin.as_deref().unwrap_or("A")),
        inv_out: Istr::from(tech.inv_out_pin.as_deref().unwrap_or("Y")),
        area_model,
        arrival_model,
        depth_model,
    })
}

/// Number of consumers each class has in a chosen cover (primary outputs count as consumers).
fn cover_refs(
    matches: &[Match],
    pick: &[Option<usize>],
    required: &[(usize, bool)],
    is_free: &[bool],
    n_classes: usize,
) -> Vec<f32> {
    let mut r = vec![0.0f32; n_classes];
    for &(n, g) in required {
        let c = class_of(n, g);
        if !is_free[c] {
            r[c] += 1.0;
        }
    }
    let mut seen = std::collections::HashSet::new();
    for c in 0..n_classes {
        let Some(mi) = pick[c] else { continue };
        if !seen.insert(mi) {
            continue; // a shared instance has one set of inputs regardless of its output count
        }
        for (p, &leaf) in matches[mi].leaves.iter().enumerate() {
            let ic = class_of(leaf, matches[mi].leaf_neg[p]);
            if !is_free[ic] {
                r[ic] += 1.0;
            }
        }
    }
    r
}

/// Marginal cost of making each class available, given the current selection: a class that is already
/// realised costs nothing, everything else costs the cheapest way to build it.
///
/// Passing an all-`None` selection degenerates to the classic tree-cost **area flow**.
fn marginal_cost(
    dense: &Dense,
    matches: &[Match],
    by_out: &[Vec<usize>],
    pick: &[Option<usize>],
    is_free: &[bool],
    n_classes: usize,
    w: &[f64],
    est_refs: &[f32],
    share: bool,
) -> Vec<f32> {
    let is_inv = |m: &Match| m.leaves.first() == Some(&m.out_node);
    let mut marg = vec![f32::INFINITY; n_classes];
    for c in 0..n_classes {
        if is_free[c] || pick[c].is_some() {
            marg[c] = 0.0;
        }
    }
    for &d in &dense.order {
        /* 1) Non-inverter realisations of both phases. A cut match only reads strictly earlier
        nodes, so both are final once this node is reached. */
        for phase in [false, true] {
            let c = class_of(d, phase);
            if marg[c] == 0.0 {
                continue;
            }
            let mut best = f32::INFINITY;
            let mut best_size = u32::MAX;
            for &mi in &by_out[c] {
                let m = &matches[mi];
                if is_inv(m) {
                    continue;
                }
                let mut s = w[mi];
                let mut ok = true;
                for (p, &leaf) in m.leaves.iter().enumerate() {
                    let ic = class_of(leaf, m.leaf_neg[p]);
                    if marg[ic].is_infinite() {
                        ok = false;
                        break;
                    }
                    s += f64::from(marg[ic]);
                }
                /* emap's area-mode comparison is area-flow, then arrival, then **smaller cut**
                (`compare_map<DO_AREA>`). Taking a bare minimum kept whichever match happened to
                come first on a tie, which in a flat-cost landscape (many library cells share an
                area) systematically prefers the wider cell. */
                if ok {
                    /* emap amortises a shared gate over the *sum* of its roots' reference counts:
                    `AF = (Area(G) + |roots|*SUM AF(l)) / SUM est_refs(p)`. Charging the full gate
                    area to each root independently hides the sharing from the flow, which is why
                    the flow alone never picks shared instances and the decision has to be
                    recovered later by the pair pass. */
                    let s = if share && !m.extra_outs.is_empty() {
                        s / 2.0
                    } else {
                        s
                    };
                    let cand = (s as f32, m.leaves.len() as u32);
                    if cand < (best, best_size) {
                        best = cand.0;
                        best_size = cand.1;
                    }
                }
            }
            if best < marg[c] {
                /* Amortise over the estimated number of consumers (mockturtle emap:
                `flows[phase] /= est_refs[phase]`). Without this the tree-cost flow charges a
                shared leaf once per consumer, which systematically over-prices wide cuts and is
                why a plain flow converges to an AND2-and-inverter cover. */
                marg[c] = best / est_refs[c];
            }
        }

        /* 2) The inverter option in **both** directions — emap's `match_drop_phase`: a node whose
        negative phase has a cheap realisation can be built from it plus an inverter instead
        of needing a positive-phase match of its own. Each direction is priced from the
        other phase's *non-inverter* value (captured before either is updated), so the two
        can never bootstrap each other into a two-cycle. */
        let base = [marg[class_of(d, false)], marg[class_of(d, true)]];
        for phase in [true] {
            let c = class_of(d, phase);
            let b = base[usize::from(!phase)];
            if b.is_infinite() {
                continue;
            }
            for &mi in &by_out[c] {
                let m = &matches[mi];
                if !is_inv(m) {
                    continue;
                }
                let v = (f64::from(b) + w[mi]) as f32 / est_refs[c];
                if v < marg[c] {
                    marg[c] = v;
                }
            }
        }
    }
    marg
}

/// A node must not have **both** phases realised by an inverter — that is a two-cycle, and the
/// pricing in [`marginal_cost`] deliberately does not rule it out (each direction is priced from the
/// other's non-inverter value). Re-realise the offending side(s) from a non-inverter match.
fn repair_inverter_cycles(
    matches: &[Match],
    by_out: &[Vec<usize>],
    cost: &[f32],
    w: &[f64],
    dense: &Dense,
    n_classes: usize,
    pick: &mut [Option<usize>],
) {
    let is_inv = |m: &Match| m.leaves.first() == Some(&m.out_node);
    let alt = |c: usize| -> Option<usize> {
        let mut best = usize::MAX;
        let mut bv = f32::INFINITY;
        for &mi in &by_out[c] {
            let m = &matches[mi];
            if is_inv(m) {
                continue;
            }
            let mut s = w[mi];
            let mut ok = true;
            for (p, &leaf) in m.leaves.iter().enumerate() {
                let ic = class_of(leaf, m.leaf_neg[p]);
                if cost[ic].is_infinite() {
                    ok = false;
                    break;
                }
                s += f64::from(cost[ic]);
            }
            if ok && (s as f32) < bv {
                bv = s as f32;
                best = mi;
            }
        }
        (best != usize::MAX).then_some(best)
    };
    for &d in &dense.order {
        let cp = class_of(d, false);
        let cn = class_of(d, true);
        if cp >= n_classes || cn >= n_classes {
            continue;
        }
        let both = pick[cp].is_some_and(|i| is_inv(&matches[i]))
            && pick[cn].is_some_and(|i| is_inv(&matches[i]));
        if !both {
            continue;
        }
        if let Some(mi) = alt(cp) {
            pick[cp] = Some(mi);
        }
        if let Some(mj) = alt(cn) {
            pick[cn] = Some(mj);
        }
    }
}

/// Realise the required classes by walking down from the primary outputs and taking, for each class,
/// the cheapest match under `cost`. Sharing falls out of the `visited` set: a class is realised once
/// however many parents ask for it.
fn select_by_cost(
    matches: &[Match],
    by_out: &[Vec<usize>],
    required: &[(usize, bool)],
    is_free: &[bool],
    cost: &[f32],
    w: &[f64],
    n_classes: usize,
    share: bool,
) -> Vec<Option<usize>> {
    let mut pick: Vec<Option<usize>> = vec![None; n_classes];
    let mut visited = vec![false; n_classes];
    let mut stack: Vec<usize> = required.iter().map(|&(n, g)| class_of(n, g)).collect();
    while let Some(c) = stack.pop() {
        if visited[c] {
            continue;
        }
        visited[c] = true;
        if is_free[c] {
            continue;
        }
        let mut best = usize::MAX;
        let mut best_v = f32::INFINITY;
        let mut best_sz = u32::MAX;
        for &mi in &by_out[c] {
            let m = &matches[mi];
            /* Only `(n,neg) <- (n,pos)` inverters: allowing the reverse makes the walk able to pick
            both directions of one node, a two-cycle. The positive phase always has a direct
            match, so nothing is lost. */
            if m.leaves.first() == Some(&m.out_node) && !m.out_neg {
                continue;
            }
            let mut s = w[mi];
            let mut ok = true;
            for (p, &leaf) in m.leaves.iter().enumerate() {
                let ic = class_of(leaf, m.leaf_neg[p]);
                if cost[ic].is_infinite() {
                    ok = false;
                    break;
                }
                s += f64::from(cost[ic]);
            }
            if ok {
                let s = if share && !m.extra_outs.is_empty() {
                    s / 2.0
                } else {
                    s
                };
                let cand = (s as f32, m.leaves.len() as u32);
                if cand < (best_v, best_sz) {
                    best_v = cand.0;
                    best_sz = cand.1;
                    best = mi;
                }
            }
        }
        if best == usize::MAX {
            continue;
        }
        pick[c] = Some(best);
        for o in &matches[best].extra_outs {
            pick[class_of(o.node, o.neg)] = Some(best);
        }
        let kids: Vec<usize> = matches[best]
            .leaves
            .iter()
            .enumerate()
            .map(|(p, &l)| class_of(l, matches[best].leaf_neg[p]))
            .collect();
        for ic in kids {
            if !is_free[ic] {
                stack.push(ic);
            }
        }
    }
    pick
}

/// Drop realisations that no primary output actually needs. The walk down from the outputs can mark
/// a class visited along one path and then never use it; those cells would still be paid for.
fn prune_reachable(
    pick: &mut [Option<usize>],
    matches: &[Match],
    required: &[(usize, bool)],
    is_free: &[bool],
    n_classes: usize,
) {
    let mut reach = vec![false; n_classes];
    let mut stack: Vec<usize> = required.iter().map(|&(n, g)| class_of(n, g)).collect();
    while let Some(c) = stack.pop() {
        if reach[c] || is_free[c] {
            continue;
        }
        reach[c] = true;
        let Some(mi) = pick[c] else { continue };
        /* A multi-output instance realises *all* of its output pins at once. Those extra pins are
        outputs, not inputs, so the downward walk alone would never mark them reachable and they
        would be garbage-collected along with the instance that produces them. */
        for o in &matches[mi].extra_outs {
            let oc = class_of(o.node, o.neg);
            if !reach[oc] {
                stack.push(oc);
            }
        }
        for (p, &leaf) in matches[mi].leaves.iter().enumerate() {
            stack.push(class_of(leaf, matches[mi].leaf_neg[p]));
        }
    }
    for (c, p) in pick.iter_mut().enumerate() {
        if p.is_some() && !reach[c] {
            *p = None;
        }
    }
}

fn pick_area(matches: &[Match], pick: &[Option<usize>]) -> f32 {
    /* A multi-output instance appears once per output class, so the area must be counted once per
     *instance*, not once per class. */
    let mut seen = std::collections::HashSet::new();
    pick.iter()
        .filter_map(|p| *p)
        .filter(|i| seen.insert(*i))
        .map(|i| matches[i].area)
        .sum()
}

/// Minimum achievable arrival per class over the whole match set — a lower bound on the arrival of
/// *any* cover. Same per-node ordering discipline as [`marginal_cost`]: non-inverter realisations
/// first, then the `(n,neg) <- (n,pos)` inverter that depends on the same node.
fn arrival_flow(
    dense: &Dense,
    matches: &[Match],
    by_out: &[Vec<usize>],
    is_free: &[bool],
    n_classes: usize,
) -> Vec<f32> {
    let is_inv = |m: &Match| m.leaves.first() == Some(&m.out_node);
    let mut arr = vec![f32::INFINITY; n_classes];
    for c in 0..n_classes {
        if is_free[c] {
            arr[c] = 0.0;
        }
    }
    for &d in &dense.order {
        for pass_inv in [false, true] {
            for phase in [false, true] {
                let c = class_of(d, phase);
                if arr[c] == 0.0 {
                    continue;
                }
                let mut best = f32::INFINITY;
                for &mi in &by_out[c] {
                    let m = &matches[mi];
                    if is_inv(m) != pass_inv {
                        continue;
                    }
                    if pass_inv && !m.out_neg {
                        continue; // see `marginal_cost`
                    }
                    let mut a = 0.0f32;
                    let mut ok = true;
                    for (p, &leaf) in m.leaves.iter().enumerate() {
                        let ic = class_of(leaf, m.leaf_neg[p]);
                        if arr[ic].is_infinite() {
                            ok = false;
                            break;
                        }
                        a = a.max(arr[ic] + m.pin_delay[p]);
                    }
                    if ok && a < best {
                        best = a;
                    }
                }
                if best < arr[c] {
                    arr[c] = best;
                }
            }
        }
    }
    arr
}

/// Backward required-time propagation over the whole match set: the latest arrival each class may
/// have without pushing a primary output past `t_target`.
///
/// A class feeds several consumers when its sub-graph is shared, so its required time is the
/// **minimum** over them. Within one node the negative phase is resolved first, because the only
/// same-node edge is the inverter `(n,pos) -> (n,neg)`.
fn required_times(
    dense: &Dense,
    matches: &[Match],
    by_out: &[Vec<usize>],
    required: &[(usize, bool)],
    is_free: &[bool],
    n_classes: usize,
    t_target: f32,
) -> Vec<f32> {
    let mut req = vec![f32::INFINITY; n_classes];
    for &(n, g) in required {
        let c = class_of(n, g);
        if !is_free[c] && t_target < req[c] {
            req[c] = t_target;
        }
    }
    for &d in dense.order.iter().rev() {
        for phase in [true, false] {
            let o = class_of(d, phase);
            if req[o].is_infinite() {
                continue;
            }
            for &mi in &by_out[o] {
                let m = &matches[mi];
                if m.leaves.first() == Some(&m.out_node) && !m.out_neg {
                    continue;
                }
                for (p, &leaf) in m.leaves.iter().enumerate() {
                    let ic = class_of(leaf, m.leaf_neg[p]);
                    if is_free[ic] {
                        continue;
                    }
                    let r = req[o] - m.pin_delay[p];
                    if r < req[ic] {
                        req[ic] = r;
                    }
                }
            }
        }
    }
    req
}

/// Timing-constrained version of [`select_by_cost`]: among the matches that still meet `req[c]`,
/// take the cheapest under `cost`. When a class has *no* timing-feasible match the fastest one is
/// taken anyway, so the walk always produces a complete cover.
fn select_timed(
    matches: &[Match],
    by_out: &[Vec<usize>],
    required: &[(usize, bool)],
    is_free: &[bool],
    arr: &[f32],
    req: &[f32],
    cost: &[f32],
    n_classes: usize,
) -> Vec<Option<usize>> {
    let mut pick: Vec<Option<usize>> = vec![None; n_classes];
    let mut visited = vec![false; n_classes];
    let mut stack: Vec<usize> = required.iter().map(|&(n, g)| class_of(n, g)).collect();
    while let Some(c) = stack.pop() {
        if visited[c] {
            continue;
        }
        visited[c] = true;
        if is_free[c] {
            continue;
        }
        let limit = req[c];
        let mut best = usize::MAX;
        let mut best_v = f32::INFINITY;
        let mut fb = usize::MAX;
        let mut fb_v = f32::INFINITY;
        for &mi in &by_out[c] {
            let m = &matches[mi];
            if m.leaves.first() == Some(&m.out_node) && !m.out_neg {
                continue;
            }
            let mut a = 0.0f32;
            let mut s = f64::from(m.area);
            let mut ok = true;
            for (p, &leaf) in m.leaves.iter().enumerate() {
                let ic = class_of(leaf, m.leaf_neg[p]);
                if cost[ic].is_infinite() {
                    ok = false;
                    break;
                }
                a = a.max(arr[ic] + m.pin_delay[p]);
                s += f64::from(cost[ic]);
            }
            if !ok {
                continue;
            }
            let v = s as f32;
            if v < fb_v {
                fb_v = v;
                fb = mi;
            }
            if a <= limit + 1e-6 && v < best_v {
                best_v = v;
                best = mi;
            }
        }
        let chosen = if best != usize::MAX { best } else { fb };
        if chosen == usize::MAX {
            continue;
        }
        pick[c] = Some(chosen);
        for o in &matches[chosen].extra_outs {
            pick[class_of(o.node, o.neg)] = Some(chosen);
        }
        let kids: Vec<usize> = matches[chosen]
            .leaves
            .iter()
            .enumerate()
            .map(|(p, &l)| class_of(l, matches[chosen].leaf_neg[p]))
            .collect();
        for ic in kids {
            if !is_free[ic] {
                stack.push(ic);
            }
        }
    }
    pick
}

/// Delay-oriented selection: for every class take the match with the smallest **own** arrival,
/// which is what a conventional delay-driven mapper does. Produces the delay-optimal cover of the
/// match set and seeds the timing-aware area recovery.
fn select_delay(
    matches: &[Match],
    by_out: &[Vec<usize>],
    required: &[(usize, bool)],
    is_free: &[bool],
    arr: &[f32],
    n_classes: usize,
) -> Vec<Option<usize>> {
    let mut pick: Vec<Option<usize>> = vec![None; n_classes];
    let mut visited = vec![false; n_classes];
    let mut stack: Vec<usize> = required.iter().map(|&(n, g)| class_of(n, g)).collect();
    while let Some(c) = stack.pop() {
        if visited[c] {
            continue;
        }
        visited[c] = true;
        if is_free[c] {
            continue;
        }
        let mut best = usize::MAX;
        let mut best_a = f32::INFINITY;
        for &mi in &by_out[c] {
            let m = &matches[mi];
            if m.leaves.first() == Some(&m.out_node) && !m.out_neg {
                continue;
            }
            let mut a = 0.0f32;
            let mut ok = true;
            for (p, &leaf) in m.leaves.iter().enumerate() {
                let ic = class_of(leaf, m.leaf_neg[p]);
                if arr[ic].is_infinite() {
                    ok = false;
                    break;
                }
                a = a.max(arr[ic] + m.pin_delay[p]);
            }
            if ok && a < best_a {
                best_a = a;
                best = mi;
            }
        }
        if best == usize::MAX {
            continue;
        }
        pick[c] = Some(best);
        for o in &matches[best].extra_outs {
            pick[class_of(o.node, o.neg)] = Some(best);
        }
        let kids: Vec<usize> = matches[best]
            .leaves
            .iter()
            .enumerate()
            .map(|(p, &l)| class_of(l, matches[best].leaf_neg[p]))
            .collect();
        for ic in kids {
            if !is_free[ic] {
                stack.push(ic);
            }
        }
    }
    pick
}

/// Which output inverter, if any, an instance takes on the class `c` it is realising.
fn out_inv_of(m: &Match, c: usize) -> Option<bool> {
    if class_of(m.out_node, m.out_neg) == c {
        return Some(m.out_inv);
    }
    m.extra_outs
        .iter()
        .find(|o| class_of(o.node, o.neg) == c)
        .map(|o| o.inv)
}

/// Arrival of the **chosen** cover. Unlike [`arrival_flow`] this is exact, which matters: a match
/// that looks timing-feasible against the min-achievable lower bound can still be slow once its own
/// inputs are realised.
fn cover_arrival(
    dense: &Dense,
    matches: &[Match],
    pick: &[Option<usize>],
    is_free: &[bool],
    n_classes: usize,
    inv_delay: f32,
) -> Vec<f32> {
    let mut arr = vec![f32::INFINITY; n_classes];
    for c in 0..n_classes {
        if is_free[c] {
            arr[c] = 0.0;
        }
    }
    for &d in &dense.order {
        for pass_inv in [false, true] {
            for phase in [false, true] {
                let c = class_of(d, phase);
                if is_free[c] || pick[c].is_none() {
                    continue;
                }
                let m = &matches[pick[c].expect("checked")];
                let inv = m.leaves.first() == Some(&m.out_node);
                if inv != pass_inv {
                    continue;
                }
                let mut a = 0.0f32;
                for (p, &leaf) in m.leaves.iter().enumerate() {
                    let ic = class_of(leaf, m.leaf_neg[p]);
                    if arr[ic].is_infinite() {
                        a = f32::INFINITY;
                        break;
                    }
                    a = a.max(arr[ic] + m.pin_delay[p]);
                }
                arr[c] = a;
            }
        }
    }
    arr
}

/// Required times propagated along the **chosen** cover up to `t_target` at the primary outputs.
/// A shared class feeds several consumers, so its required time is the tightest of them.
fn required_times_cover(
    dense: &Dense,
    matches: &[Match],
    pick: &[Option<usize>],
    required: &[(usize, bool)],
    is_free: &[bool],
    n_classes: usize,
    t_target: f32,
) -> Vec<f32> {
    let mut req = vec![f32::INFINITY; n_classes];
    for &(n, g) in required {
        let c = class_of(n, g);
        if !is_free[c] && t_target < req[c] {
            req[c] = t_target;
        }
    }
    for &d in dense.order.iter().rev() {
        // the only same-node edge is (n,pos) -> (n,neg), so resolve `neg` first
        for phase in [true, false] {
            let o = class_of(d, phase);
            if req[o].is_infinite() {
                continue;
            }
            let Some(mi) = pick[o] else { continue };
            let m = &matches[mi];
            for (p, &leaf) in m.leaves.iter().enumerate() {
                let ic = class_of(leaf, m.leaf_neg[p]);
                if is_free[ic] {
                    continue;
                }
                let r = req[o] - m.pin_delay[p];
                if r < req[ic] {
                    req[ic] = r;
                }
            }
        }
    }
    req
}

/// Stage-S4 cover under an arrival target.
///
/// Starts from the delay-optimal cover, then repeatedly re-prices with already-realised classes at
/// cost zero and re-selects the cheapest timing-feasible realisation. `arr` and `req` are recomputed
/// from the **actual** cover each round, so the feasibility test is exact at the point it is applied.
/// Sweeping `t_target` from the delay-optimal arrival upwards traces the area–delay Pareto front.
fn timed_cover(
    dense: &Dense,
    matches: &[Match],
    by_out: &[Vec<usize>],
    required: &[(usize, bool)],
    is_free: &[bool],
    n_classes: usize,
    t_target: f32,
    rounds: usize,
    w: &[f64],
    est_refs: &[f32],
    inv_delay: f32,
) -> Vec<usize> {
    let arr_min = arrival_flow(dense, matches, by_out, is_free, n_classes);
    let mut pick = select_delay(matches, by_out, required, is_free, &arr_min, n_classes);
    prune_reachable(&mut pick, matches, required, is_free, n_classes);

    let arr0 = cover_arrival(dense, matches, &pick, is_free, n_classes, inv_delay);
    let mut best_area = pick_area(matches, &pick);
    let mut best_arr = required
        .iter()
        .map(|&(n, g)| arr0[class_of(n, g)])
        .fold(0.0f32, f32::max);
    let mut best_pick = pick.clone();

    /* Ranking uses the plain tree-cost area flow, not the sharing-aware marginal of the pure-area
    path. The marginal marks already-realised classes as free, which is right for area but is a
    trap here: starting from the delay-optimal cover every alternative match would appear more
    expensive than the incumbent, and the recovery could never move. This is the classical
    arrange: rank by area flow, filter by required time. */
    let none: Vec<Option<usize>> = vec![None; n_classes];
    let flow = marginal_cost(
        dense, matches, by_out, &none, is_free, n_classes, w, est_refs, false,
    );
    for _ in 0..rounds {
        let arr = cover_arrival(dense, matches, &pick, is_free, n_classes, inv_delay);
        let req = required_times_cover(
            dense, matches, &pick, required, is_free, n_classes, t_target,
        );
        let mut cand = select_timed(
            matches, by_out, required, is_free, &arr, &req, &flow, n_classes,
        );
        prune_reachable(&mut cand, matches, required, is_free, n_classes);
        if !cover_complete(matches, &cand, required, is_free) {
            break;
        }
        let carr = cover_arrival(dense, matches, &cand, is_free, n_classes, inv_delay);
        let cmax = required
            .iter()
            .map(|&(n, g)| carr[class_of(n, g)])
            .fold(0.0f32, f32::max);
        let carea = pick_area(matches, &cand);
        // keep only Pareto-improving points within the arrival target
        if carea < best_area && cmax <= t_target * (1.0 + 1e-3) {
            best_area = carea;
            best_arr = cmax;
            best_pick = cand.clone();
        }
        if cand == pick {
            break;
        }
        pick = cand;
    }

    let _ = best_arr;
    let mut is_picked = vec![false; matches.len()];
    for p in best_pick.iter().flatten() {
        is_picked[*p] = true;
    }
    (0..matches.len()).filter(|&i| is_picked[i]).collect()
}

/// Materialise any class that is reachable from the primary outputs but has no chosen realisation.
///
/// A swap in [`exact_area_pass`] can make previously-dead classes live again; those must be given a
/// realisation before the cover is valid. Ranking uses the same weight as the calling cover.
fn fill_missing(
    matches: &[Match],
    by_out: &[Vec<usize>],
    required: &[(usize, bool)],
    is_free: &[bool],
    n_classes: usize,
    w: &[f64],
    pick: &mut [Option<usize>],
) {
    let mut visited = vec![false; n_classes];
    let mut stack: Vec<usize> = required.iter().map(|&(n, g)| class_of(n, g)).collect();
    while let Some(c) = stack.pop() {
        if visited[c] {
            continue;
        }
        visited[c] = true;
        if is_free[c] {
            continue;
        }
        if pick[c].is_none() {
            let mut best = usize::MAX;
            let mut best_v = f64::INFINITY;
            /* Fallback for classes whose every realisation has infinite *flow* cost. `w` is a tree
            cost produced by the area flow, and a class can end up with `w == INFINITY` for all of
            its matches even though a legal realisation exists (the flow's own thinning drops the
            match it would have needed). The old code then left the class unrealised, which made
            the whole cover incomplete — the greedy's `if w[mi] < best_v` test can never select an
            `INFINITY` match because `INF < INF` is false. Since `greedy_cover` is the mandatory
            fallback, it must be able to finish, so fall back to the cheapest-area match. */
            let mut fb = usize::MAX;
            let mut fb_area = f32::INFINITY;
            for &mi in &by_out[c] {
                let m = &matches[mi];
                if m.leaves.first() == Some(&m.out_node) && !m.out_neg {
                    continue;
                }
                if w[mi] < best_v {
                    best_v = w[mi];
                    best = mi;
                }
                if m.area < fb_area {
                    fb_area = m.area;
                    fb = mi;
                }
            }
            if best == usize::MAX {
                best = fb;
            }
            if best == usize::MAX {
                continue;
            }
            pick[c] = Some(best);
        }
        let Some(mi) = pick[c] else { continue };
        let kids: Vec<usize> = matches[mi]
            .leaves
            .iter()
            .enumerate()
            .map(|(p, &l)| class_of(l, matches[mi].leaf_neg[p]))
            .collect();
        for ic in kids {
            if !is_free[ic] {
                stack.push(ic);
            }
        }
    }
}

/// Build the incremental reference-count state for a cover: `refs[c]` is the number of selected
/// matches that consume `c` as a leaf, plus one permanent unit for a required (primary-output) class
/// so that a required class can never be reclaimed.
fn inc_state_init(
    matches: &[Match],
    pick: &[Option<usize>],
    required: &[(usize, bool)],
    is_free: &[bool],
    n_classes: usize,
) -> IncState {
    let mut is_req = vec![false; n_classes];
    for &(n, g) in required {
        let c = class_of(n, g);
        if !is_free[c] {
            is_req[c] = true;
        }
    }
    let mut st = IncState {
        refs: vec![0u32; n_classes],
        pick: pick.to_vec(),
    };
    for c in 0..n_classes {
        if is_req[c] {
            st.refs[c] += 1;
        }
    }
    let mut counted = vec![false; matches.len()];
    for c in 0..n_classes {
        let Some(mi) = st.pick[c] else { continue };
        if counted[mi] {
            continue; // a shared instance's inputs count once, not once per output
        }
        counted[mi] = true;
        for o in &matches[mi].extra_outs {
            let oc = class_of(o.node, o.neg);
            if !is_free[oc] {
                st.refs[oc] += 1;
            }
        }
        for (p, &l) in matches[mi].leaves.iter().enumerate() {
            let ic = class_of(l, matches[mi].leaf_neg[p]);
            if !is_free[ic] {
                st.refs[ic] += 1;
            }
        }
    }
    st
}

/// Incremental state for [`exact_area_pass_fast`]: reference counts plus the chosen realisation.
///
/// `refs[c]` counts, for class `c`: +1 when it is required, +1 for every selected instance that
/// consumes it as an input, and +1 for every selected instance that produces it (a multi-output
/// pin). A class is materialised exactly while `refs[c] > 0`, which makes freeing a purely local
/// operation — no garbage collection over the whole cover.
#[derive(Clone)]
struct IncState {
    refs: Vec<u32>,
    pick: Vec<Option<usize>>,
}

fn inc_free(
    c: usize,
    matches: &[Match],
    is_free: &[bool],
    is_req: &[bool],
    st: &mut IncState,
) -> f32 {
    if is_free[c] || st.refs[c] == 0 {
        return 0.0;
    }
    /* A required class (a primary output) is never freed. Its `refs` entry is permanent, so it must
    not be decremented here either: doing so let a required class be reclaimed as soon as one of
    its consumers was rewritten, which silently broke the cover. */
    if is_req[c] {
        return 0.0;
    }
    st.refs[c] -= 1;
    if st.refs[c] > 0 {
        return 0.0;
    }
    let Some(mi) = st.pick[c] else { return 0.0 };
    st.pick[c] = None;
    let mut cost = matches[mi].area;
    for o in &matches[mi].extra_outs {
        cost += inc_free(class_of(o.node, o.neg), matches, is_free, is_req, st);
    }
    let kids: Vec<usize> = matches[mi]
        .leaves
        .iter()
        .enumerate()
        .map(|(p, &l)| class_of(l, matches[mi].leaf_neg[p]))
        .collect();
    for ic in kids {
        cost += inc_free(ic, matches, is_free, is_req, st);
    }
    cost
}

/// Drop the current realisation of `c` itself (its references stay), returning the area released.
fn inc_free_match(
    c: usize,
    matches: &[Match],
    is_free: &[bool],
    is_req: &[bool],
    st: &mut IncState,
) -> f32 {
    let Some(mi) = st.pick[c] else { return 0.0 };
    st.pick[c] = None;
    let mut cost = matches[mi].area;
    for o in &matches[mi].extra_outs {
        cost += inc_free(class_of(o.node, o.neg), matches, is_free, is_req, st);
    }
    let kids: Vec<usize> = matches[mi]
        .leaves
        .iter()
        .enumerate()
        .map(|(p, &l)| class_of(l, matches[mi].leaf_neg[p]))
        .collect();
    for ic in kids {
        cost += inc_free(ic, matches, is_free, is_req, st);
    }
    cost
}

/// Take one more reference to `c`, materialising it (and whatever that needs) when it was dead.
fn inc_add(
    c: usize,
    matches: &[Match],
    by_out: &[Vec<usize>],
    is_free: &[bool],
    w: &[f64],
    st: &mut IncState,
) -> f32 {
    if is_free[c] {
        return 0.0;
    }
    st.refs[c] += 1;
    if st.refs[c] > 1 {
        return 0.0;
    }
    let mut best = usize::MAX;
    let mut bv = f64::INFINITY;
    for &mi in &by_out[c] {
        let m = &matches[mi];
        if m.leaves.first() == Some(&m.out_node) && !m.out_neg {
            continue;
        }
        /* only ordinary single-output instances whose *primary* output is `c`: taking one where `c`
        is merely an extra pin would leave the primary class unrealised, and a shared instance's
        joint saving is invisible to a per-class walk anyway. Those are the pair pass's business. */
        if !m.extra_outs.is_empty() || class_of(m.out_node, m.out_neg) != c {
            continue;
        }
        if w[mi] < bv {
            bv = w[mi];
            best = mi;
        }
    }
    if best == usize::MAX {
        /* Nothing can realise this class, so the candidate that needed it is not a legal cover.
        Returning infinity makes it lose the ranking instead of leaving `refs = 1, pick = None`
        behind — the old implementation caught the same case with `cover_complete`. */
        st.refs[c] = 0;
        return f32::INFINITY;
    }
    st.pick[c] = Some(best);
    st.refs[c] = 1;
    inc_add_match_body(c, best, matches, by_out, is_free, w, st)
}

/* Remove the body of a match that has already been superseded at its primary class.
`inc_free_match` cannot be used for this: it clears `st.pick[c]` and would clobber the
replacement. More importantly, callers must install the replacement FIRST, so that any leaf
shared between the old and the new match is already ref-protected and is not transiently
freed and then re-realised with a different (and possibly sub-cone-pulling) match. */
fn inc_free_match_at(
    old_mi: usize,
    matches: &[Match],
    is_free: &[bool],
    is_req: &[bool],
    st: &mut IncState,
) -> f32 {
    let mut cost = matches[old_mi].area;
    for o in &matches[old_mi].extra_outs {
        cost += inc_free(class_of(o.node, o.neg), matches, is_free, is_req, st);
    }
    let kids: Vec<usize> = matches[old_mi]
        .leaves
        .iter()
        .enumerate()
        .map(|(p, &l)| class_of(l, matches[old_mi].leaf_neg[p]))
        .collect();
    for ic in kids {
        cost += inc_free(ic, matches, is_free, is_req, st);
    }
    cost
}

fn inc_add_match_body(
    c: usize,
    mi: usize,
    matches: &[Match],
    by_out: &[Vec<usize>],
    is_free: &[bool],
    w: &[f64],
    st: &mut IncState,
) -> f32 {
    /* Set the realisation first: the ranking loop commits `st.pick` wholesale, so forgetting this
    left the class with no producer after commit and silently produced an incomplete cover. */
    st.pick[c] = Some(mi);
    let mut cost = matches[mi].area;
    for o in &matches[mi].extra_outs {
        let oc = class_of(o.node, o.neg);
        if !is_free[oc] {
            st.refs[oc] += 1;
            if st.pick[oc].is_none() {
                st.pick[oc] = Some(mi);
            }
        }
    }
    let kids: Vec<usize> = matches[mi]
        .leaves
        .iter()
        .enumerate()
        .map(|(p, &l)| class_of(l, matches[mi].leaf_neg[p]))
        .collect();
    for ic in kids {
        cost += inc_add(ic, matches, by_out, is_free, w, st);
    }
    cost
}

/// Exact-area reverse remapping with **ref-counted incremental deltas**.
///
/// The previous version cloned the whole cover, ran `fill_missing` + `prune_reachable` and resummed
/// the area for every candidate; on `multiplier` that was 452 s of a 465 s run. Here a candidate is
/// priced by freeing its victim's realisation and materialising the candidate's inputs against
/// reference counts, touching only the affected cone. Cloning `refs`/`pick` is a plain memcpy, so a
/// trial costs a cone walk plus two memcpys rather than a hashed graph traversal.
///
/// The DAG semantics are the same: a class alive for another consumer is not freed, and a class that
/// becomes live again is materialised, so the delta is the true change in covered area.
fn exact_area_pass_fast(
    dense: &Dense,
    matches: &[Match],
    by_out: &[Vec<usize>],
    required: &[(usize, bool)],
    is_free: &[bool],
    n_classes: usize,
    w: &[f64],
    pick: &mut [Option<usize>],
    rounds: usize,
) {
    let mut is_req = vec![false; n_classes];
    for &(n, g) in required {
        let c = class_of(n, g);
        if !is_free[c] {
            is_req[c] = true;
        }
    }
    let mut st = inc_state_init(matches, pick, required, is_free, n_classes);

    let mut area = pick_area(matches, pick);
    for _ in 0..rounds {
        let mut improved = false;
        for &d in dense.order.iter().rev() {
            for phase in [true, false] {
                let c = class_of(d, phase);
                let Some(cur) = st.pick[c] else { continue };
                /* Replacing one output of a shared multi-output instance would free inputs the
                other output still needs. Those classes are left to the joint pair pass. */
                if !matches[cur].extra_outs.is_empty()
                    || class_of(matches[cur].out_node, matches[cur].out_neg) != c
                {
                    continue;
                }
                /* The incremental delta is exact for a single-class replacement: `inc_add`
                installs the candidate and ref-protects every leaf it shares with the outgoing
                match *before* `inc_free_match_at` tears the old body down — precisely what the
                global trial did (overwrite `pick[c]`, prune afterwards). The global
                `fill_missing` + `prune_reachable` + `cover_complete` + `pick_area` that used to
                settle each candidate is therefore redundant, and was an O(n_classes) pass per
                candidate per class per round. Keeping only the best delta is enough; measured
                identical to K = 4 and to the old pass on ISCAS/EPFL. */
                let mut best_delta = -1e-6f32;
                let mut best_trial: Option<IncState> = None;
                for &mi in &by_out[c] {
                    if mi == cur {
                        continue;
                    }
                    let m = &matches[mi];
                    if m.leaves.first() == Some(&m.out_node) && !m.out_neg {
                        continue;
                    }
                    if !m.extra_outs.is_empty() || class_of(m.out_node, m.out_neg) != c {
                        continue;
                    }
                    let mut trial = IncState {
                        refs: st.refs.clone(),
                        pick: st.pick.clone(),
                    };
                    let added = inc_add_match_body(c, mi, matches, by_out, is_free, w, &mut trial);
                    let freed = inc_free_match_at(cur, matches, is_free, &is_req, &mut trial);
                    let delta = added - freed;
                    if delta < best_delta {
                        best_delta = delta;
                        best_trial = Some(trial);
                    }
                }
                if let Some(t) = best_trial {
                    st = t;
                    area += best_delta;
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
        /* `area` is only bookkeeping now; re-derive it from the cover once per round so that
        accumulating thousands of tiny f32 deltas cannot drift away from the true area. */
        area = pick_area(matches, &st.pick);
    }
    pick.copy_from_slice(&st.pick);
}

/// Restricted-neighbourhood ILP refinement.
///
/// The full covering ILP is far too large for CBC on the big circuits — `multiplier`, `sqrt`, `log2`
/// and `div` all time out even at 1200 s, so their covers are only whatever the greedy coordinate
/// descent settled on and carry no optimality proof (unlike `adder` / `sin`, where CBC returns the
/// same answer as the greedy). This re-solves small, topologically contiguous windows **exactly**.
/// That is strictly stronger than the 1-opt/2-opt swap search, which could only try a truncated set
/// of candidate pairs, while still keeping every individual subproblem inside CBC's reach.
///
/// A window is a band of consecutive nodes in dense (topological) order, closed under the *currently
/// realised* ancestors of those nodes. Classes outside the window keep their present realisation and
/// are treated as free inputs, so the local objective is the true global area up to a constant.
/// Returns `true` if any window improved the cover.
#[allow(clippy::too_many_arguments)]
fn lns_pass(
    dense: &Dense,
    matches: &[Match],
    by_out: &[Vec<usize>],
    required: &[(usize, bool)],
    is_free: &[bool],
    n_classes: usize,
    w: &[f64],
    pick: &mut [Option<usize>],
    windows: usize,
    window_nodes: usize,
    secs_per_window: f64,
    dbg: bool,
) -> bool {
    use coin_cbc::{Col, Model, Sense};

    /* Only single-output matches take part: a shared instance realises two classes at once, which
    does not fit the "exactly one producer per class" row, and the joint pair pass owns those. */
    let mut prod: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, m) in matches.iter().enumerate() {
        if m.extra_outs.is_empty() {
            prod[class_of(m.out_node, m.out_neg)].push(i);
        }
    }

    let mut improved_any = false;
    let order_len = dense.order.len();
    for win in 0..windows {
        let span = (order_len / windows.max(1)).max(1);
        let start = (win * span).min(order_len.saturating_sub(1));
        let end = (start + window_nodes).min(order_len);
        if start >= end {
            break;
        }

        /* Window closure: the band's classes plus every currently realised ancestor. */
        let mut in_s = vec![false; n_classes];
        let mut members: Vec<usize> = Vec::new();
        let mut stack: Vec<usize> = Vec::new();
        for &d in &dense.order[start..end] {
            stack.push(class_of(d, false));
            stack.push(class_of(d, true));
        }
        while let Some(c) = stack.pop() {
            if in_s[c] || is_free[c] || prod[c].is_empty() {
                continue; // free, or nothing single-output can realise it here: leave it fixed
            }
            in_s[c] = true;
            members.push(c);
            if let Some(mi) = pick[c] {
                let m = &matches[mi];
                if !m.extra_outs.is_empty() {
                    continue;
                }
                for (p, &l) in m.leaves.iter().enumerate() {
                    stack.push(class_of(l, m.leaf_neg[p]));
                }
            }
        }
        if dbg {
            eprintln!(
                "  lns win {win}: nodes [{start},{end}) members={}",
                members.len()
            );
        }
        if members.len() < 4 {
            continue;
        }

        let mut model = Model::default();
        model.set_parameter("log", "0");
        model.set_parameter("seconds", &format!("{}", secs_per_window.max(1.0) as u64));
        model.set_obj_sense(Sense::Minimize);

        /* One binary per match whose primary class is in the window. A match that needs a leaf which
        is neither free nor inside the window is **excluded**, not merely penalised: the outside
        cover provides those leaves only as currently realised, so allowing such a variable would
        let the model propose a locally cheaper solution that does not extend to a cover (it would
        then fail `cover_complete` and be thrown away, wasting the window). */
        let mut var: Vec<(usize, Col)> = Vec::new();
        for &c in &members {
            for &i in &prod[c] {
                let m = &matches[i];
                let feasible = m.leaves.iter().enumerate().all(|(p, &l)| {
                    let ic = class_of(l, m.leaf_neg[p]);
                    is_free[ic] || in_s[ic]
                });
                if !feasible {
                    continue;
                }
                let col = model.add_binary();
                model.set_obj_coeff(col, f64::from(matches[i].area));
                var.push((i, col));
            }
        }
        let mut col_of: HashMap<usize, Col> = HashMap::new();
        for &(i, col) in &var {
            col_of.insert(i, col);
        }

        for &c in &members {
            /* Exactly one realisation — this is what makes the window a real re-solve rather than a
            relaxation that can share one fractional cell. Classes with no feasible in-window
            match keep their current pick and get no row. */
            let avail: Vec<usize> = prod[c]
                .iter()
                .copied()
                .filter(|i| col_of.contains_key(i))
                .collect();
            if avail.is_empty() {
                continue;
            }
            let row = model.add_row();
            model.set_row_equal(row, 1.0);
            for i in avail {
                model.set_weight(row, col_of[&i], 1.0);
            }
        }
        for &(i, col) in &var {
            let m = &matches[i];
            let oc = class_of(m.out_node, m.out_neg);
            for (p, &l) in m.leaves.iter().enumerate() {
                let ic = class_of(l, m.leaf_neg[p]);
                if is_free[ic] || !in_s[ic] || ic == oc {
                    continue; // provided by the unchanged outside cover, or an inverter self-edge
                }
                let row = model.add_row();
                model.set_row_upper(row, 0.0);
                model.set_weight(row, col, 1.0);
                for &j in &prod[ic] {
                    if let Some(&cj) = col_of.get(&j) {
                        model.set_weight(row, cj, -1.0);
                    }
                }
            }
        }

        let sol = model.solve();
        let integral = var.iter().all(|&(_, col)| {
            let v = sol.col(col);
            (v - 0.0).abs() <= 1e-6 || (v - 1.0).abs() <= 1e-6
        });
        if !integral {
            if dbg {
                eprintln!("  lns win {win}: non-integral, skipped");
            }
            continue;
        }

        let mut cand = pick.to_vec();
        for &c in &members {
            for &i in &prod[c] {
                if let Some(&col) = col_of.get(&i) {
                    if sol.col(col) > 0.5 {
                        cand[c] = Some(i);
                    }
                }
            }
        }
        if !cover_complete(matches, &cand, required, is_free)
            || first_cycle(matches, &cand).is_some()
        {
            continue;
        }
        let na = pick_area(matches, &cand);
        let oa = pick_area(matches, pick);
        if na < oa - 1e-6 {
            let _ = w;
            pick.copy_from_slice(&cand);
            improved_any = true;
            if dbg {
                eprintln!(
                    "  lns win {win}: {oa:.3} -> {na:.3} (members={})",
                    members.len()
                );
            }
        }
    }
    improved_any
}

/// Two-swap extension of [`exact_area_pass_fast`].
///
/// The 1-opt pass is a *local* optimum of the true cover area, but a cover can be 1-opt optimal and
/// still far from 2-opt optimal: replacing `A` alone may cost, and replacing `B` alone may cost,
/// while replacing both simultaneously orphans a shared cone that neither orphans on its own. This is
/// the single-output analogue of the joint multi-output pair search, and it is the mechanism that
/// could let the cover escape the "many small cells" basin emap is better at avoiding.
///
/// Candidates are the cheapest alternative realisation per class (skipping classes realised by a
/// shared instance, which the multi-output pass owns); the `width` best by local gain are paired up
/// and every pair is settled with the full repaired-area test.
#[allow(clippy::too_many_arguments)]
fn exact_pairs_pass(
    dense: &Dense,
    matches: &[Match],
    by_out: &[Vec<usize>],
    required: &[(usize, bool)],
    is_free: &[bool],
    n_classes: usize,
    w: &[f64],
    pick: &mut [Option<usize>],
    width: usize,
    rounds: usize,
) {
    let mut st = inc_state_init(matches, pick, required, is_free, n_classes);
    let mut area = pick_area(matches, &st.pick);
    for _ in 0..rounds {
        let mut cands: Vec<(f32, usize, usize)> = Vec::new();
        for c in 0..n_classes {
            let Some(cur) = st.pick[c] else { continue };
            if is_free[c]
                || !matches[cur].extra_outs.is_empty()
                || class_of(matches[cur].out_node, matches[cur].out_neg) != c
            {
                continue;
            }
            let mut best_alt = usize::MAX;
            let mut best_gain = -1e-6f32;
            for &mi in &by_out[c] {
                if mi == cur {
                    continue;
                }
                let m = &matches[mi];
                if m.leaves.first() == Some(&m.out_node) && !m.out_neg {
                    continue;
                }
                if !m.extra_outs.is_empty() || class_of(m.out_node, m.out_neg) != c {
                    continue;
                }
                let gain = matches[cur].area - m.area;
                if gain > best_gain {
                    best_gain = gain;
                    best_alt = mi;
                }
            }
            if best_alt != usize::MAX {
                cands.push((best_gain, c, best_alt));
            }
        }
        if cands.len() < 2 {
            break;
        }
        cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        cands.truncate(width);
        let base = st.pick.clone();
        let mut best: Option<(Vec<Option<usize>>, f32)> = None;
        for a in 0..cands.len() {
            for b in (a + 1)..cands.len() {
                let (ca, ma) = (cands[a].1, cands[a].2);
                let (cb, mb) = (cands[b].1, cands[b].2);
                if ca == cb {
                    continue;
                }
                let mut cand = base.clone();
                cand[ca] = Some(ma);
                cand[cb] = Some(mb);
                fill_missing(matches, by_out, required, is_free, n_classes, w, &mut cand);
                prune_reachable(&mut cand, matches, required, is_free, n_classes);
                if !cover_complete(matches, &cand, required, is_free)
                    || first_cycle(matches, &cand).is_some()
                {
                    continue;
                }
                let na = pick_area(matches, &cand);
                if best.as_ref().map_or(true, |(_, v)| na < *v - 1e-6) {
                    best = Some((cand, na));
                }
            }
        }
        match best {
            Some((cand, na)) if na < area - 1e-6 => {
                st = inc_state_init(matches, &cand, required, is_free, n_classes);
                area = na;
            }
            _ => break,
        }
    }
    pick.copy_from_slice(&st.pick);
}

/// Exact-area reverse remapping — the counterpart of mockturtle emap's
/// `compute_mapping_exact_reversed` (`ela_rounds`). Tree-cost area flow systematically over-prices
/// wide cuts, because it charges every leaf even when that leaf is shared and therefore already paid
/// for. Here the cost of a candidate is the **actual area of the resulting cover**: swap one
/// realisation, materialise whatever became live, garbage-collect whatever the primary outputs no
/// longer reach, and keep the swap only if the total drops. Sweeping from the outputs backwards is
/// what lets a swap release the references the next, cheaper swap depends on.
fn exact_area_pass(
    dense: &Dense,
    matches: &[Match],
    by_out: &[Vec<usize>],
    required: &[(usize, bool)],
    is_free: &[bool],
    n_classes: usize,
    w: &[f64],
    pick: &mut [Option<usize>],
    rounds: usize,
) {
    fill_missing(matches, by_out, required, is_free, n_classes, w, pick);
    prune_reachable(pick, matches, required, is_free, n_classes);
    let mut area = pick_area(matches, pick);
    for _ in 0..rounds {
        let mut improved = false;
        for &d in dense.order.iter().rev() {
            for phase in [true, false] {
                let c = class_of(d, phase);
                let Some(cur) = pick[c] else { continue };
                let mut best = cur;
                let mut best_area = area;
                let mut best_trial: Option<Vec<Option<usize>>> = None;
                for &mi in &by_out[c] {
                    if mi == cur {
                        continue;
                    }
                    let m = &matches[mi];
                    if m.leaves.first() == Some(&m.out_node) && !m.out_neg {
                        continue;
                    }
                    let mut trial = pick.to_vec();
                    trial[c] = Some(mi);
                    fill_missing(matches, by_out, required, is_free, n_classes, w, &mut trial);
                    prune_reachable(&mut trial, matches, required, is_free, n_classes);
                    if !cover_complete(matches, &trial, required, is_free) {
                        continue;
                    }
                    let a = pick_area(matches, &trial);
                    if a < best_area - 1e-6 {
                        best_area = a;
                        best = mi;
                        best_trial = Some(trial);
                    }
                }
                if best != cur {
                    if let Some(t) = best_trial {
                        pick.copy_from_slice(&t);
                    } else {
                        pick[c] = Some(best);
                    }
                    area = best_area;
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
}

/// Joint pass for shared multi-output instances.
///
/// The per-class DP is structurally blind to these: it prices each class independently, so a
/// multi-output instance that covers two classes for one area always looks more expensive than the
/// cheaper of the two single-output alternatives. The saving only exists **jointly**, which is the
/// whole reason the joint formulation is interesting. This pass evaluates candidate instances
/// against the *sum* of the two classes' current realisations and commits when the pair is cheaper.
fn multioutput_pair_pass(
    dense: &Dense,
    matches: &[Match],
    by_out: &[Vec<usize>],
    required: &[(usize, bool)],
    is_free: &[bool],
    n_classes: usize,
    w: &[f64],
    pick: &mut [Option<usize>],
    rounds: usize,
) {
    let mut cands: Vec<usize> = (0..matches.len())
        .filter(|&i| !matches[i].extra_outs.is_empty())
        .collect();
    // deepest first: committing an inner pair can only help its consumers
    cands.sort_by_key(|&i| std::cmp::Reverse(matches[i].out_node));
    if cands.is_empty() {
        return;
    }
    fill_missing(matches, by_out, required, is_free, n_classes, w, pick);
    prune_reachable(pick, matches, required, is_free, n_classes);

    let dbg = std::env::var_os("EMAPPER_DEBUG_MO").is_some();
    let is_req = {
        let mut v = vec![false; n_classes];
        for &(n, g) in required {
            let c = class_of(n, g);
            if !is_free[c] {
                v[c] = true;
            }
        }
        v
    };

    /* Joint selection by **incremental area delta**, the same discipline as
    `exact_area_pass_fast`.
    The previous version gated every candidate on `m.area < pick[c1].area + pick[c2].area` before
    running a global trial. That local test is wrong for exactly the pairs that matter: our cover
    often realises an adder's `CON` cheaply by exploiting reconvergence — e.g. `CON` as a 2-input
    `AND2` over two already-materialised internal nodes, at 0.09 against `FAx1`'s 0.28 — so the
    locally-dearer joint cell is rejected even though switching to it **orphans the whole cone**
    those two internal nodes needed. Only the global delta can see that, so it is now computed for
    every candidate whose two outputs are already live, and the global `fill_missing` +
    `prune_reachable` + `cover_complete` + `pick_area` that used to settle each candidate is
    replaced by reference-counted bookkeeping. */
    let mut st = inc_state_init(matches, pick, required, is_free, n_classes);
    let mut area = pick_area(matches, pick);
    let mut n_both_live = 0usize;
    let mut n_committed = 0usize;
    let mut n_broken = 0usize;
    let mut eval_log: Vec<(f32, usize, usize, Istr, f32, f32)> = Vec::new();
    let mut best_area_seen = area;
    let mut best_pick_seen = st.pick.clone();

    for _ in 0..rounds {
        let mut improved = false;
        /* Best-first: order the live candidates by their local saving. Committing the most promising
        swaps first matters more than the size of the pool — adding more (weaker) candidates while
        keeping the arbitrary node-depth order made results *worse*, because a mediocre swap taken
        early blocks several good ones. The local gap is only an ordering key here, never a filter:
        as a filter it rejects exactly the pairs whose value is the cone they orphan. */
        let mut order: Vec<(f32, usize)> = Vec::new();
        for &mi in &cands {
            let m = &matches[mi];
            let c1 = class_of(m.out_node, m.out_neg);
            let Some(o) = m.extra_outs.first() else {
                continue;
            };
            let c2 = class_of(o.node, o.neg);
            if c1 == c2 {
                continue;
            }
            /* Both roots must already be realised. Allowing a missing root was measured: with a
            small candidate window it is *worse* (it admits pairs whose repair re-materialises the
            root, derailing the trajectory), and with a window large enough to be neutral it buys
            0.2% for 2.5x the runtime. Keeping the restriction is both simpler and better. */
            let (Some(o1), Some(o2)) = (st.pick[c1], st.pick[c2]) else {
                continue;
            };
            if o1 == mi && o2 == mi {
                continue;
            }
            let cur = if o1 == o2 {
                matches[o1].area
            } else {
                matches[o1].area + matches[o2].area
            };
            order.push((m.area - cur, mi));
        }
        order.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        for &(_, mi) in &order {
            let m = &matches[mi];
            let c1 = class_of(m.out_node, m.out_neg);
            let Some(o) = m.extra_outs.first() else {
                continue;
            };
            let c2 = class_of(o.node, o.neg);
            let (Some(o1), Some(o2)) = (st.pick[c1], st.pick[c2]) else {
                continue;
            };
            if o1 == mi && o2 == mi {
                continue;
            }
            n_both_live += 1;

            let mut trial = st.clone();
            /* Install the shared instance first, so a leaf it has in common with either outgoing
            realisation is already ref-protected and is not transiently freed and re-realised
            through a different match — the bug that cost the exact pass 2.6%. */
            let added = inc_add_match_body(c1, mi, matches, by_out, is_free, w, &mut trial);
            // `inc_add_match_body` only fills the extra output when it is None; make it explicit.
            trial.pick[c2] = Some(mi);
            let mut freed = 0.0;
            if o1 != mi {
                freed += inc_free_match_at(o1, matches, is_free, &is_req, &mut trial);
            }
            if o2 != mi && o2 != o1 {
                freed += inc_free_match_at(o2, matches, is_free, &is_req, &mut trial);
            }
            let delta = added - freed;
            /* No incremental filter here. The reference-counted delta is unreliable in *both*
            directions on shared cones — it missed the globally-profitable swap above, and it
            over-reports on deep cones — so every candidate whose two outputs are live is settled
            with the repaired cover. `delta` is kept for reporting only. */
            let _ = delta;
            {
                /* Repair before accepting. The reference-counted removal can drop an edge that is
                still live (the drift is rare and only shows up on deep shared cones), and an
                incomplete cover makes the whole solve fall back to greedy-only. Repairing only on
                the commits that are actually taken keeps the cost at O(n_classes) per *commit*
                instead of per candidate, and rebuilding the state there also resets any drift
                the incremental bookkeeping has accumulated. */
                let mut cand = trial.pick.clone();
                fill_missing(matches, by_out, required, is_free, n_classes, w, &mut cand);
                prune_reachable(&mut cand, matches, required, is_free, n_classes);
                /* A complete cover is not necessarily an *acyclic* one: the repair re-selects
                realisations from `by_out`, which can reintroduce a class into its own cone. That
                is a latch in the netlist, and CEC rejects it. */
                let cyclic = first_cycle(matches, &cand).is_some();
                if cyclic || !cover_complete(matches, &cand, required, is_free) {
                    if dbg && n_broken <= 3 {
                        n_broken += 1;
                        eprintln!(
                            "    unrepairable commit c1={c1} c2={c2} cell={} added={added:.3} freed={freed:.3} delta={delta:+.3}",
                            m.cell_name
                        );
                    }
                    continue;
                }
                /* The incremental delta is only a *filter* here: the cross-check above showed it can
                be wildly optimistic on deep shared cones (it claimed -1.12 where the repaired
                cover was 2.63 worse). Acceptance is therefore on the repaired area, so the pass
                stays monotone. */
                let new_area = pick_area(matches, &cand);
                if dbg {
                    let cur_a = if o1 == o2 {
                        matches[o1].area
                    } else {
                        matches[o1].area + matches[o2].area
                    };
                    eval_log.push((new_area - area, c1, c2, m.cell_name.clone(), cur_a, m.area));
                }
                /* Accept a plateau move as well: converting a pair to a shared instance can leave the
                total unchanged while removing cells that block the *next* conversion. */
                let force = std::env::var_os("ILP_MO_FORCE").is_some();
                let tol = if std::env::var_os("ILP_MO_TIES").is_some() {
                    1e-6
                } else {
                    -1e-6
                };
                if !force && new_area >= area + tol {
                    continue;
                }
                st = inc_state_init(matches, &cand, required, is_free, n_classes);
                area = new_area;
                improved = true;
                n_committed += 1;
                if area < best_area_seen {
                    best_area_seen = area;
                    best_pick_seen = st.pick.clone();
                }
                if dbg && n_committed <= 6 {
                    eprintln!(
                        "    commit: c1={c1} c2={c2} cell={} m.area={:.3} added={added:.3} freed={freed:.3} delta={delta:+.3} area={area:.3}",
                        m.cell_name, m.area
                    );
                }
            }
        }
        if !improved {
            break;
        }
        area = pick_area(matches, &st.pick);
    }

    /* ---- phase 2: joint pairs ------------------------------------------------------------------
    A single conversion can fail to pay on its own while two of them pay together, because the two
    together orphan a shared cone that neither orphans alone. The 1-opt pass above cannot see that.
    Take the most promising both-live candidates by local gap and settle every *pair* of them with
    the same repaired-area test, committing the best and repeating. */
    if std::env::var_os("ILP_MO_PAIRS").is_some() {
        let width: usize = std::env::var("ILP_MO_PAIR_WIDTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(24);
        for _sweep in 0..8 {
            let mut scored: Vec<(f32, usize)> = Vec::new();
            for &mi in &cands {
                let m = &matches[mi];
                let c1 = class_of(m.out_node, m.out_neg);
                let Some(o) = m.extra_outs.first() else {
                    continue;
                };
                let c2 = class_of(o.node, o.neg);
                let (Some(o1), Some(o2)) = (st.pick[c1], st.pick[c2]) else {
                    continue;
                };
                if o1 == mi && o2 == mi {
                    continue;
                }
                let cur = if o1 == o2 {
                    matches[o1].area
                } else {
                    matches[o1].area + matches[o2].area
                };
                scored.push((m.area - cur, mi));
            }
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(width);
            let mut best_here: Option<(f32, Vec<Option<usize>>)> = None;
            for a in 0..scored.len() {
                for b in (a + 1)..scored.len() {
                    let mut cand = st.pick.clone();
                    for &(_, mi) in [&scored[a], &scored[b]] {
                        let m = &matches[mi];
                        cand[class_of(m.out_node, m.out_neg)] = Some(mi);
                        for o in &m.extra_outs {
                            cand[class_of(o.node, o.neg)] = Some(mi);
                        }
                    }
                    fill_missing(matches, by_out, required, is_free, n_classes, w, &mut cand);
                    prune_reachable(&mut cand, matches, required, is_free, n_classes);
                    if !cover_complete(matches, &cand, required, is_free)
                        || first_cycle(matches, &cand).is_some()
                    {
                        continue;
                    }
                    let na = pick_area(matches, &cand);
                    if best_here.as_ref().map_or(true, |(v, _)| na < *v - 1e-6) {
                        best_here = Some((na, cand));
                    }
                }
            }
            match best_here {
                Some((na, cand)) if na < area - 1e-6 => {
                    st = inc_state_init(matches, &cand, required, is_free, n_classes);
                    area = na;
                    n_committed += 2;
                    if area < best_area_seen {
                        best_area_seen = area;
                        best_pick_seen = st.pick.clone();
                    }
                }
                _ => break,
            }
        }
    }

    if std::env::var_os("ILP_MO_FORCE").is_some() {
        st.pick = best_pick_seen;
    }
    pick.copy_from_slice(&st.pick);
    if dbg {
        eval_log.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("  best 12 rejected/committed candidates by (new_area - area):");
        for (imp, c1, c2, cell, cur_a, m_a) in eval_log.iter().take(12) {
            eprintln!("    d={imp:+.4} c1={c1} c2={c2} {cell}  cur={cur_a:.3} m={m_a:.3}");
        }
        eprintln!(
            "mo-pair: cands={} live_classes={} both_live={} committed={} area={}",
            cands.len(),
            pick.iter().filter(|p| p.is_some()).count(),
            n_both_live,
            n_committed,
            pick_area(matches, pick)
        );
    }
}

/// Stage-S4 cover: area flow followed by coordinate descent on the **DAG** objective.
///
/// The first pass ranks by tree cost, which over-counts shared sub-graphs. Each later round re-prices
/// with every already-realised class at cost zero and re-selects from the primary outputs, so reusing
/// a net that is already materialised is free — that is what turns a tree-cost ranking into a DAG
/// cover. Used as the CBC MIP start and as the fallback when the MIP returns nothing integral.
#[must_use]
fn ps_pairs_enabled() -> bool {
    std::env::var_os("ILP_EXACT_PAIRS").is_some()
}

fn greedy_cover(
    dense: &Dense,
    matches: &[Match],
    required: &[(usize, bool)],
    is_free: &[bool],
    n_classes: usize,
    rounds: usize,
    w: &[f64],
    exact_rounds: usize,
    est_refs: &mut [f32],
) -> Vec<usize> {
    let mut by_out: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, m) in matches.iter().enumerate() {
        by_out[class_of(m.out_node, m.out_neg)].push(i);
        /* A shared instance realises **all** of its output pins, so its extra outputs index this
        match too. Omitting them made a class whose only surviving realisation is an extra pin
        look unrealisable, and since `greedy_cover` is the mandatory fallback the whole solve then
        failed with "neither CBC nor the greedy pass produced a complete cover" — this is what
        broke `multiplier`, `log2` and `sin` under the default match cap. Every other `by_out`
        builder in this file already includes them. */
        for o in &m.extra_outs {
            by_out[class_of(o.node, o.neg)].push(i);
        }
    }

    let none: Vec<Option<usize>> = vec![None; n_classes];
    /* emap shares a multi-output gate's area across both of its roots when it prices the flow.
    Charging it in full to each root hides the sharing, so the flow never selects shared instances
    and the decision has to be recovered later by the pair pass. Gated for A/B measurement. */
    let share_mo = std::env::var_os("ILP_MO_FLOW_SHARE").is_some();
    /* Starting-point diversity. The flow is a coordinate descent whose fixed point depends on where
    it starts, and on the large circuits where CBC cannot run at all (`used_greedy`) that fixed
    point *is* the answer. `ILP_START=delay` seeds it from the delay-optimal cover instead of from
    scratch, marking the whole delay cover as "already realised" for the first marginal pricing —
    a completely different basin from the tree-cost start. */
    let delay_seed: Option<Vec<Option<usize>>> =
        if std::env::var("ILP_START").as_deref() == Ok("delay") {
            let arr_min = arrival_flow(dense, matches, &by_out, is_free, n_classes);
            let mut s = select_delay(matches, &by_out, required, is_free, &arr_min, n_classes);
            fill_missing(matches, &by_out, required, is_free, n_classes, w, &mut s);
            prune_reachable(&mut s, matches, required, is_free, n_classes);
            if cover_complete(matches, &s, required, is_free) {
                Some(s)
            } else {
                None
            }
        } else {
            None
        };
    let start_ref = delay_seed.as_ref().unwrap_or(&none);
    let flow = marginal_cost(
        dense, matches, &by_out, start_ref, is_free, n_classes, w, est_refs, share_mo,
    );
    let mut pick = select_by_cost(
        matches, &by_out, required, is_free, &flow, w, n_classes, share_mo,
    );
    repair_inverter_cycles(matches, &by_out, &flow, w, dense, n_classes, &mut pick);
    fill_missing(matches, &by_out, required, is_free, n_classes, w, &mut pick);
    prune_reachable(&mut pick, matches, required, is_free, n_classes);
    let mut best_area = pick_area(matches, &pick);
    let mut best_pick = pick.clone();
    if let Some(s) = &delay_seed {
        let sa = pick_area(matches, s);
        if sa < best_area {
            best_area = sa;
            best_pick = s.clone();
            pick = s.clone();
        }
    }

    for iter in 0..rounds {
        /* Refine the reference estimate from the cover actually produced, exactly as mockturtle
        emap does (`est_refs = max(1, coef*est_refs + (1-coef)*map_refs)`, `coef = 1/(i+1)^2`).
        A single static fanout estimate leaves the flow partly blind to real sharing. */
        {
            let refs = cover_refs(matches, &pick, required, is_free, n_classes);
            let coef = 1.0 / ((iter as f32 + 1.0) * (iter as f32 + 1.0));
            for c in 0..n_classes {
                est_refs[c] = (coef * est_refs[c] + (1.0 - coef) * refs[c]).max(1.0);
            }
        }
        let marg = marginal_cost(
            dense, matches, &by_out, &pick, is_free, n_classes, w, est_refs, share_mo,
        );
        let mut cand = select_by_cost(
            matches, &by_out, required, is_free, &marg, w, n_classes, share_mo,
        );
        repair_inverter_cycles(matches, &by_out, &marg, w, dense, n_classes, &mut cand);
        fill_missing(matches, &by_out, required, is_free, n_classes, w, &mut cand);
        prune_reachable(&mut cand, matches, required, is_free, n_classes);
        let area = pick_area(matches, &cand);
        if area < best_area {
            best_area = area;
            best_pick = cand.clone();
        }
        if cand == pick {
            break;
        }
        pick = cand;
    }

    /* Area flow is a tree cost, so it over-prices wide cuts — every leaf is charged even when it is
    shared and therefore already paid for. The exact pass repairs that by measuring the real cover
    area after each swap (emap's `ela_rounds`). */
    /* Order matters: the exact pass rewrites single-output realisations and knows nothing about
    shared instances, so the joint pair pass runs **after** it to re-establish them, then the
    descent gets one more chance to exploit the freed area. */
    /* The ref-counted pass used to settle ~2.6% worse than the original "clone, fill, prune,
    re-measure" pass; that gap was a free-then-add ordering bug (see `inc_free_match_at`) and is
    gone. It is now bit-identical on ISCAS/EPFL while being up to 3x faster (c5315 26.5 s ->
    7.6 s, multiplier 465 s -> 19 s), so it is the default. `EMAPPER_EXACT_OLD=1` keeps the
    original available as a reference implementation. */
    let use_mo_pair = std::env::var_os("ILP_MO_NO_PAIR").is_none();
    let has_mo = use_mo_pair && matches.iter().any(|m| !m.extra_outs.is_empty());
    let mo_rounds: usize = std::env::var("ILP_MO_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let mut mo_pass = |pick: &mut Vec<Option<usize>>, w: &[f64]| {
        multioutput_pair_pass(
            dense, matches, &by_out, required, is_free, n_classes, w, pick, mo_rounds,
        );
        prune_reachable(pick, matches, required, is_free, n_classes);
    };
    if ps_pairs_enabled() {
        let width: usize = std::env::var("ILP_EXACT_PAIR_WIDTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(24);
        let rounds: usize = std::env::var("ILP_EXACT_PAIRS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        exact_pairs_pass(
            dense,
            matches,
            &by_out,
            required,
            is_free,
            n_classes,
            w,
            &mut best_pick,
            width,
            rounds,
        );
        best_area = pick_area(matches, &best_pick);
    }
    /* Joint multi-output selection runs after the exact pass. Running it on both sides of the exact
    pass as well was measured: identical on adder and 0.01 worse on multiplier for 1.5x the runtime,
    so the pre-pass is not kept. */
    /* Randomised multi-start. The cover is provably optimal within local neighbourhoods (1-opt,
    2-opt, and exact ILP over 1500-node windows), so the only way to test whether it is *globally*
    optimal for this match set is to leave the basin: perturb a random slice of realisations, run
    the same descent, and keep the best. `ILP_SHAKE=<restarts>[,<classes-per-shake>]`. */
    if let Ok(spec) = std::env::var("ILP_SHAKE") {
        let nums: Vec<usize> = spec
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .collect();
        let restarts: usize = nums.first().copied().unwrap_or(8);
        /* The remaining numbers are the perturbation strengths, cycled across restarts. Small
        perturbations refine inside a basin, large ones jump between basins; a run that only uses
        one size leaves most of the space unvisited. */
        let sizes: Vec<usize> = if nums.len() > 1 {
            nums[1..].to_vec()
        } else {
            vec![40]
        };
        let mut rng: u64 = 0x9E3779B97F4A7C15;
        let mut besta = pick_area(matches, &best_pick);
        for r in 0..restarts {
            let per = sizes[r % sizes.len()].max(1);
            let mut p = best_pick.clone();
            for _ in 0..per {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let c = (rng as usize) % n_classes;
                if is_free[c] {
                    continue;
                }
                let opts = &by_out[c];
                if opts.is_empty() {
                    continue;
                }
                p[c] = Some(opts[(rng >> 20) as usize % opts.len()]);
            }
            fill_missing(matches, &by_out, required, is_free, n_classes, w, &mut p);
            prune_reachable(&mut p, matches, required, is_free, n_classes);
            if !cover_complete(matches, &p, required, is_free) || first_cycle(matches, &p).is_some()
            {
                continue;
            }
            if exact_rounds > 0 {
                exact_area_pass_fast(
                    dense,
                    matches,
                    &by_out,
                    required,
                    is_free,
                    n_classes,
                    w,
                    &mut p,
                    exact_rounds,
                );
            }
            if has_mo {
                mo_pass(&mut p, w);
            }
            let a = pick_area(matches, &p);
            if a < besta - 1e-6 {
                besta = a;
                best_pick = p;
                if std::env::var_os("EMAPPER_DEBUG_SHAKE").is_some() {
                    eprintln!("  shake: improved to {a:.3}");
                }
            }
        }
        best_area = besta;
    }

    let use_fast_exact = std::env::var_os("EMAPPER_EXACT_OLD").is_none();
    if exact_rounds > 0 {
        if !use_fast_exact {
            exact_area_pass(
                dense,
                matches,
                &by_out,
                required,
                is_free,
                n_classes,
                w,
                &mut best_pick,
                exact_rounds,
            );
        } else {
            exact_area_pass_fast(
                dense,
                matches,
                &by_out,
                required,
                is_free,
                n_classes,
                w,
                &mut best_pick,
                exact_rounds,
            );
        }
    }
    if has_mo {
        mo_pass(&mut best_pick, w);
        best_area = pick_area(matches, &best_pick);
    }

    /* Restricted-neighbourhood ILP refinement: the only lever that can still move the large
    circuits, where CBC cannot solve the full model and the cover is therefore greedy-only. */
    if let Ok(spec) = std::env::var("ILP_LNS") {
        let parts: Vec<f64> = spec
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect();
        let wins = parts.first().copied().unwrap_or(8.0).max(1.0) as usize;
        let nodes = parts.get(1).copied().unwrap_or(400.0).max(4.0) as usize;
        let secs = parts.get(2).copied().unwrap_or(5.0).max(1.0);
        let dbg = std::env::var_os("EMAPPER_DEBUG_LNS").is_some();
        for _ in 0..3 {
            if !lns_pass(
                dense,
                matches,
                &by_out,
                required,
                is_free,
                n_classes,
                w,
                &mut best_pick,
                wins,
                nodes,
                secs,
                dbg,
            ) {
                break;
            }
            best_area = pick_area(matches, &best_pick);
        }
    }

    let mut is_picked = vec![false; matches.len()];
    for p in best_pick.iter().flatten() {
        is_picked[*p] = true;
    }
    (0..matches.len()).filter(|&i| is_picked[i]).collect()
}

/// Arrival / depth of a chosen cover, computed directly from the matches instead of from model
/// columns (which are absent in an area-only solve).
///
/// Must be run over nodes in topological order: within one node the non-inverter realisations are
/// resolved first, then the inverter realisation, which is the only same-node edge.
fn evaluate_cover(
    dense: &Dense,
    matches: &[Match],
    chosen: &[Option<usize>],
    is_free: &[bool],
    n_classes: usize,
    inv_delay: f32,
) -> (Vec<f32>, Vec<u32>) {
    let mut arr = vec![0.0f32; n_classes];
    let mut lvl = vec![0u32; n_classes];
    let is_inv = |m: &Match, d: usize| m.leaves.len() == 1 && m.leaves[0] == d;
    for &d in &dense.order {
        for pass_inverters in [false, true] {
            for phase in [false, true] {
                let c = class_of(d, phase);
                if is_free[c] {
                    continue;
                }
                let Some(mi) = chosen[c] else { continue };
                let m = &matches[mi];
                if is_inv(m, d) != pass_inverters {
                    continue;
                }
                if pass_inverters {
                    let ic = class_of(d, !phase);
                    arr[c] = arr[ic] + m.pin_delay[0];
                    lvl[c] = lvl[ic] + 1;
                } else {
                    let mut a = 0.0f32;
                    let mut l = 0u32;
                    for (p, &leaf) in m.leaves.iter().enumerate() {
                        let ic = class_of(leaf, m.leaf_neg[p]);
                        a = a.max(arr[ic] + m.pin_delay[p]);
                        l = l.max(lvl[ic] + 1);
                    }
                    if out_inv_of(m, c) == Some(true) {
                        a += inv_delay;
                    }
                    arr[c] = a;
                    lvl[c] = l;
                }
            }
        }
    }
    (arr, lvl)
}

/// Whether every class reachable from the required set has a chosen realisation.
fn cover_complete(
    matches: &[Match],
    chosen: &[Option<usize>],
    required: &[(usize, bool)],
    is_free: &[bool],
) -> bool {
    let mut seen: HashSet<usize> = HashSet::new();
    let mut stack: Vec<usize> = required
        .iter()
        .map(|&(n, g)| class_of(n, g))
        .filter(|&c| !is_free[c])
        .collect();
    while let Some(c) = stack.pop() {
        if !seen.insert(c) {
            continue;
        }
        let Some(mi) = chosen[c] else {
            return false;
        };
        let m = &matches[mi];
        for (p, &leaf) in m.leaves.iter().enumerate() {
            let ic = class_of(leaf, m.leaf_neg[p]);
            if !is_free[ic] {
                stack.push(ic);
            }
        }
    }
    true
}

/// Return some class on a cycle of the chosen cover, if one exists.
fn first_cycle(matches: &[Match], chosen: &[Option<usize>]) -> Option<usize> {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        White,
        Grey,
        Black,
    }
    let n = chosen.len();
    let mut mark = vec![Mark::White; n];
    // Iterative DFS over the class graph induced by the chosen matches.
    for root in 0..n {
        if chosen[root].is_none() || mark[root] != Mark::White {
            continue;
        }
        let mut stack: Vec<(usize, usize)> = vec![(root, 0)];
        mark[root] = Mark::Grey;
        while let Some(&mut (c, ref mut idx)) = stack.last_mut() {
            let mi = chosen[c]?;
            let m = &matches[mi];
            if *idx < m.leaves.len() {
                let p = *idx;
                *idx += 1;
                let ic = class_of(m.leaves[p], m.leaf_neg[p]);
                if chosen[ic].is_none() {
                    continue;
                }
                match mark[ic] {
                    Mark::Grey => return Some(ic),
                    Mark::White => {
                        mark[ic] = Mark::Grey;
                        stack.push((ic, 0));
                    }
                    Mark::Black => {}
                }
            } else {
                mark[c] = Mark::Black;
                stack.pop();
            }
        }
    }
    None
}

/// Exact don't-care measurement over cuts.
///
/// For a cut with leaves `L`, the *care set* is the image of the leaf functions over the primary
/// input space; a library cell `G` may legally implement the cut's function `F` whenever `F` and `G`
/// agree on that image. The image is a strict subset of `B^|L|` exactly when the leaves reconverge.
///
/// This computes the image **exactly** whenever the leaves' joint primary-input support fits in
/// `support_limit` variables, by bit-parallel simulation over all `2^m` assignments to that support.
/// Primary inputs outside the joint support cannot influence the leaves, so fixing them to zero is
/// exact, not an approximation.
///
/// The point of the measurement is to decide whether building care-set-aware matching (stage S0) is
/// worth the SAT machinery it needs: if `unreachable_points` is near zero, it is not.
#[derive(Clone, Debug, Default)]
pub struct DcStats {
    pub cuts_total: usize,
    pub cuts_measured: usize,
    pub cuts_with_dc: usize,
    pub unreachable_points: u64,
    pub total_points: u64,
    /// `support_hist[m]` = number of cuts whose leaves' joint support has popcount `m`.
    pub support_hist: Vec<usize>,
    /// Cuts where some library cell becomes legal **only** thanks to the don't-care set, and that
    /// cell is cheaper than every cell matching the function exactly. This is the actual headroom of
    /// stage S0 — a non-empty care set by itself proves nothing.
    pub cuts_dc_helps: usize,
    pub area_exact: f32,
    pub area_with_dc: f32,
}

#[must_use]
pub fn measure_cut_dc(
    aig: &Aig,
    tech: &TechLib,
    ps: &IlpParams,
    support_limit: u32,
    cone_limit: usize,
) -> DcStats {
    let dense = Dense::build(aig);
    let n_nodes = dense.len();

    let mut pi_var: HashMap<usize, usize> = HashMap::new();
    let mut n_pi = 0usize;
    for d in 0..n_nodes {
        if dense.is_pi[d] {
            pi_var.insert(d, n_pi);
            n_pi += 1;
        }
    }
    let sw = n_pi.div_ceil(64).max(1);

    // leaves' joint support, as a bitset over primary inputs
    let mut support: Vec<Vec<u64>> = vec![vec![0u64; sw]; n_nodes];
    for &d in &dense.order {
        match aig.node_type(dense.node(d)) {
            AigNode::Input(_) => {
                let i = pi_var[&d];
                support[d][i / 64] |= 1u64 << (i % 64);
            }
            AigNode::And => {
                let (e1, e2) = aig.try_unwrap_and(dense.node(d)).expect("and");
                let a = dense.dense(aig.edge_source(e1));
                let b = dense.dense(aig.edge_source(e2));
                let mut s = vec![0u64; sw];
                for w in 0..sw {
                    s[w] = support[a][w] | support[b][w];
                }
                support[d] = s;
            }
            _ => {}
        }
    }

    let cuts = enumerate_cuts(aig, &dense, ps.max_cut_leaves, ps.max_cuts_per_node);
    let mut st = DcStats {
        support_hist: vec![0; n_pi + 1],
        ..Default::default()
    };

    for &d in &dense.order {
        if dense.is_pi[d] || dense.is_zero[d] {
            continue;
        }
        for cut in &cuts[d] {
            if cut.len() < 2 || (cut.len() == 1 && cut[0] == d) {
                continue;
            }
            let k = cut.len();
            st.cuts_total += 1;

            let mut sup = vec![0u64; sw];
            for &l in cut {
                for w in 0..sw {
                    sup[w] |= support[l][w];
                }
            }
            let m: u32 = sup.iter().map(|w| w.count_ones()).sum();
            st.support_hist[m as usize] += 1;
            if m > support_limit {
                continue;
            }
            st.cuts_measured += 1;

            // primary inputs of the support, in variable order
            let mut vars: Vec<usize> = Vec::with_capacity(m as usize);
            for (w, &word) in sup.iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let b = bits.trailing_zeros();
                    vars.push(w * 64 + b as usize);
                    bits &= bits - 1;
                }
            }

            let bits = 1usize << m;
            let words = bits.div_ceil(64);
            // bit j of variable i is (j >> i) & 1
            /* bit j of variable i is `(j >> i) & 1`; it is periodic with period `2^(i+1)` in j,
            so each word is either all-zeros, all-ones, or a fixed 64-bit pattern. Getting this
            wrong costs a factor of 64 in the hottest loop. */
            let pattern = |i: usize| -> Vec<u64> {
                let mut v = vec![0u64; words];
                if i >= 6 {
                    let period = 1usize << (i - 6);
                    for (w, slot) in v.iter_mut().enumerate() {
                        *slot = if (w / period) % 2 == 1 { u64::MAX } else { 0 };
                    }
                } else {
                    let half = 1usize << i;
                    let mut base = 0u64;
                    for b in 0..64 {
                        if ((b / half) % 2) == 1 {
                            base |= 1u64 << b;
                        }
                    }
                    for slot in v.iter_mut() {
                        *slot = base;
                    }
                }
                v
            };

            /* The cone must extend down to the primary inputs: the leaves are *functions* of the
            PIs, so their values have to be simulated rather than treated as free variables.
            Primaries outside the leaves' joint support are fixed to zero, which is exact because
            they cannot influence the leaves. */
            let mut cone: std::collections::HashSet<usize> = std::collections::HashSet::new();
            let mut stack = vec![d];
            while let Some(x) = stack.pop() {
                if !cone.insert(x) {
                    continue;
                }
                if dense.is_pi[x] || dense.is_zero[x] {
                    continue;
                }
                let Some((e1, e2)) = aig.try_unwrap_and(dense.node(x)) else {
                    continue;
                };
                stack.push(dense.dense(aig.edge_source(e1)));
                stack.push(dense.dense(aig.edge_source(e2)));
            }
            if cone.len() > cone_limit {
                continue;
            }

            let mut vals: HashMap<usize, Vec<u64>> = HashMap::new();
            let mut var_of: HashMap<usize, usize> = HashMap::new();
            for (i, &p) in vars.iter().enumerate() {
                var_of.insert(p, i);
            }
            for &x in &dense.order {
                if !cone.contains(&x) {
                    continue;
                }
                let node = dense.node(x);
                let v = match aig.node_type(node) {
                    AigNode::Zero => vec![0u64; words],
                    AigNode::Input(_) => match var_of.get(&pi_var[&x]) {
                        Some(&i) => pattern(i),
                        None => vec![0u64; words],
                    },
                    AigNode::And => {
                        let (e1, e2) = aig.try_unwrap_and(node).expect("and");
                        let a = dense.dense(aig.edge_source(e1));
                        let b = dense.dense(aig.edge_source(e2));
                        let (ia, ib) = (aig.is_edge_inverted(e1), aig.is_edge_inverted(e2));
                        let mut o = vec![0u64; words];
                        let va = &vals[&a];
                        let vb = &vals[&b];
                        for w in 0..words {
                            let xv = if ia { !va[w] } else { va[w] };
                            let yv = if ib { !vb[w] } else { vb[w] };
                            o[w] = xv & yv;
                        }
                        o
                    }
                    AigNode::Output(_) => unreachable!(),
                };
                vals.insert(x, v);
            }

            let mut seen = vec![false; 1usize << k];
            for j in 0..bits {
                let (w, b) = (j / 64, j % 64);
                let mut tuple = 0usize;
                for (i, &l) in cut.iter().enumerate() {
                    if (vals[&l][w] >> b) & 1 == 1 {
                        tuple |= 1 << i;
                    }
                }
                seen[tuple] = true;
            }
            let img = seen.iter().filter(|s| **s).count() as u64;
            let all = 1u64 << k;
            st.total_points += all;
            if img < all {
                st.cuts_with_dc += 1;
                st.unreachable_points += all - img;
            }

            /* The decisive number: does the don't-care set make a *cheaper* cell legal? */
            let Some(raw) = cut_tt(
                aig,
                dense.node(d),
                &cut.iter().map(|&x| dense.node(x)).collect::<Vec<_>>(),
                k,
            ) else {
                continue;
            };
            let f = tt9_from_u64(raw);
            let mut care = tt9_zero();
            for (t, &r) in seen.iter().enumerate() {
                if r {
                    tt9_bit_set(&mut care, t);
                }
            }
            let exact = tech.candidates(k as u8, f).and_then(|c| {
                c.iter()
                    .filter(|v| !v.out_neg)
                    .map(|v| v.area)
                    .reduce(f32::min)
            });
            let with_dc = tech
                .index
                .iter()
                .filter(|((n, _), _)| *n == k as u8)
                .flat_map(|((_, vtt), vs)| {
                    // legal iff (F xor G) & care == 0
                    let legal = (0..(1usize << k).min(512)).all(|b| {
                        tt9_bit_get(&f, b) == tt9_bit_get(vtt, b) || !tt9_bit_get(&care, b)
                    });
                    if legal {
                        vs.iter()
                            .filter(|v| !v.out_neg)
                            .map(|v| v.area)
                            .reduce(f32::min)
                    } else {
                        None
                    }
                })
                .reduce(f32::min);
            if exact.is_some() {
                st.area_exact += exact.expect("checked");
                if let Some(wd) = with_dc {
                    st.area_with_dc += wd;
                    if wd < exact.expect("checked") - 1e-6 {
                        st.cuts_dc_helps += 1;
                    }
                } else {
                    st.area_with_dc += exact.expect("checked");
                }
            }
        }
    }
    st
}

/// Multi-output payload measurement.
///
/// A multi-output cell instance produces several output functions from **one** set of input pins and
/// is charged once. Whether that is worth anything depends on two things being true at once:
/// some pair of subject-graph nodes must share a small cut, and the two cut functions must land on
/// two outputs of the same cell with a **consistent pin permutation and inversion**.
///
/// This counts that, before any joint-extraction machinery is built for it.
#[derive(Clone, Debug, Default)]
pub struct MoPayload {
    pub nodes: usize,
    pub leaf_sets_with_pairs: usize,
    pub pairs: usize,
    /// pairs where some multi-output cell realises both cut functions together
    pub pairs_available: usize,
    /// of those, pairs where the shared instance is cheaper than the best *single* cells for the two
    /// functions separately (unrealisable single cells count as "wins", so this over-states)
    pub pairs_cheaper: usize,
    pub saving_total: f32,
    pub best_saving: f32,
    /// pairs where the two functions have no single-cell implementation at the cell's arity
    pub pairs_with_unrealisable_half: usize,
}

fn mo_groups(tech: &TechLib) -> HashMap<String, Vec<String>> {
    let mut g: HashMap<String, Vec<String>> = HashMap::new();
    for c in &tech.cells {
        let e = g.entry(c.name.clone()).or_default();
        if !e.contains(&c.out_pin) {
            e.push(c.out_pin.clone());
        }
    }
    g.retain(|_, v| v.len() >= 2);
    g
}

fn best_single(tech: &TechLib, k: u8, tt: Tt9) -> f32 {
    tech.candidates(k, tt)
        .and_then(|c| {
            c.iter()
                .filter(|v| !v.out_neg)
                .map(|v| v.area)
                .reduce(f32::min)
        })
        .unwrap_or(f32::INFINITY)
}

/// Pin keys `(perm, neg_mask, out_neg)` under which `pin` realises `tt` at arity `k`.
///
/// Unlike a single-output match, a *shared* instance may take the output inverter on either pin
/// independently — you cannot fold it into the instance, but you can put one inverter after it. So
/// `out_neg` is part of the key rather than being filtered away, and the caller charges `inv_area`
/// for each negated output.
type PinKey = (Vec<u8>, u16, bool);
fn pin_keys(
    tech: &TechLib,
    name: &str,
    pin: &str,
    k: u8,
    tt: Tt9,
) -> std::collections::HashSet<PinKey> {
    tech.candidates(k, tt)
        .map(|c| {
            c.iter()
                .filter(|v| v.cell_name == name && v.out_pin == pin)
                .map(|v| (v.perm.clone(), v.neg_mask, v.out_neg))
                .collect()
        })
        .unwrap_or_default()
}

#[must_use]
pub fn measure_multioutput_payload(
    aig: &Aig,
    tech: &TechLib,
    ps: &IlpParams,
    max_leaves: usize,
    max_pairs_per_set: usize,
) -> MoPayload {
    let dense = Dense::build(aig);
    let cuts = enumerate_cuts(aig, &dense, max_leaves, 1_000_000);
    let groups = mo_groups(tech);
    let mut st = MoPayload::default();

    // leaf set -> nodes for which it is a cut
    let mut by_leaves: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for &d in &dense.order {
        if dense.is_pi[d] || dense.is_zero[d] {
            continue;
        }
        st.nodes += 1;
        for cut in &cuts[d] {
            if cut.len() < 2 || cut.len() > max_leaves {
                continue;
            }
            by_leaves.entry(cut.clone()).or_default().push(d);
        }
    }

    if std::env::var_os("ILP_MO_DEBUG").is_some() {
        let mut buckets: Vec<(&Vec<usize>, &Vec<usize>)> =
            by_leaves.iter().filter(|(_, n)| n.len() >= 2).collect();
        buckets.sort_by_key(|(l, n)| (std::cmp::Reverse(n.len()), l.len()));
        eprintln!("leaf-set buckets with >=2 nodes: {}", buckets.len());
        let mut xor3 = 0usize;
        let mut maj3i = 0usize;
        let mut both = 0usize;
        for (l, n) in by_leaves.iter().filter(|(_, n)| n.len() >= 2) {
            let k = l.len();
            let idx: Vec<NodeIndex> = l.iter().map(|&x| dense.node(x)).collect();
            let mut has_x = false;
            let mut has_m = false;
            for &d in n {
                if let Some(raw) = cut_tt(aig, dense.node(d), &idx, k) {
                    // XOR3 = 0x96 over 3 leaves; !MAJ3 = 0x17
                    if raw == 0x96 {
                        has_x = true;
                    }
                    if raw == 0x17 {
                        has_m = true;
                    }
                }
            }
            if has_x {
                xor3 += 1;
            }
            if has_m {
                maj3i += 1;
            }
            if has_x && has_m {
                both += 1;
            }
        }
        eprintln!("buckets containing XOR3={xor3} !MAJ3={maj3i} BOTH={both}");
        // how much XOR structure exists at all, by arity
        for kk in 2..=4usize {
            let mut xor2 = 0usize;
            let mut xnor2 = 0usize;
            let mut tot = 0usize;
            for (l, n) in by_leaves.iter().filter(|(l, _)| l.len() == kk) {
                let idx: Vec<NodeIndex> = l.iter().map(|&x| dense.node(x)).collect();
                for &d in n {
                    tot += 1;
                    if let Some(raw) = cut_tt(aig, dense.node(d), &idx, kk) {
                        if kk == 2 && raw == 0x6 {
                            xor2 += 1;
                        }
                        if kk == 2 && raw == 0x9 {
                            xnor2 += 1;
                        }
                    }
                }
            }
            eprintln!("k={kk}: (node,cut) entries={tot}  XOR2={xor2} XNOR2={xnor2}");
        }
        for (l, n) in buckets.iter().take(8) {
            let k = l.len();
            let idx: Vec<NodeIndex> = l.iter().map(|&x| dense.node(x)).collect();
            let mut tts: Vec<String> = Vec::new();
            for &d in n.iter() {
                if let Some(raw) = cut_tt(aig, dense.node(d), &idx, k) {
                    tts.push(format!("{:x}", raw));
                }
            }
            eprintln!("  size={k} nodes={} tt={tts:?}", n.len());
        }
    }
    for (leaves, nodes) in &by_leaves {
        if nodes.len() < 2 {
            continue;
        }
        st.leaf_sets_with_pairs += 1;
        let k = leaves.len();
        let nodes_idx: Vec<NodeIndex> = leaves.iter().map(|&x| dense.node(x)).collect();
        let mut pairs = 0usize;
        'outer: for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                if pairs >= max_pairs_per_set {
                    break 'outer;
                }
                pairs += 1;
                st.pairs += 1;
                let n1 = nodes[i];
                let n2 = nodes[j];
                let (Some(r1), Some(r2)) = (
                    cut_tt(aig, dense.node(n1), &nodes_idx, k),
                    cut_tt(aig, dense.node(n2), &nodes_idx, k),
                ) else {
                    continue;
                };
                let t1 = tt9_from_u64(r1);
                let t2 = tt9_from_u64(r2);
                let s1 = best_single(tech, k as u8, t1);
                let s2 = best_single(tech, k as u8, t2);
                if !s1.is_finite() || !s2.is_finite() {
                    st.pairs_with_unrealisable_half += 1;
                }
                let single = if s1.is_finite() && s2.is_finite() {
                    s1 + s2
                } else {
                    f32::INFINITY
                };
                for (name, pins) in &groups {
                    let area = tech
                        .cells
                        .iter()
                        .find(|c| &c.name == name)
                        .map_or(f32::INFINITY, |c| c.area);
                    let inv = tech.inv_area;
                    // both ordered assignments of the two outputs; the pins must share a
                    // permutation and inversion, but each may take its own output inverter
                    let mut best_cost = f32::INFINITY;
                    for (pa, pb) in [(0usize, 1usize), (1usize, 0usize)] {
                        let Some(oa) = pins.get(pa) else { continue };
                        let Some(ob) = pins.get(pb) else { continue };
                        let ka = pin_keys(tech, name, oa, k as u8, t1);
                        if ka.is_empty() {
                            continue;
                        }
                        let kb = pin_keys(tech, name, ob, k as u8, t2);
                        for (perm, mask, na) in &ka {
                            for (perm2, mask2, nb) in &kb {
                                if perm != perm2 || mask != mask2 {
                                    continue;
                                }
                                let c = area
                                    + if *na { inv } else { 0.0 }
                                    + if *nb { inv } else { 0.0 };
                                if c < best_cost {
                                    best_cost = c;
                                }
                            }
                        }
                    }
                    if best_cost.is_finite() {
                        st.pairs_available += 1;
                        if best_cost < single - 1e-6 {
                            st.pairs_cheaper += 1;
                            let sv = if single.is_finite() {
                                single - best_cost
                            } else {
                                best_cost
                            };
                            st.saving_total += sv;
                            if sv > st.best_saving {
                                st.best_saving = sv;
                            }
                        }
                    }
                }
            }
        }
    }
    st
}

// ---------------------------------------------------------------------------------------------
// netlist emission
// ---------------------------------------------------------------------------------------------

fn sanitize_ident(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 1);
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() || out.as_bytes()[0].is_ascii_digit() {
        out.insert(0, 'n');
    }
    out
}

/// Primary inputs (dense id, port name) and primary outputs (driver dense id, inverted, port name).
pub type Ports = (Vec<(usize, String)>, Vec<(usize, bool, String)>);

#[must_use]
pub fn collect_ports(aig: &Aig, dense: &Dense) -> Ports {
    use itertools::Itertools;
    let pi = aig
        .graph()
        .node_indices()
        .filter_map(|n| match aig.node_type(n) {
            AigNode::Input(var) => Some((var, dense.dense(n))),
            _ => None,
        })
        .sorted_by_key(|(var, _)| *var)
        .enumerate()
        .map(|(k, (var, d))| {
            let name = aig
                .symbol_input(var)
                .cloned()
                .unwrap_or_else(|| format!("$i{k:02}"));
            (d, sanitize_ident(&name))
        })
        .collect::<Vec<_>>();

    let po = aig
        .graph()
        .externals(petgraph::Direction::Outgoing)
        .filter(|n| matches!(aig.node_type(*n), AigNode::Output(_)))
        .filter_map(|o| {
            let pos = match aig.node_type(o) {
                AigNode::Output(i) => i,
                _ => return None,
            };
            let e = aig
                .graph()
                .edges_directed(o, petgraph::Direction::Incoming)
                .next()
                .expect("output driven");
            Some((pos, dense.dense(e.source()), *e.weight()))
        })
        .sorted_by_key(|(pos, _, _)| *pos)
        .enumerate()
        .map(|(k, (pos, driver, inv))| {
            let name = aig
                .symbol_output(pos)
                .cloned()
                .unwrap_or_else(|| format!("$o{k:02}"));
            (driver, inv, sanitize_ident(&name))
        })
        .collect::<Vec<_>>();

    (pi, po)
}

/// Required `(dense node, phase)` classes implied by the primary outputs.
#[must_use]
pub fn required_classes(aig: &Aig) -> (Vec<(usize, bool)>, Ports) {
    let dense = Dense::build(aig);
    let ports = collect_ports(aig, &dense);
    let req = ports
        .1
        .iter()
        .filter(|(d, _, _)| !dense.is_zero[*d])
        .map(|(d, inv, _)| (*d, *inv))
        .collect();
    (req, ports)
}

/// Emit the covered netlist as Verilog.
pub fn write_verilog<W: std::io::Write>(
    aig: &Aig,
    cov: &IlpCover,
    mut w: W,
    module_name: &str,
) -> std::io::Result<()> {
    let dense = Dense::build(aig);
    let (pi, po) = collect_ports(aig, &dense);
    let pi_by_node: HashMap<usize, String> = pi.iter().cloned().collect();
    let po_names: HashSet<String> = po.iter().map(|(_, _, n)| n.clone()).collect();
    /* Name the class a primary output reads directly as the port itself, so the driving cell writes
    the port instead of an internal net that an `assign` then copies. ABC charges a buffer for
    every `assign`-driven output, and on output-heavy circuits that is large: the 129-output
    `adder` measured 49.48 instead of its true 40.45, i.e. exactly 129 * 0.07. A class driven by
    more than one primary output keeps the first as the port and the rest fall back to `assign`. */
    let mut po_class: HashMap<(usize, bool), String> = HashMap::new();
    for (d, inv, n) in &po {
        po_class.entry((*d, *inv)).or_insert_with(|| n.clone());
    }

    let net_of = |node: usize, neg: bool| -> String {
        if dense.is_zero[node] {
            return if neg { "1'b1".into() } else { "1'b0".into() };
        }
        if dense.is_pi[node] && !neg {
            return pi_by_node
                .get(&node)
                .cloned()
                .unwrap_or_else(|| format!("n{node}"));
        }
        if let Some(port) = po_class.get(&(node, neg)) {
            return port.clone();
        }
        if neg {
            format!("n{node}_neg")
        } else {
            format!("n{node}")
        }
    };

    // --- collect the live cone ---------------------------------------------------------------
    let mut live: HashSet<usize> = HashSet::new();
    let mut stack: Vec<usize> = po
        .iter()
        .filter(|(d, _, _)| !dense.is_zero[*d])
        .map(|(d, inv, _)| class_of(*d, *inv))
        .collect();
    let mut instances: Vec<usize> = Vec::new();
    while let Some(c) = stack.pop() {
        if !live.insert(c) {
            continue;
        }
        let n = class_node(c);
        if dense.is_zero[n] || (dense.is_pi[n] && !class_neg(c)) {
            continue;
        }
        let Some(mi) = cov.chosen[c] else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("class {c} required but not realised"),
            ));
        };
        instances.push(mi);
        let m = &cov.matches[mi];
        /* A shared instance realises *all* of its output pins at once. Walking only the leaves
        misses them, and an extra pin whose class is not marked live here would still be wired to
        that class's net below — putting two drivers on one net. `prune_reachable` already walks
        the extra pins for exactly this reason. */
        for o in &m.extra_outs {
            stack.push(class_of(o.node, o.neg));
        }
        for (p, &leaf) in m.leaves.iter().enumerate() {
            stack.push(class_of(leaf, m.leaf_neg[p]));
        }
    }
    instances.sort_unstable();
    instances.dedup();

    // --- header -----------------------------------------------------------------------------
    let mut port_list: Vec<String> = pi.iter().map(|(_, n)| n.clone()).collect();
    port_list.extend(po.iter().map(|(_, _, n)| n.clone()));
    writeln!(w, "module {}({});", module_name, port_list.join(", "))?;
    for (_, n) in &pi {
        writeln!(w, "  input {n};")?;
    }
    for (_, _, n) in &po {
        writeln!(w, "  output {n};")?;
    }

    // --- wires ------------------------------------------------------------------------------
    let mut wires: Vec<String> = Vec::new();
    for &c in &live {
        let n = class_node(c);
        if dense.is_zero[n] || (dense.is_pi[n] && !class_neg(c)) {
            continue;
        }
        let name = net_of(n, class_neg(c));
        if po_names.contains(&name) {
            continue;
        }
        wires.push(name);
    }
    wires.sort();
    wires.dedup();
    for x in &wires {
        writeln!(w, "  wire {x};")?;
    }

    // --- instances --------------------------------------------------------------------------
    for (k, &mi) in instances.iter().enumerate() {
        let m = &cov.matches[mi];
        let mut conns: Vec<String> = Vec::new();
        for (p, pin) in m.pin_names.iter().enumerate() {
            conns.push(format!(".{}({})", pin, net_of(m.leaves[p], m.leaf_neg[p])));
        }
        /* Every output pin of the instance is driven here. A pin that needs an output inverter
        cannot fold it into the cell and cannot share it either, so it gets its own inverter
        between the pin and the class's net. */
        let mut invs: Vec<(String, String)> = Vec::new();
        /* A pin is only tied to its class's net when this instance actually *drives* that class. A
        shared instance can be selected for one of its outputs while the other output's class is
        realised by a different match; wiring both anyway short-circuits the two drivers. The
        unused pin is parked on a private net so the cell and its area stay as modelled. */
        let mut outs: Vec<(Istr, String, bool)> = Vec::new();
        let mut dangling: Vec<String> = Vec::new();
        let mut pin_net = |node: usize, neg: bool, tag: usize| -> String {
            /* `chosen[c] == Some(mi)` alone is not enough: the class must also be *live*, because
            only live classes get a wire (or are primary ports). An instance reached through one
            of its outputs can carry a `chosen` entry for another output that no primary output
            reaches; connecting it produced a net that was driven but never declared, and ABC
            rejected the Verilog with "Actual net nNNNN is missing". */
            let c = class_of(node, neg);
            if live.contains(&c) && cov.chosen.get(c) == Some(&Some(mi)) {
                net_of(node, neg)
            } else {
                let d = format!("unused_g{k}_{tag}");
                dangling.push(d.clone());
                d
            }
        };
        outs.push((
            m.out_pin.clone(),
            pin_net(m.out_node, m.out_neg, 0),
            m.out_inv,
        ));
        for (t, o) in m.extra_outs.iter().enumerate() {
            outs.push((o.pin.clone(), pin_net(o.node, o.neg, t + 1), o.inv));
        }
        for (pin, net, inv) in &outs {
            if *inv {
                let phys = format!("{net}_pin");
                writeln!(w, "  wire {phys};")?;
                conns.push(format!(".{pin}({phys})"));
                invs.push((phys, net.clone()));
            } else {
                conns.push(format!(".{pin}({net})"));
            }
        }
        for d in &dangling {
            writeln!(w, "  wire {d};")?;
        }
        writeln!(w, "  {} g{} (", m.cell_name, k)?;
        writeln!(w, "    {}", conns.join(", "))?;
        writeln!(w, "  );")?;
        for (t, (a, y)) in invs.iter().enumerate() {
            /* One index per *inverted output*, not per cell: `g{k}i` collided whenever a shared
            instance needed an inverter on both of its outputs. */
            writeln!(
                w,
                "  {} g{}i{} ( .{}({}), .{}({}) );",
                cov.inv_cell, k, t, cov.inv_in, a, cov.inv_out, y
            )?;
        }
    }

    // --- output assigns ---------------------------------------------------------------------
    for (driver, inv, name) in &po {
        if dense.is_zero[*driver] {
            writeln!(w, "  assign {} = 1'b{};", name, u8::from(*inv))?;
            continue;
        }
        let net = net_of(*driver, *inv);
        if &net == name {
            continue; // the cell already drives the port directly
        }
        writeln!(w, "  assign {name} = {net};")?;
    }

    writeln!(w, "endmodule")?;
    Ok(())
}
