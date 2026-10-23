use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use itertools::Itertools;
use petgraph::prelude::*;
use rayon::prelude::*;

use crate::{
    aig_graph::{Aig as AigGraph, AigNode},
    emap_cover::{
        build_mapped_cover_binding, extract_mapped_cover_alternatives, MappedCoverChoice,
    },
    emap_cut::{compute_aig_comb_depth, enumerate_merged_cuts_mockturtle, Cut},
    emap_egg::{sexp_to_tt9, EggRewriteBundle},
    supergate_db,
    techlib::{CellVariant, MappingAreaCost, MappingObjective, PhaseArrival, TechLib},
    tt9::{tt9_bit_get, tt9_bit_set, tt9_not_masked, tt9_zero, Tt9},
};

pub use crate::emap_cover::{MappedCoverGraph, MappedCoverNode};

#[derive(Clone, Debug)]
pub struct EmapParams {
    pub max_cuts: usize,
    pub max_inputs: usize,
    pub wire_delay: f32,
    pub use_egg: bool,
    pub cache_egg: bool,
    pub cache_tt: bool,
    pub objective: MappingObjective,
    /// If true, Stage 1b rematch uses delay-first on critical ANDs and area-first elsewhere.
    /// Stage 1b always runs after Stage 1a; if false, Stage 1b uses [`MappingObjective::DelayFirst`] for every node.
    pub critical_path_mixed_objective: bool,
    /// Number of area-flow style recovery passes after delay pass.
    pub area_flow_passes: usize,
    /// Number of exact-area style recovery passes after area-flow pass.
    pub exact_area_passes: usize,
    /// Reject oversized extracted AIG covers to avoid QoR collapse from expression blow-up.
    pub max_cover_nodes: usize,
    /// Use dual-phase delay flow (considers both pos and neg bindings per node) in Stage 1.
    /// Gives lower delay than the single-phase flow at similar runtime.  Defaults to true.
    pub use_dual_phase_delay: bool,
    /// Distinguish mapper intent from mockturtle-style emap naming.
    /// This implementation targets an egraph-based mapper flow.
    pub mapper_name: &'static str,
    /// Area term used to **compare** candidate cells during mapping (mockturtle-style flow vs
    /// physical). [`NodeBinding::area`] is always **physical** for reporting.
    pub mapping_area_cost: MappingAreaCost,
    /// After egg saturation, collect up to this many **root** cover DAG enodes that tie on
    /// `(delay, area)` for the current objective, materialize each, and pick the best binding downstream.
    pub max_egg_cover_alternatives: usize,
    /// Egg saturation iteration limit for cover extraction. `0` uses the `EMAPPER_EGG_ITER_LIMIT`
    /// environment default (30). Cut mapping uses a small value: a cut AIG is tiny, and the
    /// default 30 iterations can take seconds per cut without changing the extracted cover.
    pub egg_iter_limit: usize,
    /// Egg saturation node limit for cover extraction. `0` uses the `EMAPPER_EGG_NODE_LIMIT`
    /// environment default (10_000).
    pub egg_node_limit: usize,
    /// Stage-4 level-recovery passes after the exact-area passes. Each pass is monotone (a binding
    /// is replaced only when it is strictly shallower *and* no slower), so more passes can only
    /// help; `0` disables the stage (`EMAPPER_DEPTH_RECOVERY`).
    pub depth_recovery_passes: usize,
    /// Enable the bounded two-level supergate fallback in the direct matcher (see
    /// [`crate::supergate`]). Part of the mapping portfolio because it is better on c5315
    /// (564.45/145.41 vs 571.33/149.52) and clearly worse on c7552 (461.90/19 levels vs
    /// 408.99/17) with everything else at its default, so neither setting dominates.
    pub use_two_level_cover: bool,
    /// Force the phase-arrival strictness instead of deriving it from the objective: `Some(true)`
    /// charges an inverter for every negated pin, `Some(false)` credits a leaf's native negative
    /// phase. `None` keeps the historical objective-derived behaviour.
    ///
    /// Not a tuning knob: the two models win on *different* designs (with everything else at its
    /// default, the strict model is better on c432/c880/c1355 and the crediting one on
    /// c2670/c5315), so [`emap_with_params`] evaluates both and keeps the better mapping.
    pub pessimistic_phase: Option<bool>,
}

impl Default for EmapParams {
    fn default() -> Self {
        Self {
            max_cuts: 16,
            max_inputs: 9,
            wire_delay: 0.0,
            use_egg: true,
            cache_egg: true,
            cache_tt: true,
            objective: MappingObjective::DelayFirst,
            critical_path_mixed_objective: false,
            area_flow_passes: 1,
            exact_area_passes: 1,
            max_cover_nodes: 64,
            use_dual_phase_delay: true,
            mapper_name: "egraph-based-mapper",
            mapping_area_cost: MappingAreaCost::Flow,
            max_egg_cover_alternatives: 1,
            egg_iter_limit: 0,
            egg_node_limit: 0,
            depth_recovery_passes: 1,
            use_two_level_cover: true,
            pessimistic_phase: None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum NodeBindingKind {
    Direct {
        cell: CellVariant,
    },
    /// Gate-level mapped cover extracted from the rewritten cut.
    MappedCover {
        graph: MappedCoverGraph,
        /// Per entry in [`NodeBinding::leaves`] (same order as [`MappedCoverChoice::used_leaf_indices`]):
        /// mockturtle-style delay from that leaf through the cover to the root output.
        pin_to_root_delay: Vec<f32>,
    },
}

#[derive(Clone, Debug)]
pub struct NodeBinding {
    pub node: usize,
    pub cut: Cut,
    /// Leaf nodes in variable order a,b,c,... (constant 0 is omitted).
    pub leaves: Vec<usize>,
    pub kind: NodeBindingKind,
    /// Whether the binding directly realizes the complemented node phase.
    pub phase_neg: bool,
    /// Arrival of the phase directly realized by this binding.
    pub native_delay: f32,
    /// Arrival of the positive logical node phase.
    pub delay: f32,
    /// Physical area (matches emitted netlist / [`TechLib::variant_area_physical_invs`]).
    pub area: f32,
    /// Area used only when **ranking** cuts (see [`EmapParams::mapping_area_cost`]).
    /// Equals [`NodeBinding::area`] when comparing with physical cost.
    pub mapping_cost_area: f32,
}

#[derive(Clone, Debug, Default)]
struct DualNodeBinding {
    pos: Option<NodeBinding>,
    neg: Option<NodeBinding>,
}

#[derive(Clone, Copy, Debug, Default)]
struct PhaseFlow {
    pos: f32,
    neg: f32,
}

#[derive(Clone, Debug)]
struct PhaseBindingChoice {
    binding: Option<NodeBinding>,
    arrival: f32,
    flow: f32,
}

impl Default for PhaseBindingChoice {
    fn default() -> Self {
        Self {
            binding: None,
            arrival: f32::INFINITY,
            flow: f32::INFINITY,
        }
    }
}

#[derive(Clone, Debug)]
struct NodePhaseMatch {
    best: [PhaseBindingChoice; 2],
    best_alternative: [PhaseBindingChoice; 2],
    required: [f32; 2],
    map_refs: [u32; 2],
    same_match: bool,
    /// Telemetry: best arrival offered by a `MappedCover` binding, per phase (`INFINITY` if none).
    cover_arrival: [f32; 2],
    /// Telemetry: best arrival offered by a `Direct` (single-cell) binding, per phase.
    direct_arrival: [f32; 2],
    /// Telemetry: physical area of the best `MappedCover` / `Direct` candidate, per phase.
    cover_area: [f32; 2],
    direct_area: [f32; 2],
}

impl Default for NodePhaseMatch {
    fn default() -> Self {
        Self {
            best: [PhaseBindingChoice::default(), PhaseBindingChoice::default()],
            best_alternative: [PhaseBindingChoice::default(), PhaseBindingChoice::default()],
            required: [f32::INFINITY; 2],
            map_refs: [0; 2],
            same_match: false,
            cover_arrival: [f32::INFINITY; 2],
            direct_arrival: [f32::INFINITY; 2],
            cover_area: [f32::INFINITY; 2],
            direct_area: [f32::INFINITY; 2],
        }
    }
}

/// For each cut leaf, whether **positive** / **negative** phase is available without inserting an
/// extra inverter (matches netlist: `n_k` vs `n_k_neg`, PI, const).
fn native_phases_for_leaves(
    leaves: &[usize],
    best: &[Option<NodeBinding>],
    pi_nodes: &HashSet<usize>,
) -> (Vec<bool>, Vec<bool>) {
    let mut pos = Vec::with_capacity(leaves.len());
    let mut neg = Vec::with_capacity(leaves.len());
    for &leaf in leaves {
        if leaf == 0 {
            pos.push(true);
            neg.push(true);
        } else if pi_nodes.contains(&leaf) {
            pos.push(true);
            neg.push(false);
        } else {
            match best.get(leaf).and_then(|b| b.as_ref()) {
                None => {
                    pos.push(true);
                    neg.push(false);
                }
                Some(b) => {
                    pos.push(!b.phase_neg);
                    neg.push(b.phase_neg);
                }
            }
        }
    }
    (pos, neg)
}

/// Per-pin mockturtle-style flows: `phase_flow[leaf].pos` / `.neg` for cut leaves in order.
fn pin_flows_for_leaves(
    leaves: &[usize],
    phase_flow: &[PhaseFlow],
) -> Option<(Vec<f32>, Vec<f32>)> {
    let mut fp = Vec::with_capacity(leaves.len());
    let mut fn_ = Vec::with_capacity(leaves.len());
    for &l in leaves {
        let pf = phase_flow.get(l)?;
        fp.push(pf.pos);
        fn_.push(pf.neg);
    }
    Some((fp, fn_))
}

/// Same as [`native_phases_for_leaves`], using dual-phase bindings from `delay_flow_pass`.
fn native_phases_from_dual(
    leaves: &[usize],
    best_dual: &[DualNodeBinding],
    pi_nodes: &HashSet<usize>,
) -> (Vec<bool>, Vec<bool>) {
    let mut pos = Vec::with_capacity(leaves.len());
    let mut neg = Vec::with_capacity(leaves.len());
    for &leaf in leaves {
        if leaf == 0 {
            pos.push(true);
            neg.push(true);
        } else if pi_nodes.contains(&leaf) {
            pos.push(true);
            neg.push(false);
        } else {
            match best_dual.get(leaf) {
                None => {
                    pos.push(true);
                    neg.push(false);
                }
                Some(bd) if bd.pos.is_none() && bd.neg.is_none() => {
                    pos.push(true);
                    neg.push(false);
                }
                Some(bd) => {
                    pos.push(bd.pos.is_some());
                    neg.push(bd.neg.is_some());
                }
            }
        }
    }
    (pos, neg)
}

fn dual_binding_phase_arrival(
    binding: &DualNodeBinding,
    fallback_pos: f32,
    tech: &TechLib,
) -> PhaseArrival {
    let mut phase = PhaseArrival {
        pos: fallback_pos,
        neg: fallback_pos + tech.inv_delay,
    };
    if let Some(posb) = &binding.pos {
        phase.pos = phase.pos.min(binding_positive_arrival(posb, tech));
        phase.neg = phase.neg.min(binding_negative_arrival(posb, tech));
    }
    if let Some(negb) = &binding.neg {
        phase.pos = phase.pos.min(binding_positive_arrival(negb, tech));
        phase.neg = phase.neg.min(binding_negative_arrival(negb, tech));
    }
    phase
}

fn choose_binding_for_phase(
    best: &[Option<NodeBinding>],
    node: usize,
    _phase_neg: bool,
) -> Option<NodeBinding> {
    best.get(node).and_then(|b| b.clone())
}

fn choose_dual_binding_for_request_mask(
    binding: &DualNodeBinding,
    request_mask: u8,
    tech: &TechLib,
) -> Option<NodeBinding> {
    let candidates = [binding.pos.clone(), binding.neg.clone()];
    let mut best_choice: Option<(f32, f32, NodeBinding)> = None;
    for cand in candidates.into_iter().flatten() {
        let mut req_delay = 0.0f32;
        if (request_mask & 0b01) != 0 {
            req_delay = req_delay.max(binding_positive_arrival(&cand, tech));
        }
        if (request_mask & 0b10) != 0 {
            req_delay = req_delay.max(binding_negative_arrival(&cand, tech));
        }
        let needs_opposite_phase = ((request_mask & 0b01) != 0 && cand.phase_neg)
            || ((request_mask & 0b10) != 0 && !cand.phase_neg);
        let area = cand.area
            + if needs_opposite_phase {
                tech.inv_area
            } else {
                0.0
            };
        match &best_choice {
            None => best_choice = Some((req_delay, area, cand)),
            Some((bd, ba, _)) if (req_delay, area) < (*bd, *ba) => {
                best_choice = Some((req_delay, area, cand))
            }
            _ => {}
        }
    }
    best_choice.map(|(_, _, binding)| binding)
}

fn binding_request_delay(binding: &NodeBinding, request_mask: u8, tech: &TechLib) -> f32 {
    let mut req_delay = 0.0f32;
    if (request_mask & 0b01) != 0 {
        req_delay = req_delay.max(binding_positive_arrival(binding, tech));
    }
    if (request_mask & 0b10) != 0 {
        req_delay = req_delay.max(binding_negative_arrival(binding, tech));
    }
    req_delay
}

fn binding_request_area(binding: &NodeBinding, request_mask: u8, tech: &TechLib) -> f32 {
    let needs_opposite_phase = ((request_mask & 0b01) != 0 && binding.phase_neg)
        || ((request_mask & 0b10) != 0 && !binding.phase_neg);
    binding.area
        + if needs_opposite_phase {
            tech.inv_area
        } else {
            0.0
        }
}

fn choose_dual_bindings_for_request_mask(
    binding: &DualNodeBinding,
    request_mask: u8,
    tech: &TechLib,
) -> Vec<NodeBinding> {
    if request_mask == 0 {
        return Vec::new();
    }
    if request_mask == 0b01 || request_mask == 0b10 {
        return choose_dual_binding_for_request_mask(binding, request_mask, tech)
            .into_iter()
            .collect();
    }

    let single = choose_dual_binding_for_request_mask(binding, request_mask, tech);
    let pos = choose_dual_binding_for_request_mask(binding, 0b01, tech);
    let neg = choose_dual_binding_for_request_mask(binding, 0b10, tech);

    let mut dual = HashMap::<(usize, bool), (NodeBinding, u8)>::new();
    for (choice, served_mask) in [(pos, 0b01u8), (neg, 0b10u8)] {
        if let Some(binding) = choice {
            dual.entry((binding.node, binding.phase_neg))
                .and_modify(|(_, mask)| *mask |= served_mask)
                .or_insert((binding, served_mask));
        }
    }

    let dual_choice = (!dual.is_empty()).then(|| {
        let dual_delay = dual
            .values()
            .map(|(binding, mask)| binding_request_delay(binding, *mask, tech))
            .fold(0.0, f32::max);
        let dual_area = dual
            .values()
            .map(|(binding, mask)| binding_request_area(binding, *mask, tech))
            .sum::<f32>();
        let dual_bindings = dual
            .into_values()
            .map(|(binding, _)| binding)
            .collect::<Vec<_>>();
        (dual_delay, dual_area, dual_bindings)
    });

    match (single, dual_choice) {
        (Some(single_binding), Some((dual_delay, dual_area, dual_bindings))) => {
            let single_delay = binding_request_delay(&single_binding, request_mask, tech);
            let single_area = binding_request_area(&single_binding, request_mask, tech);
            if (dual_delay, dual_area) < (single_delay, single_area) {
                dual_bindings
            } else {
                vec![single_binding]
            }
        }
        (Some(single_binding), None) => vec![single_binding],
        (None, Some((_, _, dual_bindings))) => dual_bindings,
        (None, None) => Vec::new(),
    }
}

fn drop_redundant_phase_delay(
    pos: Option<NodeBinding>,
    neg: Option<NodeBinding>,
    tech: &TechLib,
) -> (Option<NodeBinding>, Option<NodeBinding>) {
    let (Some(posb), Some(negb)) = (pos.as_ref(), neg.as_ref()) else {
        return (pos, neg);
    };

    let eps = 1e-3f32;
    let pos_arr = binding_positive_arrival(&posb, tech);
    let neg_arr = binding_negative_arrival(&negb, tech);
    let pos_via_neg = binding_positive_arrival(&negb, tech);
    let neg_via_pos = binding_negative_arrival(&posb, tech);

    let keep_neg_only = pos_via_neg <= pos_arr + eps;
    let keep_pos_only = neg_via_pos <= neg_arr + eps;

    if keep_pos_only && keep_neg_only {
        let pos_cost = (neg_via_pos.max(pos_arr), posb.area + tech.inv_area);
        let neg_cost = (pos_via_neg.max(neg_arr), negb.area + tech.inv_area);
        if pos_cost <= neg_cost {
            (pos, None)
        } else {
            (None, neg)
        }
    } else if keep_pos_only {
        (pos, None)
    } else if keep_neg_only {
        (None, neg)
    } else {
        (pos, neg)
    }
}

fn next_required_phases(binding: &NodeBinding, aig: &AigGraph) -> Vec<(usize, bool)> {
    match &binding.kind {
        NodeBindingKind::Direct { cell } => binding
            .leaves
            .iter()
            .take(cell.n_inputs as usize)
            .enumerate()
            .filter_map(|(i, &leaf)| {
                (leaf != 0 && aig.node_type(NodeIndex::new(leaf)) == AigNode::And)
                    .then_some((leaf, ((cell.neg_mask >> i) & 1) == 1))
            })
            .collect(),
        NodeBindingKind::MappedCover { .. } => cover_used_leaves(binding)
            .into_iter()
            .filter(|&leaf| leaf != 0 && aig.node_type(NodeIndex::new(leaf)) == AigNode::And)
            .map(|leaf| (leaf, false))
            .collect(),
    }
}

/// Collect the bindings to materialise, choosing each node's **phase realisation** from the demand
/// its consumers place on it.
///
/// `best` gives one binding per node, so a consumer that needs a node's complemented output pays an
/// explicit inverter. `neg_alt` holds the stage-1 negative-phase realisation of the same node, which
/// provides that complemented output natively. When every consumer needs the *negative* phase, using
/// `neg_alt` removes those inverters at no extra cell, provided the positive phase is not also
/// needed (materialising both would be a second cell, which measured worse).
///
/// Measured on c432 the emitted netlist carries 93 inverters against `emap`'s 30, of which 66 are
/// Direct-binding pin inversions (AND2x2 alone 31); this pass targets exactly those.
fn collect_phase_aware_bindings(
    aig: &AigGraph,
    tech: &TechLib,
    best: &[Option<NodeBinding>],
    neg_alt: &[Option<NodeBinding>],
) -> Vec<NodeBinding> {
    // Pass 1: count, per node, how many consumers need each phase.
    let mut want_pos: HashMap<usize, usize> = HashMap::new();
    let mut want_neg: HashMap<usize, usize> = HashMap::new();
    {
        let mut frontier = aig
            .graph()
            .externals(Outgoing)
            .flat_map(|o| aig.graph().edges_directed(o, Incoming))
            .filter_map(|edge| match aig.node_type(edge.source()) {
                AigNode::And => Some((edge.source().index(), *edge.weight())),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut seen = HashSet::<(usize, bool)>::new();
        while let Some((n, neg)) = frontier.pop() {
            if !seen.insert((n, neg)) {
                continue;
            }
            *if neg {
                want_neg.entry(n)
            } else {
                want_pos.entry(n)
            }
            .or_insert(0) += 1;
            let Some(b) = best.get(n).and_then(|b| b.as_ref()) else {
                continue;
            };
            for dep in next_required_phases(b, aig) {
                if !seen.contains(&dep) {
                    frontier.push(dep);
                }
            }
        }
    }

    // Pass 2: pick the realisation, then collect.
    collect_required_bindings_with(aig, |node, want_neg_phase| {
        let pos = best.get(node).and_then(|b| b.clone());
        let alt = neg_alt.get(node).and_then(|b| b.clone());
        let swap = alt.is_some()
            && want_pos.get(&node).copied().unwrap_or(0) == 0
            && want_neg.get(&node).copied().unwrap_or(0) > 0;
        let chosen = if swap { alt } else { pos };
        match chosen {
            Some(b) if b.phase_neg == want_neg_phase => Some(b),
            other => other,
        }
    })
}

/// Ensure the binding set is closed under its own leaves.
///
/// `collect_required_bindings_with` walks from the primary outputs and silently skips a leave whose
/// `select` returns `None`, so a node that is *consumed* but never *mapped* stays out of the result
/// while other bindings still reference it. The writer then inserts an inverter on a net that no cell
/// drives, and ABC rejects the netlist (`Actual net n1233 is missing`, c5315 line 1641) -- which is
/// also why `run_abc_cec` returned -1 there.
///
/// Repair by mapping any such node with its trivial two-input cut: a 2-input function always lands on
/// some library variant (input polarity is free in the NP index), so this cannot fail while the leaf
/// is an AND node. Cascades are handled by iterating to a fixpoint.
fn close_binding_set(
    aig: &AigGraph,
    tech: &TechLib,
    bindings: &mut Vec<NodeBinding>,
    best: &[Option<NodeBinding>],
    arrival: &[f32],
    wire_delay: f32,
) {
    let len = best
        .len()
        .max(bindings.iter().map(|b| b.node + 1).max().unwrap_or(0));
    let mut closed_total = 0usize;
    let mut rounds = 0usize;
    loop {
        let bound: HashSet<usize> = bindings.iter().map(|b| b.node).collect();
        let mut missing: Vec<usize> = Vec::new();
        // Every node consumed as a leaf must be driven...
        for b in bindings.iter() {
            for &leaf in &b.leaves {
                if leaf == 0 || leaf >= len || bound.contains(&leaf) || missing.contains(&leaf) {
                    continue;
                }
                if matches!(aig.node_type(NodeIndex::new(leaf)), AigNode::And) {
                    missing.push(leaf);
                }
            }
        }
        // ...and so must every primary-output driver. A PO whose driver is unmapped made the writer
        // emit `assign <port> = <port>;` (both sides normalise to the port name through
        // `apply_po_rename`), which is a combinational loop ABC rejects -- c5315 line 8228, where
        // `n5214` does not exist anywhere in the file because the node was never mapped.
        for out in aig.graph().externals(Outgoing) {
            for edge in aig.graph().edges_directed(out, Incoming) {
                let d = edge.source().index();
                if d != 0
                    && d < len
                    && !bound.contains(&d)
                    && !missing.contains(&d)
                    && matches!(aig.node_type(NodeIndex::new(d)), AigNode::And)
                {
                    missing.push(d);
                }
            }
        }
        if missing.is_empty() {
            if std::env::var_os("EMAPPER_DEBUG_PO").is_some() {
                eprintln!(
                    "[close-binding-set] bindings={} closed_added={} rounds={}",
                    bindings.len(),
                    closed_total,
                    rounds
                );
            }
            if std::env::var("EMAPPER_DEBUG_PO").is_ok_and(|v| v == "2") {
                let by_node: HashMap<usize, &NodeBinding> =
                    bindings.iter().map(|b| (b.node, b)).collect();
                let mut rows = Vec::new();
                for out in aig.graph().externals(Outgoing) {
                    for edge in aig.graph().edges_directed(out, Incoming) {
                        let d = edge.source().index();
                        let port = match aig.node_type(out) {
                            AigNode::Output(pos) => aig
                                .symbol_output(pos)
                                .cloned()
                                .unwrap_or_else(|| format!("#{pos}")),
                            _ => "?".to_string(),
                        };
                        rows.push(format!(
                            "port={} out={} d={} inv={} type={:?} bound={} phase_neg={:?}",
                            port,
                            out.index(),
                            d,
                            *edge.weight(),
                            aig.node_type(NodeIndex::new(d)),
                            by_node.contains_key(&d),
                            by_node.get(&d).map(|b| b.phase_neg)
                        ));
                    }
                }
                for r in rows.iter().take(200) {
                    eprintln!("[po-dump] {r}");
                }
                eprintln!("[po-dump] total PO edges={}", rows.len());
            }
            if std::env::var_os("EMAPPER_DEBUG_PO").is_some() {
                let bound: HashSet<usize> = bindings.iter().map(|b| b.node).collect();
                let mut bad = Vec::new();
                for out in aig.graph().externals(Outgoing) {
                    for edge in aig.graph().edges_directed(out, Incoming) {
                        let d = edge.source().index();
                        if d == 0 || d >= len {
                            continue;
                        }
                        if !bound.contains(&d) {
                            bad.push(format!(
                                "out{} d={} type={:?} best={} inv={}",
                                out.index(),
                                d,
                                aig.node_type(NodeIndex::new(d)),
                                best.get(d).map(|b| b.is_some()).unwrap_or(false),
                                *edge.weight()
                            ));
                        }
                    }
                }
                eprintln!(
                    "[po-closure] bindings={} len={} unbound_po_drivers={} first={:?}",
                    bindings.len(),
                    len,
                    bad.len(),
                    &bad[..bad.len().min(6)]
                );
            }
            return;
        }
        missing.sort_unstable();
        let mut added = 0usize;
        for node in missing {
            let Some((xe, ye)) = aig.try_unwrap_and(NodeIndex::new(node)) else {
                continue;
            };
            let x = aig.edge_source(xe).index();
            let y = aig.edge_source(ye).index();
            let lit = |v: &str, inv: bool| {
                if inv {
                    format!("(~ {v})")
                } else {
                    v.to_string()
                }
            };
            let expr = format!(
                "(& {} {})",
                lit("a", aig.is_edge_inverted(xe)),
                lit("b", aig.is_edge_inverted(ye))
            );
            let Some((n_inputs, tt)) = crate::emap_egg::sexp_to_tt9(&expr) else {
                continue;
            };
            let leaves = vec![x, y];
            let phases = leaves
                .iter()
                .map(|&l| {
                    let d = arrival.get(l).copied().unwrap_or(0.0);
                    PhaseArrival {
                        pos: d,
                        neg: d + tech.inv_delay,
                    }
                })
                .collect::<Vec<_>>();
            let native = native_phases_for_leaves(
                &leaves,
                best,
                &aig.input_nodes().iter().map(|n| n.index()).collect(),
            );
            let native_ref = Some((native.0.as_slice(), native.1.as_slice()));
            let Some((cell_delay, _, physical_a, variant)) = tech.best_for_tt_phase_arrivals(
                n_inputs,
                tt,
                &phases,
                MappingObjective::DelayFirst,
                native_ref,
                None,
                crate::techlib::MappingAreaCost::Physical,
            ) else {
                continue;
            };
            let d = cell_delay + wire_delay;
            let cut = Cut::new(
                {
                    let mut v = leaves.clone();
                    v.sort_unstable();
                    v
                },
                node,
                {
                    let mut v = leaves.clone();
                    v.push(node);
                    v.sort_unstable();
                    v.dedup();
                    v
                },
            );
            bindings.push(NodeBinding {
                node,
                cut,
                leaves,
                kind: NodeBindingKind::Direct {
                    cell: variant.clone(),
                },
                phase_neg: false,
                native_delay: d,
                delay: d,
                area: physical_a,
                mapping_cost_area: physical_a,
            });
            added += 1;
            closed_total += 1;
        }
        if added == 0 {
            if std::env::var_os("EMAPPER_DEBUG_PO").is_some() {
                eprintln!(
                    "[close-binding-set] bindings={} closed_added={} rounds={} (no progress)",
                    bindings.len(),
                    closed_total,
                    rounds
                );
            }
            return;
        }
        rounds += 1;
    }
}

fn collect_required_bindings_with<F>(aig: &AigGraph, mut select: F) -> Vec<NodeBinding>
where
    F: FnMut(usize, bool) -> Option<NodeBinding>,
{
    let mut frontier = aig
        .graph()
        .externals(Outgoing)
        .flat_map(|o| aig.graph().edges_directed(o, Incoming))
        .filter_map(|edge| match aig.node_type(edge.source()) {
            AigNode::And => Some((edge.source().index(), *edge.weight())),
            _ => None,
        })
        .collect::<Vec<_>>();

    let mut visited = HashSet::<(usize, bool)>::new();
    let mut selected = HashMap::<(usize, bool), NodeBinding>::new();

    while let Some((n, req_neg)) = frontier.pop() {
        if !visited.insert((n, req_neg)) {
            continue;
        }
        let Some(binding) = select(n, req_neg) else {
            continue;
        };
        selected
            .entry((binding.node, binding.phase_neg))
            .or_insert_with(|| binding.clone());
        for dep in next_required_phases(&binding, aig) {
            if !visited.contains(&dep) {
                frontier.push(dep);
            }
        }
    }

    let mut bindings = selected.into_values().collect::<Vec<_>>();
    bindings.sort_by_key(|b| (b.node, b.phase_neg));
    bindings
}

fn collect_required_bindings(aig: &AigGraph, best: &[Option<NodeBinding>]) -> Vec<NodeBinding> {
    collect_required_bindings_with(aig, |node, phase_neg| {
        choose_binding_for_phase(best, node, phase_neg)
    })
}

fn compute_structural_request_masks(aig: &AigGraph, len: usize) -> Vec<u8> {
    let mut request_mask = vec![0u8; len];
    for node in aig.graph().node_indices() {
        if !matches!(aig.node_type(node), AigNode::And) {
            continue;
        }
        for edge in aig.graph().edges_directed(node, Outgoing) {
            request_mask[node.index()] |= if *edge.weight() { 0b10 } else { 0b01 };
        }
    }
    request_mask
}

fn phase_arrival_from_bindings(bindings: &[NodeBinding], tech: &TechLib) -> Option<PhaseArrival> {
    if bindings.is_empty() {
        return None;
    }
    let mut phase = PhaseArrival {
        pos: f32::INFINITY,
        neg: f32::INFINITY,
    };
    for binding in bindings {
        phase.pos = phase.pos.min(binding_positive_arrival(binding, tech));
        phase.neg = phase.neg.min(binding_negative_arrival(binding, tech));
    }
    Some(phase)
}

fn compute_est_refs(aig: &AigGraph, len: usize) -> Vec<f32> {
    let mut est_refs = vec![1.0f32; len];
    for node in aig.graph().node_indices() {
        let idx = node.index();
        let fanout = aig.graph().edges_directed(node, Outgoing).count() as f32;
        est_refs[idx] = fanout.max(1.0);
    }
    est_refs
}

fn binding_output_arrival(binding: &NodeBinding, phase_neg: bool, tech: &TechLib) -> f32 {
    if phase_neg {
        binding_negative_arrival(binding, tech)
    } else {
        binding_positive_arrival(binding, tech)
    }
}

/// Whether the Stage-1a delay pass runs the required-time **area refinement** of
/// [`refine_best_matches`].
///
/// The refinement replaces a node's delay-optimal binding with the lowest-flow alternative that
/// still meets the *forward-pass* required time. Those required times are derived from the forward
/// pass's own arrival estimate while the alternatives' stored arrivals were measured against the
/// forward pass's leaf arrivals, so a chain of substitutions can push the recomputed primary-output
/// arrival far above the estimate the required times were built from: on `c432` the forward pass
/// reaches 199.9 while the assembled netlist after refinement measures 675.2. Set
/// `EMAPPER_REFINE=0` to keep the delay-optimal solution and leave area recovery to Stages 2/3,
/// which do re-measure the primary-output delay before accepting a pass.
fn refine_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("EMAPPER_REFINE")
            .map_or(true, |v| !(v == "0" || v.eq_ignore_ascii_case("false")))
    })
}

/// `EMAPPER_DUAL_PHASE=1`: also materialise the negative-phase realization the delay flow kept,
/// for nodes on the critical path.
///
/// Measured on c432/c880 (egg 2/500, af=1/ea=1) this trades **-3 % delay for +10 % area**, because
/// the negative realization is a second standard cell rather than a free inversion -- while the
/// delay flow's arrival model credits it as native. Default **off**: the phase asymmetry is a cost
/// model decision, not a bug, and the area price is not worth the delay at the current QoR level.
fn dual_phase_materialization() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("EMAPPER_DUAL_PHASE").is_ok_and(|v| v == "1"))
}

/// `EMAPPER_CUT_POOL_RANK=arrival` ranks each node's surviving cut pool by the arrival the fold
/// measured, instead of the historical flow order.
///
/// Default **off**: measured on c432/c880 (egg 2/500) the arrival ranking keeps the *wide* cuts and
/// both designs then exceed a 300 s timeout, because every extra wide cut at every node is a new
/// e-graph cover extraction. The flow order (small cuts first) is what keeps the cut pool cheap;
/// changing the ranking without also shrinking the extraction budget just moves the cost.
fn cut_pool_rank_by_arrival() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("EMAPPER_CUT_POOL_RANK").is_ok_and(|v| v.eq_ignore_ascii_case("arrival"))
    })
}

/// Price every negated pin as `positive + inv_delay` instead of crediting a leaf's **native**
/// negative phase.
///
/// `EMAPPER_PESSIMISTIC_PHASE=0` restores the historical credit. Default **on**, because that credit
/// describes a netlist this mapper does not emit: `collect_required_bindings` materialises one
/// binding per node, so a node's opposite phase always costs an explicit inverter, while the
/// credited model assumed a native one. Measured (egg 2/500, af=1/ea=1, `nec=1`):
///
/// | design | credit (old) | honest (default) |
/// | --- | --- | --- |
/// | c432 | 15.82 / 627.4 / 18 | **16.03 / 509.6 / 17** |
/// | c880 | 22.55 / 446.5 / 19 | **23.51 / 425.3 / 18** |
///
/// i.e. delay -19 % / -4.7 % for area +1.3 % / +4.3 %. The delay gap to `emap` is the far larger
/// of the two (2.2x vs 1.06x), and a mapper whose own arrival model is 2.5x better than what it
/// emits cannot be tuned further, so the honest model is the right default.
fn pessimistic_phase_for(ps: &EmapParams, objective: MappingObjective) -> bool {
    if let Some(forced) = ps.pessimistic_phase {
        return forced;
    }
    if let Ok(v) = std::env::var("EMAPPER_PESSIMISTIC_PHASE") {
        return !(v == "0" || v.eq_ignore_ascii_case("false"));
    }
    // With no explicit override, follow the declared objective. Measured (egg 2/500, af=1/ea=1,
    // nec=1): the honest model is clearly better for delay (c432 479 vs 597, c880 425 vs 447,
    // c5315 424 vs 424 with -4.6 area points worse), while the optimistic credit is better for
    // area (c880 22.34 vs 23.39, c5315 129.0 vs 124.4). Tying the strictness to the objective
    // lets each run get the model it is actually optimising for.
    objective == MappingObjective::DelayFirst
}

/// `EMAPPER_EGG_MIN_LEAVES`: minimum cut width (leaves) that still gets an e-graph saturation.
fn egg_min_leaves() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("EMAPPER_EGG_MIN_LEAVES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
    })
}

/// Process-wide e-graph attempt budget for one mapping run (`EMAPPER_EGG_MAX_ATTEMPTS`, default
/// 20 000; `0` = unlimited).
///
/// Egg is the mapper's entire runtime on the larger designs -- c5315 finishes in 2 s with
/// `EMAPPER_USE_EGG=0` and does not finish at all (>120 s) with it on, while the QoR it buys on
/// c432/c880 is one to three levels and a few percent of area. A hard budget keeps the mapper
/// usable on those designs while leaving the small ones (c432: ~7 000 attempts) untouched.
static EGG_ATTEMPTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn egg_attempt_budget() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("EMAPPER_EGG_MAX_ATTEMPTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20_000)
    })
}

fn reset_egg_budget() {
    EGG_ATTEMPTS.store(0, std::sync::atomic::Ordering::Relaxed);
    EGG_DEADLINE.with(|d| *d.borrow_mut() = None);
}

thread_local! {
    /// Wall-clock deadline for the e-graph stage of one mapping run.
    static EGG_DEADLINE: std::cell::RefCell<Option<std::time::Instant>> =
        const { std::cell::RefCell::new(None) };
}

/// `EMAPPER_EGG_TIME_BUDGET` (seconds, default 10; `0` = unlimited): wall-clock cap on the e-graph
/// stage per mapping run.
///
/// An attempt count is not enough: on c7552 `EMAPPER_EGG_MAX_ATTEMPTS=2000` still exceeded 300 s
/// while the same design finishes in **3 s** with `EMAPPER_USE_EGG=0` (114.54/538.8/23, nec=1), i.e.
/// single extractions there are orders of magnitude more expensive than on c432. A time cap keeps
/// the designs that benefit from egg (c432 479 vs 597 delay) while making the large ones terminate
/// with whatever the stage produced so far.
fn egg_time_budget_secs() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("EMAPPER_EGG_TIME_BUDGET")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10)
    })
}

fn egg_budget_available() -> bool {
    let secs = egg_time_budget_secs();
    if secs > 0 {
        let expired = EGG_DEADLINE.with(|d| {
            let mut slot = d.borrow_mut();
            let deadline = slot.get_or_insert_with(|| {
                std::time::Instant::now() + std::time::Duration::from_secs(secs)
            });
            std::time::Instant::now() >= *deadline
        });
        if expired {
            return false;
        }
    }
    let budget = egg_attempt_budget();
    if budget == 0 {
        return true;
    }
    EGG_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < budget
}

/// `EMAPPER_CUT_POOL_DIVERSE=1` round-robins each node's surviving cut pool by cut width instead of
/// keeping the `max_cuts` smallest cuts.
///
/// Default **off**, because it does not address the actual bottleneck and is unsafe: the pool is
/// built only from cuts that produced a binding (`fold_dual_phase_scored_into_node_match` pushes to
/// `kept_cuts` per scored binding), so an *unmappable* wide cut is never in the pool to begin with
/// and diversity cannot recover it. Measured with egg on: c432 unchanged, **c880 TIMEOUT** -- the
/// extra wide cuts that do survive each cost a fresh e-graph extraction.
fn cut_pool_diverse_default_note() {}
///
/// The pool kept per node is what the next level merges from, and the historical rule takes the
/// **smallest** cuts (`enumerate_merged_cuts_mockturtle` sorts by ascending flow = sum of estimated
/// references). Wide cuts therefore disappear before they can ever be merged further up, which
/// shows directly in the mapped result: on c5315, 746 of 1096 bindings use 2-input cuts, while
/// `emap` on the same AIG uses AO33/OAI21/OR3-class gates that absorb two AIG levels each.
/// Round-robin over cut width keeps the pool size (and therefore the cost) identical while making
/// sure every width is represented.
fn cut_pool_diverse() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("EMAPPER_CUT_POOL_DIVERSE").is_ok_and(|v| v == "1"))
}

/// Pick up to `limit` distinct cuts from `kept_cuts`, round-robin over cut width, **widest first**.
fn diverse_cut_pool(kept_cuts: &[(Cut, f32, f32)], limit: usize) -> Vec<Cut> {
    let max_w = kept_cuts
        .iter()
        .map(|(c, _, _)| c.input_count())
        .max()
        .unwrap_or(1);
    let mut buckets: Vec<Vec<Cut>> = vec![Vec::new(); max_w + 2];
    for (cut, _, _) in kept_cuts {
        let w = cut.input_count();
        if !buckets[w].iter().any(|c| c == cut) {
            buckets[w].push(cut.clone());
        }
    }
    let mut cursors = vec![0usize; max_w + 2];
    let mut out: Vec<Cut> = Vec::with_capacity(limit);
    let mut progressed = true;
    while out.len() < limit && progressed {
        progressed = false;
        for w in (1..=max_w).rev() {
            if out.len() >= limit {
                break;
            }
            if let Some(cut) = buckets[w].get(cursors[w]) {
                out.push(cut.clone());
                cursors[w] += 1;
                progressed = true;
            }
        }
    }
    out
}

/// `EMAPPER_SUPERGATE_RELAXED=1` adds a bucket-agnostic database fallback (default off).
///
/// Measured: c432/c880 byte-identical results, c5315 goes from 8 s to a timeout -- the extra hits
/// buy no QoR and each one costs a cover extraction.
fn supergate_db_relaxed() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("EMAPPER_SUPERGATE_RELAXED").is_ok_and(|v| v == "1"))
}

/// See the call site in `score_cut_for_phase_core`: artificial area credit for cover bindings.
fn cover_area_bonus() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("EMAPPER_COVER_AREA_BONUS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0)
    })
}

/// `EMAPPER_DELAY_ITERS` (default **0**): required-time rematch iterations after Stage 1.
///
/// Measured (egg 2/500, af=1/ea=1): `1` or `4` iterations leave c432 **byte-identical**
/// (16.03/509.6/17) -- Stage 1b already is one such rematch, so the fixpoint converges immediately
/// and the delay gap is not attributable to insufficient required-time iteration. On c880 it makes
/// area slightly better (23.51 -> 23.21) but delay clearly worse (425.3 -> 465.7). Default 0.
fn delay_fixpoint_iters() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("EMAPPER_DELAY_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
    })
}

/// `EMAPPER_DEPTH_DELAY_SLACK` set (any value, including 0): snapshot the pre-level mapping so the
/// caller can restore it if Stage 4/5 made the worst primary output slower.
fn depth_delay_slack_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("EMAPPER_DEPTH_DELAY_SLACK").is_ok())
}

/// `EMAPPER_PHASE_DEMAND=0` disables the consumer-phase-driven binding selection (default on).
fn phase_demand_selection() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("EMAPPER_PHASE_DEMAND")
            .map_or(true, |v| !(v == "0" || v.eq_ignore_ascii_case("false")))
    })
}

fn better_phase_choice(
    _candidate_binding: &NodeBinding,
    candidate_arrival: f32,
    candidate_flow: f32,
    current: &PhaseBindingChoice,
    prefer_flow: bool,
) -> bool {
    let cand = (
        if prefer_flow {
            candidate_flow
        } else {
            candidate_arrival
        },
        if prefer_flow {
            candidate_arrival
        } else {
            candidate_flow
        },
    );
    if current.binding.is_none() {
        return true;
    }
    let cur = (
        if prefer_flow {
            current.flow
        } else {
            current.arrival
        },
        if prefer_flow {
            current.arrival
        } else {
            current.flow
        },
    );
    cand < cur
}

fn fold_dual_phase_scored_into_node_match(
    scored: DualNodeBinding,
    tech: &TechLib,
    aig: &AigGraph,
    phase_flow: &[PhaseFlow],
    node_match: &mut NodePhaseMatch,
    kept_cuts: &mut Vec<(Cut, f32, f32)>,
) {
    if let Some(posb) = scored.pos {
        let pos_arrival = binding_positive_arrival(&posb, tech);
        let pos_flow = binding_phase_flow(&posb, aig, phase_flow);
        match &posb.kind {
            NodeBindingKind::MappedCover { .. } => {
                if pos_arrival < node_match.cover_arrival[0] {
                    node_match.cover_arrival[0] = pos_arrival;
                    node_match.cover_area[0] = posb.area;
                }
            }
            NodeBindingKind::Direct { .. } => {
                if pos_arrival < node_match.direct_arrival[0] {
                    node_match.direct_arrival[0] = pos_arrival;
                    node_match.direct_area[0] = posb.area;
                }
            }
        }
        if better_phase_choice(&posb, pos_arrival, pos_flow, &node_match.best[0], false) {
            node_match.best[0] = PhaseBindingChoice {
                binding: Some(posb.clone()),
                arrival: pos_arrival,
                flow: pos_flow,
            };
        }
        if better_phase_choice(
            &posb,
            pos_arrival,
            pos_flow,
            &node_match.best_alternative[0],
            true,
        ) {
            node_match.best_alternative[0] = PhaseBindingChoice {
                binding: Some(posb.clone()),
                arrival: pos_arrival,
                flow: pos_flow,
            };
        }
        kept_cuts.push((posb.cut.clone(), pos_arrival, pos_flow));
    }

    if let Some(negb) = scored.neg {
        let neg_arrival = binding_negative_arrival(&negb, tech);
        let neg_flow = binding_phase_flow(&negb, aig, phase_flow);
        match &negb.kind {
            NodeBindingKind::MappedCover { .. } => {
                node_match.cover_arrival[1] = node_match.cover_arrival[1].min(neg_arrival);
            }
            NodeBindingKind::Direct { .. } => {
                node_match.direct_arrival[1] = node_match.direct_arrival[1].min(neg_arrival);
            }
        }
        if better_phase_choice(&negb, neg_arrival, neg_flow, &node_match.best[1], false) {
            node_match.best[1] = PhaseBindingChoice {
                binding: Some(negb.clone()),
                arrival: neg_arrival,
                flow: neg_flow,
            };
        }
        if better_phase_choice(
            &negb,
            neg_arrival,
            neg_flow,
            &node_match.best_alternative[1],
            true,
        ) {
            node_match.best_alternative[1] = PhaseBindingChoice {
                binding: Some(negb.clone()),
                arrival: neg_arrival,
                flow: neg_flow,
            };
        }
        kept_cuts.push((negb.cut.clone(), neg_arrival, neg_flow));
    }
}

fn refine_best_matches_copy_refinement(
    node_match: &mut NodePhaseMatch,
    phase: usize,
    both_phases: bool,
    tech: &TechLib,
) {
    let bg = node_match.best_alternative[phase].clone();
    if bg.binding.is_none() {
        return;
    }

    node_match.best[phase] = bg.clone();
    if !both_phases {
        return;
    }

    node_match.same_match = true;
    let other = phase ^ 1;
    node_match.best[other] = PhaseBindingChoice {
        binding: None,
        arrival: bg.arrival + tech.inv_delay,
        flow: bg.flow + tech.inv_area,
    };
}

fn refine_best_matches(node_match: &mut NodePhaseMatch, tech: &TechLib) {
    const EPS: f32 = 1e-3;
    let g0 = node_match.best_alternative[0].clone();
    let g1 = node_match.best_alternative[1].clone();

    if node_match.map_refs[0] > 0 && node_match.map_refs[1] > 0 {
        if node_match.same_match {
            if g0.binding.is_some()
                && g1.binding.is_some()
                && g0.arrival <= node_match.required[0] + EPS
                && g1.arrival <= node_match.required[1] + EPS
            {
                node_match.same_match = false;
                refine_best_matches_copy_refinement(node_match, 0, false, tech);
                refine_best_matches_copy_refinement(node_match, 1, false, tech);
                return;
            }
            let use_phase = if g0.binding.is_none() { 1 } else { 0 };
            let use_alt = &node_match.best_alternative[use_phase];
            if use_alt.binding.is_some()
                && use_alt.arrival <= node_match.required[use_phase] + EPS
                && use_alt.arrival + tech.inv_delay <= node_match.required[use_phase ^ 1] + EPS
            {
                refine_best_matches_copy_refinement(node_match, use_phase, true, tech);
            }
            return;
        }

        if g0.binding.is_some() && g0.arrival <= node_match.required[0] + EPS {
            refine_best_matches_copy_refinement(
                node_match,
                0,
                g1.binding.is_none() && g0.arrival + tech.inv_delay <= node_match.required[1] + EPS,
                tech,
            );
        }
        if g1.binding.is_some() && g1.arrival <= node_match.required[1] + EPS {
            refine_best_matches_copy_refinement(
                node_match,
                1,
                g0.binding.is_none() && g1.arrival + tech.inv_delay <= node_match.required[0] + EPS,
                tech,
            );
        }
        return;
    }

    if node_match.map_refs[0] > 0 {
        if g0.binding.is_some() && g0.arrival <= node_match.required[0] + EPS {
            node_match.same_match = false;
            refine_best_matches_copy_refinement(node_match, 0, false, tech);
        } else if g0.binding.is_none()
            && g1.binding.is_some()
            && g1.arrival + tech.inv_delay <= node_match.required[0] + EPS
        {
            refine_best_matches_copy_refinement(node_match, 1, true, tech);
        }
    } else if node_match.map_refs[1] > 0 {
        if g1.binding.is_some() && g1.arrival <= node_match.required[1] + EPS {
            node_match.same_match = false;
            refine_best_matches_copy_refinement(node_match, 1, false, tech);
        } else if g1.binding.is_none()
            && g0.binding.is_some()
            && g0.arrival + tech.inv_delay <= node_match.required[1] + EPS
        {
            refine_best_matches_copy_refinement(node_match, 0, true, tech);
        }
    }
}

fn select_alternatives(node_match: &mut NodePhaseMatch, tech: &TechLib, est_ref: f32) {
    const EPS: f32 = 1e-3;

    for (phase_idx, phase_neg) in [(0usize, false), (1usize, true)] {
        let best = node_match.best[phase_idx].clone();
        let alt = node_match.best_alternative[phase_idx].clone();
        let Some(ref alt_binding) = alt.binding else {
            continue;
        };
        let replace = match best.binding.as_ref() {
            None => true,
            Some(best_binding) => {
                let best_arr = binding_output_arrival(best_binding, phase_neg, tech);
                let alt_arr = binding_output_arrival(&alt_binding, phase_neg, tech);
                alt_arr <= best_arr + EPS && alt.flow < best.flow - EPS
            }
        };
        if replace {
            node_match.best[phase_idx] = alt;
        }
    }

    let pos = node_match.best[0].clone();
    let neg = node_match.best[1].clone();
    match (pos.binding.as_ref(), neg.binding.as_ref()) {
        (Some(posb), Some(negb)) => {
            let pos_arr = binding_positive_arrival(posb, tech);
            let neg_arr = binding_negative_arrival(negb, tech);
            let pos_via_neg_arr = binding_positive_arrival(negb, tech);
            let neg_via_pos_arr = binding_negative_arrival(posb, tech);
            let pos_via_neg_flow = neg.flow + tech.inv_area / est_ref;
            let neg_via_pos_flow = pos.flow + tech.inv_area / est_ref;

            let drop_pos = pos_via_neg_arr <= pos_arr + EPS && pos_via_neg_flow <= pos.flow + EPS;
            let drop_neg = neg_via_pos_arr <= neg_arr + EPS && neg_via_pos_flow <= neg.flow + EPS;

            if drop_pos && drop_neg {
                let pos_only = (
                    neg_via_pos_arr.max(pos_arr),
                    pos.flow + tech.inv_area / est_ref,
                );
                let neg_only = (
                    pos_via_neg_arr.max(neg_arr),
                    neg.flow + tech.inv_area / est_ref,
                );
                if pos_only <= neg_only {
                    node_match.best[1].binding = None;
                    node_match.best[1].flow = neg_via_pos_flow;
                } else {
                    node_match.best[0].binding = None;
                    node_match.best[0].flow = pos_via_neg_flow;
                }
                node_match.same_match = true;
            } else if drop_pos {
                node_match.best[0].binding = None;
                node_match.best[0].flow = pos_via_neg_flow;
                node_match.same_match = true;
            } else if drop_neg {
                node_match.best[1].binding = None;
                node_match.best[1].flow = neg_via_pos_flow;
                node_match.same_match = true;
            }
        }
        (Some(_), None) | (None, Some(_)) => {
            node_match.same_match = true;
        }
        (None, None) => {}
    }
}

fn binding_phase_flow(binding: &NodeBinding, aig: &AigGraph, phase_flow: &[PhaseFlow]) -> f32 {
    let mut flow = binding.area;
    match &binding.kind {
        NodeBindingKind::Direct { cell } => {
            for i in 0..(cell.n_inputs as usize) {
                let Some(&leaf) = binding.leaves.get(i) else {
                    continue;
                };
                if leaf == 0 || leaf >= phase_flow.len() {
                    continue;
                }
                if !matches!(aig.node_type(NodeIndex::new(leaf)), AigNode::And) {
                    continue;
                }
                let child_phase_neg = ((cell.neg_mask >> i) & 1) == 1;
                flow += if child_phase_neg {
                    phase_flow[leaf].neg
                } else {
                    phase_flow[leaf].pos
                };
            }
        }
        NodeBindingKind::MappedCover { .. } => {
            let deps = binding_dependency_leaves(binding, phase_flow.len());
            for leaf in deps {
                if !matches!(aig.node_type(NodeIndex::new(leaf)), AigNode::And) {
                    continue;
                }
                flow += phase_flow[leaf].pos;
            }
        }
    }
    flow
}

fn propagate_required_to_leaves(
    aig: &AigGraph,
    binding: &NodeBinding,
    served_phase_neg: bool,
    required_at_output: f32,
    phase_match: &mut [NodePhaseMatch],
    tech: &TechLib,
    wire_delay: f32,
) {
    let extra_out_inv = if binding.phase_neg != served_phase_neg {
        tech.inv_delay
    } else {
        0.0
    };
    match &binding.kind {
        NodeBindingKind::Direct { cell } => {
            let out_inv = if cell.out_neg { tech.inv_delay } else { 0.0 };
            for i in 0..(cell.n_inputs as usize) {
                let Some(&leaf) = binding.leaves.get(i) else {
                    continue;
                };
                if leaf == 0 || leaf >= phase_match.len() {
                    continue;
                }
                if !matches!(aig.node_type(NodeIndex::new(leaf)), AigNode::And) {
                    continue;
                }
                let leaf_phase = usize::from(((cell.neg_mask >> i) & 1) == 1);
                let inv_d = if leaf_phase == 1 { tech.inv_delay } else { 0.0 };
                let cand = required_at_output
                    - wire_delay
                    - extra_out_inv
                    - out_inv
                    - cell.pin_delay[i]
                    - inv_d;
                phase_match[leaf].required[leaf_phase] =
                    phase_match[leaf].required[leaf_phase].min(cand);
                phase_match[leaf].map_refs[leaf_phase] =
                    phase_match[leaf].map_refs[leaf_phase].saturating_add(1);
            }
        }
        NodeBindingKind::MappedCover { .. } => {
            let cand = required_at_output - wire_delay - extra_out_inv;
            for &leaf in &binding.leaves {
                if leaf == 0 || leaf >= phase_match.len() {
                    continue;
                }
                if !matches!(aig.node_type(NodeIndex::new(leaf)), AigNode::And) {
                    continue;
                }
                phase_match[leaf].required[0] = phase_match[leaf].required[0].min(cand);
                phase_match[leaf].map_refs[0] = phase_match[leaf].map_refs[0].saturating_add(1);
            }
        }
    }
}

#[derive(Clone, Debug)]
struct MapOnceResult {
    best: Vec<Option<NodeBinding>>,
    /// A valid **negative-phase** realization per node, when the pass kept one (`phase_neg == true`).
    ///
    /// `best` materialises one binding per node, so the only way to obtain the complemented output
    /// of a node without an inverter is to also keep the binding that realizes `~f` natively. The
    /// delay flow computes exactly that (`DualNodeBinding::neg`); dropping it here is what made the
    /// emitted netlist 3x slower than the arrival model the mapper optimised.
    neg_alt: Vec<Option<NodeBinding>>,
    arrival: Vec<f32>,
    perf: EmapPerf,
    /// Per-node cut pool after Stage 1 (mockturtle-style: fixed for later passes).
    cuts: Vec<Vec<Cut>>,
}

/// Technology mapping result for an AIG.
///
/// ## QoR numbers vs. ABC `print_stats` on emitted Verilog
///
/// - **`total_area`**: Sum of [`NodeBinding::area`] over bindings. The mapper charges inverter
///   area per *variant* (negated inputs / `out_neg`), while the netlist may **share** one inverter
///   wire (`…_inv`) across multiple cell inputs, and may add **extra** inverters not folded into
///   these sums (e.g. output phase, const fallback). ABC `area` is the sum of genlib cell areas over
///   **each instance** after `read_genlib; read -m`.
/// - **`max_delay`**: Maximum PO arrival from the mapper’s arrival propagation (genlib pin delays).
///   ABC `delay` is recomputed on the mapped network with ABC’s view of the graph; same numeric
///   scale (library units) but path/criticality can differ slightly.
/// - **`bindings.len()`**: Count of mapped AIG node bindings (one logical cover per AND node), not
///   ABC `nd` (node count in ABC’s internal network after reading Verilog).
#[derive(Clone, Debug, Default)]
pub struct EmapResult {
    pub bindings: Vec<NodeBinding>,
    /// Sum of per-binding area models; see struct docs vs. ABC instance area.
    pub total_area: f32,
    /// Area of the **emitted** netlist: one cell per binding plus one inverter per *deduplicated*
    /// inverted net, i.e. exactly what [`write_mapped_verilog`] materialises.
    ///
    /// `total_area` charges an `inv_area` for every inverted pin of every binding, while the writer
    /// collects the inverted nets into a `HashSet` and inserts a single inverter per net. When one
    /// inverted signal feeds several consumers the model charges it several times: measured on c432
    /// `total_area` is 16.030 while the emitted Verilog sums to 15.710 (93 inverters emitted for a
    /// model that charges ~67 equivalent ones). Prefer this field for head-to-head comparisons.
    pub materialized_area: f32,
    /// Worst PO delay in mapper’s delay model; see struct docs vs. ABC `print_stats` delay.
    pub max_delay: f32,
    pub perf: EmapPerf,
    pub mapper_name: &'static str,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EmapPerf {
    pub cut_gen: Duration,
    pub egg: Duration,
    pub tt: Duration,
    pub tech_match: Duration,
    pub direct_tt_hit: u64,
    pub egg_attempt: u64,
    pub egg_build_ok: u64,
    pub egg_coverable_hit: u64,
    pub egg_proxy_fallback: u64,
    pub egg_reject_non_improving: u64,
    pub egg_miss: u64,
    pub direct_variant_reject: u64,
    pub supergate_hit: u64,
    pub supergate_store: u64,
}

impl EmapPerf {
    #[must_use]
    pub fn total_accounted(&self) -> Duration {
        self.cut_gen + self.egg + self.tt + self.tech_match
    }

    #[must_use]
    pub fn ratio_percent(&self) -> (f64, f64, f64, f64) {
        let total = self.total_accounted().as_secs_f64();
        if total <= f64::EPSILON {
            return (0.0, 0.0, 0.0, 0.0);
        }
        (
            self.cut_gen.as_secs_f64() * 100.0 / total,
            self.egg.as_secs_f64() * 100.0 / total,
            self.tt.as_secs_f64() * 100.0 / total,
            self.tech_match.as_secs_f64() * 100.0 / total,
        )
    }

    fn merge_from(&mut self, other: &Self) {
        self.cut_gen += other.cut_gen;
        self.egg += other.egg;
        self.tt += other.tt;
        self.tech_match += other.tech_match;
        self.direct_tt_hit += other.direct_tt_hit;
        self.egg_attempt += other.egg_attempt;
        self.egg_build_ok += other.egg_build_ok;
        self.egg_coverable_hit += other.egg_coverable_hit;
        self.egg_proxy_fallback += other.egg_proxy_fallback;
        self.egg_reject_non_improving += other.egg_reject_non_improving;
        self.egg_miss += other.egg_miss;
        self.direct_variant_reject += other.direct_variant_reject;
        self.supergate_hit += other.supergate_hit;
        self.supergate_store += other.supergate_store;
    }
}

type TtExprCache = HashMap<String, (u8, Tt9)>;
type CoverChoiceCache = HashMap<
    (
        String,
        MappingObjective,
        u32,
        Option<u64>,
        Option<u64>,
        usize,
    ),
    Option<Vec<MappedCoverChoice>>,
>;

#[inline]
fn merge_tt_cache(into: &mut TtExprCache, from: TtExprCache) {
    into.extend(from);
}

#[inline]
fn merge_cover_cache(into: &mut CoverChoiceCache, from: CoverChoiceCache) {
    into.extend(from);
}

/// When unset or non-`0`/`false`, score multiple candidate cuts per AND node in parallel (rayon).
#[inline]
fn emapper_parallel_cuts_within_node() -> bool {
    match std::env::var("EMAPPER_PARALLEL_CUTS") {
        Ok(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
        _ => true,
    }
}

/// Stable bucket for cover-cache keys: `u32::MAX` means no timing limit.
#[inline]
/// Bucket a required output time into the `supergate_db`/cover-cache key space.
///
/// `None` and non-finite/effectively-unbounded values map to `u32::MAX`, which is the bucket used
/// by cut mapping without a delay constraint (including `epmapping`'s calls).
#[must_use]
pub fn cover_cache_req_bits(req: Option<f32>) -> u32 {
    match req {
        None => u32::MAX,
        Some(r) if !r.is_finite() || r > f32::MAX * 0.49 => u32::MAX,
        Some(r) => r.to_bits(),
    }
}

#[inline]
fn cover_cache_native_digest(np: &[bool], nn: &[bool]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    np.hash(&mut h);
    nn.hash(&mut h);
    h.finish()
}

#[inline]
fn cover_cache_phase_digest(phases: &[PhaseArrival]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for p in phases {
        p.pos.to_bits().hash(&mut h);
        p.neg.to_bits().hash(&mut h);
    }
    h.finish()
}

#[inline]
fn finite_required_at(required: Option<&[f32]>, index: usize) -> Option<f32> {
    required
        .and_then(|r| r.get(index).copied())
        .filter(|&x| x.is_finite() && x < f32::MAX * 0.5)
}

#[inline]
fn finite_phase_required(r: f32) -> Option<f32> {
    (r.is_finite() && r < f32::MAX * 0.5).then_some(r)
}

/// Egraph-based technology mapper on AIG-style cuts/expressions.
///
/// - Traverses nodes in topological order
/// - Stage 1 enumerates per-node cuts (merge children + trivial); the resulting cut pool is
///   **fixed** for area-flow / exact-area recovery (mockturtle-style: later passes only re-match).
/// - For each cut, performs direct techlib matching first; falls back to egg path if needed
/// - Stores one best binding per node and backtraces from outputs
pub fn emap(
    aig: &AigGraph,
    tech: &TechLib,
    max_cuts: usize,
    max_inputs: usize,
    wire_delay: f32,
) -> EmapResult {
    emap_with_params(
        aig,
        tech,
        EmapParams {
            max_cuts,
            max_inputs,
            wire_delay,
            ..Default::default()
        },
    )
}

/// Merge performance counters from a sub-pass into the accumulator.
fn accumulate_perf(acc: &mut EmapPerf, delta: EmapPerf) {
    acc.cut_gen += delta.cut_gen;
    acc.egg += delta.egg;
    acc.tt += delta.tt;
    acc.tech_match += delta.tech_match;
    acc.direct_tt_hit += delta.direct_tt_hit;
    acc.egg_attempt += delta.egg_attempt;
    acc.egg_build_ok += delta.egg_build_ok;
    acc.egg_coverable_hit += delta.egg_coverable_hit;
    acc.egg_proxy_fallback += delta.egg_proxy_fallback;
    acc.egg_reject_non_improving += delta.egg_reject_non_improving;
    acc.egg_miss += delta.egg_miss;
    acc.direct_variant_reject += delta.direct_variant_reject;
    acc.supergate_hit += delta.supergate_hit;
    acc.supergate_store += delta.supergate_store;
}

/// Where the emitted inverters come from, using the writer's `inv_srcs` rule.
///
/// Returns `(pin_inversions, output_inversions, po_inversions, per_cell)` where `per_cell` counts the
/// pin inversions attributed to each standard-cell name (the dominant term), so the inverter cost can
/// be attacked at its source instead of guessed at.
#[must_use]
pub fn inverter_breakdown(
    aig: &AigGraph,
    tech: &TechLib,
    res: &EmapResult,
) -> (usize, usize, usize, Vec<(String, usize)>) {
    let has_neg: HashSet<usize> = res
        .bindings
        .iter()
        .filter(|b| b.phase_neg)
        .map(|b| b.node)
        .collect();
    let pi_nodes: HashSet<usize> = aig.input_nodes().iter().map(|n| n.index()).collect();
    let mut pin_inv: HashSet<(usize, usize)> = HashSet::new();
    let mut per_cell: HashMap<String, usize> = HashMap::new();
    let mut out_neg = 0usize;
    for b in &res.bindings {
        if let NodeBindingKind::Direct { cell } = &b.kind {
            for i in 0..usize::from(cell.n_inputs) {
                let Some(&leaf) = b.leaves.get(i) else {
                    continue;
                };
                if leaf == 0 || ((cell.neg_mask >> i) & 1) == 0 {
                    continue;
                }
                if !has_neg.contains(&leaf) && pin_inv.insert((leaf, b.node)) {
                    *per_cell.entry(cell.cell_name.clone()).or_insert(0) += 1;
                }
            }
            if cell.out_neg {
                out_neg += 1;
            }
        }
    }
    let mut po_inv = 0usize;
    for out in aig.graph().externals(Outgoing) {
        for edge in aig.graph().edges_directed(out, Incoming) {
            let d = edge.source().index();
            if *edge.weight() && d != 0 && !pi_nodes.contains(&d) && !has_neg.contains(&d) {
                po_inv += 1;
            }
        }
    }
    let mut v: Vec<(String, usize)> = per_cell.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    (pin_inv.len(), out_neg, po_inv, v)
}

/// Area of the netlist [`write_mapped_verilog`] would emit: bare cell area per binding plus one
/// inverter per deduplicated inverted net.
///
/// Mirrors the writer's `inv_srcs` rule: an inverter is emitted for an inverted pin only when the
/// leaf has no negative-phase binding of its own, and for an inverted primary output that is not a
/// primary input. Cover-internal inversion is already part of the cover's cells (the cover graph
/// carries its own inverters as cells), so only the Direct-binding pins and the POs are considered.
#[must_use]
pub fn materialized_area(aig: &AigGraph, tech: &TechLib, res: &EmapResult) -> f32 {
    fn base_area(tech: &TechLib, name: &str, fallback: f32) -> f32 {
        tech.cells
            .iter()
            .find(|c| c.name == name)
            .map_or(fallback, |c| c.area)
    }

    let has_neg_binding: HashSet<usize> = res
        .bindings
        .iter()
        .filter(|b| b.phase_neg)
        .map(|b| b.node)
        .collect();
    let pi_nodes: HashSet<usize> = aig.input_nodes().iter().map(|n| n.index()).collect();

    let mut area = 0.0f32;
    let mut inv_nets: HashSet<usize> = HashSet::new();
    for b in &res.bindings {
        match &b.kind {
            NodeBindingKind::Direct { cell } => {
                area += base_area(tech, &cell.cell_name, cell.area);
                for i in 0..usize::from(cell.n_inputs) {
                    let Some(&leaf) = b.leaves.get(i) else {
                        continue;
                    };
                    if leaf == 0 || ((cell.neg_mask >> i) & 1) == 0 {
                        continue;
                    }
                    if !has_neg_binding.contains(&leaf) {
                        inv_nets.insert(leaf);
                    }
                }
                if cell.out_neg {
                    inv_nets.insert(usize::MAX - b.node);
                }
            }
            NodeBindingKind::MappedCover { graph, .. } => {
                for node in &graph.nodes {
                    if let MappedCoverNode::Cell { cell, .. } = node {
                        area += base_area(tech, &cell.cell_name, cell.area);
                    }
                }
            }
        }
    }
    for out in aig.graph().externals(Outgoing) {
        for edge in aig.graph().edges_directed(out, Incoming) {
            let driver = edge.source().index();
            if *edge.weight()
                && driver != 0
                && !pi_nodes.contains(&driver)
                && !has_neg_binding.contains(&driver)
            {
                inv_nets.insert(driver);
            }
        }
    }
    area + inv_nets.len() as f32 * tech.inv_area
}

pub fn emap_with_params(aig: &AigGraph, tech: &TechLib, ps: EmapParams) -> EmapResult {
    /* Two independent either/ors, neither of which dominates:
    - phase-arrival strictness: the strict model is better on c432 (360.4 vs 391.7), c880
      (355.1 vs 369.6) and c1355 (343.2 vs 350.4); crediting a native negative phase is better on
      c2670 (290.6 vs 323.3) and c5315 (564.5 vs 618.4);
    - dual-phase Stage-1 flow: turning it off keeps c7552 and c880 at the same delay for ~2 %
      less area (130.30 -> 128.02, 24.41 -> 23.79), while costing c2670 and c5315 delay.
    - the two-level supergate fallback: enabling it is better on c5315 (564.45 vs 571.33, and
      cheaper in area), disabling it is far better on c7552 (408.99/17 levels vs 461.90/19).
    So map all eight combinations and keep the best `(max_delay, materialized_area)`. Monotone by
    construction, so it can only match or beat any single setting; `EMAPPER_PHASE_PORTFOLIO=0`
    restores the single-model behaviour. Cost is 8 mapping runs (c5315 ~4 s -> ~30 s). */
    if !std::env::var("EMAPPER_PHASE_PORTFOLIO").is_ok_and(|v| v == "0") {
        let mut best: Option<EmapResult> = None;
        for (pessimistic, dual_phase, two_level) in [
            (true, true, true),
            (false, true, true),
            (true, false, true),
            (false, false, true),
            (true, true, false),
            (false, true, false),
            (true, false, false),
            (false, false, false),
        ] {
            let mut variant = ps.clone();
            variant.pessimistic_phase = Some(pessimistic);
            variant.use_dual_phase_delay = ps.use_dual_phase_delay && dual_phase;
            variant.use_two_level_cover = ps.use_two_level_cover && two_level;
            let produced = emap_core(aig, tech, variant);
            let better = best.as_ref().is_none_or(|b| {
                (produced.max_delay, produced.materialized_area)
                    < (b.max_delay, b.materialized_area)
            });
            if better {
                best = Some(produced);
            }
        }
        if let Some(r) = best {
            return r;
        }
    }
    emap_core(aig, tech, ps)
}

fn emap_core(aig: &AigGraph, tech: &TechLib, ps: EmapParams) -> EmapResult {
    assert!(!petgraph::algo::is_cyclic_directed(aig.graph()));

    reset_egg_budget();
    WIDE_TT_CANDIDATES.store(0, std::sync::atomic::Ordering::Relaxed);
    WIDE_TT_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
    let egg_bundle_ref = ps.use_egg.then(|| tech.egg_mapping_bundle().as_ref());

    // ── Stage 1a: delay flow ────────────────────────────────────────────────
    // Dual-phase sweep produces tighter delay bounds by propagating the
    // cheaper phase (pos / neg) to every fanout without extra inverters.
    let mut ps_delay = ps.clone();
    ps_delay.objective = MappingObjective::DelayFirst;

    let MapOnceResult {
        mut best,
        neg_alt: mut stage1_neg_alt,
        mut arrival,
        mut perf,
        cuts: mut stage1_cuts,
        ..
    } = if ps_delay.use_dual_phase_delay {
        delay_flow_pass(aig, tech, &ps_delay, egg_bundle_ref)
    } else {
        constrained_single_phase_pass(
            aig,
            tech,
            &ps_delay,
            None,
            None,
            egg_bundle_ref,
            None,
            false,
        )
    };

    // ── Stage 1b: rematch on the Stage 1a cut pool with known critical path ─
    // Enables supergate leaf-delay policy (and optional mixed per-node objective when
    // `critical_path_mixed_objective` is set) before area recovery.
    let crit_after_s1a = collect_critical_nodes(aig, tech, &best, &arrival, ps_delay.wire_delay);
    let required_after_s1a =
        compute_required_times(aig, tech, &best, &arrival, ps_delay.wire_delay);
    let MapOnceResult {
        best: best2,
        perf: perf2,
        cuts: cuts2,
        ..
    } = constrained_single_phase_pass(
        aig,
        tech,
        &ps_delay,
        Some(&crit_after_s1a),
        Some(&required_after_s1a),
        egg_bundle_ref,
        Some(&stage1_cuts),
        ps_delay.critical_path_mixed_objective,
    );
    best = best2;
    stage1_cuts = cuts2;
    accumulate_perf(&mut perf, perf2);
    let mut final_bindings_override: Option<Vec<NodeBinding>> = None;

    // Recompute arrivals phase-aware after Stage 1.
    arrival = recompute_binding_arrivals(aig, tech, &mut best, ps_delay.wire_delay);

    /* ── Stage 1c: required-time rematch to a fixpoint ─────────────────────────────────────────
    Stage 1a is a greedy forward delay flow: every node picks the cut that minimises its arrival
    given the leaf arrivals available at that moment. Stage 1b rematches once on the frozen cut
    pool, but with required times derived from the same greedy solution. Recomputing the required
    times from the *achieved* mapping and rematching again moves the binding choices onto the
    critical path, which is what the delay-oriented mappers do (and what the area passes already
    do for area). Accept-only-if-the-primary-output-delay-improves keeps this monotone, so it can
    never regress. */
    let delay_iters = delay_fixpoint_iters();
    for _ in 0..delay_iters {
        let cur = primary_output_max_delay(aig, tech, &best, &arrival);
        let crit = collect_critical_nodes(aig, tech, &best, &arrival, ps_delay.wire_delay);
        let required = compute_required_times(aig, tech, &best, &arrival, ps_delay.wire_delay);
        let MapOnceResult {
            best: r_best,
            perf: r_perf,
            ..
        } = constrained_single_phase_pass(
            aig,
            tech,
            &ps_delay,
            Some(&crit),
            Some(&required),
            egg_bundle_ref,
            Some(&stage1_cuts),
            false,
        );
        accumulate_perf(&mut perf, r_perf);
        let mut r_best = r_best;
        let r_arrival = recompute_binding_arrivals(aig, tech, &mut r_best, ps_delay.wire_delay);
        let new_delay = primary_output_max_delay(aig, tech, &r_best, &r_arrival);
        if new_delay < cur - 1e-3 {
            best = r_best;
            arrival = r_arrival;
        } else {
            break;
        }
    }
    let delay_budget = primary_output_max_delay(aig, tech, &best, &arrival);
    let delay_required = compute_required_times(aig, tech, &best, &arrival, ps_delay.wire_delay);

    // ── Stage 2: area-flow recovery ─────────────────────────────────────────
    for _ in 0..ps.area_flow_passes {
        let (mut best2, _arrival2, perf2) = area_flow_recovery_pass(
            aig,
            tech,
            &ps,
            &best,
            &arrival,
            Some(&delay_required),
            egg_bundle_ref,
            &stage1_cuts,
        );
        let arrival2 = recompute_binding_arrivals(aig, tech, &mut best2, ps.wire_delay);
        if primary_output_max_delay(aig, tech, &best2, &arrival2) <= delay_budget + 1e-3 {
            best = best2;
            arrival = arrival2;
            final_bindings_override = None;
        }
        accumulate_perf(&mut perf, perf2);
    }

    // ── Stage 3: exact-area recovery ────────────────────────────────────────
    for _ in 0..ps.exact_area_passes {
        let (mut best2, _arrival2, perf2) = exact_area_recovery_pass(
            aig,
            tech,
            &ps,
            &best,
            &arrival,
            Some(&delay_required),
            egg_bundle_ref,
            &stage1_cuts,
        );
        let arrival2 = recompute_binding_arrivals(aig, tech, &mut best2, ps.wire_delay);
        if primary_output_max_delay(aig, tech, &best2, &arrival2) <= delay_budget + 1e-3 {
            best = best2;
            arrival = arrival2;
            final_bindings_override = None;
        }
        accumulate_perf(&mut perf, perf2);
    }

    /* Snapshot the pre-level mapping. Stage 4/5 normally can only remove depth at equal delay, but
    `EMAPPER_DEPTH_DELAY_SLACK` lets a level be bought with a bounded delay increase, and that is
    no longer monotone in delay -- so keep the old mapping and restore it if the worst primary
    output got slower. */
    let pre_level = if depth_delay_slack_enabled() {
        Some((best.clone(), arrival.clone()))
    } else {
        None
    };

    // ── Stage 4: level (cell-depth) recovery ────────────────────────────────
    // Monotone in both delay and level, so it can only remove depth the earlier passes left behind.
    for _ in 0..ps.depth_recovery_passes {
        let perf2 = depth_recovery_pass(
            aig,
            tech,
            &ps,
            &mut best,
            &mut arrival,
            Some(&delay_required),
            egg_bundle_ref,
            &stage1_cuts,
        );
        accumulate_perf(&mut perf, perf2);
    }
    arrival = recompute_binding_arrivals(aig, tech, &mut best, ps.wire_delay);

    /* ── Stage 5: re-run area recovery against the *post-level* delay ─────────
    Stage 4 buys a large delay reduction (c432 484.8 -> 387.8 at slack=inf) but pays area for the
    shallower bindings. Stages 2/3 ran before that and used the (much looser) Stage-1 budget, so
    re-running them here with the achieved delay as the new budget spends the slack on area
    instead of leaving it on the table. Whatever they take back, Stage 6 then re-levels. */
    if ps.depth_recovery_passes > 0 {
        let level_budget = primary_output_max_delay(aig, tech, &best, &arrival);
        let level_required = compute_required_times(aig, tech, &best, &arrival, ps.wire_delay);
        for _ in 0..ps.area_flow_passes.max(ps.exact_area_passes) {
            let (mut best2, _, perf2) = area_flow_recovery_pass(
                aig,
                tech,
                &ps,
                &best,
                &arrival,
                Some(&level_required),
                egg_bundle_ref,
                &stage1_cuts,
            );
            accumulate_perf(&mut perf, perf2);
            let arrival2 = recompute_binding_arrivals(aig, tech, &mut best2, ps.wire_delay);
            if primary_output_max_delay(aig, tech, &best2, &arrival2) <= level_budget + 1e-3 {
                best = best2;
                arrival = arrival2;
            }

            let (mut best3, _, perf3) = exact_area_recovery_pass(
                aig,
                tech,
                &ps,
                &best,
                &arrival,
                Some(&level_required),
                egg_bundle_ref,
                &stage1_cuts,
            );
            accumulate_perf(&mut perf, perf3);
            let arrival3 = recompute_binding_arrivals(aig, tech, &mut best3, ps.wire_delay);
            if primary_output_max_delay(aig, tech, &best3, &arrival3) <= level_budget + 1e-3 {
                best = best3;
                arrival = arrival3;
            }

            // Re-level after the area round: the area objective is free to pick deeper bindings.
            let perf4 = depth_recovery_pass(
                aig,
                tech,
                &ps,
                &mut best,
                &mut arrival,
                Some(&level_required),
                egg_bundle_ref,
                &stage1_cuts,
            );
            accumulate_perf(&mut perf, perf4);
            arrival = recompute_binding_arrivals(aig, tech, &mut best, ps.wire_delay);
        }
    }

    if let Some((pre_best, pre_arrival)) = pre_level {
        let before = primary_output_max_delay(aig, tech, &pre_best, &pre_arrival);
        let after = primary_output_max_delay(aig, tech, &best, &arrival);
        if after > before + 1e-3 {
            best = pre_best;
            arrival = pre_arrival;
        }
    }

    /* Phase-aware collection: a node's negative output is taken from the negative-phase realization
    the delay flow kept, instead of always paying an inverter on top of the positive-phase binding.
    `collect_required_bindings_with` keys its result by `(node, phase_neg)` and the writer keys its
    lookup the same way, so materialising both realizations is exactly what the data model
    supports.

    A negative-phase realization is a **second cell**, not a free inversion, so which realization
    to materialise is a real choice, and it is made from `neg_alt`. `stage1_neg_alt` comes from
    Stage 1a, but Stages 1b/1c/2/3/4/5 replace most of `best` afterwards, so the two arrays can
    disagree: on c5315 that produced an `OR4 -> INV -> OR4 -> INV` alternation on the critical
    path (each stage's realized phase no longer matching what its consumers expect), while
    refreshing them for the *final* cuts fixes c5315 (-4.2 %) but hurts c2670 (+13.8 %). Neither
    is uniformly better, so evaluate both and keep the better `(max_delay, total_area)`. The
    choice is monotone -- it can only match or beat the historical behaviour. */
    let phase_crit = collect_critical_nodes(aig, tech, &best, &arrival, ps.wire_delay);
    let dual_phase = dual_phase_materialization();
    let dual_neg = if dual_phase {
        let (_p, n) = recompute_dual_arrivals(aig, tech, &mut best, &stage1_neg_alt, ps.wire_delay);
        Some(n)
    } else {
        None
    };

    let collect_with = |neg_alt: &[Option<NodeBinding>]| -> Vec<NodeBinding> {
        if let Some(ov) = final_bindings_override.as_ref() {
            return ov.clone();
        }
        if phase_demand_selection() {
            collect_phase_aware_bindings(aig, tech, &best, neg_alt)
        } else {
            collect_required_bindings_with(aig, |node, want_neg| {
                let pick = |v: &Vec<Option<NodeBinding>>| v.get(node).and_then(|b| b.clone());
                if dual_phase && want_neg && phase_crit.contains(&node) {
                    pick(&stage1_neg_alt).or_else(|| pick(&best))
                } else {
                    pick(&best)
                }
            })
        }
    };

    let evaluate = |neg_alt: &[Option<NodeBinding>]| -> (Vec<NodeBinding>, f32, f32) {
        let mut bindings = collect_with(neg_alt);
        close_binding_set(aig, tech, &mut bindings, &best, &arrival, ps.wire_delay);
        bindings.sort_by_key(|b| (b.node, b.phase_neg));
        let total_area = bindings.iter().map(|b| b.area).sum::<f32>();
        let breakdown = po_arrival_breakdown(aig, tech, &bindings, &arrival, dual_neg.as_ref());
        let max_delay = breakdown.iter().map(|r| r.0).fold(0.0, f32::max);
        (bindings, total_area, max_delay)
    };

    let mut chosen = evaluate(&stage1_neg_alt);
    if std::env::var("EMAPPER_REFRESH_NEG_ALT").map_or(true, |v| v != "0") {
        let mut refreshed = stage1_neg_alt.clone();
        let perf2 = refresh_negative_alternatives(
            aig,
            tech,
            &ps,
            &best,
            &arrival,
            egg_bundle_ref,
            &mut refreshed,
        );
        accumulate_perf(&mut perf, perf2);
        let alternative = evaluate(&refreshed);
        if (alternative.2, alternative.1) < (chosen.2, chosen.1) {
            chosen = alternative;
        }
    }
    let (bindings, total_area, max_delay) = chosen;

    if std::env::var("EMAPPER_DEBUG_PO").is_ok_and(|v| v == "3") {
        let mut rows = po_arrival_breakdown(aig, tech, &bindings, &arrival, dual_neg.as_ref());
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        eprintln!("[po-crit] internal_po_max={max_delay:.2}");
        for (v, driver, neg, bound, kind) in rows.iter().take(8) {
            eprintln!("  {v:8.2} driver=n{driver} neg={neg} in_bindings={bound} {kind}");
        }
    }

    if std::env::var_os("EMAPPER_DEBUG_CUTS").is_some() {
        let cand = WIDE_TT_CANDIDATES.load(std::sync::atomic::Ordering::Relaxed);
        let hits = WIDE_TT_HITS.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "[wide-tt] compound(>=3-input) function lookups={cand} matched={hits} ({:.1}%)",
            100.0 * hits as f64 / cand.max(1) as f64
        );
    }

    let mut result = EmapResult {
        bindings,
        total_area,
        materialized_area: 0.0,
        max_delay,
        perf,
        mapper_name: ps.mapper_name,
    };
    result.materialized_area = materialized_area(aig, tech, &result);
    result
}

/// Leaf-flow plan for a **fixed** binding map (`best`), used by the area-recovery passes.
///
/// Those passes used to hand `score_cut` a `None` flow map, so a `MappedCover` binding was ranked on
/// its raw physical area (`build_mapped_cover_binding` sets `mapping_cost_area = cover.area`) while a
/// `Direct` binding was ranked on the mockturtle flow proxy `cell.area + Σ leaf_flow`. Comparing the
/// two directly makes a cover look 3-4x more expensive for the same logic -- measured on c432 the
/// cover wins the node-level area comparison at **0 of 176** nodes, which is why the final mapping
/// only kept 11 covers and degenerated to 2-input cells.
/// `EMAPPER_RECOVERY_FLOW=1`: hand the area-recovery passes the leaf-flow map (default off).
///
/// Measured on c432/c5315 with the flow map threaded in: cover bindings stayed at 11 / 0, area and
/// delay both got slightly **worse** (c432 16.03/509.6 -> 16.13/524.8, c5315 134.15 -> 135.66) and
/// c880 went from 12 s to a timeout. So the asymmetric flow ranking was not what was keeping covers
/// out -- unifying the metric does not make them win.
fn recovery_flow_ref(pf: &[PhaseFlow]) -> Option<&[PhaseFlow]> {
    static V: OnceLock<bool> = OnceLock::new();
    if *V.get_or_init(|| std::env::var("EMAPPER_RECOVERY_FLOW").is_ok_and(|v| v == "1")) {
        Some(pf)
    } else {
        None
    }
}

fn plan_phase_flow(
    aig: &AigGraph,
    tech: &TechLib,
    best: &[Option<NodeBinding>],
    est_refs: &[f32],
    phase_flow: &mut [PhaseFlow],
) {
    for pi in aig.input_nodes() {
        let i = pi.index();
        phase_flow[i] = PhaseFlow {
            pos: 0.0,
            neg: tech.inv_area / est_refs[i].max(1.0),
        };
    }
    let mut topo = petgraph::visit::Topo::new(aig.graph());
    while let Some(node) = topo.next(aig.graph()) {
        let idx = node.index();
        if !matches!(aig.node_type(node), AigNode::And) || idx >= phase_flow.len() {
            continue;
        }
        if let Some(b) = best.get(idx).and_then(|b| b.as_ref()) {
            let f = binding_phase_flow(b, aig, &*phase_flow);
            let refs = est_refs.get(idx).copied().unwrap_or(1.0).max(1.0);
            phase_flow[idx] = PhaseFlow {
                pos: f / refs,
                neg: (f + tech.inv_area) / refs,
            };
        }
    }
}

fn area_flow_recovery_pass(
    aig: &AigGraph,
    tech: &TechLib,
    ps: &EmapParams,
    best_in: &[Option<NodeBinding>],
    arrival_in: &[f32],
    required: Option<&[f32]>,
    egg_bundle: Option<&EggRewriteBundle>,
    cuts: &[Vec<Cut>],
) -> (Vec<Option<NodeBinding>>, Vec<f32>, EmapPerf) {
    let len = best_in.len();
    debug_assert_eq!(
        cuts.len(),
        len,
        "cut pool length must match mapping slot count"
    );
    let mut best = best_in.to_vec();
    let mut arrival = arrival_in.to_vec();
    let mut perf = EmapPerf::default();
    let est_refs_flow = compute_est_refs(aig, len);
    let mut phase_flow: Vec<PhaseFlow> = vec![PhaseFlow::default(); len];
    plan_phase_flow(aig, tech, best_in, &est_refs_flow, &mut phase_flow);
    let mut tt_cache: HashMap<String, (u8, Tt9)> = HashMap::new();
    let mut cover_cache: CoverChoiceCache = HashMap::new();
    let mut topo_nodes = Vec::new();
    let mut topo = petgraph::visit::Topo::new(aig.graph());
    while let Some(node) = topo.next(aig.graph()) {
        topo_nodes.push(node.index());
    }

    let mut topo_rank = vec![usize::MAX; len];
    for (pos, &node) in topo_nodes.iter().enumerate() {
        topo_rank[node] = pos;
    }
    let mut fanouts = compute_binding_fanouts(&best);
    let critical_nodes = collect_critical_nodes(aig, tech, &best, &arrival, ps.wire_delay);

    for &idx in &topo_nodes {
        if !matches!(aig.node_type(NodeIndex::new(idx)), AigNode::And) {
            continue;
        }
        let Some(old_binding) = best[idx].clone() else {
            continue;
        };
        let refs = compute_binding_refs(aig, &best);
        let mut best_cand = old_binding.clone();
        let mut best_delta = 0.0f32;
        for cut in cuts
            .get(idx)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
            .iter()
            .cloned()
        {
            if let Some(cand) = score_cut(
                aig,
                tech,
                &arrival,
                &best,
                &cut,
                ps,
                MappingObjective::AreaFirst,
                &mut tt_cache,
                &mut cover_cache,
                &mut perf,
                egg_bundle,
                required,
                recovery_flow_ref(&phase_flow),
                Some(&critical_nodes),
            ) {
                let within_required = required.is_none_or(|req| {
                    cand.delay <= req.get(idx).copied().unwrap_or(f32::INFINITY) + 1e-3
                });
                if !within_required {
                    continue;
                }
                let delta = local_exact_area_delta(idx, &old_binding, &cand, &best, &refs);
                if delta < best_delta
                    && replacement_respects_timing(
                        &mut best,
                        &mut arrival,
                        &fanouts,
                        idx,
                        &cand,
                        tech,
                        ps.wire_delay,
                        required,
                        &topo_rank,
                    )
                {
                    best_delta = delta;
                    best_cand = cand;
                }
            }
        }
        if best_delta < -1e-6 {
            commit_binding_replacement(
                &mut best,
                &mut arrival,
                &mut fanouts,
                idx,
                best_cand,
                tech,
                ps.wire_delay,
                required,
                &topo_rank,
            );
        }
    }

    (best, arrival, perf)
}

fn exact_area_recovery_pass(
    aig: &AigGraph,
    tech: &TechLib,
    ps: &EmapParams,
    best_in: &[Option<NodeBinding>],
    arrival_in: &[f32],
    required: Option<&[f32]>,
    egg_bundle: Option<&EggRewriteBundle>,
    cuts: &[Vec<Cut>],
) -> (Vec<Option<NodeBinding>>, Vec<f32>, EmapPerf) {
    let len = best_in.len();
    debug_assert_eq!(
        cuts.len(),
        len,
        "cut pool length must match mapping slot count"
    );
    let mut best = best_in.to_vec();
    let mut arrival = arrival_in.to_vec();
    let mut perf = EmapPerf::default();
    let est_refs_flow = compute_est_refs(aig, len);
    let mut phase_flow: Vec<PhaseFlow> = vec![PhaseFlow::default(); len];
    plan_phase_flow(aig, tech, best_in, &est_refs_flow, &mut phase_flow);
    let mut tt_cache: HashMap<String, (u8, Tt9)> = HashMap::new();
    let mut cover_cache: CoverChoiceCache = HashMap::new();
    let mut topo_nodes = Vec::new();
    let mut topo = petgraph::visit::Topo::new(aig.graph());
    while let Some(node) = topo.next(aig.graph()) {
        topo_nodes.push(node.index());
    }

    let mut topo_rank = vec![usize::MAX; len];
    for (pos, &node) in topo_nodes.iter().enumerate() {
        topo_rank[node] = pos;
    }
    let mut fanouts = compute_binding_fanouts(&best);
    let critical_nodes = collect_critical_nodes(aig, tech, &best, &arrival, ps.wire_delay);

    for &idx in topo_nodes.iter().rev() {
        if !matches!(aig.node_type(NodeIndex::new(idx)), AigNode::And) {
            continue;
        }
        let Some(old_binding) = best[idx].clone() else {
            continue;
        };
        let refs = compute_binding_refs(aig, &best);
        let mffc_nodes = collect_mffc_nodes(idx, &best, &refs);
        let mut best_node = idx;
        let mut best_cand = old_binding;
        let mut best_delta = 0.0f32;
        for m in mffc_nodes {
            let Some(old_m) = best[m].clone() else {
                continue;
            };
            let cands_for_m = cuts[m].clone();
            for cut in cands_for_m {
                if let Some(cand) = score_cut(
                    aig,
                    tech,
                    &arrival,
                    &best,
                    &cut,
                    ps,
                    MappingObjective::AreaFirst,
                    &mut tt_cache,
                    &mut cover_cache,
                    &mut perf,
                    egg_bundle,
                    required,
                    recovery_flow_ref(&phase_flow),
                    Some(&critical_nodes),
                ) {
                    let within_required = required.is_none_or(|req| {
                        cand.delay <= req.get(m).copied().unwrap_or(f32::INFINITY) + 1e-3
                    });
                    if !within_required {
                        continue;
                    }
                    let delta = local_exact_area_delta(m, &old_m, &cand, &best, &refs);
                    if delta < best_delta
                        && replacement_respects_timing(
                            &mut best,
                            &mut arrival,
                            &fanouts,
                            m,
                            &cand,
                            tech,
                            ps.wire_delay,
                            required,
                            &topo_rank,
                        )
                    {
                        best_delta = delta;
                        best_node = m;
                        best_cand = cand;
                    }
                }
            }
        }
        if best_delta < -1e-6 {
            commit_binding_replacement(
                &mut best,
                &mut arrival,
                &mut fanouts,
                best_node,
                best_cand,
                tech,
                ps.wire_delay,
                required,
                &topo_rank,
            );
        }
    }
    (best, arrival, perf)
}

fn compute_binding_refs(aig: &AigGraph, best: &[Option<NodeBinding>]) -> Vec<u32> {
    let mut refs = vec![0u32; best.len()];
    for b in best.iter().flatten() {
        for &leaf in &b.leaves {
            if leaf != 0 && leaf < best.len() && best[leaf].is_some() {
                refs[leaf] = refs[leaf].saturating_add(1);
            }
        }
    }
    for driver in aig
        .graph()
        .externals(Outgoing)
        .flat_map(|o| aig.graph().neighbors_directed(o, Incoming))
        .map(|d| d.index())
    {
        if driver < best.len() && best[driver].is_some() {
            refs[driver] = refs[driver].saturating_add(1);
        }
    }
    refs
}

fn local_exact_area_delta(
    node: usize,
    oldb: &NodeBinding,
    newb: &NodeBinding,
    best: &[Option<NodeBinding>],
    refs: &[u32],
) -> f32 {
    let mut tmp = refs.to_vec();
    fn deref_node(node: usize, best: &[Option<NodeBinding>], refs: &mut [u32]) -> f32 {
        if node == 0 || node >= refs.len() || best[node].is_none() {
            return 0.0;
        }
        if refs[node] == 0 {
            return 0.0;
        }
        refs[node] -= 1;
        if refs[node] > 0 {
            return 0.0;
        }
        let b = best[node].as_ref().expect("binding exists");
        let mut removed = b.area;
        for &leaf in &b.leaves {
            removed += deref_node(leaf, best, refs);
        }
        removed
    }

    fn ref_node(node: usize, best: &[Option<NodeBinding>], refs: &mut [u32]) -> f32 {
        if node == 0 || node >= refs.len() || best[node].is_none() {
            return 0.0;
        }
        let was = refs[node];
        refs[node] = refs[node].saturating_add(1);
        if was > 0 {
            return 0.0;
        }
        let b = best[node].as_ref().expect("binding exists");
        let mut added = b.area;
        for &leaf in &b.leaves {
            added += ref_node(leaf, best, refs);
        }
        added
    }

    let mut removed = oldb.area;
    for &leaf in &oldb.leaves {
        removed += deref_node(leaf, best, &mut tmp);
    }

    let mut added = newb.area;
    for &leaf in &newb.leaves {
        if leaf == node {
            continue;
        }
        added += ref_node(leaf, best, &mut tmp);
    }

    added - removed
}

fn collect_mffc_nodes(root: usize, best: &[Option<NodeBinding>], refs: &[u32]) -> Vec<usize> {
    fn walk(
        n: usize,
        best: &[Option<NodeBinding>],
        refs: &[u32],
        out: &mut Vec<usize>,
        seen: &mut HashSet<usize>,
    ) {
        if n == 0 || n >= best.len() || !seen.insert(n) || best[n].is_none() {
            return;
        }
        out.push(n);
        let b = best[n].as_ref().expect("binding exists");
        for &leaf in &b.leaves {
            if leaf != 0 && leaf < refs.len() && refs[leaf] == 1 {
                walk(leaf, best, refs, out, seen);
            }
        }
    }
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    walk(root, best, refs, &mut out, &mut seen);
    out
}

fn compute_binding_fanouts(best: &[Option<NodeBinding>]) -> Vec<Vec<usize>> {
    let mut fanouts = vec![Vec::new(); best.len()];
    for binding in best.iter().flatten() {
        for leaf in binding_dependency_leaves(binding, best.len()) {
            fanouts[leaf].push(binding.node);
        }
    }
    fanouts
}

/// The cut leaves a cover graph **actually references**, in variable order.
///
/// `NodeBinding::leaves` is dense (indexed by cut variable) so the writer can resolve
/// `MappedCoverNode::Input(li)`. Dependency/required-time analysis must instead use only the
/// referenced variables -- otherwise a cover pulls in every leaf of its cut, which perturbs the
/// required-time propagation and the phase-demand statistics. The graph itself records exactly which
/// variables it uses, so no extra field is needed.
///
/// Both call sites were measured against the alternative (dense everywhere, i.e. only
/// `next_required_phases` filtered): sum of delay gaps over c432/c880/c5315/c7552 is **217.8 %** with
/// both filtered vs **243.6 %** with only one, and the sum of area gaps ties (60.5 %) -- so filtering
/// in both places is the better default. Reverting only `binding_dependency_leaves` does recover
/// c5315's area (126.58 -> 124.80) but costs far more delay elsewhere.
fn cover_used_leaves(binding: &NodeBinding) -> Vec<usize> {
    let NodeBindingKind::MappedCover { graph, .. } = &binding.kind else {
        return binding.leaves.clone();
    };
    let mut vars: Vec<usize> = graph
        .nodes
        .iter()
        .filter_map(|n| match n {
            MappedCoverNode::Input(li) => Some(*li),
            _ => None,
        })
        .collect();
    vars.sort_unstable();
    vars.dedup();
    vars.into_iter()
        .filter_map(|li| binding.leaves.get(li).copied())
        .collect()
}

fn binding_dependency_leaves(binding: &NodeBinding, len: usize) -> Vec<usize> {
    let mut leaves = cover_used_leaves(binding)
        .iter()
        .copied()
        .filter(|&leaf| leaf != 0 && leaf < len)
        .collect_vec();
    leaves.sort_unstable();
    leaves.dedup();
    leaves
}

fn update_binding_fanouts(
    fanouts: &mut [Vec<usize>],
    node: usize,
    old_binding: &NodeBinding,
    new_binding: &NodeBinding,
    len: usize,
) {
    for leaf in binding_dependency_leaves(old_binding, len) {
        fanouts[leaf].retain(|&fo| fo != node);
    }
    for leaf in binding_dependency_leaves(new_binding, len) {
        if !fanouts[leaf].contains(&node) {
            fanouts[leaf].push(node);
        }
    }
}

fn collect_tfo_nodes(root: usize, fanouts: &[Vec<usize>], topo_rank: &[usize]) -> Vec<usize> {
    let mut stack = vec![root];
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        out.push(node);
        for &fo in fanouts.get(node).into_iter().flatten() {
            stack.push(fo);
        }
    }
    out.sort_unstable_by_key(|&node| topo_rank.get(node).copied().unwrap_or(usize::MAX));
    out
}

fn try_binding_replacement(
    best: &mut [Option<NodeBinding>],
    arrival: &mut [f32],
    fanouts: &mut [Vec<usize>],
    node: usize,
    new_binding: NodeBinding,
    tech: &TechLib,
    wire_delay: f32,
    required: Option<&[f32]>,
    topo_rank: &[usize],
    commit: bool,
) -> bool {
    let Some(old_binding) = best[node].clone() else {
        return false;
    };
    let affected = collect_tfo_nodes(node, fanouts, topo_rank);
    let saved_arrivals = affected
        .iter()
        .map(|&n| (n, arrival.get(n).copied().unwrap_or(0.0)))
        .collect_vec();

    best[node] = Some(new_binding.clone());
    let mut legal = true;
    for &n in &affected {
        let Some(binding) = best[n].as_mut() else {
            continue;
        };
        let Some(delay) = recompute_binding_delay(binding, arrival, tech, wire_delay) else {
            legal = false;
            break;
        };
        let lim = required
            .and_then(|req| req.get(n).copied())
            .unwrap_or(f32::INFINITY);
        if delay > lim + 1e-3 {
            legal = false;
            break;
        }
        binding.delay = delay;
        arrival[n] = delay;
    }

    if legal && commit {
        update_binding_fanouts(fanouts, node, &old_binding, &new_binding, best.len());
        return true;
    }

    best[node] = Some(old_binding);
    for (n, old_delay) in saved_arrivals {
        arrival[n] = old_delay;
        if let Some(binding) = best[n].as_mut() {
            binding.delay = old_delay;
        }
    }
    legal
}

fn replacement_respects_timing(
    best: &mut [Option<NodeBinding>],
    arrival: &mut [f32],
    fanouts: &[Vec<usize>],
    node: usize,
    new_binding: &NodeBinding,
    tech: &TechLib,
    wire_delay: f32,
    required: Option<&[f32]>,
    topo_rank: &[usize],
) -> bool {
    let mut fanouts_tmp = fanouts.to_vec();
    try_binding_replacement(
        best,
        arrival,
        &mut fanouts_tmp,
        node,
        new_binding.clone(),
        tech,
        wire_delay,
        required,
        topo_rank,
        false,
    )
}

fn commit_binding_replacement(
    best: &mut [Option<NodeBinding>],
    arrival: &mut [f32],
    fanouts: &mut [Vec<usize>],
    node: usize,
    new_binding: NodeBinding,
    tech: &TechLib,
    wire_delay: f32,
    required: Option<&[f32]>,
    topo_rank: &[usize],
) -> bool {
    try_binding_replacement(
        best,
        arrival,
        fanouts,
        node,
        new_binding,
        tech,
        wire_delay,
        required,
        topo_rank,
        true,
    )
}

/// Stage-1 delay flow: dual-phase sweep.
/// Considers both positive and negative bindings at every AND node, propagating the
/// cheaper-phase arrival directly to fanout, which reduces delay vs. the single-phase
/// path by avoiding redundant inverters.  This is the default Stage-1 implementation.
/// Build a node's cut pool for the next level: **matchable cuts first**, then the narrowest
/// remaining structural candidates.
///
/// A node's pool serves two consumers with opposite preferences:
/// - *direct matching at this node* wants the wide cuts that a compound cell can realise;
/// - *merging at the parents* wants narrow cuts, because two cuts of width `w1 + w2` merge into a
///   cut with at most `w1 + w2` inputs and anything above `max_inputs` is filtered away.
///
/// Keeping only the matchable cuts (the historical behaviour) collapses the pool to ~1.9 cuts per
/// node, and because candidates are the cross product of the fanins' pools that collapse is a fixed
/// point: 1.9^2 ~= 3.6 candidates, of which ~2 match again. Keeping only *wide* cuts is just as bad
/// in the other direction: two 8-input pools merge into almost nothing valid.
fn balanced_cut_pool<'a>(
    matchable: impl Iterator<Item = &'a Cut>,
    candidates: &[Cut],
    limit: usize,
) -> Vec<Cut> {
    let limit = limit.max(1);
    /* `EMAPPER_POOL_MAX_WIDTH` (0 = no cap, default): the pool a node hands to its parents is capped
    by cut width, while matching at the node itself still sees every candidate.
    The two roles want opposite things. Candidates are the cross product of the fanins' pools, so
    a wide-heavy pool makes most merged pairs exceed `max_inputs` and get filtered: measured on
    c432 the candidate count pins at ~11 per node however large `max_cuts` is, and the mapper then
    has no shallow alternative to the slowest cells in the library. Restricting the pool is what
    unbottles it -- `EMAPPER_MAX_INPUTS=7` (which also removes 8/9-input cuts from the candidates)
    drops c432 from 15 to **13 levels at the same delay**, matching mockturtle, and makes the run
    1.6x faster. This knob gets that effect without also losing the wide cells. */
    let pool_max_width = std::env::var("EMAPPER_POOL_MAX_WIDTH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(6);
    let mut pool: Vec<Cut> = Vec::with_capacity(limit);
    // Keyed on the leaf set: see the note in `enumerate_merged_cuts_mockturtle`.
    let mut seen: HashSet<Vec<usize>> = HashSet::with_capacity(limit * 2);
    for c in matchable {
        if pool.len() >= limit {
            return pool;
        }
        if seen.insert(c.input_indices().to_vec()) {
            pool.push(c.clone());
        }
    }
    // `candidates` arrives in ascending area-flow order, i.e. narrowest cones first, which is the
    // order that keeps upstream merges inside `max_inputs`.
    for c in candidates {
        if pool.len() >= limit {
            break;
        }
        if pool_max_width > 0 && c.input_count() > pool_max_width {
            continue;
        }
        if seen.insert(c.input_indices().to_vec()) {
            pool.push(c.clone());
        }
    }
    if pool.is_empty() {
        pool.extend(candidates.iter().take(limit).cloned());
    }
    pool
}

/// Pick up to `limit` structurally enumerated cuts, round-robin over cut width (**widest first**).
///
/// Kept for `EMAPPER_CUT_POOL_DIVERSE=1`; see [`balanced_cut_pool`] for the default policy.
fn structural_cut_pool(candidates: &[Cut], limit: usize) -> Vec<Cut> {
    let limit = limit.max(1);
    if candidates.len() <= limit {
        return candidates.to_vec();
    }
    let max_w = candidates.iter().map(Cut::input_count).max().unwrap_or(1);
    let mut buckets: Vec<Vec<Cut>> = vec![Vec::new(); max_w + 2];
    for c in candidates {
        let w = c.input_count();
        if !buckets[w].contains(c) {
            buckets[w].push(c.clone());
        }
    }
    let mut cursors = vec![0usize; max_w + 2];
    let mut out: Vec<Cut> = Vec::with_capacity(limit);
    let mut progressed = true;
    while out.len() < limit && progressed {
        progressed = false;
        for w in (1..=max_w).rev() {
            if out.len() >= limit {
                break;
            }
            if let Some(cut) = buckets[w].get(cursors[w]) {
                out.push(cut.clone());
                cursors[w] += 1;
                progressed = true;
            }
        }
    }
    out
}

/// `EMAPPER_SINGLE_PHASE_ARRIVAL` (default **off**; `1`/`true` enables the honest single-binding
/// arrival map).
///
/// The forward delay flow publishes a node's leaf-arrival map as the **minimum over the best
/// positive *and* best negative realization**. That is optimistic, because
/// `collect_phase_aware_bindings` materialises **one** realization per node, so the phase the flow
/// did not keep still costs an inverter in the emitted netlist -- the `[arrival-mismatch]` telemetry
/// in `EMAPPER_DEBUG_COVER` reports exactly that (177 of 208 ANDs on c432, gaps up to 16.3 == one
/// inverter delay).
///
/// Making the map honest was expected to help, but it does not: measured with `EMAPPER_USE_EGG=0`
/// (area/delay/depth), c432 16.83/486.5/16 -> 16.91/539.1/16 and c2670 55.68/352.4/15 ->
/// 55.57/368.3/15, with c880/c1355/c7552 unchanged. The optimism is therefore not what drives the
/// delay gap, so the historical behaviour stays the default and the honest map is opt-in.
fn single_phase_arrival() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("EMAPPER_SINGLE_PHASE_ARRIVAL").is_ok_and(|v| v == "1"))
}

/// Cut-pool telemetry for `EMAPPER_DEBUG_CUTS=1` (see `delay_flow_pass`).
#[derive(Default)]
struct CutPoolStats {
    and_nodes: usize,
    candidates: usize,
    scored: usize,
    pool_widths: HashMap<usize, usize>,
    chosen_widths: HashMap<usize, usize>,
    choice_gap: f64,
    choice_gap_max: f32,
    both_widths: usize,
    two_faster: usize,
    two_advantage: f64,
    wide_advantage: f64,
    wide_only: usize,
    two_only: usize,
}

impl CutPoolStats {
    fn observe(&mut self, candidates: usize, kept: usize, pool: &[Cut]) {
        self.and_nodes += 1;
        self.candidates += candidates;
        self.scored += kept;
        for c in pool {
            *self.pool_widths.entry(c.input_count()).or_insert(0) += 1;
        }
    }

    fn observe_width_arrivals(&mut self, two: f32, wide: f32) {
        if two.is_finite() && wide.is_finite() {
            self.both_widths += 1;
            if two <= wide {
                self.two_faster += 1;
                self.two_advantage += f64::from(wide - two);
            } else {
                self.wide_advantage += f64::from(two - wide);
            }
        } else if wide.is_finite() {
            self.wide_only += 1;
        } else if two.is_finite() {
            self.two_only += 1;
        }
    }

    fn observe_choice(&mut self, width: usize, chosen_arrival: f32, best_arrival: f32) {
        *self.chosen_widths.entry(width).or_insert(0) += 1;
        if chosen_arrival.is_finite() && best_arrival.is_finite() {
            self.choice_gap += f64::from(chosen_arrival - best_arrival);
            self.choice_gap_max = self.choice_gap_max.max(chosen_arrival - best_arrival);
        }
    }

    fn report(&self, tag: &str) {
        if self.and_nodes == 0 {
            return;
        }
        let fmt_hist = |m: &HashMap<usize, usize>| {
            let mut v = m.iter().collect::<Vec<_>>();
            v.sort_by_key(|(w, _)| **w);
            v.iter()
                .map(|(w, n)| format!("{w}:{n}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        let hist = fmt_hist(&self.pool_widths);
        let total: usize = self.pool_widths.values().sum();
        let wide: usize = self
            .pool_widths
            .iter()
            .filter(|(w, _)| **w >= 3)
            .map(|(_, n)| *n)
            .sum();
        let chosen = self.chosen_widths.values().sum::<usize>().max(1);
        let chosen_wide: usize = self
            .chosen_widths
            .iter()
            .filter(|(w, _)| **w >= 3)
            .map(|(_, n)| *n)
            .sum();
        eprintln!(
            "[cut-pool {tag}] ands={} candidates/and={:.1} scored/and={:.1} pool/and={:.1} \
             width_hist({hist}) wide(>=3)={:.1}% | chosen_hist({}) wide={:.1}% \
             mean_chosen_minus_best_arrival={:.2} max={:.2} | \
             both_widths={} two_faster={} mean_two_advantage={:.2} mean_wide_advantage={:.2} \
             wide_only={} two_only={}",
            self.and_nodes,
            self.candidates as f64 / self.and_nodes as f64,
            self.scored as f64 / self.and_nodes as f64,
            total as f64 / self.and_nodes as f64,
            100.0 * wide as f64 / total.max(1) as f64,
            fmt_hist(&self.chosen_widths),
            100.0 * chosen_wide as f64 / chosen as f64,
            self.choice_gap / chosen as f64,
            self.choice_gap_max,
            self.both_widths,
            self.two_faster,
            self.two_advantage / self.two_faster.max(1) as f64,
            self.wide_advantage / (self.both_widths - self.two_faster).max(1) as f64,
            self.wide_only,
            self.two_only,
        );
    }
}

fn delay_flow_pass(
    aig: &AigGraph,
    tech: &TechLib,
    ps: &EmapParams,
    egg_bundle: Option<&EggRewriteBundle>,
) -> MapOnceResult {
    let len = aig
        .graph()
        .node_indices()
        .map(|n| n.index())
        .max()
        .unwrap_or(0)
        + 1;
    let mut cuts: Vec<Vec<Cut>> = vec![Vec::new(); len];
    let mut best_dual: Vec<DualNodeBinding> = vec![DualNodeBinding::default(); len];
    let mut phase_match = vec![NodePhaseMatch::default(); len];
    let mut arrival: Vec<f32> = vec![0.0; len];
    let mut phase_arrival: Vec<PhaseArrival> = vec![
        PhaseArrival {
            pos: 0.0,
            neg: tech.inv_delay,
        };
        len
    ];
    let mut phase_flow: Vec<PhaseFlow> = vec![PhaseFlow::default(); len];
    let mut tt_cache: HashMap<String, (u8, Tt9)> = HashMap::new();
    let mut cover_cache: CoverChoiceCache = HashMap::new();
    let mut perf = EmapPerf::default();
    let structural_request_mask = compute_structural_request_masks(aig, len);
    let est_refs = compute_est_refs(aig, len);
    let comb_depth = compute_aig_comb_depth(aig, len);
    /* `EMAPPER_DEBUG_CUTS=1`: report how many enumerated cuts survive to a binding and enter the
    per-node pool, plus the width histogram of that pool. Wide cuts are what let a binding
    collapse several AIG levels into one compound cell; if the pool is all 2-input cuts the
    mapping degenerates into AND2 chains no matter how large `max_cuts`/`max_inputs` are. */
    let mut cut_dbg: Option<CutPoolStats> =
        std::env::var_os("EMAPPER_DEBUG_CUTS").map(|_| CutPoolStats::default());

    for pi in aig.input_nodes() {
        cuts[pi.index()] = vec![Cut::single_node(pi.index())];
        arrival[pi.index()] = 0.0;
        phase_arrival[pi.index()] = PhaseArrival {
            pos: 0.0,
            neg: tech.inv_delay,
        };
        phase_flow[pi.index()] = PhaseFlow {
            pos: 0.0,
            neg: tech.inv_area / est_refs[pi.index()],
        };
    }

    // Capture AND nodes in true topological order for use in the reverse pass.
    let mut topo_and_order: Vec<usize> = Vec::new();

    let mut topo = petgraph::visit::Topo::new(aig.graph());
    while let Some(node) = topo.next(aig.graph()) {
        let idx = node.index();
        match aig.node_type(node) {
            AigNode::Zero => {
                cuts[idx] = vec![Cut::single_node(idx)];
                arrival[idx] = 0.0;
                phase_arrival[idx] = PhaseArrival {
                    pos: 0.0,
                    neg: tech.inv_delay,
                };
                phase_flow[idx] = PhaseFlow::default();
            }
            AigNode::Input(_) | AigNode::Output(_) => {}
            AigNode::And => {
                topo_and_order.push(idx);
                let Some((x_edge, y_edge)) = aig.try_unwrap_and(node) else {
                    continue;
                };
                let x = aig.edge_source(x_edge);
                let y = aig.edge_source(y_edge);

                let mut trivial_inputs = vec![x.index(), y.index()];
                trivial_inputs.sort_unstable();
                trivial_inputs.dedup();
                let mut trivial_nodes = trivial_inputs.clone();
                trivial_nodes.push(idx);
                trivial_nodes.sort_unstable();
                let trivial = Cut::new(trivial_inputs, idx, trivial_nodes);

                let t_cut = Instant::now();
                let budget = ps.max_cuts.saturating_mul(10).max(1);
                let candidates = enumerate_merged_cuts_mockturtle(
                    &cuts[x.index()],
                    &cuts[y.index()],
                    idx,
                    trivial,
                    ps.max_inputs,
                    budget,
                    &est_refs,
                    &comb_depth,
                );
                perf.cut_gen += t_cut.elapsed();
                let candidates_len = candidates.len();

                let mut node_match = NodePhaseMatch::default();
                let mut kept_cuts = Vec::<(Cut, f32, f32)>::new();

                let req_pos = finite_phase_required(phase_match[idx].required[0]);
                let req_neg = finite_phase_required(phase_match[idx].required[1]);

                if emapper_parallel_cuts_within_node() && candidates.len() > 1 {
                    let rows: Vec<(DualNodeBinding, TtExprCache, CoverChoiceCache, EmapPerf)> =
                        candidates
                            .par_iter()
                            .map(|cut| {
                                let mut tt_l = TtExprCache::default();
                                let mut cover_l = CoverChoiceCache::default();
                                let mut perf_l = EmapPerf::default();
                                let scored = score_cut_dual_phase_for_map(
                                    aig,
                                    tech,
                                    &phase_arrival,
                                    cut,
                                    ps,
                                    MappingObjective::DelayFirst,
                                    &mut tt_l,
                                    &mut cover_l,
                                    &mut perf_l,
                                    egg_bundle,
                                    &best_dual,
                                    &phase_flow,
                                    req_pos,
                                    req_neg,
                                    None,
                                );
                                (scored, tt_l, cover_l, perf_l)
                            })
                            .collect();
                    for (scored, tt_l, cover_l, perf_l) in rows {
                        merge_tt_cache(&mut tt_cache, tt_l);
                        merge_cover_cache(&mut cover_cache, cover_l);
                        perf.merge_from(&perf_l);
                        fold_dual_phase_scored_into_node_match(
                            scored,
                            tech,
                            aig,
                            &phase_flow,
                            &mut node_match,
                            &mut kept_cuts,
                        );
                    }
                } else {
                    for cut in &candidates {
                        let scored = score_cut_dual_phase_for_map(
                            aig,
                            tech,
                            &phase_arrival,
                            cut,
                            ps,
                            MappingObjective::DelayFirst,
                            &mut tt_cache,
                            &mut cover_cache,
                            &mut perf,
                            egg_bundle,
                            &best_dual,
                            &phase_flow,
                            req_pos,
                            req_neg,
                            None,
                        );
                        fold_dual_phase_scored_into_node_match(
                            scored,
                            tech,
                            aig,
                            &phase_flow,
                            &mut node_match,
                            &mut kept_cuts,
                        );
                    }
                }

                /* Cut pool for the *next* level.
                The pool used to be built only from the cuts that happened to produce a binding
                (`kept_cuts`), deduplicated and truncated to `max_cuts`. On c432 that leaves
                **1.9 cuts per node**, and since a node's candidates are the cross product of its
                fanins' pools, the next level then sees 1.9^2 ~= 3.6 candidates and collapses
                again: the pool is a fixed point at ~2 cuts and the enumeration can never build
                the wide cones that compound cells (OR3/AO22/AO211/AO331...) need. Measured cell
                mix before this change: 53 AND2 out of 118 cells, against 2 for `emap`.
                Build the pool from the **structural** candidates instead, round-robining over
                cut width so no width is starved by the ascending-flow order. */
                let limit = ps.max_cuts.max(1);
                cuts[idx] = if cut_pool_rank_by_arrival() {
                    kept_cuts
                        .iter()
                        .map(|(cut, _, _)| cut.clone())
                        .unique()
                        .take(limit)
                        .collect()
                } else {
                    balanced_cut_pool(kept_cuts.iter().map(|(cut, _, _)| cut), &candidates, limit)
                };
                if let Some(d) = cut_dbg.as_mut() {
                    d.observe(candidates_len, kept_cuts.len(), &cuts[idx]);
                    let (mut two, mut wide) = (f32::INFINITY, f32::INFINITY);
                    for (c, a, _) in &kept_cuts {
                        if c.input_count() <= 2 {
                            two = two.min(*a);
                        } else {
                            wide = wide.min(*a);
                        }
                    }
                    d.observe_width_arrivals(two, wide);
                }

                select_alternatives(&mut node_match, tech, est_refs[idx]);
                let pre_choice_arrival = node_match.best[0].arrival;
                let (best_pos, best_neg) = drop_redundant_phase_delay(
                    node_match.best[0].binding.clone(),
                    node_match.best[1].binding.clone(),
                    tech,
                );
                if best_pos.is_some() || best_neg.is_some() {
                    node_match.best[0].binding = best_pos.clone();
                    node_match.best[1].binding = best_neg.clone();
                    phase_match[idx] = node_match.clone();
                    best_dual[idx].pos = best_pos;
                    best_dual[idx].neg = best_neg;
                    if let Some(d) = cut_dbg.as_mut() {
                        let (w, a) = match best_dual[idx].pos.as_ref() {
                            Some(b) => (b.cut.input_count(), binding_positive_arrival(b, tech)),
                            None => (0, f32::INFINITY),
                        };
                        d.observe_choice(w, a, pre_choice_arrival);
                    }
                    let req_mask = if structural_request_mask[idx] == 0 {
                        0b01
                    } else {
                        structural_request_mask[idx]
                    };
                    let realized =
                        choose_dual_bindings_for_request_mask(&best_dual[idx], req_mask, tech);
                    phase_arrival[idx] = if single_phase_arrival() {
                        /* Publish the arrival of the single binding that will be materialised; a
                        phase it does not realise natively is charged one inverter by
                        `binding_{positive,negative}_arrival`. */
                        choose_dual_binding_for_request_mask(&best_dual[idx], req_mask, tech)
                            .map_or_else(
                                || dual_binding_phase_arrival(&best_dual[idx], f32::INFINITY, tech),
                                |b| PhaseArrival {
                                    pos: binding_positive_arrival(&b, tech),
                                    neg: binding_negative_arrival(&b, tech),
                                },
                            )
                    } else {
                        phase_arrival_from_bindings(&realized, tech).unwrap_or_else(|| {
                            dual_binding_phase_arrival(&best_dual[idx], f32::INFINITY, tech)
                        })
                    };
                    arrival[idx] = phase_arrival[idx].pos;
                    phase_flow[idx] = match (&best_dual[idx].pos, &best_dual[idx].neg) {
                        (Some(_posb), Some(_negb)) => PhaseFlow {
                            pos: phase_match[idx].best[0].flow / est_refs[idx],
                            neg: phase_match[idx].best[1].flow / est_refs[idx],
                        },
                        (Some(_posb), None) => PhaseFlow {
                            pos: phase_match[idx].best[0].flow / est_refs[idx],
                            neg: (phase_match[idx].best[0].flow + tech.inv_area) / est_refs[idx],
                        },
                        (None, Some(_negb)) => PhaseFlow {
                            pos: (phase_match[idx].best[1].flow + tech.inv_area) / est_refs[idx],
                            neg: phase_match[idx].best[1].flow / est_refs[idx],
                        },
                        (None, None) => PhaseFlow::default(),
                    };
                } else {
                    let mut fanins = vec![
                        (x.index(), aig.is_edge_inverted(x_edge)),
                        (y.index(), aig.is_edge_inverted(y_edge)),
                    ];
                    fanins.sort_unstable();
                    fanins.dedup();
                    if fanins.len() == 2 {
                        let leaves_fb = vec![fanins[0].0, fanins[1].0];
                        let arr_fb = vec![arrival[leaves_fb[0]], arrival[leaves_fb[1]]];
                        let lit = |v: &str, inv: bool| {
                            if inv {
                                format!("(~ {v})")
                            } else {
                                v.to_string()
                            }
                        };
                        let expr_fb =
                            format!("(& {} {})", lit("a", fanins[0].1), lit("b", fanins[1].1));
                        if let Some((n_inputs, tt)) = sexp_to_tt9(&expr_fb) {
                            if let Some((d, a, v)) = tech.best_for_tt(
                                n_inputs,
                                tt,
                                &arr_fb,
                                MappingObjective::DelayFirst,
                            ) {
                                let cut_fb = Cut::new(leaves_fb.clone(), idx, {
                                    let mut n = leaves_fb.clone();
                                    n.push(idx);
                                    n.sort_unstable();
                                    n
                                });
                                let bfb = NodeBinding {
                                    node: idx,
                                    cut: cut_fb.clone(),
                                    leaves: leaves_fb,
                                    kind: NodeBindingKind::Direct { cell: v.clone() },
                                    phase_neg: false,
                                    native_delay: d + ps.wire_delay,
                                    delay: d + ps.wire_delay,
                                    area: a,
                                    mapping_cost_area: a,
                                };
                                arrival[idx] = bfb.delay;
                                best_dual[idx].pos = Some(bfb);
                                phase_arrival[idx] = dual_binding_phase_arrival(
                                    &best_dual[idx],
                                    f32::INFINITY,
                                    tech,
                                );
                                let flow = binding_phase_flow(
                                    best_dual[idx].pos.as_ref().expect("fallback binding"),
                                    aig,
                                    &phase_flow,
                                );
                                phase_flow[idx] = PhaseFlow {
                                    pos: flow / est_refs[idx],
                                    neg: (flow + tech.inv_area) / est_refs[idx],
                                };
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(d) = cut_dbg.as_ref() {
        d.report("stage1a");
    }

    let mut request_mask = vec![0u8; len];
    let mut delay_budget = 0.0f32;
    for o in aig.graph().externals(Outgoing) {
        for edge in aig.graph().edges_directed(o, Incoming) {
            let driver = edge.source().index();
            let phase = usize::from(*edge.weight());
            if driver < phase_match.len() {
                let arr = phase_match[driver].best[phase].arrival;
                if arr.is_finite() {
                    delay_budget = delay_budget.max(arr);
                } else {
                    delay_budget = delay_budget.max(
                        phase_arrival[driver].pos + if phase == 1 { tech.inv_delay } else { 0.0 },
                    );
                }
            }
            if aig.node_type(edge.source()) == AigNode::And {
                let mask = if *edge.weight() { 0b10 } else { 0b01 };
                request_mask[driver] |= mask;
                phase_match[driver].map_refs[phase] =
                    phase_match[driver].map_refs[phase].saturating_add(1);
            }
        }
    }
    for o in aig.graph().externals(Outgoing) {
        for edge in aig.graph().edges_directed(o, Incoming) {
            let driver = edge.source().index();
            let phase = usize::from(*edge.weight());
            if driver < phase_match.len() {
                phase_match[driver].required[phase] =
                    phase_match[driver].required[phase].min(delay_budget);
            }
        }
    }

    let mut selected = vec![None; len];
    for &idx in topo_and_order.iter().rev() {
        if request_mask[idx] == 0 {
            continue;
        }
        // Only call refine_best_matches when required times have actually been
        // set by propagate_required_to_leaves.  When required == INFINITY the
        // condition inside refine_best_matches is trivially true for every
        // alternative, which would silently replace delay-optimal selections
        // with area-optimal ones and blow up the critical path.
        let has_required =
            phase_match[idx].required[0].is_finite() || phase_match[idx].required[1].is_finite();
        if has_required && refine_enabled() {
            refine_best_matches(&mut phase_match[idx], tech);
        }
        best_dual[idx].pos = phase_match[idx].best[0].binding.clone();
        best_dual[idx].neg = phase_match[idx].best[1].binding.clone();
        let Some(binding) =
            choose_dual_binding_for_request_mask(&best_dual[idx], request_mask[idx], tech)
        else {
            continue;
        };
        let served_phase_neg = (request_mask[idx] & 0b01) == 0 && (request_mask[idx] & 0b10) != 0;
        if phase_match[idx].same_match
            && phase_match[idx].map_refs[served_phase_neg as usize ^ 1] > 0
        {
            let other = served_phase_neg as usize ^ 1;
            let use_phase = served_phase_neg as usize;
            phase_match[idx].required[use_phase] = phase_match[idx].required[use_phase]
                .min(phase_match[idx].required[other] - tech.inv_delay);
        }
        if (request_mask[idx] & 0b01) != 0 {
            propagate_required_to_leaves(
                aig,
                &binding,
                false,
                phase_match[idx].required[0],
                &mut phase_match,
                tech,
                ps.wire_delay,
            );
        }
        if (request_mask[idx] & 0b10) != 0 {
            propagate_required_to_leaves(
                aig,
                &binding,
                true,
                phase_match[idx].required[1],
                &mut phase_match,
                tech,
                ps.wire_delay,
            );
        }
        for (leaf, leaf_neg) in next_required_phases(&binding, aig) {
            request_mask[leaf] |= if leaf_neg { 0b10 } else { 0b01 };
        }
        if let Ok(t) = std::env::var("EMAPPER_DEBUG_NODE") {
            if t.split(',')
                .filter_map(|x| x.parse::<usize>().ok())
                .any(|x| x == idx)
            {
                let posb = best_dual[idx].pos.as_ref();
                let negb = best_dual[idx].neg.as_ref();
                let desc = |b: Option<&NodeBinding>| {
                    b.map_or("none".to_string(), |b| match &b.kind {
                        NodeBindingKind::Direct { cell } => format!(
                            "{} neg_mask={:#x} out_neg={} phase_neg={} native={:.2}",
                            cell.cell_name,
                            cell.neg_mask,
                            cell.out_neg,
                            b.phase_neg,
                            b.native_delay
                        ),
                        NodeBindingKind::MappedCover { .. } => format!(
                            "cover phase_neg={} native={:.2}",
                            b.phase_neg, b.native_delay
                        ),
                    })
                };
                eprintln!(
                    "[rev {idx}] req_mask={:#04b} struct={:#04b} chosen: {} | dual.pos={} | dual.neg={}",
                    request_mask[idx], structural_request_mask[idx], desc(Some(&binding)), desc(posb), desc(negb)
                );
            }
        }
        selected[idx] = Some(binding);
    }

    if std::env::var_os("EMAPPER_DEBUG_COVER").is_some() {
        let mut both = 0usize;
        let mut cover_better = 0usize;
        let mut cover_only = 0usize;
        let mut direct_only = 0usize;
        let mut sum_gap = 0.0f64;
        let mut best_gain = 0.0f32;
        for idx in &topo_and_order {
            let c = phase_match[*idx].cover_arrival[0];
            let d = phase_match[*idx].direct_arrival[0];
            match (c.is_finite(), d.is_finite()) {
                (true, true) => {
                    both += 1;
                    if c < d {
                        cover_better += 1;
                        sum_gap += f64::from(d - c);
                        best_gain = best_gain.max(d - c);
                    }
                }
                (true, false) => cover_only += 1,
                (false, true) => direct_only += 1,
                (false, false) => {}
            }
        }
        let fwd_po = aig
            .graph()
            .externals(Outgoing)
            .flat_map(|o| aig.graph().edges_directed(o, Incoming))
            .map(|e| {
                let d = e.source().index();
                arrival.get(d).copied().unwrap_or(0.0)
                    + if *e.weight() { tech.inv_delay } else { 0.0 }
            })
            .fold(0.0f32, f32::max);
        let (mut cw_delay, mut cw_area, mut dw_delay, mut dw_area) =
            (0usize, 0usize, 0usize, 0usize);
        let (mut sum_cdep, mut sum_ddep, mut sum_carea, mut sum_darea) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for idx in &topo_and_order {
            let ca = phase_match[*idx].cover_arrival[0];
            let da = phase_match[*idx].direct_arrival[0];
            let car = phase_match[*idx].cover_area[0];
            let dar = phase_match[*idx].direct_area[0];
            if ca.is_finite() && da.is_finite() {
                if ca < da {
                    cw_delay += 1;
                } else {
                    dw_delay += 1;
                }
                if car < dar {
                    cw_area += 1;
                } else {
                    dw_area += 1;
                }
                sum_cdep += f64::from(ca);
                sum_ddep += f64::from(da);
                sum_carea += f64::from(car);
                sum_darea += f64::from(dar);
            }
        }
        let n = (cw_delay + dw_delay).max(1) as f64;
        eprintln!(
            "[cover-telemetry] phase_model_po={:.2} forward_delay_budget={:.2} ands={} both={} cover_better={} cover_only={} direct_only={} \
             mean_gain_when_better={:.2} max_gain={:.2} | \
             cover_wins_delay={cw_delay} direct_wins_delay={dw_delay} cover_wins_area={cw_area} direct_wins_area={dw_area} \
             mean_cover(delay={:.2},area={:.4}) mean_direct(delay={:.2},area={:.4})",
            fwd_po,
            delay_budget,
            topo_and_order.len(),
            both,
            cover_better,
            cover_only,
            direct_only,
            if cover_better > 0 { sum_gap / cover_better as f64 } else { 0.0 },
            best_gain,
            sum_cdep / n,
            sum_carea / n,
            sum_ddep / n,
            sum_darea / n
        );
    }

    let force_pos = std::env::var_os("EMAPPER_POS_ASSEMBLY").is_some();
    let mut best = vec![None; len];
    for &idx in &topo_and_order {
        let req_mask = if force_pos || structural_request_mask[idx] == 0 {
            0b01
        } else {
            structural_request_mask[idx]
        };
        if let Some(binding) = choose_dual_binding_for_request_mask(&best_dual[idx], req_mask, tech)
        {
            best[idx] = Some(binding);
        }
    }
    if !force_pos {
        for binding in selected.into_iter().flatten() {
            let node = binding.node;
            best[node] = Some(binding);
        }
    }
    let debug_pre = std::env::var_os("EMAPPER_DEBUG_COVER").is_some().then(|| {
        best.iter()
            .map(|b| b.as_ref().map(|x| (x.native_delay, x.delay, x.phase_neg)))
            .collect::<Vec<_>>()
    });
    arrival = recompute_binding_arrivals(aig, tech, &mut best, ps.wire_delay);
    if let Some(pre) = &debug_pre {
        let mut rows: Vec<(f32, usize)> = Vec::new();
        for &idx in &topo_and_order {
            if let (Some(Some((pn, _, _))), Some(b)) = (pre.get(idx), best[idx].as_ref()) {
                if (b.native_delay - pn).abs() > 1.0 {
                    rows.push(((b.native_delay - pn).abs(), idx));
                }
            }
        }
        // Ascending node index: `topo_and_order` is topological, so this is the *origin* of the
        // divergence (the first node whose recomputed value disagrees with its own estimate).
        rows.sort_by_key(|r| r.1);
        for &(gap, idx) in rows.iter().take(4) {
            let b = best[idx].as_ref().unwrap();
            let (pn, pd, pneg) = pre[idx].unwrap();
            let desc = match &b.kind {
                NodeBindingKind::Direct { cell } => format!(
                    "direct:{} neg_mask={:#x} out_neg={}",
                    cell.cell_name, cell.neg_mask, cell.out_neg
                ),
                NodeBindingKind::MappedCover { .. } => "cover".to_string(),
            };
            eprintln!(
                "[node {idx}] gap={gap:.2} pre_native={pn:.2} pre_delay={pd:.2} pre_phase_neg={pneg} \
                 post_native={:.2} post_delay={:.2} est={:.2} recomputed_pos={:.2} | {desc}",
                b.native_delay, b.delay, phase_match[idx].best[0].arrival, arrival[idx]
            );
            for (i, &leaf) in b.leaves.iter().enumerate() {
                eprintln!(
                    "    pin{i} leaf={leaf} pre_leaf_native={:?} recomputed_leaf_pos={:?} \
                     recomputed_leaf_neg={:?} pin_delay={:?}",
                    pre.get(leaf).and_then(|x| *x).map(|(n, _, _)| n),
                    best.get(leaf).and_then(|x| x.as_ref()).map(|x| x.delay),
                    best.get(leaf)
                        .and_then(|x| x.as_ref())
                        .map(|x| x.native_delay + if x.phase_neg { 0.0 } else { tech.inv_delay }),
                    match &b.kind {
                        NodeBindingKind::Direct { cell } => cell.pin_delay.get(i).copied(),
                        NodeBindingKind::MappedCover { .. } => None,
                    }
                );
            }
        }
    }
    if std::env::var_os("EMAPPER_DEBUG_COVER").is_some() {
        let mut rows: Vec<(f32, usize, String, f32)> = Vec::new();
        for &idx in &topo_and_order {
            let est = phase_match[idx].best[0].arrival;
            let rec = arrival[idx];
            if est.is_finite() && rec.is_finite() && (rec - est).abs() > 1.0 {
                let kind = match best[idx].as_ref().map(|b| &b.kind) {
                    Some(NodeBindingKind::Direct { cell }) => format!("direct:{}", cell.cell_name),
                    Some(NodeBindingKind::MappedCover { .. }) => "cover".to_string(),
                    None => "none".to_string(),
                };
                rows.push((rec - est, idx, kind, est));
            }
        }
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        eprintln!(
            "[arrival-mismatch] nodes_with_gap={} worst={:?}",
            rows.len(),
            &rows[..rows.len().min(8)]
        );
    }
    let neg_alt = best_dual.iter().map(|d| d.neg.clone()).collect::<Vec<_>>();
    MapOnceResult {
        best,
        neg_alt,
        arrival,
        perf,
        cuts,
    }
}

/// Single-phase mapping pass. Used for the initial non-dual Stage 1a or for **Stage 1b**
/// rematch on a fixed cut pool after Stage 1a.
///
/// When `precomputed_cuts` is `Some`, re-matches on that fixed cut pool (mockturtle-style)
/// without re-enumerating cuts.
///
/// `mix_objective_by_critical_path`: if true and `critical_nodes` is `Some`, critical ANDs use
/// [`MappingObjective::DelayFirst`] and off-critical use [`MappingObjective::AreaFirst`]; if false,
/// every node uses `ps.objective` (still pass `critical_nodes` into [`score_cut`] for supergate
/// / zero-leaf policy).
fn constrained_single_phase_pass(
    aig: &AigGraph,
    tech: &TechLib,
    ps: &EmapParams,
    critical_nodes: Option<&HashSet<usize>>,
    required: Option<&[f32]>,
    egg_bundle: Option<&EggRewriteBundle>,
    precomputed_cuts: Option<&[Vec<Cut>]>,
    mix_objective_by_critical_path: bool,
) -> MapOnceResult {
    let len = aig
        .graph()
        .node_indices()
        .map(|n| n.index())
        .max()
        .unwrap_or(0)
        + 1;
    let rematch_only = precomputed_cuts.is_some();
    let mut cuts: Vec<Vec<Cut>> = if let Some(pc) = precomputed_cuts {
        assert_eq!(
            pc.len(),
            len,
            "precomputed cuts length must match AIG node slot count"
        );
        pc.to_vec()
    } else {
        vec![Vec::new(); len]
    };
    let mut best: Vec<Option<NodeBinding>> = vec![None; len];
    let mut arrival: Vec<f32> = vec![0.0; len];
    let mut tt_cache: HashMap<String, (u8, Tt9)> = HashMap::new();
    let mut cover_cache: CoverChoiceCache = HashMap::new();
    let mut perf = EmapPerf::default();
    let est_refs = compute_est_refs(aig, len);
    let comb_depth = compute_aig_comb_depth(aig, len);
    let mut phase_flow: Vec<PhaseFlow> = vec![PhaseFlow::default(); len];
    for pi in aig.input_nodes() {
        let i = pi.index();
        phase_flow[i] = PhaseFlow {
            pos: 0.0,
            neg: tech.inv_area / est_refs[i].max(1.0),
        };
    }

    for pi in aig.input_nodes() {
        if !rematch_only {
            cuts[pi.index()] = vec![Cut::single_node(pi.index())];
        }
        arrival[pi.index()] = 0.0;
    }

    let mut topo = petgraph::visit::Topo::new(aig.graph());
    while let Some(node) = topo.next(aig.graph()) {
        let idx = node.index();
        match aig.node_type(node) {
            AigNode::Zero => {
                if !rematch_only {
                    cuts[idx] = vec![Cut::single_node(idx)];
                }
                arrival[idx] = 0.0;
                phase_flow[idx] = PhaseFlow::default();
            }
            AigNode::Input(_) | AigNode::Output(_) => {}
            AigNode::And => {
                let Some((x_edge, y_edge)) = aig.try_unwrap_and(node) else {
                    continue;
                };
                let x = aig.edge_source(x_edge);
                let y = aig.edge_source(y_edge);

                let mut trivial_inputs = vec![x.index(), y.index()];
                trivial_inputs.sort_unstable();
                trivial_inputs.dedup();
                let mut trivial_nodes = trivial_inputs.clone();
                trivial_nodes.push(idx);
                trivial_nodes.sort_unstable();
                let trivial = Cut::new(trivial_inputs, idx, trivial_nodes);

                let candidates = if rematch_only {
                    cuts.get(idx).cloned().unwrap_or_default()
                } else {
                    let t_cut = Instant::now();
                    let budget = ps.max_cuts.saturating_mul(10).max(1);
                    let c = enumerate_merged_cuts_mockturtle(
                        &cuts[x.index()],
                        &cuts[y.index()],
                        idx,
                        trivial,
                        ps.max_inputs,
                        budget,
                        &est_refs,
                        &comb_depth,
                    );
                    perf.cut_gen += t_cut.elapsed();
                    c
                };

                let node_obj = match critical_nodes {
                    Some(set) if mix_objective_by_critical_path => {
                        if set.contains(&idx) {
                            MappingObjective::DelayFirst
                        } else {
                            MappingObjective::AreaFirst
                        }
                    }
                    _ => ps.objective,
                };

                let mut scored = Vec::new();
                if emapper_parallel_cuts_within_node() && candidates.len() > 1 {
                    let rows: Vec<(Option<NodeBinding>, TtExprCache, CoverChoiceCache, EmapPerf)> =
                        candidates
                            .par_iter()
                            .map(|cut| {
                                let mut tt_l = TtExprCache::default();
                                let mut cover_l = CoverChoiceCache::default();
                                let mut perf_l = EmapPerf::default();
                                let b = score_cut(
                                    aig,
                                    tech,
                                    &arrival,
                                    &best,
                                    cut,
                                    ps,
                                    node_obj,
                                    &mut tt_l,
                                    &mut cover_l,
                                    &mut perf_l,
                                    egg_bundle,
                                    required,
                                    Some(&phase_flow),
                                    critical_nodes,
                                );
                                let b = match b {
                                    Some(b) => {
                                        if let Some(req) = required {
                                            let lim =
                                                req.get(idx).copied().unwrap_or(f32::INFINITY);
                                            if b.delay <= lim + 1e-3 {
                                                Some(b)
                                            } else {
                                                None
                                            }
                                        } else {
                                            Some(b)
                                        }
                                    }
                                    None => None,
                                };
                                (b, tt_l, cover_l, perf_l)
                            })
                            .collect();
                    for (b, tt_l, cover_l, perf_l) in rows {
                        merge_tt_cache(&mut tt_cache, tt_l);
                        merge_cover_cache(&mut cover_cache, cover_l);
                        perf.merge_from(&perf_l);
                        if let Some(b) = b {
                            scored.push(b);
                        }
                    }
                } else {
                    for cut in candidates {
                        if let Some(b) = score_cut(
                            aig,
                            tech,
                            &arrival,
                            &best,
                            &cut,
                            ps,
                            node_obj,
                            &mut tt_cache,
                            &mut cover_cache,
                            &mut perf,
                            egg_bundle,
                            required,
                            Some(&phase_flow),
                            critical_nodes,
                        ) {
                            if let Some(req) = required {
                                let lim = req.get(idx).copied().unwrap_or(f32::INFINITY);
                                if b.delay <= lim + 1e-3 {
                                    scored.push(b);
                                }
                            } else {
                                scored.push(b);
                            }
                        }
                    }
                }
                let best_binding = scored
                    .iter()
                    .min_by(|a, b| cmp_binding_objective(a, b, node_obj))
                    .cloned();

                if let Some(best_binding) = best_binding {
                    if std::env::var("EMAPPER_TRACE_BINDINGS")
                        .ok()
                        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                        .unwrap_or(false)
                    {
                        match &best_binding.kind {
                            NodeBindingKind::Direct { cell } => {
                                eprintln!(
                                    "trace_bind node={} leaves={:?} cell={} pins={:?} neg_mask={} out_neg={}",
                                    idx,
                                    best_binding.leaves,
                                    cell.cell_name,
                                    cell.pin_names,
                                    cell.neg_mask,
                                    cell.out_neg
                                );
                            }
                            NodeBindingKind::MappedCover { graph, .. } => {
                                eprintln!(
                                    "trace_bind node={} leaves={:?} mapped_cover_nodes={}",
                                    idx,
                                    best_binding.leaves,
                                    graph.nodes.len()
                                );
                            }
                        }
                    }
                    arrival[idx] = best_binding.delay;
                    let f = binding_phase_flow(&best_binding, aig, &phase_flow);
                    phase_flow[idx] = PhaseFlow {
                        pos: f / est_refs[idx].max(1.0),
                        neg: (f + tech.inv_area) / est_refs[idx].max(1.0),
                    };
                    if !rematch_only {
                        cuts[idx] = cut_pool_from_scored(
                            scored,
                            node_obj,
                            ps.max_cuts.max(1),
                            &best_binding,
                        );
                    }
                    best[idx] = Some(best_binding);
                } else {
                    // Functional safety fallback for AIG-style node: force-map using its two
                    // non-const fanins so downstream netlist never leaves undriven mapped nodes.
                    let mut fanins = vec![
                        (x.index(), aig.is_edge_inverted(x_edge)),
                        (y.index(), aig.is_edge_inverted(y_edge)),
                    ];
                    fanins.sort_unstable();
                    fanins.dedup();
                    if fanins.len() == 2 {
                        let leaves_fb = vec![fanins[0].0, fanins[1].0];
                        let arr_fb = vec![arrival[leaves_fb[0]], arrival[leaves_fb[1]]];
                        let lit = |v: &str, inv: bool| {
                            if inv {
                                format!("(~ {v})")
                            } else {
                                v.to_string()
                            }
                        };
                        let expr_fb =
                            format!("(& {} {})", lit("a", fanins[0].1), lit("b", fanins[1].1));
                        if let Some((n_inputs, tt)) = sexp_to_tt9(&expr_fb) {
                            if let Some((d, a, v)) = tech.best_for_tt(
                                n_inputs,
                                tt,
                                &arr_fb,
                                MappingObjective::DelayFirst,
                            ) {
                                let cut_fb = Cut::new(leaves_fb.clone(), idx, {
                                    let mut n = leaves_fb.clone();
                                    n.push(idx);
                                    n.sort_unstable();
                                    n
                                });
                                let bfb = NodeBinding {
                                    node: idx,
                                    cut: cut_fb.clone(),
                                    leaves: leaves_fb,
                                    kind: NodeBindingKind::Direct { cell: v.clone() },
                                    phase_neg: false,
                                    native_delay: d + ps.wire_delay,
                                    delay: d + ps.wire_delay,
                                    area: a,
                                    mapping_cost_area: a,
                                };
                                arrival[idx] = bfb.delay;
                                best[idx] = Some(bfb);
                                cuts[idx] = vec![cut_fb];
                                continue;
                            }
                        }
                    }
                    cuts[idx] = vec![Cut::new(vec![idx], idx, vec![idx])];
                }
            }
        }
    }
    // Single-phase pass: no dual choice is kept, so callers fall back to `best` for both phases.
    let neg_alt = vec![None; best.len()];
    MapOnceResult {
        best,
        neg_alt,
        arrival,
        perf,
        cuts,
    }
}

fn compute_required_times(
    aig: &AigGraph,
    tech: &TechLib,
    best: &[Option<NodeBinding>],
    arrival: &[f32],
    wire_delay: f32,
) -> Vec<f32> {
    let max_po = primary_output_max_delay(aig, tech, best, arrival);
    let mut req = vec![max_po; best.len()];
    let mut nodes = aig.graph().node_indices().map(|n| n.index()).collect_vec();
    nodes.sort_unstable_by(|a, b| b.cmp(a));
    for n in nodes {
        let Some(Some(b)) = best.get(n) else {
            continue;
        };
        let rn = req[n];
        match &b.kind {
            NodeBindingKind::Direct { cell } => {
                let out_inv = if cell.out_neg { tech.inv_delay } else { 0.0 };
                for i in 0..(cell.n_inputs as usize) {
                    let leaf = b.leaves[i];
                    if leaf == 0 || leaf >= req.len() {
                        continue;
                    }
                    let inv_d = if ((cell.neg_mask >> i) & 1) == 1 {
                        tech.inv_delay
                    } else {
                        0.0
                    };
                    let cand = rn - wire_delay - out_inv - cell.pin_delay[i] - inv_d;
                    req[leaf] = req[leaf].min(cand);
                }
            }
            NodeBindingKind::MappedCover { .. } => {
                for &leaf in &b.leaves {
                    if leaf != 0 && leaf < req.len() {
                        req[leaf] = req[leaf].min(rn);
                    }
                }
            }
        }
    }
    req
}

/// Observed delay at a PO driven by `driver` when the AIG edge to the PO has `edge_inverted`.
///
/// For a non-inverted edge, the PO sees the positive logical phase, which matches `arrival[driver]`
/// after `recompute_binding_arrivals`. For an inverted edge, the PO sees the negative logical
/// phase: use `binding_negative_arrival` when a mapped AND binding exists, otherwise approximate
/// with `arrival[driver] + inv_delay` (e.g. PI).
fn primary_output_observed_delay(
    tech: &TechLib,
    driver: usize,
    edge_inverted: bool,
    best: &[Option<NodeBinding>],
    arrival: &[f32],
) -> f32 {
    let arr = arrival.get(driver).copied().unwrap_or(0.0);
    if !edge_inverted {
        return arr;
    }
    if let Some(Some(b)) = best.get(driver) {
        return binding_negative_arrival(b, tech);
    }
    arr + tech.inv_delay
}

fn primary_output_max_delay(
    aig: &AigGraph,
    tech: &TechLib,
    best: &[Option<NodeBinding>],
    arrival: &[f32],
) -> f32 {
    aig.graph()
        .externals(Outgoing)
        .flat_map(|o| aig.graph().edges_directed(o, Incoming))
        .map(|edge| {
            primary_output_observed_delay(
                tech,
                edge.source().index(),
                *edge.weight(),
                best,
                arrival,
            )
        })
        .fold(0.0, f32::max)
}

fn binding_positive_arrival(binding: &NodeBinding, tech: &TechLib) -> f32 {
    if binding.phase_neg {
        binding.native_delay + tech.inv_delay
    } else {
        binding.native_delay
    }
}

fn binding_negative_arrival(binding: &NodeBinding, tech: &TechLib) -> f32 {
    if binding.phase_neg {
        binding.native_delay
    } else {
        binding.native_delay + tech.inv_delay
    }
}

fn cell_output_phase_arrival(
    cell: &CellVariant,
    fanin_phases: &[PhaseArrival],
    tech: &TechLib,
) -> Option<PhaseArrival> {
    if fanin_phases.len() != cell.n_inputs as usize {
        return None;
    }
    let mut t = 0.0f32;
    for (i, arrival) in fanin_phases.iter().enumerate() {
        let ph = if ((cell.neg_mask >> i) & 1) == 1 {
            arrival.neg
        } else {
            arrival.pos
        };
        t = t.max(ph + cell.pin_delay[i]);
    }
    if cell.out_neg {
        t += tech.inv_delay;
    }
    Some(if cell.out_neg {
        PhaseArrival {
            pos: t + tech.inv_delay,
            neg: t,
        }
    } else {
        PhaseArrival {
            pos: t,
            neg: t + tech.inv_delay,
        }
    })
}

fn recompute_binding_native_delay(
    binding: &NodeBinding,
    arrival: &[f32],
    tech: &TechLib,
    wire_delay: f32,
) -> Option<f32> {
    match &binding.kind {
        NodeBindingKind::Direct { cell } => {
            let mut fanin_phases = Vec::with_capacity(cell.n_inputs as usize);
            for i in 0..(cell.n_inputs as usize) {
                let leaf = *binding.leaves.get(i)?;
                let d = *arrival.get(leaf)?;
                fanin_phases.push(PhaseArrival {
                    pos: d,
                    neg: d + tech.inv_delay,
                });
            }
            let out = cell_output_phase_arrival(cell, &fanin_phases, tech)?;
            Some(out.pos + wire_delay)
        }
        NodeBindingKind::MappedCover {
            graph,
            pin_to_root_delay,
        } => {
            if !pin_to_root_delay.is_empty() && pin_to_root_delay.len() == binding.leaves.len() {
                let mut t = 0.0f32;
                for (k, &pr) in pin_to_root_delay.iter().enumerate() {
                    let leaf = *binding.leaves.get(k)?;
                    let d = *arrival.get(leaf)?;
                    t = t.max(d + pr);
                }
                Some(t + wire_delay)
            } else {
                let mut node_phase: Vec<Option<PhaseArrival>> = vec![None; graph.nodes.len()];
                for (idx, node) in graph.nodes.iter().enumerate() {
                    let phase = match node {
                        MappedCoverNode::Const(_) => PhaseArrival {
                            pos: 0.0,
                            neg: tech.inv_delay,
                        },
                        MappedCoverNode::Input(li) => {
                            let leaf = *binding.leaves.get(*li)?;
                            let d = *arrival.get(leaf)?;
                            PhaseArrival {
                                pos: d,
                                neg: d + tech.inv_delay,
                            }
                        }
                        MappedCoverNode::Cell { cell, fanins } => {
                            let mut fanin_phases = Vec::with_capacity(fanins.len());
                            for &fi in fanins {
                                fanin_phases.push(*node_phase.get(fi)?.as_ref()?);
                            }
                            cell_output_phase_arrival(cell, &fanin_phases, tech)?
                        }
                    };
                    node_phase[idx] = Some(phase);
                }
                let root = node_phase.get(graph.root)?.as_ref()?;
                Some(root.pos + wire_delay)
            }
        }
    }
}

fn recompute_binding_delay(
    binding: &NodeBinding,
    arrival: &[f32],
    tech: &TechLib,
    wire_delay: f32,
) -> Option<f32> {
    let native = recompute_binding_native_delay(binding, arrival, tech, wire_delay)?;
    Some(if binding.phase_neg {
        native + tech.inv_delay
    } else {
        native
    })
}

#[inline]
fn cmp_binding_objective(
    a: &NodeBinding,
    b: &NodeBinding,
    obj: MappingObjective,
) -> std::cmp::Ordering {
    let ka = match obj {
        MappingObjective::DelayFirst => (a.delay, a.mapping_cost_area),
        MappingObjective::AreaFirst => (a.mapping_cost_area, a.delay),
    };
    let kb = match obj {
        MappingObjective::DelayFirst => (b.delay, b.mapping_cost_area),
        MappingObjective::AreaFirst => (b.mapping_cost_area, b.delay),
    };
    ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
}

/// Up to `k` **distinct** cuts for the cut pool: `winner` first (so rematch passes always see it),
/// then other bindings in `node_obj` order.
fn cut_pool_from_scored(
    mut scored: Vec<NodeBinding>,
    node_obj: MappingObjective,
    k: usize,
    winner: &NodeBinding,
) -> Vec<Cut> {
    let k = k.max(1);
    scored.sort_by(|a, b| cmp_binding_objective(a, b, node_obj));
    let mut seen: HashSet<Cut> = HashSet::new();
    let mut out = Vec::new();
    seen.insert(winner.cut.clone());
    out.push(winner.cut.clone());
    for b in scored {
        if seen.insert(b.cut.clone()) {
            out.push(b.cut);
            if out.len() >= k {
                break;
            }
        }
    }
    out
}

/// Build a mapped-cover binding and refresh delay fields from the global `arrival` slice.
fn mapped_cover_binding_for_arrivals(
    cut_output_index: usize,
    cut: &Cut,
    leaves: &[usize],
    cover: &MappedCoverChoice,
    arrival: &[f32],
    tech: &TechLib,
    wire_delay: f32,
) -> Option<NodeBinding> {
    let mut b = build_mapped_cover_binding(cut_output_index, cut, leaves, cover, tech, wire_delay);
    b.native_delay = recompute_binding_native_delay(&b, arrival, tech, wire_delay)?;
    b.delay = recompute_binding_delay(&b, arrival, tech, wire_delay)?;
    Some(b)
}

/// Among structurally distinct egg covers (same DP cost class), pick the best [`NodeBinding`]
/// under `objective` using **current** leaf arrivals.
fn pick_best_mapped_cover_binding(
    covers: &[MappedCoverChoice],
    cut_output_index: usize,
    cut: &Cut,
    leaves: &[usize],
    arrival: &[f32],
    tech: &TechLib,
    wire_delay: f32,
    objective: MappingObjective,
) -> Option<(NodeBinding, usize)> {
    let mut best_i = 0usize;
    let mut best_b = mapped_cover_binding_for_arrivals(
        cut_output_index,
        cut,
        leaves,
        &covers[0],
        arrival,
        tech,
        wire_delay,
    )?;
    for (i, cc) in covers.iter().enumerate().skip(1) {
        let b = mapped_cover_binding_for_arrivals(
            cut_output_index,
            cut,
            leaves,
            cc,
            arrival,
            tech,
            wire_delay,
        )?;
        if cmp_binding_objective(&b, &best_b, objective) == std::cmp::Ordering::Less {
            best_b = b;
            best_i = i;
        }
    }
    Some((best_b, best_i))
}

/// Native-phase arrival of a binding evaluated against explicit leaf phase-arrival maps.
///
/// Same formula as [`recompute_binding_arrivals`], but parameterised on the leaf maps so it can be
/// used for a node's positive *and* negative realization in one topological sweep.
fn native_delay_from_phase_maps(
    binding: &NodeBinding,
    pos_arr: &[f32],
    neg_arr: &[f32],
    tech: &TechLib,
    wire_delay: f32,
) -> Option<f32> {
    match &binding.kind {
        NodeBindingKind::Direct { cell } => {
            let mut delay = 0.0f32;
            for i in 0..(cell.n_inputs as usize) {
                let leaf = *binding.leaves.get(i)?;
                let d = if ((cell.neg_mask >> i) & 1) == 1 {
                    neg_arr.get(leaf).copied().unwrap_or(0.0)
                } else {
                    pos_arr.get(leaf).copied().unwrap_or(0.0)
                } + cell.pin_delay[i];
                delay = delay.max(d);
            }
            if cell.out_neg {
                delay += tech.inv_delay;
            }
            Some(delay + wire_delay)
        }
        // Covers are built against the positive phase of their leaves (see `verilog_phase_needs`).
        NodeBindingKind::MappedCover { .. } => {
            recompute_binding_native_delay(binding, pos_arr, tech, wire_delay)
        }
    }
}

/// Topological arrival recomputation over **both** phase realizations of every node.
///
/// `pos` holds the (single) binding the area stages settled on; `neg_alt` holds the negative-phase
/// realization the delay flow kept. Both are valid self-contained implementations of the same node,
/// so each phase of a node is available at the cheaper of the two -- which is exactly the model the
/// delay flow optimises and the model the emitted netlist realises once `collect_required_bindings`
/// materialises the per-phase bindings. Returns `(pos_arrival, neg_arrival)`.
fn recompute_dual_arrivals(
    aig: &AigGraph,
    tech: &TechLib,
    pos: &mut [Option<NodeBinding>],
    neg_alt: &[Option<NodeBinding>],
    wire_delay: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = pos.len();
    let mut pos_arr = vec![0.0f32; n];
    let mut neg_arr = vec![tech.inv_delay; n];

    let mut topo = petgraph::visit::Topo::new(aig.graph());
    while let Some(node) = topo.next(aig.graph()) {
        let idx = node.index();
        if !matches!(aig.node_type(node), AigNode::And) {
            continue;
        }
        let mut best_pos = f32::INFINITY;
        let mut best_neg = f32::INFINITY;
        let mut consider = |b: &NodeBinding, pos_arr: &[f32], neg_arr: &[f32]| {
            if let Some(nd) = native_delay_from_phase_maps(b, pos_arr, neg_arr, tech, wire_delay) {
                let mut c = b.clone();
                c.native_delay = nd;
                c.delay = binding_positive_arrival(&c, tech);
                (
                    binding_positive_arrival(&c, tech),
                    binding_negative_arrival(&c, tech),
                )
            } else {
                (f32::INFINITY, f32::INFINITY)
            }
        };
        // Clone out of the mutable slot first: `consider` needs `pos_arr` while `pos` is borrowed.
        let pos_binding = pos[idx].clone();
        for b in [
            pos_binding.as_ref(),
            neg_alt.get(idx).and_then(|x| x.as_ref()),
        ]
        .into_iter()
        .flatten()
        {
            let (p, ng) = consider(b, &pos_arr, &neg_arr);
            best_pos = best_pos.min(p);
            best_neg = best_neg.min(ng);
        }
        if let Some(b) = pos[idx].as_mut() {
            if let Some(nd) = native_delay_from_phase_maps(b, &pos_arr, &neg_arr, tech, wire_delay)
            {
                b.native_delay = nd;
                b.delay = binding_positive_arrival(b, tech);
            }
        }
        if best_pos.is_finite() {
            pos_arr[idx] = best_pos;
        }
        if best_neg.is_finite() {
            neg_arr[idx] = best_neg;
        }
    }
    (pos_arr, neg_arr)
}

fn recompute_binding_arrivals(
    aig: &AigGraph,
    tech: &TechLib,
    best: &mut [Option<NodeBinding>],
    wire_delay: f32,
) -> Vec<f32> {
    let n = best.len();
    // pos_arrival[i] = delay to obtain the POSITIVE output of node i.
    // neg_arrival[i] = delay to obtain the NEGATIVE output of node i.
    // For PI nodes both are 0 / inv_delay; updated as AND nodes are processed.
    let mut pos_arrival = vec![0.0f32; n];
    let mut neg_arrival = vec![tech.inv_delay; n];

    let mut topo = petgraph::visit::Topo::new(aig.graph());
    while let Some(node) = topo.next(aig.graph()) {
        let idx = node.index();
        match aig.node_type(node) {
            AigNode::Zero | AigNode::Input(_) | AigNode::Output(_) => {}
            AigNode::And => {
                let Some(binding) = best[idx].as_mut() else {
                    continue;
                };
                // Recompute native delay using phase-aware leaf arrivals.
                // When a binding pin has neg_mask[i]=1 the Verilog wires the
                // NEGATIVE output of the leaf (either via an explicit inverter
                // for a pos-phase leaf, or directly from a neg-phase leaf's
                // native output).  Using neg_arrival[leaf] directly avoids
                // double-counting inv_delay when the leaf has a neg binding.
                let native_delay = match &binding.kind {
                    NodeBindingKind::Direct { cell } => {
                        let mut delay = 0.0f32;
                        for i in 0..(cell.n_inputs as usize) {
                            let leaf = match binding.leaves.get(i) {
                                Some(&l) => l,
                                None => {
                                    delay = f32::INFINITY;
                                    break;
                                }
                            };
                            let d = if ((cell.neg_mask >> i) & 1) == 1 {
                                // Pin wants the inverted signal: use neg_arrival which
                                // already accounts for the inverter (real or virtual).
                                neg_arrival.get(leaf).copied().unwrap_or(0.0)
                            } else {
                                pos_arrival.get(leaf).copied().unwrap_or(0.0)
                            } + cell.pin_delay[i];
                            delay = delay.max(d);
                        }
                        if cell.out_neg {
                            delay += tech.inv_delay;
                        }
                        delay + wire_delay
                    }
                    NodeBindingKind::MappedCover { .. } => {
                        // MappedCover bindings come only from the stable path
                        // (egg covers) where all leaves are pos-phase, so the
                        // original helper is correct.
                        match recompute_binding_native_delay(
                            binding,
                            &pos_arrival,
                            tech,
                            wire_delay,
                        ) {
                            Some(d) => d,
                            None => continue,
                        }
                    }
                };
                binding.native_delay = native_delay;
                binding.delay = binding_positive_arrival(binding, tech);
                pos_arrival[idx] = binding.delay;
                neg_arrival[idx] = binding_negative_arrival(binding, tech);
            }
        }
    }
    pos_arrival
}

fn collect_critical_nodes(
    aig: &AigGraph,
    tech: &TechLib,
    best: &[Option<NodeBinding>],
    arrival: &[f32],
    wire_delay: f32,
) -> HashSet<usize> {
    let mut crit = HashSet::new();
    let mut po_delays = Vec::<f32>::new();
    for o in aig.graph().externals(Outgoing) {
        for edge in aig.graph().edges_directed(o, Incoming) {
            po_delays.push(primary_output_observed_delay(
                tech,
                edge.source().index(),
                *edge.weight(),
                best,
                arrival,
            ));
        }
    }
    let max_po = po_delays.into_iter().fold(0.0f32, f32::max);
    let eps = 1e-3f32;
    let mut stack = Vec::<usize>::new();
    for o in aig.graph().externals(Outgoing) {
        for edge in aig.graph().edges_directed(o, Incoming) {
            let driver = edge.source().index();
            let d = primary_output_observed_delay(tech, driver, *edge.weight(), best, arrival);
            if (d - max_po).abs() <= eps {
                stack.push(driver);
            }
        }
    }
    stack.sort_unstable();
    stack.dedup();

    while let Some(n) = stack.pop() {
        if !crit.insert(n) {
            continue;
        }
        let Some(b) = &best[n] else {
            continue;
        };
        match &b.kind {
            NodeBindingKind::Direct { cell } => {
                let out_inv = if cell.out_neg { tech.inv_delay } else { 0.0 };
                let target = b.delay - wire_delay - out_inv;
                for i in 0..(cell.n_inputs as usize) {
                    let leaf = b.leaves[i];
                    if leaf == 0 {
                        continue;
                    }
                    let inv_d = if ((cell.neg_mask >> i) & 1) == 1 {
                        tech.inv_delay
                    } else {
                        0.0
                    };
                    let d = arrival[leaf] + cell.pin_delay[i] + inv_d;
                    if (d - target).abs() <= eps
                        && matches!(aig.node_type(NodeIndex::new(leaf)), AigNode::And)
                    {
                        stack.push(leaf);
                    }
                }
            }
            NodeBindingKind::MappedCover { .. } => {
                for &leaf in &b.leaves {
                    if leaf != 0 && matches!(aig.node_type(NodeIndex::new(leaf)), AigNode::And) {
                        stack.push(leaf);
                    }
                }
            }
        }
    }
    crit
}

/// When `critical_nodes` is `Some`, splits nodes on the critical path vs off it:
/// - **Critical**: real leaf arrivals for cut mapping; **no** supergate DB lookup/store (covers there
///   are keyed without per-leaf timing; timing-sensitive mapping must use egg/direct with real arrivals).
/// - **Non-critical**: leaf arrivals forced to **zero** (phase `pos=0`, `neg=inv_delay`) for mapping;
///   **supergate** lookup/store enabled (canonical delay-agnostic covers).
///
/// When `critical_nodes` is `None` (e.g. Stage 1a before criticality is known), preserves legacy behavior:
/// real arrivals and full supergate use.
#[inline]
fn cut_mapping_supergate_policy(
    critical_nodes: Option<&HashSet<usize>>,
    cut_output: usize,
) -> (bool, bool) {
    match critical_nodes {
        None => (false, true),
        Some(c) => {
            let on_crit = c.contains(&cut_output);
            (!on_crit, !on_crit)
        }
    }
}

fn apply_zero_leaf_arrivals_for_cut_mapping(
    tech: &TechLib,
    input_arrivals: &mut [f32],
    input_phase_arrivals: &mut [PhaseArrival],
) {
    input_arrivals.fill(0.0);
    for p in input_phase_arrivals.iter_mut() {
        p.pos = 0.0;
        p.neg = tech.inv_delay;
    }
}

/// Per-primary-output arrival of a **collected** binding set, in the same model the result uses.
///
/// Returns `(arrival, driver, driver_is_negated, driver_has_a_binding, label)`. This is the single
/// definition of the reported `max_delay`: `EMAPPER_DEBUG_PO=3` prints the same rows it maximises
/// over, so a stuck critical path can be attributed to a node instead of guessed at.
#[allow(clippy::type_complexity)]
fn po_arrival_breakdown(
    aig: &AigGraph,
    tech: &TechLib,
    bindings: &[NodeBinding],
    arrival: &[f32],
    dual_neg: Option<&Vec<f32>>,
) -> Vec<(f32, usize, bool, bool, String)> {
    let selected = bindings
        .iter()
        .map(|b| ((b.node, b.phase_neg), b))
        .collect::<HashMap<_, _>>();
    let mut out = Vec::new();
    for o in aig.graph().externals(Outgoing) {
        for edge in aig.graph().edges_directed(o, Incoming) {
            let driver = edge.source().index();
            let neg = *edge.weight();
            let row = match selected
                .get(&(driver, neg))
                .or_else(|| selected.get(&(driver, false)))
            {
                Some(binding) => {
                    let v = if neg {
                        binding_negative_arrival(binding, tech)
                    } else {
                        binding_positive_arrival(binding, tech)
                    };
                    (v, driver, neg, true, binding_kind_label(binding))
                }
                None if !neg => (
                    arrival.get(driver).copied().unwrap_or(0.0),
                    driver,
                    neg,
                    false,
                    "no-binding".to_string(),
                ),
                None => (
                    dual_neg
                        .and_then(|d| d.get(driver).copied())
                        .unwrap_or_else(|| {
                            arrival.get(driver).copied().unwrap_or(0.0) + tech.inv_delay
                        }),
                    driver,
                    neg,
                    false,
                    "no-binding+inv".to_string(),
                ),
            };
            out.push(row);
        }
    }
    out
}

/// Short label for a binding's implementation, used by the `EMAPPER_DEBUG_PO=3` critical-PO dump.
fn binding_kind_label(binding: &NodeBinding) -> String {
    let phase = if binding.phase_neg { "neg" } else { "pos" };
    match &binding.kind {
        NodeBindingKind::Direct { cell } => format!(
            "direct:{} neg_mask={:#x} out_neg={} {phase} cut_w={}",
            cell.cell_name,
            cell.neg_mask,
            cell.out_neg,
            binding.cut.input_count()
        ),
        NodeBindingKind::MappedCover { graph, .. } => {
            format!("cover cells={} {phase}", graph.nodes.len())
        }
    }
}

/// Levels (cell depth) of a binding's **native** output: the structural twin of
/// [`recompute_binding_native_delay`].
///
/// The mapper's own `depth` accounting ignored this. An inverted pin or a negated cell output is a
/// real inverter instance in the emitted netlist, so it costs one level exactly as `inv_delay`
/// costs time. On c432 the binding graph says 17 levels while ABC measures `lev = 20` on the emitted
/// Verilog (58 `INVx1` + 9 `BUFx2` for 125 cells), and those uncounted levels are what a delay
/// optimisation is supposed to be trading against.
fn binding_native_level(binding: &NodeBinding, level: &[u32]) -> u32 {
    match &binding.kind {
        NodeBindingKind::Direct { cell } => {
            let mut deepest = 0u32;
            for i in 0..usize::from(cell.n_inputs) {
                let Some(&leaf) = binding.leaves.get(i) else {
                    continue;
                };
                let leaf_level = level.get(leaf).copied().unwrap_or(0);
                let inv = u32::from(((cell.neg_mask >> i) & 1) == 1);
                deepest = deepest.max(leaf_level.saturating_add(inv));
            }
            1u32.saturating_add(deepest)
                .saturating_add(u32::from(cell.out_neg))
        }
        NodeBindingKind::MappedCover { graph, .. } => {
            let mut node_level = vec![0u32; graph.nodes.len()];
            for (i, node) in graph.nodes.iter().enumerate() {
                node_level[i] = match node {
                    MappedCoverNode::Const(_) => 0,
                    MappedCoverNode::Input(li) => binding
                        .leaves
                        .get(*li)
                        .and_then(|l| level.get(*l))
                        .copied()
                        .unwrap_or(0),
                    MappedCoverNode::Cell { cell, fanins } => {
                        let mut deepest = 0u32;
                        for (k, &fi) in fanins.iter().enumerate() {
                            let leaf_level = node_level.get(fi).copied().unwrap_or(0);
                            let inv = u32::from(((cell.neg_mask >> k) & 1) == 1);
                            deepest = deepest.max(leaf_level.saturating_add(inv));
                        }
                        1u32.saturating_add(deepest)
                            .saturating_add(u32::from(cell.out_neg))
                    }
                };
            }
            node_level.get(graph.root).copied().unwrap_or(0)
        }
    }
}

fn binding_positive_level(binding: &NodeBinding, level: &[u32]) -> u32 {
    let native = binding_native_level(binding, level);
    if binding.phase_neg {
        native.saturating_add(1)
    } else {
        native
    }
}

/// Recompute every node's **negative-phase** realization against the final `best` bindings.
///
/// See the call site in [`emap_with_params`]: `collect_phase_aware_bindings` decides whether to
/// materialise a node's negative output natively by consulting `neg_alt`, so that array has to
/// describe the cuts the mapping actually ended up with.
fn refresh_negative_alternatives(
    aig: &AigGraph,
    tech: &TechLib,
    ps: &EmapParams,
    best: &[Option<NodeBinding>],
    arrival: &[f32],
    egg_bundle: Option<&EggRewriteBundle>,
    neg_alt: &mut [Option<NodeBinding>],
) -> EmapPerf {
    let mut perf = EmapPerf::default();
    let mut tt_cache: TtExprCache = HashMap::new();
    let mut cover_cache: CoverChoiceCache = HashMap::new();
    for idx in topo_and_nodes(aig) {
        let Some(cut) = best
            .get(idx)
            .and_then(|b| b.as_ref())
            .map(|b| b.cut.clone())
        else {
            continue;
        };
        let refreshed = score_cut_for_phase(
            aig,
            tech,
            arrival,
            best,
            &cut,
            ps,
            true,
            MappingObjective::DelayFirst,
            &mut tt_cache,
            &mut cover_cache,
            &mut perf,
            egg_bundle,
            None,
            None,
            None,
        );
        if let Some(slot) = neg_alt.get_mut(idx) {
            *slot = refreshed;
        }
    }
    perf
}

/// Maximum cell depth of a **mapping**, counting exactly the levels the emitter produces.
///
/// This is the number to compare against `depth_view` of the mapped network or ABC's `lev`.
/// Two things the mapper's own accounting never had:
/// - a level notion at all (it optimises pin-delay arrivals), and
/// - the distinction between a leaf whose **negative phase is materialised** (a `phase_neg` binding,
///   so the writer connects `n{k}_neg` directly) and one where the writer must insert an inverter.
///
/// Tracking `(positive, negative)` levels per node and charging an inverter only for the latter is
/// what makes this match the netlist; charging for every inverted pin over-estimated c432 by two
/// levels, and ignoring `out_neg` (as the benchmark's hand-rolled walk did) under-estimated.
#[must_use]
pub fn mapped_depth(aig: &AigGraph, res: &EmapResult) -> u32 {
    let len = aig
        .graph()
        .node_indices()
        .map(|n| n.index())
        .max()
        .unwrap_or(0)
        + 1;
    let mut level_pos = vec![0u32; len];
    let mut level_neg = vec![1u32; len];
    level_neg[0] = 0; // the constant's complement is free (`const_sig`)
    let mut by_node: HashMap<usize, &NodeBinding> = HashMap::new();
    for b in &res.bindings {
        by_node.insert(b.node, b);
    }
    let mut worst = 0u32;
    for idx in topo_and_nodes(aig) {
        let Some(binding) = by_node.get(&idx).copied() else {
            continue;
        };
        let native = match &binding.kind {
            NodeBindingKind::Direct { cell } => {
                let mut deepest = 0u32;
                for i in 0..usize::from(cell.n_inputs) {
                    let Some(&leaf) = binding.leaves.get(i) else {
                        continue;
                    };
                    let l = if ((cell.neg_mask >> i) & 1) == 1 {
                        level_neg.get(leaf).copied().unwrap_or(1)
                    } else {
                        level_pos.get(leaf).copied().unwrap_or(0)
                    };
                    deepest = deepest.max(l);
                }
                1u32.saturating_add(deepest)
                    .saturating_add(u32::from(cell.out_neg))
            }
            NodeBindingKind::MappedCover { graph, .. } => {
                let mut node_level = vec![0u32; graph.nodes.len()];
                for (i, node) in graph.nodes.iter().enumerate() {
                    node_level[i] = match node {
                        MappedCoverNode::Const(_) => 0,
                        MappedCoverNode::Input(li) => binding
                            .leaves
                            .get(*li)
                            .map(|l| level_pos.get(*l).copied().unwrap_or(0))
                            .unwrap_or(0),
                        MappedCoverNode::Cell { cell, fanins } => {
                            let mut deepest = 0u32;
                            for (k, &fi) in fanins.iter().enumerate() {
                                let l = node_level.get(fi).copied().unwrap_or(0);
                                let inv = u32::from(((cell.neg_mask >> k) & 1) == 1);
                                deepest = deepest.max(l.saturating_add(inv));
                            }
                            1u32.saturating_add(deepest)
                                .saturating_add(u32::from(cell.out_neg))
                        }
                    };
                }
                node_level.get(graph.root).copied().unwrap_or(0)
            }
        };
        if binding.phase_neg {
            level_neg[idx] = native;
            level_pos[idx] = native + 1;
        } else {
            level_pos[idx] = native;
            level_neg[idx] = native + 1;
        }
        worst = worst.max(level_pos[idx]);
        worst = worst.max(level_neg[idx]);
    }
    worst
}

/// AND nodes of `aig` in topological order.
fn topo_and_nodes(aig: &AigGraph) -> Vec<usize> {
    let mut order = Vec::new();
    let mut topo = petgraph::visit::Topo::new(aig.graph());
    while let Some(node) = topo.next(aig.graph()) {
        if aig.node_type(node) == AigNode::And {
            order.push(node.index());
        }
    }
    order
}

/// Reduce the **level** of the mapping without increasing its modelled delay.
///
/// Runs after the area recovery passes. Nodes are visited in topological order so a node whose
/// leaves were just rewritten is re-scored against the updated arrivals and levels, and a candidate
/// is accepted only when it is strictly shallower *and* no slower than the binding it replaces.
/// That makes the pass monotone in both delay and level, so it cannot regress QoR; the only
/// question is how much of the current depth is avoidable at equal delay.
fn depth_recovery_pass(
    aig: &AigGraph,
    tech: &TechLib,
    ps: &EmapParams,
    best: &mut [Option<NodeBinding>],
    arrival: &mut [f32],
    required: Option<&[f32]>,
    egg_bundle: Option<&EggRewriteBundle>,
    cuts: &[Vec<Cut>],
) -> EmapPerf {
    let len = best.len();
    let mut perf = EmapPerf::default();
    if len == 0 {
        return perf;
    }
    /* Two ways to bound what a level may cost:
    - `EMAPPER_DEPTH_AREA_SLACK` (per node, default 1.0): the replacement must stay within this
      factor of the area of the binding it replaces.
    - `EMAPPER_DEPTH_AREA_BUDGET` (global, default 0.0 = off): the **total** area the pass may add,
      as a fraction of the mapping's area. When set, the per-node cap is lifted and the budget
      governs instead, because the per-node cap is a poor proxy: it blocks an expensive
      level reduction even when a much cheaper one elsewhere would pay for it. */
    let area_slack = std::env::var("EMAPPER_DEPTH_AREA_SLACK")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);
    /* Default 0.05: measured on c432/c880/c1355/c2670/c5315/c7552 the level pass needs a global
    budget to be useful at all, and 5 % of the mapping's area is where the delay it removes stops
    costing more than it is worth. 0 restores strict area neutrality; 0.10 buys another 1-2 levels
    on c432/c1355 at up to +14 % area. */
    let area_budget_frac = std::env::var("EMAPPER_DEPTH_AREA_BUDGET")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.10);
    let area_slack = if area_budget_frac > 0.0 {
        f32::INFINITY
    } else {
        area_slack
    };
    let depth_delay_slack = std::env::var("EMAPPER_DEPTH_DELAY_SLACK")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.03);
    let total_before: f32 = best.iter().flatten().map(|b| b.area).sum();
    let area_budget = total_before * area_budget_frac;
    let mut area_spent = 0.0f32;
    let mut tt_cache: TtExprCache = HashMap::new();
    let mut cover_cache: CoverChoiceCache = HashMap::new();
    let mut level = vec![0u32; len];
    let order = topo_and_nodes(aig);

    for &idx in &order {
        let Some(current) = best[idx].clone() else {
            continue;
        };
        let current_level = binding_positive_level(&current, &level);
        level[idx] = current_level;
        let current_delay = binding_positive_arrival(&current, tech);
        let node_required = required
            .and_then(|r| r.get(idx).copied())
            .filter(|v| v.is_finite() && *v < f32::MAX * 0.5);

        // `(level, area, delay)`: among equally shallow candidates take the cheapest, because the
        // delay is already bounded by the binding being replaced.
        let mut winner: Option<(u32, f32, f32, NodeBinding)> = None;
        for cut in &cuts[idx] {
            let Some(cand) = score_cut(
                aig,
                tech,
                arrival,
                best,
                cut,
                ps,
                MappingObjective::DelayFirst,
                &mut tt_cache,
                &mut cover_cache,
                &mut perf,
                egg_bundle,
                required,
                None,
                None,
            ) else {
                continue;
            };
            let cand_level = binding_positive_level(&cand, &level);
            if cand_level >= current_level {
                continue;
            }
            /* `EMAPPER_DEPTH_DELAY_SLACK` (fraction, default 0 = never slower): how much extra
            modelled delay a **shallower** binding may cost. At 0 the pass only removes depth the
            mapping already paid nothing for; a positive value lets it buy levels, which is the
            trade mockturtle's composed supergates get for free. Non-monotone in delay, so the
            caller snapshots and restores if the worst primary output regresses. */
            if cand.delay > current_delay * (1.0 + depth_delay_slack) + 1e-3 {
                continue;
            }
            if cand.area > current.area * area_slack + 1e-4 {
                continue;
            }
            if node_required.is_some_and(|req| cand.delay > req + 1e-3) {
                continue;
            }
            let extra = (cand.area - current.area).max(0.0);
            if extra > 0.0 && area_spent + extra > area_budget + 1e-6 {
                continue;
            }
            let better = winner.as_ref().is_none_or(|(w_level, w_area, w_delay, _)| {
                (cand_level, cand.area, cand.delay) < (*w_level, *w_area, *w_delay)
            });
            if better {
                winner = Some((cand_level, cand.area, cand.delay, cand));
            }
        }
        if let Some((new_level, _, new_delay, binding)) = winner {
            area_spent += (binding.area - current.area).max(0.0);
            arrival[idx] = new_delay;
            level[idx] = new_level;
            best[idx] = Some(binding);
        }
    }
    perf
}

fn score_cut(
    aig: &AigGraph,
    tech: &TechLib,
    arrival: &[f32],
    best: &[Option<NodeBinding>],
    cut: &Cut,
    ps: &EmapParams,
    objective: MappingObjective,
    tt_cache: &mut HashMap<String, (u8, Tt9)>,
    cover_cache: &mut CoverChoiceCache,
    perf: &mut EmapPerf,
    egg_bundle: Option<&EggRewriteBundle>,
    required: Option<&[f32]>,
    phase_flow: Option<&[PhaseFlow]>,
    critical_nodes: Option<&HashSet<usize>>,
) -> Option<NodeBinding> {
    score_cut_dual_phase(
        aig,
        tech,
        arrival,
        best,
        cut,
        ps,
        objective,
        tt_cache,
        cover_cache,
        perf,
        egg_bundle,
        required,
        phase_flow,
        critical_nodes,
    )
    .pos
}

fn score_cut_dual_phase(
    aig: &AigGraph,
    tech: &TechLib,
    arrival: &[f32],
    best: &[Option<NodeBinding>],
    cut: &Cut,
    ps: &EmapParams,
    objective: MappingObjective,
    tt_cache: &mut HashMap<String, (u8, Tt9)>,
    cover_cache: &mut CoverChoiceCache,
    perf: &mut EmapPerf,
    egg_bundle: Option<&EggRewriteBundle>,
    required: Option<&[f32]>,
    phase_flow: Option<&[PhaseFlow]>,
    critical_nodes: Option<&HashSet<usize>>,
) -> DualNodeBinding {
    let rq = finite_required_at(required, cut.output_index());
    DualNodeBinding {
        pos: score_cut_for_phase(
            aig,
            tech,
            arrival,
            best,
            cut,
            ps,
            false,
            objective,
            tt_cache,
            cover_cache,
            perf,
            egg_bundle,
            rq,
            phase_flow,
            critical_nodes,
        ),
        neg: None,
    }
}

fn score_cut_for_phase(
    aig: &AigGraph,
    tech: &TechLib,
    arrival: &[f32],
    best: &[Option<NodeBinding>],
    cut: &Cut,
    ps: &EmapParams,
    target_phase_neg: bool,
    objective: MappingObjective,
    tt_cache: &mut HashMap<String, (u8, Tt9)>,
    cover_cache: &mut CoverChoiceCache,
    perf: &mut EmapPerf,
    egg_bundle: Option<&EggRewriteBundle>,
    required_at_output: Option<f32>,
    phase_flow: Option<&[PhaseFlow]>,
    critical_nodes: Option<&HashSet<usize>>,
) -> Option<NodeBinding> {
    let (expr0, leaves, mut input_arrivals) = cut_to_sexp_and_arrivals(aig, arrival, cut);
    let pi_nodes: HashSet<usize> = aig.input_nodes().iter().map(|n| n.index()).collect();
    let (native_pos, native_neg) = native_phases_for_leaves(&leaves, best, &pi_nodes);
    let mut input_phase_arrivals = leaves
        .iter()
        .enumerate()
        .map(|(i, &leaf)| {
            if let Some(Some(binding)) = best.get(leaf) {
                PhaseArrival {
                    pos: binding_positive_arrival(binding, tech),
                    neg: binding_negative_arrival(binding, tech),
                }
            } else {
                let pos = input_arrivals.get(i).copied().unwrap_or(0.0);
                PhaseArrival {
                    pos,
                    neg: pos + tech.inv_delay,
                }
            }
        })
        .collect::<Vec<_>>();
    let (zero_leaf, allow_supergate) =
        cut_mapping_supergate_policy(critical_nodes, cut.output_index());
    if zero_leaf {
        apply_zero_leaf_arrivals_for_cut_mapping(
            tech,
            &mut input_arrivals,
            &mut input_phase_arrivals,
        );
    }
    score_cut_for_phase_core(
        aig,
        tech,
        cut,
        ps,
        objective,
        tt_cache,
        cover_cache,
        perf,
        egg_bundle,
        expr0,
        leaves,
        input_arrivals,
        input_phase_arrivals,
        &native_pos,
        &native_neg,
        target_phase_neg,
        required_at_output,
        phase_flow,
        allow_supergate,
        arrival,
    )
}

/// Debug aid: report *why* an e-graph cover extraction failed.
///
/// Enabled with `EMAPPER_DEBUG_EGG=1`. Prints each distinct reason once (up to 24), together with a
/// sample expression, then the running total on stderr. This is the only place the extraction error
/// string is available -- the production path deliberately collapses it to `None`.
fn egg_fail_note(reason: &str, expr: &str) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static TOTAL: AtomicU64 = AtomicU64::new(0);
    let total = TOTAL.fetch_add(1, Ordering::Relaxed);
    if std::env::var_os("EMAPPER_DEBUG_EGG").is_none() {
        return;
    }
    if total < 24 {
        let shown = if expr.len() > 160 { &expr[..160] } else { expr };
        eprintln!("[egg-fail #{total}] reason={reason} expr={shown}");
    }
}

/// `Result` -> `Option` for the e-graph cover path, with [`egg_fail_note`] instrumentation.
fn egg_cover_or_note(
    r: Result<Vec<crate::emap_cover::MappedCoverChoice>, String>,
    expr: &str,
) -> Option<Vec<crate::emap_cover::MappedCoverChoice>> {
    match r {
        Ok(v) if !v.is_empty() => Some(v),
        Ok(_) => {
            egg_fail_note("empty_cover_vec", expr);
            None
        }
        Err(e) => {
            egg_fail_note(&e, expr);
            None
        }
    }
}

/// Variables an `n`-input truth table actually depends on, in ascending order.
///
/// A cut's leaves are its cone boundary, but the cone's function need not depend on every one of
/// them. Requiring an `n`-input cell for a function with only `m < n` essential variables throws
/// away every compound cell that could implement it, which is one reason a node with a rich cut
/// pool still ends up on a 2-input cell.
pub(crate) fn tt_support(tt: &Tt9, n: u8) -> Vec<usize> {
    let mut support = Vec::new();
    for i in 0..n {
        let bit = 1usize << i;
        let mut dependent = false;
        for a in 0..(1usize << n) {
            if (a & bit) == 0 && tt9_bit_get(tt, a) != tt9_bit_get(tt, a | bit) {
                dependent = true;
                break;
            }
        }
        if dependent {
            support.push(i as usize);
        }
    }
    support
}

/// Re-express an `n`-input truth table over only `support` (a subset of its input indices).
pub(crate) fn tt_reduce(tt: &Tt9, support: &[usize]) -> Tt9 {
    let mut out = tt9_zero();
    for a in 0..(1usize << support.len()) {
        let mut full = 0usize;
        for (j, &v) in support.iter().enumerate() {
            if ((a >> j) & 1) == 1 {
                full |= 1 << v;
            }
        }
        if tt9_bit_get(tt, full) {
            tt9_bit_set(&mut out, a);
        }
    }
    out
}

/// Prepare a direct match over `support` variables instead of all cut leaves.
///
/// Returns the reduced `(n_inputs, tt, leaves, phase_arrivals, native_pos, native_neg)`.
#[allow(clippy::type_complexity)]
fn reduce_to_support(
    tt: Tt9,
    support: &[usize],
    leaves: &[usize],
    input_phase_arrivals: &[PhaseArrival],
    native_pos: &[bool],
    native_neg: &[bool],
) -> Option<(u8, Tt9, Vec<usize>, Vec<PhaseArrival>, Vec<bool>, Vec<bool>)> {
    let mut r_leaves = Vec::with_capacity(support.len());
    let mut r_arrivals = Vec::with_capacity(support.len());
    let mut r_pos = Vec::with_capacity(support.len());
    let mut r_neg = Vec::with_capacity(support.len());
    for &i in support {
        r_leaves.push(*leaves.get(i)?);
        r_arrivals.push(*input_phase_arrivals.get(i)?);
        r_pos.push(native_pos.get(i).copied().unwrap_or(false));
        r_neg.push(native_neg.get(i).copied().unwrap_or(false));
    }
    Some((
        support.len() as u8,
        tt_reduce(&tt, support),
        r_leaves,
        r_arrivals,
        r_pos,
        r_neg,
    ))
}

/// `EMAPPER_DEBUG_CUTS` counters: how many cuts present a compound (>= 3 input) function, and how
/// many of those a library cell actually realises.
static WIDE_TT_CANDIDATES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WIDE_TT_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn score_cut_for_phase_core(
    aig: &AigGraph,
    tech: &TechLib,
    cut: &Cut,
    ps: &EmapParams,
    objective: MappingObjective,
    tt_cache: &mut HashMap<String, (u8, Tt9)>,
    cover_cache: &mut CoverChoiceCache,
    perf: &mut EmapPerf,
    egg_bundle: Option<&EggRewriteBundle>,
    expr0: String,
    leaves: Vec<usize>,
    input_arrivals: Vec<f32>,
    input_phase_arrivals: Vec<PhaseArrival>,
    native_pos: &[bool],
    native_neg: &[bool],
    target_phase_neg: bool,
    required_at_output: Option<f32>,
    phase_flow: Option<&[PhaseFlow]>,
    allow_supergate: bool,
    arrival: &[f32],
) -> Option<NodeBinding> {
    // Fast path: if cut TT directly matches a techlib cell, skip egg.
    let t_tt_direct = Instant::now();
    let direct_tt = if ps.cache_tt {
        if let Some(v) = tt_cache.get(&expr0) {
            Some(*v)
        } else {
            let v = sexp_to_tt9(&expr0);
            if let Some(v2) = v {
                tt_cache.insert(expr0.clone(), v2);
            }
            v
        }
    } else {
        sexp_to_tt9(&expr0)
    };
    perf.tt += t_tt_direct.elapsed();
    /* Wide-function accounting for `EMAPPER_DEBUG_CUTS`: distinguishes "the enumeration never
    presents a compound function" from "it does and no cell realises it". Those need opposite
    fixes, and on c880/c7552 the mapper's critical paths are dominated by 2-input cells while
    mockturtle's use AOI/OAI compound cells, so which of the two it is decides the next move. */
    if direct_tt.is_some_and(|(n, _)| n >= 3) {
        WIDE_TT_CANDIDATES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let direct_key = direct_tt.map(|(n_inputs, tt)| {
        let target_tt = if target_phase_neg {
            tt9_not_masked(&tt, n_inputs)
        } else {
            tt
        };
        (n_inputs, target_tt)
    });
    if let Some((n_inputs, tt)) = direct_tt {
        /* Two lookup plans: the cut as enumerated, then -- only if that misses -- the same function
        re-expressed over its essential variables only. Both are exact; the second just stops
        insisting on a cell with the same arity as the cut. */
        let reduced_plan = tt_support(&tt, n_inputs);
        let plans: [Option<(u8, Tt9, Vec<usize>, Vec<PhaseArrival>, Vec<bool>, Vec<bool>)>; 2] = [
            (n_inputs as usize <= input_phase_arrivals.len()).then(|| {
                (
                    n_inputs,
                    tt,
                    leaves.clone(),
                    input_phase_arrivals[..n_inputs as usize].to_vec(),
                    native_pos[..n_inputs as usize].to_vec(),
                    native_neg[..n_inputs as usize].to_vec(),
                )
            }),
            (reduced_plan.len() < n_inputs as usize)
                .then(|| {
                    reduce_to_support(
                        tt,
                        &reduced_plan,
                        &leaves,
                        &input_phase_arrivals,
                        native_pos,
                        native_neg,
                    )
                })
                .flatten(),
        ];
        for (n, tt_plan, leaves_plan, arrivals, np, nn) in plans.into_iter().flatten() {
            if n == 0 {
                continue;
            }
            let t_match = Instant::now();
            let target_tt = if target_phase_neg {
                tt9_not_masked(&tt_plan, n)
            } else {
                tt_plan
            };
            let native = Some((np.as_slice(), nn.as_slice()));
            let leaf_flow_owned = phase_flow.and_then(|pf| pin_flows_for_leaves(&leaves_plan, pf));
            let leaf_flow_ref = leaf_flow_owned
                .as_ref()
                .map(|(a, b)| (a.as_slice(), b.as_slice()));
            let target_match = tech.best_for_tt_phase_arrivals(
                n,
                target_tt,
                &arrivals,
                objective,
                native,
                leaf_flow_ref,
                ps.mapping_area_cost,
            );
            perf.tech_match += t_match.elapsed();
            if let Some((cell_delay, mapping_a, physical_a, variant_ref)) = target_match {
                if !direct_variant_matches_tt(variant_ref, target_tt, n, tech) {
                    perf.direct_variant_reject += 1;
                    continue;
                }
                perf.direct_tt_hit += 1;
                if n >= 3 {
                    WIDE_TT_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                let native_delay = cell_delay + ps.wire_delay;
                return Some(NodeBinding {
                    node: cut.output_index(),
                    cut: cut.clone(),
                    leaves: leaves_plan,
                    kind: NodeBindingKind::Direct {
                        cell: variant_ref.clone(),
                    },
                    phase_neg: target_phase_neg,
                    native_delay,
                    delay: if target_phase_neg {
                        native_delay + tech.inv_delay
                    } else {
                        native_delay
                    },
                    area: physical_a,
                    mapping_cost_area: mapping_a,
                });
            }
        }
    }

    /* Step 0 of the slow path: a bounded **two-level supergate**. mockturtle matches a cut against
    composed supergates, so it can use functions such as AO221/AO32 that no single asap7 cell
    realises; emapper otherwise falls back to 2-input cells for those cones (22-30 % of nodes).
    See `crate::supergate`. */
    let two_level_max_leaves = std::env::var("EMAPPER_TWO_LEVEL_MAX_LEAVES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5);
    if ps.use_two_level_cover && leaves.len() <= two_level_max_leaves {
        if let Some(cov) = crate::supergate::two_level_cover(
            aig,
            tech,
            cut,
            &leaves,
            &input_phase_arrivals,
            objective,
            target_phase_neg,
        ) {
            let choice = crate::emap_cover::MappedCoverChoice {
                graph: cov.graph,
                delay: 0.0,
                area: cov.area,
                used_leaf_indices: (0..leaves.len()).collect(),
                pin_to_root_delay: Vec::new(),
            };
            if let Some(mut b) = mapped_cover_binding_for_arrivals(
                cut.output_index(),
                cut,
                &leaves,
                &choice,
                arrival,
                tech,
                ps.wire_delay,
            ) {
                b.phase_neg = target_phase_neg;
                b.delay = if target_phase_neg {
                    b.native_delay + tech.inv_delay
                } else {
                    b.native_delay
                };
                perf.supergate_hit += 1;
                return Some(b);
            }
        }
    }

    // Slow path: direct single-cell match failed.
    //
    // Persistent supergate DB (default `supergates.jsonl`, or `EMAPPER_SUPERGATE_DB`):
    // consult **before** the egg path and **before**
    // `should_try_egg_for_cut` — previously this lived only inside that gate, so cuts with a trivial
    // cone shape never consulted the DB even when a cover was cached on disk.
    //
    // On the timing-critical path we disable supergate (`allow_supergate`): DB entries are keyed
    // without leaf-delay diversity; mapping uses real arrivals instead.
    if allow_supergate && ps.use_egg && !target_phase_neg {
        if let Some((n_inputs, target_tt)) = direct_key {
            let req_bits = cover_cache_req_bits(required_at_output);
            /* The database buckets covers by the **exact** required-time bits it was recorded
            under: 51 184 of its 126 911 entries are unconstrained (`u32::MAX`) and the rest are
            spread over 3 050 distinct constrained buckets, so an exact hit in the constrained
            stages (1b/2/3) is essentially impossible and every one of those cuts fell through to
            the e-graph. `lookup_relaxed` exists for exactly this ("what unconstrained
            post-mapping optimization needs") but was never called. */
            let hit =
                supergate_db::lookup(n_inputs, target_tt, objective, req_bits).or_else(|| {
                    if supergate_db_relaxed() {
                        supergate_db::lookup_relaxed(n_inputs, target_tt, objective)
                    } else {
                        None
                    }
                });
            if let Some(cc) = hit {
                perf.supergate_hit += 1;
                return Some(build_mapped_cover_binding(
                    cut.output_index(),
                    cut,
                    &leaves,
                    &cc,
                    tech,
                    ps.wire_delay,
                ));
            }
        }
    }

    // Egg equality saturation + in-memory cover cache (only when multi-gate decomposition applies).
    if ps.use_egg && should_try_egg_for_cut(aig, cut, ps) {
        if target_phase_neg {
            return None;
        }
        perf.egg_attempt += 1;
        let t_egg = Instant::now();
        let cut_native =
            if native_pos.len() == native_neg.len() && native_pos.len() == input_arrivals.len() {
                Some((native_pos, native_neg))
            } else {
                None
            };
        let native_key = cut_native.map(|(np, nn)| cover_cache_native_digest(np, nn));
        let phase_key = (input_phase_arrivals.len() == input_arrivals.len())
            .then(|| cover_cache_phase_digest(&input_phase_arrivals));
        let input_phase_ref = (input_phase_arrivals.len() == input_arrivals.len())
            .then_some(input_phase_arrivals.as_slice());
        let n_alt = ps.max_egg_cover_alternatives.max(1).min(64);
        let cache_key = (
            expr0.clone(),
            objective,
            cover_cache_req_bits(required_at_output),
            native_key,
            phase_key,
            n_alt,
        );
        let cached = if ps.cache_egg {
            if let Some(v) = cover_cache.get(&cache_key) {
                v.clone()
            } else {
                let computed = extract_mapped_cover_alternatives(
                    &expr0,
                    tech,
                    &input_arrivals,
                    objective,
                    ps,
                    egg_bundle,
                    required_at_output,
                    input_phase_ref,
                    cut_native,
                    n_alt,
                );
                let computed = egg_cover_or_note(computed, &expr0);
                cover_cache.insert(cache_key.clone(), computed.clone());
                computed
            }
        } else {
            egg_cover_or_note(
                extract_mapped_cover_alternatives(
                    &expr0,
                    tech,
                    &input_arrivals,
                    objective,
                    ps,
                    egg_bundle,
                    required_at_output,
                    input_phase_ref,
                    cut_native,
                    n_alt,
                ),
                &expr0,
            )
        };
        perf.egg += t_egg.elapsed();
        if let Some(covers) = cached.filter(|v| !v.is_empty()) {
            let (binding, best_i) = pick_best_mapped_cover_binding(
                &covers,
                cut.output_index(),
                cut,
                &leaves,
                arrival,
                tech,
                ps.wire_delay,
                objective,
            )?;
            if allow_supergate {
                if let Some((n_inputs, target_tt)) = direct_key {
                    supergate_db::store(
                        n_inputs,
                        target_tt,
                        objective,
                        cover_cache_req_bits(required_at_output),
                        &covers[best_i],
                    );
                    perf.supergate_store += 1;
                }
            }
            perf.egg_coverable_hit += 1;
            perf.egg_build_ok += 1;
            /* Rank a cover on the **same** area proxy as a Direct binding.
            `build_mapped_cover_binding` sets `mapping_cost_area = cover.area` (the raw physical
            area of every cell inside the cover), while a Direct binding is ranked on
            `cell.area + Σ leaf_flow` -- the mockturtle area-flow proxy, which charges only the
            node's **marginal** area and defers the leaves' cost to their own bindings. Comparing
            the two directly makes a cover look 3-4x more expensive for the same logic (measured
            on c432: mean cover area 0.6539 vs direct 0.1527, and the cover wins on area at
            0 of 176 nodes), so the area passes discard every cover and the mapping degenerates
            into 2-input cells -- exactly the cut-width gap vs `emap`. */
            let mut binding = binding;
            /* `EMAPPER_COVER_AREA_BONUS`: artificial area credit for covers, to answer "would the
            mapping improve if covers were selected at all?" independently of the (asymmetric)
            cost model. Larger values make covers win the area passes. */
            let bonus = cover_area_bonus();
            if bonus > 0.0 {
                // Only the *ranking* proxy is adjusted; `binding.area` stays the physical area that
                // the recovery passes and the report use.
                binding.mapping_cost_area = (binding.mapping_cost_area - bonus).max(0.0);
            }
            if ps.mapping_area_cost == crate::techlib::MappingAreaCost::Flow {
                if let Some(pf) = phase_flow {
                    let mut leaf_flow_sum = 0.0f32;
                    for leaf in &binding.leaves {
                        if let Some(f) = pf.get(*leaf) {
                            leaf_flow_sum += f.pos;
                        }
                    }
                    binding.mapping_cost_area = binding.area + leaf_flow_sum;
                }
            }
            return Some(binding);
        }
        perf.egg_miss += 1;
        return None;
    }

    let t_tt = Instant::now();
    let (n_inputs, tt) = if ps.cache_tt {
        if let Some(v) = tt_cache.get(&expr0) {
            *v
        } else {
            let v = sexp_to_tt9(&expr0)?;
            tt_cache.insert(expr0.clone(), v);
            v
        }
    } else {
        sexp_to_tt9(&expr0)?
    };
    perf.tt += t_tt.elapsed();
    let mut arrivals = input_phase_arrivals;
    if n_inputs as usize > arrivals.len() {
        return None;
    }
    arrivals.truncate(n_inputs as usize);
    let n = n_inputs as usize;
    let native = Some((&native_pos[..n], &native_neg[..n]));
    let leaf_flow_owned = phase_flow.and_then(|pf| pin_flows_for_leaves(leaves.get(..n)?, pf));
    let leaf_flow_ref = leaf_flow_owned
        .as_ref()
        .map(|(a, b)| (a.as_slice(), b.as_slice()));

    let t_match = Instant::now();
    let target_tt = if target_phase_neg {
        tt9_not_masked(&tt, n_inputs)
    } else {
        tt
    };
    let (cell_delay, mapping_a, physical_a, variant_ref) = tech.best_for_tt_phase_arrivals(
        n_inputs,
        target_tt,
        &arrivals,
        objective,
        native,
        leaf_flow_ref,
        ps.mapping_area_cost,
    )?;
    if !direct_variant_matches_tt(variant_ref, target_tt, n_inputs, tech) {
        perf.direct_variant_reject += 1;
        return None;
    }
    perf.tech_match += t_match.elapsed();
    let delay = cell_delay + ps.wire_delay;

    Some(NodeBinding {
        node: cut.output_index(),
        cut: cut.clone(),
        leaves,
        kind: NodeBindingKind::Direct {
            cell: variant_ref.clone(),
        },
        phase_neg: target_phase_neg,
        native_delay: delay,
        delay: if target_phase_neg {
            delay + tech.inv_delay
        } else {
            delay
        },
        area: physical_a,
        mapping_cost_area: mapping_a,
    })
}

fn score_cut_dual_phase_for_map(
    aig: &AigGraph,
    tech: &TechLib,
    arrival: &[PhaseArrival],
    cut: &Cut,
    ps: &EmapParams,
    objective: MappingObjective,
    tt_cache: &mut HashMap<String, (u8, Tt9)>,
    cover_cache: &mut CoverChoiceCache,
    perf: &mut EmapPerf,
    egg_bundle: Option<&EggRewriteBundle>,
    best_dual: &[DualNodeBinding],
    phase_flow: &[PhaseFlow],
    required_pos: Option<f32>,
    required_neg: Option<f32>,
    critical_nodes: Option<&HashSet<usize>>,
) -> DualNodeBinding {
    let (expr0, leaves, mut input_phase_arrivals) =
        cut_to_sexp_and_phase_arrivals(aig, arrival, cut);
    let (zero_leaf, allow_supergate) =
        cut_mapping_supergate_policy(critical_nodes, cut.output_index());
    let mut input_arrivals: Vec<f32> = input_phase_arrivals.iter().map(|a| a.pos).collect();
    if zero_leaf {
        apply_zero_leaf_arrivals_for_cut_mapping(
            tech,
            &mut input_arrivals,
            &mut input_phase_arrivals,
        );
    }
    let pi_nodes: HashSet<usize> = aig.input_nodes().iter().map(|n| n.index()).collect();
    let (native_pos, native_neg) = native_phases_from_dual(&leaves, best_dual, &pi_nodes);
    let pos_arrival_f32: Vec<f32> = arrival.iter().map(|p| p.pos).collect();
    /* `EMAPPER_PESSIMISTIC_PHASE=1` refuses to credit a leaf's native negative phase: every
    negated pin is priced as `positive + inv_delay`, i.e. one explicit inverter. This isolates
    how much of the delay gap comes from the phase credit. */
    if pessimistic_phase_for(ps, objective) {
        /* `EMAPPER_NATIVE_NEG_CREDIT=1`: keep the credit for leaves that **do** have a native
        negative realization and charge the inverter only for the others.
        The blanket override was added to stop the flow crediting a negative phase that
        `collect_phase_aware_bindings` would not materialise, but it also erased a credit that is
        real: a leaf bound with `phase_neg == true` genuinely produces `~f` for free, and
        `variant_area_physical_invs` already credits it on the area side. Without the credit the
        delay model prices a compound cell that needs negated inputs (e.g. XOR as AO22 with two
        inverted pins, ~22 delay per level) at `pos + inv_delay` per negated pin, i.e. +14.77,
        which loses to the dedicated XOR2x2/XNOR2x2 cells -- the two slowest cells in asap7
        (36.3/36.9 pin delay) and the ones sitting on emapper's critical paths. */
        let native_neg_credit = std::env::var("EMAPPER_NATIVE_NEG_CREDIT")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        for (i, a) in input_phase_arrivals.iter_mut().enumerate() {
            if !native_neg_credit || !native_neg.get(i).copied().unwrap_or(false) {
                a.neg = a.pos + tech.inv_delay;
            }
        }
    }
    if let Ok(target) = std::env::var("EMAPPER_DEBUG_NODE") {
        if target.parse::<usize>() == Ok(cut.output_index()) {
            eprintln!(
                "[debug-node {}] cut_in={} leaves={:?} input_phase_arrivals={:?}",
                cut.output_index(),
                cut.input_count(),
                leaves,
                input_phase_arrivals
            );
        }
    }
    DualNodeBinding {
        pos: score_cut_for_phase_core(
            aig,
            tech,
            cut,
            ps,
            objective,
            tt_cache,
            cover_cache,
            perf,
            egg_bundle,
            expr0.clone(),
            leaves.clone(),
            input_arrivals.clone(),
            input_phase_arrivals.clone(),
            &native_pos,
            &native_neg,
            false,
            required_pos,
            Some(phase_flow),
            allow_supergate,
            &pos_arrival_f32,
        ),
        neg: score_cut_for_phase_core(
            aig,
            tech,
            cut,
            ps,
            objective,
            tt_cache,
            cover_cache,
            perf,
            egg_bundle,
            expr0,
            leaves,
            input_arrivals,
            input_phase_arrivals,
            &native_pos,
            &native_neg,
            true,
            required_neg,
            Some(phase_flow),
            allow_supergate,
            &pos_arrival_f32,
        ),
    }
}

fn direct_variant_matches_tt(
    v: &CellVariant,
    target_tt: Tt9,
    n_inputs: u8,
    tech: &TechLib,
) -> bool {
    usize::from(n_inputs) == v.n_inputs as usize && tech.variant_tt_wiring(v) == Some(target_tt)
}

/// Number of nodes in the **tree-expanded** s-expression of a cut cone.
///
/// `Cut::node_count()` counts deduplicated AIG nodes, but `cut_to_sexp_and_arrivals` writes the
/// cone as a tree: a node reachable through two paths is emitted twice, and `egg::RecExpr::parse`
/// does not hash-cons, so the parsed expression is longer than `node_count()`. The extraction
/// rejects on that parsed length (`expr_too_large`), so the cheap pre-filter has to measure the
/// same quantity or it lets through cuts the extractor will refuse. Measured on `c432` that
/// mismatch (83 > 64) was silently killing the extraction for the wide cuts that need it.
fn cut_expr_size(aig: &AigGraph, cut: &Cut) -> usize {
    let leaves: HashSet<usize> = cut.input_indices().iter().copied().collect();
    let mut size: HashMap<usize, usize> = HashMap::new();
    // `Cut` keeps `nodes` sorted ascending and AIG indices are topological, so this order has every
    // child before its parent.
    for node in cut.nodes() {
        let idx = node.index();
        let s = if idx == 0 || leaves.contains(&idx) || aig.node_type(node) != AigNode::And {
            1
        } else if let Some((xe, ye)) = aig.try_unwrap_and(node) {
            1 + size.get(&aig.edge_source(xe).index()).copied().unwrap_or(1)
                + size.get(&aig.edge_source(ye).index()).copied().unwrap_or(1)
        } else {
            1
        };
        size.insert(idx, s);
    }
    size.get(&cut.output_index()).copied().unwrap_or(1)
}

fn should_try_egg_for_cut(aig: &AigGraph, cut: &Cut, ps: &EmapParams) -> bool {
    // Egg fallback is for discovering a multi-gate implementation of the cut.
    // If the cone has only one internal node beyond its leaves, direct single-
    // cell TT matching is the relevant mechanism, not e-graph cover search.
    if cut.node_count() <= cut.input_count() + 1 {
        return false;
    }
    /* `EMAPPER_EGG_MIN_LEAVES` (default 1 = current behaviour): skip the saturation for narrow cuts.
    Every qualifying cut otherwise pays a full e-graph run, and `secs_egg` dominates the runtime
    of the larger designs. */
    if cut.input_count() < egg_min_leaves() {
        return false;
    }
    if !egg_budget_available() {
        return false;
    }

    // Avoid building/parsing an expression only to reject it later. Same metric as the extractor's
    // own `expr_too_large` check (tree-expanded length, not deduplicated node count).
    cut_expr_size(aig, cut) <= ps.max_cover_nodes
}

fn cut_to_sexp_and_arrivals(
    aig: &AigGraph,
    arrival: &[f32],
    cut: &Cut,
) -> (String, Vec<usize>, Vec<f32>) {
    // Assign leaves to variables a,b,c,... in the order of cut.inputs (excluding const 0).
    let mut leaf_vars: HashMap<usize, char> = HashMap::new();
    let mut leaves = Vec::<usize>::new();
    let mut input_arrivals = Vec::<f32>::new();

    let mut next = b'a';
    for &leaf in cut.input_indices() {
        if leaf == 0 {
            continue;
        }
        let v = char::from(next);
        leaf_vars.insert(leaf, v);
        leaves.push(leaf);
        input_arrivals.push(arrival[leaf]);
        next += 1;
    }

    fn build(
        aig: &AigGraph,
        node: usize,
        leaf_vars: &HashMap<usize, char>,
        memo: &mut HashMap<usize, String>,
    ) -> String {
        if let Some(s) = memo.get(&node) {
            return s.clone();
        }
        let s = if node == 0 {
            "0".to_string()
        } else if let Some(v) = leaf_vars.get(&node) {
            v.to_string()
        } else {
            match aig.node_type(NodeIndex::new(node)) {
                AigNode::Input(_) => {
                    // Uncut input treated as variable (should be in leaf_vars)
                    "?".to_string()
                }
                AigNode::Zero => "0".to_string(),
                AigNode::Output(_) => {
                    // Should not happen in cut internal nodes; treat as driver
                    let mut edges = aig.graph().edges_directed(NodeIndex::new(node), Incoming);
                    let e = edges.next().expect("output driven");
                    let driver = e.source().index();
                    let inv = *e.weight();
                    let inner = build(aig, driver, leaf_vars, memo);
                    if inv {
                        format!("(~ {inner})")
                    } else {
                        inner
                    }
                }
                AigNode::And => {
                    let (x_edge, y_edge) = aig.try_unwrap_and(NodeIndex::new(node)).unwrap();
                    let x = aig.edge_source(x_edge).index();
                    let y = aig.edge_source(y_edge).index();
                    let xi = aig.is_edge_inverted(x_edge);
                    let yi = aig.is_edge_inverted(y_edge);
                    let xs = build(aig, x, leaf_vars, memo);
                    let ys = build(aig, y, leaf_vars, memo);
                    let xs = if xi { format!("(~ {xs})") } else { xs };
                    let ys = if yi { format!("(~ {ys})") } else { ys };
                    format!("(& {xs} {ys})")
                }
            }
        };
        memo.insert(node, s.clone());
        s
    }

    let mut memo = HashMap::new();
    let expr = build(aig, cut.output_index(), &leaf_vars, &mut memo);
    (expr, leaves, input_arrivals)
}

fn cut_to_sexp_and_phase_arrivals(
    aig: &AigGraph,
    arrival: &[PhaseArrival],
    cut: &Cut,
) -> (String, Vec<usize>, Vec<PhaseArrival>) {
    let (expr, leaves, _) =
        cut_to_sexp_and_arrivals(aig, &arrival.iter().map(|a| a.pos).collect::<Vec<_>>(), cut);
    let input_arrivals = leaves
        .iter()
        .map(|&leaf| arrival.get(leaf).copied().unwrap_or_default())
        .collect::<Vec<_>>();
    (expr, leaves, input_arrivals)
}

/// Convert a binding to a simple instance-level fragment.
///
/// Returns `(inputs, output)` where each signal is `(node_index, inverted)`.
#[must_use]
pub fn binding_as_instance(binding: &NodeBinding) -> Option<(Vec<(usize, bool)>, (usize, bool))> {
    let NodeBindingKind::Direct { cell } = &binding.kind else {
        return None;
    };
    let mut ins = Vec::new();
    let n = cell.n_inputs as usize;
    for i in 0..n {
        let src_leaf = binding.leaves[i];
        let inv = ((cell.neg_mask >> i) & 1) == 1;
        ins.push((src_leaf, inv));
    }
    let out = (binding.node, binding.phase_neg);
    Some((ins, out))
}

/// Which logical phases of each AIG node index are referenced when emitting Verilog fanins / POs.
fn verilog_phase_needs(
    res: &EmapResult,
    po_ports: &[(usize, bool, String)],
) -> (HashSet<usize>, HashSet<usize>) {
    let mut needs_pos = HashSet::<usize>::new();
    let mut needs_neg = HashSet::<usize>::new();

    for (driver, inv, _) in po_ports {
        if *driver == 0 {
            continue;
        }
        if *inv {
            needs_neg.insert(*driver);
        } else {
            needs_pos.insert(*driver);
        }
    }

    for b in &res.bindings {
        match &b.kind {
            NodeBindingKind::Direct { cell } => {
                for i in 0..usize::from(cell.n_inputs) {
                    let leaf = b.leaves[i];
                    if leaf == 0 {
                        continue;
                    }
                    let pin_inv = ((cell.neg_mask >> i) & 1) == 1;
                    if pin_inv {
                        needs_neg.insert(leaf);
                    } else {
                        needs_pos.insert(leaf);
                    }
                }
            }
            NodeBindingKind::MappedCover { graph, .. } => {
                for n in &graph.nodes {
                    if let MappedCoverNode::Input(li) = n {
                        let leaf = b.leaves.get(*li).copied().unwrap_or(0);
                        if leaf != 0 {
                            needs_pos.insert(leaf);
                        }
                    }
                }
            }
        }
    }

    (needs_pos, needs_neg)
}

/// Write a gate-level Verilog netlist for the mapping result.
///
/// - Instantiates mapped standard cells by name using named port connections.
/// - Handles output inversion (`CellVariant.out_neg`) by inserting inverter cells (or `assign`).
/// - Primary inputs/outputs are derived from AIGER symbols when available, otherwise `$iXX` / `$oXX`.
/// - Like mockturtle-style mapping dumps: **does not** insert a buffer cell per PI; PIs drive logic
///   directly by port name. Net aliases use `assign`, not BUF cells.
pub fn write_mapped_verilog<T: std::io::Write>(
    aig: &AigGraph,
    tech: &TechLib,
    res: &EmapResult,
    mut w: T,
    module_name: &str,
) -> std::io::Result<()> {
    // Build binding lookup.
    let mut binds: HashMap<(usize, bool), NodeBinding> = HashMap::new();
    let mut bind_nodes = HashSet::<usize>::new();
    for b in &res.bindings {
        binds.insert((b.node, b.phase_neg), b.clone());
        bind_nodes.insert(b.node);
    }

    // Collect primary inputs, ordered by AIGER variable index.
    let pi_ports = aig
        .graph()
        .node_indices()
        .filter_map(|n| match aig.node_type(n) {
            AigNode::Input(var) => Some((var, n.index())),
            _ => None,
        })
        .sorted_by_key(|(var, _)| *var)
        .enumerate()
        .map(|(k, (var, node))| {
            let name = aig
                .symbol_input(var)
                .cloned()
                .unwrap_or_else(|| format!("$i{k:02}"));
            (node, sanitize_ident(&name))
        })
        .collect::<Vec<_>>();
    let pi_nodes: HashSet<usize> = pi_ports.iter().map(|(n, _)| *n).collect();
    let pi_by_node: HashMap<usize, String> = pi_ports.iter().cloned().collect();

    // Collect primary outputs (output nodes are externals(Outgoing)), ordered by output position id.
    let out_nodes = aig
        .graph()
        .externals(Outgoing)
        .filter(|n| matches!(aig.node_type(*n), AigNode::Output(_)))
        .collect::<Vec<_>>();
    let po_ports = out_nodes
        .into_iter()
        .filter_map(|o| {
            let pos_id = match aig.node_type(o) {
                AigNode::Output(i) => i,
                _ => return None,
            };
            let mut edges = aig.graph().edges_directed(o, Incoming);
            let e = edges.next().expect("output driven");
            assert!(edges.next().is_none(), "output driven by multiple nodes");
            let driver = e.source().index();
            let inv = *e.weight();
            Some((pos_id, driver, inv))
        })
        .sorted_by_key(|(pos_id, _, _)| *pos_id)
        .enumerate()
        .map(|(k, (pos_id, driver, inv))| {
            let name = aig
                .symbol_output(pos_id)
                .cloned()
                .unwrap_or_else(|| format!("$o{k:02}"));
            (driver, inv, sanitize_ident(&name))
        })
        .collect::<Vec<_>>();

    let (needs_pos, _needs_neg) = verilog_phase_needs(res, &po_ports);
    let mut needs_wire_pos = needs_pos.clone();
    for b in &res.bindings {
        if !b.phase_neg {
            needs_wire_pos.insert(b.node);
        }
    }

    // Module header.
    let port_list = pi_ports
        .iter()
        .map(|(_, n)| n.clone())
        .chain(po_ports.iter().map(|(_, _, n)| n.clone()))
        .collect::<Vec<_>>();
    writeln!(w, "module {}({});", module_name, port_list.join(", "))?;
    for (_, n) in &pi_ports {
        writeln!(w, "  input {};", n)?;
    }
    let has_binding = |node: usize| bind_nodes.contains(&node);
    let has_phase_binding = |node: usize, phase_neg: bool| binds.contains_key(&(node, phase_neg));
    let net_base = |node: usize| -> String {
        pi_by_node
            .get(&node)
            .cloned()
            .unwrap_or_else(|| format!("n{}", node))
    };
    let signal_for_phase = |node: usize, phase_neg: bool| {
        if phase_neg {
            if has_phase_binding(node, true) {
                format!("n{}_neg", node)
            } else {
                // Must match inv pre-pass: `inv_srcs` keys are `net_base(leaf)` and wires are `{src}_inv`.
                format!("{}_inv", net_base(node))
            }
        } else {
            net_base(node)
        }
    };

    // Net name needed at a PO (before merging a net into an output port).
    let po_driver_net = |driver: usize, inv: bool| -> Option<String> {
        if driver == 0 {
            return None;
        }
        Some(if inv {
            if has_phase_binding(driver, true) {
                format!("n{}_neg", driver)
            } else {
                format!("{}_inv", net_base(driver))
            }
        } else {
            net_base(driver)
        })
    };

    let mut input_ref_count: HashMap<String, usize> = HashMap::new();
    for b in &res.bindings {
        match &b.kind {
            NodeBindingKind::Direct { cell } => {
                for i in 0..usize::from(cell.n_inputs) {
                    let leaf = b.leaves[i];
                    if leaf == 0 {
                        continue;
                    }
                    let pin_inv = ((cell.neg_mask >> i) & 1) == 1;
                    let sig = signal_for_phase(leaf, pin_inv);
                    *input_ref_count.entry(sig).or_insert(0) += 1;
                }
            }
            NodeBindingKind::MappedCover { graph, .. } => {
                for n in &graph.nodes {
                    if let MappedCoverNode::Input(li) = n {
                        let leaf = b.leaves.get(*li).copied().unwrap_or(0);
                        if leaf != 0 {
                            let sig = signal_for_phase(leaf, false);
                            *input_ref_count.entry(sig).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
    }

    let mut po_by_net: HashMap<String, Vec<String>> = HashMap::new();
    for (driver, inv, port) in &po_ports {
        if let Some(net) = po_driver_net(*driver, *inv) {
            po_by_net.entry(net).or_default().push(port.clone());
        }
    }

    let mut po_net_rename: HashMap<String, String> = HashMap::new();
    for (net, ports) in &po_by_net {
        if ports.len() == 1 {
            po_net_rename.insert(net.clone(), ports[0].clone());
        }
    }

    let apply_po_rename = |s: &str| -> String {
        po_net_rename
            .get(s)
            .cloned()
            .unwrap_or_else(|| s.to_string())
    };

    let mut po_port_needs_wire: HashSet<String> = HashSet::new();
    for (net, port) in &po_net_rename {
        if input_ref_count.get(net).copied().unwrap_or(0) > 0 {
            po_port_needs_wire.insert(port.clone());
        }
    }

    for (_, _, n) in &po_ports {
        if po_port_needs_wire.contains(n) {
            writeln!(w, "  output wire {};", n)?;
        } else {
            writeln!(w, "  output {};", n)?;
        }
    }

    // Internal wires for mapped node outputs (not PIs — ports drive directly, mockturtle-style).
    let mut wires = HashSet::<usize>::new();
    for b in &res.bindings {
        wires.insert(b.node);
    }
    for n in &wires {
        if needs_wire_pos.contains(n) {
            let nname = format!("n{n}");
            if !po_net_rename.contains_key(&nname) {
                writeln!(w, "  wire n{n};")?;
            }
        }
    }
    for b in &res.bindings {
        if b.phase_neg {
            let nname = format!("n{}_neg", b.node);
            if !po_net_rename.contains_key(&nname) {
                writeln!(w, "  wire n{}_neg;", b.node)?;
            }
        }
    }

    // Inverter cells (or assign) for explicit inversions.
    let inv_cell = tech.inv_cell.clone();
    let inv_in_pin = tech.inv_in_pin.clone().unwrap_or_else(|| "A".to_string());
    let inv_out_pin = tech.inv_out_pin.clone().unwrap_or_else(|| "Y".to_string());
    let buf_cell = tech.buf_cell.clone();
    let buf_in_pin = tech.buf_in_pin.clone().unwrap_or_else(|| "A".to_string());
    let buf_out_pin = tech.buf_out_pin.clone().unwrap_or_else(|| "Y".to_string());

    fn emit_assign<T: std::io::Write>(w: &mut T, dst: &str, src: &str) -> std::io::Result<()> {
        // Never emit a self-assignment: `assign x = x;` is a combinational loop, and ABC rejects the
        // whole netlist with "Network contains a combinational loop".
        if dst == src {
            return Ok(());
        }
        writeln!(w, "  assign {} = {};", dst, src)
    }
    fn const_sig(v: bool) -> &'static str {
        if v {
            "1'b1"
        } else {
            "1'b0"
        }
    }

    fn emit_inverter<T: std::io::Write>(
        w: &mut T,
        inv_cell: &Option<String>,
        inv_in_pin: &str,
        inv_out_pin: &str,
        uid: usize,
        src: &str,
        dst: &str,
    ) -> std::io::Result<()> {
        if let Some(inv_cell) = inv_cell.as_ref() {
            writeln!(w, "  {} inv{} (", inv_cell, uid)?;
            writeln!(w, "    .{}({}),", inv_in_pin, src)?;
            writeln!(w, "    .{}({})", inv_out_pin, dst)?;
            writeln!(w, "  );")?;
        } else {
            writeln!(w, "  assign {} = ~{};", dst, src)?;
        }
        Ok(())
    }
    fn emit_buffer<T: std::io::Write>(
        w: &mut T,
        buf_cell: &Option<String>,
        buf_in_pin: &str,
        buf_out_pin: &str,
        uid: usize,
        src: &str,
        dst: &str,
    ) -> std::io::Result<()> {
        if let Some(buf_cell) = buf_cell.as_ref() {
            writeln!(w, "  {} buf{} (", buf_cell, uid)?;
            writeln!(w, "    .{}({}),", buf_in_pin, src)?;
            writeln!(w, "    .{}({})", buf_out_pin, dst)?;
            writeln!(w, "  );")?;
        } else {
            emit_assign(w, dst, src)?;
        }
        Ok(())
    }

    let sf = |node: usize, phase_neg: bool| apply_po_rename(&signal_for_phase(node, phase_neg));

    // Precompute all inversions needed (so we never emit wires inside a cell instance).
    let mut inv_srcs: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // - Input negations for each instantiated cell
    for b in &res.bindings {
        if let NodeBindingKind::Direct { cell } = &b.kind {
            for i in 0..(cell.n_inputs as usize) {
                let leaf = b.leaves[i];
                let inv = ((cell.neg_mask >> i) & 1) == 1;
                if inv && leaf != 0 {
                    if !has_phase_binding(leaf, true) {
                        inv_srcs.insert(net_base(leaf));
                    }
                }
            }
            // - Output inversion for a mapped cell (raw -> node)
            if cell.out_neg {
                inv_srcs.insert(format!("n{}_raw", b.node));
            }
        }
    }
    // - Output inversion at POs (driver -> PO); skip direct PI->PO cases because those are emitted
    //   as dedicated PO drivers (not precomputed *_inv aliases).
    for (driver, inv, _) in &po_ports {
        if *inv && *driver != 0 && !pi_nodes.contains(driver) {
            if !has_phase_binding(*driver, true) {
                inv_srcs.insert(net_base(*driver));
            }
        }
    }

    // Emit inversion wires and instances.
    /* `out_neg` cells drive `n{k}_raw`, and the inversion instance emitted just below reads it, but
    the cell body that declares `n{k}_raw` comes later in the file -- so ABC rejected the netlist
    with "Actual net n{k}_raw is missing" (c432 line 276) and `run_abc_cec` could never verify a
    mapping (`nec = -1`). Declare every `_raw` source up front, before the instances that read it. */
    for src in inv_srcs.iter() {
        if src.ends_with("_raw") {
            writeln!(w, "  wire {src};")?;
        }
    }

    let mut inv_uid: usize = 0;
    for src in inv_srcs.iter() {
        let raw_inv_net = format!("{src}_inv");
        let dst = apply_po_rename(&raw_inv_net);
        // If this net merges onto a primary output port (e.g. `n53_inv` -> `G426`), do not
        // declare `wire G426` — the inverter drives the output port directly.
        let dst_is_po_port = po_ports.iter().any(|(_, _, p)| p == &dst);
        if !dst_is_po_port {
            writeln!(w, "  wire {};", dst)?;
        }
        /* A net that drives a primary output is *renamed* to the port name, so a node whose positive
        phase is that port has no `n{k}` net at all. The destination below is renamed but the
        source used to be passed through verbatim, which emitted `INV(.A(n1233), .Y(n1233_inv))`
        for a node driven as `.Y(G5285)` -- ABC rejected it with "Actual net n1233 is missing"
        (c5315 line 1641) and `run_abc_cec` could not verify the design. Rename the source too;
        `*_raw` sources are not in `po_net_rename`, so they pass through unchanged. */
        let src_named = apply_po_rename(src);
        emit_inverter(
            &mut w,
            &inv_cell,
            &inv_in_pin,
            &inv_out_pin,
            inv_uid,
            &src_named,
            &dst,
        )?;
        inv_uid += 1;
    }

    // Instantiate cells (order doesn't strictly matter for Verilog).
    let mut ordered = res.bindings.clone();
    ordered.sort_by_key(|b| b.node);
    for b in &ordered {
        let out_net = format!("n{}", b.node);
        // When a PO renames `n{k}` to an output port (e.g. `G426`), every driver of the positive
        // phase must use that name — including the inverter that reconstructs `n{k}` from `n{k}_neg`.
        let out_net_named = apply_po_rename(&out_net);
        match &b.kind {
            NodeBindingKind::Direct { cell } => {
                let inst = format!("u{}", b.node);
                let phase_net = if b.phase_neg {
                    format!("n{}_neg", b.node)
                } else {
                    out_net.clone()
                };
                let raw_out_net = if cell.out_neg {
                    format!("n{}_raw", b.node)
                } else {
                    phase_net.clone()
                };
                if cell.out_neg && !inv_srcs.contains(&raw_out_net) {
                    writeln!(w, "  wire {};", raw_out_net)?;
                }

                writeln!(w, "  {} {} (", cell.cell_name, inst)?;
                let mut conns = Vec::<String>::new();
                for i in 0..(cell.n_inputs as usize) {
                    let pin = &cell.pin_names[i];
                    let leaf = b.leaves[i];
                    let inv = ((cell.neg_mask >> i) & 1) == 1;
                    let sig = if leaf == 0 {
                        const_sig(inv).to_string()
                    } else if inv {
                        if !has_binding(leaf) && !pi_nodes.contains(&leaf) {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                format!("unmapped internal leaf n{leaf} in direct binding"),
                            ));
                        }
                        sf(leaf, true)
                    } else {
                        if !has_binding(leaf) && !pi_nodes.contains(&leaf) {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                format!("unmapped internal leaf n{leaf} in direct binding"),
                            ));
                        }
                        sf(leaf, false)
                    };
                    conns.push(format!("    .{}({})", pin, sig));
                }
                let out_pin_net = if cell.out_neg {
                    raw_out_net.clone()
                } else {
                    apply_po_rename(&phase_net)
                };
                conns.push(format!("    .{}({})", cell.out_pin, out_pin_net));
                writeln!(w, "{}\n  );", conns.join(",\n"))?;

                if cell.out_neg {
                    let inv_out = format!("{}_inv", raw_out_net);
                    emit_assign(&mut w, &apply_po_rename(&phase_net), &inv_out)?;
                }
                if b.phase_neg && needs_pos.contains(&b.node) {
                    emit_inverter(
                        &mut w,
                        &inv_cell,
                        &inv_in_pin,
                        &inv_out_pin,
                        inv_uid,
                        &apply_po_rename(&phase_net),
                        &out_net_named,
                    )?;
                    inv_uid += 1;
                }
            }
            NodeBindingKind::MappedCover { graph, .. } => {
                let phase_net = if b.phase_neg {
                    format!("n{}_neg", b.node)
                } else {
                    out_net.clone()
                };
                // Cover DAG: only `Cell` nodes need a declared wire. `Const` / `Input` are inlined
                // (no `assign n…_cK = pi;`) so fanins connect directly to const0/const1 or PI/mapped nets.
                let mut tmp_net: Vec<String> = Vec::with_capacity(graph.nodes.len());
                for (idx, n) in graph.nodes.iter().enumerate() {
                    match n {
                        MappedCoverNode::Const(v) => {
                            let s = const_sig(*v);
                            tmp_net.push(s.to_string());
                        }
                        MappedCoverNode::Input(li) => {
                            if *li >= b.leaves.len() {
                                return Err(std::io::Error::new(
                                    std::io::ErrorKind::Other,
                                    "cover input index overflow",
                                ));
                            }
                            let leaf = b.leaves[*li];
                            let src = if leaf == 0 {
                                const_sig(false).to_string()
                            } else {
                                if !has_binding(leaf) && !pi_nodes.contains(&leaf) {
                                    return Err(std::io::Error::new(
                                        std::io::ErrorKind::Other,
                                        format!("unmapped internal leaf n{leaf} in cover binding"),
                                    ));
                                }
                                sf(leaf, false)
                            };
                            tmp_net.push(src);
                        }
                        MappedCoverNode::Cell { cell, fanins } => {
                            let wname = format!("n{}_c{}", b.node, idx);
                            writeln!(w, "  wire {};", wname)?;
                            tmp_net.push(wname.clone());
                            let mut in_sigs = Vec::<String>::new();
                            for (i, fanin) in fanins.iter().enumerate() {
                                let src = tmp_net.get(*fanin).ok_or_else(|| {
                                    std::io::Error::new(
                                        std::io::ErrorKind::Other,
                                        "mapped cover fanin overflow",
                                    )
                                })?;
                                let mut sig = src.clone();
                                if ((cell.neg_mask >> i) & 1) == 1 {
                                    let inv_in = format!("{}_invin{}_u{}_c{}", src, i, b.node, idx);
                                    writeln!(w, "  wire {};", inv_in)?;
                                    emit_inverter(
                                        &mut w,
                                        &inv_cell,
                                        &inv_in_pin,
                                        &inv_out_pin,
                                        inv_uid,
                                        src,
                                        &inv_in,
                                    )?;
                                    inv_uid += 1;
                                    sig = inv_in;
                                }
                                in_sigs.push(sig);
                            }
                            let mut out_sig = wname.clone();
                            if cell.out_neg {
                                out_sig = format!("{}_raw_u{}_c{}", wname, b.node, idx);
                                writeln!(w, "  wire {};", out_sig)?;
                            }
                            writeln!(w, "  {} u{}_c{} (", cell.cell_name, b.node, idx)?;
                            for (i, sig) in in_sigs.iter().enumerate() {
                                writeln!(w, "    .{}({}),", cell.pin_names[i], sig)?;
                            }
                            writeln!(w, "    .{}({})", cell.out_pin, out_sig)?;
                            writeln!(w, "  );")?;
                            if cell.out_neg {
                                emit_inverter(
                                    &mut w,
                                    &inv_cell,
                                    &inv_in_pin,
                                    &inv_out_pin,
                                    inv_uid,
                                    &out_sig,
                                    &wname,
                                )?;
                                inv_uid += 1;
                            }
                        }
                    }
                }
                let root_sig = &tmp_net[graph.root];
                let phase_dst = apply_po_rename(&phase_net);
                if root_sig.as_str() != phase_dst.as_str() {
                    emit_assign(&mut w, &phase_dst, root_sig)?;
                }
                if b.phase_neg && needs_pos.contains(&b.node) {
                    emit_inverter(
                        &mut w,
                        &inv_cell,
                        &inv_in_pin,
                        &inv_out_pin,
                        inv_uid,
                        &apply_po_rename(&phase_net),
                        &out_net_named,
                    )?;
                    inv_uid += 1;
                }
            }
        }
    }

    // Drive primary outputs.
    for (driver, inv, port) in &po_ports {
        if *driver != 0 && !has_binding(*driver) && !pi_nodes.contains(driver) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "unmapped internal driver n{} for primary output {}",
                    driver, port
                ),
            ));
        }
        if *driver == 0 {
            emit_assign(&mut w, port, const_sig(*inv))?;
            continue;
        }
        let want_neg = *inv;
        if pi_nodes.contains(driver) && !want_neg {
            /* `net_base` already returns a primary input's **port name**, and `po_net_rename`'s keys
            are built from `net_base(driver)` -- so when a PI drives an output directly, the PI's
            own port name becomes a rename key and `sf` (which applies `apply_po_rename`) collapses
            it onto the *destination* port: `assign G5214 = G5214;`. The self-assign guard then
            drops it and the output is left undriven (c5315: G5214/G5215/G5216/G5218). A PI's name
            is already final, so bypass the rename here. */
            emit_assign(&mut w, port, &signal_for_phase(*driver, false))?;
            continue;
        }
        /* A **primary input** driving an inverted output (`G = ~PI`): the rename guard below assumes
        some *cell* drives the renamed net, which is false for a PI, so those outputs used to come
        out completely undriven (c5315: 27 of them; ABC tied 14 to constant 0 and CEC reported 0).
        Emit the inverter here -- but only when `inv_srcs` has not already produced it. When a cell
        also consumes `~PI`, the pin loop inserts `n{k}` into `inv_srcs` and the emission loop's
        destination `apply_po_rename("n{k}_inv")` *is* this port, which is exactly how the
        duplicate `INV(.A(G165), .Y(G5195))` pair appeared. */
        if pi_nodes.contains(driver) && want_neg {
            let base = net_base(*driver);
            let already =
                inv_srcs.contains(&base) && apply_po_rename(&format!("{base}_inv")) == *port;
            if !already {
                emit_inverter(
                    &mut w,
                    &inv_cell,
                    &inv_in_pin,
                    &inv_out_pin,
                    inv_uid,
                    &signal_for_phase(*driver, false),
                    port,
                )?;
                inv_uid += 1;
            }
            continue;
        }
        if let Some(net) = po_driver_net(*driver, want_neg) {
            if po_net_rename.get(&net).is_some_and(|p| p == port) {
                continue;
            }
        }
        if want_neg {
            if pi_nodes.contains(driver) {
                emit_inverter(
                    &mut w,
                    &inv_cell,
                    &inv_in_pin,
                    &inv_out_pin,
                    inv_uid,
                    &signal_for_phase(*driver, false),
                    port,
                )?;
            } else {
                emit_buffer(
                    &mut w,
                    &buf_cell,
                    &buf_in_pin,
                    &buf_out_pin,
                    inv_uid,
                    &signal_for_phase(*driver, true),
                    port,
                )?;
            }
            inv_uid += 1;
        } else {
            emit_buffer(
                &mut w,
                &buf_cell,
                &buf_in_pin,
                &buf_out_pin,
                inv_uid,
                &signal_for_phase(*driver, false),
                port,
            )?;
            inv_uid += 1;
        }
    }
    writeln!(
        w,
        "// verilog: constants are literals; PO nets may merge onto output ports (no BUF)"
    )?;
    writeln!(w, "endmodule")?;
    Ok(())
}

fn sanitize_ident(s: &str) -> String {
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        let ok = ch.is_ascii_alphanumeric() || ch == '_' || (i == 0 && ch == '$');
        out.push(if ok { ch } else { '_' });
    }
    if out.is_empty() {
        "_".to_string()
    } else {
        out
    }
}
