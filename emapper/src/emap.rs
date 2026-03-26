use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use itertools::Itertools;
use petgraph::prelude::*;

use crate::{
    aig_graph::{Aig as AigGraph, AigNode},
    emap_cover::{build_mapped_cover_binding, extract_mapped_cover_from_expr, MappedCoverChoice},
    emap_cut::Cut,
    emap_egg::{sexp_to_tt64, EggRewriteBundle},
    techlib::{CellVariant, MappingObjective, PhaseArrival, TechLib},
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
    /// When `Some`, emapper writes a temporary Verilog netlist and runs ABC
    /// `read_genlib` / `read -m` / `print_stats` so reported QoR matches ABC.
    pub abc_genlib_path: Option<PathBuf>,
    /// Module name for Verilog sent to ABC (must match `write_mapped_verilog`).
    pub verilog_module_name: &'static str,
}

impl Default for EmapParams {
    fn default() -> Self {
        Self {
            max_cuts: 16,
            max_inputs: 6,
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
            abc_genlib_path: None,
            verilog_module_name: "top",
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
    pub area: f32,
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
}

impl Default for NodePhaseMatch {
    fn default() -> Self {
        Self {
            best: [PhaseBindingChoice::default(), PhaseBindingChoice::default()],
            best_alternative: [PhaseBindingChoice::default(), PhaseBindingChoice::default()],
            required: [f32::INFINITY; 2],
            map_refs: [0; 2],
            same_match: false,
        }
    }
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

fn choose_dual_binding_for_phase(
    best: &[DualNodeBinding],
    node: usize,
    phase_neg: bool,
    tech: &TechLib,
) -> Option<NodeBinding> {
    let binding = best.get(node)?;
    choose_dual_binding_for_request_mask(binding, if phase_neg { 0b10 } else { 0b01 }, tech)
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
        let mut dual_bindings = dual
            .into_values()
            .map(|(binding, _)| binding)
            .collect::<Vec<_>>();
        // Two physical bindings for the same AIG node would both drive `n{k}` / `n{k}_neg`
        // in the current Verilog emitter (illegal multi-driver; ABC read_verilog -m asserts).
        // Collapse to a single binding (pick by delay/area under full request_mask).
        if dual_bindings.len() == 2 && dual_bindings[0].node == dual_bindings[1].node {
            let b0 = &dual_bindings[0];
            let b1 = &dual_bindings[1];
            let d0 = binding_request_delay(b0, request_mask, tech);
            let d1 = binding_request_delay(b1, request_mask, tech);
            let a0 = binding_request_area(b0, request_mask, tech);
            let a1 = binding_request_area(b1, request_mask, tech);
            let pick = if (d0, a0) <= (d1, a1) {
                b0.clone()
            } else {
                b1.clone()
            };
            dual_bindings = vec![pick];
        }
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
        NodeBindingKind::MappedCover { .. } => binding
            .leaves
            .iter()
            .copied()
            .filter(|&leaf| leaf != 0 && aig.node_type(NodeIndex::new(leaf)) == AigNode::And)
            .map(|leaf| (leaf, false))
            .collect(),
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

fn collect_required_bindings_dual(
    aig: &AigGraph,
    best: &[DualNodeBinding],
    tech: &TechLib,
) -> Vec<NodeBinding> {
    let len = best.len();
    let mut topo_nodes = aig.graph().node_indices().map(|n| n.index()).collect_vec();
    topo_nodes.sort_unstable();
    let mut request_mask = vec![0u8; len];
    for o in aig.graph().externals(Outgoing) {
        for edge in aig.graph().edges_directed(o, Incoming) {
            let driver = edge.source().index();
            if aig.node_type(edge.source()) == AigNode::And {
                request_mask[driver] |= if *edge.weight() { 0b10 } else { 0b01 };
            }
        }
    }

    let mut selected = HashMap::<(usize, bool), NodeBinding>::new();
    for &idx in topo_nodes.iter().rev() {
        if request_mask[idx] == 0 || !matches!(aig.node_type(NodeIndex::new(idx)), AigNode::And) {
            continue;
        }
        for binding in choose_dual_bindings_for_request_mask(&best[idx], request_mask[idx], tech) {
            for (leaf, leaf_neg) in next_required_phases(&binding, aig) {
                request_mask[leaf] |= if leaf_neg { 0b10 } else { 0b01 };
            }
            selected
                .entry((binding.node, binding.phase_neg))
                .or_insert(binding);
        }
    }

    let mut bindings = selected.into_values().collect::<Vec<_>>();
    bindings.sort_by_key(|b| (b.node, b.phase_neg));
    bindings
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
    arrival: Vec<f32>,
    perf: EmapPerf,
    final_bindings: Option<Vec<NodeBinding>>,
}

#[derive(Clone, Debug, Default)]
pub struct EmapResult {
    pub bindings: Vec<NodeBinding>,
    /// Total area: ABC `print_stats` when `EmapParams::abc_genlib_path` is set; else sum of binding areas.
    pub total_area: f32,
    /// Worst delay: ABC `print_stats` when ABC stats run; else emapper PO arrival model.
    pub max_delay: f32,
    /// Gate count (`nd` in ABC): from ABC when available; else `bindings.len()`.
    pub gate_count: usize,
    /// Logic depth (`lev` in ABC): from ABC when available; else binding-tree depth estimate.
    pub logic_depth: usize,
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
}

/// Binding-tree depth estimate (used when ABC `print_stats` is not run).
pub(crate) fn binding_tree_depth(bindings: &[NodeBinding]) -> usize {
    type BindMap<'a> = HashMap<(usize, bool), &'a NodeBinding>;
    let bind_map: BindMap = bindings
        .iter()
        .map(|b| ((b.node, b.phase_neg), b))
        .collect();
    let bind_nodes: HashSet<usize> = bindings.iter().map(|b| b.node).collect();
    let mut memo = HashMap::<(usize, bool), u32>::new();

    fn depth(
        node: usize,
        phase_neg: bool,
        binds: &BindMap<'_>,
        bind_nodes: &HashSet<usize>,
        memo: &mut HashMap<(usize, bool), u32>,
    ) -> u32 {
        if let Some(&d) = memo.get(&(node, phase_neg)) {
            return d;
        }
        let d = if let Some(b) = binds.get(&(node, phase_neg)) {
            let mut best = 0u32;
            match &b.kind {
                NodeBindingKind::Direct { cell } => {
                    for (i, &leaf) in b.leaves.iter().take(cell.n_inputs as usize).enumerate() {
                        if leaf != 0 && bind_nodes.contains(&leaf) {
                            let neg = ((cell.neg_mask >> i) & 1) == 1;
                            best = best.max(depth(leaf, neg, binds, bind_nodes, memo));
                        }
                    }
                }
                NodeBindingKind::MappedCover { .. } => {
                    for &leaf in &b.leaves {
                        if leaf != 0 && bind_nodes.contains(&leaf) {
                            best = best.max(depth(leaf, false, binds, bind_nodes, memo));
                        }
                    }
                }
            }
            best + 1
        } else if phase_neg && bind_nodes.contains(&node) {
            depth(node, false, binds, bind_nodes, memo) + 1
        } else {
            0
        };
        memo.insert((node, phase_neg), d);
        d
    }

    bindings
        .iter()
        .map(|b| depth(b.node, b.phase_neg, &bind_map, &bind_nodes, &mut memo))
        .max()
        .unwrap_or(0) as usize
}

/// Egraph-based technology mapper on AIG-style cuts/expressions.
///
/// - Traverses nodes in topological order
/// - Enumerates cuts (merge children cuts + trivial cut)
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
}

pub fn emap_with_params(aig: &AigGraph, tech: &TechLib, ps: EmapParams) -> EmapResult {
    assert!(!petgraph::algo::is_cyclic_directed(aig.graph()));

    let egg_bundle_ref = ps.use_egg.then(|| tech.egg_rewrite_bundle().as_ref());

    // ── Stage 1a: delay flow ────────────────────────────────────────────────
    // Dual-phase sweep produces tighter delay bounds by propagating the
    // cheaper phase (pos / neg) to every fanout without extra inverters.
    let mut ps_delay = ps.clone();
    ps_delay.objective = MappingObjective::DelayFirst;

    let MapOnceResult {
        mut best,
        mut arrival,
        mut perf,
        final_bindings: mut final_bindings_override,
    } = if ps_delay.use_dual_phase_delay {
        delay_flow_pass(aig, tech, &ps_delay, egg_bundle_ref)
    } else {
        constrained_single_phase_pass(aig, tech, &ps_delay, None, None, egg_bundle_ref)
    };

    // ── Stage 1b: optional mixed-objective refinement ───────────────────────
    // After we have arrival times from Stage 1a, re-sweep with per-node
    // required times to pull non-critical nodes toward a mixed area/delay
    // objective, without relaxing the critical path.
    if ps_delay.critical_path_mixed_objective {
        let critical_nodes =
            collect_critical_nodes(aig, tech, &best, &arrival, ps_delay.wire_delay);
        let required = compute_required_times(aig, tech, &best, &arrival, ps_delay.wire_delay);
        let MapOnceResult {
            best: best2,
            perf: perf2,
            ..
        } = constrained_single_phase_pass(
            aig,
            tech,
            &ps_delay,
            Some(&critical_nodes),
            Some(&required),
            egg_bundle_ref,
        );
        best = best2;
        accumulate_perf(&mut perf, perf2);
        final_bindings_override = None;
    }

    // Recompute arrivals phase-aware after Stage 1.
    arrival = recompute_binding_arrivals(aig, tech, &mut best, ps_delay.wire_delay);
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
        );
        let arrival2 = recompute_binding_arrivals(aig, tech, &mut best2, ps.wire_delay);
        if primary_output_max_delay(aig, tech, &best2, &arrival2) <= delay_budget + 1e-3 {
            best = best2;
            arrival = arrival2;
            final_bindings_override = None;
        }
        accumulate_perf(&mut perf, perf2);
    }

    let bindings = final_bindings_override.unwrap_or_else(|| collect_required_bindings(aig, &best));

    let model_total_area = bindings.iter().map(|b| b.area).sum::<f32>();
    let selected_phases = bindings
        .iter()
        .map(|b| ((b.node, b.phase_neg), b.clone()))
        .collect::<HashMap<_, _>>();
    let model_max_delay = aig
        .graph()
        .externals(Outgoing)
        .flat_map(|o| aig.graph().edges_directed(o, Incoming))
        .map(|edge| {
            let driver = edge.source().index();
            if !*edge.weight() {
                arrival[driver]
            } else if let Some(binding) = selected_phases
                .get(&(driver, true))
                .or_else(|| selected_phases.get(&(driver, false)))
            {
                binding_negative_arrival(binding, tech)
            } else {
                arrival.get(driver).copied().unwrap_or(0.0) + tech.inv_delay
            }
        })
        .fold(0.0, f32::max);

    let mut total_area = model_total_area;
    let mut max_delay = model_max_delay;
    let mut gate_count = bindings.len();
    let mut logic_depth = binding_tree_depth(&bindings);

    if let Some(ref gl) = ps.abc_genlib_path {
        let tmp = std::env::temp_dir().join(format!(
            "emapper_abc_{}_{}.v",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let partial = EmapResult {
            bindings: bindings.clone(),
            total_area: model_total_area,
            max_delay: model_max_delay,
            gate_count: bindings.len(),
            logic_depth,
            perf: EmapPerf::default(),
            mapper_name: ps.mapper_name,
        };
        if let Ok(f) = std::fs::File::create(&tmp) {
            let mut bw = std::io::BufWriter::new(f);
            if write_mapped_verilog(aig, tech, &partial, &mut bw, ps.verilog_module_name).is_ok() {
                let _ = bw.flush();
                if let Some(st) = crate::abc_stats::run_abc_print_stats(gl, &tmp) {
                    total_area = st.area;
                    max_delay = st.delay;
                    gate_count = st.nd;
                    logic_depth = st.lev;
                }
            }
        }
        let _ = std::fs::remove_file(&tmp);
    }

    EmapResult {
        bindings,
        total_area,
        max_delay,
        gate_count,
        logic_depth,
        perf,
        mapper_name: ps.mapper_name,
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
) -> (Vec<Option<NodeBinding>>, Vec<f32>, EmapPerf) {
    let len = best_in.len();
    let mut best = best_in.to_vec();
    let mut arrival = arrival_in.to_vec();
    let mut perf = EmapPerf::default();
    let mut cuts: Vec<Vec<Cut>> = vec![Vec::new(); len];
    let mut tt_cache: HashMap<String, (u8, u64)> = HashMap::new();
    let mut cover_cache: HashMap<(String, MappingObjective), Option<MappedCoverChoice>> =
        HashMap::new();
    let mut topo_nodes = Vec::new();

    for pi in aig.input_nodes() {
        cuts[pi.index()] = vec![Cut::single_node(pi.index())];
    }

    let mut topo = petgraph::visit::Topo::new(aig.graph());
    while let Some(node) = topo.next(aig.graph()) {
        let idx = node.index();
        topo_nodes.push(idx);
        match aig.node_type(node) {
            AigNode::Zero => cuts[idx] = vec![Cut::single_node(idx)],
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
                let t_cut = Instant::now();
                let mut candidates = cuts[x.index()]
                    .iter()
                    .cartesian_product(&cuts[y.index()])
                    .map(|(xc, yc)| Cut::union2(xc, yc, idx))
                    .filter(|c| c.input_count() <= ps.max_inputs)
                    .collect_vec();
                candidates.push(trivial);
                candidates = candidates.into_iter().unique().collect_vec();
                perf.cut_gen += t_cut.elapsed();
                cuts[idx] = candidates;
            }
        }
    }

    let mut topo_rank = vec![usize::MAX; len];
    for (pos, &node) in topo_nodes.iter().enumerate() {
        topo_rank[node] = pos;
    }
    let mut fanouts = compute_binding_fanouts(&best);

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
        for cut in cuts[idx].clone() {
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
) -> (Vec<Option<NodeBinding>>, Vec<f32>, EmapPerf) {
    let len = best_in.len();
    let mut best = best_in.to_vec();
    let mut arrival = arrival_in.to_vec();
    let mut perf = EmapPerf::default();
    let mut cuts: Vec<Vec<Cut>> = vec![Vec::new(); len];
    let mut tt_cache: HashMap<String, (u8, u64)> = HashMap::new();
    let mut cover_cache: HashMap<(String, MappingObjective), Option<MappedCoverChoice>> =
        HashMap::new();
    let mut topo_nodes = Vec::new();

    for pi in aig.input_nodes() {
        cuts[pi.index()] = vec![Cut::single_node(pi.index())];
    }

    let mut topo = petgraph::visit::Topo::new(aig.graph());
    while let Some(node) = topo.next(aig.graph()) {
        let idx = node.index();
        topo_nodes.push(idx);
        match aig.node_type(node) {
            AigNode::Zero => cuts[idx] = vec![Cut::single_node(idx)],
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
                let mut candidates = cuts[x.index()]
                    .iter()
                    .cartesian_product(&cuts[y.index()])
                    .map(|(xc, yc)| Cut::union2(xc, yc, idx))
                    .filter(|c| c.input_count() <= ps.max_inputs)
                    .take(ps.max_cuts.saturating_mul(10))
                    .collect_vec();
                candidates.push(trivial);
                candidates = candidates.into_iter().unique().collect_vec();
                cuts[idx] = candidates.clone();
            }
        }
    }

    let mut topo_rank = vec![usize::MAX; len];
    for (pos, &node) in topo_nodes.iter().enumerate() {
        topo_rank[node] = pos;
    }
    let mut fanouts = compute_binding_fanouts(&best);

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

fn binding_dependency_leaves(binding: &NodeBinding, len: usize) -> Vec<usize> {
    let mut leaves = binding
        .leaves
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
    let mut tt_cache: HashMap<String, (u8, u64)> = HashMap::new();
    let mut cover_cache: HashMap<(String, MappingObjective), Option<MappedCoverChoice>> =
        HashMap::new();
    let mut perf = EmapPerf::default();
    let structural_request_mask = compute_structural_request_masks(aig, len);
    let est_refs = compute_est_refs(aig, len);

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
                let mut candidates = cuts[x.index()]
                    .iter()
                    .cartesian_product(&cuts[y.index()])
                    .map(|(xc, yc)| Cut::union2(xc, yc, idx))
                    .filter(|c| c.input_count() <= ps.max_inputs)
                    .take(ps.max_cuts.saturating_mul(10))
                    .collect_vec();
                candidates.push(trivial);
                candidates = candidates.into_iter().unique().collect_vec();
                perf.cut_gen += t_cut.elapsed();

                let mut node_match = NodePhaseMatch::default();
                let mut kept_cuts = Vec::<(Cut, f32, f32)>::new();

                for cut in candidates {
                    let scored = score_cut_dual_phase_for_map(
                        aig,
                        tech,
                        &phase_arrival,
                        &cut,
                        ps,
                        MappingObjective::DelayFirst,
                        &mut tt_cache,
                        &mut cover_cache,
                        &mut perf,
                        egg_bundle,
                    );

                    if let Some(posb) = scored.pos {
                        let pos_arrival = binding_positive_arrival(&posb, tech);
                        let pos_flow = binding_phase_flow(&posb, aig, &phase_flow);
                        if better_phase_choice(
                            &posb,
                            pos_arrival,
                            pos_flow,
                            &node_match.best[0],
                            false,
                        ) {
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
                        let neg_flow = binding_phase_flow(&negb, aig, &phase_flow);
                        if better_phase_choice(
                            &negb,
                            neg_arrival,
                            neg_flow,
                            &node_match.best[1],
                            false,
                        ) {
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

                select_alternatives(&mut node_match, tech, est_refs[idx]);
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
                    let req_mask = if structural_request_mask[idx] == 0 {
                        0b01
                    } else {
                        structural_request_mask[idx]
                    };
                    let realized =
                        choose_dual_bindings_for_request_mask(&best_dual[idx], req_mask, tech);
                    phase_arrival[idx] = phase_arrival_from_bindings(&realized, tech)
                        .unwrap_or_else(|| {
                            dual_binding_phase_arrival(&best_dual[idx], f32::INFINITY, tech)
                        });
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
                    if !kept_cuts.is_empty() {
                        cuts[idx] = kept_cuts
                            .into_iter()
                            .sorted_by(|a, b| {
                                (a.1, a.2)
                                    .partial_cmp(&(b.1, b.2))
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(cut, _, _)| cut)
                            .unique()
                            .take(ps.max_cuts.max(1))
                            .collect();
                    }
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
                        if let Some((n_inputs, tt)) = sexp_to_tt64(&expr_fb) {
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
        if has_required {
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
        selected[idx] = Some(binding);
    }

    let final_bindings = collect_required_bindings_dual(aig, &best_dual, tech);
    let mut best = vec![None; len];
    for &idx in &topo_and_order {
        let req_mask = if structural_request_mask[idx] == 0 {
            0b01
        } else {
            structural_request_mask[idx]
        };
        if let Some(binding) = choose_dual_binding_for_request_mask(&best_dual[idx], req_mask, tech)
        {
            best[idx] = Some(binding);
        }
    }
    for binding in selected.into_iter().flatten() {
        let node = binding.node;
        best[node] = Some(binding);
    }
    arrival = recompute_binding_arrivals(aig, tech, &mut best, ps.wire_delay);
    MapOnceResult {
        best,
        arrival,
        perf,
        final_bindings: Some(final_bindings),
    }
}

/// Single-phase mapping pass.  Used for the optional mixed-objective refinement (Stage 1b)
/// where per-node `required` times and `critical_nodes` are already known, so the
/// dual-phase sweep is not needed.
fn constrained_single_phase_pass(
    aig: &AigGraph,
    tech: &TechLib,
    ps: &EmapParams,
    critical_nodes: Option<&HashSet<usize>>,
    required: Option<&[f32]>,
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
    let mut best: Vec<Option<NodeBinding>> = vec![None; len];
    let mut arrival: Vec<f32> = vec![0.0; len];
    let mut tt_cache: HashMap<String, (u8, u64)> = HashMap::new();
    let mut cover_cache: HashMap<(String, MappingObjective), Option<MappedCoverChoice>> =
        HashMap::new();
    let mut perf = EmapPerf::default();

    for pi in aig.input_nodes() {
        cuts[pi.index()] = vec![Cut::single_node(pi.index())];
        arrival[pi.index()] = 0.0;
    }

    let mut topo = petgraph::visit::Topo::new(aig.graph());
    while let Some(node) = topo.next(aig.graph()) {
        let idx = node.index();
        match aig.node_type(node) {
            AigNode::Zero => {
                cuts[idx] = vec![Cut::single_node(idx)];
                arrival[idx] = 0.0;
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

                let t_cut = Instant::now();
                let mut candidates = cuts[x.index()]
                    .iter()
                    .cartesian_product(&cuts[y.index()])
                    .map(|(xc, yc)| Cut::union2(xc, yc, idx))
                    .filter(|c| c.input_count() <= ps.max_inputs)
                    .take(ps.max_cuts.saturating_mul(10))
                    .collect_vec();
                candidates.push(trivial);
                candidates = candidates.into_iter().unique().collect_vec();
                perf.cut_gen += t_cut.elapsed();

                let node_obj = if let Some(set) = critical_nodes {
                    if set.contains(&idx) {
                        MappingObjective::DelayFirst
                    } else {
                        MappingObjective::AreaFirst
                    }
                } else {
                    ps.objective
                };

                let mut scored = Vec::new();
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
                scored.sort_by(|a, b| {
                    let ka = match node_obj {
                        MappingObjective::DelayFirst => (a.delay, a.area),
                        MappingObjective::AreaFirst => (a.area, a.delay),
                    };
                    let kb = match node_obj {
                        MappingObjective::DelayFirst => (b.delay, b.area),
                        MappingObjective::AreaFirst => (b.area, b.delay),
                    };
                    ka.partial_cmp(&kb).unwrap()
                });
                scored.truncate(ps.max_cuts.max(1));

                if let Some(best_binding) = scored.first().cloned() {
                    if std::env::var("EMAP_TRACE_BINDINGS")
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
                            NodeBindingKind::MappedCover { graph } => {
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
                    best[idx] = Some(best_binding.clone());
                    cuts[idx] = scored.into_iter().map(|b| b.cut).collect();
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
                        if let Some((n_inputs, tt)) = sexp_to_tt64(&expr_fb) {
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
    MapOnceResult {
        best,
        arrival,
        perf,
        final_bindings: None,
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

fn recompute_binding_native_delay(
    binding: &NodeBinding,
    arrival: &[f32],
    tech: &TechLib,
    wire_delay: f32,
) -> Option<f32> {
    match &binding.kind {
        NodeBindingKind::Direct { cell } => {
            let mut delay = 0.0f32;
            for i in 0..(cell.n_inputs as usize) {
                let leaf = *binding.leaves.get(i)?;
                let mut d = *arrival.get(leaf)? + cell.pin_delay[i];
                if ((cell.neg_mask >> i) & 1) == 1 {
                    d += tech.inv_delay;
                }
                delay = delay.max(d);
            }
            if cell.out_neg {
                delay += tech.inv_delay;
            }
            Some(delay + wire_delay)
        }
        NodeBindingKind::MappedCover { graph } => {
            let mut node_delay = vec![0.0f32; graph.nodes.len()];
            for (idx, node) in graph.nodes.iter().enumerate() {
                node_delay[idx] = match node {
                    MappedCoverNode::Const(_) => 0.0,
                    MappedCoverNode::Input(li) => *arrival.get(*binding.leaves.get(*li)?)?,
                    MappedCoverNode::Cell { cell, fanins } => {
                        let mut delay = 0.0f32;
                        for (i, &fanin) in fanins.iter().enumerate() {
                            let mut d = *node_delay.get(fanin)? + cell.pin_delay[i];
                            if ((cell.neg_mask >> i) & 1) == 1 {
                                d += tech.inv_delay;
                            }
                            delay = delay.max(d);
                        }
                        if cell.out_neg {
                            delay += tech.inv_delay;
                        }
                        delay
                    }
                };
            }
            Some(*node_delay.get(graph.root)? + wire_delay)
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

fn score_cut(
    aig: &AigGraph,
    tech: &TechLib,
    arrival: &[f32],
    best: &[Option<NodeBinding>],
    cut: &Cut,
    ps: &EmapParams,
    objective: MappingObjective,
    tt_cache: &mut HashMap<String, (u8, u64)>,
    cover_cache: &mut HashMap<(String, MappingObjective), Option<MappedCoverChoice>>,
    perf: &mut EmapPerf,
    egg_bundle: Option<&EggRewriteBundle>,
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
    tt_cache: &mut HashMap<String, (u8, u64)>,
    cover_cache: &mut HashMap<(String, MappingObjective), Option<MappedCoverChoice>>,
    perf: &mut EmapPerf,
    egg_bundle: Option<&EggRewriteBundle>,
) -> DualNodeBinding {
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
    tt_cache: &mut HashMap<String, (u8, u64)>,
    cover_cache: &mut HashMap<(String, MappingObjective), Option<MappedCoverChoice>>,
    perf: &mut EmapPerf,
    egg_bundle: Option<&EggRewriteBundle>,
) -> Option<NodeBinding> {
    let (expr0, leaves, input_arrivals) = cut_to_sexp_and_arrivals(aig, arrival, cut);
    let input_phase_arrivals = leaves
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
    score_cut_for_phase_core(
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
        target_phase_neg,
    )
}

fn score_cut_for_phase_core(
    tech: &TechLib,
    cut: &Cut,
    ps: &EmapParams,
    objective: MappingObjective,
    tt_cache: &mut HashMap<String, (u8, u64)>,
    cover_cache: &mut HashMap<(String, MappingObjective), Option<MappedCoverChoice>>,
    perf: &mut EmapPerf,
    egg_bundle: Option<&EggRewriteBundle>,
    expr0: String,
    leaves: Vec<usize>,
    input_arrivals: Vec<f32>,
    input_phase_arrivals: Vec<PhaseArrival>,
    target_phase_neg: bool,
) -> Option<NodeBinding> {
    // Fast path: if cut TT directly matches a techlib cell, skip egg.
    let t_tt_direct = Instant::now();
    let direct_tt = if ps.cache_tt {
        if let Some(v) = tt_cache.get(&expr0) {
            Some(*v)
        } else {
            let v = sexp_to_tt64(&expr0);
            if let Some(v2) = v {
                tt_cache.insert(expr0.clone(), v2);
            }
            v
        }
    } else {
        sexp_to_tt64(&expr0)
    };
    perf.tt += t_tt_direct.elapsed();
    if let Some((n_inputs, tt)) = direct_tt {
        let mut arrivals = input_phase_arrivals.clone();
        if n_inputs as usize <= arrivals.len() {
            arrivals.truncate(n_inputs as usize);
            let t_match = Instant::now();
            let target_tt = if target_phase_neg {
                (!tt) & truth_table_mask(n_inputs)
            } else {
                tt
            };
            let target_match =
                tech.best_for_tt_phase_arrivals(n_inputs, target_tt, &arrivals, objective);
            perf.tech_match += t_match.elapsed();
            if let Some((cell_delay, cell_area, variant_ref)) = target_match {
                if !direct_variant_matches_tt(variant_ref, target_tt, n_inputs, tech) {
                    perf.direct_variant_reject += 1;
                } else {
                    perf.direct_tt_hit += 1;
                    let native_delay = cell_delay + ps.wire_delay;
                    return Some(NodeBinding {
                        node: cut.output_index(),
                        cut: cut.clone(),
                        leaves,
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
                        area: cell_area,
                    });
                }
            }
        }
    }

    // Slow path: direct TT miss.
    // Run egg rewriting, then extract a gate-level mapped cover.
    if ps.use_egg && should_try_egg_for_cut(cut, ps) {
        if target_phase_neg {
            return None;
        }
        perf.egg_attempt += 1;
        let t_egg = Instant::now();
        let cache_key = (expr0.clone(), objective);
        let cached = if ps.cache_egg {
            if let Some(v) = cover_cache.get(&cache_key) {
                v.clone()
            } else {
                let computed = extract_mapped_cover_from_expr(
                    &expr0,
                    tech,
                    &input_arrivals,
                    objective,
                    ps,
                    egg_bundle,
                );
                cover_cache.insert(cache_key.clone(), computed.clone());
                computed
            }
        } else {
            extract_mapped_cover_from_expr(&expr0, tech, &input_arrivals, objective, ps, egg_bundle)
        };
        perf.egg += t_egg.elapsed();
        if let Some(cc) = cached {
            perf.egg_coverable_hit += 1;
            perf.egg_build_ok += 1;
            return Some(build_mapped_cover_binding(
                cut.output_index(),
                cut,
                &leaves,
                &cc,
                ps.wire_delay,
            ));
        }
        perf.egg_miss += 1;
        return None;
    }

    let t_tt = Instant::now();
    let (n_inputs, tt) = if ps.cache_tt {
        if let Some(v) = tt_cache.get(&expr0) {
            *v
        } else {
            let v = sexp_to_tt64(&expr0)?;
            tt_cache.insert(expr0.clone(), v);
            v
        }
    } else {
        sexp_to_tt64(&expr0)?
    };
    perf.tt += t_tt.elapsed();
    let mut arrivals = input_phase_arrivals;
    if n_inputs as usize > arrivals.len() {
        return None;
    }
    arrivals.truncate(n_inputs as usize);

    let t_match = Instant::now();
    let target_tt = if target_phase_neg {
        (!tt) & truth_table_mask(n_inputs)
    } else {
        tt
    };
    let (cell_delay, cell_area, variant_ref) =
        tech.best_for_tt_phase_arrivals(n_inputs, target_tt, &arrivals, objective)?;
    if !direct_variant_matches_tt(variant_ref, target_tt, n_inputs, tech) {
        perf.direct_variant_reject += 1;
        return None;
    }
    perf.tech_match += t_match.elapsed();
    let delay = cell_delay + ps.wire_delay;
    let area = cell_area;

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
        area,
    })
}

fn score_cut_dual_phase_for_map(
    aig: &AigGraph,
    tech: &TechLib,
    arrival: &[PhaseArrival],
    cut: &Cut,
    ps: &EmapParams,
    objective: MappingObjective,
    tt_cache: &mut HashMap<String, (u8, u64)>,
    cover_cache: &mut HashMap<(String, MappingObjective), Option<MappedCoverChoice>>,
    perf: &mut EmapPerf,
    egg_bundle: Option<&EggRewriteBundle>,
) -> DualNodeBinding {
    let (expr0, leaves, input_phase_arrivals) = cut_to_sexp_and_phase_arrivals(aig, arrival, cut);
    let input_arrivals = input_phase_arrivals
        .iter()
        .map(|a| a.pos)
        .collect::<Vec<_>>();
    DualNodeBinding {
        pos: score_cut_for_phase_core(
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
            false,
        ),
        neg: score_cut_for_phase_core(
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
            true,
        ),
    }
}

fn direct_variant_matches_tt(
    v: &CellVariant,
    target_tt: u64,
    n_inputs: u8,
    tech: &TechLib,
) -> bool {
    if usize::from(n_inputs) != v.n_inputs as usize {
        return false;
    }
    let Some(cell) = tech.cells.iter().find(|c| c.name == v.cell_name) else {
        return false;
    };
    let mut var_pos = std::collections::BTreeMap::<String, usize>::new();
    for (i, name) in cell.vars.iter().enumerate() {
        var_pos.insert(name.clone(), i);
    }
    let n = usize::from(n_inputs);
    let mut out = 0u64;
    for a in 0..(1u64 << n) {
        let mut old = 0u64;
        for i in 0..n {
            let mut bit = ((a >> i) & 1) == 1;
            if ((v.neg_mask >> i) & 1) == 1 {
                bit = !bit;
            }
            let Some(pin) = v.pin_names.get(i) else {
                return false;
            };
            let Some(&oidx) = var_pos.get(pin) else {
                return false;
            };
            if bit {
                old |= 1u64 << (oidx as u64);
            }
        }
        let mut y = ((cell.tt64 >> old) & 1) == 1;
        if v.out_neg {
            y = !y;
        }
        if y {
            out |= 1u64 << a;
        }
    }
    out == target_tt
}

fn should_try_egg_for_cut(cut: &Cut, ps: &EmapParams) -> bool {
    // Egg fallback is for discovering a multi-gate implementation of the cut.
    // If the cone has only one internal node beyond its leaves, direct single-
    // cell TT matching is the relevant mechanism, not e-graph cover search.
    if cut.node_count() <= cut.input_count() + 1 {
        return false;
    }

    // Avoid building/parsing an expression only to reject it later.
    cut.node_count() <= ps.max_cover_nodes
}

fn truth_table_mask(n_inputs: u8) -> u64 {
    let bits = 1usize << usize::from(n_inputs);
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
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

/// Write a gate-level Verilog netlist for the mapping result.
///
/// - Instantiates mapped standard cells by name using named port connections.
/// - Handles output inversion (`CellVariant.out_neg`) with an inverter cell when available, else
///   a Verilog `assign` inverter (same idea as mockturtle `write_verilog` / emap-style emission).
/// - Primary inputs use module port names directly on cell pins (no PI→`n<id>` buffer), matching
///   mockturtle emap’s mapped-network + verilog naming.
/// - A buffer cell is emitted only for a non-inverted PO driven straight from a PI when the
///   library defines a buffer—mirroring `emap_impl::finalize_cover`’s `insert_buffers` / PO path.
/// - Other mapped POs: first output per `(driver, inv)` is driven **structurally** (cell output pin
///   on the module `output` net), matching mockturtle `write_verilog_with_cell`; extra POs sharing
///   the same signal use a buffer when available, else `assign`.
/// - Remaining internal stitching (covers, `out_neg`, inverters) still uses `assign` where needed.
/// - Primary I/O names come from AIGER symbols when available; otherwise `pi_{k}` / `po_{k}`
///   (avoiding `$`-prefixed identifiers, which are not portable across Verilog tools).
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
    // Same AIG node may appear twice in `res.bindings` with different `phase_neg` when
    // `collect_required_bindings_dual` picks two implementations (see `choose_dual_bindings_for_request_mask`).
    // In that case instance names (`uK`) and MappedCover internal nets (`nK_cI`) must not collide.
    let mut binding_count_per_node: HashMap<usize, usize> = HashMap::new();
    for b in &res.bindings {
        *binding_count_per_node.entry(b.node).or_insert(0) += 1;
    }
    let dual_phase_nodes: HashSet<usize> = binding_count_per_node
        .into_iter()
        .filter(|(_, c)| *c > 1)
        .map(|(n, _)| n)
        .collect();
    let inst_slug = |node: usize, phase_neg: bool| -> String {
        if dual_phase_nodes.contains(&node) {
            format!("u{}_{}", node, if phase_neg { "n" } else { "p" })
        } else {
            format!("u{}", node)
        }
    };
    let cover_net_prefix = |node: usize, phase_neg: bool| -> String {
        if dual_phase_nodes.contains(&node) {
            format!("n{}_{}", node, if phase_neg { "n" } else { "p" })
        } else {
            format!("n{}", node)
        }
    };

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
                .unwrap_or_else(|| format!("pi_{k}"));
            (node, sanitize_ident(&name))
        })
        .collect::<Vec<_>>();
    let pi_nodes: HashSet<usize> = pi_ports.iter().map(|(n, _)| *n).collect();
    let pi_name: HashMap<usize, String> = pi_ports.iter().cloned().collect();

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
                .unwrap_or_else(|| format!("po_{k}"));
            (driver, inv, sanitize_ident(&name))
        })
        .collect::<Vec<_>>();

    // Primary output port per (driver, inv) for mockturtle-style structural PO hookup.
    let mut po_by_key: HashMap<(usize, bool), Vec<String>> = HashMap::new();
    for (driver, inv, port) in &po_ports {
        po_by_key
            .entry((*driver, *inv))
            .or_default()
            .push(port.clone());
    }
    // Net that drives a PO before structural renaming (must match a binding output, not a free `_inv`).
    let po_signal_net = |node: usize, inv_po: bool| -> String {
        let pos = || {
            pi_name
                .get(&node)
                .cloned()
                .unwrap_or_else(|| format!("n{}", node))
        };
        if inv_po {
            if binds.contains_key(&(node, true)) {
                format!("n{}_neg", node)
            } else {
                format!("{}_inv", pos())
            }
        } else {
            pos()
        }
    };
    let mut structural_primary: HashMap<(usize, bool), String> = HashMap::new();
    for ((driver, inv_po), ports) in &po_by_key {
        if *driver == 0 || ports.is_empty() {
            continue;
        }
        if pi_nodes.contains(driver) && !*inv_po && tech.buf_cell.is_some() {
            continue;
        }
        if !bind_nodes.contains(driver) {
            continue;
        }
        let n = po_signal_net(*driver, *inv_po);
        let is_binding_sink = n == format!("n{}", driver) || n == format!("n{}_neg", driver);
        if !is_binding_sink {
            continue;
        }
        structural_primary.insert((*driver, *inv_po), ports[0].clone());
    }
    // Mockturtle-style structural PO hookup (cell output pin on `output` net, no assign)
    // is not compatible with ABC `read_verilog -m` on non-trivial netlists: the same
    // output net can also need to fan out to internal gates (e.g. neg-phase + trailing INV),
    // or ABC triggers Abc_ObjAddFanin assertions while building the netlist.
    // Always drive POs via internal `n{k}` / `n{k}_neg` and the final `assign`/buffer block.
    structural_primary.clear();

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
    for (_, _, n) in &po_ports {
        writeln!(w, "  output {};", n)?;
    }

    // Internal wires for mapped node outputs (omit `n{k}` / `n{k}_neg` when replaced by a PO port).
    for &k in &bind_nodes {
        let omit_n =
            structural_primary.contains_key(&(k, false)) && !binds.contains_key(&(k, true));
        if !omit_n {
            writeln!(w, "  wire n{};", k)?;
        }
    }
    for &k in &bind_nodes {
        if binds.contains_key(&(k, true)) && !structural_primary.contains_key(&(k, true)) {
            writeln!(w, "  wire n{}_neg;", k)?;
        }
    }
    // Constants (no assigns): instantiate CONST cells if present.
    // If library has no native constants, synthesize:
    //   const0 = (s & ~s), const1 = ~const0
    writeln!(w, "  wire const0;")?;
    writeln!(w, "  wire const1;")?;
    if let (Some(c0), Some(p0)) = (tech.const0_cell.as_ref(), tech.const0_out_pin.as_ref()) {
        writeln!(w, "  {} const0_u ( .{}(const0) );", c0, p0)?;
    }
    if let (Some(c1), Some(p1)) = (tech.const1_cell.as_ref(), tech.const1_out_pin.as_ref()) {
        writeln!(w, "  {} const1_u ( .{}(const1) );", c1, p1)?;
    }

    // Inverter / buffer info (explicit inversions; buffer only for emap-style PI→PO buffering).
    let inv_cell = tech.inv_cell.clone();
    let inv_in_pin = tech.inv_in_pin.clone().unwrap_or_else(|| "A".to_string());
    let inv_out_pin = tech.inv_out_pin.clone().unwrap_or_else(|| "Y".to_string());
    let buf_cell = tech.buf_cell.clone();
    let buf_in_pin = tech.buf_in_pin.clone().unwrap_or_else(|| "A".to_string());
    let buf_out_pin = tech.buf_out_pin.clone().unwrap_or_else(|| "Y".to_string());

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

    fn emit_assign_connect<T: std::io::Write>(
        w: &mut T,
        dst: &str,
        src: &str,
    ) -> std::io::Result<()> {
        writeln!(w, "  assign {} = {};", dst, src)?;
        Ok(())
    }

    let has_binding = |node: usize| bind_nodes.contains(&node);
    let has_phase_binding = |node: usize, phase_neg: bool| binds.contains_key(&(node, phase_neg));
    let pos_signal = |node: usize| -> String {
        pi_name
            .get(&node)
            .cloned()
            .unwrap_or_else(|| format!("n{}", node))
    };
    let signal_for_phase = |node: usize, phase_neg: bool| {
        if let Some(p) = structural_primary.get(&(node, phase_neg)) {
            return p.clone();
        }
        if phase_neg {
            if has_phase_binding(node, true) {
                format!("n{}_neg", node)
            } else {
                format!("{}_inv", pos_signal(node))
            }
        } else {
            pos_signal(node)
        }
    };

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
                        inv_srcs.insert(signal_for_phase(leaf, false));
                    }
                }
            }
            // - Output inversion for a mapped cell (raw -> node)
            if cell.out_neg {
                inv_srcs.insert(format!("{}_raw", cover_net_prefix(b.node, b.phase_neg)));
            }
        }
    }
    // - Output inversion at POs (driver -> PO)
    for (driver, inv, _) in &po_ports {
        if *inv && *driver != 0 {
            if !has_phase_binding(*driver, true) {
                inv_srcs.insert(signal_for_phase(*driver, false));
            }
        }
    }

    // Inverter sources may be `n{k}_raw` (cell output before polarity fix); declare those wires
    // before instances so ABC/mapped-Verilog readers see nets declared before use.
    for src in inv_srcs.iter() {
        // `n5_raw` and dual-phase `n5_p_raw` / `n5_n_raw` must be declared before use.
        if src.starts_with('n') && src.ends_with("_raw") {
            writeln!(w, "  wire {};", src)?;
        }
    }

    // Emit inversion wires and instances.
    let mut inv_uid: usize = 0;
    let mut buf_count: usize = 0;
    for src in inv_srcs.iter() {
        let dst = format!("{}_inv", src);
        writeln!(w, "  wire {};", dst)?;
        emit_inverter(
            &mut w,
            &inv_cell,
            &inv_in_pin,
            &inv_out_pin,
            inv_uid,
            src,
            &dst,
        )?;
        inv_uid += 1;
    }

    // Instantiate cells (order doesn't strictly matter for Verilog).
    let mut ordered = res.bindings.clone();
    ordered.sort_by_key(|b| b.node);
    for b in &ordered {
        let out_net = structural_primary
            .get(&(b.node, false))
            .cloned()
            .unwrap_or_else(|| format!("n{}", b.node));
        match &b.kind {
            NodeBindingKind::Direct { cell } => {
                let inst = inst_slug(b.node, b.phase_neg);
                let phase_net = if b.phase_neg {
                    structural_primary
                        .get(&(b.node, true))
                        .cloned()
                        .unwrap_or_else(|| format!("n{}_neg", b.node))
                } else {
                    out_net.clone()
                };
                let raw_out_net = if cell.out_neg {
                    format!("{}_raw", cover_net_prefix(b.node, b.phase_neg))
                } else {
                    phase_net.clone()
                };

                writeln!(w, "  {} {} (", cell.cell_name, inst)?;
                let mut conns = Vec::<String>::new();
                for i in 0..(cell.n_inputs as usize) {
                    let pin = &cell.pin_names[i];
                    let leaf = b.leaves[i];
                    let inv = ((cell.neg_mask >> i) & 1) == 1;
                    let sig = if leaf == 0 {
                        if inv {
                            "const1".to_string()
                        } else {
                            "const0".to_string()
                        }
                    } else if inv {
                        if !has_binding(leaf) && !pi_nodes.contains(&leaf) {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                format!("unmapped internal leaf n{leaf} in direct binding"),
                            ));
                        }
                        signal_for_phase(leaf, true)
                    } else {
                        if !has_binding(leaf) && !pi_nodes.contains(&leaf) {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                format!("unmapped internal leaf n{leaf} in direct binding"),
                            ));
                        }
                        signal_for_phase(leaf, false)
                    };
                    conns.push(format!("    .{}({})", pin, sig));
                }
                conns.push(format!("    .{}({})", cell.out_pin, raw_out_net));
                writeln!(w, "{}\n  );", conns.join(",\n"))?;

                if cell.out_neg {
                    let inv_out = format!("{}_inv", raw_out_net);
                    emit_assign_connect(&mut w, &phase_net, &inv_out)?;
                }
                if b.phase_neg {
                    emit_inverter(
                        &mut w,
                        &inv_cell,
                        &inv_in_pin,
                        &inv_out_pin,
                        inv_uid,
                        &phase_net,
                        &out_net,
                    )?;
                    inv_uid += 1;
                }
            }
            NodeBindingKind::MappedCover { graph } => {
                let cover_pfx = cover_net_prefix(b.node, b.phase_neg);
                let phase_net = if b.phase_neg {
                    structural_primary
                        .get(&(b.node, true))
                        .cloned()
                        .unwrap_or_else(|| format!("n{}_neg", b.node))
                } else {
                    out_net.clone()
                };
                let mut tmp_net: Vec<String> = Vec::with_capacity(graph.nodes.len());
                for i in 0..graph.nodes.len() {
                    let n = format!("{}_c{}", cover_pfx, i);
                    tmp_net.push(n.clone());
                    writeln!(w, "  wire {};", n)?;
                }
                for (idx, n) in graph.nodes.iter().enumerate() {
                    match n {
                        MappedCoverNode::Const(v) => {
                            let src = if *v { "const1" } else { "const0" };
                            emit_assign_connect(&mut w, &tmp_net[idx], src)?;
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
                                "const0".to_string()
                            } else {
                                if !has_binding(leaf) && !pi_nodes.contains(&leaf) {
                                    return Err(std::io::Error::new(
                                        std::io::ErrorKind::Other,
                                        format!("unmapped internal leaf n{leaf} in cover binding"),
                                    ));
                                }
                                signal_for_phase(leaf, false)
                            };
                            emit_assign_connect(&mut w, &tmp_net[idx], &src)?;
                        }
                        MappedCoverNode::Cell { cell, fanins } => {
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
                                    let inv_in = format!(
                                        "{}_invin{}_{}_c{}",
                                        src,
                                        i,
                                        inst_slug(b.node, b.phase_neg),
                                        idx
                                    );
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
                            let mut out_sig = tmp_net[idx].clone();
                            if cell.out_neg {
                                out_sig = format!(
                                    "{}_raw_{}_c{}",
                                    tmp_net[idx],
                                    inst_slug(b.node, b.phase_neg),
                                    idx
                                );
                                writeln!(w, "  wire {};", out_sig)?;
                            }
                            writeln!(
                                w,
                                "  {} {}_c{} (",
                                cell.cell_name,
                                inst_slug(b.node, b.phase_neg),
                                idx
                            )?;
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
                                    &tmp_net[idx],
                                )?;
                                inv_uid += 1;
                            }
                        }
                    }
                }
                emit_assign_connect(&mut w, &phase_net, &tmp_net[graph.root])?;
                if b.phase_neg {
                    emit_inverter(
                        &mut w,
                        &inv_cell,
                        &inv_in_pin,
                        &inv_out_pin,
                        inv_uid,
                        &phase_net,
                        &out_net,
                    )?;
                    inv_uid += 1;
                }
            }
        }
    }

    // Fallback constants for libraries without native CONST cells.
    if tech.const0_cell.is_none() || tech.const1_cell.is_none() {
        let seed = if let Some((_, port)) = pi_ports.first() {
            port.clone()
        } else if let Some(b0) = ordered.first() {
            format!("n{}", b0.node)
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "cannot synthesize constants without PI or mapped node",
            ));
        };
        writeln!(w, "  wire const_seed_inv;")?;
        emit_inverter(
            &mut w,
            &inv_cell,
            &inv_in_pin,
            &inv_out_pin,
            inv_uid,
            &seed,
            "const_seed_inv",
        )?;
        inv_uid += 1;
        let (_, _, andv) = tech
            .best_for_tt(2, 0b1000, &[0.0, 0.0], MappingObjective::DelayFirst)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "no 2-input AND for const fallback",
                )
            })?;
        let and_ins = [seed.clone(), "const_seed_inv".to_string()];
        let mut pin_sig = [String::new(), String::new()];
        for i in 0..2usize {
            let src = and_ins
                .get(i)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "and perm overflow"))?
                .clone();
            if ((andv.neg_mask >> i) & 1) == 1 {
                let ninv = format!("const_and_in{}_inv", i);
                writeln!(w, "  wire {};", ninv)?;
                emit_inverter(
                    &mut w,
                    &inv_cell,
                    &inv_in_pin,
                    &inv_out_pin,
                    inv_uid,
                    &src,
                    &ninv,
                )?;
                inv_uid += 1;
                pin_sig[i] = ninv;
            } else {
                pin_sig[i] = src;
            }
        }
        let and_out = if andv.out_neg {
            writeln!(w, "  wire const0_raw;")?;
            "const0_raw".to_string()
        } else {
            "const0".to_string()
        };
        writeln!(w, "  {} const0_and_u (", andv.cell_name)?;
        writeln!(w, "    .{}({}),", andv.pin_names[0], pin_sig[0])?;
        writeln!(w, "    .{}({}),", andv.pin_names[1], pin_sig[1])?;
        writeln!(w, "    .{}({})", andv.out_pin, and_out)?;
        writeln!(w, "  );")?;
        if andv.out_neg {
            emit_inverter(
                &mut w,
                &inv_cell,
                &inv_in_pin,
                &inv_out_pin,
                inv_uid,
                "const0_raw",
                "const0",
            )?;
            inv_uid += 1;
        }
        emit_inverter(
            &mut w,
            &inv_cell,
            &inv_in_pin,
            &inv_out_pin,
            inv_uid,
            "const0",
            "const1",
        )?;
        inv_uid += 1;
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
            let src = if *inv {
                "const1".to_string()
            } else {
                "const0".to_string()
            };
            emit_assign_connect(&mut w, port, &src)?;
            continue;
        }

        let src = if *inv {
            signal_for_phase(*driver, true)
        } else {
            signal_for_phase(*driver, false)
        };

        // PI → PO: buffer when library has one (emap-style), else assign.
        if pi_nodes.contains(driver) && !*inv {
            if let Some(bc) = buf_cell.as_ref() {
                writeln!(w, "  {} buf{} (", bc, inv_uid)?;
                writeln!(w, "    .{}({}),", buf_in_pin, src)?;
                writeln!(w, "    .{}({})", buf_out_pin, port)?;
                writeln!(w, "  );")?;
                buf_count += 1;
                inv_uid += 1;
            } else {
                emit_assign_connect(&mut w, port, &src)?;
            }
            continue;
        }

        // Mapped PO: primary port is driven structurally from the binding; skip here.
        if structural_primary
            .get(&(*driver, *inv))
            .is_some_and(|p| p == port)
        {
            continue;
        }
        // Same (driver, inv) as a structural primary: fanout to extra POs (mockturtle buffer split).
        if let Some(primary) = structural_primary.get(&(*driver, *inv)) {
            if let Some(bc) = buf_cell.as_ref() {
                writeln!(w, "  {} buf{} (", bc, inv_uid)?;
                writeln!(w, "    .{}({}),", buf_in_pin, primary)?;
                writeln!(w, "    .{}({})", buf_out_pin, port)?;
                writeln!(w, "  );")?;
                buf_count += 1;
                inv_uid += 1;
            } else {
                emit_assign_connect(&mut w, port, primary)?;
            }
            continue;
        }

        emit_assign_connect(&mut w, port, &src)?;
    }
    writeln!(w, "// buffers_inserted: {}", buf_count)?;
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
