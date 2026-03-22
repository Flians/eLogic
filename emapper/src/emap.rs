use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use itertools::Itertools;
use petgraph::prelude::*;

use crate::{
    aig_graph::{Aig as AigGraph, AigNode},
    emap_cut::Cut,
    emap_egg::{rewrite_to_gategraph, sexp_to_tt64, AIG, GateEGraph},
    techlib::{CellVariant, MappingObjective, TechLib},
};

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
    /// Distinguish mapper intent from mockturtle-style emap naming.
    /// This implementation targets an egraph-based mapper flow.
    pub mapper_name: &'static str,
}

impl Default for EmapParams {
    fn default() -> Self {
        Self {
            max_cuts: 16,
            max_inputs: 6,
            wire_delay: 0.0,
            use_egg: false,
            cache_egg: true,
            cache_tt: true,
            objective: MappingObjective::DelayFirst,
            critical_path_mixed_objective: false,
            area_flow_passes: 1,
            exact_area_passes: 1,
            max_cover_nodes: 64,
            mapper_name: "egraph-based-mapper",
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
pub enum MappedCoverNode {
    Const(bool),
    Input(usize),
    Cell {
        cell: CellVariant,
        fanins: Vec<usize>,
    },
}

#[derive(Clone, Debug)]
pub struct MappedCoverGraph {
    pub nodes: Vec<MappedCoverNode>,
    pub root: usize,
}

#[derive(Clone, Debug)]
pub struct NodeBinding {
    pub node: usize,
    pub cut: Cut,
    /// Leaf nodes in variable order a,b,c,... (constant 0 is omitted).
    pub leaves: Vec<usize>,
    pub kind: NodeBindingKind,
    pub delay: f32,
    pub area: f32,
}

#[derive(Clone, Debug, Default)]
pub struct EmapResult {
    pub bindings: Vec<NodeBinding>,
    pub total_area: f32,
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
}

#[derive(Clone, Debug)]
struct MappedCoverChoice {
    graph: MappedCoverGraph,
    delay: f32,
    area: f32,
    used_leaf_indices: Vec<usize>,
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

pub fn emap_with_params(aig: &AigGraph, tech: &TechLib, ps: EmapParams) -> EmapResult {
    assert!(!petgraph::algo::is_cyclic_directed(aig.graph()));

    // Stage 1: delay flow baseline.
    let mut ps_delay = ps.clone();
    ps_delay.objective = MappingObjective::DelayFirst;
    let (mut best, mut arrival, mut perf) = map_once(aig, tech, &ps_delay, None, None);

    // Optional mixed objective pass over delay baseline.
    if ps_delay.critical_path_mixed_objective {
        let critical_nodes =
            collect_critical_nodes(aig, tech, &best, &arrival, ps_delay.wire_delay);
        let required = compute_required_times(aig, tech, &best, &arrival, ps_delay.wire_delay);
        let (best2, arrival2, perf2) =
            map_once(aig, tech, &ps_delay, Some(&critical_nodes), Some(&required));
        best = best2;
        arrival = arrival2;
        perf = perf2;
    }

    // Stage 2: area-flow recovery passes.
    for _ in 0..ps.area_flow_passes {
        let mut ps_area = ps.clone();
        ps_area.objective = MappingObjective::AreaFirst;
        ps_area.critical_path_mixed_objective = false;
        let (best2, arrival2, perf2) = map_once(aig, tech, &ps_area, None, None);
        best = best2;
        arrival = arrival2;
        perf.cut_gen += perf2.cut_gen;
        perf.egg += perf2.egg;
        perf.tt += perf2.tt;
        perf.tech_match += perf2.tech_match;
        perf.direct_tt_hit += perf2.direct_tt_hit;
        perf.egg_attempt += perf2.egg_attempt;
        perf.egg_build_ok += perf2.egg_build_ok;
        perf.egg_coverable_hit += perf2.egg_coverable_hit;
        perf.egg_proxy_fallback += perf2.egg_proxy_fallback;
        perf.egg_reject_non_improving += perf2.egg_reject_non_improving;
        perf.egg_miss += perf2.egg_miss;
    }

    // Stage 3: exact-area recovery (local ref/deref style delta).
    for _ in 0..ps.exact_area_passes {
        let (best2, arrival2, perf2) = exact_area_recovery_pass(aig, tech, &ps, &best, &arrival);
        best = best2;
        arrival = arrival2;
        perf.cut_gen += perf2.cut_gen;
        perf.egg += perf2.egg;
        perf.tt += perf2.tt;
        perf.tech_match += perf2.tech_match;
        perf.direct_tt_hit += perf2.direct_tt_hit;
        perf.egg_attempt += perf2.egg_attempt;
        perf.egg_build_ok += perf2.egg_build_ok;
        perf.egg_coverable_hit += perf2.egg_coverable_hit;
        perf.egg_proxy_fallback += perf2.egg_proxy_fallback;
        perf.egg_reject_non_improving += perf2.egg_reject_non_improving;
        perf.egg_miss += perf2.egg_miss;
    }

    // Backtrace from outputs: pick nodes that feed primary outputs.
    let mut frontier = aig
        .graph()
        .externals(Outgoing)
        .flat_map(|o| aig.graph().neighbors_directed(o, Incoming))
        .filter(|n| aig.node_type(*n) == AigNode::And)
        .map(|n| n.index())
        .collect::<Vec<_>>();

    let mut visited = HashSet::<usize>::new();
    let mut bindings = Vec::<NodeBinding>::new();

    while let Some(n) = frontier.pop() {
        if !visited.insert(n) {
            continue;
        }
        let Some(b) = best[n].clone() else {
            continue;
        };
        // Push internal dependencies to frontier based on actual binding semantics.
        let dep_iter: Box<dyn Iterator<Item = usize>> = match &b.kind {
            NodeBindingKind::Direct { cell } => {
                Box::new(b.leaves.iter().copied().take(cell.n_inputs as usize))
            }
            NodeBindingKind::MappedCover { .. } => Box::new(b.leaves.iter().copied()),
        };
        for leaf in dep_iter.filter(|&i| i != 0) {
            if aig.node_type(NodeIndex::new(leaf)) == AigNode::And {
                frontier.push(leaf);
            }
        }
        bindings.push(b);
    }

    let total_area = bindings.iter().map(|b| b.area).sum::<f32>();
    let max_delay = aig
        .graph()
        .externals(Outgoing)
        .flat_map(|o| aig.graph().neighbors_directed(o, Incoming))
        .map(|d| arrival[d.index()])
        .fold(0.0, f32::max);

    EmapResult {
        bindings,
        total_area,
        max_delay,
        perf,
        mapper_name: ps.mapper_name,
    }
}

fn exact_area_recovery_pass(
    aig: &AigGraph,
    tech: &TechLib,
    ps: &EmapParams,
    best_in: &[Option<NodeBinding>],
    arrival_in: &[f32],
) -> (Vec<Option<NodeBinding>>, Vec<f32>, EmapPerf) {
    let len = best_in.len();
    let mut best = best_in.to_vec();
    let mut arrival = arrival_in.to_vec();
    let mut perf = EmapPerf::default();
    let mut cuts: Vec<Vec<Cut>> = vec![Vec::new(); len];
    let mut tt_cache: HashMap<String, (u8, u64)> = HashMap::new();
    let mut cover_cache: HashMap<(String, MappingObjective), Option<MappedCoverChoice>> =
        HashMap::new();

    for pi in aig.input_nodes() {
        cuts[pi.index()] = vec![Cut::single_node(pi.index())];
    }

    let mut topo = petgraph::visit::Topo::new(aig.graph());
    while let Some(node) = topo.next(aig.graph()) {
        let idx = node.index();
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
                    let old_delay = old_m.delay;
                    let cands_for_m = if m == idx {
                        candidates.clone()
                    } else {
                        cuts[m].clone()
                    };
                    let mut scored = Vec::new();
                    for cut in cands_for_m {
                        if let Some(cand) = score_cut(
                            aig,
                            tech,
                            &arrival,
                            &cut,
                            ps,
                            MappingObjective::AreaFirst,
                            &mut tt_cache,
                            &mut cover_cache,
                            &mut perf,
                        ) {
                            if cand.delay <= old_delay + 1e-3 {
                                scored.push(cand);
                            }
                        }
                    }
                    for cand in scored {
                        let delta = local_exact_area_delta(m, &old_m, &cand, &best, &refs);
                        if delta < best_delta {
                            best_delta = delta;
                            best_node = m;
                            best_cand = cand;
                        }
                    }
                }
                if best_delta < -1e-6 {
                    arrival[best_node] = best_cand.delay;
                    best[best_node] = Some(best_cand);
                }
            }
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

fn map_once(
    aig: &AigGraph,
    tech: &TechLib,
    ps: &EmapParams,
    critical_nodes: Option<&HashSet<usize>>,
    required: Option<&[f32]>,
) -> (Vec<Option<NodeBinding>>, Vec<f32>, EmapPerf) {
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
                        &cut,
                        ps,
                        node_obj,
                        &mut tt_cache,
                        &mut cover_cache,
                        &mut perf,
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
    (best, arrival, perf)
}

fn compute_required_times(
    aig: &AigGraph,
    tech: &TechLib,
    best: &[Option<NodeBinding>],
    arrival: &[f32],
    wire_delay: f32,
) -> Vec<f32> {
    let max_po = aig
        .graph()
        .externals(Outgoing)
        .flat_map(|o| aig.graph().neighbors_directed(o, Incoming))
        .map(|d| arrival[d.index()])
        .fold(0.0, f32::max);
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

fn collect_critical_nodes(
    aig: &AigGraph,
    tech: &TechLib,
    best: &[Option<NodeBinding>],
    arrival: &[f32],
    wire_delay: f32,
) -> HashSet<usize> {
    let mut crit = HashSet::new();
    let max_po = aig
        .graph()
        .externals(Outgoing)
        .flat_map(|o| aig.graph().neighbors_directed(o, Incoming))
        .map(|d| arrival[d.index()])
        .fold(0.0, f32::max);
    let eps = 1e-3f32;
    let mut stack = aig
        .graph()
        .externals(Outgoing)
        .flat_map(|o| aig.graph().neighbors_directed(o, Incoming))
        .filter(|d| (arrival[d.index()] - max_po).abs() <= eps)
        .map(|d| d.index())
        .collect_vec();

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
    cut: &Cut,
    ps: &EmapParams,
    objective: MappingObjective,
    tt_cache: &mut HashMap<String, (u8, u64)>,
    cover_cache: &mut HashMap<(String, MappingObjective), Option<MappedCoverChoice>>,
    perf: &mut EmapPerf,
) -> Option<NodeBinding> {
    let (expr0, leaves, input_arrivals) = cut_to_sexp_and_arrivals(aig, arrival, cut);

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
        let mut arrivals = input_arrivals.clone();
        if n_inputs as usize <= arrivals.len() {
            arrivals.truncate(n_inputs as usize);
            let t_match = Instant::now();
            if let Some((cell_delay, cell_area, variant_ref)) =
                tech.best_for_tt(n_inputs, tt, &arrivals, objective)
            {
                perf.tech_match += t_match.elapsed();
                if !direct_variant_matches_tt(variant_ref, tt, n_inputs, tech) {
                    perf.direct_variant_reject += 1;
                } else {
                    perf.direct_tt_hit += 1;
                    return Some(NodeBinding {
                        node: cut.output_index(),
                        cut: cut.clone(),
                        leaves,
                        kind: NodeBindingKind::Direct {
                            cell: variant_ref.clone(),
                        },
                        delay: cell_delay + ps.wire_delay,
                        area: cell_area,
                    });
                }
            }
            perf.tech_match += t_match.elapsed();
        }
    }

    // Slow path: direct TT miss.
    // Run egg rewriting, then extract a gate-level mapped cover.
    if ps.use_egg {
        perf.egg_attempt += 1;
        let t_egg = Instant::now();
        let cache_key = (expr0.clone(), objective);
        let cached = if ps.cache_egg {
            if let Some(v) = cover_cache.get(&cache_key) {
                v.clone()
            } else {
                let computed =
                    extract_mapped_cover_from_expr(&expr0, tech, &input_arrivals, objective, ps);
                cover_cache.insert(cache_key.clone(), computed.clone());
                computed
            }
        } else {
            extract_mapped_cover_from_expr(&expr0, tech, &input_arrivals, objective, ps)
        };
        perf.egg += t_egg.elapsed();
        if let Some(cc) = cached {
            perf.egg_coverable_hit += 1;
            perf.egg_build_ok += 1;
            return Some(NodeBinding {
                node: cut.output_index(),
                cut: cut.clone(),
                leaves: cc.used_leaf_indices.iter().map(|&i| leaves[i]).collect(),
                kind: NodeBindingKind::MappedCover {
                    graph: cc.graph.clone(),
                },
                delay: cc.delay + ps.wire_delay,
                area: cc.area,
            });
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
    let mut arrivals = input_arrivals;
    if n_inputs as usize > arrivals.len() {
        return None;
    }
    arrivals.truncate(n_inputs as usize);

    let t_match = Instant::now();
    let (cell_delay, cell_area, variant_ref) =
        tech.best_for_tt(n_inputs, tt, &arrivals, objective)?;
    if !direct_variant_matches_tt(variant_ref, tt, n_inputs, tech) {
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
        delay,
        area,
    })
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

#[derive(Clone, Debug)]
enum CoverChoiceKind {
    Const(bool),
    Input(usize),
    Gate { cell: CellVariant, fanins: Vec<usize> },
}

#[derive(Clone, Debug)]
struct CoverChoice {
    delay: f32,
    area: f32,
    kind: CoverChoiceKind,
}

fn better_choice(
    lhs: &CoverChoice,
    rhs: &CoverChoice,
    objective: MappingObjective,
) -> std::cmp::Ordering {
    match objective {
        MappingObjective::DelayFirst => (lhs.delay, lhs.area)
            .partial_cmp(&(rhs.delay, rhs.area))
            .unwrap_or(std::cmp::Ordering::Equal),
        MappingObjective::AreaFirst => (lhs.area, lhs.delay)
            .partial_cmp(&(rhs.area, rhs.delay))
            .unwrap_or(std::cmp::Ordering::Equal),
    }
}

fn gate_choice_for_variant(
    cell: &CellVariant,
    fanins: &[usize],
    child_delays: &[f32],
    child_area: f32,
    tech: &TechLib,
) -> Option<CoverChoice> {
    if fanins.len() != cell.n_inputs as usize || child_delays.len() != fanins.len() {
        return None;
    }
    let mut delay = 0.0f32;
    for (i, &child_delay) in child_delays.iter().enumerate() {
        let mut d = child_delay + cell.pin_delay[i];
        if ((cell.neg_mask >> i) & 1) == 1 {
            d += tech.inv_delay;
        }
        delay = delay.max(d);
    }
    if cell.out_neg {
        delay += tech.inv_delay;
    }
    let invs = (0..fanins.len())
        .filter(|&i| ((cell.neg_mask >> i) & 1) == 1)
        .count() as f32
        + if cell.out_neg { 1.0 } else { 0.0 };
    let area = cell.area + invs * tech.inv_area;
    Some(CoverChoice {
        delay,
        area: child_area + area,
        kind: CoverChoiceKind::Gate {
            cell: cell.clone(),
            fanins: fanins.to_vec(),
        },
    })
}

fn extract_mapped_cover_from_expr(
    expr: &str,
    tech: &TechLib,
    input_arrivals: &[f32],
    objective: MappingObjective,
    ps: &EmapParams,
) -> Option<MappedCoverChoice> {
    let rec: egg::RecExpr<AIG> = expr.parse().ok()?;
    if rec.as_ref().len() > ps.max_cover_nodes {
        return None;
    }
    let (egraph, root, gate_variants) = rewrite_to_gategraph(expr, tech, input_arrivals.len())?;
    let mut best = HashMap::<usize, Option<CoverChoice>>::new();

    fn best_eclass(
        id: egg::Id,
        egraph: &GateEGraph,
        gate_variants: &HashMap<String, CellVariant>,
        input_arrivals: &[f32],
        tech: &TechLib,
        objective: MappingObjective,
        best: &mut HashMap<usize, Option<CoverChoice>>,
        visiting: &mut HashSet<usize>,
    ) -> Option<CoverChoice> {
        let key = usize::from(id);
        if let Some(v) = best.get(&key) {
            return v.clone();
        }
        if !visiting.insert(key) {
            return None;
        }

        let mut choice = None::<CoverChoice>;

        for enode in &egraph[id].nodes {
            let op = enode.op.as_str();
            let cand = match (op, enode.children.as_slice()) {
                ("0", []) => Some(CoverChoice {
                    delay: 0.0,
                    area: 0.0,
                    kind: CoverChoiceKind::Const(false),
                }),
                ("1", []) => Some(CoverChoice {
                    delay: 0.0,
                    area: 0.0,
                    kind: CoverChoiceKind::Const(true),
                }),
                (_, []) if op.len() == 1 => {
                    let bytes = op.as_bytes();
                    if !(b'a'..=b'f').contains(&bytes[0]) {
                        None
                    } else {
                        let li = (bytes[0] - b'a') as usize;
                        input_arrivals.get(li).map(|&d| CoverChoice {
                            delay: d,
                            area: 0.0,
                            kind: CoverChoiceKind::Input(li),
                        })
                    }
                }
                ("~", [a]) => {
                    let child = best_eclass(
                        *a,
                        egraph,
                        gate_variants,
                        input_arrivals,
                        tech,
                        objective,
                        best,
                        visiting,
                    )?;
                    let cell = tech.exact_inv_variant()?;
                    gate_choice_for_variant(
                        cell,
                        &[usize::from(*a)],
                        &[child.delay],
                        child.area,
                        tech,
                    )
                }
                ("&", [a, b]) => {
                    let lhs = best_eclass(
                        *a,
                        egraph,
                        gate_variants,
                        input_arrivals,
                        tech,
                        objective,
                        best,
                        visiting,
                    )?;
                    let rhs = best_eclass(
                        *b,
                        egraph,
                        gate_variants,
                        input_arrivals,
                        tech,
                        objective,
                        best,
                        visiting,
                    )?;
                    let cell = tech.exact_and2_variant()?;
                    gate_choice_for_variant(
                        cell,
                        &[usize::from(*a), usize::from(*b)],
                        &[lhs.delay, rhs.delay],
                        lhs.area + rhs.area,
                        tech,
                    )
                }
                _ => gate_variants.get(op).and_then(|cell| {
                    let mut fanins = Vec::with_capacity(enode.children.len());
                    let mut child_delays = Vec::with_capacity(enode.children.len());
                    let mut child_area = 0.0f32;
                    for child in enode.children.iter().copied() {
                        let child_choice = best_eclass(
                            child,
                            egraph,
                            gate_variants,
                            input_arrivals,
                            tech,
                            objective,
                            best,
                            visiting,
                        )?;
                        fanins.push(usize::from(child));
                        child_delays.push(child_choice.delay);
                        child_area += child_choice.area;
                    }
                    gate_choice_for_variant(cell, &fanins, &child_delays, child_area, tech)
                }),
            };
            choice = match (choice, cand) {
                (Some(cur), Some(newc)) if better_choice(&newc, &cur, objective).is_lt() => Some(newc),
                (None, Some(newc)) => Some(newc),
                (cur, _) => cur,
            };
        }

        visiting.remove(&key);
        best.insert(key, choice.clone());
        choice
    }

    let mut visiting = HashSet::<usize>::new();
    let root_choice = best_eclass(
        root,
        &egraph,
        &gate_variants,
        input_arrivals,
        tech,
        objective,
        &mut best,
        &mut visiting,
    )?;

    let mut graph = MappedCoverGraph {
        nodes: Vec::new(),
        root: 0,
    };
    let mut input_cache = HashMap::<usize, usize>::new();
    let mut const_cache = HashMap::<bool, usize>::new();
    let mut memo = HashMap::<usize, usize>::new();

    fn materialize(
        sg_idx: usize,
        best: &HashMap<usize, Option<CoverChoice>>,
        graph: &mut MappedCoverGraph,
        input_cache: &mut HashMap<usize, usize>,
        const_cache: &mut HashMap<bool, usize>,
        memo: &mut HashMap<usize, usize>,
    ) -> Option<usize> {
        if let Some(&id) = memo.get(&sg_idx) {
            return Some(id);
        }
        let choice = best.get(&sg_idx)?.as_ref()?;
        let id = match &choice.kind {
            CoverChoiceKind::Const(v) => {
                if let Some(&id) = const_cache.get(v) {
                    id
                } else {
                    let id = graph.nodes.len();
                    graph.nodes.push(MappedCoverNode::Const(*v));
                    const_cache.insert(*v, id);
                    id
                }
            }
            CoverChoiceKind::Input(li) => {
                if let Some(&id) = input_cache.get(li) {
                    id
                } else {
                    let id = graph.nodes.len();
                    graph.nodes.push(MappedCoverNode::Input(*li));
                    input_cache.insert(*li, id);
                    id
                }
            }
            CoverChoiceKind::Gate { cell, fanins } => {
                let fanins = fanins
                    .iter()
                    .map(|fanin| materialize(*fanin, best, graph, input_cache, const_cache, memo))
                    .collect::<Option<Vec<_>>>()?;
                let id = graph.nodes.len();
                graph.nodes.push(MappedCoverNode::Cell {
                    cell: cell.clone(),
                    fanins,
                });
                id
            }
        };
        memo.insert(sg_idx, id);
        Some(id)
    }

    graph.root = materialize(
        usize::from(root),
        &best,
        &mut graph,
        &mut input_cache,
        &mut const_cache,
        &mut memo,
    )?;

    let mut used_leaf_indices = graph
        .nodes
        .iter()
        .filter_map(|n| match n {
            MappedCoverNode::Input(i) => Some(*i),
            _ => None,
        })
        .collect::<Vec<_>>();
    used_leaf_indices.sort_unstable();
    used_leaf_indices.dedup();

    Some(MappedCoverChoice {
        graph,
        delay: root_choice.delay,
        area: root_choice.area,
        used_leaf_indices,
    })
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
    let out = (binding.node, cell.out_neg);
    Some((ins, out))
}

/// Write a gate-level Verilog netlist for the mapping result.
///
/// - Instantiates mapped standard cells by name using named port connections.
/// - Handles output inversion (`CellVariant.out_neg`) by inserting a Verilog `assign` inverter.
/// - Primary inputs/outputs are derived from AIGER symbols when available, otherwise `$iXX` / `$oXX`.
pub fn write_mapped_verilog<T: std::io::Write>(
    aig: &AigGraph,
    tech: &TechLib,
    res: &EmapResult,
    mut w: T,
    module_name: &str,
) -> std::io::Result<()> {
    // Build binding lookup.
    let mut binds: HashMap<usize, NodeBinding> = HashMap::new();
    for b in &res.bindings {
        binds.insert(b.node, b.clone());
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

    // Internal wires for mapped node outputs.
    let mut wires = HashSet::<usize>::new();
    for b in &res.bindings {
        wires.insert(b.node);
    }
    // Also include PI nodes as internal nets.
    for (n, _) in &pi_ports {
        wires.insert(*n);
    }
    for n in &wires {
        writeln!(w, "  wire n{};", n)?;
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

    // Inverter / buffer info (for explicit inversions and no-assign connections).
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
            Ok(())
        } else {
            // If no buffer cell exists, fallback to a direct connection (would require assign).
            // Caller ensures a buffer exists to keep netlist assign-free.
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "no buffer cell available for assign-free netlist",
            ))
        }
    }

    // Precompute all inversions needed (so we never emit wires inside a cell instance).
    let mut inv_srcs: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // - Input negations for each instantiated cell
    for b in &res.bindings {
        if let NodeBindingKind::Direct { cell } = &b.kind {
            for i in 0..(cell.n_inputs as usize) {
                let leaf = b.leaves[i];
                let inv = ((cell.neg_mask >> i) & 1) == 1;
                if inv && leaf != 0 {
                    inv_srcs.insert(format!("n{}", leaf));
                }
            }
            // - Output inversion for a mapped cell (raw -> node)
            if cell.out_neg {
                inv_srcs.insert(format!("n{}_raw", b.node));
            }
        }
    }
    // - Output inversion at POs (driver -> PO)
    for (driver, inv, _) in &po_ports {
        if *inv && *driver != 0 {
            inv_srcs.insert(format!("n{}", driver));
        }
    }

    // Emit inversion wires and instances.
    let mut inv_uid: usize = 0;
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
        let out_net = format!("n{}", b.node);
        match &b.kind {
            NodeBindingKind::Direct { cell } => {
                let inst = format!("u{}", b.node);
                let raw_out_net = if cell.out_neg {
                    format!("n{}_raw", b.node)
                } else {
                    out_net.clone()
                };
                if cell.out_neg {
                    writeln!(w, "  wire {};", raw_out_net)?;
                }

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
                        if !binds.contains_key(&leaf) && !pi_nodes.contains(&leaf) {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                format!("unmapped internal leaf n{leaf} in direct binding"),
                            ));
                        }
                        format!("n{}_inv", leaf)
                    } else {
                        if !binds.contains_key(&leaf) && !pi_nodes.contains(&leaf) {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                format!("unmapped internal leaf n{leaf} in direct binding"),
                            ));
                        }
                        format!("n{}", leaf)
                    };
                    conns.push(format!("    .{}({})", pin, sig));
                }
                conns.push(format!("    .{}({})", cell.out_pin, raw_out_net));
                writeln!(w, "{}\n  );", conns.join(",\n"))?;

                if cell.out_neg {
                    let inv_out = format!("{}_inv", raw_out_net);
                    emit_buffer(
                        &mut w,
                        &buf_cell,
                        &buf_in_pin,
                        &buf_out_pin,
                        inv_uid,
                        &inv_out,
                        &out_net,
                    )?;
                    inv_uid += 1;
                }
            }
            NodeBindingKind::MappedCover { graph } => {
                let mut tmp_net: Vec<String> = Vec::with_capacity(graph.nodes.len());
                for i in 0..graph.nodes.len() {
                    let n = format!("n{}_c{}", b.node, i);
                    tmp_net.push(n.clone());
                    writeln!(w, "  wire {};", n)?;
                }
                for (idx, n) in graph.nodes.iter().enumerate() {
                    match n {
                        MappedCoverNode::Const(v) => {
                            let src = if *v { "const1" } else { "const0" };
                            emit_buffer(
                                &mut w,
                                &buf_cell,
                                &buf_in_pin,
                                &buf_out_pin,
                                inv_uid,
                                src,
                                &tmp_net[idx],
                            )?;
                            inv_uid += 1;
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
                                if !binds.contains_key(&leaf) && !pi_nodes.contains(&leaf) {
                                    return Err(std::io::Error::new(
                                        std::io::ErrorKind::Other,
                                        format!("unmapped internal leaf n{leaf} in cover binding"),
                                    ));
                                }
                                format!("n{}", leaf)
                            };
                            emit_buffer(
                                &mut w,
                                &buf_cell,
                                &buf_in_pin,
                                &buf_out_pin,
                                inv_uid,
                                &src,
                                &tmp_net[idx],
                            )?;
                            inv_uid += 1;
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
                                    let inv_in =
                                        format!("{}_invin{}_u{}_c{}", src, i, b.node, idx);
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
                                out_sig = format!("{}_raw_u{}_c{}", tmp_net[idx], b.node, idx);
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
                                    &tmp_net[idx],
                                )?;
                                inv_uid += 1;
                            }
                        }
                    }
                }
                emit_buffer(
                    &mut w,
                    &buf_cell,
                    &buf_in_pin,
                    &buf_out_pin,
                    inv_uid,
                    &tmp_net[graph.root],
                    &out_net,
                )?;
                inv_uid += 1;
            }
        }
    }

    // No assigns for PI connections: use port name directly (PIs do not need nX nets).
    // To keep existing internal naming, we buffer each PI port into its node net n<id>.
    for (node, port) in &pi_ports {
        let dst = format!("n{}", node);
        emit_buffer(
            &mut w,
            &buf_cell,
            &buf_in_pin,
            &buf_out_pin,
            inv_uid,
            port,
            &dst,
        )?;
        inv_uid += 1;
    }

    // Fallback constants for libraries without native CONST cells.
    if tech.const0_cell.is_none() || tech.const1_cell.is_none() {
        let seed = if let Some((n, _)) = pi_ports.first() {
            format!("n{}", n)
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
        if *driver != 0 && !binds.contains_key(driver) && !pi_nodes.contains(driver) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "unmapped internal driver n{} for primary output {}",
                    driver, port
                ),
            ));
        }
        let src = if *driver == 0 {
            if *inv {
                "const1".to_string()
            } else {
                "const0".to_string()
            }
        } else if *inv {
            format!("n{}_inv", driver)
        } else {
            format!("n{}", driver)
        };
        emit_buffer(
            &mut w,
            &buf_cell,
            &buf_in_pin,
            &buf_out_pin,
            inv_uid,
            &src,
            port,
        )?;
        inv_uid += 1;
    }

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
