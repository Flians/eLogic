use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::{
    binding::NodeBindingKind,
    params::EmapParams,
    emap_egg::{
        rewrite_to_gategraph_with_limits, sexp_to_tt9, EggRewriteBundle, GateEGraph, AIG,
    },
    techlib::{Cell, CellVariant, MappingAreaCost, MappingObjective, PhaseArrival, TechLib},
    tt9::Tt9,
};

const REQ_TIMING_EPS: f32 = 1e-3;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MappedCoverNode {
    Const(bool),
    Input(usize),
    Cell {
        cell: CellVariant,
        fanins: Vec<usize>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MappedCoverGraph {
    pub nodes: Vec<MappedCoverNode>,
    pub root: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MappedCoverChoice {
    pub graph: MappedCoverGraph,
    pub delay: f32,
    pub area: f32,
    pub used_leaf_indices: Vec<usize>,
    /// Per **variable index** `i` in `0..n_inputs` (same length as `input_arrivals` to
    /// [`extract_mapped_cover_from_expr`]): mockturtle-style delay from that cut input through the
    /// cover to the root output — genlib `pin_delay` along the path, plus input negation inverters,
    /// plus each cell's `out_neg` inverter on the path (see [`compute_pin_to_root_delays`]).
    #[serde(default)]
    pub pin_to_root_delay: Vec<f32>,
}

/// Per-variable-index delays from each primary input of the cover to the root output, aligned with
/// mockturtle `supergate.tdelay`: walk root-to-leaves accumulating `pin_delay` and inverter costs.
///
/// `max_var` is the number of cut variables (length of `input_arrivals` in extraction); unused
/// indices stay `0`.
#[must_use]
pub fn compute_pin_to_root_delays(graph: &MappedCoverGraph, tech: &TechLib, max_var: usize) -> Vec<f32> {
    if max_var == 0 {
        return Vec::new();
    }
    let mut tdelay = vec![0.0f32; max_var];
    fn walk(
        node_idx: usize,
        delay_from_here_to_cover_root: f32,
        graph: &MappedCoverGraph,
        tech: &TechLib,
        tdelay: &mut [f32],
    ) {
        match &graph.nodes[node_idx] {
            MappedCoverNode::Const(_) => {}
            MappedCoverNode::Input(li) => {
                if *li < tdelay.len() {
                    tdelay[*li] = tdelay[*li].max(delay_from_here_to_cover_root);
                }
            }
            MappedCoverNode::Cell { cell, fanins } => {
                for (i, &fi) in fanins.iter().enumerate() {
                    let mut edge = cell.pin_delay.get(i).copied().unwrap_or(0.0);
                    if ((cell.neg_mask >> i) & 1) == 1 {
                        edge += tech.inv_delay;
                    }
                    if cell.out_neg {
                        edge += tech.inv_delay;
                    }
                    walk(
                        fi,
                        delay_from_here_to_cover_root + edge,
                        graph,
                        tech,
                        tdelay,
                    );
                }
            }
        }
    }
    walk(graph.root, 0.0, graph, tech, &mut tdelay);
    tdelay
}

#[derive(Clone, Debug)]
enum CoverChoiceKind {
    Const(bool),
    Input(usize),
    Gate {
        cell: CellVariant,
        fanins: Vec<usize>,
    },
}

#[derive(Clone, Debug)]
struct CoverChoice {
    /// Primary delay metric for tie-breaking: positive-phase arrival at this e-class root.
    delay: f32,
    /// Positive / negative phase arrivals (matches [`TechLib::best_variant_from_candidates`]).
    phase: PhaseArrival,
    /// Sum of gate areas over **distinct** e-class ids in this cover (DAG-accurate).
    area: f32,
    kind: CoverChoiceKind,
    /// Per e-class (`egg::Id`) gate area for the chosen enode; used to merge without
    /// double-counting shared subgraphs (multi-fanout), cf. `UniqueNodeExtractor` in mig_egg.
    class_areas: HashMap<usize, f32>,
}

#[derive(Clone, Debug)]
enum CoverDagNodeOp {
    Const(bool),
    Input(usize),
    Not,
    And,
    Gate(String),
}

#[derive(Clone, Debug)]
struct CoverDagNode {
    op: CoverDagNodeOp,
    children: Vec<usize>,
}

#[derive(Clone, Debug)]
struct CoverDag {
    root_class: usize,
    class_nodes: HashMap<usize, Vec<usize>>,
    nodes: Vec<CoverDagNode>,
    parents: HashMap<usize, Vec<usize>>,
}

#[derive(Default)]
struct UniqueQueue<T> {
    queued: HashSet<T>,
    queue: VecDeque<T>,
}

impl<T> UniqueQueue<T>
where
    T: Clone + Eq + std::hash::Hash,
{
    fn insert(&mut self, value: T) {
        if self.queued.insert(value.clone()) {
            self.queue.push_back(value);
        }
    }

    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for value in iter {
            self.insert(value);
        }
    }

    fn pop(&mut self) -> Option<T> {
        let value = self.queue.pop_front()?;
        self.queued.remove(&value);
        Some(value)
    }
}

fn union_class_areas(lhs: &HashMap<usize, f32>, rhs: &HashMap<usize, f32>) -> HashMap<usize, f32> {
    let mut out = lhs.clone();
    for (k, v) in rhs {
        match out.get(k) {
            None => {
                out.insert(*k, *v);
            }
            Some(ev) => {
                // The same e-class can appear in multiple child covers when subgraphs
                // merge; enode/tech paths may assign slightly different gate areas.
                // Take the max so the merged map stays a conservative per-class cost.
                out.insert(*k, ev.max(*v));
            }
        }
    }
    out
}

fn merge_child_class_areas(
    self_id: usize,
    child_ids: &[usize],
    best_map: &HashMap<usize, CoverChoice>,
) -> Option<(Vec<PhaseArrival>, HashMap<usize, f32>)> {
    let mut child_phases = Vec::with_capacity(child_ids.len());
    let mut largest_child_idx = None;
    let mut largest_size = 0usize;

    for (index, child_id) in child_ids.iter().copied().enumerate() {
        let child_choice = best_map.get(&child_id)?;
        if child_choice.class_areas.contains_key(&self_id) {
            return None;
        }
        child_phases.push(child_choice.phase);
        if child_choice.class_areas.len() > largest_size {
            largest_size = child_choice.class_areas.len();
            largest_child_idx = Some(index);
        }
    }

    let mut merged = largest_child_idx
        .map(|index| best_map[&child_ids[index]].class_areas.clone())
        .unwrap_or_default();

    for (index, child_id) in child_ids.iter().copied().enumerate() {
        if Some(index) == largest_child_idx {
            continue;
        }
        merged = union_class_areas(&merged, &best_map[&child_id].class_areas);
    }

    Some((child_phases, merged))
}

#[inline]
fn input_phase_arrival_for_leaf(
    li: usize,
    phase_arrivals: Option<&[PhaseArrival]>,
    scalar_arrivals: &[f32],
    tech: &TechLib,
) -> Option<PhaseArrival> {
    let d = scalar_arrivals.get(li).copied()?;
    let phase = match phase_arrivals {
        Some(pa) if pa.len() == scalar_arrivals.len() => pa.get(li).copied(),
        _ => None,
    };
    Some(phase.unwrap_or(PhaseArrival {
        pos: d,
        neg: d + tech.inv_delay,
    }))
}

fn build_cover_dag(egraph: &GateEGraph, root: egg::Id) -> CoverDag {
    let root_class = usize::from(root);
    /* Sorted, not `HashSet`: the iteration order below assigns `node_id`s and `parents` entries, and
       the DP breaks cost ties by first-encountered. A hash order makes the extracted cover (and the
       netlist built from it) vary between processes even though the cost is identical. */
    let mut reachable = std::collections::BTreeSet::<usize>::new();
    let mut stack = vec![root_class];
    while let Some(class_id) = stack.pop() {
        if !reachable.insert(class_id) {
            continue;
        }
        for enode in &egraph[egg::Id::from(class_id)].nodes {
            for child in enode.children.iter().copied() {
                stack.push(usize::from(child));
            }
        }
    }

    let mut class_nodes = HashMap::<usize, Vec<usize>>::new();
    let mut parents = HashMap::<usize, Vec<usize>>::new();
    for &class_id in &reachable {
        class_nodes.insert(class_id, Vec::new());
        parents.insert(class_id, Vec::new());
    }

    let mut nodes = Vec::<CoverDagNode>::new();
    for &class_id in &reachable {
        for enode in &egraph[egg::Id::from(class_id)].nodes {
            let op = match (enode.op.as_str(), enode.children.as_slice()) {
                ("0", []) => CoverDagNodeOp::Const(false),
                ("1", []) => CoverDagNodeOp::Const(true),
                (symbol, []) if symbol.len() == 1 => {
                    let byte = symbol.as_bytes()[0];
                    /* Cut leaves are `a`..=`i`: `sexp_to_tt9`, `gate_expr_n_inputs` and the
                       `GateTtFold` analysis all accept nine. Recognizing only `a`..=`f` here made
                       leaves `g`/`h`/`i` look like library-cell symbols, so every cut wider than six
                       leaves had unpriced e-classes and the cover extraction failed with
                       `root_choice_missing` -- which disabled the whole multi-cell path for exactly
                       the wide cuts it exists for. */
                    if (b'a'..=b'i').contains(&byte) {
                        CoverDagNodeOp::Input((byte - b'a') as usize)
                    } else {
                        CoverDagNodeOp::Gate(symbol.to_string())
                    }
                }
                ("~", [_]) => CoverDagNodeOp::Not,
                ("&", [_, _]) => CoverDagNodeOp::And,
                (symbol, _) => CoverDagNodeOp::Gate(symbol.to_string()),
            };
            let children = enode
                .children
                .iter()
                .copied()
                .map(usize::from)
                .collect::<Vec<_>>();
            let node_id = nodes.len();
            nodes.push(CoverDagNode {
                op,
                children: children.clone(),
            });
            class_nodes.entry(class_id).or_default().push(node_id);
            for child in children {
                if reachable.contains(&child) {
                    parents.entry(child).or_default().push(class_id);
                }
            }
        }
    }

    CoverDag {
        root_class,
        class_nodes,
        nodes,
        parents,
    }
}

/// Per-pin native pos/neg for a lib gate: cut leaves use `cut_native`; internal e-class roots
/// default to positive phase only (matches unmapped behaviour).
fn native_pin_phases_for_fanins(
    fanins: &[usize],
    best_map: &HashMap<usize, CoverChoice>,
    cut_native: Option<(&[bool], &[bool])>,
) -> Option<(Vec<bool>, Vec<bool>)> {
    let (np, nn) = cut_native?;
    let mut pos = Vec::with_capacity(fanins.len());
    let mut neg = Vec::with_capacity(fanins.len());
    for &cid in fanins {
        match best_map.get(&cid).map(|c| &c.kind) {
            Some(CoverChoiceKind::Input(li)) if (*li as usize) < np.len() && (*li as usize) < nn.len() => {
                pos.push(np[*li]);
                neg.push(nn[*li]);
            }
            _ => {
                pos.push(true);
                neg.push(false);
            }
        }
    }
    Some((pos, neg))
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

/// Same `(delay, area)` under the current objective (within epsilon).
fn choice_optimal_tie(a: &CoverChoice, b: &CoverChoice, objective: MappingObjective) -> bool {
    const EPS: f32 = 1e-3;
    match objective {
        MappingObjective::DelayFirst => {
            (a.delay - b.delay).abs() <= EPS && (a.area - b.area).abs() <= EPS
        }
        MappingObjective::AreaFirst => {
            (a.area - b.area).abs() <= EPS && (a.delay - b.delay).abs() <= EPS
        }
    }
}

fn cover_choice_kind_eq(a: &CoverChoiceKind, b: &CoverChoiceKind) -> bool {
    match (a, b) {
        (CoverChoiceKind::Const(x), CoverChoiceKind::Const(y)) => x == y,
        (CoverChoiceKind::Input(i), CoverChoiceKind::Input(j)) => i == j,
        (
            CoverChoiceKind::Gate { cell: ca, fanins: fa },
            CoverChoiceKind::Gate { cell: cb, fanins: fb },
        ) => {
            ca.cell_name == cb.cell_name
                && ca.neg_mask == cb.neg_mask
                && ca.out_neg == cb.out_neg
                && fa == fb
        }
        _ => false,
    }
}

/// Evaluate one DAG `node_id` in e-class `key` using child choices from `best_map` (same logic as
/// the inner loop of cover DP).
fn cover_choice_for_node_id(
    node_id: usize,
    key: usize,
    cover_dag: &CoverDag,
    tech: &TechLib,
    gate_cells: &HashMap<String, Cell>,
    input_arrivals: &[f32],
    input_phase_arrivals: Option<&[PhaseArrival]>,
    objective: MappingObjective,
    ps: &EmapParams,
    best_map: &HashMap<usize, CoverChoice>,
    cut_native_phases: Option<(&[bool], &[bool])>,
    required_output_delay: Option<f32>,
) -> Option<CoverChoice> {
    let node = cover_dag.nodes.get(node_id)?;
    let cand = match &node.op {
        CoverDagNodeOp::Const(false) => {
            let phase = PhaseArrival {
                pos: 0.0,
                neg: tech.inv_delay,
            };
            Some(CoverChoice {
                delay: phase.pos,
                phase,
                area: 0.0,
                kind: CoverChoiceKind::Const(false),
                class_areas: HashMap::new(),
            })
        }
        CoverDagNodeOp::Const(true) => {
            let phase = PhaseArrival {
                pos: 0.0,
                neg: tech.inv_delay,
            };
            Some(CoverChoice {
                delay: phase.pos,
                phase,
                area: 0.0,
                kind: CoverChoiceKind::Const(true),
                class_areas: HashMap::new(),
            })
        }
        CoverDagNodeOp::Input(li) => input_phase_arrival_for_leaf(
            *li,
            input_phase_arrivals,
            input_arrivals,
            tech,
        )
        .map(|phase| CoverChoice {
            delay: phase.pos,
            phase,
            area: 0.0,
            kind: CoverChoiceKind::Input(*li),
            class_areas: HashMap::new(),
        }),
        // A raw AIG node is a legal cover element only if the library really has the matching
        // cell: an inverter for `~` and an exact 2-input AND for `&`. When present, its cost IS
        // that cell's library area/delay (or unit delay in the depth objective); when absent,
        // `exact_*_variant()` is `None`, the arm yields no choice, and the node is effectively
        // infinite cost -- so extraction can never emit an AIG node that has no gate-level
        // implementation and every returned cover is directly materializable.
        CoverDagNodeOp::Not => (|| {
            let child_id = *node.children.first()?;
            let child = best_map.get(&child_id)?;
            if child.class_areas.contains_key(&key) {
                return None;
            }
            let cell = tech.exact_inv_variant()?;
            gate_choice_for_variant(
                key,
                cell,
                &[child_id],
                std::slice::from_ref(&child.phase),
                child.class_areas.clone(),
                tech,
                best_map,
                cut_native_phases,
            )
        })(),
        CoverDagNodeOp::And => (|| {
            let child_ids = node.children.as_slice();
            let (child_phases, merged) = merge_child_class_areas(key, child_ids, best_map)?;
            let cell = tech.exact_and2_variant()?;
            gate_choice_for_variant(
                key,
                cell,
                child_ids,
                &child_phases,
                merged,
                tech,
                best_map,
                cut_native_phases,
            )
        })(),
        CoverDagNodeOp::Gate(symbol) => (|| {
            let cell = gate_cells.get(symbol.as_str())?;
            let fanins = node.children.clone();
            let (child_phases, merged) = merge_child_class_areas(key, &fanins, best_map)?;
            gate_choice_for_cell(
                key,
                cell,
                &fanins,
                &child_phases,
                merged,
                tech,
                objective,
                best_map,
                cut_native_phases,
                ps.mapping_area_cost,
            )
        })(),
    };
    cand.filter(|c| required_output_delay.map_or(true, |r| c.delay <= r + REQ_TIMING_EPS))
}

/// Materialize a cover from the DP's per-class choices, using `root_choice` for the root class and
/// the DP winners for every other class.
///
/// `root_choice` is passed explicitly instead of being written into a cloned `best` map: the
/// multi-solution path asks for up to 16 root alternatives per cut, and cloning the whole DP map
/// (hundreds of `CoverChoice`, each with its own `HashMap`) for every alternative dominated the
/// runtime of `EPMAPPING_EGG_MULTI`.
fn materialize_mapped_cover_from_best(
    root_key: usize,
    best: &HashMap<usize, CoverChoice>,
    root_choice: &CoverChoice,
    tech: &TechLib,
    input_arrivals_len: usize,
) -> Result<MappedCoverChoice, String> {
    let mut graph = MappedCoverGraph {
        nodes: Vec::new(),
        root: 0,
    };
    let mut input_cache = HashMap::<usize, usize>::new();
    let mut const_cache = HashMap::<bool, usize>::new();
    let mut memo = HashMap::<usize, usize>::new();

    #[allow(clippy::too_many_arguments)]
    fn materialize(
        sg_idx: usize,
        best: &HashMap<usize, CoverChoice>,
        root_key: usize,
        root_choice: &CoverChoice,
        graph: &mut MappedCoverGraph,
        input_cache: &mut HashMap<usize, usize>,
        const_cache: &mut HashMap<bool, usize>,
        memo: &mut HashMap<usize, usize>,
    ) -> Option<usize> {
        if let Some(&id) = memo.get(&sg_idx) {
            return Some(id);
        }
        /* The root uses the caller's alternative; every other class uses the DP winner. A
           self-referential root therefore keeps the alternative in its own subtree, exactly as the
           previous `best_alt[root] = alternative` clone did. */
        let choice = if sg_idx == root_key {
            root_choice
        } else {
            best.get(&sg_idx)?
        };
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
                    .map(|fanin| {
                        materialize(
                            *fanin,
                            best,
                            root_key,
                            root_choice,
                            graph,
                            input_cache,
                            const_cache,
                            memo,
                        )
                    })
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
        root_key,
        best,
        root_key,
        root_choice,
        &mut graph,
        &mut input_cache,
        &mut const_cache,
        &mut memo,
    )
    .ok_or_else(|| "materialize_root_failed".to_string())?;

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

    let pin_to_root_delay = compute_pin_to_root_delays(&graph, tech, input_arrivals_len);

    Ok(MappedCoverChoice {
        graph,
        delay: root_choice.delay,
        area: root_choice.area,
        used_leaf_indices,
        pin_to_root_delay,
    })
}

fn gate_choice_for_variant(
    self_id: usize,
    cell: &CellVariant,
    fanins: &[usize],
    child_phases: &[PhaseArrival],
    merged_child_areas: HashMap<usize, f32>,
    tech: &TechLib,
    best_map: &HashMap<usize, CoverChoice>,
    cut_native: Option<(&[bool], &[bool])>,
) -> Option<CoverChoice> {
    if fanins.len() != cell.n_inputs as usize || child_phases.len() != fanins.len() {
        return None;
    }
    let mut t = 0.0f32;
    for (i, arrival) in child_phases.iter().enumerate() {
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
    let phase = if cell.out_neg {
        PhaseArrival {
            pos: t + tech.inv_delay,
            neg: t,
        }
    } else {
        PhaseArrival {
            pos: t,
            neg: t + tech.inv_delay,
        }
    };
    let native_pins = native_pin_phases_for_fanins(fanins, best_map, cut_native);
    let gate_area = tech.variant_area_physical_invs(
        cell,
        fanins.len(),
        native_pins
            .as_ref()
            .map(|(a, b)| (a.as_slice(), b.as_slice())),
    );
    let mut class_areas = merged_child_areas;
    class_areas.insert(self_id, gate_area);
    let area = class_areas.values().sum::<f32>();
    Some(CoverChoice {
        delay: phase.pos,
        phase,
        area,
        kind: CoverChoiceKind::Gate {
            cell: cell.clone(),
            fanins: fanins.to_vec(),
        },
        class_areas,
    })
}

fn gate_choice_for_cell(
    self_id: usize,
    cell: &Cell,
    fanins: &[usize],
    child_phases: &[PhaseArrival],
    merged_child_areas: HashMap<usize, f32>,
    tech: &TechLib,
    objective: MappingObjective,
    best_map: &HashMap<usize, CoverChoice>,
    cut_native: Option<(&[bool], &[bool])>,
    mapping_area_cost: MappingAreaCost,
) -> Option<CoverChoice> {
    let native_for_pins = native_pin_phases_for_fanins(fanins, best_map, cut_native);
    let native_ref = native_for_pins
        .as_ref()
        .map(|(a, b)| (a.as_slice(), b.as_slice()));
    let (_, _, _, variant) = tech.best_for_cell_phase_arrivals(
        cell,
        child_phases,
        objective,
        native_ref,
        None,
        mapping_area_cost,
    )?;
    gate_choice_for_variant(
        self_id,
        variant,
        fanins,
        child_phases,
        merged_child_areas,
        tech,
        best_map,
        cut_native,
    )
}

/// Full list of materialized covers (first is the DP winner; further entries are root e-class ties).
fn extract_mapped_cover_covers_vec(
    expr: &str,
    tech: &TechLib,
    input_arrivals: &[f32],
    objective: MappingObjective,
    ps: &EmapParams,
    egg_bundle: Option<&EggRewriteBundle>,
    required_output_delay: Option<f32>,
    input_phase_arrivals: Option<&[PhaseArrival]>,
    cut_native_phases: Option<(&[bool], &[bool])>,
) -> Result<Vec<MappedCoverChoice>, String> {
    let rec: egg::RecExpr<AIG> = expr.parse().map_err(|_| "expr_parse_failed".to_string())?;
    if rec.as_ref().len() > ps.max_cover_nodes {
        return Err(format!(
            "expr_too_large(len={},max={})",
            rec.as_ref().len(),
            ps.max_cover_nodes
        ));
    }
    let (egraph, root, gate_cells) = rewrite_to_gategraph_with_limits(
        expr,
        tech,
        input_arrivals.len(),
        egg_bundle,
        ps.egg_iter_limit,
        ps.egg_node_limit,
    )
    .ok_or_else(|| "rewrite_to_gategraph_failed".to_string())?;
    let cover_dag = build_cover_dag(&egraph, root);
    let mut best = HashMap::<usize, CoverChoice>::new();
    let root_key = cover_dag.root_class;

    let compute_best = |key: usize, best_map: &HashMap<usize, CoverChoice>| {
        let mut choice = None::<CoverChoice>;
        for &node_id in cover_dag.class_nodes.get(&key).into_iter().flatten() {
            let cand = cover_choice_for_node_id(
                node_id,
                key,
                &cover_dag,
                tech,
                gate_cells.as_ref(),
                input_arrivals,
                input_phase_arrivals,
                objective,
                ps,
                best_map,
                cut_native_phases,
                required_output_delay,
            );

            choice = match (choice, cand) {
                (Some(cur), Some(newc)) if better_choice(&newc, &cur, objective).is_lt() => {
                    Some(newc)
                }
                (None, Some(newc)) => Some(newc),
                (cur, _) => cur,
            };
        }
        choice
    };

    let mut q = UniqueQueue::<usize>::default();
    /* Deterministic worklist order (see `build_cover_dag`): the fixpoint keeps the first choice it
       sees on a cost tie, so both the seed order and the re-queued parent order must be sorted. */
    let mut seed: Vec<usize> = cover_dag.class_nodes.keys().copied().collect();
    seed.sort_unstable();
    q.extend(seed);

    while let Some(key) = q.pop() {
        let new_choice = compute_best(key, &best);
        let changed = match (best.get(&key), new_choice.as_ref()) {
            (None, None) => false,
            (None, Some(_)) => true,
            (Some(_), None) => false,
            (Some(oldc), Some(newc)) => better_choice(newc, oldc, objective).is_lt(),
        };
        if changed {
            if let Some(new_choice) = new_choice {
                best.insert(key, new_choice);
            }
            if let Some(ps) = cover_dag.parents.get(&key) {
                let mut ps = ps.clone();
                ps.sort_unstable();
                q.extend(ps);
            }
        }
    }

    let root_choice = best.get(&root_key).cloned().ok_or_else(|| {
        let mut input_nodes = 0usize;
        let mut const_nodes = 0usize;
        let mut not_nodes = 0usize;
        let mut and_nodes = 0usize;
        let mut gate_nodes = 0usize;
        for node in &cover_dag.nodes {
            match node.op {
                CoverDagNodeOp::Const(_) => const_nodes += 1,
                CoverDagNodeOp::Input(_) => input_nodes += 1,
                CoverDagNodeOp::Not => not_nodes += 1,
                CoverDagNodeOp::And => and_nodes += 1,
                CoverDagNodeOp::Gate(_) => gate_nodes += 1,
            }
        }
        let mut root_kind_counts = HashMap::<String, usize>::new();
        for &node_id in cover_dag
            .class_nodes
            .get(&root_key)
            .into_iter()
            .flatten()
        {
            let label = match &cover_dag.nodes[node_id].op {
                CoverDagNodeOp::Const(false) => "const0".to_string(),
                CoverDagNodeOp::Const(true) => "const1".to_string(),
                CoverDagNodeOp::Input(_) => "input".to_string(),
                CoverDagNodeOp::Not => "not".to_string(),
                CoverDagNodeOp::And => "and".to_string(),
                CoverDagNodeOp::Gate(_) => "gate".to_string(),
            };
            *root_kind_counts.entry(label).or_insert(0) += 1;
        }
        let mut root_kinds = root_kind_counts.into_iter().collect::<Vec<_>>();
        root_kinds.sort_by(|a, b| a.0.cmp(&b.0));
        let empty_best = HashMap::<usize, CoverChoice>::new();
        let seedable_classes = cover_dag
            .class_nodes
            .keys()
            .copied()
            .filter(|&class_id| compute_best(class_id, &empty_best).is_some())
            .count();
        format!(
            "root_choice_missing best_classes={} total_classes={} seedable_classes={} nodes[input={},const={},not={},and={},gate={}] root_kinds={:?}",
            best.len(),
            cover_dag.class_nodes.len(),
            seedable_classes,
            input_nodes,
            const_nodes,
            not_nodes,
            and_nodes,
            gate_nodes,
            root_kinds
        )
    })?;
    if let Some(r) = required_output_delay {
        if root_choice.delay > r + REQ_TIMING_EPS {
            return Err("root_required_violation".to_string());
        }
    }

    let max_alt = ps
        .max_egg_cover_alternatives
        .max(1)
        .min(64);
    let reference = root_choice.clone();
    let mut covers: Vec<MappedCoverChoice> = Vec::new();
    covers.push(materialize_mapped_cover_from_best(
        root_key,
        &best,
        &reference,
        tech,
        input_arrivals.len(),
    )?);

    if max_alt > 1 {
        let mut node_ids: Vec<usize> = cover_dag
            .class_nodes
            .get(&root_key)
            .into_iter()
            .flatten()
            .copied()
            .collect();
        node_ids.sort_unstable();
        let mut seen_kinds = vec![reference.kind.clone()];
        for &node_id in &node_ids {
            let Some(c) = cover_choice_for_node_id(
                node_id,
                root_key,
                &cover_dag,
                tech,
                gate_cells.as_ref(),
                input_arrivals,
                input_phase_arrivals,
                objective,
                ps,
                &best,
                cut_native_phases,
                required_output_delay,
            ) else {
                continue;
            };
            if !choice_optimal_tie(&c, &reference, objective) {
                continue;
            }
            if seen_kinds
                .iter()
                .any(|k| cover_choice_kind_eq(k, &c.kind))
            {
                continue;
            }
            seen_kinds.push(c.kind.clone());
            covers.push(materialize_mapped_cover_from_best(
                root_key,
                &best,
                &c,
                tech,
                input_arrivals.len(),
            )?);
            if covers.len() >= max_alt {
                break;
            }
        }
    }

    if covers.is_empty() {
        return Err("empty_cover".to_string());
    }
    Ok(covers)
}

pub(crate) fn extract_mapped_cover_from_expr(
    expr: &str,
    tech: &TechLib,
    input_arrivals: &[f32],
    objective: MappingObjective,
    ps: &EmapParams,
    egg_bundle: Option<&EggRewriteBundle>,
    required_output_delay: Option<f32>,
    input_phase_arrivals: Option<&[PhaseArrival]>,
    cut_native_phases: Option<(&[bool], &[bool])>,
) -> Result<MappedCoverChoice, String> {
    extract_mapped_cover_covers_vec(
        expr,
        tech,
        input_arrivals,
        objective,
        ps,
        egg_bundle,
        required_output_delay,
        input_phase_arrivals,
        cut_native_phases,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| "empty_cover".to_string())
}

/// Extract up to `max_alternatives` distinct root enode covers that tie on `(delay, area)` for the
/// current objective (same child choices from DP; only the root e-class implementation differs).
pub fn extract_mapped_cover_alternatives(
    expr: &str,
    tech: &TechLib,
    input_arrivals: &[f32],
    objective: MappingObjective,
    ps: &EmapParams,
    egg_bundle: Option<&EggRewriteBundle>,
    required_output_delay: Option<f32>,
    input_phase_arrivals: Option<&[PhaseArrival]>,
    cut_native_phases: Option<(&[bool], &[bool])>,
    max_alternatives: usize,
) -> Result<Vec<MappedCoverChoice>, String> {
    let mut p = ps.clone();
    p.max_egg_cover_alternatives = max_alternatives.max(1).min(64);
    extract_mapped_cover_covers_vec(
        expr,
        tech,
        input_arrivals,
        objective,
        &p,
        egg_bundle,
        required_output_delay,
        input_phase_arrivals,
        cut_native_phases,
    )
}

pub fn build_mapped_cover_binding(
    cut_output_index: usize,
    cut: &crate::emap_cut::Cut,
    leaves: &[usize],
    cover: &MappedCoverChoice,
    tech: &TechLib,
    wire_delay: f32,
) -> crate::binding::NodeBinding {
    /* Kept **indexed by cut variable** (dense) to match a cover graph's `MappedCoverNode::Input(li)`,
       which carries a variable index; see the emapper copy of this function for the c7552 abort this
       used to cause. */
    let pin_to_root_delay =
        if cover.pin_to_root_delay.len() == leaves.len() && !cover.pin_to_root_delay.is_empty() {
            cover.pin_to_root_delay.clone()
        } else {
            compute_pin_to_root_delays(&cover.graph, tech, leaves.len())
        };
    crate::binding::NodeBinding {
        node: cut_output_index,
        cut: cut.clone(),
        leaves: leaves.to_vec(),
        kind: NodeBindingKind::MappedCover {
            graph: cover.graph.clone(),
            pin_to_root_delay,
        },
        phase_neg: false,
        native_delay: cover.delay + wire_delay,
        delay: cover.delay + wire_delay,
        area: cover.area,
        mapping_cost_area: cover.area,
    }
}

/// Expand a [`MappedCoverGraph`] node to an AIG s-expression string.
///
/// Each `Cell` node is expanded to an AIG using the cell's `Expr` function,
/// with the variant's `neg_mask` and `out_neg` applied to its fanin connections.
/// Returns `None` if the underlying `TechLib` does not contain the referenced cell.
fn materialized_sexp(
    node_idx: usize,
    graph: &MappedCoverGraph,
    tech: &crate::techlib::TechLib,
    memo: &mut HashMap<usize, String>,
) -> Option<String> {
    if let Some(s) = memo.get(&node_idx) {
        return Some(s.clone());
    }
    let s = match &graph.nodes[node_idx] {
        MappedCoverNode::Const(false) => "0".to_string(),
        MappedCoverNode::Const(true) => "1".to_string(),
        MappedCoverNode::Input(li) => ((b'a' + *li as u8) as char).to_string(),
        MappedCoverNode::Cell { cell, fanins } => {
            // Resolve fanin AIG sexps first.
            let mut fanin_sexps: Vec<String> = Vec::with_capacity(fanins.len());
            for &fi in fanins {
                fanin_sexps.push(materialized_sexp(fi, graph, tech, memo)?);
            }
            // Apply neg_mask: each fanin[i] is optionally inverted before connecting.
            let mut fanin_sexps_neg: Vec<String> = Vec::with_capacity(fanins.len());
            for (i, s) in fanin_sexps.iter().enumerate() {
                if ((cell.neg_mask >> i) & 1) == 1 {
                    fanin_sexps_neg.push(format!("(~ {})", s));
                } else {
                    fanin_sexps_neg.push(s.clone());
                }
            }
            // Look up the base Cell to get its Expr and variable order.
            let base_cell = tech.cells.iter().find(|c| c.name == cell.cell_name)?;
            // Build a mapping from cell variable name -> fanin AIG sexp, using
            // pin_names to identify which variant input connects to which cell variable.
            let mut var_to_sexp: HashMap<&str, &str> = HashMap::new();
            for (i, pin_name) in cell.pin_names.iter().enumerate() {
                if let Some(sexp) = fanin_sexps_neg.get(i) {
                    var_to_sexp.insert(pin_name.as_str(), sexp.as_str());
                }
            }
            let expanded = expand_expr(&base_cell.expr, &var_to_sexp)?;
            if cell.out_neg {
                format!("(~ {})", expanded)
            } else {
                expanded
            }
        }
    };
    memo.insert(node_idx, s.clone());
    Some(s)
}

/// Lower an [`Expr`] to an AIG s-expression by substituting variables.
fn expand_expr(expr: &crate::techlib::Expr, vars: &HashMap<&str, &str>) -> Option<String> {
    match expr {
        crate::techlib::Expr::Const(false) => Some("0".to_string()),
        crate::techlib::Expr::Const(true) => Some("1".to_string()),
        crate::techlib::Expr::Var(v) => vars.get(v.as_str()).map(|s| (*s).to_string()),
        crate::techlib::Expr::Not(a) => Some(format!("(~ {})", expand_expr(a, vars)?)),
        crate::techlib::Expr::And(a, b) => Some(format!(
            "(& {} {})",
            expand_expr(a, vars)?,
            expand_expr(b, vars)?
        )),
        crate::techlib::Expr::Or(a, b) => {
            let la = expand_expr(a, vars)?;
            let rb = expand_expr(b, vars)?;
            Some(format!("(~ (& (~ {la}) (~ {rb})))"))
        }
    }
}

/// Map a cut expression to an AIG s-expression using technology-library-aware cell cover.
///
/// - `expr`: the input AIG s-expression (variables `a`..`f`).
/// - `tech`: the loaded [`TechLib`].
/// - `input_arrivals`: arrival times (or logical depths) for each input, length = number of inputs.
/// - `objective`: optimisation goal.
/// - `required_output_delay`: if `Some(r)` and `r` is finite, discard covers with root delay above `r`.
///
/// Returns the full [`MappedCoverChoice`] on success, containing the cell-level graph,
/// real techlib delay/area, and which input indices were actually used.
/// Cut cones passed from C++ `build_cut_expr` can be large AIGs; the in-mapper default (64) and
/// the previous hard-coded 256 were far too small and made every `map_egg` extraction fail.
const MAP_EGG_CUT_MAX_NODES: usize = 65_536;

/// [`EmapParams`] for standalone cut mapping, with small equality-saturation limits.
///
/// A cut AIG has only a handful of nodes, so the mapper-wide defaults (30 iterations / 10k nodes)
/// spend seconds per cut without changing the extracted cover. The limits stay overridable with
/// `MAP_EGG_EGG_ITER_LIMIT` / `MAP_EGG_EGG_NODE_LIMIT`.
fn cut_mapping_params() -> EmapParams {
    fn usize_env(key: &str, default: usize) -> usize {
        std::env::var(key)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }
    EmapParams {
        max_cover_nodes: MAP_EGG_CUT_MAX_NODES,
        egg_iter_limit: usize_env("MAP_EGG_EGG_ITER_LIMIT", 2),
        egg_node_limit: usize_env("MAP_EGG_EGG_NODE_LIMIT", 500),
        ..Default::default()
    }
}

pub fn map_cut_to_cover(
    expr: &str,
    tech: &crate::techlib::TechLib,
    input_arrivals: &[f32],
    objective: crate::techlib::MappingObjective,
    required_output_delay: Option<f32>,
) -> Option<MappedCoverChoice> {
    let ps = cut_mapping_params();
    let egg_bundle = tech.egg_rewrite_bundle();
    extract_mapped_cover_from_expr(
        expr,
        tech,
        input_arrivals,
        objective,
        &ps,
        Some(egg_bundle.as_ref()),
        required_output_delay,
        None,
        None,
    )
    .ok()
}

pub fn map_cut_to_cover_with_reason(
    expr: &str,
    tech: &crate::techlib::TechLib,
    input_arrivals: &[f32],
    objective: crate::techlib::MappingObjective,
    required_output_delay: Option<f32>,
) -> Result<MappedCoverChoice, String> {
    let ps = cut_mapping_params();
    let egg_bundle = tech.egg_rewrite_bundle();
    extract_mapped_cover_from_expr(
        expr,
        tech,
        input_arrivals,
        objective,
        &ps,
        Some(egg_bundle.as_ref()),
        required_output_delay,
        None,
        None,
    )
}

/// Like [`map_cut_to_cover_with_reason`] but saturates with the **cover-only** (`AIG -> gate`)
/// rule set.
///
/// Use this when `expr` is already an AIG (the `map_egg` large-cut path: the gate-level subgraph
/// is expanded to AIG before rewriting). The one-way rule set reaches every cell cover without
/// the gate -> AIG expansion that makes full bidirectional saturation blow up.
/// Egg rewrite bundle for a cut-level saturation: the cover-only one by default, or the full
/// bidirectional one when `MAP_EGG_FULL_BUNDLE=1`.
///
/// The cover-only bundle rewrites `AIG -> cell` only, which is enough to reach every cell cover and
/// keeps the e-graph small. The full bundle adds `cell -> AIG` expansion, so saturation can take a
/// cell apart again and re-cover it differently — strictly more search (better covers on some cuts,
/// more e-graph nodes and time on all of them). It is opt-in for exactly that reason.
fn cut_egg_bundle(tech: &TechLib) -> &std::sync::Arc<EggRewriteBundle> {
    static FULL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let full = *FULL.get_or_init(|| {
        std::env::var("MAP_EGG_FULL_BUNDLE").is_ok_and(|v| v == "1")
    });
    if full {
        tech.egg_rewrite_bundle()
    } else {
        tech.egg_cover_bundle()
    }
}

pub fn map_cut_to_cover_cover_only(
    expr: &str,
    tech: &crate::techlib::TechLib,
    input_arrivals: &[f32],
    objective: crate::techlib::MappingObjective,
    required_output_delay: Option<f32>,
) -> Result<MappedCoverChoice, String> {
    let ps = cut_mapping_params();
    let egg_bundle = cut_egg_bundle(tech);
    extract_mapped_cover_from_expr(
        expr,
        tech,
        input_arrivals,
        objective,
        &ps,
        Some(egg_bundle.as_ref()),
        required_output_delay,
        None,
        None,
    )
}

/// Egg cover extraction returning up to `max_alternatives` distinct root implementations.
///
/// Same as [`map_cut_to_cover_cover_only`] but returns several covers of the cut function instead of
/// one, so the caller can pick among them (e.g. a unit-depth/area Pareto selection).
pub fn map_cut_to_cover_alternatives(
    expr: &str,
    tech: &crate::techlib::TechLib,
    input_arrivals: &[f32],
    objective: crate::techlib::MappingObjective,
    required_output_delay: Option<f32>,
    max_alternatives: usize,
) -> Result<Vec<MappedCoverChoice>, String> {
    let ps = cut_mapping_params();
    let egg_bundle = cut_egg_bundle(tech);
    extract_mapped_cover_alternatives(
        expr,
        tech,
        input_arrivals,
        objective,
        &ps,
        Some(egg_bundle.as_ref()),
        required_output_delay,
        None,
        None,
        max_alternatives,
    )
}

/// Map a small cut (typically `<= 4` inputs) directly against the technology library.
///
/// This is the "cut mapping library" fast path: the cut's truth table is computed from the AIG
/// s-expression and the best single standard-cell variant is selected via
/// [`TechLib::best_for_tt_phase_arrivals`]. No e-graph saturation is performed, so this is cheap
/// and is meant for small cuts where one library cell usually implements the function.
///
/// Returns `Err("no_library_match")` when no library variant realizes the cut function.
pub fn map_cut_to_library(
    expr: &str,
    tech: &crate::techlib::TechLib,
    input_arrivals: &[f32],
    objective: crate::techlib::MappingObjective,
    required_output_delay: Option<f32>,
) -> Result<MappedCoverChoice, String> {
    let (n_inputs, tt) = sexp_to_tt9(expr).ok_or_else(|| "aig_sexp_parse_failed".to_string())?;
    map_tt_to_library(tt, n_inputs, tech, input_arrivals, objective, required_output_delay)
}

/// Map a **truth table** directly against the technology library.
///
/// Same as [`map_cut_to_library`] but keyed by the truth table instead of an AIG s-expression. This
/// is the entry point for don't-care aware matching: the caller enumerates the completions of an
/// incompletely specified cut function and matches each one with this function, keeping the best.
pub fn map_tt_to_library(
    tt: Tt9,
    n_inputs: u8,
    tech: &crate::techlib::TechLib,
    input_arrivals: &[f32],
    objective: crate::techlib::MappingObjective,
    required_output_delay: Option<f32>,
) -> Result<MappedCoverChoice, String> {
    if n_inputs == 0 {
        let value = crate::tt9::tt9_bit_get(&tt, 0);
        return Ok(MappedCoverChoice {
            graph: MappedCoverGraph {
                nodes: vec![MappedCoverNode::Const(value)],
                root: 0,
            },
            delay: 0.0,
            area: 0.0,
            used_leaf_indices: Vec::new(),
            pin_to_root_delay: Vec::new(),
        });
    }
    let n = n_inputs as usize;
    if input_arrivals.len() < n {
        return Err("arrivals_len_mismatch".to_string());
    }
    let arrivals: Vec<PhaseArrival> = input_arrivals
        .iter()
        .take(n)
        .map(|&a| PhaseArrival {
            pos: a,
            neg: a + tech.inv_delay,
        })
        .collect();
    let (delay, _mapping_area, physical_area, variant) = tech
        .best_for_tt_phase_arrivals(
            n_inputs,
            tt,
            &arrivals,
            objective,
            None,
            None,
            MappingAreaCost::Physical,
        )
        .ok_or_else(|| "no_library_match".to_string())?;
    if let Some(r) = required_output_delay {
        if delay > r + REQ_TIMING_EPS {
            return Err("root_required_violation".to_string());
        }
    }
    let mut nodes = Vec::with_capacity(n + 1);
    for i in 0..n {
        nodes.push(MappedCoverNode::Input(i));
    }
    nodes.push(MappedCoverNode::Cell {
        cell: variant.clone(),
        fanins: (0..n).collect(),
    });
    let graph = MappedCoverGraph { nodes, root: n };
    let pin_to_root_delay = compute_pin_to_root_delays(&graph, tech, n);
    Ok(MappedCoverChoice {
        graph,
        delay,
        area: physical_area,
        used_leaf_indices: (0..n).collect(),
        pin_to_root_delay,
    })
}

/// - `input_arrivals`: arrival times (or logical depths) for each input, length = number of inputs.
/// - `objective`: optimisation goal.
///
/// Returns `(aig_sexp, delay, area)` on success.
pub fn map_cut_to_aig_sexp(
    expr: &str,
    tech: &crate::techlib::TechLib,
    input_arrivals: &[f32],
    objective: crate::techlib::MappingObjective,
) -> Option<(String, f32, f32)> {
    let ps = cut_mapping_params();
    let cover =
        extract_mapped_cover_from_expr(
            expr,
            tech,
            input_arrivals,
            objective,
            &ps,
            None,
            None,
            None,
            None,
        )
        .ok()?;
    let mut memo = HashMap::<usize, String>::new();
    let sexp = materialized_sexp(cover.graph.root, &cover.graph, tech, &mut memo)?;
    Some((sexp, cover.delay, cover.area))
}
