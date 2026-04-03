use std::collections::{HashMap, HashSet};

use crate::{
    emap::{EmapParams, NodeBindingKind},
    emap_egg::{rewrite_to_gategraph, EggRewriteBundle, GateEGraph, AIG},
    techlib::{Cell, CellVariant, MappingObjective, TechLib},
};

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
pub struct MappedCoverChoice {
    pub graph: MappedCoverGraph,
    pub delay: f32,
    pub area: f32,
    pub used_leaf_indices: Vec<usize>,
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
    delay: f32,
    /// Sum of gate areas over **distinct** e-class ids in this cover (DAG-accurate).
    area: f32,
    kind: CoverChoiceKind,
    /// Per e-class (`egg::Id`) gate area for the chosen enode; used to merge without
    /// double-counting shared subgraphs (multi-fanout), cf. `UniqueNodeExtractor` in mig_egg.
    class_areas: HashMap<usize, f32>,
}

fn union_class_areas(lhs: &HashMap<usize, f32>, rhs: &HashMap<usize, f32>) -> HashMap<usize, f32> {
    let mut out = lhs.clone();
    for (k, v) in rhs {
        match out.get(k) {
            None => {
                out.insert(*k, *v);
            }
            Some(ev) => {
                debug_assert!(
                    (*ev - *v).abs() <= 1e-3_f32.max(1e-4 * ev.max(*v)),
                    "class_areas mismatch for e-class {k}: {ev} vs {v}"
                );
            }
        }
    }
    out
}

fn union_many_class_areas(maps: &[HashMap<usize, f32>]) -> HashMap<usize, f32> {
    match maps.len() {
        0 => HashMap::new(),
        1 => maps[0].clone(),
        _ => maps
            .iter()
            .skip(1)
            .fold(maps[0].clone(), |acc, m| union_class_areas(&acc, m)),
    }
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
    self_id: usize,
    cell: &CellVariant,
    fanins: &[usize],
    child_delays: &[f32],
    merged_child_areas: HashMap<usize, f32>,
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
    let gate_area = cell.area + invs * tech.inv_area;
    let mut class_areas = merged_child_areas;
    class_areas.insert(self_id, gate_area);
    let area = class_areas.values().sum::<f32>();
    Some(CoverChoice {
        delay,
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
    child_delays: &[f32],
    merged_child_areas: HashMap<usize, f32>,
    tech: &TechLib,
    objective: MappingObjective,
) -> Option<CoverChoice> {
    let (_, _, variant) = tech.best_for_cell(cell, child_delays, objective)?;
    gate_choice_for_variant(
        self_id,
        variant,
        fanins,
        child_delays,
        merged_child_areas,
        tech,
    )
}

pub(crate) fn extract_mapped_cover_from_expr(
    expr: &str,
    tech: &TechLib,
    input_arrivals: &[f32],
    objective: MappingObjective,
    ps: &EmapParams,
    egg_bundle: Option<&EggRewriteBundle>,
) -> Option<MappedCoverChoice> {
    let rec: egg::RecExpr<AIG> = expr.parse().ok()?;
    if rec.as_ref().len() > ps.max_cover_nodes {
        return None;
    }
    let (egraph, root, gate_cells) =
        rewrite_to_gategraph(expr, tech, input_arrivals.len(), egg_bundle)?;
    let mut best = HashMap::<usize, Option<CoverChoice>>::new();

    fn best_eclass(
        id: egg::Id,
        egraph: &GateEGraph,
        gate_cells: &HashMap<String, Cell>,
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
                    class_areas: HashMap::new(),
                }),
                ("1", []) => Some(CoverChoice {
                    delay: 0.0,
                    area: 0.0,
                    kind: CoverChoiceKind::Const(true),
                    class_areas: HashMap::new(),
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
                            class_areas: HashMap::new(),
                        })
                    }
                }
                ("~", [a]) => (|| {
                    let child = best_eclass(
                        *a,
                        egraph,
                        gate_cells,
                        input_arrivals,
                        tech,
                        objective,
                        best,
                        visiting,
                    )?;
                    let cell = tech.exact_inv_variant()?;
                    gate_choice_for_variant(
                        key,
                        cell,
                        &[usize::from(*a)],
                        &[child.delay],
                        child.class_areas.clone(),
                        tech,
                    )
                })(),
                ("&", [a, b]) => (|| {
                    let lhs = best_eclass(
                        *a,
                        egraph,
                        gate_cells,
                        input_arrivals,
                        tech,
                        objective,
                        best,
                        visiting,
                    )?;
                    let rhs = best_eclass(
                        *b,
                        egraph,
                        gate_cells,
                        input_arrivals,
                        tech,
                        objective,
                        best,
                        visiting,
                    )?;
                    let merged = union_class_areas(&lhs.class_areas, &rhs.class_areas);
                    let cell = tech.exact_and2_variant()?;
                    gate_choice_for_variant(
                        key,
                        cell,
                        &[usize::from(*a), usize::from(*b)],
                        &[lhs.delay, rhs.delay],
                        merged,
                        tech,
                    )
                })(),
                _ => gate_cells.get(op).and_then(|cell| {
                    let mut fanins = Vec::with_capacity(enode.children.len());
                    let mut child_delays = Vec::with_capacity(enode.children.len());
                    let mut child_maps = Vec::with_capacity(enode.children.len());
                    for child in enode.children.iter().copied() {
                        let child_choice = best_eclass(
                            child,
                            egraph,
                            gate_cells,
                            input_arrivals,
                            tech,
                            objective,
                            best,
                            visiting,
                        )?;
                        fanins.push(usize::from(child));
                        child_delays.push(child_choice.delay);
                        child_maps.push(child_choice.class_areas);
                    }
                    let merged = union_many_class_areas(&child_maps);
                    gate_choice_for_cell(key, cell, &fanins, &child_delays, merged, tech, objective)
                }),
            };
            choice = match (choice, cand) {
                (Some(cur), Some(newc)) if better_choice(&newc, &cur, objective).is_lt() => {
                    Some(newc)
                }
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
        gate_cells.as_ref(),
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

pub fn build_mapped_cover_binding(
    cut_output_index: usize,
    cut: &crate::emap_cut::Cut,
    leaves: &[usize],
    cover: &MappedCoverChoice,
    wire_delay: f32,
) -> crate::emap::NodeBinding {
    crate::emap::NodeBinding {
        node: cut_output_index,
        cut: cut.clone(),
        leaves: cover.used_leaf_indices.iter().map(|&i| leaves[i]).collect(),
        kind: NodeBindingKind::MappedCover {
            graph: cover.graph.clone(),
        },
        phase_neg: false,
        native_delay: cover.delay + wire_delay,
        delay: cover.delay + wire_delay,
        area: cover.area,
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
///
/// Returns the full [`MappedCoverChoice`] on success, containing the cell-level graph,
/// real techlib delay/area, and which input indices were actually used.
pub fn map_cut_to_cover(
    expr: &str,
    tech: &crate::techlib::TechLib,
    input_arrivals: &[f32],
    objective: crate::techlib::MappingObjective,
) -> Option<MappedCoverChoice> {
    let ps = crate::emap::EmapParams {
        max_cover_nodes: 256,
        ..Default::default()
    };
    extract_mapped_cover_from_expr(expr, tech, input_arrivals, objective, &ps, None)
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
    let ps = crate::emap::EmapParams {
        max_cover_nodes: 256,
        ..Default::default()
    };
    let cover = extract_mapped_cover_from_expr(expr, tech, input_arrivals, objective, &ps, None)?;
    let mut memo = HashMap::<usize, String>::new();
    let sexp = materialized_sexp(cover.graph.root, &cover.graph, tech, &mut memo)?;
    Some((sexp, cover.delay, cover.area))
}
