//! Bounded **two-level supergates**: implement one cut with two library cells.
//!
//! mockturtle's `emap` matches a cut against `supergate`s, and a `supergate` wraps a
//! `composed_gate` -- a *composition* of library cells with precomputed `area` and pin-to-pin
//! `tdelay` (`mockturtle/utils/include/supergate.hpp`). emapper's direct path can only bind a cut
//! to **one** cell, which is why it never emits functions such as `AO221 = (A1*A2)+(B1*B2)+C` or
//! `AO32 = (A1*A2*A3)+(B1*B2)`: those are not single library cells in asap7, yet mockturtle uses
//! them 22 times on c880 alone. Everything else in the mapper is saturated -- cut-pool width
//! 3..9 and `max_cuts` 16..512 give byte-identical results -- so the missing compositions, not the
//! search space, are what keep emapper deeper and slower.
//!
//! This module recovers a bounded form of it. Given a cut it walks the cut's own AIG cone and, for
//! each internal node `u`, tries to realise `u`'s sub-cone with one cell and the cut's root with a
//! second cell that takes `u`'s output as a single input. Both matches are ordinary library lookups,
//! so cost stays close to the direct path; the win is that a function no single cell realises can
//! still be built from two.

use std::collections::HashMap;

use crate::aig_graph::{Aig as AigGraph, AigNode};
use crate::emap::{tt_reduce, tt_support};
use crate::emap_cover::{MappedCoverGraph, MappedCoverNode};
use crate::emap_cut::Cut;
use crate::techlib::{MappingAreaCost, MappingObjective, PhaseArrival, TechLib};
use crate::tt9::{tt9_and_masked, tt9_bit_set, tt9_not_masked, tt9_zero, Tt9};

/// A two-cell realisation of a cut's function.
pub struct TwoLevelCover {
    pub graph: MappedCoverGraph,
    pub area: f32,
}

/// Truth table of variable `idx` over `n` variables.
fn var_tt(idx: usize, n: usize) -> Tt9 {
    if n == 0 {
        return tt9_zero();
    }
    let mut tt = tt9_zero();
    for a in 0..(1usize << n) {
        if ((a >> idx) & 1) == 1 {
            tt9_bit_set(&mut tt, a);
        }
    }
    tt
}

/// Truth tables of every node of `cut` over `n_vars` variables.
///
/// `extra` replaces one node by a free variable, which is how the root function is expressed when an
/// internal node of the cone is abstracted away and becomes an input of the upper cell.
fn cone_tts(
    aig: &AigGraph,
    cut: &Cut,
    leaf_var: &HashMap<usize, usize>,
    extra: Option<(usize, usize)>,
    n_vars: usize,
) -> Option<HashMap<usize, Tt9>> {
    let n = u8::try_from(n_vars).ok()?;
    let mut tts: HashMap<usize, Tt9> = HashMap::new();
    for node in cut.nodes() {
        let idx = node.index();
        if extra.is_some_and(|(enode, _)| enode == idx) {
            let (_, evar) = extra?;
            tts.insert(idx, var_tt(evar, n_vars));
            continue;
        }
        let tt = match aig.node_type(node) {
            AigNode::Zero => tt9_zero(),
            AigNode::Input(_) => var_tt(*leaf_var.get(&idx)?, n_vars),
            AigNode::And => {
                let (xe, ye) = aig.try_unwrap_and(node)?;
                let mut tx = *tts.get(&aig.edge_source(xe).index())?;
                if aig.is_edge_inverted(xe) {
                    tx = tt9_not_masked(&tx, n);
                }
                let mut ty = *tts.get(&aig.edge_source(ye).index())?;
                if aig.is_edge_inverted(ye) {
                    ty = tt9_not_masked(&ty, n);
                }
                tt9_and_masked(&tx, &ty, n)
            }
            _ => return None,
        };
        tts.insert(idx, tt);
    }
    Some(tts)
}

/// Largest cut this module will attempt. The abstraction adds one variable, and [`Tt9`] holds nine.
const MAX_CUT_LEAVES: usize = 8;
/// Cap on the number of internal split nodes tried per cut, to bound the second cone pass.
const MAX_SPLITS: usize = 6;

/// Build a two-cell realisation of `cut`'s function (negated when `target_neg`).
///
/// `leaves` is the cut's dense variable order (`a`, `b`, ...); `input_phase_arrivals[i]` is the
/// arrival of leaf `i` in both phases. Returns `None` when no split yields two matching cells.
pub fn two_level_cover(
    aig: &AigGraph,
    tech: &TechLib,
    cut: &Cut,
    leaves: &[usize],
    input_phase_arrivals: &[PhaseArrival],
    objective: MappingObjective,
    target_neg: bool,
) -> Option<TwoLevelCover> {
    let n = leaves.len();
    if n == 0 || n > MAX_CUT_LEAVES || input_phase_arrivals.len() != n {
        return None;
    }
    let leaf_var: HashMap<usize, usize> = leaves.iter().enumerate().map(|(i, &l)| (l, i)).collect();
    let root = cut.output_index();
    let base = cone_tts(aig, cut, &leaf_var, None, n)?;
    let n_u8 = u8::try_from(n).ok()?;

    let mut tried = 0usize;
    for node in cut.nodes() {
        let u = node.index();
        if u == root || aig.node_type(node) != AigNode::And || leaf_var.contains_key(&u) {
            continue;
        }
        if tried >= MAX_SPLITS {
            break;
        }
        tried += 1;

        // Lower cell: realise the split node's own cone.
        let Some(tt_u) = base.get(&u).copied() else {
            continue;
        };
        let sup_u = tt_support(&tt_u, n_u8);
        let m = sup_u.len();
        if m < 2 || m > MAX_CUT_LEAVES {
            continue;
        }
        let tt_u_red = tt_reduce(&tt_u, &sup_u);
        let arr_u: Vec<PhaseArrival> = sup_u.iter().map(|&i| input_phase_arrivals[i]).collect();
        let Some((h_delay, _, _h_phys, h_variant)) = tech.best_for_tt_phase_arrivals(
            u8::try_from(m).ok()?,
            tt_u_red,
            &arr_u,
            objective,
            None,
            None,
            MappingAreaCost::Physical,
        ) else {
            continue;
        };

        // Upper cell: the root function with `u` abstracted to one variable.
        let n2 = n + 1;
        if n2 > 9 {
            continue;
        }
        let Some(tts2) = cone_tts(aig, cut, &leaf_var, Some((u, n)), n2) else {
            continue;
        };
        let Some(mut f_root) = tts2.get(&root).copied() else {
            continue;
        };
        let n2_u8 = u8::try_from(n2).ok()?;
        if target_neg {
            f_root = tt9_not_masked(&f_root, n2_u8);
        }
        let sup_r = tt_support(&f_root, n2_u8);
        // The split is only useful if the upper cell really consumes the lower cell's output.
        if !sup_r.contains(&n) || sup_r.len() < 2 || sup_r.len() > MAX_CUT_LEAVES {
            continue;
        }
        let f_red = tt_reduce(&f_root, &sup_r);
        let k = sup_r.len();
        let arr_r: Vec<PhaseArrival> = sup_r
            .iter()
            .map(|&i| {
                if i == n {
                    PhaseArrival {
                        pos: h_delay,
                        neg: h_delay + tech.inv_delay,
                    }
                } else {
                    input_phase_arrivals[i]
                }
            })
            .collect();
        let Some((_, _, _g_phys, g_variant)) = tech.best_for_tt_phase_arrivals(
            u8::try_from(k).ok()?,
            f_red,
            &arr_r,
            objective,
            None,
            None,
            MappingAreaCost::Physical,
        ) else {
            continue;
        };

        // Cover graph: every variable as an Input, then the lower cell, then the upper cell.
        let mut nodes: Vec<MappedCoverNode> = (0..n).map(MappedCoverNode::Input).collect();
        let low = nodes.len();
        nodes.push(MappedCoverNode::Cell {
            cell: h_variant.clone(),
            fanins: sup_u.clone(),
        });
        let high = nodes.len();
        nodes.push(MappedCoverNode::Cell {
            cell: g_variant.clone(),
            fanins: sup_r
                .iter()
                .map(|&i| if i == n { low } else { i })
                .collect(),
        });
        let area = tech.variant_area_physical_invs(h_variant, m, None)
            + tech.variant_area_physical_invs(g_variant, k, None);
        return Some(TwoLevelCover {
            graph: MappedCoverGraph { nodes, root: high },
            area,
        });
    }
    None
}
