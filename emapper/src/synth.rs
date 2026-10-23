//! Cut-level **structural synthesis**: build a small library-cell network for a cut's truth table.
//!
//! This is the piece the mapper was missing. Its cover supply used to be an e-graph saturation with
//! the one-way `AIG_pattern(cell) -> gate(cell)` rules, which does not restructure the Boolean
//! network at all -- it only matches library cells against the *existing* AIG sub-expressions.
//! Measured on c432 cut functions, raising the saturation budget from 2 iterations / 500 nodes to
//! 30 / 200 000 produced **byte-identical** covers, which is why every "more search" attempt on the
//! e-graph path was inert.
//!
//! `emap` gets its wide, level-collapsing cells from `tech_library`'s offline supergate
//! precomputation; this project's C++ `epmapping` engine gets them from `synthesize_cone`
//! (SOP factoring / DSD / exact) -- and there `synth` is the *only* winning stage on c432 and the
//! second largest on c7552, while the e-graph stage wins nothing on c432. This module is the Rust
//! equivalent for a single cut.
//!
//! # Method
//!
//! Shannon decomposition with library matching:
//!
//! ```text
//! synth(f) = a cell whose native function is f           (if the library has one)
//!          | mux(x, synth(f|x=1), synth(f|x=0))          (otherwise, x a support variable)
//! ```
//!
//! The mux is realised with the library's own 4-input `(a&b)|(c&d)` cell (asap7: `AO22`), with `~x`
//! produced by one explicit inverter node. Every cell is fetched as an **identity variant** built
//! here (`perm = 0,1,..`, `neg_mask = 0`, `out_neg = false`), so graph `Input(i)` feeds cell
//! variable `i` with no permutation bookkeeping -- the same construction `TechLib::detect_and2`
//! uses for the exact-AND2 variant.

use std::collections::HashSet;

use crate::emap_cover::{
    compute_pin_to_root_delays, MappedCoverChoice, MappedCoverGraph, MappedCoverNode,
};
use crate::techlib::{CellVariant, MappingObjective, TechLib};
use crate::tt9::{
    tt9_all_ones, tt9_and_masked, tt9_bit_get, tt9_bit_set, tt9_not_masked, tt9_zero, Tt9,
};

/// Maximum mux nesting depth of a synthesised cover (`2` covers "absorb two AIG levels").
const MAX_DEPTH: u32 = 4;
// NOTE: the depth is the number of *variables eliminated* -- one mux per level -- so this strategy
// only reaches wide cut functions by building a deep mux chain, which is exactly what the mapper's
// delay model rejects. Measured against the e-graph cover extractor on real cut functions it loses
// every multi-cell case (area 2-3x, delay far worse), because the e-graph preserves the AIG
// structure and matches 2-level AND-inverter patterns onto AOI/OAI cells while mux decomposition
// destroys that structure. Kept as a working, functionally-verified module (useful as a mux/small
// function primitive source), but it is NOT wired into the mapper.

/// Zero every minterm bit outside the `1 << n` space of an `n`-variable function.
fn mask_tt(tt: &Tt9, n: u8) -> Tt9 {
    let mut out = *tt;
    let total = 1usize << usize::from(n);
    for (word, slot) in out.iter_mut().enumerate() {
        let lo = word * 64;
        let hi = lo + 64;
        if lo >= total {
            *slot = 0;
        } else if hi > total {
            let keep = total - lo;
            *slot &= (1u64 << keep) - 1;
        }
    }
    out
}

fn var_tt(idx: usize, n: u8) -> Tt9 {
    let mut tt = tt9_zero();
    for m in 0..(1usize << usize::from(n)) {
        if ((m >> idx) & 1) == 1 {
            tt9_bit_set(&mut tt, m);
        }
    }
    tt
}

fn tt_is_const(tt: &Tt9, n: u8) -> Option<bool> {
    let m = mask_tt(tt, n);
    if m.iter().all(|w| *w == 0) {
        Some(false)
    } else if m == tt9_all_ones(n) {
        Some(true)
    } else {
        None
    }
}

/// Build an identity [`CellVariant`] (no permutation, no input/output inversion) for a library cell.
fn identity_variant(cell: &crate::techlib::Cell) -> CellVariant {
    CellVariant {
        cell_name: cell.name.clone(),
        out_pin: cell.out_pin.clone(),
        n_inputs: cell.vars.len() as u8,
        perm: (0..cell.vars.len() as u8).collect(),
        neg_mask: 0,
        out_neg: false,
        area: cell.area,
        pin_delay: cell
            .vars
            .iter()
            .map(|v| cell.pin_delay.get(v).copied().unwrap_or(0.0))
            .collect(),
        pin_names: cell.vars.clone(),
    }
}

/// A library cell whose own function equals `tt` over `n` variables, as an identity variant.
fn cell_for_tt(tech: &TechLib, tt: &Tt9, n: u8) -> Option<CellVariant> {
    if n == 0 {
        return None;
    }
    let want = mask_tt(tt, n);
    let mut best: Option<&crate::techlib::Cell> = None;
    for c in &tech.cells {
        if c.vars.len() != usize::from(n) {
            continue;
        }
        if mask_tt(&c.tt, n) != want {
            continue;
        }
        best = match best {
            None => Some(c),
            Some(b) if c.area < b.area => Some(c),
            _ => best,
        };
    }
    best.map(identity_variant)
}

/// The library's `(a&b)|(c&d)` cell as an identity variant (the mux primitive).
fn mux_variant(tech: &TechLib) -> Option<CellVariant> {
    let n = 4u8;
    let ab = tt9_and_masked(&var_tt(0, n), &var_tt(1, n), n);
    let cd = tt9_and_masked(&var_tt(2, n), &var_tt(3, n), n);
    // or(x, y) = !(!x & !y)
    let or = tt9_not_masked(
        &tt9_and_masked(&tt9_not_masked(&ab, n), &tt9_not_masked(&cd, n), n),
        n,
    );
    cell_for_tt(tech, &or, n)
}

/// Split `f` on variable position `p`; both results are over the reduced variable list (position
/// `p` removed).
fn cofactor(tt: &Tt9, n: usize, p: usize) -> (Tt9, Tt9) {
    let mut f1 = tt9_zero();
    let mut f0 = tt9_zero();
    for m in 0..(1usize << n) {
        if !tt9_bit_get(tt, m) {
            continue;
        }
        let mut reduced = 0usize;
        for j in 0..n {
            if j == p {
                continue;
            }
            let dst = if j < p { j } else { j - 1 };
            reduced |= ((m >> j) & 1) << dst;
        }
        if ((m >> p) & 1) == 1 {
            tt9_bit_set(&mut f1, reduced);
        } else {
            tt9_bit_set(&mut f0, reduced);
        }
    }
    (f1, f0)
}

/// Number of original cut variables the function actually depends on.
fn support(tt: &Tt9, n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for p in 0..n {
        let (f1, f0) = cofactor(tt, n, p);
        if mask_tt(&f1, (n - 1) as u8) != mask_tt(&f0, (n - 1) as u8) {
            out.push(p);
        }
    }
    out
}

/// Append `g` to `nodes` and return its root index after the offset.
fn append(nodes: &mut Vec<MappedCoverNode>, g: &MappedCoverGraph, offset: usize) -> usize {
    for nd in &g.nodes {
        let nd = match nd {
            MappedCoverNode::Const(v) => MappedCoverNode::Const(*v),
            MappedCoverNode::Input(i) => MappedCoverNode::Input(*i),
            MappedCoverNode::Cell { cell, fanins } => MappedCoverNode::Cell {
                cell: cell.clone(),
                fanins: fanins.iter().map(|f| f + offset).collect(),
            },
        };
        nodes.push(nd);
    }
    g.root + offset
}

fn input_node(nodes: &mut Vec<MappedCoverNode>, var: usize) -> usize {
    if let Some(i) = nodes
        .iter()
        .position(|n| matches!(n, MappedCoverNode::Input(v) if *v == var))
    {
        return i;
    }
    let i = nodes.len();
    nodes.push(MappedCoverNode::Input(var));
    i
}

/// Recursive synthesis. `vars[j]` is the **original cut variable index** of reduced variable `j`.
fn build(tt: &Tt9, vars: &[usize], tech: &TechLib, depth: u32) -> Option<MappedCoverGraph> {
    let n = vars.len();
    let nu = n as u8;

    if let Some(v) = tt_is_const(tt, nu) {
        return Some(MappedCoverGraph {
            nodes: vec![MappedCoverNode::Const(v)],
            root: 0,
        });
    }
    // A single library cell realising the function outright.
    if let Some(cell) = cell_for_tt(tech, tt, nu) {
        let mut nodes: Vec<MappedCoverNode> = vars
            .iter()
            .map(|&var| MappedCoverNode::Input(var))
            .collect();
        let root = nodes.len();
        nodes.push(MappedCoverNode::Cell {
            cell,
            fanins: (0..n).collect(),
        });
        return Some(MappedCoverGraph { nodes, root });
    }
    if depth == 0 || n <= 1 || n > 9 {
        return None;
    }
    let mux = mux_variant(tech)?;
    let inv = tech.exact_inv_variant()?.clone();

    // Prefer a split whose cofactors are single cells (cheap win), then the smallest cofactors.
    let cands = support(tt, n);
    let mut best: Option<(usize, MappedCoverGraph, MappedCoverGraph)> = None;
    let mut best_key = usize::MAX;
    for &p in &cands {
        let (f1, f0) = cofactor(tt, n, p);
        let mut vars1 = vars.to_vec();
        vars1.remove(p);
        let (Some(g1), Some(g0)) = (
            build(&f1, &vars1, tech, depth - 1),
            build(&f0, &vars1, tech, depth - 1),
        ) else {
            continue;
        };
        // Key: total node count, with a bonus when both cofactors are a single cell.
        let key = g1.nodes.len() + g0.nodes.len();
        if key < best_key {
            best_key = key;
            best = Some((p, g1, g0));
        }
        if key <= 2 {
            break;
        }
    }
    let (p, g1, g0) = best?;

    let mut nodes: Vec<MappedCoverNode> = Vec::new();
    let r1 = append(&mut nodes, &g1, 0);
    let off0 = nodes.len();
    let r0 = append(&mut nodes, &g0, off0);
    let x = input_node(&mut nodes, vars[p]);
    let nx = nodes.len();
    nodes.push(MappedCoverNode::Cell {
        cell: inv,
        fanins: vec![x],
    });
    let root = nodes.len();
    nodes.push(MappedCoverNode::Cell {
        cell: mux,
        fanins: vec![x, r1, nx, r0],
    });
    Some(MappedCoverGraph { nodes, root })
}

/// Synthesise a library-cell network realising `tt` over `n_inputs` cut variables.
///
/// Returns `None` when the library cannot realise the function within [`MAX_DEPTH`] mux levels.
#[must_use]
pub fn synthesize_cut_cover(tt: Tt9, n_inputs: u8, tech: &TechLib) -> Option<MappedCoverGraph> {
    if n_inputs > 9 {
        return None;
    }
    let vars = (0..usize::from(n_inputs)).collect::<Vec<_>>();
    build(&mask_tt(&tt, n_inputs), &vars, tech, MAX_DEPTH)
}

/// Evaluate a synthesised graph back to a truth table (used by tests and callers to verify that a
/// synthesis result really implements the requested function).
#[must_use]
pub fn graph_tt(graph: &MappedCoverGraph, tech: &TechLib, n_inputs: u8) -> Option<Tt9> {
    let n = n_inputs;
    let total = 1usize << usize::from(n);
    let mut vals: Vec<Option<Tt9>> = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let v = match node {
            MappedCoverNode::Const(b) => {
                if *b {
                    tt9_all_ones(n)
                } else {
                    tt9_zero()
                }
            }
            MappedCoverNode::Input(li) => {
                if *li >= usize::from(n) {
                    return None;
                }
                var_tt(*li, n)
            }
            MappedCoverNode::Cell { cell, fanins } => {
                let base = tech.cells.iter().find(|c| c.name == cell.cell_name)?;
                if fanins.len() != cell.n_inputs as usize || cell.n_inputs == 0 {
                    return None;
                }
                let child = fanins
                    .iter()
                    .map(|f| vals.get(*f).copied().flatten())
                    .collect::<Option<Vec<_>>>()?;
                let mut out = tt9_zero();
                for m in 0..total {
                    let mut idx = 0usize;
                    for (bit, c) in child.iter().enumerate() {
                        if tt9_bit_get(c, m) {
                            idx |= 1usize << bit;
                        }
                    }
                    if tt9_bit_get(&base.tt, idx) {
                        tt9_bit_set(&mut out, m);
                    }
                }
                // `synthesize_cut_cover` only ever emits identity variants.
                if cell.neg_mask != 0
                    || cell.out_neg
                    || cell.perm.iter().enumerate().any(|(i, p)| i != *p as usize)
                {
                    return None;
                }
                out
            }
        };
        vals.push(Some(v));
    }
    vals.get(graph.root).copied().flatten()
}

/// Costed, materialisable synthesis result: the same shape the e-graph cover path returns, so it
/// can be offered to the mapper as an ordinary candidate.
#[must_use]
pub fn synthesize_cut_choice(
    tt: Tt9,
    n_inputs: u8,
    tech: &TechLib,
    input_arrivals: &[f32],
) -> Option<MappedCoverChoice> {
    let graph = synthesize_cut_cover(tt, n_inputs, tech)?;
    if graph_tt(&graph, tech, n_inputs)? != mask_tt(&tt, n_inputs) {
        return None;
    }
    let mut area = 0.0f32;
    for node in &graph.nodes {
        if let MappedCoverNode::Cell { cell, .. } = node {
            area += cell.area;
        }
    }
    let mut used = graph
        .nodes
        .iter()
        .filter_map(|n| match n {
            MappedCoverNode::Input(i) => Some(*i),
            _ => None,
        })
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    used.sort_unstable();
    let pin_to_root_delay = compute_pin_to_root_delays(&graph, tech, usize::from(n_inputs));
    let mut delay = 0.0f32;
    for &li in &used {
        let a = input_arrivals.get(li).copied().unwrap_or(0.0);
        delay = delay.max(a + pin_to_root_delay.get(li).copied().unwrap_or(0.0));
    }
    Some(MappedCoverChoice {
        graph,
        delay,
        area,
        used_leaf_indices: used,
        pin_to_root_delay,
    })
}

/// `MappingObjective` is part of the public signature for symmetry with the e-graph extractor; the
/// synthesis itself is objective-independent (the mapper prices the result).
#[must_use]
pub fn synthesize_cut_choice_for_objective(
    tt: Tt9,
    n_inputs: u8,
    tech: &TechLib,
    input_arrivals: &[f32],
    _objective: MappingObjective,
) -> Option<MappedCoverChoice> {
    synthesize_cut_choice(tt, n_inputs, tech, input_arrivals)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tech() -> TechLib {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tools/mockturtle/experiments/cell_libraries/asap7.genlib");
        TechLib::from_genlib(&p, 9).expect("asap7.genlib")
    }

    fn tt_of(sexp: &str) -> (u8, Tt9) {
        crate::emap_egg::sexp_to_tt9(sexp).expect("sexp parses")
    }

    /// Every synthesised network must implement exactly the requested function.
    #[test]
    fn synthesis_is_functionally_exact() {
        let tech = tech();
        let cases = [
            "(& a b)",
            "(~ (& a b))",
            "(~ (& (~ a) b))",
            "(& (~ (& (~ a) b)) (~ (& (~ c) d)))",
            "(& (& (~ (& (~ a) b)) (~ (& (~ c) d))) (& (~ (& (~ e) f)) (~ (& (~ g) h))))",
            "(& (~ (& (~ (& (~ a) b)) (~ (& (& c d) (& e f))))) (& g (~ h)))",
        ];
        let mut ok = 0usize;
        for s in cases {
            let (n, tt) = tt_of(s);
            if let Some(g) = synthesize_cut_cover(tt, n, &tech) {
                assert_eq!(
                    graph_tt(&g, &tech, n).expect("graph evaluates"),
                    mask_tt(&tt, n),
                    "synthesised network for {s} is not equivalent"
                );
                ok += 1;
            }
        }
        assert!(ok > 0, "synthesiser produced nothing for any case");
        eprintln!("synthesised {ok}/{} cases", cases.len());
    }

    /// A function no single cell realises must come back as a genuine multi-cell network.
    ///
    /// Scope note: the synthesizer eliminates one variable per mux level, so it only reaches
    /// functions whose cofactors bottom out in single cells within `MAX_DEPTH` levels. Wide cut
    /// functions (7-9 leaves) are out of reach -- which is why this module is not wired into the
    /// mapper (see the `MAX_DEPTH` note).
    #[test]
    fn four_input_function_gets_a_multi_cell_cover() {
        let tech = tech();
        let (n, tt) = tt_of("(& (~ (& (~ a) b)) (~ (& (~ c) d)))");
        assert!(
            cell_for_tt(&tech, &tt, n).is_none(),
            "this function should not be a single library cell"
        );
        let g = synthesize_cut_cover(tt, n, &tech).expect("multi-cell synthesis");
        let cells = g
            .nodes
            .iter()
            .filter(|x| matches!(x, MappedCoverNode::Cell { .. }))
            .count();
        assert!(
            cells > 1,
            "expected a multi-cell network, got {cells} cell(s)"
        );
        assert_eq!(graph_tt(&g, &tech, n).unwrap(), mask_tt(&tt, n));
    }

    /// Costed choice must be self-consistent with the graph.
    #[test]
    fn costed_choice_matches_graph() {
        let tech = tech();
        let (n, tt) = tt_of("(& (~ (& (~ a) b)) (~ (& (~ c) d)))");
        let arrivals = vec![0.0f32; usize::from(n)];
        let c = synthesize_cut_choice(tt, n, &tech, &arrivals).expect("choice");
        let mut area = 0.0f32;
        for node in &c.graph.nodes {
            if let MappedCoverNode::Cell { cell, .. } = node {
                area += cell.area;
            }
        }
        assert!((c.area - area).abs() < 1e-6);
        assert_eq!(c.pin_to_root_delay.len(), usize::from(n));
    }
}
