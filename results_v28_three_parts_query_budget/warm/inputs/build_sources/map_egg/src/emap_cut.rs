use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    usize,
};

use crate::aig_graph::{Aig as AigGraph, AigNode};

use itertools::Itertools;
use petgraph::prelude::*;

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Cut {
    inputs: Vec<usize>,
    output: usize,
    nodes: Vec<usize>,
}

impl Cut {
    #[must_use]
    pub fn new(inputs: Vec<usize>, output: usize, nodes: Vec<usize>) -> Self {
        assert!((0..inputs.len() - 1).all(|i| inputs[i] <= inputs[i + 1]));
        assert!((0..nodes.len() - 1).all(|i| nodes[i] <= nodes[i + 1]));
        assert!(nodes.len() == 1 || !inputs.contains(&output));
        assert!(nodes.contains(&output));
        Self {
            inputs,
            output,
            nodes,
        }
    }

    #[must_use]
    pub fn single_node(node: usize) -> Self {
        Self::new(vec![node], node, vec![node])
    }

    #[must_use]
    pub fn input_count(&self) -> usize {
        if self.inputs.contains(&0) {
            self.inputs.len() - 1
        } else {
            self.inputs.len()
        }
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Merge two cuts into a new cut, rooted at `output`.
    ///
    /// This method is linear in the number of nodes in `x`, `y` and `z`.
    #[must_use]
    pub fn union2(x: &Self, y: &Self, output: usize) -> Self {
        let nodes = [x.output, y.output];
        let inputs = x
            .inputs
            .iter()
            .merge(&y.inputs)
            .dedup()
            .copied()
            .collect::<Vec<_>>();
        let nodes = x
            .nodes
            .iter()
            .merge(&y.nodes)
            .merge(&nodes)
            .dedup()
            .copied()
            .collect::<Vec<_>>();
        Self {
            inputs,
            output,
            nodes,
        }
    }

    pub fn inputs(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.inputs.iter().map(|node| NodeIndex::new(*node))
    }

    #[must_use]
    pub fn input_indices(&self) -> &[usize] {
        &self.inputs
    }

    #[must_use]
    pub const fn output_index(&self) -> usize {
        self.output
    }

    pub fn nodes(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.nodes.iter().map(|node| NodeIndex::new(*node))
    }
}

/// Combinational depth from PIs / const (AND depth = 1 + max(children); outputs match driver).
#[must_use]
pub fn compute_aig_comb_depth(aig: &AigGraph, len: usize) -> Vec<u32> {
    let mut d = vec![0u32; len];
    let mut topo = petgraph::visit::Topo::new(aig.graph());
    while let Some(node) = topo.next(aig.graph()) {
        let idx = node.index();
        if idx >= len {
            continue;
        }
        d[idx] = match aig.node_type(node) {
            AigNode::Zero | AigNode::Input(_) => 0,
            AigNode::And => {
                let Some((xe, ye)) = aig.try_unwrap_and(node) else {
                    continue;
                };
                let x = aig.edge_source(xe).index();
                let y = aig.edge_source(ye).index();
                let dx = d.get(x).copied().unwrap_or(0);
                let dy = d.get(y).copied().unwrap_or(0);
                dx.max(dy).saturating_add(1)
            }
            AigNode::Output(_) => aig
                .graph()
                .edges_directed(node, Incoming)
                .next()
                .map(|e| d.get(e.source().index()).copied().unwrap_or(0))
                .unwrap_or(0),
        };
    }
    d
}

/// Structural ordering for cut sets before tech mapping (cf. mockturtle cut prioritization):
/// ascending Σ `est_ref` over the cone, then max combinational depth, then cone size, then leaves.
pub fn sort_cuts_by_flow_depth_cone(est_refs: &[f32], comb_depth: &[u32], cuts: &mut [Cut]) {
    fn metrics(est_refs: &[f32], comb_depth: &[u32], c: &Cut) -> (f32, u32, usize) {
        let mut sum_flow = 0.0f32;
        let mut max_dep = 0u32;
        for n in c.nodes() {
            let i = n.index();
            sum_flow += est_refs.get(i).copied().unwrap_or(1.0);
            max_dep = max_dep.max(comb_depth.get(i).copied().unwrap_or(0));
        }
        (sum_flow, max_dep, c.node_count())
    }
    cuts.sort_by(|a, b| {
        let ma = metrics(est_refs, comb_depth, a);
        let mb = metrics(est_refs, comb_depth, b);
        ma.0.total_cmp(&mb.0)
            .then_with(|| ma.1.cmp(&mb.1))
            .then_with(|| ma.2.cmp(&mb.2))
            .then_with(|| a.input_indices().cmp(b.input_indices()))
    });
}

/// Mockturtle [`cut_enumeration`](https://github.com/lsils/mockturtle/blob/master/include/mockturtle/algorithms/cut_enumeration.hpp)-style
/// merge for one AND: take **all** pairs of child cuts, merge with [`Cut::union2`], add `trivial`,
/// deduplicate, structurally sort, then keep the best `cut_limit` (same role as
/// `cut_enumeration_params::cut_limit`).
///
/// This replaces a blind Cartesian **enumeration order** cap: every distinct merged cut from the
/// full child cross-product is considered before the limit (child lists should stay bounded by the
/// same limit from the previous level, as in mockturtle).
#[must_use]
pub fn enumerate_merged_cuts_mockturtle(
    left: &[Cut],
    right: &[Cut],
    output_idx: usize,
    trivial: Cut,
    max_inputs: usize,
    cut_limit: usize,
    est_refs: &[f32],
    comb_depth: &[u32],
) -> Vec<Cut> {
    let cut_limit = cut_limit.max(1);
    let mut seen: HashSet<Cut> = HashSet::new();
    let mut out = Vec::new();
    if seen.insert(trivial.clone()) {
        out.push(trivial);
    }
    for (xc, yc) in left.iter().cartesian_product(right.iter()) {
        let cut = Cut::union2(xc, yc, output_idx);
        if cut.input_count() <= max_inputs && seen.insert(cut.clone()) {
            out.push(cut);
        }
    }
    sort_cuts_by_flow_depth_cone(est_refs, comb_depth, &mut out);
    out.truncate(cut_limit);
    out
}

pub struct Mapper<'a> {
    max_cuts: usize,
    max_inputs: usize,
    lut_area: &'a [u32],
    lut_delay: &'a [&'a [i32]],
    wire_delay: i32,
    aig: &'a AigGraph,
    cuts: Vec<Vec<Cut>>,
    depth: Vec<i32>,
    max_depth: i32,
    required: Vec<i32>,
    area_flow: Vec<f32>,
    edge_flow: Vec<f32>,
    references: Vec<u32>,
}

impl<'a> Mapper<'a> {
    #[must_use]
    pub fn new(
        max_cuts: usize,
        max_inputs: usize,
        lut_area: &'a [u32],
        lut_delay: &'a [&'a [i32]],
        wire_delay: i32,
        aig: &'a AigGraph,
    ) -> Self {
        assert!(!petgraph::algo::is_cyclic_directed(aig.graph()));

        let len = aig
            .graph()
            .node_indices()
            .map(|node| node.index())
            .max()
            .unwrap()
            + 1;
        Self {
            max_cuts,
            max_inputs,
            lut_area,
            lut_delay,
            wire_delay,
            aig,
            cuts: vec![Vec::new(); len],
            depth: vec![-1000; len],
            max_depth: -1000,
            required: vec![-1; len],
            area_flow: vec![0.0; len],
            edge_flow: vec![0.0; len],
            references: vec![0; len],
        }
    }

    #[inline]
    pub fn cut_dep_mffc_prefixexp(&self, cut: &Cut) -> (usize, usize, String) {
        let mut discovered: HashMap<usize, String> = cut
            .inputs
            .iter()
            .enumerate()
            .map(|(index, pi)| {
                (
                    *pi,
                    String::from_utf8(vec!['a' as u8 + index as u8]).unwrap(),
                )
            })
            .collect::<HashMap<usize, String>>();
        discovered.insert(0, "0".into());
        let mut expsi_odegs: HashMap<usize, (usize, usize)> = cut
            .inputs
            .iter()
            .map(|pi| (*pi, (usize::MAX, 0 as usize)))
            .collect::<HashMap<usize, (usize, usize)>>();
        let mut max_dep: usize = 0;
        fn inv_wrap(s: String, inv: bool) -> String {
            if inv {
                format!("(~ {s})")
            } else {
                s
            }
        }
        fn build_expr(
            this: &Mapper<'_>,
            node: usize,
            discovered: &mut HashMap<usize, String>,
            expsi_odegs: &mut HashMap<usize, (usize, usize)>,
            dep: usize,
            max_dep: &mut usize,
        ) -> String {
            if let Some(s) = discovered.get(&node) {
                if let Some(v) = expsi_odegs.get_mut(&node) {
                    v.1 += 1;
                }
                return s.clone();
            }
            *max_dep = std::cmp::max(*max_dep, dep);
            let (x_edge, y_edge) = this
                .aig
                .try_unwrap_and(NodeIndex::new(node))
                .expect("AND node with less than two inputs");
            let x = this.aig.edge_source(x_edge).index();
            let y = this.aig.edge_source(y_edge).index();
            let xs = inv_wrap(
                build_expr(this, x, discovered, expsi_odegs, dep + 1, max_dep),
                this.aig.is_edge_inverted(x_edge),
            );
            let ys = inv_wrap(
                build_expr(this, y, discovered, expsi_odegs, dep + 1, max_dep),
                this.aig.is_edge_inverted(y_edge),
            );
            let expr = format!("(& {xs} {ys})");
            expsi_odegs.insert(node, (0, 1));
            discovered.insert(node, expr.clone());
            expr
        }
        let original_expr = build_expr(
            self,
            cut.output,
            &mut discovered,
            &mut expsi_odegs,
            1,
            &mut max_dep,
        );
        let mffc: usize = 1 + cut
            .nodes
            .iter()
            .filter(|node| {
                !cut.inputs.contains(node)
                    && **node != cut.output
                    && self.aig.output_degree(NodeIndex::new(**node))
                        <= expsi_odegs.get(&node).unwrap().1
            })
            .count();

        (mffc, max_dep, original_expr)
    }

    #[must_use]
    #[inline]
    pub fn cut_depth(&self, cut: &Cut) -> i32 {
        cut.inputs
            .iter()
            .filter(|node| **node != 0)
            .sorted_by_key(|node| self.depth[**node])
            .rev()
            .enumerate()
            .map(|(index, node)| self.depth[*node] + self.lut_delay[cut.input_count()][index])
            .max()
            .unwrap_or(0)
            + self.wire_delay
    }

    #[must_use]
    #[inline]
    pub fn area_flow(&self, cut: &Cut) -> f32 {
        (cut.inputs
            .iter()
            .map(|node| self.area_flow[*node])
            .sum::<f32>()
            + self.lut_area[cut.input_count()] as f32)
            / (self.references[cut.output].min(1) as f32)
    }

    #[must_use]
    #[inline]
    pub fn edge_flow(&self, cut: &Cut) -> f32 {
        (cut.inputs
            .iter()
            .map(|node| self.edge_flow[*node])
            .sum::<f32>()
            + cut.input_count() as f32)
            / (self.references[cut.output].min(1) as f32)
    }

    #[must_use]
    #[inline]
    pub fn exact_area(&mut self, cut: &Cut) -> u32 {
        let references = self.references.clone();
        if let Some(repr_cut) = self.cuts[cut.output].first() {
            if cut == repr_cut {
                let _area2 = self.exact_area_ref(cut);
                let area1 = self.exact_area_deref(cut);
                //assert_eq!(area1, area2);
                self.references = references;
                return area1;
            }
        }

        let area1 = self.exact_area_deref(cut);
        let _area2 = self.exact_area_ref(cut);
        //assert_eq!(area1, area2);
        self.references = references;
        area1
    }

    fn exact_area_deref(&mut self, cut: &Cut) -> u32 {
        let mut area = self.lut_area[cut.input_count()];
        for input in &cut.inputs {
            if !self.aig.input_nodes().contains(&NodeIndex::new(*input))
                && self.references[*input] >= 1
            {
                assert!(
                    self.references[*input] >= 1,
                    "reference count of {} is zero before decrementing",
                    *input
                );
                self.references[*input] -= 1;
                if self.references[*input] == 0 {
                    area += self.exact_area_deref(&self.cuts[*input][0].clone());
                }
            }
        }
        area
    }

    fn exact_area_ref(&mut self, cut: &Cut) -> u32 {
        let mut area = self.lut_area[cut.input_count()];
        for input in &cut.inputs {
            if !self.aig.input_nodes().contains(&NodeIndex::new(*input)) {
                if self.references[*input] == 0 {
                    area += self.exact_area_ref(&self.cuts[*input][0].clone());
                }
                self.references[*input] += 1;
            }
        }
        area
    }

    #[must_use]
    #[inline]
    pub fn exact_edge(&mut self, cut: &Cut) -> usize {
        let references = self.references.clone();
        if let Some(repr_cut) = self.cuts[cut.output].first() {
            if cut == repr_cut {
                let _area2 = self.exact_edge_deref(cut);
                let area1 = self.exact_edge_ref(cut);
                self.references = references;
                //assert_eq!(area1, area2);
                return area1;
            }
        }

        let area1 = self.exact_edge_ref(cut);
        //let area2 = self.exact_edge_deref(cut);
        self.references = references;
        //assert_eq!(area1, area2);
        area1
    }

    fn exact_edge_deref(&mut self, cut: &Cut) -> usize {
        let mut area = cut.input_count();
        for input in &cut.inputs {
            if !self.aig.input_nodes().contains(&NodeIndex::new(*input))
                && self.references[*input] >= 1
            {
                assert!(
                    self.references[*input] >= 1,
                    "reference count of {} is zero before decrementing",
                    *input
                );
                self.references[*input] -= 1;
                if self.references[*input] == 0 {
                    area += self.exact_edge_deref(&self.cuts[*input][0].clone());
                }
            }
        }
        area
    }

    fn exact_edge_ref(&mut self, cut: &Cut) -> usize {
        let mut area = cut.input_count();
        for input in &cut.inputs {
            if !self.aig.input_nodes().contains(&NodeIndex::new(*input)) {
                if self.references[*input] == 0 {
                    area += self.exact_edge_ref(&self.cuts[*input][0].clone());
                }
                self.references[*input] += 1;
            }
        }
        area
    }

    #[must_use]
    #[inline]
    pub fn cut_rank_depth(&mut self, lhs: &Cut, rhs: &Cut) -> Ordering {
        let lhs = self.cut_depth(lhs);
        let rhs = self.cut_depth(rhs);
        lhs.cmp(&rhs)
    }

    #[must_use]
    #[inline]
    pub fn cut_rank_size(&mut self, lhs: &Cut, rhs: &Cut) -> Ordering {
        let lhs = lhs.input_count();
        let rhs = rhs.input_count();
        lhs.cmp(&rhs)
    }

    #[must_use]
    #[inline]
    pub fn cut_rank_area_flow(&mut self, lhs: &Cut, rhs: &Cut) -> Ordering {
        let lhs = self.area_flow(lhs);
        let rhs = self.area_flow(rhs);
        lhs.partial_cmp(&rhs).unwrap()
    }

    #[must_use]
    #[inline]
    pub fn cut_rank_edge_flow(&mut self, lhs: &Cut, rhs: &Cut) -> Ordering {
        let lhs = self.edge_flow(lhs);
        let rhs = self.edge_flow(rhs);
        lhs.partial_cmp(&rhs).unwrap()
    }

    #[must_use]
    #[inline]
    pub fn cut_rank_fanin_refs(&mut self, lhs: &Cut, rhs: &Cut) -> Ordering {
        let lhs = lhs
            .inputs
            .iter()
            .map(|node| self.references[*node])
            .sum::<u32>()
            / (lhs.inputs.len() as u32);
        let rhs = rhs
            .inputs
            .iter()
            .map(|node| self.references[*node])
            .sum::<u32>()
            / (rhs.inputs.len() as u32);
        lhs.cmp(&rhs)
    }

    #[must_use]
    #[inline]
    pub fn cut_rank_exact_area(&mut self, lhs: &Cut, rhs: &Cut) -> Ordering {
        let lhs = self.exact_area(lhs);
        let rhs = self.exact_area(rhs);
        lhs.cmp(&rhs)
    }

    #[must_use]
    #[inline]
    pub fn cut_rank_exact_edge(&mut self, lhs: &Cut, rhs: &Cut) -> Ordering {
        let lhs = self.exact_edge(lhs);
        let rhs = self.exact_edge(rhs);
        lhs.cmp(&rhs)
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn compute_cuts<F1, F2, F3>(&mut self, sort_first: F1, sort_second: F2, sort_third: F3)
    where
        F1: Fn(&mut Self, &Cut, &Cut) -> Ordering,
        F2: Fn(&mut Self, &Cut, &Cut) -> Ordering,
        F3: Fn(&mut Self, &Cut, &Cut) -> Ordering,
    {
        for node in self.aig.graph().node_indices() {
            self.depth[node.index()] = -1000;
            self.area_flow[node.index()] = 0.0;
            self.edge_flow[node.index()] = 0.0;
            self.references[node.index()] = 0;
        }

        for node in self.aig.input_nodes() {
            self.cuts[node.index()] = vec![Cut::single_node(node.index())];
            self.depth[node.index()] = 0;
            self.area_flow[node.index()] = 0.0;
            self.edge_flow[node.index()] = 0.0;
        }

        for node in self.aig.graph().externals(Outgoing) {
            if let AigNode::Output(node) = self.aig.graph()[node] {
                self.references[node as usize] = 1;
            }
        }

        let mut iter = petgraph::visit::Topo::new(self.aig.graph());

        let mut cut_count = 0;

        while let Some(node) = iter.next(&self.aig.graph()) {
            if let Some((x_edge, y_edge)) = self.aig.try_unwrap_and(node) {
                let x = self.aig.edge_source(x_edge);
                let y = self.aig.edge_source(y_edge);

                // Compute the trivial cut of this node.
                //let mut inputs = vec![x.index(), y.index(), z.index()];
                //let mut nodes = vec![x.index(), y.index(), z.index(), node.index()];
                let inputs = vec![node.index()];
                let nodes = vec![node.index()];
                //inputs.sort_unstable();
                //nodes.sort_unstable();
                let cut = Cut::new(inputs, node.index(), nodes);

                let max_inputs = self.max_inputs;
                let required = self.required[node.index()];

                let cuts = self.cuts[x.index()]
                    .iter()
                    .cartesian_product(&self.cuts[y.index()])
                    .map(|(x_cut, y_cut)| Cut::union2(x_cut, y_cut, node.index()))
                    .chain(self.cuts[node.index()].first().cloned())
                    .filter(|candidate| candidate.input_count() <= max_inputs)
                    .filter(|candidate| required < 0 || self.cut_depth(candidate) <= required)
                    .collect_vec();

                let mut cuts = cuts
                    .into_iter()
                    .sorted_by(|lhs, rhs| {
                        sort_first(self, lhs, rhs)
                            .then_with(|| sort_second(self, lhs, rhs))
                            .then_with(|| sort_third(self, lhs, rhs))
                    })
                    .dedup()
                    .take(self.max_cuts)
                    .collect_vec();

                cuts.push(cut);

                assert!(!cuts.is_empty());

                cut_count += cuts.len();

                self.cuts[node.index()] = cuts;
                let best_cut = &self.cuts[node.index()][0];

                for input in &best_cut.inputs {
                    self.references[*input] += 1;
                    self.area_flow[*input] = self.area_flow(&self.cuts[*input][0]);
                    self.edge_flow[*input] = self.edge_flow(&self.cuts[*input][0]);
                }

                self.depth[node.index()] = self.cut_depth(best_cut);
                self.area_flow[node.index()] = self.area_flow(best_cut);
                self.edge_flow[node.index()] = self.edge_flow(best_cut);

                self.max_depth = self.max_depth.max(self.depth[node.index()]);
            }
        }

        println!("Found {} cuts", cut_count);
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn map_luts(&mut self, print_stats: bool) -> Vec<Cut> {
        let mut frontier = self
            .aig
            .graph()
            .externals(Outgoing)
            .flat_map(|output| self.aig.graph().neighbors_directed(output, Incoming))
            .filter(|node| self.aig.node_type(*node) == AigNode::And)
            .collect::<Vec<_>>();
        let mut mapping = Vec::new();
        let mut mapped_nodes = Vec::new();
        let input_nodes = self.aig.input_nodes();

        while let Some(node) = frontier.pop() {
            let cut = &self.cuts[node.index()][0];
            for input in cut.inputs().filter(|node| node.index() != 0) {
                if !mapped_nodes.contains(&input) && !input_nodes.contains(&input) {
                    frontier.push(input);
                }
            }

            mapped_nodes.push(NodeIndex::new(cut.output));
            mapping.push(cut.clone());
        }

        if print_stats {
            println!("Mapped to {} LUTs", mapping.len());
            println!(
                "Estimated area: {} units",
                mapping
                    .iter()
                    .map(|cut| self.lut_area[cut.input_count()])
                    .sum::<u32>()
            );

            for i in 1..=self.max_inputs {
                println!(
                    "LUT{}: {}",
                    i,
                    mapping.iter().filter(|cut| cut.input_count() == i).count()
                );
            }

            println!("Maximum delay: {}", self.max_depth);
        }

        for node in self.aig.graph().node_indices() {
            self.required[node.index()] = self.max_depth;
        }

        let mut required_dfs = petgraph::visit::DfsPostOrder::empty(self.aig.graph());
        let pis = self.aig.input_nodes();
        for pi in pis {
            required_dfs.move_to(pi);
            while let Some(node) = required_dfs.next(self.aig.graph()) {
                if let Some(cut) = self.cuts[node.index()].first() {
                    let required = self.required[node.index()] - self.wire_delay;
                    for (index, input) in cut
                        .inputs()
                        .filter(|node| node.index() != 0)
                        .sorted_by_key(|node| self.depth[node.index()])
                        .rev()
                        .enumerate()
                    {
                        if input.index() != node.index() {
                            self.required[input.index()] = self.required[input.index()]
                                .min(required - self.lut_delay[cut.input_count()][index]);
                            assert!(
                                self.required[input.index()] >= 0,
                                "node {} has negative required time of {}",
                                input.index(),
                                self.required[input.index()]
                            );
                        }
                    }
                }
            }
        }

        mapping
    }
}
