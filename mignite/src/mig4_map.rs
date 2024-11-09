use std::{cmp::Ordering, collections::HashMap, usize};

use crate::{
    mig4::{self, MigNode},
    mig4_egg::{simplify, CCost},
};

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

    /// Merge three cuts into a new cut, rooted at `output`.
    ///
    /// This method is linear in the number of nodes in `x`, `y` and `z`.
    #[must_use]
    pub fn union(x: &Self, y: &Self, z: &Self, output: usize) -> Self {
        let nodes = [x.output, y.output, z.output];
        let inputs = x
            .inputs
            .iter()
            .merge(&y.inputs)
            .merge(&z.inputs)
            .dedup()
            .copied()
            .collect::<Vec<_>>();
        let nodes = x
            .nodes
            .iter()
            .merge(&y.nodes)
            .merge(&z.nodes)
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

    pub fn nodes(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.nodes.iter().map(|node| NodeIndex::new(*node))
    }
}

pub struct Mapper<'a> {
    max_cuts: usize,
    max_inputs: usize,
    lut_area: &'a [u32],
    lut_delay: &'a [&'a [i32]],
    wire_delay: i32,
    mig: &'a mig4::Mig,
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
        mig: &'a mig4::Mig,
    ) -> Self {
        assert!(!petgraph::algo::is_cyclic_directed(mig.graph()));

        let len = mig
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
            mig,
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
        let mut original_expr: String = String::new();
        let mut max_dep: usize = 0;
        let mut cur_dep: usize = 0;

        let mut stack: Vec<usize> = Vec::new();
        stack.push(cut.output);
        while let Some(node) = stack.last() {
            if *node == usize::MAX {
                cur_dep -= 1;
                original_expr.push_str(")");
                stack.pop(); // pop MAX flag
                let cur_root: usize = stack.pop().unwrap(); // pop M/A root
                let cur_expsi_odeg = expsi_odegs.get_mut(&cur_root).unwrap();
                cur_expsi_odeg.1 += 1; // visited again
                discovered.insert(cur_root, original_expr[cur_expsi_odeg.0..].to_string());
                continue;
            }
            if discovered.contains_key(node) {
                original_expr.push_str(" ");
                original_expr += discovered.get(node).unwrap();
                expsi_odegs.get_mut(&node).unwrap().1 += 1; // visited again
                stack.pop();
            } else {
                cur_dep += 1;
                max_dep = std::cmp::max(max_dep, cur_dep);
                expsi_odegs.insert(*node, (original_expr.len(), 1)); // record the starting index
                if *node == cut.output {
                    original_expr.push_str("(M");
                } else {
                    original_expr.push_str(" (M");
                }
                let (x_edge, y_edge, z_edge) = self
                    .mig
                    .try_unwrap_majority(NodeIndex::new(*node))
                    .expect("majority node with less than three inputs");
                stack.push(usize::MAX);
                let x = self.mig.edge_source(x_edge);
                stack.push(x.index());
                let y = self.mig.edge_source(y_edge);
                stack.push(y.index());
                let z = self.mig.edge_source(z_edge);
                stack.push(z.index());
            }
        }
        let mffc: usize = 1 + cut
            .nodes
            .iter()
            .filter(|node| {
                !cut.inputs.contains(node)
                    && **node != cut.output
                    && self.mig.output_degree(NodeIndex::new(**node))
                        <= expsi_odegs.get(&node).unwrap().1
            })
            .count();

        (mffc, max_dep, original_expr)
    }

    #[inline]
    pub fn rebuild_cut(
        &self,
        cut: &Cut,
        expr: &String,
    ) -> (
        (usize, bool),
        Vec<NodeIndex>,
        StableGraph<MigNode, bool, Directed>,
    ) {
        let const_false: usize = 0;
        let mut signal_stack: Vec<(usize, bool)> = Vec::new();

        let mut graph: StableGraph<MigNode, bool, Directed> = StableGraph::new();
        let inputs = cut
            .inputs()
            .into_iter()
            .map(|node| {
                if node.index() == 0 {
                    graph.add_node(MigNode::Zero)
                } else {
                    graph.add_node(self.mig.node_type(node))
                }
            })
            .collect::<Vec<NodeIndex>>();

        for start in expr.chars() {
            // skip space
            if start.is_whitespace() {
                continue;
            }

            if start == '(' {
                signal_stack.push((usize::MAX, false));
            } else if start == ')' {
                // 1. collect children
                let mut children: Vec<(usize, bool)> = Vec::new();
                while !signal_stack.is_empty() {
                    let cid = signal_stack.pop().unwrap();
                    if cid.0 == usize::MAX {
                        break;
                    }
                    children.push(cid);
                }
                // 2. build node
                let num_ins = children.len();
                let mut new_node = (0, false);
                if num_ins == 1 {
                    // NOT
                    new_node = (children[0].0, !children[0].1);
                } else if num_ins == 2 { // AND
                     // new_node = ntk.create_and(children[0], children[1]);
                } else if num_ins == 3 {
                    // MAJ
                    // new_node = ntk.create_maj(children[0], children[1], children[2]);
                    let new_node = graph.add_node(MigNode::Majority);
                    for node in children {
                        graph.add_edge(NodeIndex::new(node.0), new_node, node.1);
                    }
                } else {
                    assert!(false);
                }
                signal_stack.push(new_node);
            } else {
                if start == '~' || start == 'M' || start == '&' {
                    // pass
                } else {
                    if start == '0' {
                        signal_stack.push((const_false, false));
                    } else if start == '1' {
                        signal_stack.push((const_false, true));
                    } else {
                        signal_stack
                            .push((inputs[(start as u8 - 'a' as u8) as usize].index(), false));
                    }
                }
            }
        }
        let new_root = signal_stack.last().unwrap();
        (*new_root, inputs, graph)
    }

    pub fn rewrite_expr(&self, expr: &String, ori_deps: Option<&[u32]>) -> (String, CCost) {
        let (best, inital_cost, best_cost) = simplify(&expr, ori_deps, None);
        (best.to_string(), best_cost)
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
                let area2 = self.exact_area_ref(cut);
                let area1 = self.exact_area_deref(cut);
                //assert_eq!(area1, area2);
                self.references = references;
                return area1;
            }
        }

        let area1 = self.exact_area_deref(cut);
        let area2 = self.exact_area_ref(cut);
        //assert_eq!(area1, area2);
        self.references = references;
        area1
    }

    fn exact_area_deref(&mut self, cut: &Cut) -> u32 {
        let mut area = self.lut_area[cut.input_count()];
        for input in &cut.inputs {
            if !self.mig.input_nodes().contains(&NodeIndex::new(*input))
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
            if !self.mig.input_nodes().contains(&NodeIndex::new(*input)) {
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
                let area2 = self.exact_edge_deref(cut);
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
            if !self.mig.input_nodes().contains(&NodeIndex::new(*input))
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
            if !self.mig.input_nodes().contains(&NodeIndex::new(*input)) {
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
        for node in self.mig.graph().node_indices() {
            self.depth[node.index()] = -1000;
            self.area_flow[node.index()] = 0.0;
            self.edge_flow[node.index()] = 0.0;
            self.references[node.index()] = 0;
        }

        for node in self.mig.input_nodes() {
            self.cuts[node.index()] = vec![Cut::single_node(node.index())];
            self.depth[node.index()] = 0;
            self.area_flow[node.index()] = 0.0;
            self.edge_flow[node.index()] = 0.0;
        }

        for node in self.mig.graph().externals(Outgoing) {
            if let mig4::MigNode::Output(node) = self.mig.graph()[node] {
                self.references[node as usize] = 1;
            }
        }

        let mut iter = petgraph::visit::Topo::new(self.mig.graph());

        let mut cut_count = 0;

        while let Some(node) = iter.next(&self.mig.graph()) {
            if let Some((x_edge, y_edge, z_edge)) = self.mig.try_unwrap_majority(node) {
                let x = self.mig.edge_source(x_edge);
                let y = self.mig.edge_source(y_edge);
                let z = self.mig.edge_source(z_edge);

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
                    .cartesian_product(&self.cuts[z.index()])
                    .map(|((x_cut, y_cut), z_cut)| Cut::union(x_cut, y_cut, z_cut, node.index()))
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
                let (mffc, max_dep, original_expr) = self.cut_dep_mffc_prefixexp(best_cut);
                let (aft_expr, cost) = self.rewrite_expr(
                    &original_expr,
                    Some(
                        best_cut
                            .inputs()
                            .map(|pi| self.depth[pi.index()] as u32)
                            .collect::<Vec<u32>>()
                            .as_slice(),
                    ),
                );
                self.rebuild_cut(best_cut, &aft_expr);

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
            .mig
            .graph()
            .externals(Outgoing)
            .flat_map(|output| self.mig.graph().neighbors_directed(output, Incoming))
            .filter(|node| self.mig.node_type(*node) == mig4::MigNode::Majority)
            .collect::<Vec<_>>();
        let mut mapping = Vec::new();
        let mut mapped_nodes = Vec::new();
        let input_nodes = self.mig.input_nodes();

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

        for node in self.mig.graph().node_indices() {
            self.required[node.index()] = self.max_depth;
        }

        let mut required_dfs = petgraph::visit::DfsPostOrder::empty(self.mig.graph());
        let pis = self.mig.input_nodes();
        for pi in pis {
            required_dfs.move_to(pi);
            while let Some(node) = required_dfs.next(self.mig.graph()) {
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
