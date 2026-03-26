use std::collections::HashMap;
use std::convert::TryFrom;
use std::io::Write;

use petgraph::{prelude::*, visit::EdgeRef};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MigNode {
    Input(u32),
    Output(u32),
    Zero,
    Majority,
}

#[derive(Default, Debug)]
pub struct Mig {
    graph: StableGraph<MigNode, bool, Directed>,
    input_symbols: HashMap<u32, String>,
    output_symbols: HashMap<u32, String>,
    zero: NodeIndex,
}

impl Mig {
    #[allow(clippy::similar_names)]
    #[must_use]
    pub fn from_aiger(path: &str) -> Self {
        let bytes = std::fs::read(path).unwrap();
        if bytes.starts_with(b"aig") {
            Self::from_aig_binary(&bytes)
        } else {
            Self::from_aag_ascii(path)
        }
    }

    fn init_with_m(m: usize) -> Self {
        let mut mig = Self::default();
        mig.zero = mig.graph.add_node(MigNode::Zero);
        assert_eq!(mig.zero.index(), 0);

        // Pre-emptively mark every variable as a majority gate.
        for variable in 1..=m {
            let node_index = mig.graph.add_node(MigNode::Majority);
            assert_eq!(node_index.index(), variable);
        }
        mig
    }

    fn from_aag_ascii(path: &str) -> Self {
        let file = std::fs::File::open(path).unwrap();
        let reader = aiger::Reader::from_reader(file).unwrap();

        let mut mig = Self::init_with_m(reader.header().m);

        let mut inputs = vec![];
        let mut outputs = vec![];

        for record in reader.records() {
            match record.unwrap() {
                aiger::Aiger::Input(l) => {
                    let index = u32::try_from(l.variable()).unwrap();
                    mig.graph[NodeIndex::new(l.variable())] = MigNode::Input(index);
                    inputs.push(index);
                }
                aiger::Aiger::Latch { .. } => todo!("latches"),
                aiger::Aiger::Output(l) => {
                    let pos = u32::try_from(outputs.len()).unwrap();
                    let output_node = mig.graph.add_node(MigNode::Output(pos));
                    mig.graph
                        .add_edge(NodeIndex::new(l.variable()), output_node, l.is_inverted());
                    outputs.push(pos);
                }
                aiger::Aiger::AndGate { output, inputs } => {
                    let input0 = inputs[0];
                    let input1 = inputs[1];

                    let gate = NodeIndex::new(output.variable());

                    mig.graph.add_edge(
                        NodeIndex::new(input0.variable()),
                        gate,
                        input0.is_inverted(),
                    );
                    mig.graph.add_edge(
                        NodeIndex::new(input1.variable()),
                        gate,
                        input1.is_inverted(),
                    );
                    mig.graph.add_edge(mig.zero, gate, false);
                }
                aiger::Aiger::Symbol {
                    type_spec,
                    position,
                    symbol,
                } => {
                    let index = match type_spec {
                        aiger::Symbol::Input => inputs[position],
                        aiger::Symbol::Output => outputs[position],
                        aiger::Symbol::Latch => continue,
                    };

                    match type_spec {
                        aiger::Symbol::Input => {
                            mig.input_symbols.insert(index, symbol);
                        }
                        aiger::Symbol::Output => {
                            mig.output_symbols.insert(index, symbol);
                        }
                        aiger::Symbol::Latch => {}
                    }
                }
            }
        }

        mig
    }

    fn from_aig_binary(bytes: &[u8]) -> Self {
        // Minimal AIGER binary parser for combinational circuits (L=0).
        // Format: "aig M I L O A\n" then L latch lines (ascii), O output lines (ascii literals),
        // then A AND gates in binary delta encoding, then optional symbol/comment section.
        let mut pos = 0usize;
        fn read_line<'a>(bytes: &'a [u8], pos: &mut usize) -> Option<&'a str> {
            if *pos >= bytes.len() {
                return None;
            }
            let start = *pos;
            while *pos < bytes.len() && bytes[*pos] != b'\n' {
                *pos += 1;
            }
            if *pos >= bytes.len() {
                return None;
            }
            let end = *pos;
            *pos += 1;
            std::str::from_utf8(&bytes[start..end]).ok()
        }

        let header = read_line(bytes, &mut pos).expect("missing AIG header");
        let mut it = header.split_whitespace();
        let magic = it.next().expect("missing magic");
        assert_eq!(magic, "aig", "expected binary aig header");
        let m: usize = it.next().unwrap().parse().unwrap();
        let i_cnt: usize = it.next().unwrap().parse().unwrap();
        let l_cnt: usize = it.next().unwrap().parse().unwrap();
        let o_cnt: usize = it.next().unwrap().parse().unwrap();
        let a_cnt: usize = it.next().unwrap().parse().unwrap();

        assert_eq!(l_cnt, 0, "binary AIG with latches not supported yet");

        let mut mig = Self::init_with_m(m);

        // Inputs are implicit: literals 2,4,...,2I correspond to variables 1..I.
        for v in 1..=i_cnt {
            mig.graph[NodeIndex::new(v)] = MigNode::Input(u32::try_from(v).unwrap());
        }

        // Outputs: ascii literals (possibly inverted).
        let mut output_pos = Vec::<u32>::new();
        for out_p in 0..o_cnt {
            let l = read_line(bytes, &mut pos).expect("missing output literal");
            let lit: usize = l.trim().parse().unwrap();
            let var = lit / 2;
            let inv = (lit & 1) == 1;
            let pos_id = u32::try_from(out_p).unwrap();
            let output_node = mig.graph.add_node(MigNode::Output(pos_id));
            mig.graph.add_edge(NodeIndex::new(var), output_node, inv);
            output_pos.push(pos_id);
        }

        // AND gates: delta encoding.
        fn read_uvarint(bytes: &[u8], pos: &mut usize) -> u64 {
            let mut x = 0u64;
            let mut s = 0u32;
            loop {
                let b = bytes.get(*pos).copied().expect("unexpected EOF in varint");
                *pos += 1;
                x |= u64::from(b & 0x7F) << s;
                if (b & 0x80) == 0 {
                    break;
                }
                s += 7;
            }
            x
        }

        // LHS literals start at 2*(I+1) and increase by 2.
        let mut lhs_lit = 2 * (i_cnt + 1);
        for _ in 0..a_cnt {
            let d0 = read_uvarint(bytes, &mut pos) as usize;
            let d1 = read_uvarint(bytes, &mut pos) as usize;
            let rhs0 = lhs_lit - d0;
            let rhs1 = rhs0 - d1;

            let lhs_var = lhs_lit / 2;
            let gate = NodeIndex::new(lhs_var);

            let rhs0_var = rhs0 / 2;
            let rhs1_var = rhs1 / 2;
            let rhs0_inv = (rhs0 & 1) == 1;
            let rhs1_inv = (rhs1 & 1) == 1;

            mig.graph.add_edge(NodeIndex::new(rhs0_var), gate, rhs0_inv);
            mig.graph.add_edge(NodeIndex::new(rhs1_var), gate, rhs1_inv);
            mig.graph.add_edge(mig.zero, gate, false);

            lhs_lit += 2;
        }

        // Parse optional symbol table (ascii) if present.
        while let Some(line) = read_line(bytes, &mut pos) {
            if line.starts_with('c') {
                break;
            }
            if line.len() < 2 {
                continue;
            }
            let (tag, rest) = line.split_at(1);
            let mut it = rest.splitn(2, ' ');
            let pos_str = it.next().unwrap_or("");
            let name = it.next().unwrap_or("").to_string();
            if name.is_empty() {
                continue;
            }
            let p: usize = pos_str.trim().parse().unwrap_or(usize::MAX);
            match tag {
                "i" => {
                    if p < i_cnt {
                        let var = p + 1;
                        mig.input_symbols.insert(var as u32, name);
                    }
                }
                "o" => {
                    if p < output_pos.len() {
                        mig.output_symbols.insert(output_pos[p], name);
                    }
                }
                _ => {}
            }
        }

        mig
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn to_graphviz(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;

        writeln!(f, "strict digraph {{")?;

        for node in self.graph.node_indices() {
            match self.graph[node] {
                MigNode::Input(index) => {
                    writeln!(
                        f,
                        "{} [shape=box,color=blue,label=\"Input {} ({})\"];",
                        node.index(),
                        node.index(),
                        self.input_symbols
                            .get(&index)
                            .map(String::as_str)
                            .unwrap_or("")
                    )?;
                }
                MigNode::Output(index) => {
                    writeln!(
                        f,
                        "{} [shape=box,color=green,label=\"Output {} ({})\"];",
                        node.index(),
                        node.index(),
                        self.output_symbols
                            .get(&index)
                            .map(String::as_str)
                            .unwrap_or("")
                    )?;
                }
                MigNode::Zero => {}
                MigNode::Majority => {
                    let (x_edge, y_edge, z_edge) = self
                        .try_unwrap_majority(node)
                        .expect("majority node with less than three inputs");
                    let (x, _) = self.graph().edge_endpoints(x_edge).unwrap();
                    let (y, _) = self.graph().edge_endpoints(y_edge).unwrap();
                    let (z, _) = self.graph().edge_endpoints(z_edge).unwrap();
                    let x_is_inverted = *self
                        .graph()
                        .edge_weight(x_edge)
                        .expect("edge from x to node has no weight");
                    let y_is_inverted = *self
                        .graph()
                        .edge_weight(y_edge)
                        .expect("edge from y to node has no weight");
                    let z_is_inverted = *self
                        .graph()
                        .edge_weight(z_edge)
                        .expect("edge from z to node has no weight");
                    if (x == self.zero && x_is_inverted)
                        || (y == self.zero && y_is_inverted)
                        || (z == self.zero && z_is_inverted)
                    {
                        writeln!(f, "{} [label=\"OR {0}\"];", node.index())?;
                    } else if (x == self.zero && !x_is_inverted)
                        || (y == self.zero && !y_is_inverted)
                        || (z == self.zero && !z_is_inverted)
                    {
                        writeln!(f, "{} [label=\"AND {0}\"];", node.index())?;
                    } else {
                        writeln!(f, "{} [label=\"Majority {0}\"];", node.index())?;
                    }
                }
            }
        }

        for edge in self.graph.edge_indices() {
            let (to, from) = self.graph.edge_endpoints(edge).unwrap();

            match self.graph[to] {
                MigNode::Input(_) | MigNode::Output(_) | MigNode::Majority => {
                    write!(f, "{} -> {}", to.index(), from.index())?;
                }
                MigNode::Zero => {
                    writeln!(f, "z{} [label=\"\", shape=point];", from.index())?;
                    write!(f, "z{} -> {0}", from.index())?;
                }
            }

            writeln!(
                f,
                " {};",
                if self.graph[edge] {
                    "[dir=both,arrowtail=odot]"
                } else {
                    ""
                }
            )?;
        }

        writeln!(f, "}}")
    }

    pub(crate) const fn graph(&self) -> &StableGraph<MigNode, bool> {
        &self.graph
    }

    pub(crate) fn graph_mut(&mut self) -> &mut StableGraph<MigNode, bool> {
        &mut self.graph
    }

    #[must_use]
    pub fn is_edge_inverted(&self, edge: EdgeIndex) -> bool {
        self.graph[edge]
    }

    #[must_use]
    pub fn edge_source(&self, edge: EdgeIndex) -> NodeIndex {
        self.graph.edge_endpoints(edge).unwrap().0
    }

    pub fn add_edge(&mut self, source: NodeIndex, target: NodeIndex, inverted: bool) -> EdgeIndex {
        self.graph.add_edge(source, target, inverted)
    }

    pub fn remove_edge(&mut self, edge: EdgeIndex) -> bool {
        self.graph
            .remove_edge(edge)
            .unwrap_or_else(|| panic!("tried to remove edge {} twice", edge.index()))
    }

    #[must_use]
    pub fn node_type(&self, node: NodeIndex) -> MigNode {
        self.graph[node]
    }

    pub fn add_node(&mut self, kind: MigNode) -> NodeIndex {
        self.graph.add_node(kind)
    }

    pub fn remove_node(&mut self, node: NodeIndex) -> MigNode {
        self.graph.remove_node(node).unwrap()
    }

    #[must_use]
    pub fn input_nodes(&self) -> Vec<NodeIndex> {
        self.graph().externals(Incoming).collect::<Vec<_>>()
    }

    #[must_use]
    pub fn output_edges(&self, node: NodeIndex) -> petgraph::stable_graph::Neighbors<bool, u32> {
        self.graph.neighbors_directed(node, Outgoing)
    }

    #[must_use]
    pub fn output_degree(&self, node: NodeIndex) -> usize {
        self.output_edges(node).count()
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    #[must_use]
    pub fn symbol_input(&self, var: u32) -> Option<&String> {
        self.input_symbols.get(&var)
    }

    #[must_use]
    pub fn symbol_output(&self, pos: u32) -> Option<&String> {
        self.output_symbols.get(&pos)
    }

    #[must_use]
    pub fn symbol(&self, index: u32) -> Option<&String> {
        self.input_symbols
            .get(&index)
            .or_else(|| self.output_symbols.get(&index))
    }

    #[must_use]
    pub fn try_unwrap_majority(
        &self,
        node: NodeIndex,
    ) -> Option<(EdgeIndex, EdgeIndex, EdgeIndex)> {
        match self.graph[node] {
            MigNode::Input(_) | MigNode::Output(_) | MigNode::Zero => None,
            MigNode::Majority => {
                let mut iter = self.graph.edges_directed(node, Incoming);
                match (iter.next(), iter.next(), iter.next()) {
                    (Some(x), Some(y), Some(z)) => {
                        if let Some(w) = iter.next() {
                            dbg!(w);
                            panic!(
                                "majority gate {} has {} inputs",
                                node.index(),
                                iter.count() + 4
                            );
                        }
                        Some((x.id(), y.id(), z.id()))
                    }
                    (Some(x), Some(y), None) => panic!(
                        "majority gate {} has two inputs: {} and {}",
                        node.index(),
                        x.id().index(),
                        y.id().index()
                    ),
                    (Some(x), None, None) => panic!(
                        "majority gate {} has one input: {}",
                        node.index(),
                        x.id().index()
                    ),
                    (None, None, None) => panic!("majority gate {} has zero inputs", node.index()),
                    (_, _, _) => unreachable!(),
                }
            }
        }
    }

    #[must_use]
    pub fn try_unwrap_and(&self, node: NodeIndex) -> Option<(EdgeIndex, EdgeIndex)> {
        let (x_edge, y_edge, z_edge) = self.try_unwrap_majority(node)?;
        let x = self.edge_source(x_edge);
        let y = self.edge_source(y_edge);
        let z = self.edge_source(z_edge);
        let x_is_inverted = self.is_edge_inverted(x_edge);
        let y_is_inverted = self.is_edge_inverted(y_edge);
        let z_is_inverted = self.is_edge_inverted(z_edge);

        if x == self.zero && !x_is_inverted {
            Some((y_edge, z_edge))
        } else if y == self.zero && !y_is_inverted {
            Some((x_edge, z_edge))
        } else if z == self.zero && !z_is_inverted {
            Some((x_edge, y_edge))
        } else {
            None
        }
    }

    #[must_use]
    pub fn try_unwrap_or(&self, node: NodeIndex) -> Option<(EdgeIndex, EdgeIndex)> {
        let (x_edge, y_edge, z_edge) = self.try_unwrap_majority(node)?;
        let x = self.edge_source(x_edge);
        let y = self.edge_source(y_edge);
        let z = self.edge_source(z_edge);
        let x_is_inverted = self.is_edge_inverted(x_edge);
        let y_is_inverted = self.is_edge_inverted(y_edge);
        let z_is_inverted = self.is_edge_inverted(z_edge);

        if x == self.zero && x_is_inverted {
            Some((y_edge, z_edge))
        } else if y == self.zero && y_is_inverted {
            Some((x_edge, z_edge))
        } else if z == self.zero && z_is_inverted {
            Some((x_edge, y_edge))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Mig, MigNode};

    #[test]
    fn try_unwrap_majority() {
        let mut mig = Mig::default();

        let node_input0 = mig.graph.add_node(MigNode::Input(0));
        let node_input1 = mig.graph.add_node(MigNode::Input(1));
        let node_input2 = mig.graph.add_node(MigNode::Input(2));
        let node_majority = mig.graph.add_node(MigNode::Majority);

        let edge_input0 = mig.graph.add_edge(node_input0, node_majority, false);
        let edge_input1 = mig.graph.add_edge(node_input1, node_majority, false);
        let edge_input2 = mig.graph.add_edge(node_input2, node_majority, false);

        // This is the best way I've got for order-independent comparison of
        // tuples
        let edges = mig
            .try_unwrap_majority(node_majority)
            .expect("try_unwrap_majority should return Some(_) for a majority gate");
        let edges = [edges.0, edges.1, edges.2];

        assert!(edges.contains(&edge_input0));
        assert!(edges.contains(&edge_input1));
        assert!(edges.contains(&edge_input2));
    }

    #[test]
    fn try_unwrap_majority_not_a_majority() {
        let mut mig = Mig::default();

        let node_input = mig.graph.add_node(MigNode::Input(0));
        let node_output = mig.graph.add_node(MigNode::Output(1));
        let node_zero = mig.graph.add_node(MigNode::Zero);

        assert_eq!(mig.try_unwrap_majority(node_input), None);
        assert_eq!(mig.try_unwrap_majority(node_output), None);
        assert_eq!(mig.try_unwrap_majority(node_zero), None);
    }
}
