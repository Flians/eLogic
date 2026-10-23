use std::collections::HashMap;
use std::convert::TryFrom;
use std::io::Write;

use petgraph::{prelude::*, visit::EdgeRef};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AigNode {
    Input(u32),
    Output(u32),
    Zero,
    And,
}

#[derive(Clone, Default, Debug)]
pub struct Aig {
    graph: StableGraph<AigNode, bool, Directed>,
    input_symbols: HashMap<u32, String>,
    output_symbols: HashMap<u32, String>,
    zero: NodeIndex,
}

impl Aig {
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
        let mut aig = Self::default();
        aig.zero = aig.graph.add_node(AigNode::Zero);
        assert_eq!(aig.zero.index(), 0);

        for variable in 1..=m {
            let node_index = aig.graph.add_node(AigNode::And);
            assert_eq!(node_index.index(), variable);
        }
        aig
    }

    fn from_aag_ascii(path: &str) -> Self {
        let file = std::fs::File::open(path).unwrap();
        let reader = aiger::Reader::from_reader(file).unwrap();

        let mut aig = Self::init_with_m(reader.header().m);

        let mut inputs = vec![];
        let mut outputs = vec![];

        for record in reader.records() {
            match record.unwrap() {
                aiger::Aiger::Input(l) => {
                    let index = u32::try_from(l.variable()).unwrap();
                    aig.graph[NodeIndex::new(l.variable())] = AigNode::Input(index);
                    inputs.push(index);
                }
                aiger::Aiger::Latch { .. } => todo!("latches"),
                aiger::Aiger::Output(l) => {
                    let pos = u32::try_from(outputs.len()).unwrap();
                    let output_node = aig.graph.add_node(AigNode::Output(pos));
                    aig.graph
                        .add_edge(NodeIndex::new(l.variable()), output_node, l.is_inverted());
                    outputs.push(pos);
                }
                aiger::Aiger::AndGate { output, inputs } => {
                    let input0 = inputs[0];
                    let input1 = inputs[1];
                    let gate = NodeIndex::new(output.variable());

                    aig.graph.add_edge(
                        NodeIndex::new(input0.variable()),
                        gate,
                        input0.is_inverted(),
                    );
                    aig.graph.add_edge(
                        NodeIndex::new(input1.variable()),
                        gate,
                        input1.is_inverted(),
                    );
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
                            aig.input_symbols.insert(index, symbol);
                        }
                        aiger::Symbol::Output => {
                            aig.output_symbols.insert(index, symbol);
                        }
                        aiger::Symbol::Latch => {}
                    }
                }
            }
        }

        aig
    }

    fn from_aig_binary(bytes: &[u8]) -> Self {
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

        let mut aig = Self::init_with_m(m);

        for v in 1..=i_cnt {
            aig.graph[NodeIndex::new(v)] = AigNode::Input(u32::try_from(v).unwrap());
        }

        let mut output_pos = Vec::<u32>::new();
        for out_p in 0..o_cnt {
            let l = read_line(bytes, &mut pos).expect("missing output literal");
            let lit: usize = l.trim().parse().unwrap();
            let var = lit / 2;
            let inv = (lit & 1) == 1;
            let pos_id = u32::try_from(out_p).unwrap();
            let output_node = aig.graph.add_node(AigNode::Output(pos_id));
            aig.graph.add_edge(NodeIndex::new(var), output_node, inv);
            output_pos.push(pos_id);
        }

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

            aig.graph.add_edge(NodeIndex::new(rhs0_var), gate, rhs0_inv);
            aig.graph.add_edge(NodeIndex::new(rhs1_var), gate, rhs1_inv);

            lhs_lit += 2;
        }

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
                        aig.input_symbols.insert(var as u32, name);
                    }
                }
                "o" => {
                    if p < output_pos.len() {
                        aig.output_symbols.insert(output_pos[p], name);
                    }
                }
                _ => {}
            }
        }

        aig
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn to_graphviz(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;

        writeln!(f, "strict digraph {{")?;

        for node in self.graph.node_indices() {
            match self.graph[node] {
                AigNode::Input(index) => {
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
                AigNode::Output(index) => {
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
                AigNode::Zero => {}
                AigNode::And => {
                    writeln!(f, "{} [label=\"AND {}\"];", node.index(), node.index())?;
                }
            }
        }

        for edge in self.graph.edge_indices() {
            let (to, from) = self.graph.edge_endpoints(edge).unwrap();

            match self.graph[to] {
                AigNode::Input(_) | AigNode::Output(_) | AigNode::And => {
                    write!(f, "{} -> {}", to.index(), from.index())?;
                }
                AigNode::Zero => {
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

    pub(crate) const fn graph(&self) -> &StableGraph<AigNode, bool> {
        &self.graph
    }

    /// All node indices currently in the graph, in ascending index order.
    ///
    /// Includes primary inputs, the constant node and output nodes, so
    /// `node_indices().map(|n| n.index()).max() + 1` is the dense index bound used by the
    /// mapper's per-node vectors.
    pub fn node_indices(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.graph.node_indices()
    }

    pub(crate) fn graph_mut(&mut self) -> &mut StableGraph<AigNode, bool> {
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
    pub fn node_type(&self, node: NodeIndex) -> AigNode {
        self.graph[node]
    }

    pub fn add_node(&mut self, kind: AigNode) -> NodeIndex {
        self.graph.add_node(kind)
    }

    pub fn remove_node(&mut self, node: NodeIndex) -> AigNode {
        self.graph.remove_node(node).unwrap()
    }

    #[must_use]
    pub fn input_nodes(&self) -> Vec<NodeIndex> {
        self.graph().externals(Incoming).collect::<Vec<_>>()
    }

    #[must_use]
    pub fn output_edges(
        &self,
        node: NodeIndex,
    ) -> petgraph::stable_graph::Neighbors<'_, bool, u32> {
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

    /// Attach a name to a primary input. Needed when an [`Aig`] is built programmatically (e.g. by
    /// re-extracting a restructured e-graph) so that equivalence checking can still match ports.
    pub fn set_symbol_input(&mut self, var: u32, name: String) {
        self.input_symbols.insert(var, name);
    }

    /// Attach a name to a primary output.
    pub fn set_symbol_output(&mut self, pos: u32, name: String) {
        self.output_symbols.insert(pos, name);
    }

    #[must_use]
    pub fn try_unwrap_and(&self, node: NodeIndex) -> Option<(EdgeIndex, EdgeIndex)> {
        match self.graph[node] {
            AigNode::Input(_) | AigNode::Output(_) | AigNode::Zero => None,
            AigNode::And => {
                let mut iter = self.graph.edges_directed(node, Incoming);
                match (iter.next(), iter.next()) {
                    (Some(x), Some(y)) => {
                        if let Some(w) = iter.next() {
                            dbg!(w);
                            panic!("AND gate {} has {} inputs", node.index(), iter.count() + 3);
                        }
                        Some((x.id(), y.id()))
                    }
                    (Some(x), None) => {
                        panic!(
                            "AND gate {} has one input: {}",
                            node.index(),
                            x.id().index()
                        )
                    }
                    (None, None) => panic!("AND gate {} has zero inputs", node.index()),
                    _ => unreachable!(),
                }
            }
        }
    }

    pub fn cleanup_graph(&mut self) {
        let mut hash: HashMap<[(NodeIndex<u32>, bool); 2], NodeIndex> = HashMap::new();

        let node_indices = self.graph().node_indices().collect::<Vec<_>>();
        for node in node_indices {
            if let Some((x_edge, y_edge)) = self.try_unwrap_and(node) {
                let x = self.edge_source(x_edge);
                let y = self.edge_source(y_edge);
                let x_is_inverted = self.is_edge_inverted(x_edge);
                let y_is_inverted = self.is_edge_inverted(y_edge);

                let mut children = [(x, x_is_inverted), (y, y_is_inverted)];
                children.sort_unstable_by_key(|(src, inv)| (src.index(), *inv));

                if let Some(dest) = hash.get(&children) {
                    let mut outputs = self.output_edges(node).detach();
                    while let Some((edge, output)) = outputs.next(self.graph()) {
                        let inverted = self.remove_edge(edge);
                        self.add_edge(*dest, output, inverted);
                    }
                    continue;
                }

                hash.insert(children, node);
            }
        }

        let mut did_something = true;
        while did_something {
            let nodes = self.node_count();

            *self.graph_mut() = self.graph().filter_map(
                |node, kind| {
                    if *kind == AigNode::And
                        && self.graph().neighbors_directed(node, Outgoing).count() == 0
                    {
                        None
                    } else {
                        Some(*kind)
                    }
                },
                |_edge, kind| Some(*kind),
            );

            did_something = self.node_count() < nodes;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Aig, AigNode};

    #[test]
    fn try_unwrap_and() {
        let mut aig = Aig::default();

        let node_input0 = aig.graph.add_node(AigNode::Input(0));
        let node_input1 = aig.graph.add_node(AigNode::Input(1));
        let node_and = aig.graph.add_node(AigNode::And);

        let edge_input0 = aig.graph.add_edge(node_input0, node_and, false);
        let edge_input1 = aig.graph.add_edge(node_input1, node_and, false);

        let edges = aig
            .try_unwrap_and(node_and)
            .expect("try_unwrap_and should return Some(_) for an AND gate");
        let edges = [edges.0, edges.1];

        assert!(edges.contains(&edge_input0));
        assert!(edges.contains(&edge_input1));
    }

    #[test]
    fn try_unwrap_and_not_an_and() {
        let mut aig = Aig::default();

        let node_input = aig.graph.add_node(AigNode::Input(0));
        let node_output = aig.graph.add_node(AigNode::Output(1));
        let node_zero = aig.graph.add_node(AigNode::Zero);

        assert_eq!(aig.try_unwrap_and(node_input), None);
        assert_eq!(aig.try_unwrap_and(node_output), None);
        assert_eq!(aig.try_unwrap_and(node_zero), None);
    }
}
