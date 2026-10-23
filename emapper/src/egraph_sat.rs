//! **S1 + S2** — build an e-graph from the AIG and saturate it with AIG-level identities, then
//! extract a restructured AIG.
//!
//! This is the controlled half of the v2 experiment: the *same* covering algorithm (S4/S5) runs on
//! the result, so any change in QoR is attributable to the e-graph's extra realisation space and
//! nothing else.
//!
//! # Representation
//!
//! The language is `Pi | 0 | 1 | ~x | x & y`. The flow doc's trap **T1** ("polarity is an e-node
//! attribute, not a separate `~` e-node") is only sound for *input* polarity: `And` here is a plain
//! two-input AND and inversion is its own e-node. That deviation is deliberate —
//! `~x & ~x → ~x` and `a & 1 → a` are not expressible when a class cannot name its own complement,
//! so a polarity-on-inputs-only language cannot host the leaf-eliminating rules this stage exists
//! for.
//!
//! # Why this rule set
//!
//! Commutativity and associativity of `&` are *not* included: a k-feasible cut already absorbs
//! them, because the cut's truth table is invariant under re-association and pin permutation.
//! What pays is the rules that remove logic — idempotence, complement, constants, absorption —
//! plus factoring, which is the one rule that changes the cut structure without changing size.

use std::collections::HashMap;
use std::time::Instant;

use egg::{define_language, rewrite as rw, Id, Language, RecExpr, Rewrite, Runner};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::{EdgeRef, Topo};
use petgraph::Direction;

use crate::aig_graph::{Aig, AigNode};
use crate::techlib::TechLib;

define_language! {
    pub enum AigLang {
        "0" = Zero,
        "1" = One,
        "~" = Not(Id),
        "&" = And([Id; 2]),
        Pi(u32),
        // a library cell instance: the symbol is the cell name, the children are its pins
        Cell(egg::Symbol, Vec<Id>),
    }
}

/// Compact logical AND with explicit input phases; this is not a library cell.
/// All original live ANDs remain actual counted enodes. Output complements
/// remain nameable through Not, so no conditional equivalence is introduced.
pub(crate) fn literal_inputs(node: &AigLang) -> Option<(Id, Id, u8)> {
    match node {
        AigLang::And([a,b]) => Some((*a,*b,0)),
        AigLang::Cell(symbol, children) if children.len()==2 => {
            let phase = match symbol.as_str() {
                "@egm_and_np" => 1, "@egm_and_pn" => 2, "@egm_and_nn" => 3,
                _ => return None,
            };
            Some((children[0],children[1],phase))
        }
        _ => None,
    }
}

/// Bounded exact three/four-input logical operator (historical helper name).
/// Full truth and every child are stored in the counted e-node; no hidden cone.
pub(crate) fn tt3_inputs(node:&AigLang)->Option<(&[Id],u64)>{
 if let AigLang::Cell(sym,args)=node{let (prefix,maximum)=match args.len(){3=>("@egm_tt3_",255),4=>("@egm_tt4_",65535),_=>return None};let tt=u64::from_str_radix(sym.as_str().strip_prefix(prefix)?,16).ok()?;if tt<=maximum{return Some((args,tt))}}None
}

/// Exact XOR/XNOR internal operator. Its operands are real counted classes,
/// not external boundary functions or library cells.
pub(crate) fn xor_inputs(node:&AigLang)->Option<(Id,Id,u8)>{
 if let AigLang::Cell(sym,args)=node{if args.len()==2{return match sym.as_str(){"@egm_xor"=>Some((args[0],args[1],0)),"@egm_xnor"=>Some((args[0],args[1],1)),_=>None}}}None
}

#[derive(Clone, Debug)]
pub struct SatParams {
    pub iter_limit: usize,
    pub node_limit: usize,
    /// Set to drop the factoring rule, to measure its contribution separately.
    pub use_factoring: bool,
    /// Include the `a & ~(~a & ~b) => a` absorption rule. It merges a class with one of its own
    /// subterms, which makes the e-graph cyclic; the class then has no acyclic realisation and
    /// S3 extraction can fail outright. Kept switchable so the cost of the rule is measurable.
    pub use_absorption: bool,
    /// Only accept the restructured AIG when it is strictly smaller. Turning this off admits
    /// equal-size restructurings, which is how the "does the e-graph add anything at equal cost?"
    /// case is measured.
    pub require_improvement: bool,
}

impl Default for SatParams {
    fn default() -> Self {
        Self {
            iter_limit: 16,
            node_limit: 20_000,
            use_factoring: true,
            use_absorption: true,
            require_improvement: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SatStats {
    pub nodes_before: usize,
    pub nodes_after: usize,
    /// `&` nodes only — the meaningful size measure (`nodes_*` also counts PIs, the constant and the
    /// output nodes).
    pub ands_before: usize,
    pub ands_after: usize,
    /// `false` when saturation found nothing and the original AIG was returned unchanged.
    pub improved: bool,
    /// E-classes left without a finite cost because the e-graph is cyclic. Absorption
    /// (`a & ~(~a & ~b) => a`) merges a class with one of its own subterms, which is legal for egg
    /// and makes the graph cyclic by construction.
    pub cyclic_classes: usize,
    pub eclasses: usize,
    pub enodes: usize,
    pub iters: usize,
    pub stop_reason: String,
    pub secs: f64,
}

/// The AIG-level identities that actually remove logic (plus factoring).
pub(crate) fn rules(use_factoring: bool, use_absorption: bool) -> Vec<Rewrite<AigLang, ()>> {
    let mut r = vec![
        rw!("inv-inv"; "(~ (~ ?a))" => "?a"),
        rw!("not-zero"; "(~ 0)" => "1"),
        rw!("not-one"; "(~ 1)" => "0"),
        rw!("and-idem"; "(& ?a ?a)" => "?a"),
        rw!("and-compl"; "(& ?a (~ ?a))" => "0"),
        rw!("and-one"; "(& ?a 1)" => "?a"),
        rw!("and-zero"; "(& ?a 0)" => "0"),
        rw!("and-comm"; "(& ?a ?b)" => "(& ?b ?a)"),
    ];
    if use_absorption {
        // a & ~(~a & ~b) = a & (a | b) = a
        r.push(rw!("and-absorb"; "(& ?a (~ (& (~ ?a) (~ ?b))))" => "?a"));
    }
    if use_factoring {
        // (a&b) & (a&c) = a & (b&c): exposes the shared term `a` to the matcher.
        r.push(rw!("and-factor"; "(& (& ?a ?b) (& ?a ?c))" => "(& ?a (& ?b ?c))"));
    }
    r
}

/// AIG -> e-graph input. Returns the flat expression plus `(output position, root)` pairs.
pub(crate) fn build_expr(aig: &Aig) -> Option<(RecExpr<AigLang>, Vec<(u32, Id)>)> {
    let mut expr = RecExpr::default();
    let mut id_of: HashMap<(usize, bool), Id> = HashMap::new();

    let mut order: Vec<NodeIndex> = Vec::new();
    let mut topo = Topo::new(aig.graph());
    while let Some(n) = topo.next(aig.graph()) {
        if !matches!(aig.node_type(n), AigNode::Output(_)) {
            order.push(n);
        }
    }

    let mut phase = |expr: &mut RecExpr<AigLang>,
                     id_of: &mut HashMap<(usize, bool), Id>,
                     i: usize,
                     base: Id| {
        id_of.insert((i, false), base);
        let neg = expr.add(AigLang::Not(base));
        id_of.insert((i, true), neg);
    };

    for n in order {
        let i = n.index();
        match aig.node_type(n) {
            AigNode::Zero => {
                let z = expr.add(AigLang::Zero);
                phase(&mut expr, &mut id_of, i, z);
            }
            AigNode::Input(var) => {
                let v = expr.add(AigLang::Pi(var));
                phase(&mut expr, &mut id_of, i, v);
            }
            AigNode::And => {
                let (e1, e2): (EdgeIndex, EdgeIndex) = aig.try_unwrap_and(n)?;
                let a = (aig.edge_source(e1).index(), aig.is_edge_inverted(e1));
                let b = (aig.edge_source(e2).index(), aig.is_edge_inverted(e2));
                let ca = *id_of.get(&a)?;
                let cb = *id_of.get(&b)?;
                let and = expr.add(AigLang::And([ca, cb]));
                phase(&mut expr, &mut id_of, i, and);
            }
            AigNode::Output(_) => unreachable!("outputs are filtered out of the walk"),
        }
    }

    let mut roots = Vec::new();
    for o in aig.graph().externals(Direction::Outgoing) {
        let pos = match aig.node_type(o) {
            AigNode::Output(p) => p,
            _ => continue,
        };
        let e = aig.graph().edges_directed(o, Direction::Incoming).next()?;
        let key = (e.source().index(), *e.weight());
        roots.push((pos, *id_of.get(&key)?));
    }
    roots.sort_by_key(|(p, _)| *p);
    Some((expr, roots))
}

/// RecExpr positions are not e-graph ids: hash-consing can collapse duplicate enodes on import.
/// Translate every position explicitly before recording any primary-output root.
pub(crate) fn import_aig(aig: &Aig) -> Option<(egg::EGraph<AigLang, ()>, Vec<(u32, Id)>)> {
    import_aig_admitted(aig, &mut crate::egraph_budget::NodeAdmission::new(usize::MAX))
}
pub(crate) fn import_aig_admitted(aig: &Aig, admission: &mut crate::egraph_budget::NodeAdmission) -> Option<(egg::EGraph<AigLang, ()>, Vec<(u32, Id)>)> {
    let (expr, roots) = build_expr(aig)?;
    let mut graph = egg::EGraph::default();
    let mut translated: Vec<Id> = Vec::with_capacity(expr.as_ref().len());
    for node in expr.as_ref() {
        translated.push(
            admission.import(&mut graph,
                node.clone().map_children(|child| translated[usize::from(child)]), "ordinary")?,
        );
    }
    graph.rebuild();
    let roots = roots
        .into_iter()
        .map(|(pos, id)| (pos, graph.find(translated[usize::from(id)])))
        .collect();
    Some((graph, roots))
}

/// Pick, for every e-class the cheapest e-node, where the cost is the number of `&` nodes
/// (inversions are free).
///
/// This deliberately replaces `egg::Extractor`: `Extractor::find_best` returns one `RecExpr` per
/// root, and **each such expression has its own `Id` space**. Keying a shared memo on those `Id`s
/// silently aliases classes between primary outputs, which is how the first version of this module
/// produced a non-equivalent netlist. Working on the e-graph's own (globally unique) class ids keeps
/// cross-output sharing intact and is correct by construction.
pub(crate) fn cheapest_nodes(
    egraph: &egg::EGraph<AigLang, ()>,
) -> std::collections::HashMap<Id, AigLang> {
    let mut cost: std::collections::HashMap<Id, usize> = std::collections::HashMap::new();
    let mut chosen: std::collections::HashMap<Id, AigLang> = std::collections::HashMap::new();
    loop {
        let mut changed = false;
        for class in egraph.classes() {
            let mut best = usize::MAX;
            let mut best_node: Option<AigLang> = None;
            for node in &class.nodes {
                /* Cost in **AIG nodes**, not in e-nodes: `~x` is an inverted edge and costs nothing,
                and so do constants and primary inputs. `AstSize` would charge 1 for every `~`,
                which makes the extractor prefer de-Morgan-expanded forms that are free in the
                netlist. */
                let mut c = if xor_inputs(node).is_some(){3}else if let Some((args,_))=tt3_inputs(node){args.len()-1}else{usize::from(literal_inputs(node).is_some())};
                let mut ok = true;
                for child in node.children() {
                    match cost.get(&egraph[*child].id) {
                        Some(v) => c = c.saturating_add(*v),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok && c < best {
                    best = c;
                    best_node = Some(node.clone());
                }
            }
            if let Some(bn) = best_node {
                if cost.get(&class.id).is_none_or(|v| best < *v) {
                    cost.insert(class.id, best);
                    chosen.insert(class.id, bn);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    chosen
}

pub(crate) fn cheapest_nodes_queued(
    egraph: &egg::EGraph<AigLang, ()>,
) -> std::collections::HashMap<Id, AigLang> {
    let mut cost: std::collections::HashMap<Id, usize> = std::collections::HashMap::new();
    let mut chosen: std::collections::HashMap<Id, AigLang> = std::collections::HashMap::new();
    let mut parents: std::collections::HashMap<Id, Vec<Id>> = std::collections::HashMap::new();
    let mut queue = std::collections::VecDeque::new();
    let mut queued = std::collections::HashSet::new();
    for class in egraph.classes() {
        queue.push_back(class.id);
        queued.insert(class.id);
        for node in &class.nodes {
            for child in node.children() {
                parents
                    .entry(egraph.find(*child))
                    .or_default()
                    .push(class.id);
            }
        }
    }
    while let Some(id) = queue.pop_front() {
        queued.remove(&id);
        let class = &egraph[id];
        let mut best = usize::MAX;
        let mut best_node = None;
        for node in &class.nodes {
            let mut value = if xor_inputs(node).is_some(){3}else if let Some((args,_))=tt3_inputs(node){args.len()-1}else{usize::from(literal_inputs(node).is_some())};
            let mut available = true;
            for child in node.children() {
                match cost.get(&egraph.find(*child)) {
                    Some(v) => value = value.saturating_add(*v),
                    None => {
                        available = false;
                        break;
                    }
                }
            }
            // MAX is a finite saturated cost, not an unreachable sentinel.
            if available && (best_node.is_none() || value < best) {
                best = value;
                best_node = Some(node.clone());
            }
        }
        if let Some(node) = best_node {
            if cost.get(&id).is_none_or(|v| best < *v) {
                cost.insert(id, best);
                chosen.insert(id, node);
                if let Some(users) = parents.get(&id) {
                    for &parent in users {
                        if queued.insert(parent) {
                            queue.push_back(parent);
                        }
                    }
                }
            }
        }
    }
    chosen
}

/// Rebuild an [`Aig`] from the chosen e-nodes, preserving primary input/output names.
struct Builder {
    aig: Aig,
    zero: NodeIndex,
    /// keyed by the e-graph's class id, which is unique across all primary outputs
    memo: HashMap<Id, (NodeIndex, bool)>,
}

impl Builder {
    fn new() -> Self {
        let mut aig = Aig::default();
        let zero = aig.add_node(AigNode::Zero);
        debug_assert_eq!(zero.index(), 0);
        Self {
            aig,
            zero,
            memo: HashMap::new(),
        }
    }

    fn pi(&mut self, var: u32) -> NodeIndex {
        for n in self.aig.graph().node_indices() {
            if self.aig.node_type(n) == AigNode::Input(var) {
                return n;
            }
        }
        self.aig.add_node(AigNode::Input(var))
    }

    /// `(node, inverted)` — the class's value is `node`'s output, complemented if `inverted`.
    fn emit(
        &mut self,
        egraph: &egg::EGraph<AigLang, ()>,
        chosen: &HashMap<Id, AigLang>,
        class: Id,
    ) -> Option<(NodeIndex, bool)> {
        if let Some(&r) = self.memo.get(&class) {
            return Some(r);
        }
        let node = chosen.get(&class)?.clone();
        let r = match node {
            AigLang::Zero => (self.zero, false),
            AigLang::One => (self.zero, true),
            AigLang::Pi(var) => (self.pi(var), false),
            AigLang::Not(x) => {
                let (n, inv) = self.emit(egraph, chosen, egraph[x].id)?;
                (n, !inv)
            }
            AigLang::And([x, y]) => {
                let (nx, ix) = self.emit(egraph, chosen, egraph[x].id)?;
                let (ny, iy) = self.emit(egraph, chosen, egraph[y].id)?;
                let n = self.aig.add_node(AigNode::And);
                self.aig.add_edge(nx, n, ix);
                self.aig.add_edge(ny, n, iy);
                (n, false)
            }
            // cells only exist in the S3 rule set; S1+S2 extracts an AIG
            AigLang::Cell(..) => return None,
        };
        self.memo.insert(class, r);
        Some(r)
    }
}

/// Saturate the AIG in an e-graph and re-extract the smallest equivalent AIG.
#[must_use]
pub fn restructure(aig: &Aig, ps: &SatParams) -> Option<(Aig, SatStats)> {
    let t0 = Instant::now();
    let mut admission=crate::egraph_budget::NodeAdmission::new(ps.node_limit);
    let (graph, roots) = import_aig_admitted(aig,&mut admission)?;
    let admission=std::sync::Arc::new(std::sync::Mutex::new(admission));
    let nodes_before = aig.graph().node_indices().count();
    let ands_before = aig
        .graph()
        .node_indices()
        .filter(|n| aig.node_type(*n) == AigNode::And)
        .count();

    let mut runner = Runner::<AigLang, (), ()>::default()
        .with_iter_limit(ps.iter_limit)
        .with_node_limit(ps.node_limit)
        .with_egraph(graph);
    // Track every translated primary-output root, including duplicate expressions.
    runner.roots = roots.iter().map(|(_, id)| *id).collect();
    let mut rewrites=rules(ps.use_factoring, ps.use_absorption);
    let _skips=crate::egraph_budget::reserve_rewrites(&mut rewrites,admission);
    let runner = runner.run(&rewrites);

    let stop_reason = format!("{:?}", runner.stop_reason);
    let chosen = cheapest_nodes(&runner.egraph);
    let cyclic_classes = runner
        .egraph
        .classes()
        .filter(|c| !chosen.contains_key(&c.id))
        .count();

    // Reconstruct, but fall back to the original AIG if any primary output has no acyclic
    // realisation. Reporting `None` here would silently drop the benchmark from the A/B.
    let mut out = Builder::new();
    let mut reconstructed = true;
    for (pos, id) in &roots {
        /* `chosen` is keyed by canonical class ids; a rewrite root id may have been remapped by
        union-find, so it must be canonicalised before the lookup. */
        match out.emit(&runner.egraph, &chosen, runner.egraph.find(*id)) {
            Some((node, inv)) => {
                let o = out.aig.add_node(AigNode::Output(*pos));
                out.aig.add_edge(node, o, inv);
            }
            None => {
                reconstructed = false;
                break;
            }
        }
    }
    if !reconstructed {
        return Some((
            aig.clone(),
            SatStats {
                nodes_before,
                nodes_after: nodes_before,
                ands_before,
                ands_after: ands_before,
                improved: false,
                cyclic_classes,
                eclasses: runner.egraph.number_of_classes(),
                enodes: runner.egraph.total_number_of_nodes(),
                iters: runner.iterations.len(),
                stop_reason: format!("{stop_reason}+extract-failed"),
                secs: t0.elapsed().as_secs_f64(),
            },
        ));
    }
    for n in aig.graph().node_indices() {
        if let AigNode::Input(var) = aig.node_type(n) {
            if let Some(name) = aig.symbol_input(var) {
                out.aig.set_symbol_input(var, name.clone());
            }
        }
    }
    for (pos, _) in &roots {
        if let Some(name) = aig.symbol_output(*pos) {
            out.aig.set_symbol_output(*pos, name.clone());
        }
    }
    out.aig.cleanup_graph();
    let nodes_after = out.aig.graph().node_indices().count();
    let ands_after = out
        .aig
        .graph()
        .node_indices()
        .filter(|n| out.aig.node_type(*n) == AigNode::And)
        .count();

    /* Only take the reconstruction when it is strictly smaller. `cheapest_nodes` ranks by *tree*
    cost, which double-counts shared sub-graphs, so the reconstructed AIG can legitimately come
    out larger (c2670: 716 -> 728). Turning that into a regression would corrupt the A/B. */
    let improved = ands_after < ands_before || !ps.require_improvement;
    if !improved {
        let mut stats = SatStats {
            nodes_before,
            nodes_after: nodes_before,
            ands_before,
            ands_after: ands_before,
            improved,
            cyclic_classes,
            eclasses: runner.egraph.number_of_classes(),
            enodes: runner.egraph.total_number_of_nodes(),
            iters: runner.iterations.len(),
            stop_reason: stop_reason.clone(),
            secs: t0.elapsed().as_secs_f64(),
        };
        stats.stop_reason = format!("{stop_reason}+no-improvement");
        return Some((aig.clone(), stats));
    }

    Some((
        out.aig,
        SatStats {
            nodes_before,
            nodes_after,
            ands_before,
            ands_after,
            improved,
            cyclic_classes,
            eclasses: runner.egraph.number_of_classes(),
            enodes: runner.egraph.total_number_of_nodes(),
            iters: runner.iterations.len(),
            stop_reason,
            secs: t0.elapsed().as_secs_f64(),
        },
    ))
}

// ---------------------------------------------------------------------------------------------
// S3 — one-way AIG -> Cell cover rules inside the e-graph
// ---------------------------------------------------------------------------------------------

/// Add a cell e-node for every library cell, rewriting the cell's AIG pattern into it.
///
/// **One-way only.** The reverse (`Cell -> AIG`) would let the extractor expand a cell back into
/// gates, which is precisely the resynthesis effect S3 is supposed to avoid creating at extraction
/// time; keeping the direction single also matches trap T6 in the flow doc.
fn cell_rules(tech: &TechLib) -> Vec<Rewrite<AigLang, ()>> {
    let mut r = Vec::new();
    for cell in &tech.cells {
        if cell.vars.is_empty() || cell.vars.len() > 6 || cell.name.starts_with('_') {
            continue;
        }
        let Some(pat) = crate::emap_egg::base_aig_pattern(cell) else {
            continue;
        };
        let Ok(lhs) = pat.parse::<egg::Pattern<AigLang>>() else {
            continue;
        };
        let args = (0..cell.vars.len())
            .map(|i| format!("?i{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let rhs_str = if args.is_empty() {
            format!("({})", cell.name)
        } else {
            format!("({} {})", cell.name, args)
        };
        let Ok(rhs) = rhs_str.parse::<egg::Pattern<AigLang>>() else {
            continue;
        };
        if let Ok(rw) = Rewrite::new(format!("cover-{}", cell.name), lhs, rhs) {
            r.push(rw);
        }
    }
    r
}

/// Realise every e-class by a **library cell**, never by a bare `&`.
///
/// This is the e-graph analogue of cut-based matching: a class carries both the AIG decomposition
/// and whatever cell e-nodes saturation put into it, and extraction must leave only cells behind.
/// `&` therefore costs a hard infinity. Costs are scaled to integers so that is representable.
fn cheapest_cells(
    egraph: &egg::EGraph<AigLang, ()>,
    area: &HashMap<String, usize>,
    inv_area: usize,
    and2_area: usize,
) -> HashMap<Id, AigLang> {
    const INF: usize = usize::MAX / 4;
    let cost_of = |node: &AigLang| -> usize {
        match node {
            AigLang::Cell(sym, _) => area.get(sym.as_str()).copied().unwrap_or(INF),
            AigLang::Not(_) => inv_area,
            /* A bare `&` is not a cell, but it *must* have a price: forbidding it outright makes
            any class without a single-cell cover unrealisable, and the failure cascades to the
            primary outputs (2 of 15 benchmarks). Pricing it at AND2 is exact whenever a 2-input
            AND exists, and the emitter then instantiates one. */
            AigLang::And(_) => and2_area,
            AigLang::Zero | AigLang::One | AigLang::Pi(_) => 0,
        }
    };
    let mut cost: HashMap<Id, usize> = HashMap::new();
    let mut chosen: HashMap<Id, AigLang> = HashMap::new();
    loop {
        let mut changed = false;
        for class in egraph.classes() {
            let mut best = INF;
            let mut best_node: Option<AigLang> = None;
            for node in &class.nodes {
                let mut c = cost_of(node);
                if c >= INF {
                    continue;
                }
                let mut ok = true;
                for child in node.children() {
                    match cost.get(&egraph[*child].id) {
                        Some(v) if *v < INF => c = c.saturating_add(*v),
                        _ => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok && c < best {
                    best = c;
                    best_node = Some(node.clone());
                }
            }
            if let Some(bn) = best_node {
                if cost.get(&class.id).is_none_or(|v| best < *v) {
                    cost.insert(class.id, best);
                    chosen.insert(class.id, bn);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    chosen
}

/// A cell-only netlist mapped directly off the saturated e-graph.
pub struct CellCover {
    pub verilog: String,
    pub model_area: f32,
    pub cells: usize,
    pub stats: SatStats,
}

struct CellEmitter<'a> {
    tech: &'a TechLib,
    inv_name: String,
    and2: String,
    body: Vec<String>,
    wires: Vec<String>,
    net: HashMap<Id, String>,
    counter: usize,
    model_area: f32,
}

impl CellEmitter<'_> {
    fn out_pin(&self, name: &str) -> Option<String> {
        self.tech
            .cells
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.out_pin.clone())
    }
    fn pins(&self, name: &str) -> Option<Vec<String>> {
        self.tech
            .cells
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.vars.clone())
    }
    fn area_of(&self, name: &str) -> f32 {
        self.tech
            .cells
            .iter()
            .find(|c| c.name == name)
            .map_or(0.0, |c| c.area)
    }

    fn instance(&mut self, cell: &str, args: &[String], out: &str) {
        let pins = self.pins(cell).unwrap_or_default();
        let op = self.out_pin(cell).unwrap_or_else(|| "Y".into());
        let mut conns: Vec<String> = Vec::new();
        for (i, a) in args.iter().enumerate() {
            let p = pins.get(i).cloned().unwrap_or_else(|| format!("P{i}"));
            conns.push(format!(".{p}({a})"));
        }
        conns.push(format!(".{op}({out})"));
        let inst = format!("g{}", self.counter);
        self.counter += 1;
        self.model_area += self.area_of(cell);
        self.body
            .push(format!("  {cell} {inst} ( {} );\n", conns.join(", ")));
    }

    fn emit(
        &mut self,
        egraph: &egg::EGraph<AigLang, ()>,
        chosen: &HashMap<Id, AigLang>,
        class: Id,
        pi: &HashMap<u32, String>,
    ) -> Option<String> {
        if let Some(n) = self.net.get(&class) {
            return Some(n.clone());
        }
        let node = chosen.get(&class)?.clone();
        let net = match node {
            AigLang::Zero => "1'b0".to_string(),
            AigLang::One => "1'b1".to_string(),
            AigLang::Pi(v) => pi.get(&v).cloned().unwrap_or_else(|| format!("i{v}")),
            AigLang::Not(x) => {
                let a = self.emit(egraph, chosen, egraph[x].id, pi)?;
                let name = format!("n{}", class);
                let inv = self.inv_name.clone();
                self.instance(&inv, &[a], &name);
                name
            }
            AigLang::Cell(sym, args) => {
                let mut as_ = Vec::with_capacity(args.len());
                for a in &args {
                    as_.push(self.emit(egraph, chosen, egraph[*a].id, pi)?);
                }
                let name = format!("n{}", class);
                let cell = sym.as_str().to_string();
                self.instance(&cell, &as_, &name);
                name
            }
            AigLang::And([x, y]) => {
                let a = self.emit(egraph, chosen, egraph[x].id, pi)?;
                let b = self.emit(egraph, chosen, egraph[y].id, pi)?;
                let name = format!("n{}", class);
                let cell = self.and2.clone();
                self.instance(&cell, &[a, b], &name);
                name
            }
        };
        if !net.starts_with("1'b") && !pi.values().any(|p| *p == net) {
            self.wires.push(net.clone());
        }
        self.net.insert(class, net.clone());
        Some(net)
    }
}

/// Map by putting cell matches **inside** the e-graph (EqMap's architecture) rather than matching
/// cells against cut truth tables outside it.
#[must_use]
pub fn map_with_cells(aig: &Aig, tech: &TechLib, ps: &SatParams) -> Option<CellCover> {
    let t0 = Instant::now();
    let mut admission=crate::egraph_budget::NodeAdmission::new(ps.node_limit);
    let (graph, roots) = import_aig_admitted(aig,&mut admission)?;
    let admission=std::sync::Arc::new(std::sync::Mutex::new(admission));
    let nodes_before = aig.graph().node_indices().count();
    let ands_before = aig
        .graph()
        .node_indices()
        .filter(|n| aig.node_type(*n) == AigNode::And)
        .count();

    let mut rules = rules(ps.use_factoring, ps.use_absorption);
    rules.extend(cell_rules(tech));
    let _skips=crate::egraph_budget::reserve_rewrites(&mut rules,admission);

    let mut runner = Runner::<AigLang, (), ()>::default()
        .with_iter_limit(ps.iter_limit)
        .with_node_limit(ps.node_limit)
        .with_egraph(graph);
    runner.roots = roots.iter().map(|(_, id)| *id).collect();
    let runner = runner.run(&rules);
    let stop_reason = format!("{:?}", runner.stop_reason);

    let area: HashMap<String, usize> = tech
        .cells
        .iter()
        .map(|c| (c.name.clone(), (f64::from(c.area) * 1000.0) as usize))
        .collect();
    let inv_area = (f64::from(tech.inv_area) * 1000.0) as usize;
    let and2 = (f64::from(tech.and2_area) * 1000.0) as usize;
    let chosen = cheapest_cells(&runner.egraph, &area, inv_area, and2);
    /* `cheapest_cells` can leave a class unrealisable: the saturated e-graph may contain a cycle of
    realisations (A's only cover needs B, B's needs A). mockturtle emap has the same hazard — its
    `--no_assert` flag exists precisely because the "everything is mapped" assert can fire — so
    this is reported rather than papered over. */
    if roots
        .iter()
        .any(|(_, id)| !chosen.contains_key(&runner.egraph.find(*id)))
    {
        eprintln!("s3: at least one primary output has no cell cover; giving up");
        return None;
    }

    use itertools::Itertools;
    let pi: Vec<(u32, String)> = aig
        .graph()
        .node_indices()
        .filter_map(|n| match aig.node_type(n) {
            AigNode::Input(var) => Some(var),
            _ => None,
        })
        .sorted_unstable()
        .map(|var| {
            (
                var,
                aig.symbol_input(var)
                    .cloned()
                    .unwrap_or_else(|| format!("i{var}")),
            )
        })
        .collect();
    let pi_map: HashMap<u32, String> = pi.iter().cloned().collect();

    let mut em = CellEmitter {
        tech,
        inv_name: tech.inv_cell.clone()?,
        and2: tech.and2_variant.as_ref().map(|v| v.cell_name.clone())?,
        body: Vec::new(),
        wires: Vec::new(),
        net: HashMap::new(),
        counter: 0,
        model_area: 0.0,
    };
    let mut assigns: Vec<(String, String)> = Vec::new();
    for (pos, id) in &roots {
        let net = em.emit(&runner.egraph, &chosen, runner.egraph.find(*id), &pi_map)?;
        let name = aig
            .symbol_output(*pos)
            .cloned()
            .unwrap_or_else(|| format!("o{pos}"));
        assigns.push((name, net));
    }
    let mut out = String::new();
    let mut ports: Vec<String> = pi.iter().map(|(_, n)| n.clone()).collect();
    ports.extend(assigns.iter().map(|(n, _)| n.clone()));
    out.push_str(&format!("module top({});\n", ports.join(", ")));
    for (_, n) in &pi {
        out.push_str(&format!("  input {n};\n"));
    }
    for (n, _) in &assigns {
        out.push_str(&format!("  output {n};\n"));
    }
    em.wires.sort();
    em.wires.dedup();
    for w in &em.wires {
        if !assigns.iter().any(|(n, _)| n == w) {
            out.push_str(&format!("  wire {w};\n"));
        }
    }
    for line in &em.body {
        out.push_str(line);
    }
    for (n, net) in &assigns {
        out.push_str(&format!("  assign {n} = {net};\n"));
    }
    out.push_str("endmodule\n");

    let secs = t0.elapsed().as_secs_f64();
    let cyclic = runner
        .egraph
        .classes()
        .filter(|c| !chosen.contains_key(&c.id))
        .count();
    Some(CellCover {
        model_area: em.model_area,
        cells: em.counter,
        verilog: out,
        stats: SatStats {
            nodes_before,
            nodes_after: nodes_before,
            ands_before,
            ands_after: ands_before,
            improved: false,
            cyclic_classes: cyclic,
            eclasses: runner.egraph.number_of_classes(),
            enodes: runner.egraph.total_number_of_nodes(),
            iters: runner.iterations.len(),
            stop_reason,
            secs,
        },
    })
}

/// Saturation limits from the environment (`EMAPPER_EG_ITER`, `EMAPPER_EG_NODES`).
#[must_use]
pub fn sat_params_from_env() -> SatParams {
    let mut ps = SatParams::default();
    if let Ok(v) = std::env::var("EMAPPER_EG_ITER") {
        if let Ok(n) = v.parse() {
            ps.iter_limit = n;
        }
    }
    if let Ok(v) = std::env::var("EMAPPER_EG_NODES") {
        if let Ok(n) = v.parse() {
            ps.node_limit = n;
        }
    }
    if std::env::var_os("EMAPPER_EG_NO_FACTOR").is_some() {
        ps.use_factoring = false;
    }
    if std::env::var_os("EMAPPER_EG_ALLOW_EQUAL").is_some() {
        ps.require_improvement = false;
    }
    if std::env::var_os("EMAPPER_EG_NO_ABSORB").is_some() {
        ps.use_absorption = false;
    }
    ps
}

#[cfg(test)]
mod overflow_witness_tests {
    use super::*;
    #[test]
    fn saturated_tree_cost_still_has_finite_dag_witness() {
        let mut graph = egg::EGraph::<AigLang, ()>::default();
        let mut root = graph.add(AigLang::Pi(0));
        for _ in 0..(usize::BITS + 10) {
            root = graph.add(AigLang::And([root, root]));
        }
        graph.rebuild();
        let chosen = cheapest_nodes_queued(&graph);
        assert_eq!(chosen.len(), graph.number_of_classes());
        assert!(chosen.contains_key(&graph.find(root)));
    }
}
