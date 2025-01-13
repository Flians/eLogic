use egg::{Id, Language};
use std::ffi::CString;
use std::fmt;
use std::iter::Sum;
use std::ops::Add;
use std::ops::AddAssign;
use std::os::raw::c_char;
use std::{cmp, default};
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct CCost {
    dep: u32,
    aom: u32,
    inv: u32,
}

impl Default for CCost {
    fn default() -> Self {
        CCost {
            dep: 0,
            aom: 0,
            inv: 0,
        }
    }
}

impl fmt::Display for CCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dep={}, aom={}, inv={}", self.dep, self.aom, self.inv)
    }
}

impl CCost {
    pub fn merge(a: &CCost, b: &CCost) -> CCost {
        CCost {
            dep: a.dep.max(b.dep), // for add between children
            aom: a.aom + b.aom,
            inv: a.inv + b.inv,
        }
    }

    fn max() -> Self {
        CCost {
            dep: u32::MAX,
            aom: u32::MAX,
            inv: u32::MAX,
        }
    }

    const MASK: u64 = 0x1FFFFF;
    const SHIFT_X: u64 = 42;
    const SHIFT_Y: u64 = 21;

    pub fn encode(&self) -> f64 {
        let packed = ((self.dep as u64 & Self::MASK) << Self::SHIFT_X)
            | ((self.aom as u64 & Self::MASK) << Self::SHIFT_Y)
            | (self.inv as u64 & Self::MASK);

        f64::from_bits(packed)
    }

    pub fn decode(value: f64) -> Self {
        let packed = value.to_bits();

        let x = ((packed >> Self::SHIFT_X) & Self::MASK) as u32;
        let y = ((packed >> Self::SHIFT_Y) & Self::MASK) as u32;
        let z = (packed & Self::MASK) as u32;

        CCost {
            dep: x,
            aom: y,
            inv: z,
        }
    }
}

impl Add for CCost {
    type Output = CCost;

    fn add(self, other: CCost) -> Self::Output {
        CCost {
            dep: self.dep + other.dep, // for add between parent and child
            aom: self.aom + other.aom,
            inv: self.inv + other.inv,
        }
    }
}

impl Add<&CCost> for &CCost {
    type Output = CCost;

    fn add(self, other: &CCost) -> Self::Output {
        CCost {
            dep: self.dep + other.dep, // for add between parent and child
            aom: self.aom + other.aom,
            inv: self.inv + other.inv,
        }
    }
}

impl AddAssign for CCost {
    fn add_assign(&mut self, other: Self) {
        self.dep += other.dep; // for add between parent and child
        self.aom += other.aom;
        self.inv += other.inv;
    }
}

impl Sum for CCost {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(CCost::default(), |acc, cost| CCost::merge(&acc, &cost))
    }
}

impl<'a> Sum<&'a CCost> for CCost {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a CCost>,
    {
        iter.fold(CCost::default(), |acc, cost| CCost::merge(&acc, cost))
    }
}

impl PartialOrd for CCost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let src = (self.dep, self.aom, self.inv);
        let tar = (other.dep, other.aom, other.inv);
        Some(src.cmp(&tar))
    }

    fn lt(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(std::cmp::Ordering::Less)
    }
}
impl Ord for CCost {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

egg::define_language! {
    pub enum MIG {
        Bool(u8),
        "&" = And([Id; 2]),
        "M" = Maj([Id; 3]),
        "~" = Not(Id),
        Symbol(egg::Symbol),
    }
}

// count (prefix expression, operator size)
fn to_prefix(expr: &egg::RecExpr<MIG>, var_dep: &[u32]) -> (String, u32, u32, u32) {
    fn helper(
        expr: &egg::RecExpr<MIG>,
        id: Id,
        inv_count: &mut u32,
        ops_count: &mut u32,
        visited: &mut std::collections::HashMap<Id, (String, u32)>,
        current_depth: u32,
        max_depth: &mut u32,
        var_dep: &[u32],
    ) -> String {
        let node = &expr[id];

        let mut existed = false;
        let mut reupdate = false;
        let mut cur_dep = current_depth;
        let mut cur_exp = String::default();
        if let Some((expr_str, cached_depth)) = visited.get_mut(&id) {
            cur_dep = match node {
                MIG::Symbol(sym) => {
                    let chr_v = sym.as_str().as_bytes();
                    let index = (chr_v[0] as u8) - ('a' as u8);
                    current_depth + var_dep.get(index as usize).copied().unwrap_or(0)
                }
                _ => current_depth,
            };

            if *cached_depth < cur_dep {
                reupdate = true;
                *cached_depth = cur_dep;
                *max_depth = (*max_depth).max(cur_dep);
            }

            existed = true;
            cur_exp = expr_str.clone();
        };

        if existed {
            if reupdate {
                // continue to update children's depth
                match node {
                    MIG::And(children) => {
                        for &child_id in children.iter() {
                            helper(
                                expr,
                                child_id,
                                inv_count,
                                ops_count,
                                visited,
                                cur_dep + 1,
                                max_depth,
                                var_dep,
                            );
                        }
                    }
                    MIG::Maj(children) => {
                        for &child_id in children.iter() {
                            helper(
                                expr,
                                child_id,
                                inv_count,
                                ops_count,
                                visited,
                                cur_dep + 1,
                                max_depth,
                                var_dep,
                            );
                        }
                    }
                    MIG::Not(child_id) => {
                        helper(
                            expr, *child_id, inv_count, ops_count, visited, cur_dep, max_depth,
                            var_dep,
                        );
                    }
                    _ => {}
                }
            }

            return cur_exp;
        }

        let result = match node {
            MIG::Bool(value) => (format!("{}", value), current_depth),
            MIG::And(children) => {
                *ops_count += 1;
                let children_expr: Vec<String> = children
                    .iter()
                    .map(|&child_id| {
                        helper(
                            expr,
                            child_id,
                            inv_count,
                            ops_count,
                            visited,
                            current_depth + 1,
                            max_depth,
                            var_dep,
                        )
                    })
                    .collect();
                (
                    format!("({} {})", node.to_string(), children_expr.join(" ")),
                    current_depth,
                )
            }
            MIG::Maj(children) => {
                *ops_count += 1;
                let children_expr: Vec<String> = children
                    .iter()
                    .map(|&child_id| {
                        helper(
                            expr,
                            child_id,
                            inv_count,
                            ops_count,
                            visited,
                            current_depth + 1,
                            max_depth,
                            var_dep,
                        )
                    })
                    .collect();
                (
                    format!("({} {})", node.to_string(), children_expr.join(" ")),
                    current_depth,
                )
            }
            MIG::Not(child_id) => {
                *inv_count += 1;
                let child_expr = helper(
                    expr,
                    *child_id,
                    inv_count,
                    ops_count,
                    visited,
                    current_depth,
                    max_depth,
                    var_dep,
                );
                (
                    format!("({} {})", node.to_string(), child_expr),
                    current_depth,
                )
            }
            MIG::Symbol(sym) => {
                let chr_v = sym.as_str().as_bytes();
                let index = (chr_v[0] as u8) - ('a' as u8);
                let symbol_depth = var_dep.get(index as usize).copied().unwrap_or(0);
                (format!("{}", sym), current_depth + symbol_depth)
            }
        };

        *max_depth = (*max_depth).max(result.1);
        visited.insert(id, result.clone());
        result.0
    }

    let root_id = Id::from(expr.as_ref().len() - 1);
    let mut max_depth = 0;
    let mut ops_count = 0;
    let mut inv_count = 0;
    let mut visited = std::collections::HashMap::new();

    let prefix_expr = helper(
        expr,
        root_id,
        &mut inv_count,
        &mut ops_count,
        &mut visited,
        0,
        &mut max_depth,
        var_dep,
    );

    (prefix_expr, max_depth, ops_count, inv_count)
}

trait AsVariable {
    fn as_variable(&self) -> Option<&str>;
}

impl AsVariable for MIG {
    fn as_variable(&self) -> Option<&str> {
        match self {
            MIG::Symbol(name) => Some(name.as_str()),
            _ => None,
        }
    }
}

type CEGraph = egg::EGraph<MIG, ConstantFold>;
type CRewrite = egg::Rewrite<MIG, ConstantFold>;

#[derive(Default, Clone)]
pub struct ConstantFold;
impl egg::Analysis<MIG> for ConstantFold {
    type Data = Option<(u8, egg::PatternAst<MIG>)>;
    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> egg::DidMerge {
        egg::merge_option(to, from, |a, b| {
            assert_eq!(a.0, b.0, "Merged non-equal constants");
            egg::DidMerge(false, false)
        })
    }

    fn make(egraph: &mut CEGraph, enode: &MIG) -> Self::Data {
        let x = |i: &egg::Id| egraph[*i].data.as_ref().map(|c| c.0);
        let result = match enode {
            MIG::Bool(c) => Some((*c, c.to_string().parse().unwrap())),
            MIG::Symbol(_) => None,
            MIG::And([a, b]) => Some((
                (x(a)? & x(b)? == 1) as u8,
                format!("(& {} {})", x(a)?, x(b)?).parse().unwrap(),
            )),
            MIG::Maj([a, b, c]) => Some((
                (x(a)? + x(b)? + x(c)? > 1) as u8,
                format!("(M {} {} {})", x(a)?, x(b)?, x(c)?)
                    .parse()
                    .unwrap(),
            )),
            MIG::Not(a) => Some((1u8 ^ x(a)?, format!("(~ {})", x(a)?).parse().unwrap())),
        };
        // println!("Make: {:?} -> {:?}", enode, result);
        result
    }

    fn modify(egraph: &mut CEGraph, id: egg::Id) {
        if let Some(c) = egraph[id].data.clone() {
            egraph.union_instantiations(
                &c.1,
                &c.0.to_string().parse().unwrap(),
                &Default::default(),
                "analysis".to_string(),
            );
        }
    }
}

#[derive(Clone)]
pub struct MIGCostFn_dsi<'a> {
    egraph: &'a CEGraph,
    visited: std::collections::HashMap<egg::Id, CCost>,
    original_dep: &'a [u32],
}
impl<'a> MIGCostFn_dsi<'a> {
    pub fn new(graph: &'a CEGraph, vars_: &'a [u32]) -> Self {
        Self {
            egraph: graph,
            visited: std::collections::HashMap::default(),
            original_dep: vars_,
        }
    }

    pub fn reset(&mut self) {
        self.visited.clear();
    }

    pub fn cal_cur_cost_bystr(&self, enode: &str) -> CCost {
        let op_depth = match enode {
            "&" => 1 as u32,
            "M" => 1 as u32,
            "~" => 0 as u32,
            "0" => 0 as u32,
            "1" => 0 as u32,
            _ => {
                let chr_v = enode.as_bytes();
                let index = (chr_v[0] as u8) - ('a' as u8);
                self.original_dep[index as usize]
            }
        };
        let op_area = match enode {
            "&" => 1 as u32,
            "M" => 1 as u32,
            "~" => 0 as u32,
            _ => 0,
        };
        let op_inv = match enode {
            "&" => 0 as u32,
            "M" => 0 as u32,
            "~" => 1 as u32,
            _ => 0 as u32,
        };
        CCost {
            dep: op_depth,
            aom: op_area,
            inv: op_inv,
        }
    }

    pub fn cal_cur_cost(&self, enode: &MIG) -> CCost {
        let op_depth = match enode {
            MIG::And(..) => 1 as u32,
            MIG::Maj(..) => 1 as u32,
            MIG::Not(..) => 0 as u32,
            MIG::Symbol(v) => {
                let chr_v = v.as_str().as_bytes();
                let index = (chr_v[0] as u8) - ('a' as u8);
                self.original_dep[index as usize]
            }
            _ => 0 as u32,
        };
        let op_area = match enode {
            MIG::And(..) => 1 as u32,
            MIG::Maj(..) => 1 as u32,
            MIG::Not(..) => 0 as u32,
            _ => 0 as u32,
        };
        let op_inv = match enode {
            MIG::And(..) => 0 as u32,
            MIG::Maj(..) => 0 as u32,
            MIG::Not(..) => 1 as u32,
            _ => 0 as u32,
        };
        CCost {
            dep: op_depth,
            aom: op_area,
            inv: op_inv,
        }
    }
}
impl<'a> egg::CostFunction<MIG> for MIGCostFn_dsi<'a> {
    type Cost = CCost;
    fn cost<C>(&mut self, enode: &MIG, mut costs: C) -> Self::Cost
    where
        C: FnMut(egg::Id) -> Self::Cost,
    {
        let cur_cost = self.cal_cur_cost(enode);
        let new_cost = cur_cost
            + enode.fold(Default::default(), |sum, id| {
                Self::Cost::merge(&sum, &costs(id))
            });
        /*
        Self::Cost {
            dep: op_depth + enode.fold(0, |max, id| max.max(costs(id).dep)),
            aom: enode.fold(op_area, |sum, id| sum + costs(id).aom),
            inv: enode.fold(op_inv, |sum, id| sum + costs(id).inv),
        }
        */
        /*
        let cid = self.egraph.lookup(enode.clone()).unwrap();
        self.visited
            .entry(cid)
            .and_modify(|existing_cost| {
                if new_cost < *existing_cost {
                    *existing_cost = new_cost;
                }
            })
            .or_insert(new_cost)
            .to_owned()
        */
        new_cost
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "lp")))]
impl<'a> egg::LpCostFunction<MIG, ConstantFold> for MIGCostFn_dsi<'a> {
    fn node_cost(&mut self, _egraph: &CEGraph, _eclass: egg::Id, _enode: &MIG) -> f64 {
        match _enode {
            MIG::And(..) => 1 as f64,
            MIG::Maj(..) => 1 as f64,
            MIG::Not(..) => 0 as f64,
            _ => 0 as f64,
        }
    }
}

macro_rules! rule {
    ($name:ident, $left:literal, $right:literal) => {
        #[allow(dead_code)]
        fn $name() -> CRewrite {
            egg::rewrite!(stringify!($name); $left => $right)
        }
    };
    ($name:ident, $name2:ident, $left:literal, $right:literal) => {
        rule!($name, $left, $right);
        rule!($name2, $right, $left);
    };
}

rule! {neg_false, "(~ false)", "true"}
rule! {neg_true, "(~ true)", "false"}
rule! {true_false,     "1",                     "(~ 0)"                                                         }
rule! {false_true,     "0",                     "(~ 1)"                                                         }
rule! {double_neg,     double_neg_flip,         "(~ (~ ?a))",                 "?a"                              }
rule! {neg,            neg_flip,                "(~ (M ?a ?b ?c))",           "(M (~ ?a) (~ ?b) (~ ?c))"        }
rule! {distri,         distri_flip,             "(M ?a ?b (M ?c ?d ?e))",     "(M (M ?a ?b ?c) (M ?a ?b ?d) ?e)"}
rule! {com_associ,     com_associ_flip,         "(M ?a ?b (M ?c (~ ?b) ?d))", "(M ?a ?b (M ?c ?a ?d))"          }
rule! {relevance,      "(M ?a ?b (M ?c ?d (M ?a ?b ?e)))", "(M ?a ?b (M ?c ?d (M (~ ?b) ?b ?e)))"    }
rule! {associ,         "(M ?a ?b (M ?c ?b ?d))","(M ?d ?b (M ?c ?b ?a))"    }
rule! {comm_lm,        "(M ?a ?b ?c)",          "(M ?b ?a ?c)"              }
rule! {comm_lr,        "(M ?a ?b ?c)",          "(M ?c ?b ?a)"              }
rule! {comm_mr,        "(M ?a ?b ?c)",          "(M ?a ?c ?b)"              }
rule! {maj_2_equ,      "(M ?a ?b ?b)",          "?b"                        }
rule! {maj_2_com,      "(M ?a ?b (~ ?b))",      "?a"                        }

rule! {associ_and,     "(& ?a (& ?b ?c))",      "(& ?b (& ?a ?c))"       }
rule! {comm_and,       "(& ?a ?b)",             "(& ?b ?a)"              }
rule! {comp_and,       "(& ?a (~ ?a))",         "0"                      }
rule! {dup_and,        "(& ?a ?a)",             "?a"                     }
rule! {and_true,       "(& ?a 1)",              "?a"                     }
rule! {and_false,      "(& ?a 0)",              "0"                      }
// add (M ?a ?a ?b) => ?a
rule! {maj_dup,        "(M ?a ?a ?b)",          "?a"                     }

fn prove_something(name: &str, start: &str, rewrites: &[CRewrite], goals: &[&str]) {
    println!("Proving {}", name);

    let start_expr: egg::RecExpr<_> = start.parse().unwrap();
    let goal_exprs: Vec<egg::RecExpr<_>> = goals.iter().map(|g| g.parse().unwrap()).collect();

    let mut runner = egg::Runner::default()
        .with_iter_limit(20)
        .with_node_limit(5_000)
        .with_expr(&start_expr);

    // println!("r0: {:?}", runner.roots[0]);

    // we are assume the input expr is true
    // this is needed for the soundness of lem_imply
    let true_id = runner.egraph.add(MIG::Bool(1u8));
    let root = runner.roots[0];
    runner.egraph.union(root, true_id);
    runner.egraph.rebuild();

    let egraph = runner.run(rewrites).egraph;

    for (i, (goal_expr, goal_str)) in goal_exprs.iter().zip(goals).enumerate() {
        println!("Trying to prove goal {}: {}", i, goal_str);
        let equivs = egraph.equivs(&start_expr, goal_expr);
        if equivs.is_empty() {
            panic!("Couldn't prove goal {}: {}", i, goal_str);
        }
    }
}

#[cxx::bridge]
mod ffi {
    #[derive(Debug, Eq, PartialEq)]
    struct CCost {
        // bef_expr: String,
        // bef_dep: u32,
        // bef_size: u32,
        // bef_invs: u32,
        aft_expr: String,
        aft_dep: u32,
        aft_size: u32,
        aft_invs: u32,
    }

    extern "Rust" {
        unsafe fn simplify_depth(s: &str, vars: *const u32, size: usize) -> CCost;
        unsafe fn simplify_size(s: &str, vars: *const u32, size: usize) -> CCost;
        unsafe fn free_string(s: *mut c_char);
    }
}

impl std::fmt::Display for ffi::CCost {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "(aft_expr: {},  aft_dep: {}, aft_size: {}, aft_invs: {})",
            self.aft_expr, self.aft_dep, self.aft_size, self.aft_invs
        )
    }
}

impl PartialOrd for ffi::CCost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let src = (self.aft_dep, self.aft_size);
        let tar = (other.aft_dep, other.aft_size);
        Some(src.cmp(&tar))
    }

    fn lt(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(std::cmp::Ordering::Less)
    }
}

impl Ord for ffi::CCost {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

fn free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

pub fn simplify_depth(s: &str, vars: *const u32, size: usize) -> ffi::CCost {
    let all_rules = &[
        // rules needed for contrapositive
        double_neg(),
        double_neg_flip(),
        neg(),
        neg_flip(),
        // and some others
        distri(),
        distri_flip(),
        com_associ(),
        com_associ_flip(),
        // relevance(),
        associ(),
        comm_lm(),
        comm_lr(),
        comm_mr(),
        maj_2_equ(),
        maj_2_com(),
        associ_and(),
        comm_and(),
        comp_and(),
        dup_and(),
        and_true(),
        and_false(),
        maj_dup(),
    ];
    // parse the expression, the type annotation tells it which Language to use
    let bef_expr: egg::RecExpr<MIG> = s.parse().unwrap();

    let vars_default: [u32; 26] = [0; 26];
    let mut vars_: &[u32] = &vars_default;
    if size > 0 {
        vars_ = unsafe { std::slice::from_raw_parts(vars, size) };
    }

    // create an e-graph with the given expression
    let mut runner = egg::Runner::default().with_expr(&bef_expr);
    // the Runner knows which e-class the expression given with `with_expr` is in
    let root = runner.roots[0];

    // use an Extractor to pick the best element of the root eclass
    // (cost.bef_dep, cost.bef_size, cost.bef_invs) = Extractor::new(&runner.egraph, MIGCostFn_dsi::new()).find_best_cost(root);
    // cost.bef_expr = bef_expr.to_string();

    // simplify the expression using a Runner, which runs the given rules over it
    runner = runner.run(all_rules);

    // let lp_best = egg::LpExtractor::new(&runner.egraph, MIGCostFn_dsi::new(&runner.egraph, &vars_)).solve(root);
    // println!("\nlp result: {} ", lp_best);

    // use an Extractor to pick the best element of the root eclass
    let (best_cost, best) =
        egg::Extractor::new(&runner.egraph, MIGCostFn_dsi::new(&runner.egraph, &vars_))
            .find_best(root);
    let (aft_expr, mut aft_dep, aft_size, aft_invs) = to_prefix(&best, &vars_);
    if best_cost.dep != aft_dep {
        println!(
            "{} with {:?} to {} with {},{},{}",
            s, vars_, aft_expr, aft_dep, aft_size, aft_invs
        );
        // assert_eq!(best_cost.dep, aft_dep);
        aft_dep = best_cost.dep;
    }
    // let aft_expr = best.to_string();
    let cost: ffi::CCost = ffi::CCost {
        aft_expr: aft_expr,
        aft_dep: aft_dep,
        aft_size: aft_size,
        aft_invs: aft_invs,
    };

    cost
}

mod extract;
pub use extract::Extractor;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

pub fn egg_to_serialized_egraph(
    egraph: &CEGraph,
    cost_function: &MIGCostFn_dsi,
    root_id: egg::Id,
) -> egraph_serialize::EGraph {
    let mut out = egraph_serialize::EGraph::default();
    let final_root = egraph.find(root_id);

    // First pass: create all classes
    for class in egraph.classes() {
        out.class_data.insert(
            egraph_serialize::ClassId::from(format!("{}", class.id)),
            egraph_serialize::ClassData { typ: None },
        );
    }

    // Second pass: add nodes with simplified costs
    for class in egraph.classes() {
        for (i, node) in class.nodes.iter().enumerate() {
            // Use simpler cost model like reference
            let cost = cost_function.cal_cur_cost(&node);

            // Ensure all child classes exist
            for child in node.children() {
                if !out
                    .class_data
                    .contains_key(&egraph_serialize::ClassId::from(format!("{}", child)))
                {
                    out.class_data.insert(
                        egraph_serialize::ClassId::from(format!("{}", child)),
                        egraph_serialize::ClassData { typ: None },
                    );
                }
            }

            out.add_node(
                format!("{}.{}", class.id, i),
                egraph_serialize::Node {
                    op: node.to_string(),
                    children: node
                        .children()
                        .iter()
                        .map(|id| egraph_serialize::NodeId::from(format!("{}.0", id)))
                        .collect(),
                    eclass: egraph_serialize::ClassId::from(format!("{}", class.id)),
                    cost: egraph_serialize::Cost::new(cost.encode()).unwrap(),
                    subsumed: false,
                },
            )
        }
    }

    out.root_eclasses = vec![egraph_serialize::ClassId::from(format!("{}", final_root))];
    out
}

pub fn find_root_nodes(egraph: &CEGraph) -> Vec<Id> {
    let mut roots = Vec::new();
    let mut has_parent = std::collections::HashSet::new();

    // First collect all nodes (eclass) that are children
    for class in egraph.classes() {
        for node in &class.nodes {
            for child in node.children() {
                has_parent.insert(child);
            }
        }
    }
    // Then find nodes that aren't children of any other node
    // and contain a Maj node (since that's our root operation)
    for class in egraph.classes() {
        if !has_parent.contains(&class.id) {
            roots.push(class.id);
        }
    }
    // If no roots found, look for the highest-level Maj node
    if roots.is_empty() {
        // select the first eclass with Maj，or return null if nothing found
        for class in egraph.classes() {
            if class.nodes.iter().any(|n| matches!(n, MIG::Maj(_))) {
                roots.push(class.id);
                break;
            }
        }
    }
    roots
}

/// Save the serialized e-graph to a JSON file and include root e-class IDs.
fn save_serialized_egraph_to_json(
    serialized_egraph: &egraph_serialize::EGraph,
    file_path: &PathBuf,
    root_id: &usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(&file_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &serialized_egraph)?;

    let json_string = std::fs::read_to_string(&file_path)?;
    let mut json_data: serde_json::Value = serde_json::from_str(&json_string)?;
    // Store root_id as a vector
    json_data["root_eclasses"] = serde_json::json!([root_id.to_string()]);

    let file = File::create(&file_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &json_data)?;

    Ok(())
}

pub fn simplify_size(s: &str, vars: *const u32, size: usize) -> ffi::CCost {
    let all_rules = &[
        // rules needed for contrapositive
        double_neg(),
        double_neg_flip(),
        neg(),
        neg_flip(),
        // and some others
        distri(),
        distri_flip(),
        com_associ(),
        com_associ_flip(),
        // relevance(),
        associ(),
        comm_lm(),
        comm_lr(),
        comm_mr(),
        maj_2_equ(),
        maj_2_com(),
        associ_and(),
        comm_and(),
        comp_and(),
        dup_and(),
        and_true(),
        and_false(),
        // true_false(),
        // false_true(),
    ];
    // parse the expression, the type annotation tells it which Language to use
    let expr: egg::RecExpr<MIG> = s.parse().unwrap();

    let vars_default: [u32; 26] = [0; 26];
    let mut vars_: &[u32] = &vars_default;
    if size > 0 {
        vars_ = unsafe { std::slice::from_raw_parts(vars, size) };
    }

    // create an e-graph with the given expression
    let mut runner = egg::Runner::default()
        .with_expr(&expr)
        .with_iter_limit(1000)
        .with_node_limit(5000)
        .with_time_limit(std::time::Duration::from_secs(10));
    // the Runner knows which e-class the expression given with `with_expr` is in
    let root_id = runner.roots[0];

    // simplify the expression using a Runner, which runs the given rules over it
    runner = runner.run(all_rules);
    let saturated_egraph = runner.egraph;

    // Serialize the egraph to JSON with single root
    let serialized_egraph = egg_to_serialized_egraph(
        &saturated_egraph,
        &MIGCostFn_dsi::new(&saturated_egraph, vars_),
        root_id,
    );
    // let egraph_serialize_root = [egraph_serialize::ClassId::from(root_id.to_string())];

    // Extract the result
    // #[cfg(feature = "ilp-cbc")]
    // let extractor = extract::ilp_cbc::CbcExtractor::default();
    // let extractor = extract::faster_ilp_cbc::FasterCbcExtractor::default();
    // #[cfg(not(feature = "ilp-cbc"))]
    // let extractor = extract::bottom_up::BottomUpExtractor {};
    // let extractor = extract::global_greedy_dag::GlobalGreedyDagExtractor {};
    let extractor = extract::faster_greedy_dag::FasterGreedyDagExtractor {};
    let extraction_result = extractor.extract(&serialized_egraph, &serialized_egraph.root_eclasses);

    // Get the cost
    // let tree_cost = extraction_result.tree_cost(&serialized_egraph, &egraph_serialize_root);
    let dag_cost_size =
        extraction_result.dag_cost_size(&serialized_egraph, &serialized_egraph.root_eclasses);
    let dag_cost_depth =
        extraction_result.dag_cost_depth(&serialized_egraph, &serialized_egraph.root_eclasses);
    let aft_expr = extraction_result.print_aft_expr(&serialized_egraph);
    /*
    let (aft_expr, dag_cost) = extraction_result.print_extracted_term(
        &serialized_egraph,
        &MIGCostFn_dsi::new(&saturated_egraph, vars_),
    );
    */
    let cost: ffi::CCost = ffi::CCost {
        aft_expr: aft_expr,
        aft_dep: dag_cost_depth,
        aft_size: dag_cost_size,
        aft_invs: 0,
    };

    // println!("Depth: {}, Size: {}, term: {}", cost.aft_dep, cost.aft_size, aft_expr);

    cost
}

// use pyo3::prelude::*;
/// parse an expression, simplify it using egg, and pretty print it back out
// #[pyfunction]
pub fn simplify(s: &str, var_dep: &Vec<u32>) {
    let all_rules = &[
        // rules needed for contrapositive
        double_neg(),
        double_neg_flip(),
        neg(),
        neg_flip(),
        // and some others
        distri(),
        distri_flip(),
        com_associ(),
        com_associ_flip(),
        // relevance(),
        associ(),
        comm_lm(),
        comm_lr(),
        comm_mr(),
        maj_2_equ(),
        maj_2_com(),
        associ_and(),
        comm_and(),
        comp_and(),
        dup_and(),
        and_true(),
        and_false(),
        neg_false(),
        neg_true(),
        maj_dup(),
    ];
    // parse the expression, the type annotation tells it which Language to use
    let expr: egg::RecExpr<MIG> = s.parse().unwrap();

    let vars_default: [u32; 26] = [0; 26];
    let mut vars_ = vars_default.as_ptr();
    let mut var_len = 26;
    if !var_dep.is_empty() {
        vars_ = var_dep.as_ptr();
        var_len = var_dep.len();
    }

    let cost_depth = simplify_depth(s, vars_, var_len);
    println!("\ntree cost: {} ", cost_depth);

    // create an e-graph with the given expression
    let mut runner = egg::Runner::default()
        .with_expr(&expr)
        .with_iter_limit(1000)
        .with_node_limit(5000)
        .with_time_limit(std::time::Duration::from_secs(10));
    // the Runner knows which e-class the expression given with `with_expr` is in
    let root_id = runner.roots[0];

    // simplify the expression using a Runner, which runs the given rules over it
    runner = runner.run(all_rules);
    let saturated_egraph = runner.egraph;

    // Serialize the egraph to JSON with single root
    let serialized_egraph = egg_to_serialized_egraph(
        &saturated_egraph,
        &MIGCostFn_dsi::new(&saturated_egraph, unsafe {
            std::slice::from_raw_parts(vars_, var_len)
        }),
        root_id,
    );
    // let egraph_serialize_root = [egraph_serialize::ClassId::from(root_id.to_string())];

    // Extract the result
    //#[cfg(feature = "ilp-cbc")]
    //let extractor = extract::ilp_cbc::CbcExtractor::default();
    // Extract the result using global_greedy_dag extractor
    //#[cfg(not(feature = "ilp-cbc"))]
    // let extractor = extract::bottom_up::BottomUpExtractor {};
    // let extractor = extract::global_greedy_dag::GlobalGreedyDagExtractor {};
    let extractor = extract::faster_greedy_dag::FasterGreedyDagExtractor {};
    let extraction_result = extractor.extract(&serialized_egraph, &serialized_egraph.root_eclasses);

    // Get the cost
    // let tree_cost = extraction_result.tree_cost(&serialized_egraph, &egraph_serialize_root);
    let dag_cost_size =
        extraction_result.dag_cost_size(&serialized_egraph, &serialized_egraph.root_eclasses);
    let dag_cost_depth =
        extraction_result.dag_cost_depth(&serialized_egraph, &serialized_egraph.root_eclasses);
    let aft_expr = extraction_result.print_aft_expr(&serialized_egraph);
    println!(
        "DAG cost_faster_greedy: expr: {} , depth: {}, size: {}",
        aft_expr, dag_cost_depth, dag_cost_size
    );

    let extractor1 = extract::bottom_up::BottomUpExtractor {};
    let extraction_result1 =
        extractor1.extract(&serialized_egraph, &serialized_egraph.root_eclasses);

    let dag_cost_size1 =
        extraction_result1.dag_cost_size(&serialized_egraph, &serialized_egraph.root_eclasses);
    let dag_cost_depth1 =
        extraction_result1.dag_cost_depth(&serialized_egraph, &serialized_egraph.root_eclasses);
    let aft_expr1 = extraction_result1.print_aft_expr(&serialized_egraph);
    println!(
        "DAG cost_bottomup: expr: {}, depth: {}, size: {}",
        aft_expr1, dag_cost_depth1, dag_cost_size1
    );

    let extractor2 = extract::global_greedy_dag::GlobalGreedyDagExtractor {};
    let extraction_result2 =
        extractor2.extract(&serialized_egraph, &serialized_egraph.root_eclasses);

    let dag_cost_size2 =
        extraction_result2.dag_cost_size(&serialized_egraph, &serialized_egraph.root_eclasses);
    let dag_cost_depth2 =
        extraction_result2.dag_cost_depth(&serialized_egraph, &serialized_egraph.root_eclasses);

    let aft_expr2 = extraction_result2.print_aft_expr(&serialized_egraph);
    println!(
        "DAG cost_global_greedy: expr: {}, depth: {}, size: {}",
        aft_expr2, dag_cost_depth2, dag_cost_size2,
    );

    // let (aft_expr, tcost) = extraction_result.print_extracted_term(
    //     &serialized_egraph,
    //     &MIGCostFn_dsi::new(&saturated_egraph, vars_),
    // );
    // println!("Simplified {} to {} with cost {:?}", expr, aft_expr, tcost);
}

/*
// This function name should be same as your project name
#[pymodule]
fn mig_egg(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simplify, m)?)?;
    Ok(())
}
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prove_chain() {
        /*
        let rules = &[
            // rules needed for contrapositive
            double_neg(),
            double_neg_flip(),
            neg(),
            neg_flip(),
            // and some others
            distri(),
            distri_flip(),
            com_associ(),
            com_associ_flip(),
            associ(),
            comm_lm(),
            comm_lr(),
            comm_mr(),
            maj_2_equ(),
            maj_2_com(),
        ];
        prove_something(
            "chain",
            "(M x 0 (M y 1 (M u 0 v)))",
            rules,
            &["(M x 0 (M y x (M u 0 v)))", "(M (M y x 0) x (M 0 u v))"],
        );
        */
        let empty_vec: Vec<u32> = Vec::new();

        simplify("(& 0 1)", &empty_vec);
        simplify("(& x 1)", &empty_vec);
        simplify("(& x (~ 1))", &empty_vec);
        simplify("(& x (~ x))", &empty_vec);
        simplify("(& x x)", &empty_vec);
        simplify("(& (& x b) (& b y))", &empty_vec);
        simplify("(M 1 1 1)", &empty_vec);
        simplify("(M 1 1 0)", &empty_vec);
        simplify("(M 1 0 0)", &empty_vec);
        simplify("(M 0 0 0)", &empty_vec);
        simplify("(M x 1 (~ 0))", &empty_vec);
        simplify("(M a b (M a b c))", &empty_vec);
        simplify("(M x 0 (M y 1 (M u 0 v)))", &empty_vec); // need fix for ilp
        simplify("(M (M w x (~ z)) x (M z x y))", &empty_vec);
        simplify("(M c (M c d (M e f b)) a)", &empty_vec);
        simplify("(M (~ 0) (M 0 c (~ (M 0 (M (~ 0) a b) (~ (M 0 a b))))) (M 0 (~ c) (M 0 (M (~ 0) a b) (~ (M 0 a b)))))",&empty_vec);
        simplify("(M (~ 0) (M 0 (M 0 a c) (~ (M 0 (M (~ 0) b d) (~ (M 0 b d))))) (M 0 (~ (M 0 a c)) (M 0 (M (~ 0) b d) (~ (M 0 b d)))))",&empty_vec); // need fix for ilp
        simplify("(M 0 (~ (M 0 (~ a) b)) (M 0 c (~ d)))", &empty_vec);
        simplify(
            "(M (~ 0) (M 0 a (~ (M 0 b (~ c)))) (M 0 (~ a) (M 0 b (~ c))))",
            &empty_vec,
        );
        simplify("(M (~ 0) (M 0 a (~ b)) (M 0 (~ a) b))", &empty_vec);
        simplify("(M (~ 0) (M 0 a (~ b)) (M 0 (~ a) b))", &empty_vec);
        simplify("(M (~ 0) (M (~ e) (M (~ 0) c (M 0 (~ a) b)) (M 0 (M 0 c (M 0 (~ a) b)) (M (~ 0) c (M 0 (~ a) b)))) (M (M 0 (~ d) g) h (M 0 (~ f) h)))",&empty_vec);
        simplify("(M 0 b (~ (M 0 (~ (M g (M 0 d (M a c (~ f))) (M e (M a c (~ f)) g))) (M 0 (M (~ 0) d (M a c (~ f))) g))))",&empty_vec);
        simplify("(M (~ 0) (M 0 (M 0 c (~ (M (~ 0) (M 0 a (~ b)) (M 0 (~ a) b)))) h) (M (M 0 (~ c) d) (M 0 e (~ f)) (~ (M 0 (M 0 (~ c) d) g))))",&empty_vec);
        simplify("(M (~ 0) (M 0 (M 0 c (~ (M (~ 0) (M 0 a (~ b)) (M 0 (~ a) b)))) h) (M (M 0 (~ c) d) (M 0 e (~ f)) (~ (M 0 (M 0 (~ c) d) g))))", &vec![0, 0, 2, 2, 4, 6, 5, 7]);
    }

    #[test]
    fn const_fold() {
        let start = "(M 0 1 0)";
        let start_expr: egg::RecExpr<MIG> = start.parse().unwrap();
        let end = "0";
        let end_expr: egg::RecExpr<MIG> = end.parse().unwrap();
        let mut eg: CEGraph = egg::EGraph::default();
        eg.add_expr(&start_expr);
        eg.rebuild();
        assert!(!eg.equivs(&start_expr, &end_expr).is_empty());
    }

    #[test]
    fn test_depth() {
        /*
        let mut expr = egg::RecExpr::<MIG>::default();
        let a = expr.add(MIG::Symbol(egg::Symbol::from("a")));
        let b = expr.add(MIG::Symbol(egg::Symbol::from("b")));
        let c = expr.add(MIG::Symbol(egg::Symbol::from("c")));
        let d = expr.add(MIG::Symbol(egg::Symbol::from("d")));
        let e = expr.add(MIG::Symbol(egg::Symbol::from("e")));
        let f = expr.add(MIG::Symbol(egg::Symbol::from("f")));
        let g = expr.add(MIG::Symbol(egg::Symbol::from("g")));
        let h = expr.add(MIG::Symbol(egg::Symbol::from("h")));
        let zero = expr.add(MIG::Bool(0));

        let maj1 = expr.add(MIG::Maj([zero, a, b]));
        let not_a = expr.add(MIG::Not(a));
        let maj2 = expr.add(MIG::Maj([zero, not_a, b]));
        let not_e = expr.add(MIG::Not(e));
        let maj3 = expr.add(MIG::Maj([not_e, maj2, c]));
        let not_zero = expr.add(MIG::Not(zero));
        let maj4 = expr.add(MIG::Maj([not_zero, maj3, maj2]));
        let not_d = expr.add(MIG::Not(d));
        let maj5 = expr.add(MIG::Maj([zero, not_d, g]));
        let not_f = expr.add(MIG::Not(f));
        let maj6 = expr.add(MIG::Maj([zero, not_f, h]));
        let maj7 = expr.add(MIG::Maj([maj5, h, maj6]));
        let maj8 = expr.add(MIG::Maj([not_zero, maj4, maj7]));
        */

        let all_rules = &[
            // rules needed for contrapositive
            double_neg(),
            double_neg_flip(),
            neg(),
            neg_flip(),
            // and some others
            distri(),
            distri_flip(),
            com_associ(),
            com_associ_flip(),
            // relevance(),
            associ(),
            comm_lm(),
            comm_lr(),
            comm_mr(),
            maj_2_equ(),
            maj_2_com(),
            associ_and(),
            comm_and(),
            comp_and(),
            dup_and(),
            and_true(),
            and_false(),
            // true_false(),
            // false_true(),
            neg_false(),
            neg_true(),
            maj_dup(),
        ];

        let var_dep = vec![0, 0, 2, 5, 3, 4, 5]; // 对应 a, b, c, d, e, f, g
        let expr: egg::RecExpr<MIG> =
            "(M 0 b (~ (M 0 (~ (M g (M 0 d (M a c (~ f))) (M e (M a c (~ f)) g))) (M 0 (M (~ 0) d (M a c (~ f))) g))))"
                .parse()
                .unwrap();

        let var_dep = vec![0, 0, 0, 3, 6, 4, 5]; // 对应 a, b, c, d, e, f, g
        let expr: egg::RecExpr<MIG> =
            "(M (~ 0) d (M (~ (M c (M e (M a (~ f) (M 0 a (~ b))) (M a g (M 0 a (~ b)))) (M (~ e) (M 0 f (M 0 c (M 0 (~ a) b))) (M 0 (~ g) (M 0 c (M 0 (~ a) b)))))) (M c (M (~ 0) c (M 0 (~ a) b)) (~ (M e (~ f) g))) (M 0 (~ c) (M e (M a (~ f) (M 0 a (~ b))) (M a g (M 0 a (~ b)))))))"
                .parse()
                .unwrap();

        // create an e-graph with the given expression
        let mut runner = egg::Runner::default().with_expr(&expr);
        // the Runner knows which e-class the expression given with `with_expr` is in
        let root = runner.roots[0];

        // simplify the expression using a Runner, which runs the given rules over it
        runner = runner.run(all_rules);

        // use an Extractor to pick the best element of the root eclass
        let (best_cost, best) =
            egg::Extractor::new(&runner.egraph, MIGCostFn_dsi::new(&runner.egraph, &var_dep))
                .find_best(root);

        let (prefix_expr, depth, ops_count, inv_count) = to_prefix(&best, &var_dep);

        println!("cost: {:?}", best_cost);
        println!("前缀表达式: {}", prefix_expr);
        println!("表达式深度: {}", depth);
        println!("操作符总数: {}", ops_count);
        println!("反相器总数: {}", inv_count);
    }
}
