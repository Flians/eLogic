use egg::{Id, Language};
use std::cmp;
use std::collections::HashSet;
use std::ffi::CString;
use std::ops::Add;
use std::os::raw::c_char;

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
impl CCost {
    fn merge(a: &CCost, b: &CCost) -> CCost {
        CCost {
            dep: cmp::max(a.dep, b.dep),
            aom: a.aom + b.aom,
            inv: a.inv + b.inv,
        }
    }
}
impl Add for CCost {
    type Output = CCost;

    fn add(self, other: CCost) -> Self::Output {
        CCost {
            dep: self.dep + other.dep,
            aom: self.aom + other.aom,
            inv: self.inv + other.inv,
        }
    }
}
impl Add<&CCost> for &CCost {
    type Output = CCost;

    fn add(self, other: &CCost) -> Self::Output {
        CCost {
            dep: self.dep + other.dep,
            aom: self.aom + other.aom,
            inv: self.inv + other.inv,
        }
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

pub struct MIGCostFn_dsi<'a> {
    egraph: &'a CEGraph,
    visited: HashSet<egg::Id>,
    original_dep: &'a [u32],
}
impl<'a> MIGCostFn_dsi<'a> {
    pub fn new(graph: &'a CEGraph, vars_: &'a [u32]) -> Self {
        Self {
            egraph: graph,
            visited: HashSet::new(),
            original_dep: vars_,
        }
    }

    pub fn reset(&mut self) {
        self.visited.clear();
    }
}
impl<'a> egg::CostFunction<MIG> for MIGCostFn_dsi<'a> {
    type Cost = CCost;
    fn cost<C>(&mut self, enode: &MIG, mut costs: C) -> Self::Cost
    where
        C: FnMut(egg::Id) -> Self::Cost,
    {
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
        let cur_cost = Self::Cost {
            dep: op_depth,
            aom: op_area,
            inv: op_inv,
        };
        // let cid = &self.egraph.lookup(enode.clone()).unwrap();

        cur_cost
            + enode.fold(Default::default(), |sum, id| {
                Self::Cost::merge(&sum, &costs(id))
            })
        /*
        Self::Cost {
            dep: op_depth + enode.fold(0, |max, id| max.max(costs(id).dep)),
            aom: enode.fold(op_area, |sum, id| sum + costs(id).aom),
            inv: enode.fold(op_inv, |sum, id| sum + costs(id).inv),
        }
        */
    }
}

#[derive(Default)]
pub struct ConstantFold;
impl egg::Analysis<MIG> for ConstantFold {
    type Data = Option<(u8, egg::PatternAst<MIG>)>;
    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> egg::DidMerge {
        egg::merge_option(to, from, |a, b| {
            assert_eq!(a.0, b.0, "Merged non-equal constants");
            egg::DidMerge(false, false)
        })
    }

    fn make(egraph: &CEGraph, enode: &MIG) -> Self::Data {
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

pub struct MigcostFnLp;
#[cfg_attr(docsrs, doc(cfg(feature = "lp")))]
impl egg::LpCostFunction<MIG, ConstantFold> for MigcostFnLp {
    fn node_cost(&mut self, _egraph: &CEGraph, _eclass: egg::Id, _enode: &MIG) -> f64 {
        let op_depth = match _enode {
            MIG::Maj(..) => 1 as f64,
            MIG::Not(..) => 0 as f64,
            _ => 0 as f64,
        };
        op_depth
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
        unsafe fn simplify_mig(s: &str, vars: *const u32, size: usize) -> *mut CCost;
        fn get_vec_at(numbers: &Vec<u32>, index: usize) -> u32;
        fn less_than(cost1: &CCost, cost2: &CCost) -> bool;
        unsafe fn free_string(s: *mut c_char);
        unsafe fn free_ccost(cost: *mut CCost);
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

fn get_vec_at(numbers: &Vec<u32>, index: usize) -> u32 {
    numbers[index]
}

fn less_than(cost1: &ffi::CCost, cost2: &ffi::CCost) -> bool {
    cost1 < cost2
}

fn free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

fn free_ccost(cost: *mut ffi::CCost) {
    if !cost.is_null() {
        unsafe {
            let _ = Box::from_raw(cost);
        }
    }
}

pub fn simplify_mig(s: &str, vars: *const u32, size: usize) -> *mut ffi::CCost {
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

    // use an Extractor to pick the best element of the root eclass
    let (best_cost, best) =
        egg::Extractor::new(&runner.egraph, MIGCostFn_dsi::new(&runner.egraph, &vars_))
            .find_best(root);
    let cost: ffi::CCost = ffi::CCost {
        aft_expr: best.to_string(),
        aft_dep: best_cost.dep,
        aft_size: best_cost.aom,
        aft_invs: best_cost.inv,
    };

    Box::into_raw(Box::new(cost))
}

use egraph_serialize::*;
use ordered_float::NotNan;
use rustc_hash::FxHashMap;

pub(crate) type BuildHasher = fxhash::FxBuildHasher;

#[cfg(feature = "deterministic")]
mod hashmap {
    pub(crate) type HashMap<K, V> = indexmap::IndexMap<K, V>;
    pub(crate) type HashSet<K> = indexmap::IndexSet<K>;
}
#[cfg(not(feature = "deterministic"))]
mod hashmap {
    use super::BuildHasher;
    pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasher>;
    pub(crate) type HashSet<K> = hashbrown::HashSet<K, BuildHasher>;
}

pub type Cost = NotNan<f64>;
pub const INFINITY: Cost = unsafe { NotNan::new_unchecked(std::f64::INFINITY) };

#[derive(Debug)]
pub struct GreedyDagExtractor<'a, CF: egg::CostFunction<L>, L: Language, N: egg::Analysis<L>> {
    cost_function: CF,
    costs: hashmap::HashMap<Id, (CF::Cost, L)>,
    egraph: &'a egg::EGraph<L, N>,
}

struct CostSet {
    costs: FxHashMap<ClassId, Cost>,
    total: Cost,
    choice: NodeId,
}

use std::cmp::Ordering;
fn cmp<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    // None is high
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(a), Some(b)) => a.partial_cmp(b).unwrap(),
    }
}

impl<'a, CF, L, N> GreedyDagExtractor<'a, CF, L, N>
where
    CF: egg::CostFunction<L>,
    L: Language,
    N: egg::Analysis<L>,
{
    /// Create a new `Extractor` given an `EGraph` and a
    /// `CostFunction`.
    ///
    /// The extraction does all the work on creation, so this function
    /// performs the greedy search for cheapest representative of each
    /// eclass.
    pub fn new(egraph: &'a egg::EGraph<L, N>, cost_function: CF) -> Self {
        let costs = hashmap::HashMap::default();
        let mut extractor = GreedyDagExtractor {
            costs,
            egraph,
            cost_function,
        };
        extractor.find_costs();

        extractor
    }

    /// Find the cheapest (lowest cost) represented `RecExpr` in the
    /// given eclass.
    pub fn find_best(&self, eclass: Id) -> (CF::Cost, egg::RecExpr<L>) {
        let (cost, root) = self.costs[&self.egraph.find(eclass)].clone();
        let expr = root.build_recexpr(|id| self.find_best_node(id).clone());
        (cost, expr)
    }

    /// Find the cheapest e-node in the given e-class.
    pub fn find_best_node(&self, eclass: Id) -> &L {
        &self.costs[&self.egraph.find(eclass)].1
    }

    /// Find the cost of the term that would be extracted from this e-class.
    pub fn find_best_cost(&self, eclass: Id) -> CF::Cost {
        let (cost, _) = &self.costs[&self.egraph.find(eclass)];
        cost.clone()
    }

    fn node_total_cost(&mut self, node: &L) -> Option<CF::Cost> {
        let eg = &self.egraph;
        let has_cost = |id| self.costs.contains_key(&eg.find(id));
        /*
        let cid = &self.egraph.lookup(node.clone()).unwrap();
        if has_cost(*cid) {
            return Some(self.costs[&eg.find(*cid)].0.clone());
        }
        */
        if node.all(has_cost) {
            let costs = &self.costs;
            let cost_f = |id| costs[&eg.find(id)].0.clone();
            Some(self.cost_function.cost(node, cost_f))
        } else {
            None
        }
    }

    fn find_costs(&mut self) {
        let mut did_something = true;
        while did_something {
            did_something = false;

            for class in self.egraph.classes() {
                let pass = self.make_pass(class);
                match (self.costs.get(&class.id), pass) {
                    (None, Some(new)) => {
                        self.costs.insert(class.id, new);
                        did_something = true;
                    }
                    (Some(old), Some(new)) if new.0 < old.0 => {
                        self.costs.insert(class.id, new);
                        did_something = true;
                    }
                    _ => (),
                }
            }
        }

        for class in self.egraph.classes() {
            if !self.costs.contains_key(&class.id) {
                log::warn!(
                    "Failed to compute cost for eclass {}: {:?}",
                    class.id,
                    class.nodes
                )
            }
        }
    }

    fn make_pass(&mut self, eclass: &egg::EClass<L, N::Data>) -> Option<(CF::Cost, L)> {
        let (cost, node) = eclass
            .iter()
            .map(|n| (self.node_total_cost(n), n))
            .min_by(|a, b| cmp(&a.0, &b.0))
            .unwrap_or_else(|| panic!("Can't extract, eclass is empty: {:#?}", eclass));
        cost.map(|c| (c, node.clone()))
    }

    /*
    pub fn extract(&self, egraph: &'a egg::EGraph<L, N>, _roots: &[Id]) -> ExtractionResult {
        let mut costs = FxHashMap::<ClassId, CostSet>::with_capacity_and_hasher(
            egraph.classes().len(),
            Default::default(),
        );

        let mut keep_going = true;

        let mut i = 0;
        while keep_going {
            i += 1;
            println!("iteration {}", i);
            keep_going = false;

            'node_loop: for cid in egraph.classes() {
                let mut cost_set = CostSet {
                    costs: Default::default(),
                    total: Cost::default(),
                    choice: cid.id,
                };

                // compute the cost set from the children
                for child in &cid {
                    let child_cid = egraph.nid_to_cid(child);
                    if let Some(child_cost_set) = costs.get(child_cid) {
                        // prevent a cycle
                        if child_cost_set.costs.contains_key(cid) {
                            continue 'node_loop;
                        }
                        cost_set.costs.extend(child_cost_set.costs.clone());
                    } else {
                        continue 'node_loop;
                    }
                }

                // add this node
                cost_set.costs.insert(cid.clone(), node.cost);

                cost_set.total = cost_set.costs.values().sum();

                // if the cost set is better than the current one, update it
                if let Some(old_cost_set) = costs.get(cid) {
                    if cost_set.total < old_cost_set.total {
                        costs.insert(cid.clone(), cost_set);
                        keep_going = true;
                    }
                } else {
                    costs.insert(cid.clone(), cost_set);
                    keep_going = true;
                }
            }
        }

        let mut result = ExtractionResult::default();
        for (cid, cost_set) in costs {
            result.choose(cid, cost_set.choice);
        }
        result
    }
    */
}

// use pyo3::prelude::*;
/// parse an expression, simplify it using egg, and pretty print it back out
// #[pyfunction]
pub fn simplify(s: &str) -> (String, CCost, CCost) {
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

    /*
    let mut visited: HashSet<&str> = HashSet::default();
    for node in expr.as_ref() {
        if let Some(var) = node.as_variable() {
            visited.insert(var);
        }
    }
    */

    let vars: [u32; 26] = [0; 26];
    let slice: &[u32] = &vars;

    // create an e-graph with the given expression
    let mut runner = egg::Runner::default().with_expr(&expr);
    // the Runner knows which e-class the expression given with `with_expr` is in
    let root = runner.roots[0];

    // use an Extractor to pick the best element of the root eclass
    let inital_cost =
        egg::Extractor::new(&runner.egraph, MIGCostFn_dsi::new(&runner.egraph, slice))
            .find_best_cost(root);

    // simplify the expression using a Runner, which runs the given rules over it
    runner = runner.run(all_rules);

    // use an Extractor to pick the best element of the root eclass
    let (best_cost, best) =
        GreedyDagExtractor::new(&runner.egraph, MIGCostFn_dsi::new(&runner.egraph, slice))
            .find_best(root);
    println!(
        "Simplified {} with cost {:?} to {} with cost {:?}",
        expr, inital_cost, best, best_cost
    );
    (best.to_string(), inital_cost, best_cost)
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
        simplify("(& 0 1)");
        simplify("(& x 1)");
        simplify("(& x (~ 1))");
        simplify("(& x (~ x))");
        simplify("(& x x)");
        simplify("(& (& x b) (& b y))");
        simplify("(M 1 1 1)");
        simplify("(M 1 1 0)");
        simplify("(M 1 0 0)");
        simplify("(M 0 0 0)");
        simplify("(M x 1 (~ 0))");
        simplify("(M a b (M a b c))");
        simplify("(M x 0 (M y 1 (M u 0 v)))");
        simplify("(M (M w x (~ z)) x (M z x y))");
        // simplify("(M x3 (M x3 x4 (M x5 x6 x7)) x1)");
        simplify("(M (~ 0) (M 0 c (~ (M 0 (M (~ 0) a b) (~ (M 0 a b))))) (M 0 (~ c) (M 0 (M (~ 0) a b) (~ (M 0 a b)))))");
        simplify("(M (~ 0) (M 0 (M 0 a c) (~ (M 0 (M (~ 0) b d) (~ (M 0 b d))))) (M 0 (~ (M 0 a c)) (M 0 (M (~ 0) b d) (~ (M 0 b d)))))");
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
}
