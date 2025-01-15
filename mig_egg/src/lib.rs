use colored::*;
use egg::{Id, Language};
use log::{info, warn};
use std::ffi::CString;
use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign};
use std::os::raw::c_char;
use std::{cmp, default};

// -----------------------------------------------------------------------------------
// 1. Define a cost struct (CCost) to keep track of depth (dep), area (aom), and inversions (inv).
//    It also includes methods for merging and for a custom f64 bit-encoding/decoding.
// -----------------------------------------------------------------------------------
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
    /// Merges two costs by taking the maximum depth and summing up the area and inversions.
    pub fn merge(a: &CCost, b: &CCost) -> CCost {
        CCost {
            dep: a.dep.max(b.dep),
            aom: a.aom + b.aom,
            inv: a.inv + b.inv,
        }
    }

    /// Returns a "maximum" cost.
    fn max() -> Self {
        CCost {
            dep: u32::MAX,
            aom: u32::MAX,
            inv: u32::MAX,
        }
    }

    /// Constants for bit-packing/unpacking.
    const MASK: u64 = 0x1FFFFF;
    const SHIFT_X: u64 = 42;
    const SHIFT_Y: u64 = 21;

    /// Encodes a CCost into f64 bits:
    /// bits: [21 bits for dep | 21 bits for aom | 21 bits for inv]
    pub fn encode(&self) -> f64 {
        let packed = ((self.dep as u64 & Self::MASK) << Self::SHIFT_X)
            | ((self.aom as u64 & Self::MASK) << Self::SHIFT_Y)
            | (self.inv as u64 & Self::MASK);
        f64::from_bits(packed)
    }

    /// Decodes an f64 back into a CCost struct.
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

    /// Addition merges depth by summation (used for parent + child)
    /// and sums area and inversions directly.
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

impl AddAssign for CCost {
    fn add_assign(&mut self, other: Self) {
        self.dep += other.dep;
        self.aom += other.aom;
        self.inv += other.inv;
    }
}

impl Sum for CCost {
    /// Custom summation that merges each CCost pair.
    /// For example, used for iterators of multiple costs.
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

// -----------------------------------------------------------------------------------
// 2. Define the MIG language used by egg::RecExpr
// -----------------------------------------------------------------------------------
egg::define_language! {
    /// MIG language definitions:
    ///   - Bool(u8)
    ///   - And([Id; 2])
    ///   - Maj([Id; 3])
    ///   - Not(Id)
    ///   - Symbol(egg::Symbol)
    pub enum MIG {
        Bool(u8),
        "&" = And([Id; 2]),
        "M" = Maj([Id; 3]),
        "~" = Not(Id),
        Symbol(egg::Symbol),
    }
}

// -----------------------------------------------------------------------------------
// 3. Convert a MIG expression to a prefix string and calculate depth, operator count,
//    and inversion count.
// -----------------------------------------------------------------------------------
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

        // If we have a cache hit in visited, update if new depth is deeper.
        if let Some((expr_str, cached_depth)) = visited.get_mut(&id) {
            match node {
                MIG::Symbol(sym) => {
                    let chr_v = sym.as_str().as_bytes();
                    let index = (chr_v[0] as u8) - ('a' as u8);
                    cur_dep = current_depth + var_dep.get(index as usize).copied().unwrap_or(0);
                }
                _ => {
                    cur_dep = current_depth;
                }
            }

            if *cached_depth < cur_dep {
                reupdate = true;
                *cached_depth = cur_dep;
                *max_depth = (*max_depth).max(cur_dep);
            }
            existed = true;
            cur_exp = expr_str.clone();
        }

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

        // We need to compute for a new or re-updated node:
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

        // Update global maximum depth, store in the visited map, then return.
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

// -----------------------------------------------------------------------------------
// 4. A helper trait to check if a MIG node is a variable (Symbol).
// -----------------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------------
// 5. Create type aliases for convenience
// -----------------------------------------------------------------------------------
type CEGraph = egg::EGraph<MIG, ConstantFold>;
type CRewrite = egg::Rewrite<MIG, ConstantFold>;

// -----------------------------------------------------------------------------------
// 6. The ConstantFold analysis: attempts to do constant folding for MIG expressions
// -----------------------------------------------------------------------------------
#[derive(Default, Clone)]
pub struct ConstantFold;

/// This implements partial constant folding.
/// For example, (M 0 1 0) becomes 0, etc.
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

// -----------------------------------------------------------------------------------
// 7. Define a cost function for MIG nodes that considers depth, area, and inversion count.
// -----------------------------------------------------------------------------------
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

    /// Calculate cost from a string-based operator (used in older references).
    pub fn cal_cur_cost_bystr(&self, enode: &str) -> CCost {
        let op_depth = match enode {
            "&" => 1,
            "M" => 1,
            "~" => 0,
            "0" => 0,
            "1" => 0,
            _ => {
                let chr_v = enode.as_bytes();
                let index = (chr_v[0] as u8) - b'a';
                self.original_dep[index as usize]
            }
        };
        let op_area = match enode {
            "&" => 1,
            "M" => 1,
            "~" => 0,
            _ => 0,
        };
        let op_inv = match enode {
            "&" => 0,
            "M" => 0,
            "~" => 1,
            _ => 0,
        };
        CCost {
            dep: op_depth,
            aom: op_area,
            inv: op_inv,
        }
    }

    /// Calculate cost for a MIG node, accounting for each variant (And, Maj, Not, Symbol, etc.)
    pub fn cal_cur_cost(&self, enode: &MIG) -> CCost {
        let op_depth = match enode {
            MIG::And(..) => 1,
            MIG::Maj(..) => 1,
            MIG::Not(..) => 0,
            MIG::Symbol(v) => {
                let chr_v = v.as_str().as_bytes();
                let index = (chr_v[0] as u8) - b'a';
                self.original_dep[index as usize]
            }
            _ => 0,
        };
        let op_area = match enode {
            MIG::And(..) => 1,
            MIG::Maj(..) => 1,
            MIG::Not(..) => 0,
            _ => 0,
        };
        let op_inv = match enode {
            MIG::And(..) => 0,
            MIG::Maj(..) => 0,
            MIG::Not(..) => 1,
            _ => 0,
        };
        CCost {
            dep: op_depth,
            aom: op_area,
            inv: op_inv,
        }
    }
}

/// Implementation for egg's CostFunction trait:
/// We sum children's costs using merge and add the parent's cost.
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
        new_cost
    }
}

/// Implementation for egg::LpCostFunction trait (used by ILP extraction).
#[cfg_attr(docsrs, doc(cfg(feature = "lp")))]
impl<'a> egg::LpCostFunction<MIG, ConstantFold> for MIGCostFn_dsi<'a> {
    fn node_cost(&mut self, _egraph: &CEGraph, _eclass: egg::Id, _enode: &MIG) -> f64 {
        match _enode {
            MIG::And(..) => 1.0,
            MIG::Maj(..) => 1.0,
            MIG::Not(..) => 0.0,
            _ => 0.0,
        }
    }
}

// -----------------------------------------------------------------------------------
// 8. Define rewrite rules (MIG rewrites). Some are forward and backward (like double_neg, etc.).
// -----------------------------------------------------------------------------------
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

rule! {neg_false,      "(~ 0)",                 "1"    }
rule! {neg_true,       "(~ 1)",                 "0"    }
rule! {true_false,     "1",                     "(~ 0)"}
rule! {false_true,     "0",                     "(~ 1)"}
rule! {double_neg,     double_neg_flip,         "(~ (~ ?a))",                 "?a"                              }
rule! {neg,            neg_flip,                "(~ (M ?a ?b ?c))",           "(M (~ ?a) (~ ?b) (~ ?c))"        }
rule! {distri,         distri_flip,             "(M ?a ?b (M ?c ?d ?e))",     "(M (M ?a ?b ?c) (M ?a ?b ?d) ?e)"}
rule! {com_associ,     com_associ_flip,         "(M ?a ?b (M ?c (~ ?b) ?d))", "(M ?a ?b (M ?c ?a ?d))"          }
rule! {relevance,      "(M ?a ?b (M ?c ?d (M ?a ?b ?e)))", "(M ?a ?b (M ?c ?d (M (~ ?b) ?b ?e)))"               }
rule! {associ,         "(M ?a ?b (M ?c ?b ?d))","(M ?d ?b (M ?c ?b ?a))"}
rule! {comm_lm,        "(M ?a ?b ?c)",          "(M ?b ?a ?c)"          }
rule! {comm_lr,        "(M ?a ?b ?c)",          "(M ?c ?b ?a)"          }
rule! {comm_mr,        "(M ?a ?b ?c)",          "(M ?a ?c ?b)"          }
rule! {maj_2_equ,      "(M ?a ?b ?b)",          "?b"                    }
rule! {maj_2_com,      "(M ?a ?b (~ ?b))",      "?a"                    }

rule! {associ_and,     "(& ?a (& ?b ?c))",      "(& ?b (& ?a ?c))"      }
rule! {comm_and,       "(& ?a ?b)",             "(& ?b ?a)"             }
rule! {comp_and,       "(& ?a (~ ?a))",         "0"                     }
rule! {dup_and,        "(& ?a ?a)",             "?a"                    }
rule! {and_true,       "(& ?a 1)",              "?a"                    }
rule! {and_false,      "(& ?a 0)",              "0"                     }
// add (M ?a ?a ?b) => ?a
rule! {maj_dup,        "(M ?a ?a ?b)",          "?a"                    }

// -----------------------------------------------------------------------------------
// 9. Simple function for demonstration of equivalence proofs (not strictly needed here).
// -----------------------------------------------------------------------------------
fn prove_something(name: &str, start: &str, rewrites: &[CRewrite], goals: &[&str]) {
    println!("Proving {}", name);

    let start_expr: egg::RecExpr<_> = start.parse().unwrap();
    let goal_exprs: Vec<egg::RecExpr<_>> = goals.iter().map(|g| g.parse().unwrap()).collect();

    let mut runner = egg::Runner::default()
        .with_iter_limit(20)
        .with_node_limit(5_000)
        .with_expr(&start_expr);

    // We assume the input expression is `true` for soundness.
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

// -----------------------------------------------------------------------------------
// 10. FFI interfaces via cxx::bridge for cross-language calls.
//     We expose 'simplify_depth' and 'simplify_size' plus a deallocation function.
// -----------------------------------------------------------------------------------
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
            "expr: {}, depth: {}, size: {}, invs: {}",
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

// -----------------------------------------------------------------------------------
// 11. Memory deallocation for C-string
// -----------------------------------------------------------------------------------
pub fn free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

// -----------------------------------------------------------------------------------
// 12. The main function we compare: simplify_depth, which extracts a tree (not DAG).
// -----------------------------------------------------------------------------------
pub fn simplify_depth(s: &str, vars: *const u32, size: usize) -> ffi::CCost {
    // Collect rewrite rules that might help reduce or restructure the expression
    let all_rules = &[
        double_neg(),
        double_neg_flip(),
        neg(),
        neg_flip(),
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
        associ_and(),
        comm_and(),
        comp_and(),
        dup_and(),
        and_true(),
        and_false(),
        true_false(),
        false_true(),
        neg_false(),
        neg_true(),
    ];

    // Parse the original expression
    let bef_expr: egg::RecExpr<MIG> = s.parse().unwrap();

    // Prepare the variable depth array
    let vars_default: [u32; 26] = [0; 26];
    let mut vars_: &[u32] = &vars_default;
    if size > 0 {
        vars_ = unsafe { std::slice::from_raw_parts(vars, size) };
    }

    // Create a Runner and initialize with the expression
    let mut runner = egg::Runner::default().with_expr(&bef_expr);
    let root = runner.roots[0];

    // Run the rewrite rules to saturate or partially simplify the expression
    runner = runner.run(all_rules);

    // Extract a *tree* result from the e-graph using a standard Egg Extractor
    let (best_cost, best) =
        egg::Extractor::new(&runner.egraph, MIGCostFn_dsi::new(&runner.egraph, &vars_))
            .find_best(root);

    // Convert the best expression to prefix form to get depth, size, and #inversions
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
    ffi::CCost {
        aft_expr: aft_expr,
        aft_dep: aft_dep,
        aft_size: aft_size,
        aft_invs: aft_invs,
    }
}

// -----------------------------------------------------------------------------------
// 13. Helper code for ILP extraction, to store e-graph as a JSON, etc.
//     These are used by 'simplify_size' (which uses ILP or other DAG-based extraction).
// -----------------------------------------------------------------------------------
mod extract;
pub use extract::Extractor;

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

/// Convert an egg egraph into a JSON-serializable structure (egraph_serialize::EGraph).
/// This structure is then used by ILP-based or other custom extractors.
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
            let cost = cost_function.cal_cur_cost(node);

            // Ensure all children exist
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

/// A new function to clone the e-graph into a serialized JSON structure for ILP,
/// but using a different cost function that only counts Maj operators, etc.
pub fn egg_to_serialized_egraph_for_ilp(
    egraph: &CEGraph,
    root_id: egg::Id,
    original_dep: &[u32],
) -> egraph_serialize::EGraph {
    // Define a local function to assign ILP cost
    fn cal_ilp_cost(node: &MIG, original_dep: &[u32]) -> CCost {
        match node {
            // M操作：面积定为2，depth=1，用来让Phase1更严格地减少M的数量
            MIG::Maj(..) => CCost {
                dep: 1,
                aom: 2,
                inv: 0,
            },
            // And操作：面积=1，depth=1，也会被考虑进Phase1与Phase2
            MIG::And(..) => CCost {
                dep: 1,
                aom: 1,
                inv: 0,
            },
            // Not操作：面积=1, depth=0, inv=1 (只是在ILP里区分一下)
            MIG::Not(..) => CCost {
                dep: 0,
                aom: 1,
                inv: 1,
            },
            MIG::Symbol(v) => {
                let chr_v = v.as_str().as_bytes();
                let index = (chr_v[0] as u8) - b'a';
                CCost {
                    dep: original_dep[index as usize],
                    aom: 0,
                    inv: 0,
                }
            }
            // 其他(常量0,1)都免费
            _ => CCost::default(),
        }
    }

    let mut out = egraph_serialize::EGraph::default();
    let final_root = egraph.find(root_id);

    // Create all classes first
    for class in egraph.classes() {
        out.class_data.insert(
            egraph_serialize::ClassId::from(format!("{}", class.id)),
            egraph_serialize::ClassData { typ: None },
        );
    }

    // Then add each node with the "new ILP cost"
    for class in egraph.classes() {
        for (i, node) in class.nodes.iter().enumerate() {
            // Use our custom ILP cost function
            let cost_val = cal_ilp_cost(node, original_dep).encode();

            // Make sure child classes exist
            for child in node.children() {
                let cid = egraph_serialize::ClassId::from(format!("{}", child));
                if !out.class_data.contains_key(&cid) {
                    out.class_data
                        .insert(cid, egraph_serialize::ClassData { typ: None });
                }
            }

            // Add the node
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
                    cost: egraph_serialize::Cost::new(cost_val).unwrap(),
                    subsumed: false,
                },
            )
        }
    }

    // Set the root
    out.root_eclasses = vec![egraph_serialize::ClassId::from(format!("{}", final_root))];
    out
}

/// Find root nodes in the e-graph. Typically, we look for classes that
/// are not children of any node or contain a Maj as a root operation.
pub fn find_root_nodes(egraph: &CEGraph) -> Vec<Id> {
    let mut roots = Vec::new();
    let mut has_parent = std::collections::HashSet::new();

    // Collect all child IDs
    for class in egraph.classes() {
        for node in &class.nodes {
            for child in node.children() {
                has_parent.insert(child);
            }
        }
    }

    // A root is any ID not in has_parent
    for class in egraph.classes() {
        if !has_parent.contains(&class.id) {
            roots.push(class.id);
        }
    }

    // If no root found, try the first eclass that has a Maj node
    if roots.is_empty() {
        for class in egraph.classes() {
            if class.nodes.iter().any(|n| matches!(n, MIG::Maj(_))) {
                roots.push(class.id);
                break;
            }
        }
    }
    roots
}

/// Save the serialized egraph to a JSON file, including root e-class IDs.
pub fn save_serialized_egraph_to_json(
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

// -----------------------------------------------------------------------------------
// 14. The second major function: "simplify_size", which uses a DAG-based ILP extractor
//     (e.g., "faster_ilp_cbc") to find a minimal node count. We still track depth afterwards.
// -----------------------------------------------------------------------------------
pub fn simplify_size(s: &str, vars: *const u32, size: usize) -> ffi::CCost {
    let all_rules = &[
        double_neg(),
        double_neg_flip(),
        neg(),
        neg_flip(),
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
        associ_and(),
        comm_and(),
        comp_and(),
        dup_and(),
        and_true(),
        and_false(),
        true_false(),
        false_true(),
        neg_false(),
        neg_true(),
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

    // Convert the egraph to a JSON-serializable structure
    #[cfg(not(feature = "ilp-cbc"))]
    let serialized_egraph = egg_to_serialized_egraph(
        &saturated_egraph,
        &MIGCostFn_dsi::new(&saturated_egraph, vars_),
        root_id,
    );
    #[cfg(feature = "ilp-cbc")]
    let serialized_egraph_ilp = egg_to_serialized_egraph_for_ilp(&saturated_egraph, root_id, vars_);

    // Use custom extraction code
    #[cfg(feature = "ilp-cbc")]
    let extractor = extract::faster_ilp_cbc::FasterCbcExtractor::default();
    #[cfg(feature = "ilp-cbc")]
    let extraction_result =
        extractor.extract(&serialized_egraph_ilp, &serialized_egraph_ilp.root_eclasses);
    #[cfg(feature = "ilp-cbc")]
    let dag_cost_size = extraction_result
        .dag_cost_size_enhanced(&serialized_egraph_ilp, &serialized_egraph_ilp.root_eclasses);
    #[cfg(feature = "ilp-cbc")]
    let dag_cost_depth = extraction_result
        .dag_cost_depth(&serialized_egraph_ilp, &serialized_egraph_ilp.root_eclasses);
    #[cfg(feature = "ilp-cbc")]
    let aft_expr = extraction_result.print_aft_expr(&serialized_egraph_ilp);

    // Test faster greedy extractor
    #[cfg(not(feature = "ilp-cbc"))]
    // let extractor = extract::global_greedy_dag::GlobalGreedyDagExtractor {};
    let extractor = extract::faster_greedy_dag::FasterGreedyDagExtractor {};
    #[cfg(not(feature = "ilp-cbc"))]
    let extraction_result = extractor.extract(&serialized_egraph, &serialized_egraph.root_eclasses);
    #[cfg(not(feature = "ilp-cbc"))]
    let dag_cost_size =
        extraction_result.dag_cost_size(&serialized_egraph, &serialized_egraph.root_eclasses);
    #[cfg(not(feature = "ilp-cbc"))]
    let dag_cost_depth =
        extraction_result.dag_cost_depth(&serialized_egraph, &serialized_egraph.root_eclasses);
    #[cfg(not(feature = "ilp-cbc"))]
    let aft_expr = extraction_result.print_aft_expr(&serialized_egraph);

    ffi::CCost {
        aft_expr: aft_expr,
        aft_dep: dag_cost_depth,
        aft_size: dag_cost_size,
        aft_invs: 0,
    }
}

// -----------------------------------------------------------------------------------
// 15. Function to show usage examples and comparisons
//     (simplify() calls both a tree-based approach and a DAG-based approach).
// -----------------------------------------------------------------------------------
pub fn simplify(s: &str, var_dep: &Vec<u32>) {
    // Initialize logger if not already initialized
    let _ = env_logger::try_init();

    info!("{}", "=".repeat(50).blue());
    info!("{} {}\n", "Simplifying expression:".bright_blue(), s);
    // Rewrites for the tree-based approach
    let all_rules = &[
        double_neg(),
        double_neg_flip(),
        neg(),
        neg_flip(),
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
        associ_and(),
        comm_and(),
        comp_and(),
        dup_and(),
        and_true(),
        and_false(),
        true_false(),
        false_true(),
        neg_false(),
        neg_true(),
        maj_dup(),
    ];

    // Convert the var_dep to a raw pointer if needed
    let vars_default: [u32; 26] = [0; 26];
    let mut vars_ = vars_default.as_ptr();
    let mut var_len = 26;
    if !var_dep.is_empty() {
        vars_ = var_dep.as_ptr();
        var_len = var_dep.len();
    }

    // 1. Get baseline results
    let cost_depth = simplify_depth(s, vars_, var_len);
    info!(
        "{} {} - {}",
        "Baseline".bright_green(),
        "(simplify_depth)".green(),
        format!("{}", cost_depth).green()
    );

    let baseline_size = cost_depth.aft_size;
    let baseline_depth = cost_depth.aft_dep;

    // 2. Build and saturate an e-graph
    let expr: egg::RecExpr<MIG> = s.parse().unwrap();
    let mut runner = egg::Runner::default()
        .with_expr(&expr)
        .with_iter_limit(1000)
        .with_node_limit(5000)
        .with_time_limit(std::time::Duration::from_secs(10));
    let root_id = runner.roots[0];

    runner = runner.run(all_rules);
    let saturated_egraph = runner.egraph;

    // 3. Convert for ILP/greedy extraction
    let serialized_egraph = egg_to_serialized_egraph(
        &saturated_egraph,
        &MIGCostFn_dsi::new(&saturated_egraph, unsafe {
            std::slice::from_raw_parts(vars_, var_len)
        }),
        root_id,
    );

    let serialized_egraph_ilp =
        egg_to_serialized_egraph_for_ilp(&saturated_egraph, root_id, unsafe {
            std::slice::from_raw_parts(vars_, var_len)
        });

    // 4. Track best results across all methods
    #[derive(Clone)]
    struct MethodResult {
        method: String,
        size: u32,
        depth: u32,
        expr: String,
    }

    let mut results = Vec::new();
    results.push(MethodResult {
        method: "Baseline".to_string(),
        size: baseline_size,
        depth: baseline_depth,
        expr: cost_depth.aft_expr.clone(),
    });

    // Helper function to print results and collect them
    let mut print_result = |method: &str, expr: &str, depth: u32, size: u32| {
        let is_worse = size > baseline_size || (size == baseline_size && depth > baseline_depth);

        if is_worse {
            warn!(
                "{:<25} {} {}",
                method.bright_red(),
                "- expr:".bright_red(),
                format!(
                    "{}, depth: {}, size: {} (worse than baseline)",
                    expr, depth, size
                )
                .red()
            );
        } else {
            info!(
                "{:<25} {} {}",
                method.bright_green(),
                "- expr:".bright_green(),
                format!("{}, depth: {}, size: {}", expr, depth, size).green()
            );
        }
        results.push(MethodResult {
            method: method.to_string(),
            size,
            depth,
            expr: expr.to_string(),
        });
    };

    // 5. Compare different extractors
    #[cfg(feature = "ilp-cbc")]
    let extractor = extract::faster_ilp_cbc::FasterCbcExtractor::default();

    let extraction_result =
        extractor.extract(&serialized_egraph_ilp, &serialized_egraph_ilp.root_eclasses);
    let dag_cost_size = extraction_result
        .dag_cost_size_enhanced(&serialized_egraph_ilp, &serialized_egraph_ilp.root_eclasses);
    let dag_cost_depth = extraction_result
        .dag_cost_depth(&serialized_egraph_ilp, &serialized_egraph_ilp.root_eclasses);
    let aft_expr = extraction_result.print_aft_expr(&serialized_egraph_ilp);
    print_result(
        "DAG-based (faster ILP)",
        &aft_expr,
        dag_cost_depth,
        dag_cost_size,
    );

    // Test faster greedy extractor
    let extractor1 = extract::faster_greedy_dag::FasterGreedyDagExtractor {};
    let extraction_result1 =
        extractor1.extract(&serialized_egraph, &serialized_egraph.root_eclasses);
    let dag_cost_size1 = extraction_result1
        .dag_cost_size_enhanced(&serialized_egraph, &serialized_egraph.root_eclasses);
    let dag_cost_depth1 =
        extraction_result1.dag_cost_depth(&serialized_egraph, &serialized_egraph.root_eclasses);
    let aft_expr1 = extraction_result1.print_aft_expr(&serialized_egraph);

    print_result(
        "DAG-based (faster greedy)",
        &aft_expr1,
        dag_cost_depth1,
        dag_cost_size1,
    );

    // Test global greedy extractor
    let extractor2 = extract::global_greedy_dag::GlobalGreedyDagExtractor {};
    let extraction_result2 =
        extractor2.extract(&serialized_egraph, &serialized_egraph.root_eclasses);
    let dag_cost_size2 = extraction_result2
        .dag_cost_size_enhanced(&serialized_egraph, &serialized_egraph.root_eclasses);
    let dag_cost_depth2 =
        extraction_result2.dag_cost_depth(&serialized_egraph, &serialized_egraph.root_eclasses);
    let aft_expr2 = extraction_result2.print_aft_expr(&serialized_egraph);
    print_result(
        "DAG-based (global greedy)",
        &aft_expr2,
        dag_cost_depth2,
        dag_cost_size2,
    );

    // Find the best result
    let best_result = results.iter().min_by_key(|r| (r.size, r.depth)).unwrap();

    // 6. Print summary
    info!("\n{}", "Summary".bright_blue().bold());
    info!(
        "{} {} - size: {}, depth: {}",
        "Best result from".bright_green(),
        best_result.method.bright_green(),
        best_result.size.to_string().green(),
        best_result.depth.to_string().green()
    );

    // Check if ALL methods are worse than baseline
    let all_worse = results
        .iter()
        .skip(1)
        .all(|r| r.size > baseline_size || (r.size == baseline_size && r.depth > baseline_depth));

    if all_worse {
        warn!(
            "{}",
            "⚠️  Note: All DAG-based methods performed worse than baseline for this expression"
                .bright_red()
        );
    }

    info!(
        "{} {}",
        "Best expression:".bright_green(),
        best_result.expr.green()
    );

    info!("{}\n", "=".repeat(50).blue());
}

// -----------------------------------------------------------------------------------
// 16. Basic tests
// -----------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_fold() {
        let start = "(M 0 1 0)";
        let start_expr: egg::RecExpr<MIG> = start.parse().unwrap();
        let end = "0";
        let end_expr: egg::RecExpr<MIG> = end.parse().unwrap();
        let mut eg: CEGraph = egg::EGraph::default();
        eg.add_expr(&start_expr);
        eg.rebuild();

        // We expect "0" to be proven equivalent because (M 0 1 0) should fold to 0
        assert!(!eg.equivs(&start_expr, &end_expr).is_empty());
    }

    #[test]
    fn test_depth() {
        // Initialize logger with default configuration
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .filter_module("egg", log::LevelFilter::Error)
            .filter_module("mig_egg::extract::faster_ilp_cbc", log::LevelFilter::Off)
            .is_test(true)
            .try_init();
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
            double_neg(),
            double_neg_flip(),
            neg(),
            neg_flip(),
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
            associ_and(),
            comm_and(),
            comp_and(),
            dup_and(),
            and_true(),
            and_false(),
            true_false(),
            false_true(),
            neg_false(),
            neg_true(),
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

        let mut runner = egg::Runner::default().with_expr(&expr);
        let root = runner.roots[0];
        runner = runner.run(all_rules);

        let (best_cost, best) =
            egg::Extractor::new(&runner.egraph, MIGCostFn_dsi::new(&runner.egraph, &var_dep))
                .find_best(root);

        let (prefix_expr, depth, ops_count, inv_count) = to_prefix(&best, &var_dep);
        println!("default cost: {:?}", best_cost);
        println!("after_expr: {}", prefix_expr);
        println!("after_dep: {}", depth);
        println!("after_size: {}", ops_count);
        println!("after_invs: {}", inv_count);
    }

    #[test]
    fn prove_chain() {
        // Initialize logger with default configuration
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .filter_module("egg", log::LevelFilter::Error)
            .filter_module("mig_egg::extract::faster_ilp_cbc", log::LevelFilter::Off)
            .is_test(true)
            .try_init();
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
        simplify("(M x 0 (M y 1 (M u 0 v)))", &empty_vec);
        simplify("(M (M w x (~ z)) x (M z x y))", &empty_vec);
        simplify("(M c (M c d (M e f b)) a)", &empty_vec);
        simplify("(M (~ 0) (M 0 c (~ (M 0 (M (~ 0) a b) (~ (M 0 a b))))) (M 0 (~ c) (M 0 (M (~ 0) a b) (~ (M 0 a b)))))",&empty_vec);
        simplify("(M (~ 0) (M 0 (M 0 a c) (~ (M 0 (M (~ 0) b d) (~ (M 0 b d))))) (M 0 (~ (M 0 a c)) (M 0 (M (~ 0) b d) (~ (M 0 b d)))))",&empty_vec);
        simplify("(M 0 (~ (M 0 (~ a) b)) (M 0 c (~ d)))", &empty_vec);
        simplify(
            "(M (~ 0) (M 0 a (~ (M 0 b (~ c)))) (M 0 (~ a) (M 0 b (~ c))))",
            &empty_vec,
        );
        simplify("(M (~ 0) (M 0 a (~ b)) (M 0 (~ a) b))", &empty_vec);
        simplify("(M (~ 0) (M (~ e) (M (~ 0) c (M 0 (~ a) b)) (M 0 (M 0 c (M 0 (~ a) b)) (M (~ 0) c (M 0 (~ a) b)))) (M (M 0 (~ d) g) h (M 0 (~ f) h)))",&empty_vec);
        simplify("(M 0 b (~ (M 0 (~ (M g (M 0 d (M a c (~ f))) (M e (M a c (~ f)) g))) (M 0 (M (~ 0) d (M a c (~ f))) g))))",&empty_vec);
        simplify("(M (~ 0) (M 0 (M 0 c (~ (M (~ 0) (M 0 a (~ b)) (M 0 (~ a) b)))) h) (M (M 0 (~ c) d) (M 0 e (~ f)) (~ (M 0 (M 0 (~ c) d) g))))",&empty_vec);
        simplify("(M (~ 0) (M 0 (M 0 c (~ (M (~ 0) (M 0 a (~ b)) (M 0 (~ a) b)))) h) (M (M 0 (~ c) d) (M 0 e (~ f)) (~ (M 0 (M 0 (~ c) d) g))))", &vec![0, 0, 2, 2, 4, 6, 5, 7]);
        simplify(
            "(M (~ 0) b (M (~ (M a (~ c) e)) f (M 0 d f)))",
            &vec![0, 0, 3, 4, 2, 4],
        );
        simplify(
            "(M 0 (M (~ 0) (M 0 (M (~ 0) a b) (~ (M 0 a b))) c) (~ (M 0 (M 0 (M (~ 0) a b) (~ (M 0 a b))) c)))",
            &vec![0, 0, 2],
        );
        simplify(
            "(M 0 (M (~ 0) e (M b d (M 0 a c))) (~ (M 0 e (M b d (M 0 a c)))))",
            &vec![0, 0, 0, 0, 4],
        );
        simplify(
            "(M 0 (M (~ 0) (M 0 (M (~ 0) c f) (~ (M 0 c f))) (M b e (M 0 a d))) (~ (M 0 (M 0 (M (~ 0) c f) (~ (M 0 c f))) (M b e (M 0 a d)))))",
            &empty_vec,
        );
        simplify(
            "(M (~ 0) (M 0 d h) (M 0 (M (~ 0) d h) (M c g (M b f (M 0 a e)))))",
            &empty_vec,
        );
        simplify(
            "(M 0 (M (~ 0) f (M b d (M a c e))) (~ (M 0 f (M b d (M a c e)))))",
            &vec![0, 0, 0, 0, 4, 5],
        );
        simplify(
            "(M 0 (M (~ 0) e (M b d (M a c f))) (~ (M 0 e (M b d (M a c f)))))",
            &vec![0, 0, 0, 0, 4, 4],
        );
        simplify(
            "(M (~ 0) f (M 0 e (M b d (M a c g))))",
            &vec![0, 0, 0, 0, 4, 4, 6],
        );
        simplify(
            "(M (~ 0) (M f (M 0 a e) (M 0 f (M 0 b (~ e)))) (M (~ f) (M 0 (~ f) (M 0 d (~ e))) (M 0 c e)))",
            &empty_vec,
        );
        simplify(
            "(M (~ 0) (M 0 (~ (M (~ 0) a b)) c) (M 0 (M 0 a (~ b)) d))",
            &vec![0, 0, 2, 2],
        );
        simplify(
            "(M (~ 0) (M f (M 0 b e) (M 0 f (M 0 c (~ e)))) (M (~ f) (M 0 (~ f) (M 0 a (~ e))) (M 0 d e)))",
            &empty_vec,
        );
        simplify(
            "(M (~ 0) (M 0 (~ c) (M (~ 0) e (M (~ (M (~ 0) a b)) (M 0 (M 0 a (~ b)) h) (M (~ 0) (M (~ 0) a b) g)))) (M 0 d f))",
            &vec![0, 0, 2, 2, 6, 7, 4, 4],
        );
        simplify(
            "(M (~ 0) (M (~ e) (M 0 e (M 0 a d)) (M (~ 0) e f)) (M (~ e) (M 0 e (M 0 b (~ d))) (M (~ 0) e (M 0 c d))))",
            &vec![0, 0, 0, 0, 0, 5],
        );
    }
}
