use colored::*;
use egg::{Id, Language};
use log::{info, warn};
use std::ffi::CString;
use std::fmt;
use std::ops::{Add, AddAssign};
use std::os::raw::c_char;

// -----------------------------------------------------------------------------------
// 1. Define a cost struct (CCost) to keep track of depth (dep), area (aom), and inversions (inv).
//    It also includes methods for merging and for a custom f64 bit-encoding/decoding.
// -----------------------------------------------------------------------------------
#[derive(Debug, Clone, Copy)]
pub struct CCost {
    dep: f64,
    aom: f64,
    inv: f64,
}

impl Default for CCost {
    fn default() -> Self {
        CCost {
            dep: 0.0,
            aom: 0.0,
            inv: 0.0,
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
    pub fn merge(a: &CCost, b: &CCost, f_dep: Option<bool>) -> CCost {
        if f_dep.unwrap_or(true) {
            CCost {
                dep: a.dep.max(b.dep),
                aom: a.aom + b.aom,
                inv: a.inv + b.inv,
            }
        } else {
            CCost {
                dep: a.dep + b.dep,
                aom: a.aom.max(b.aom),
                inv: a.inv + b.inv,
            }
        }
    }

    /// Returns a "maximum" cost.
    fn max() -> Self {
        CCost {
            dep: f64::MAX,
            aom: f64::MAX,
            inv: f64::MAX,
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
        let x = ((packed >> Self::SHIFT_X) & Self::MASK) as f64;
        let y = ((packed >> Self::SHIFT_Y) & Self::MASK) as f64;
        let z = (packed & Self::MASK) as f64;

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

impl PartialEq for CCost {
    fn eq(&self, other: &Self) -> bool {
        self.dep.to_bits() == other.dep.to_bits()
            && self.aom.to_bits() == other.aom.to_bits()
            && self.inv.to_bits() == other.inv.to_bits()
    }
}
impl Eq for CCost {}

impl PartialOrd for CCost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let src = (self.dep, self.aom, self.inv);
        let tar = (other.dep, other.aom, other.inv);
        src.partial_cmp(&tar)
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
    // Dense reciprocal fanout per canonical eclass id: rcp_fanout[usize::from(id)] = 1.0 / max(1, fanout)
    rcp_fanout: Vec<f64>,
    original_dep: &'a [u32],
    first_depth: bool,
}

impl<'a> MIGCostFn_dsi<'a> {
    pub fn new(graph: &'a CEGraph, vars_: &'a [u32], f_dep: Option<bool>) -> Self {
        // Precompute fanout counts per canonical eclass id across the whole e-graph
        // Use a dense vector for speed, then store reciprocals to avoid division in the hot path
        let mut counts: Vec<u32> = Vec::new();
        for class in graph.classes() {
            for node in &class.nodes {
                for child in node.children() {
                    let idx = usize::from(graph.find(*child));
                    if idx >= counts.len() {
                        counts.resize(idx + 1, 0);
                    }
                    counts[idx] += 1;
                }
            }
        }
        let rcp_fanout: Vec<f64> = counts
            .into_iter()
            .map(|c| 1.0f64 / (c.max(1) as f64))
            .collect();
        Self {
            egraph: graph,
            rcp_fanout,
            original_dep: vars_,
            first_depth: f_dep.unwrap_or(true),
        }
    }

    pub fn reset(&mut self) {
        // self.visited.clear();
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
        if self.first_depth {
            CCost {
                dep: op_depth as f64,
                aom: op_area as f64,
                inv: op_inv as f64,
            }
        } else {
            CCost {
                dep: op_area as f64,
                aom: op_depth as f64,
                inv: op_inv as f64,
            }
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
        let mut max_dep: f64 = 0.0;
        let mut sum_inv: f64 = 0.0;
        let mut weighted_primary: f64 = 0.0;
        for id in enode.children().iter().copied() {
            let cc = costs(id);
            let idx = usize::from(self.egraph.find(id));
            let r = *self.rcp_fanout.get(idx).unwrap_or(&1.0);
            if self.first_depth {
                // depth-first: dep = max(child.dep), aom = sum(child.aom / fanout), inv = sum(child.inv)
                if cc.dep > max_dep {
                    max_dep = cc.dep;
                }
                sum_inv += cc.inv;
                weighted_primary += cc.aom * r;
            } else {
                // size-first: aom = max(child.aom), dep = sum(child.dep / fanout), inv = sum(child.inv)
                if cc.aom > max_dep {
                    max_dep = cc.aom;
                }
                sum_inv += cc.inv;
                weighted_primary += cc.dep * r;
            }
        }

        cur_cost
            + if self.first_depth {
                CCost {
                    dep: max_dep,
                    aom: weighted_primary,
                    inv: sum_inv,
                }
            } else {
                CCost {
                    dep: weighted_primary,
                    aom: max_dep,
                    inv: sum_inv,
                }
            }
    }
}

/// Implementation for egg::LpCostFunction trait (used by ILP extraction).
#[cfg_attr(docsrs, doc(cfg(feature = "lp")))]
impl<'a> egg::LpCostFunction<MIG, ConstantFold> for MIGCostFn_dsi<'a> {
    fn node_cost(&mut self, _egraph: &CEGraph, _eclass: egg::Id, enode: &MIG) -> f64 {
        let cur_cost = self.cal_cur_cost(enode);

        cur_cost.aom as f64
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

rule! {neg_false,      true_false,              "(~ 0)",                              "1"    }
rule! {neg_true,       false_true,              "(~ 1)",                              "0"    }
rule! {double_neg,     double_neg_flip,         "(~ (~ ?a))",                         "?a"   }

rule! {neg,            neg_flip,                "(~ (M ?a ?b ?c))",                   "(M (~ ?a) (~ ?b) (~ ?c))"        }
rule! {distri,         distri_flip,             "(M ?a ?b (M ?c ?d ?e))",             "(M (M ?a ?b ?c) (M ?a ?b ?d) ?e)"}
rule! {com_associ,     com_associ_flip,         "(M ?a ?b (M ?c (~ ?b) ?d))",         "(M ?a ?b (M ?c ?a ?d))"          }
rule! {maj_com_equ,    maj_equ_com,             "(M ?a ?b (~ ?b))",                   "(M ?a ?b ?a)"                    }
rule! {xnor_0,         xnor_0_flip,             "(M ?a (M 0 ?b ?c) (~ (M 1 ?b ?c)))", "(M 0 ?a (M 1 (M 0 ?b ?c) (~ (M 1 ?b ?c))))"}
rule! {xor_1,          xor_1_flip,              "(M ?a (~ (M 0 ?b ?c)) (M 1 ?b ?c))", "(M 1 ?a (M 0 (~ (M 0 ?b ?c)) (M 1 ?b ?c)))"}
rule! {pull_out_e, push_in_e,
    "(M 0 ?s (M ?a (M 0 ?t (M 1 ?d ?e)) (M 1 ?a ?b)))",
    "(M ?e (M 0 ?s (M ?a (M 1 ?a ?b) ?t)) (M 0 ?s (M ?a ?d (M ?a (M 1 ?a ?b) ?t))))"
}
rule! {relevance,      "(M ?a ?b (M ?c ?d (M ?a ?b ?e)))", "(M ?a ?b (M ?c ?d (M (~ ?b) ?b ?e)))"}
rule! {associ,         "(M ?a ?b (M ?c ?b ?d))",           "(M ?d ?b (M ?c ?b ?a))"}
rule! {comm_lm,        "(M ?a ?b ?c)",                     "(M ?b ?a ?c)"          }
rule! {comm_lr,        "(M ?a ?b ?c)",                     "(M ?c ?b ?a)"          }
rule! {comm_mr,        "(M ?a ?b ?c)",                     "(M ?a ?c ?b)"          }
rule! {maj_2_equ,      "(M ?a ?b ?b)",                     "?b"                    }
rule! {maj_2_com,      "(M ?a ?b (~ ?b))",                 "?a"                    }

rule! {associ_and,     "(& ?a (& ?b ?c))",                 "(& ?b (& ?a ?c))"      }
rule! {comm_and,       "(& ?a ?b)",                        "(& ?b ?a)"             }
rule! {comp_and,       "(& ?a (~ ?a))",                    "0"                     }
rule! {dup_and,        "(& ?a ?a)",                        "?a"                    }
rule! {and_true,       "(& ?a 1)",                         "?a"                    }
rule! {and_false,      "(& ?a 0)",                         "0"                     }

fn rules() -> Vec<CRewrite> {
    vec![
        neg_false(),
        true_false(),
        neg_true(),
        false_true(),
        double_neg(),
        double_neg_flip(),
        neg(),
        neg_flip(),
        distri(),
        distri_flip(),
        com_associ(),
        com_associ_flip(),
        maj_com_equ(),
        maj_equ_com(),
        associ(),
        comm_lm(),
        comm_lr(),
        comm_mr(),
        maj_2_equ(),
        maj_2_com(),
        xnor_0(),
        xnor_0_flip(),
        xor_1(),
        xor_1_flip(),
        // associ_and(),
        // comm_and(),
        // comp_and(),
        // dup_and(),
        // and_true(),
        // and_false(),
    ]
}

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
    #[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
    struct CCost {
        // bef_expr: String,
        // bef_dep: u32,
        // bef_size: u32,
        // bef_invs: u32,
        aft_expr: Vec<String>,
        aft_dep: u32,
        aft_size: u32,
        aft_invs: u32,
    }

    extern "Rust" {
        unsafe fn simplify_depth(
            s: &str,
            vars: *const u32,
            size: usize,
            first_depth: bool,
        ) -> CCost;
        unsafe fn simplify_size(s: &str, vars: *const u32, size: usize, first_depth: bool)
        -> CCost;
        unsafe fn simplify_best(s: &str, vars: *const u32, size: usize, first_depth: bool)
        -> CCost;
        fn merge_ccost(depth: &CCost, size: &CCost) -> CCost;
        unsafe fn free_string(s: *mut c_char);
    }
}

fn merge_ccost(depth: &ffi::CCost, size: &ffi::CCost) -> ffi::CCost {
    if depth.aft_dep < size.aft_dep && depth.aft_size <= size.aft_size {
        return depth.clone();
    }
    if size.aft_dep < depth.aft_dep && size.aft_size <= depth.aft_size {
        return size.clone();
    }

    let mut merged_expr = depth.aft_expr.clone();
    merged_expr.extend(size.aft_expr.clone());
    let unique_set: std::collections::HashSet<String> = merged_expr.into_iter().collect();
    ffi::CCost {
        aft_expr: unique_set.into_iter().collect(),
        aft_dep: depth.aft_dep,
        aft_size: depth.aft_size,
        aft_invs: depth.aft_invs,
    }
}

impl Clone for ffi::CCost {
    fn clone(&self) -> Self {
        ffi::CCost {
            aft_expr: self.aft_expr.clone(),
            aft_dep: self.aft_dep,
            aft_size: self.aft_size,
            aft_invs: self.aft_invs,
        }
    }
}

impl std::fmt::Display for ffi::CCost {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "expr: {:?}, depth: {}, size: {}, invs: {}",
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
pub fn simplify_depth(s: &str, vars: *const u32, size: usize, first_depth: bool) -> ffi::CCost {
    // Collect rewrite rules that might help reduce or restructure the expression
    let all_rules = rules();

    // Parse the original expression
    let bef_expr: egg::RecExpr<MIG> = s.parse().unwrap();

    // Prepare the variable depth array
    let vars_default: [u32; 26] = [0; 26];
    let mut vars_: &[u32] = &vars_default;
    if size > 0 {
        vars_ = unsafe { std::slice::from_raw_parts(vars, size) };
    }

    // Create a Runner and initialize with the expression
    let mut runner = egg::Runner::default()
        .with_expr(&bef_expr)
        .with_iter_limit(30)
        .with_node_limit(10000)
        .with_time_limit(std::time::Duration::from_secs(5));

    // Run the rewrite rules to saturate or partially simplify the expression
    runner = runner.run(&all_rules);
    let mut roots: Vec<Id> = runner
        .roots
        .iter()
        .map(|root| runner.egraph.find(*root))
        .collect();
    let root = roots[0];

    // Extract a *tree* result from the e-graph using a standard Egg Extractor
    let best: egg::RecExpr<MIG>;
    let mut best_cost = CCost::max();
    if true {
        (best_cost, best) = egg::Extractor::new(
            &runner.egraph,
            MIGCostFn_dsi::new(&runner.egraph, &vars_, Some(first_depth)),
        )
        .find_best(root);
    } else {
        let mut lp_extractor = egg::LpExtractor::new(
            &runner.egraph,
            MIGCostFn_dsi::new(&runner.egraph, &vars_, Some(true)),
        );
        // lp_extractor.timeout(10.0);

        (best, roots) = lp_extractor.solve_multiple(&roots);
    }

    // Convert the best expression to prefix form to get depth, size, and #inversions
    let (aft_expr, aft_dep, aft_size, aft_invs) = to_prefix(&best, &vars_);
    ffi::CCost {
        aft_expr: vec![aft_expr],
        aft_dep: aft_dep,
        aft_size: aft_size,
        aft_invs: aft_invs,
    }
}

// -----------------------------------------------------------------------------------
// 14. The second major function: "simplify_size", which uses a DAG-based ILP extractor
//     (e.g., "faster_ilp_cbc") to find a minimal node count. We still track depth afterwards.
// -----------------------------------------------------------------------------------
pub fn simplify_size(s: &str, vars: *const u32, size: usize, first_depth: bool) -> ffi::CCost {
    simplify_depth(s, vars, size, false)
}

// -----------------------------------------------------------------------------------
// 15. Function to show usage examples and comparisons
//     (simplify() calls both a tree-based approach and a DAG-based approach).
// -----------------------------------------------------------------------------------
pub fn simplify_best(s: &str, vars: *const u32, var_len: usize, first_depth: bool) -> ffi::CCost {
    // Initialize logger if not already initialized
    let _ = env_logger::try_init();

    info!("{}", "=".repeat(50).blue());
    info!("{} {}\n", "Simplifying expression:".bright_blue(), s);

    // 1. Get baseline depth-oriented results
    let cost_depth = simplify_depth(s, vars, var_len, true);

    let baseline_size = cost_depth.aft_size;
    let baseline_depth = cost_depth.aft_dep;

    // 2. Get baseline size-oriented results
    let cost_size = simplify_depth(s, vars, var_len, false);

    let baseline2_size = cost_size.aft_size;
    let baseline2_depth = cost_size.aft_dep;

    // 4. Track best results across all methods
    #[derive(Clone)]
    struct MethodResult {
        method: String,
        size: u32,
        depth: u32,
        expr: String,
    }

    let mut results = Vec::new();

    // Helper function to print results and collect them
    let mut print_result = |method: &str, expr: &str, depth: u32, size: u32| {
        let is_worse = depth > baseline_depth || (depth == baseline_depth && size > baseline_size);

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

    print_result(
        "Baseline (simplify_depth for depth)",
        &cost_depth.aft_expr[0],
        baseline_depth,
        baseline_size,
    );

    print_result(
        "Baseline (simplify_depth for size)",
        &cost_size.aft_expr[0],
        baseline2_depth,
        baseline2_size,
    );

    // Find the best result
    let best_result = results.iter().min_by_key(|r| (r.depth, r.size)).unwrap();
    let best_exprs: std::collections::HashSet<String> = results
        .iter()
        .filter(|r| (r.depth, r.size) == (best_result.depth, best_result.size))
        .map(|r| r.expr.clone())
        .collect();

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
        .all(|r| r.depth > baseline_depth || (r.depth == baseline_depth && r.size > baseline_size));

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
        format!("{:?}", best_exprs).green()
    );

    info!("{}\n", "=".repeat(50).blue());
    ffi::CCost {
        aft_expr: best_exprs.into_iter().collect(),
        aft_dep: best_result.depth,
        aft_size: best_result.size,
        aft_invs: 0,
    }
}

pub fn simplify(s: &str, var_dep: &Vec<u32>) -> ffi::CCost {
    // Convert the var_dep to a raw pointer if needed
    let vars_default: [u32; 26] = [0; 26];
    let mut vars_ = vars_default.as_ptr();
    let mut var_len = 26;
    if !var_dep.is_empty() {
        vars_ = var_dep.as_ptr();
        var_len = var_dep.len();
    }

    simplify_best(s, vars_, var_len, false)
}

// -----------------------------------------------------------------------------------
// 16. Basic tests
// -----------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    enum Expr {
        Const(bool),
        Var(char),
        Not(Box<Expr>),
        Majority(Box<Expr>, Box<Expr>, Box<Expr>),
    }

    fn parse_expression(tokens: &mut Vec<String>) -> Option<Expr> {
        tokens.pop().and_then(|token| match token.as_str() {
            "M" => {
                let a = parse_expression(tokens)?;
                let b = parse_expression(tokens)?;
                let c = parse_expression(tokens)?;
                Some(Expr::Majority(Box::new(a), Box::new(b), Box::new(c)))
            }
            "~" => Some(Expr::Not(Box::new(parse_expression(tokens)?))),
            "0" => Some(Expr::Const(false)),
            "1" => Some(Expr::Const(true)),
            var => {
                let c = var.chars().next()?;
                if c.is_ascii_lowercase() {
                    Some(Expr::Var(c))
                } else {
                    None
                }
            }
        })
    }

    fn collect_vars(expr: &Expr, vars: &mut std::collections::HashSet<char>) {
        match expr {
            Expr::Var(c) => {
                vars.insert(*c);
            }
            Expr::Not(e) => collect_vars(e, vars),
            Expr::Majority(a, b, c) => {
                collect_vars(a, vars);
                collect_vars(b, vars);
                collect_vars(c, vars);
            }
            _ => {}
        }
    }

    fn evaluate(expr: &Expr, vars: &std::collections::HashMap<char, bool>) -> bool {
        match expr {
            Expr::Const(val) => *val,
            Expr::Var(c) => *vars.get(c).expect("存在未定义变量"),
            Expr::Not(e) => !evaluate(e, vars),
            Expr::Majority(a, b, c) => {
                let a_val = evaluate(a, vars);
                let b_val = evaluate(b, vars);
                let c_val = evaluate(c, vars);
                (a_val & b_val) | (a_val & c_val) | (b_val & c_val)
            }
        }
    }

    fn is_equiv(expr1: &str, expr2: &str) -> bool {
        let preprocess = |s: &str| {
            s.replace('(', " ")
                .replace(')', " ")
                .split_whitespace()
                .rev()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        };

        let mut tokens1 = preprocess(expr1);
        let mut tokens2 = preprocess(expr2);
        let expr1 = parse_expression(&mut tokens1).expect("解析表达式1失败");
        let expr2 = parse_expression(&mut tokens2).expect("解析表达式2失败");

        let mut vars = std::collections::HashSet::new();
        collect_vars(&expr1, &mut vars);
        collect_vars(&expr2, &mut vars);
        let var_list: Vec<char> = vars.into_iter().collect();

        let n = var_list.len();
        for bits in 0..(1 << n) {
            let mut comb = std::collections::HashMap::new();
            for (i, var) in var_list.iter().enumerate() {
                comb.insert(*var, (bits >> i) & 1 == 1);
            }

            let res1 = evaluate(&expr1, &comb);
            let res2 = evaluate(&expr2, &comb);

            if res1 != res2 {
                /*
                let var_values = var_list
                    .iter()
                    .map(|c| format!("{}={}", c, comb[c]))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("不等价案例：{} => {} vs {}", var_values, res1, res2);
                */
                return false;
            }
        }

        true
    }

    #[test]
    fn process_json() {
        #[derive(serde::Serialize, serde::Deserialize, Debug)]
        struct JsonData {
            table: std::collections::HashMap<String, ffi::CCost>,
            bad_exprs: std::collections::HashSet<String>,
        }

        let data = std::fs::read_to_string("lib_expr2cost.json").unwrap();

        let mut json_data: JsonData = serde_json::from_str(&data).unwrap();

        // println!("{:#?}", json_data);

        let rules = rules();

        let mut update_count = 0;
        let item_size = json_data.table.len();
        while item_size > update_count {
            let mut processed_in_batch = 0;
            for (key_dep, entry) in json_data.table.iter_mut().skip(update_count) {
                let parts: Vec<&str> = key_dep.split('_').collect();
                let key = parts[0].to_string();
                let var_deps: Vec<u32> = parts[1..]
                    .iter()
                    .map(|s| s.parse::<u32>().unwrap())
                    .collect();

                // let key_expr: egg::RecExpr<MIG> = key.parse().unwrap();
                // let mut egraph = CEGraph::default();
                // let key_expr_id = egraph.add_expr(&key_expr);

                let single_expr = entry.aft_expr.len() == 1;
                if single_expr {
                    let opt_strs = simplify(&key, &var_deps);
                    if opt_strs.aft_dep < entry.aft_dep
                        || (opt_strs.aft_dep == entry.aft_dep && opt_strs.aft_size < entry.aft_size)
                    {
                        entry.aft_dep = opt_strs.aft_dep;
                        entry.aft_size = opt_strs.aft_size;
                        entry.aft_invs = opt_strs.aft_invs;
                        entry.aft_expr = opt_strs.aft_expr;

                        processed_in_batch += 1;
                        if processed_in_batch >= 100 {
                            break;
                        }
                    }
                }
                if false {
                    // check equivalence
                    for expr in &entry.aft_expr {
                        let mut egraph = CEGraph::default();
                        let expr_mig: egg::RecExpr<MIG> = expr.parse().unwrap();
                        let aft_expr_id = egraph.add_expr(&expr_mig);

                        /*
                        let runner = egg::Runner::default()
                            .with_egraph(egraph.clone())
                            .run(&rules);
                        if runner.egraph.find(key_expr_id) != runner.egraph.find(aft_expr_id) {
                            println!("Key '{}' is NOT equivalent to expression '{}'", key, expr);
                        }
                        */
                        if !is_equiv(&key, &expr) {
                            println!("Key '{}' is NOT equivalent to expression '{}'", key, expr);
                        }

                        let (aft_expr, aft_dep, aft_size, aft_invs) =
                            to_prefix(&egraph.id_to_expr(aft_expr_id), &var_deps);
                        if aft_dep != entry.aft_dep || aft_size != entry.aft_size {
                            println!(
                                "Key '{}' has different depth {}/{} and size {}/{}",
                                expr, aft_dep, entry.aft_dep, aft_size, entry.aft_size
                            );
                        }

                        if single_expr {
                            entry.aft_invs = aft_invs;
                        }
                    }
                }
            }

            update_count += processed_in_batch;
            if processed_in_batch > 0 {
                let file = std::fs::File::create("lib_expr2cost.json").unwrap();
                let mut writer = std::io::BufWriter::new(file);
                serde_json::to_writer_pretty(&mut writer, &json_data).unwrap();
                std::io::Write::flush(&mut writer).unwrap();
            } else {
                break;
            }
        }
    }

    #[test]
    fn prove_equiv_pair_with_core_rules() {
        let e1 = "(M 0 (~ f) (M a (~ (M (~ 0) c (M 0 (~ d) e))) (M (~ 0) a b)))";
        let e2 =
            "(M (~ e) (M 0 (~ f) (M a (M a 1 b) (~ c))) (M 0 (~ f) (M a d (M a (M a 1 b) (~ c)))))";

        // Build runner with our core rules
        let rewrites = rules();

        let expr1: egg::RecExpr<MIG> = e1.parse().unwrap();
        let expr2: egg::RecExpr<MIG> = e2.parse().unwrap();

        let mut egraph = CEGraph::default();
        let id1 = egraph.add_expr(&expr1);
        let id2 = egraph.add_expr(&expr2);

        let runner = egg::Runner::default()
            .with_egraph(egraph)
            .with_iter_limit(1000)
            .with_node_limit(100_000)
            .run(&rewrites);

        let unified = runner.egraph.find(id1) == runner.egraph.find(id2);
        let egraph_equiv = !runner.egraph.equivs(&expr1, &expr2).is_empty();
        assert!(
            is_equiv(e1, e2),
            "Expressions not equivalent by brute force"
        );
        assert!(
            (unified || egraph_equiv),
            "Expressions not proven equivalent by rules"
        );
    }

    #[test]
    fn const_fold() {
        let start = "(M 0 1 0)";
        let start_expr: egg::RecExpr<MIG> = start.parse().unwrap();
        let end = "0";
        let end_expr: egg::RecExpr<MIG> = end.parse().unwrap();
        let mut eg: CEGraph = CEGraph::default();
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

        let all_rules = rules();

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

        let var_dep = vec![2, 3, 1, 3, 0]; // 对应 a, b, c, d, e, f, g
        let expr: egg::RecExpr<MIG> =
            "(M 0 (~ (M 0 (M 0 (~ a) b) (M 0 (~ d) e))) (M 0 (~ d) (M 0 (~ c) e)))"
                .parse()
                .unwrap();

        let mut runner = egg::Runner::default().with_expr(&expr);
        let root = runner.roots[0];
        runner = runner.run(&all_rules);

        let (best_cost, best) = egg::Extractor::new(
            &runner.egraph,
            MIGCostFn_dsi::new(&runner.egraph, &var_dep, None),
        )
        .find_best(root);

        let (prefix_expr, depth, ops_count, inv_count) = to_prefix(&best, &var_dep);
        println!("default cost: {:?}", best_cost);
        println!("after_expr: {}", prefix_expr);
        println!("after_dep: {}", depth);
        println!("after_size: {}", ops_count);
        println!("after_invs: {}", inv_count);

        let expr = "Key '(M (~ 0) (M 0 (M (~ 0) a b) (~ (M 0 (M (~ 0) a b) c))) (M (~ 0) (M 0 c (~ (M 0 (M (~ 0) a b) c))) d))' is NOT equivalent to expression '(M 1 (M 0 (M 1 a b) (~ c)) (M c (~ (M b c (M 0 a c))) d))'";
        //let expr = "Key '(M (~ 0) (M 0 (~ a) (M (~ 0) g (M (~ e) (M c (~ (M (~ 0) b f)) (M 0 d (~ (M (~ 0) b f)))) (M 0 (~ e) (M (~ c) (M 0 b f) (M (~ d) f (M 0 b f))))))) (M 0 a (~ (M (~ 0) g (M (~ e) (M c (~ (M (~ 0) b f)) (M 0 d (~ (M (~ 0) b f)))) (M 0 (~ e) (M (~ c) (M 0 b f) (M (~ d) f (M 0 b f)))))))))' is NOT equivalent to expression '(~ (M 0 (~ (M 0 (M (M (M 1 b f) (~ c) (M 1 (M 1 b f) (~ d))) e (~ (M (~ c) (M f 0 (M b 0 b)) (M (M b f (~ d)) 0 (M f 0 f))))) (M 0 a (~ g)))) (M a (M (M (M 1 b f) (~ c) (M 1 (M 1 b f) (~ d))) e (~ (M (~ c) (M f 0 (M b 0 b)) (M (M b f (~ d)) 0 (M f 0 f))))) (M 1 a (~ g)))))'";
        //let expr = "Key '(M 0 (~ (M 0 c (M (~ 0) e f))) (M 0 b (~ (M 0 (~ f) (M 0 (M d (M (~ 0) (M 0 (~ a) c) (M 0 a (~ c))) (~ e)) (~ (M d (M (~ 0) (M 0 (~ a) c) (M 0 a (~ c))) e)))))))' is NOT equivalent to expression '(~ (M (M (M 1 (~ b) (M 0 c e)) (M c 1 (~ b)) f) (M 1 (M (M 1 (~ b) (M 0 c e)) (M c 1 (~ b)) f) (~ (M e d (M 0 (M c 1 a) (~ (M 0 a c)))))) (M 0 (~ f) (M d (M 0 (M c 1 a) (~ (M 0 a c))) (~ e)))))'";
        //let expr = "Key '(M (~ 0) (M 0 (~ a) (M g (~ (M (~ 0) (M 0 f (~ (M 0 (M (~ 0) e (M (~ 0) (M 0 c (~ d)) (M 0 (~ c) d))) (~ (M 0 e (M (~ 0) (M 0 c (~ d)) (M 0 (~ c) d))))))) (M 0 (M (~ 0) e (M (~ 0) (M 0 c (~ d)) (M 0 (~ c) d))) (~ (M (~ 0) f (M 0 e (M (~ 0) (M 0 c (~ d)) (M 0 (~ c) d)))))))) (M (~ 0) b g))) (M 0 a (~ (M g (~ (M (~ 0) (M 0 f (~ (M 0 (M (~ 0) e (M (~ 0) (M 0 c (~ d)) (M 0 (~ c) d))) (~ (M 0 e (M (~ 0) (M 0 c (~ d)) (M 0 (~ c) d))))))) (M 0 (M (~ 0) e (M (~ 0) (M 0 c (~ d)) (M 0 (~ c) d))) (~ (M (~ 0) f (M 0 e (M (~ 0) (M 0 c (~ d)) (M 0 (~ c) d)))))))) (M (~ 0) b g)))))' is NOT equivalent to expression '(M 0 (M 1 a (M g (M 0 (M 1 (M 0 (M 1 e (M 0 (~ (M 0 c d)) (M 1 d c))) (M (M 0 c d) 1 (~ (M 0 (M 1 d c) e)))) (~ f)) (M 1 f (M 0 (M 1 e (M 1 (M 0 c d) (~ (M 1 d c)))) (M 1 (M 0 (~ (M 0 c d)) (M 1 d c)) (~ e))))) (M 1 g b))) (~ (M 0 (M g (M 0 (M 1 (M 0 (M 1 e (M 0 (~ (M 0 c d)) (M 1 d c))) (M (M 0 c d) 1 (~ (M 0 (M 1 d c) e)))) (~ f)) (M 1 f (M 0 (M 1 e (M 1 (M 0 c d) (~ (M 1 d c)))) (M 1 (M 0 (~ (M 0 c d)) (M 1 d c)) (~ e))))) (M 1 g b)) a)))'";

        // let expr = "Key '(M (~ 0) (M 0 (~ (M 0 (~ a) b)) e) (M 0 e (~ (M 0 (~ (M 0 c d)) (M (~ 0) c d)))))' is NOT equivalent to expression '(M (M 1 a (~ b)) e (M e (~ (M 1 c d)) (M 0 c d)))'";
        // let expr = "Key '(M 0 (~ f) (M a (~ (M (~ 0) c (M 0 (~ d) e))) (M (~ 0) a b)))' is NOT equivalent to expression '(M (~ e) (M 0 (~ f) (M a (M a 1 b) (~ c))) (M 0 (~ f) (M a d (M a (M a 1 b) (~ c)))))'";
        let parts: Vec<&str> = expr.split('\'').collect();
        println!("parts: {:?}", parts);
        let expr1: egg::RecExpr<MIG> = parts[1].parse().unwrap(); //"(M x u (M y (~ u) z))".parse().unwrap();
        let expr2: egg::RecExpr<MIG> = parts[3].parse().unwrap(); //"(M x u (M y x z))".parse().unwrap();
        let mut egraph = CEGraph::default();

        let expr1_id = egraph.add_expr(&expr1);
        let expr2_id = egraph.add_expr(&expr2);

        let runner = egg::Runner::default().with_egraph(egraph).run(&all_rules);

        if runner.egraph.find(expr1_id) == runner.egraph.find(expr2_id) {
            println!("The expressions are equivalent!");
        } else {
            println!("The expressions are NOT equivalent!");
        }

        if runner.egraph.equivs(&expr1, &expr2).is_empty() {
            println!("The expressions are NOT equivalent!");
        } else {
            println!("The expressions are equivalent!");
        }

        if !is_equiv(&parts[1], &parts[3]) {
            println!("The expressions are NOT equivalent!");
        } else {
            println!("The expressions are equivalent!");
        }
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
        let rules = rules();
        prove_something(
            "chain",
            "(M x 0 (M y 1 (M u 0 v)))",
            &rules,
            &["(M x 0 (M y x (M u 0 v)))", "(M (M y x 0) x (M 0 u v))"],
        );
        */
        let empty_vec: Vec<u32> = Vec::new();

        /*
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
        */
        simplify("(M a b (M a b c))", &empty_vec);
        simplify("(M x 0 (M y 1 (M u 0 v)))", &empty_vec);
        simplify("(M (M w x (~ z)) x (M z x y))", &empty_vec);
        simplify("(M c (M c d (M e f b)) a)", &empty_vec);
        simplify(
            "(M (~ 0) (M 0 c (~ (M 0 (M (~ 0) a b) (~ (M 0 a b))))) (M 0 (~ c) (M 0 (M (~ 0) a b) (~ (M 0 a b)))))",
            &empty_vec,
        );
        simplify(
            "(M (~ 0) (M 0 (M 0 a c) (~ (M 0 (M (~ 0) b d) (~ (M 0 b d))))) (M 0 (~ (M 0 a c)) (M 0 (M (~ 0) b d) (~ (M 0 b d)))))",
            &empty_vec,
        );
        simplify("(M 0 (~ (M 0 (~ a) b)) (M 0 c (~ d)))", &empty_vec);
        simplify(
            "(M (~ 0) (M 0 a (~ (M 0 b (~ c)))) (M 0 (~ a) (M 0 b (~ c))))",
            &empty_vec,
        );
        simplify("(M (~ 0) (M 0 a (~ b)) (M 0 (~ a) b))", &empty_vec);
        simplify(
            "(M (~ 0) (M (~ e) (M (~ 0) c (M 0 (~ a) b)) (M 0 (M 0 c (M 0 (~ a) b)) (M (~ 0) c (M 0 (~ a) b)))) (M (M 0 (~ d) g) h (M 0 (~ f) h)))",
            &empty_vec,
        );
        simplify(
            "(M 0 b (~ (M 0 (~ (M g (M 0 d (M a c (~ f))) (M e (M a c (~ f)) g))) (M 0 (M (~ 0) d (M a c (~ f))) g))))",
            &empty_vec,
        );
        simplify(
            "(M (~ 0) (M 0 (M 0 c (~ (M (~ 0) (M 0 a (~ b)) (M 0 (~ a) b)))) h) (M (M 0 (~ c) d) (M 0 e (~ f)) (~ (M 0 (M 0 (~ c) d) g))))",
            &empty_vec,
        );
        simplify(
            "(M (~ 0) (M 0 (M 0 c (~ (M (~ 0) (M 0 a (~ b)) (M 0 (~ a) b)))) h) (M (M 0 (~ c) d) (M 0 e (~ f)) (~ (M 0 (M 0 (~ c) d) g))))",
            &vec![0, 0, 2, 2, 4, 6, 5, 7],
        );
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
            // worse than baseline
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
            // worse than baseline
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
            //worse than baseline
            "(M (~ 0) (M 0 (~ c) (M (~ 0) e (M (~ (M (~ 0) a b)) (M 0 (M 0 a (~ b)) h) (M (~ 0) (M (~ 0) a b) g)))) (M 0 d f))",
            &vec![0, 0, 2, 2, 6, 7, 4, 4],
        );
        simplify(
            "(M (~ 0) (M (~ e) (M 0 e (M 0 a d)) (M (~ 0) e f)) (M (~ e) (M 0 e (M 0 b (~ d))) (M (~ 0) e (M 0 c d))))",
            &vec![0, 0, 0, 0, 0, 5],
        );
    }
}
