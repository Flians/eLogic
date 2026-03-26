use egg::Id;
use std::collections::HashSet;
use std::ffi::CString;
use std::fmt;
use std::os::raw::c_char;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
pub struct CCost {
    pub dep: u32,
    pub aom: u32,
    pub inv: u32,
}

impl fmt::Display for CCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dep={}, aom={}, inv={}", self.dep, self.aom, self.inv)
    }
}

egg::define_language! {
    pub enum AIG {
        Bool(u8),
        "&" = And([Id; 2]),
        "~" = Not(Id),
        Symbol(egg::Symbol),
    }
}

#[derive(Default, Clone)]
pub struct ConstantFold;

type EGraph = egg::EGraph<AIG, ConstantFold>;
type Rewrite = egg::Rewrite<AIG, ConstantFold>;
type GateEGraph = EGraph;

/// Compute a 64-bit truth table for an expression with up to 6 variables.
///
/// Variables are expected to be single-letter symbols in `a`..`f`.
/// Returns `(n_inputs, tt64)` where bit `k` corresponds to assignment `k`.
#[must_use]
pub fn sexp_to_tt64(sexp: &str) -> Option<(u8, u64)> {
    let expr: egg::RecExpr<AIG> = sexp.parse().ok()?;
    let mut n_inputs = 0u8;

    for node in expr.as_ref() {
        if let AIG::Symbol(sym) = node {
            let s = sym.as_str().as_bytes();
            if s.len() != 1 {
                return None;
            }
            let c = s[0];
            if !(b'a'..=b'f').contains(&c) {
                return None;
            }
            n_inputs = n_inputs.max(c - b'a' + 1);
        }
    }

    if n_inputs > 6 {
        return None;
    }

    let mask = match n_inputs {
        0 => 1u64,
        6 => u64::MAX,
        _ => (1u64 << (1u32 << n_inputs)) - 1,
    };

    let var_tt = |idx: u8, n: u8| -> u64 {
        let mut tt = 0u64;
        let n = n as usize;
        for a in 0..(1u64 << n) {
            if ((a >> (idx as u64)) & 1) == 1 {
                tt |= 1u64 << a;
            }
        }
        tt
    };

    let mut values = Vec::<u64>::with_capacity(expr.as_ref().len());
    for node in expr.as_ref() {
        let v = match node {
            AIG::Bool(0) => 0u64,
            AIG::Bool(1) => mask,
            AIG::Bool(_) => return None,
            AIG::Symbol(sym) => {
                let c = sym.as_str().as_bytes()[0];
                var_tt(c - b'a', n_inputs) & mask
            }
            AIG::Not(a) => (!values[usize::from(*a)]) & mask,
            AIG::And([a, b]) => values[usize::from(*a)] & values[usize::from(*b)],
        };
        values.push(v);
    }

    let root = *values.last()?;
    Some((n_inputs, root & mask))
}

impl egg::Analysis<AIG> for ConstantFold {
    type Data = Option<(u8, egg::PatternAst<AIG>)>;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> egg::DidMerge {
        match (to.take(), from) {
            (None, None) => egg::DidMerge(false, false),
            (None, some_from @ Some(_)) => {
                *to = some_from;
                egg::DidMerge(true, false)
            }
            (some_to @ Some(_), None) => {
                *to = some_to;
                egg::DidMerge(false, false)
            }
            (Some(a), Some(b)) => {
                if a.0 == b.0 {
                    *to = Some(a);
                } else {
                    // Conservative fallback: conflicting constants mean this e-class
                    // cannot be treated as a known constant.
                    *to = None;
                }
                egg::DidMerge(true, true)
            }
        }
    }

    fn make(egraph: &mut EGraph, enode: &AIG, _id: Id) -> Self::Data {
        let x = |i: &Id| egraph[*i].data.as_ref().map(|c| c.0);
        match enode {
            AIG::Bool(c) => Some((*c, c.to_string().parse().ok()?)),
            AIG::Symbol(_) => None,
            AIG::And([a, b]) => {
                let v = (x(a)? & x(b)? == 1) as u8;
                Some((v, v.to_string().parse().ok()?))
            }
            AIG::Not(a) => {
                let v = 1u8 ^ x(a)?;
                Some((v, v.to_string().parse().ok()?))
            }
        }
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        if let Some(c) = egraph[id].data.clone() {
            egraph.union_instantiations(
                &c.1,
                &c.0.to_string()
                    .parse()
                    .unwrap_or_else(|_| "0".parse().unwrap()),
                &Default::default(),
                "const-fold".to_string(),
            );
        }
    }
}

macro_rules! rule {
    ($name:ident, $lhs:literal, $rhs:literal) => {
        fn $name() -> Rewrite {
            egg::rewrite!(stringify!($name); $lhs => $rhs)
        }
    };
    ($name:ident, $name2:ident, $lhs:literal, $rhs:literal) => {
        rule!($name, $lhs, $rhs);
        rule!($name2, $rhs, $lhs);
    };
}

rule!(double_neg, "(~ (~ ?a))", "?a");
rule!(not_zero, "(~ 0)", "1");
rule!(not_one, "(~ 1)", "0");
rule!(and_comm, "(& ?a ?b)", "(& ?b ?a)");
rule!(and_assoc, "(& ?a (& ?b ?c))", "(& ?b (& ?a ?c))");
rule!(and_comp, "(& ?a (~ ?a))", "0");
rule!(and_id, "(& ?a ?a)", "?a");
rule!(and_true, "(& ?a 1)", "?a");
rule!(and_false, "(& ?a 0)", "0");
rule!(demorgan_and, "(~ (& ?a ?b))", "(& (~ ?a) (~ ?b))");

fn rules() -> Vec<Rewrite> {
    vec![
        double_neg(),
        not_zero(),
        not_one(),
        and_comm(),
        and_assoc(),
        and_comp(),
        and_id(),
        and_true(),
        and_false(),
        demorgan_and(),
    ]
}

fn parse_var_dep(vars: *const u32, size: usize) -> Vec<u32> {
    if vars.is_null() || size == 0 {
        vec![0; 26]
    } else {
        // SAFETY: cxx passes a valid pointer/size pair for the call duration.
        unsafe { std::slice::from_raw_parts(vars, size) }.to_vec()
    }
}

fn cache_store() -> &'static Mutex<std::collections::HashMap<String, ffi::CCost>> {
    static CACHE: OnceLock<Mutex<std::collections::HashMap<String, ffi::CCost>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn make_cache_key(s: &str, vars: &[u32], objective: u8) -> String {
    let mut key = String::with_capacity(s.len() + vars.len() * 4 + 16);
    key.push_str(s);
    key.push('|');
    key.push_str(&objective.to_string());
    for v in vars {
        key.push('_');
        key.push_str(&v.to_string());
    }
    key
}

fn run_extract(s: &str, var_dep: &[u32], first_depth: bool) -> Option<(String, CCost)> {
    let _ = sexp_to_tt64(s)?;
    let (egraph, root) = rewrite_to_gategraph(s)?;
    extract_cover_from_egraph(&egraph, root, var_dep, first_depth)
}

fn rewrite_to_gategraph(s: &str) -> Option<(GateEGraph, Id)> {
    let expr: egg::RecExpr<AIG> = s.parse().ok()?;
    if expr.as_ref().is_empty() {
        return None;
    }
    let runner = egg::Runner::default()
        .with_expr(&expr)
        .with_iter_limit(40)
        .with_node_limit(20000)
        .with_time_limit(Duration::from_secs(2))
        .run(&rules());
    let root = *runner.roots.first()?;
    Some((runner.egraph, root))
}

#[derive(Clone, Debug)]
enum CoverChoiceKind {
    Const(bool),
    Input(usize),
    Not(usize),
    And(usize, usize),
}

#[derive(Clone, Debug)]
struct CoverChoice {
    cost: CCost,
    kind: CoverChoiceKind,
    class_areas: std::collections::HashMap<usize, u32>,
}

fn merge_class_areas(
    lhs: &std::collections::HashMap<usize, u32>,
    rhs: &std::collections::HashMap<usize, u32>,
) -> std::collections::HashMap<usize, u32> {
    let mut out = lhs.clone();
    for (k, v) in rhs {
        out.entry(*k).or_insert(*v);
    }
    out
}

fn better_cover_choice(lhs: &CoverChoice, rhs: &CoverChoice, first_depth: bool) -> bool {
    if first_depth {
        (lhs.cost.dep, lhs.cost.aom, lhs.cost.inv) < (rhs.cost.dep, rhs.cost.aom, rhs.cost.inv)
    } else {
        (lhs.cost.aom, lhs.cost.dep, lhs.cost.inv) < (rhs.cost.aom, rhs.cost.dep, rhs.cost.inv)
    }
}

fn extract_cover_from_egraph(
    egraph: &GateEGraph,
    root: Id,
    var_dep: &[u32],
    first_depth: bool,
) -> Option<(String, CCost)> {
    let mut best = std::collections::HashMap::<usize, Option<CoverChoice>>::new();

    fn best_eclass(
        id: Id,
        egraph: &GateEGraph,
        var_dep: &[u32],
        first_depth: bool,
        best: &mut std::collections::HashMap<usize, Option<CoverChoice>>,
        visiting: &mut std::collections::HashSet<usize>,
    ) -> Option<CoverChoice> {
        let key = usize::from(id);
        if let Some(v) = best.get(&key) {
            return v.clone();
        }
        if !visiting.insert(key) {
            return None;
        }

        let mut choice: Option<CoverChoice> = None;
        for enode in &egraph[id].nodes {
            let op = match enode {
                AIG::Bool(v) => Some(CoverChoice {
                    cost: CCost {
                        dep: 0,
                        aom: 0,
                        inv: 0,
                    },
                    kind: CoverChoiceKind::Const(*v == 1),
                    class_areas: std::collections::HashMap::new(),
                }),
                AIG::Symbol(sym) => {
                    let bytes = sym.as_str().as_bytes();
                    if bytes.len() != 1 || !(b'a'..=b'f').contains(&bytes[0]) {
                        None
                    } else {
                        let i = (bytes[0] - b'a') as usize;
                        Some(CoverChoice {
                            cost: CCost {
                                dep: var_dep.get(i).copied().unwrap_or(0),
                                aom: 0,
                                inv: 0,
                            },
                            kind: CoverChoiceKind::Input(i),
                            class_areas: std::collections::HashMap::new(),
                        })
                    }
                }
                AIG::Not(a) => {
                    let child = best_eclass(*a, egraph, var_dep, first_depth, best, visiting)?;
                    Some(CoverChoice {
                        cost: CCost {
                            dep: child.cost.dep,
                            aom: child.cost.aom,
                            inv: child.cost.inv + 1,
                        },
                        kind: CoverChoiceKind::Not(usize::from(*a)),
                        class_areas: child.class_areas.clone(),
                    })
                }
                AIG::And([a, b]) => {
                    let lhs = best_eclass(*a, egraph, var_dep, first_depth, best, visiting)?;
                    let rhs = best_eclass(*b, egraph, var_dep, first_depth, best, visiting)?;
                    let merged = merge_class_areas(&lhs.class_areas, &rhs.class_areas);
                    let mut class_areas = merged.clone();
                    class_areas.insert(key, 1);
                    Some(CoverChoice {
                        cost: CCost {
                            dep: lhs.cost.dep.max(rhs.cost.dep) + 1,
                            aom: class_areas.values().sum(),
                            inv: lhs.cost.inv + rhs.cost.inv,
                        },
                        kind: CoverChoiceKind::And(usize::from(*a), usize::from(*b)),
                        class_areas,
                    })
                }
            };

            choice = match (choice, op) {
                (Some(cur), Some(newc)) if better_cover_choice(&newc, &cur, first_depth) => {
                    Some(newc)
                }
                (None, Some(newc)) => Some(newc),
                (cur, _) => cur,
            };
        }

        visiting.remove(&key);
        best.insert(key, choice.clone());
        choice
    }

    let mut visiting = std::collections::HashSet::<usize>::new();
    let root_choice = best_eclass(root, egraph, var_dep, first_depth, &mut best, &mut visiting)?;

    fn materialize(
        id: usize,
        best: &std::collections::HashMap<usize, Option<CoverChoice>>,
        memo: &mut std::collections::HashMap<usize, String>,
    ) -> Option<String> {
        if let Some(s) = memo.get(&id) {
            return Some(s.clone());
        }
        let c = best.get(&id)?.as_ref()?;
        let s = match &c.kind {
            CoverChoiceKind::Const(v) => {
                if *v {
                    "1".to_string()
                } else {
                    "0".to_string()
                }
            }
            CoverChoiceKind::Input(i) => ((b'a' + (*i as u8)) as char).to_string(),
            CoverChoiceKind::Not(a) => format!("(~ {})", materialize(*a, best, memo)?),
            CoverChoiceKind::And(a, b) => {
                format!(
                    "(& {} {})",
                    materialize(*a, best, memo)?,
                    materialize(*b, best, memo)?
                )
            }
        };
        memo.insert(id, s.clone());
        Some(s)
    }

    let mut memo = std::collections::HashMap::<usize, String>::new();
    let expr = materialize(usize::from(root), &best, &mut memo)?;
    Some((expr, root_choice.cost))
}

#[cxx::bridge]
mod ffi {
    struct CCost {
        aft_expr: Vec<String>,
        aft_dep: u32,
        aft_size: u32,
    }

    extern "Rust" {
        unsafe fn cut_mapping(s: &str, vars: *const u32, size: usize) -> CCost;
        unsafe fn cut_mapping_with_objective(
            s: &str,
            vars: *const u32,
            size: usize,
            objective: u8,
        ) -> CCost;
        fn merge_ccost(depth: &CCost, size: &CCost) -> CCost;
        unsafe fn free_string(s: *mut c_char);
    }
}

fn merge_ccost(depth: &ffi::CCost, size: &ffi::CCost) -> ffi::CCost {
    if (depth.aft_dep, depth.aft_size) <= (size.aft_dep, size.aft_size) {
        depth.clone()
    } else {
        size.clone()
    }
}

impl Clone for ffi::CCost {
    fn clone(&self) -> Self {
        Self {
            aft_expr: self.aft_expr.clone(),
            aft_dep: self.aft_dep,
            aft_size: self.aft_size,
        }
    }
}

pub fn free_string(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: ownership was transferred via CString::into_raw.
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

pub fn cut_mapping(s: &str, vars: *const u32, size: usize) -> ffi::CCost {
    cut_mapping_with_objective(s, vars, size, 2)
}

pub fn cut_mapping_with_objective(
    s: &str,
    vars: *const u32,
    size: usize,
    objective: u8,
) -> ffi::CCost {
    let var_dep = parse_var_dep(vars, size);
    let cache_key = make_cache_key(s, &var_dep, objective);
    if let Ok(guard) = cache_store().lock() {
        if let Some(v) = guard.get(&cache_key) {
            return v.clone();
        }
    }

    let out = match objective {
        0 | 1 => {
            // Evaluate both extraction orders and keep all ties for requested objective.
            let a = run_extract(s, &var_dep, true);
            let b = run_extract(s, &var_dep, false);
            match (a, b) {
                (Some((ea, ca)), Some((eb, cb))) => {
                    let key_a = if objective == 0 {
                        (ca.dep, ca.aom)
                    } else {
                        (ca.aom, ca.dep)
                    };
                    let key_b = if objective == 0 {
                        (cb.dep, cb.aom)
                    } else {
                        (cb.aom, cb.dep)
                    };
                    let mut exprs = HashSet::new();
                    let best_cost = if key_a < key_b {
                        exprs.insert(ea);
                        ca
                    } else if key_b < key_a {
                        exprs.insert(eb);
                        cb
                    } else {
                        exprs.insert(ea);
                        exprs.insert(eb);
                        // equal objective key; keep stable representative cost
                        if (ca.dep, ca.aom) <= (cb.dep, cb.aom) { ca } else { cb }
                    };
                    ffi::CCost {
                        aft_expr: exprs.into_iter().collect(),
                        aft_dep: best_cost.dep,
                        aft_size: best_cost.aom,
                    }
                }
                (Some((e, c)), None) | (None, Some((e, c))) => ffi::CCost {
                    aft_expr: vec![e],
                    aft_dep: c.dep,
                    aft_size: c.aom,
                },
                (None, None) => ffi::CCost {
                    aft_expr: vec![s.to_string()],
                    aft_dep: 0,
                    aft_size: u32::MAX / 2,
                },
            }
        }
        _ => {
            // fallback: evaluate both only for unknown objective
            let a = run_extract(s, &var_dep, true);
            let b = run_extract(s, &var_dep, false);
            match (a, b) {
                (Some((ea, ca)), Some((eb, cb))) => {
                    let best = if (ca.dep, ca.aom) <= (cb.dep, cb.aom) {
                        (ea, ca)
                    } else {
                        (eb, cb)
                    };
                    let mut exprs = HashSet::new();
                    exprs.insert(best.0.clone());
                    ffi::CCost {
                        aft_expr: exprs.into_iter().collect(),
                        aft_dep: best.1.dep,
                        aft_size: best.1.aom,
                    }
                }
                (Some((e, c)), None) | (None, Some((e, c))) => ffi::CCost {
                    aft_expr: vec![e],
                    aft_dep: c.dep,
                    aft_size: c.aom,
                },
                (None, None) => ffi::CCost {
                    aft_expr: vec![s.to_string()],
                    aft_dep: 0,
                    aft_size: u32::MAX / 2,
                },
            }
        }
    };

    if let Ok(mut guard) = cache_store().lock() {
        guard.insert(cache_key, out.clone());
    }
    out
}
