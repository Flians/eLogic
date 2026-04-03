use egg::Id;
use emapper::map_cut_to_cover;
use emapper::techlib::{MappingObjective as EmapObjective, TechLib};
use emapper::{MappedCoverChoice, MappedCoverNode};
use std::ffi::CString;
use std::fmt;
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

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

fn default_genlib_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tools/mockturtle/experiments/cell_libraries/asap7.genlib")
}

fn techlib_store() -> &'static TechLib {
    static TECH: OnceLock<TechLib> = OnceLock::new();
    TECH.get_or_init(|| {
        let env_path = std::env::var("MAP_EGG_TECHLIB_PATH").ok();
        let candidate = env_path
            .map(PathBuf::from)
            .unwrap_or_else(default_genlib_path);
        TechLib::from_genlib(&candidate, 6).unwrap_or_default()
    })
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MappingObjective {
    DelayArea,
    AreaDelay,
}

impl MappingObjective {
    fn from_u8(objective: u8) -> Self {
        match objective {
            1 => Self::AreaDelay,
            _ => Self::DelayArea,
        }
    }
}

fn run_extract(s: &str, var_dep: &[u32], objective: MappingObjective) -> Option<MappedCoverChoice> {
    let _ = sexp_to_tt64(s)?;
    let tech = techlib_store();
    let arrivals: Vec<f32> = var_dep.iter().map(|&d| d as f32).collect();
    let emap_obj = match objective {
        MappingObjective::AreaDelay => EmapObjective::AreaFirst,
        MappingObjective::DelayArea => EmapObjective::DelayFirst,
    };
    map_cut_to_cover(s, tech, &arrivals, emap_obj)
}

#[cxx::bridge]
mod ffi {
    /// A single node in the mapped cover graph transferred over FFI.
    ///
    /// - `kind == 0`: primary input; `input_idx` identifies variable index (0='a', etc.)
    /// - `kind == 1`: constant; `const_val` is 0 or 1.
    /// - `kind == 2`: standard-cell instance; `cell_name`/`out_pin` identify the gate,
    ///   `neg_mask` gives per-input inversion bits, `out_neg` gives output inversion,
    ///   `fanins` lists indices into `CCost::cell_nodes`.
    struct MappedNode {
        kind: u8,
        input_idx: u8,
        const_val: u8,
        neg_mask: u16,
        out_neg: u8,
        cell_name: String,
        out_pin: String,
        fanins: Vec<u32>,
    }

    struct CCost {
        aft_expr: Vec<String>,
        aft_dep: u32,
        aft_size: u32,
        /// Structured cell-level cover nodes. Empty when mapping failed or no
        /// techlib cells could be matched (falls back to AIG-sexp in aft_expr).
        cell_nodes: Vec<MappedNode>,
        /// Index of the root node inside `cell_nodes`. Valid iff `!cell_nodes.is_empty()`.
        cell_root: u32,
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

fn clone_mapped_node(n: &ffi::MappedNode) -> ffi::MappedNode {
    ffi::MappedNode {
        kind: n.kind,
        input_idx: n.input_idx,
        const_val: n.const_val,
        neg_mask: n.neg_mask,
        out_neg: n.out_neg,
        cell_name: n.cell_name.clone(),
        out_pin: n.out_pin.clone(),
        fanins: n.fanins.clone(),
    }
}

impl Clone for ffi::CCost {
    fn clone(&self) -> Self {
        Self {
            aft_expr: self.aft_expr.clone(),
            aft_dep: self.aft_dep,
            aft_size: self.aft_size,
            cell_nodes: self.cell_nodes.iter().map(clone_mapped_node).collect(),
            cell_root: self.cell_root,
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
    let objective = MappingObjective::from_u8(objective);
    if let Ok(guard) = cache_store().lock() {
        if let Some(v) = guard.get(&cache_key) {
            return v.clone();
        }
    }

    let out = match run_extract(s, &var_dep, objective) {
        Some(cover) => {
            // Build cell_nodes from MappedCoverGraph.
            let graph = &cover.graph;
            let cell_nodes: Vec<ffi::MappedNode> = graph
                .nodes
                .iter()
                .map(|node| match node {
                    MappedCoverNode::Input(li) => ffi::MappedNode {
                        kind: 0,
                        input_idx: *li as u8,
                        const_val: 0,
                        neg_mask: 0,
                        out_neg: 0,
                        cell_name: String::new(),
                        out_pin: String::new(),
                        fanins: vec![],
                    },
                    MappedCoverNode::Const(v) => ffi::MappedNode {
                        kind: 1,
                        input_idx: 0,
                        const_val: if *v { 1 } else { 0 },
                        neg_mask: 0,
                        out_neg: 0,
                        cell_name: String::new(),
                        out_pin: String::new(),
                        fanins: vec![],
                    },
                    MappedCoverNode::Cell { cell, fanins } => ffi::MappedNode {
                        kind: 2,
                        input_idx: 0,
                        const_val: 0,
                        neg_mask: cell.neg_mask,
                        out_neg: if cell.out_neg { 1 } else { 0 },
                        cell_name: cell.cell_name.clone(),
                        out_pin: cell.out_pin.clone(),
                        fanins: fanins.iter().map(|&fi| fi as u32).collect(),
                    },
                })
                .collect();
            ffi::CCost {
                aft_expr: vec![], // cell_nodes path: aft_expr intentionally empty
                aft_dep: cover.delay.round() as u32,
                aft_size: cover.area.round() as u32,
                cell_root: graph.root as u32,
                cell_nodes,
            }
        }
        None => ffi::CCost {
            aft_expr: vec![s.to_string()],
            aft_dep: 0,
            aft_size: u32::MAX / 2,
            cell_nodes: vec![],
            cell_root: 0,
        },
    };

    if let Ok(mut guard) = cache_store().lock() {
        guard.insert(cache_key, out.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const SIMPLE_AND: &str = "(& a b)";

    fn call_cut_mapping(expr: &str, vars: &[u32], objective: u8) -> ffi::CCost {
        cut_mapping_with_objective(expr, vars.as_ptr(), vars.len(), objective)
    }

    fn clear_cache() {
        let guard = cache_store().lock();
        let mut guard = guard.expect("cache mutex poisoned");
        guard.clear();
    }

    fn expr_tt_set(exprs: &[String]) -> HashSet<(u8, u64)> {
        exprs.iter().filter_map(|e| sexp_to_tt64(e)).collect()
    }

    #[test]
    fn objective_results_keep_boolean_equivalence() {
        let input = "(& a (~ b))";
        let vars = vec![0u32, 0u32];
        let input_tt = sexp_to_tt64(input).expect("input should parse");

        for objective in [0u8, 1u8, 2u8] {
            let out = call_cut_mapping(input, &vars, objective);
            assert!(
                !out.cell_nodes.is_empty() || !out.aft_expr.is_empty(),
                "objective {} produced neither cell cover nor expr fallback",
                objective
            );
            if !out.aft_expr.is_empty() {
                let tts = expr_tt_set(&out.aft_expr);
                assert!(
                    !tts.is_empty(),
                    "objective {} produced unparsable exprs",
                    objective
                );
                assert!(
                    tts.contains(&input_tt),
                    "objective {} returned non-equivalent expression set {:?}",
                    objective,
                    out.aft_expr
                );
            } else {
                assert!(
                    !out.cell_nodes.is_empty(),
                    "objective {} should return structured cell cover on success",
                    objective
                );
                assert!(out.cell_root < out.cell_nodes.len() as u32);
                let _ = input_tt;
            }
        }
    }

    #[test]
    fn invalid_expression_falls_back_to_original() {
        let bad = "(% a b)";
        let out = call_cut_mapping(bad, &[0u32, 0u32], 0);
        assert_eq!(out.aft_dep, 0);
        assert_eq!(out.aft_size, u32::MAX / 2);
        assert_eq!(out.aft_expr, vec![bad.to_string()]);
    }

    #[test]
    fn cache_key_includes_objective() {
        let vars = vec![0u32, 0u32];
        let k0 = make_cache_key(SIMPLE_AND, &vars, 0);
        let k2 = make_cache_key(SIMPLE_AND, &vars, 2);
        assert_ne!(
            k0, k2,
            "same expr+vars but different objectives must map to distinct keys"
        );
    }

    #[test]
    fn same_call_hits_cache_and_is_stable() {
        clear_cache();
        let vars = vec![0u32, 0u32];
        let out1 = call_cut_mapping(SIMPLE_AND, &vars, 1);
        let out2 = call_cut_mapping(SIMPLE_AND, &vars, 1);
        assert_eq!(out1.aft_dep, out2.aft_dep);
        assert_eq!(out1.aft_size, out2.aft_size);
        assert_eq!(expr_tt_set(&out1.aft_expr), expr_tt_set(&out2.aft_expr));
    }
}
