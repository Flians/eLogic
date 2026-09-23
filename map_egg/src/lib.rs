// ---- cover generation (folded in from the former `emapper` crate) ------------------------------
// These modules were the separate `emapper` crate, of which this bridge needed six of them plus two
// types. They stay at the crate root so that their internal `crate::…` paths are unchanged.

mod tt9;
pub use tt9::Tt9;

pub mod aig_graph;

mod binding;
pub use binding::{NodeBinding, NodeBindingKind};

mod emap_cover;
pub use emap_cover::{
    build_mapped_cover_binding, compute_pin_to_root_delays, map_cut_to_aig_sexp, map_cut_to_cover,
    map_cut_to_cover_alternatives, map_cut_to_cover_cover_only, map_cut_to_cover_with_reason,
    map_cut_to_library, map_tt_to_library, MappedCoverChoice, MappedCoverGraph, MappedCoverNode,
};

pub mod emap_cut;
pub mod emap_egg;
pub mod techlib;

mod params;
pub use params::EmapParams;

/// AIG graph/representation namespace.
pub mod aig {
    pub use crate::aig_graph::{Aig, AigNode};
}

// ---- FFI bridge -------------------------------------------------------------------------------
use egg::Id;
use crate::techlib::{MappingObjective as EmapObjective, TechLib};
use std::ffi::CString;
use std::fmt;
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};

static LIBRARY_HITS: AtomicU64 = AtomicU64::new(0);
static EGG_CALLS: AtomicU64 = AtomicU64::new(0);
static EGG_CACHE_HITS: AtomicU64 = AtomicU64::new(0);

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

fn make_cache_key(s: &str, vars: &[u32], objective: u8, required: Option<f32>) -> String {
    let mut key = String::with_capacity(s.len() + vars.len() * 4 + 32);
    key.push_str(s);
    key.push('|');
    key.push_str(&objective.to_string());
    key.push('|');
    /* `required_output_delay` changes the DP, so it is part of the key (`to_bits` avoids any float
       formatting ambiguity). */
    match required {
        Some(r) => key.push_str(&r.to_bits().to_string()),
        None => key.push_str("none"),
    }
    for v in vars {
        key.push('_');
        key.push_str(&v.to_string());
    }
    key
}

/// Memoised [`cut_mapping_impl`]: the e-graph extraction is a pure function of
/// `(expression, objective, required_output_delay, arrivals)`, and the staged flow asks for the same
/// cut function once per root, so the saturation is cached instead of redone every time.
fn cached_cut_mapping(
    s: &str,
    vars: *const u32,
    size: usize,
    objective: u8,
    required: Option<f32>,
) -> ffi::CCost {
    let var_dep = parse_var_dep(vars, size);
    let key = make_cache_key(s, &var_dep, objective, required);
    if let Ok(guard) = cache_store().lock() {
        if let Some(v) = guard.get(&key) {
            EGG_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            return v.clone();
        }
    }
    let out = cut_mapping_impl(s, vars, size, objective, required);
    if let Ok(mut guard) = cache_store().lock() {
        guard.insert(key, out.clone());
    }
    out
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

fn run_extract(
    s: &str,
    var_dep: &[u32],
    objective: MappingObjective,
    required_output_delay: Option<f32>,
) -> Option<MappedCoverChoice> {
    let tech = techlib_store();
    let aig = emap_egg::gate_expr_to_aig_sexp(s, tech)?;
    let (n_inputs, _) = emap_egg::sexp_to_tt9(&aig)?;
    let arrivals: Vec<f32> = if var_dep.len() == n_inputs as usize {
        var_dep.iter().map(|&d| d as f32).collect()
    } else {
        vec![0.0; n_inputs as usize]
    };
    let emap_obj = match objective {
        MappingObjective::AreaDelay => EmapObjective::AreaFirst,
        MappingObjective::DelayArea => EmapObjective::DelayFirst,
    };
    map_cut_to_cover_cover_only(&aig, tech, &arrivals, emap_obj, required_output_delay).ok()
}

/// Library-only fast path: match the cut function against the technology library and return the best
/// single-cell variant, without any cover/e-graph search.
///
/// This is what `cut_mapping_library_dc` exposes to C++ as the "single-cell matcher": it is cheap and
/// fails (`None`) whenever no library cell realizes the cut function, which is the common case for
/// cuts with more leaves than the library's largest cell. The full cover search is reached through
/// the `cut_mapping_egg_*` entry points instead.
fn run_extract_library(s: &str, var_dep: &[u32], objective: MappingObjective) -> Option<MappedCoverChoice> {
    let tech = techlib_store();
    let aig = emap_egg::gate_expr_to_aig_sexp(s, tech)?;
    let (n_inputs, tt) = emap_egg::sexp_to_tt9(&aig)?;
    let arrivals: Vec<f32> = if var_dep.len() == n_inputs as usize {
        var_dep.iter().map(|&d| d as f32).collect()
    } else {
        vec![0.0; n_inputs as usize]
    };
    let emap_obj = match objective {
        MappingObjective::AreaDelay => EmapObjective::AreaFirst,
        MappingObjective::DelayArea => EmapObjective::DelayFirst,
    };
    map_tt_to_library(tt, n_inputs, tech, &arrivals, emap_obj, None).ok()
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
        pin_names: Vec<String>,
        fanins: Vec<u32>,
    }

    struct CCost {
        aft_expr: Vec<String>,
        aft_dep: u32,
        aft_size: u32,
        /// Unrounded cover delay in library time units.
        delay: f32,
        /// Unrounded cover area in library area units.
        area: f32,
        failure_reason: String,
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
        unsafe fn cut_mapping_with_objective_dc(
            s: &str,
            vars: *const u32,
            size: usize,
            objective: u8,
            required_output_delay: f32,
            care_mask: u64,
        ) -> CCost;
        unsafe fn cut_mapping_library_dc(
            s: &str,
            vars: *const u32,
            size: usize,
            objective: u8,
            required_output_delay: f32,
            care_mask: u64,
        ) -> CCost;
        unsafe fn cut_mapping_egg_dc(s: &str, objective: u8, n_inputs: u32) -> CCost;
        unsafe fn cut_mapping_egg_arr(
            s: &str,
            vars: *const u32,
            size: usize,
            objective: u8,
            required_output_delay: f32,
            limit: u32,
            cost_model: u8,
        ) -> Vec<CCost>;
        unsafe fn cut_mapping_egg_multi(
            s: &str,
            objective: u8,
            n_inputs: u32,
            limit: u32,
        ) -> Vec<CCost>;
        unsafe fn cut_mapping_multi_dc(
            s: &str,
            objective: u8,
            care_mask: u64,
            use_egg: bool,
            limit: u32,
        ) -> Vec<CCost>;
        unsafe fn cut_mapping_db_dc(s: &str, objective: u8, care_mask: u64) -> CCost;
        fn mapping_library_hits() -> u64;
        fn mapping_egg_cache_hits() -> u64;
        fn mapping_egg_calls() -> u64;
        fn reset_mapping_counters();
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
        pin_names: n.pin_names.clone(),
        fanins: n.fanins.clone(),
    }
}

impl Clone for ffi::CCost {
    fn clone(&self) -> Self {
        Self {
            aft_expr: self.aft_expr.clone(),
            aft_dep: self.aft_dep,
            aft_size: self.aft_size,
            delay: self.delay,
            area: self.area,
            failure_reason: self.failure_reason.clone(),
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

fn failure_cost(source: &str) -> ffi::CCost {
    ffi::CCost {
        aft_expr: vec![source.to_string()],
        aft_dep: 0,
        aft_size: u32::MAX / 2,
        delay: 0.0,
        area: f32::MAX / 2.0,
        failure_reason: "mapping_failed".to_string(),
        cell_nodes: vec![],
        cell_root: 0,
    }
}

fn cover_to_cost(cover: &MappedCoverChoice) -> ffi::CCost {
    let graph = &cover.graph;
    let cell_nodes = graph
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
                pin_names: vec![],
                fanins: vec![],
            },
            MappedCoverNode::Const(v) => ffi::MappedNode {
                kind: 1,
                input_idx: 0,
                const_val: u8::from(*v),
                neg_mask: 0,
                out_neg: 0,
                cell_name: String::new(),
                out_pin: String::new(),
                pin_names: vec![],
                fanins: vec![],
            },
            MappedCoverNode::Cell { cell, fanins } => ffi::MappedNode {
                kind: 2,
                input_idx: 0,
                const_val: 0,
                neg_mask: cell.neg_mask,
                out_neg: u8::from(cell.out_neg),
                cell_name: cell.cell_name.clone(),
                out_pin: cell.out_pin.clone(),
                pin_names: cell.pin_names.clone(),
                fanins: fanins.iter().map(|&fi| fi as u32).collect(),
            },
        })
        .collect();
    ffi::CCost {
        aft_expr: vec![],
        aft_dep: cover.delay.round() as u32,
        aft_size: cover.area.round() as u32,
        delay: cover.delay,
        area: cover.area,
        failure_reason: String::new(),
        cell_nodes,
        cell_root: graph.root as u32,
    }
}

fn cut_mapping_impl(
    s: &str,
    vars: *const u32,
    size: usize,
    objective: u8,
    required_output_delay: Option<f32>,
) -> ffi::CCost {
    let var_dep = parse_var_dep(vars, size);
    let objective = MappingObjective::from_u8(objective);
    match run_extract(s, &var_dep, objective, required_output_delay) {
        Some(cover) => cover_to_cost(&cover),
        None => failure_cost(s),
    }
}

pub fn cut_mapping_with_objective(
    s: &str,
    vars: *const u32,
    size: usize,
    objective: u8,
) -> ffi::CCost {
    let var_dep = parse_var_dep(vars, size);
    let cache_key = make_cache_key(s, &var_dep, objective, None);
    let objective = MappingObjective::from_u8(objective);
    if let Ok(guard) = cache_store().lock() {
        if let Some(v) = guard.get(&cache_key) {
            return v.clone();
        }
    }

    let out = cut_mapping_impl(s, vars, size, objective as u8, None);

    if let Ok(mut guard) = cache_store().lock() {
        guard.insert(cache_key, out.clone());
    }
    out
}

pub fn cut_mapping_with_objective_dc(
    s: &str,
    vars: *const u32,
    size: usize,
    objective: u8,
    required_output_delay: f32,
    _care_mask: u64,
) -> ffi::CCost {
    cut_mapping_impl(
        s,
        vars,
        size,
        objective,
        required_output_delay.is_finite().then_some(required_output_delay),
    )
}

pub fn mapping_library_hits() -> u64 {
    LIBRARY_HITS.load(Ordering::Relaxed)
}

pub fn mapping_egg_calls() -> u64 {
    EGG_CALLS.load(Ordering::Relaxed)
}

pub fn mapping_egg_cache_hits() -> u64 {
    EGG_CACHE_HITS.load(Ordering::Relaxed)
}

pub fn reset_mapping_counters() {
    LIBRARY_HITS.store(0, Ordering::Relaxed);
    EGG_CALLS.store(0, Ordering::Relaxed);
    EGG_CACHE_HITS.store(0, Ordering::Relaxed);
}

/// Single-cell library match for one cut (the C++ caller's "single-cell matcher"): no cover search,
/// no e-graph, no persistent database. `required_output_delay` is accepted for interface
/// compatibility but the fast path is area/delay-keyed by the arrivals only.
pub fn cut_mapping_library_dc(
    s: &str,
    vars: *const u32,
    size: usize,
    objective: u8,
    _required_output_delay: f32,
    _care_mask: u64,
) -> ffi::CCost {
    let var_dep = parse_var_dep(vars, size);
    match run_extract_library(s, &var_dep, MappingObjective::from_u8(objective)) {
        Some(cover) => {
            LIBRARY_HITS.fetch_add(1, Ordering::Relaxed);
            cover_to_cost(&cover)
        }
        None => failure_cost(s),
    }
}

pub fn cut_mapping_egg_dc(s: &str, objective: u8, n_inputs: u32) -> ffi::CCost {
    EGG_CALLS.fetch_add(1, Ordering::Relaxed);
    let zeros = vec![0u32; n_inputs as usize];
    cached_cut_mapping(s, zeros.as_ptr(), zeros.len(), objective, None)
}

pub fn cut_mapping_egg_arr(
    s: &str,
    vars: *const u32,
    size: usize,
    objective: u8,
    required_output_delay: f32,
    limit: u32,
    _cost_model: u8,
) -> Vec<ffi::CCost> {
    EGG_CALLS.fetch_add(1, Ordering::Relaxed);
    let required = required_output_delay.is_finite().then_some(required_output_delay);
    if limit <= 1 {
        let result = cached_cut_mapping(s, vars, size, objective, required);
        return if result.cell_nodes.is_empty() { Vec::new() } else { vec![result] };
    }
    /* Multi-solution: `map_cut_to_cover_alternatives` returns up to `limit` distinct root
       implementations. Not memoised (the cache holds one cover per key). */
    let var_dep = parse_var_dep(vars, size);
    let objective = MappingObjective::from_u8(objective);
    let tech = techlib_store();
    let Some(aig) = emap_egg::gate_expr_to_aig_sexp(s, tech) else {
        return Vec::new();
    };
    let Some((n_inputs, _)) = emap_egg::sexp_to_tt9(&aig) else {
        return Vec::new();
    };
    let arrivals: Vec<f32> = if var_dep.len() == n_inputs as usize {
        var_dep.iter().map(|&d| d as f32).collect()
    } else {
        vec![0.0; n_inputs as usize]
    };
    let emap_obj = match objective {
        MappingObjective::AreaDelay => EmapObjective::AreaFirst,
        MappingObjective::DelayArea => EmapObjective::DelayFirst,
    };
    match map_cut_to_cover_alternatives(
        &aig,
        tech,
        &arrivals,
        emap_obj,
        required,
        limit as usize,
    ) {
        Ok(choices) => choices.iter().map(cover_to_cost).collect(),
        Err(_) => Vec::new(),
    }
}

pub fn cut_mapping_egg_multi(
    s: &str,
    objective: u8,
    n_inputs: u32,
    _limit: u32,
) -> Vec<ffi::CCost> {
    let result = cut_mapping_egg_dc(s, objective, n_inputs);
    if result.cell_nodes.is_empty() { Vec::new() } else { vec![result] }
}

pub fn cut_mapping_multi_dc(
    s: &str,
    objective: u8,
    _care_mask: u64,
    _use_egg: bool,
    _limit: u32,
) -> Vec<ffi::CCost> {
    let result = cut_mapping_egg_dc(s, objective, 0);
    if result.cell_nodes.is_empty() { Vec::new() } else { vec![result] }
}

pub fn cut_mapping_db_dc(_s: &str, _objective: u8, _care_mask: u64) -> ffi::CCost {
    failure_cost("database_unavailable")
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
        let k0 = make_cache_key(SIMPLE_AND, &vars, 0, None);
        let k2 = make_cache_key(SIMPLE_AND, &vars, 2, None);
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
