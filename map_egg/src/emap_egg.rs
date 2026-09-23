use egg::Id;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::techlib::{Cell, Expr, TechLib};
use crate::tt9::{
    tt9_all_ones, tt9_and_masked, tt9_bit_get, tt9_bit_set, tt9_from_hex, tt9_from_u64,
    tt9_not_masked, tt9_to_hex, tt9_zero, Tt9,
};

thread_local! {
    static EGG_GATE_TT_N: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

egg::define_language! {
    pub enum AIG {
        Bool(u8),
        "&" = And([Id; 2]),
        "~" = Not(Id),
        Symbol(egg::Symbol),
    }
}

/// Truth table for an AIG s-expression; variables `a`..=`i` (up to **9** inputs).
#[must_use]
pub fn sexp_to_tt9(sexp: &str) -> Option<(u8, Tt9)> {
    let expr: egg::RecExpr<AIG> = sexp.parse().ok()?;
    let mut n_inputs = 0u8;

    for node in expr.as_ref() {
        if let AIG::Symbol(sym) = node {
            let s = sym.as_str();
            // `0`/`1` are parsed as plain symbols (the `Bool` variant has no op string).
            if s == "0" || s == "1" {
                continue;
            }
            let bytes = s.as_bytes();
            if bytes.len() != 1 {
                return None;
            }
            let c = bytes[0];
            if !(b'a'..=b'i').contains(&c) {
                return None;
            }
            n_inputs = n_inputs.max((c - b'a' + 1) as u8);
        }
    }

    if n_inputs > 9 {
        return None;
    }

    let mask_ones = tt9_all_ones(n_inputs);

    let var_tt = |idx: u8, n: u8| -> Tt9 {
        let mut tt = tt9_zero();
        for a in 0..(1usize << usize::from(n)) {
            if ((a >> usize::from(idx)) & 1) == 1 {
                tt9_bit_set(&mut tt, a);
            }
        }
        tt
    };

    let mut values = Vec::<Tt9>::with_capacity(expr.as_ref().len());
    for node in expr.as_ref() {
        let v = match node {
            AIG::Bool(0) => tt9_zero(),
            AIG::Bool(1) => mask_ones,
            AIG::Bool(_) => return None,
            AIG::Symbol(sym) => {
                let s = sym.as_str();
                match s {
                    "0" => tt9_zero(),
                    "1" => mask_ones,
                    _ => {
                        let bytes = s.as_bytes();
                        if bytes.len() != 1 {
                            return None;
                        }
                        let c = bytes[0];
                        if !(b'a'..=b'i').contains(&c) {
                            return None;
                        }
                        var_tt((c - b'a') as u8, n_inputs)
                    }
                }
            }
            AIG::Not(a) => tt9_not_masked(&values[usize::from(*a)], n_inputs),
            AIG::And([a, b]) => {
                tt9_and_masked(&values[usize::from(*a)], &values[usize::from(*b)], n_inputs)
            }
        };
        values.push(v);
    }

    let root = *values.last()?;
    Some((n_inputs, root))
}

/// Compute a 64-bit truth table for an expression with up to 6 variables (compat).
///
/// Variables `a`..`f`. Returns `None` if the expression uses more than 6 inputs.
#[must_use]
pub fn sexp_to_tt64(sexp: &str) -> Option<(u8, u64)> {
    let (n, tt) = sexp_to_tt9(sexp)?;
    if n > 6 {
        return None;
    }
    let mask = match n {
        0 => 1u64,
        6 => u64::MAX,
        _ => (1u64 << (1u32 << n)) - 1,
    };
    Some((n, tt[0] & mask))
}

pub type GateLang = egg::SymbolLang;
pub type GateEGraph = egg::EGraph<GateLang, GateTtFold>;
type GateRewrite = egg::Rewrite<GateLang, GateTtFold>;

/// Infer `n` from single-letter ops `a`..=`i` in a gate-level `RecExpr`.
fn gate_expr_n_inputs(expr: &egg::RecExpr<GateLang>) -> u8 {
    let mut n = 0u8;
    for node in expr.as_ref() {
        let op = node.op.as_str();
        if op.len() == 1 {
            let b = op.as_bytes()[0];
            if (b'a'..=b'i').contains(&b) {
                n = n.max(b - b'a' + 1);
            }
        }
    }
    n
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TtData {
    /// Number of Boolean inputs (`a`..) this TT is defined over (same for all nodes in one run).
    pub n: u8,
    pub tt: Tt9,
}

#[derive(Default)]
pub struct GateTtFold;
impl egg::Analysis<GateLang> for GateTtFold {
    type Data = Option<TtData>;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> egg::DidMerge {
        egg::merge_option(to, from, |a, b| {
            assert_eq!(a.n, b.n, "TT merge n mismatch");
            assert_eq!(a.tt, b.tt, "Merged non-equal truth tables");
            egg::DidMerge(false, false)
        })
    }

    fn make(egraph: &mut GateEGraph, enode: &GateLang, _id: Id) -> Self::Data {
        let n = EGG_GATE_TT_N.get();
        if n > 9 {
            return None;
        }
        let x = |i: &egg::Id| egraph[*i].data.as_ref();
        let op = enode.op.as_str();
        let data = match (op, enode.children.as_slice()) {
            ("0", []) => TtData {
                n,
                tt: tt9_zero(),
            },
            ("1", []) => TtData {
                n,
                tt: tt9_all_ones(n),
            },
            ("~", [a]) => {
                let ca = x(a)?;
                if ca.n != n {
                    return None;
                }
                TtData {
                    n,
                    tt: tt9_not_masked(&ca.tt, n),
                }
            }
            ("&", [a, b]) => {
                let ca = x(a)?;
                let cb = x(b)?;
                if ca.n != n || cb.n != n {
                    return None;
                }
                TtData {
                    n,
                    tt: tt9_and_masked(&ca.tt, &cb.tt, n),
                }
            }
            _ if op.len() == 1 && enode.children.is_empty() => {
                let bytes = op.as_bytes();
                if !(b'a'..=b'i').contains(&bytes[0]) {
                    return None;
                }
                let idx = bytes[0] - b'a';
                if idx >= n {
                    return None;
                }
                TtData {
                    n,
                    tt: gate_var_tt9(idx, n),
                }
            }
            _ => {
                let (arity, gate_tt) = parse_gate_symbol_full(op)?;
                let children: Vec<&TtData> = enode
                    .children
                    .iter()
                    .map(|id| x(id))
                    .collect::<Option<Vec<_>>>()?;
                TtData {
                    n,
                    tt: compose_tt9(&gate_tt, arity, &children, n)?,
                }
            }
        };
        Some(data)
    }
}

fn gate_var_tt9(idx: u8, n: u8) -> Tt9 {
    let mut tt = tt9_zero();
    for a in 0..(1usize << usize::from(n)) {
        if ((a >> usize::from(idx)) & 1) == 1 {
            tt9_bit_set(&mut tt, a);
        }
    }
    tt
}

fn compose_tt9(gate_tt: &Tt9, arity: usize, children: &[&TtData], n: u8) -> Option<Tt9> {
    if children.len() != arity {
        return None;
    }
    for c in children {
        if c.n != n {
            return None;
        }
    }
    let mut out = tt9_zero();
    for assign in 0..(1usize << usize::from(n)) {
        let mut idx = 0usize;
        for (bit, ch) in children.iter().enumerate() {
            if tt9_bit_get(&ch.tt, assign) {
                idx |= 1usize << bit;
            }
        }
        if tt9_bit_get(gate_tt, idx) {
            tt9_bit_set(&mut out, assign);
        }
    }
    Some(out)
}

fn parse_gate_symbol_full(op: &str) -> Option<(usize, Tt9)> {
    let rest = op.strip_prefix("gate:")?;
    let (arity_s, after_first) = rest.split_once(':')?;
    let arity: usize = arity_s.parse().ok()?;
    if after_first.len() >= 128 {
        let tt = tt9_from_hex(&after_first[..128])?;
        return Some((arity, tt));
    }
    let (hex_s, _) = after_first.split_once(':')?;
    let lo = u64::from_str_radix(hex_s, 16).ok()?;
    Some((arity, tt9_from_u64(lo)))
}

fn make_gate_symbol(cell: &Cell) -> String {
    format!(
        "gate:{}:{}:{}:{}",
        cell.vars.len(),
        tt9_to_hex(&cell.tt),
        cell.name,
        cell.out_pin
    )
}

/// Public wrapper around the internal gate-symbol encoding (`gate:arity:tt:name:pin`).
///
/// `map_egg` uses this to translate a cell-name gate-level subgraph into the operator strings
/// understood by [`GateTtFold`].
#[must_use]
pub fn gate_symbol_for_cell(cell: &Cell) -> String {
    make_gate_symbol(cell)
}

fn base_gate_pattern(cell: &Cell) -> String {
    let op = make_gate_symbol(cell);
    if cell.vars.is_empty() {
        op
    } else {
        let args = (0..cell.vars.len())
            .map(|i| format!("?i{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        format!("({op} {args})")
    }
}

pub(crate) fn expr_to_aig_pattern(expr: &Expr, vars: &HashMap<&str, String>) -> Option<String> {
    match expr {
        Expr::Const(false) => Some("0".to_string()),
        Expr::Const(true) => Some("1".to_string()),
        Expr::Var(v) => vars.get(v.as_str()).cloned(),
        Expr::Not(a) => Some(format!("(~ {})", expr_to_aig_pattern(a, vars)?)),
        Expr::And(a, b) => Some(format!(
            "(& {} {})",
            expr_to_aig_pattern(a, vars)?,
            expr_to_aig_pattern(b, vars)?
        )),
        Expr::Or(a, b) => {
            let lhs = expr_to_aig_pattern(a, vars)?;
            let rhs = expr_to_aig_pattern(b, vars)?;
            Some(format!("(~ (& (~ {lhs}) (~ {rhs})))"))
        }
    }
}

fn base_aig_pattern(cell: &Cell) -> Option<String> {
    let mut vars = HashMap::<&str, String>::new();
    for (idx, name) in cell.vars.iter().enumerate() {
        vars.insert(name.as_str(), format!("?i{idx}"));
    }
    expr_to_aig_pattern(&cell.expr, &vars)
}

/// Expand a cell-name gate-level s-expression into an AIG s-expression.
///
/// Accepted leaf ops are single letters `a`..`i` (cut variables) and constants `0`/`1`;
/// `&`/`~` are kept as-is; any other op is looked up by **standard-cell name** in `tech` and
/// replaced by that cell's [`Expr`] lowered to `&`/`~` with the op's children substituted for
/// the cell variables. This is the `gate-level subgraph -> AIG` step of the `map_egg` flow.
#[must_use]
pub fn gate_expr_to_aig_sexp(expr_str: &str, tech: &TechLib) -> Option<String> {
    let expr: egg::RecExpr<GateLang> = expr_str.trim().parse().ok()?;
    if expr.as_ref().is_empty() {
        return None;
    }
    let by_name: HashMap<&str, &Cell> = tech
        .cells
        .iter()
        .map(|c| (c.name.as_str(), c))
        .collect();
    expand_gate_rec_to_aig(&expr, &by_name)
}

fn expand_gate_rec_to_aig(
    expr: &egg::RecExpr<GateLang>,
    by_name: &HashMap<&str, &Cell>,
) -> Option<String> {
    let mut memo: Vec<Option<String>> = Vec::with_capacity(expr.as_ref().len());
    for node in expr.as_ref() {
        let mut child_strs = Vec::<String>::with_capacity(node.children.len());
        let mut ok = true;
        for child in &node.children {
            match memo.get(usize::from(*child)).and_then(Clone::clone) {
                Some(s) => child_strs.push(s),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            memo.push(None);
            continue;
        }
        memo.push(expand_gate_op(node.op.as_str(), &child_strs, by_name));
    }
    memo.into_iter().last().flatten()
}

fn expand_gate_op(
    op: &str,
    children: &[String],
    by_name: &HashMap<&str, &Cell>,
) -> Option<String> {
    match (op, children.len()) {
        ("0", 0) => Some("0".to_string()),
        ("1", 0) => Some("1".to_string()),
        ("~", 1) => Some(format!("(~ {})", children[0])),
        ("&", 2) => Some(format!("(& {} {})", children[0], children[1])),
        (sym, 0) if sym.len() == 1 && (b'a'..=b'i').contains(&sym.as_bytes()[0]) => {
            Some(sym.to_string())
        }
        (sym, _) => {
            let cell = by_name.get(sym)?;
            if cell.vars.len() != children.len() {
                return None;
            }
            let mut vars = HashMap::<&str, String>::new();
            for (idx, name) in cell.vars.iter().enumerate() {
                vars.insert(name.as_str(), children[idx].clone());
            }
            expr_to_aig_pattern(&cell.expr, &vars)
        }
    }
}

fn dynamic_gate_rewrites(
    tech: &TechLib,
    gate_cells: &mut HashMap<String, Cell>,
    cover_only: bool,
) -> Vec<GateRewrite> {
    let mut rules = Vec::<GateRewrite>::new();
    for cell in &tech.cells {
        if cell.vars.is_empty() || cell.vars.len() > 9 {
            continue;
        }
        let gate_symbol = make_gate_symbol(cell);
        gate_cells
            .entry(gate_symbol.clone())
            .or_insert_with(|| cell.clone());

        let lhs = match base_gate_pattern(cell).parse::<egg::Pattern<GateLang>>() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let Some(rhs_str) = base_aig_pattern(cell) else {
            continue;
        };
        let rhs = match rhs_str.parse::<egg::Pattern<GateLang>>() {
            Ok(p) => p,
            Err(_) => continue,
        };

        if !cover_only {
            if let Ok(rw) =
                egg::Rewrite::new(format!("expand-{gate_symbol}"), lhs.clone(), rhs.clone())
            {
                rules.push(rw);
            }
        }
        if let Ok(rw) = egg::Rewrite::new(format!("cover-{gate_symbol}"), rhs, lhs) {
            rules.push(rw);
        }
    }
    rules
}

macro_rules! rule_gate {
    ($name:ident, $left:literal, $right:literal) => {
        #[allow(dead_code)]
        fn $name() -> GateRewrite {
            egg::rewrite!(stringify!($name); $left => $right)
        }
    };
    ($name:ident, $name2:ident, $left:literal, $right:literal) => {
        rule_gate!($name, $left, $right);
        rule_gate!($name2, $right, $left);
    };
}

rule_gate! {gate_double_neg,     gate_double_neg_flip,    "(~ (~ ?a))",                 "?a"                              }
rule_gate! {gate_associ_and,     "(& ?a (& ?b ?c))",      "(& ?b (& ?a ?c))"       }
rule_gate! {gate_comm_and,       "(& ?a ?b)",             "(& ?b ?a)"              }
rule_gate! {gate_comp_and,       "(& ?a (~ ?a))",         "0"                      }
rule_gate! {gate_dup_and,        "(& ?a ?a)",             "?a"                     }
rule_gate! {gate_and_true,       "(& ?a 1)",              "?a"                     }
rule_gate! {gate_and_false,      "(& ?a 0)",              "0"                      }

/// Precomputed static + dynamic egg rewrites for a [`TechLib`].
///
/// Building this can take several seconds for large libraries (e.g. asap7). Prefer
/// [`TechLib::egg_rewrite_bundle`] (lazy, cached on the library) or pass `Some` from that
/// into [`rewrite_to_gategraph`]; avoid calling [`EggRewriteBundle::new`] repeatedly for the
/// same [`TechLib`].
#[derive(Debug)]
pub struct EggRewriteBundle {
    rules: Vec<GateRewrite>,
    gate_cells: Arc<HashMap<String, Cell>>,
}

impl EggRewriteBundle {
    fn static_rules() -> Vec<GateRewrite> {
        vec![
            gate_double_neg(),
            gate_double_neg_flip(),
            gate_associ_and(),
            gate_comm_and(),
            gate_comp_and(),
            gate_dup_and(),
            gate_and_true(),
            gate_and_false(),
        ]
    }

    #[must_use]
    pub fn new(tech: &TechLib) -> Self {
        let mut gate_cells = HashMap::<String, Cell>::new();
        let mut rules = Self::static_rules();
        rules.extend(dynamic_gate_rewrites(tech, &mut gate_cells, false));
        Self {
            rules,
            gate_cells: Arc::new(gate_cells),
        }
    }

    /// Rewrites for an **AIG** input: the static AIG rules plus only the one-way
    /// `AIG -> gate` (cover) rules. Starting from AIG this is sufficient to reach every cell
    /// cover, and it avoids the gate -> AIG expansion that makes saturation blow up.
    ///
    /// The direction is what makes extraction return *gate-level* solutions: an e-node stays an
    /// AIG `(& a b)` / `(~ a)` until a cover rule turns it into a library cell, and the cover DP
    /// prices a surviving raw AIG node with the library's own AND2 / inverter cell (area and delay
    /// taken from the library; unit delay under the node-depth objective). If the library has no
    /// such cell, that node has no cover choice at all -- effectively infinite cost -- so a
    /// successful extraction is always directly materializable, never a hidden AIG-only cover.
    #[must_use]
    pub fn new_cover_only(tech: &TechLib) -> Self {
        let mut gate_cells = HashMap::<String, Cell>::new();
        let mut rules = Self::static_rules();
        rules.extend(dynamic_gate_rewrites(tech, &mut gate_cells, true));
        Self {
            rules,
            gate_cells: Arc::new(gate_cells),
        }
    }

    pub(crate) fn rules(&self) -> &[GateRewrite] {
        &self.rules
    }

    /// Gate-symbol (`gate:arity:tt:name:pin`) -> [`Cell`] map used by the egg rewrites.
    #[must_use]
    pub fn gate_cells(&self) -> &HashMap<String, Cell> {
        &self.gate_cells
    }
}

impl TechLib {
    /// Lazily builds and caches egg rewrites for this library; safe to call from every path
    /// that needs [`EggRewriteBundle`].
    #[must_use]
    pub fn egg_rewrite_bundle(&self) -> &Arc<EggRewriteBundle> {
        self.egg_bundle_cache.get_or_init(|| Arc::new(EggRewriteBundle::new(self)))
    }

    /// Lazily builds and caches the cover-only (**AIG -> library cell**) rewrite bundle.
    #[must_use]
    pub fn egg_cover_bundle(&self) -> &Arc<EggRewriteBundle> {
        self.egg_cover_bundle_cache
            .get_or_init(|| Arc::new(EggRewriteBundle::new_cover_only(self)))
    }
}

/// Equality-saturation limits for [`egg::Runner`] (defaults match egg 0.11: 30 iters, 10k nodes).
///
/// Override with `EMAPPER_EGG_ITER_LIMIT` / `EMAPPER_EGG_NODE_LIMIT` — lower values **speed up** mapping
/// at the risk of missing rewrites that need more saturation.
fn egg_runner_limits_from_env() -> (usize, usize) {
    fn usize_env(key: &str, default: usize) -> usize {
        std::env::var(key)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }
    (
        usize_env("EMAPPER_EGG_ITER_LIMIT", 30),
        usize_env("EMAPPER_EGG_NODE_LIMIT", 10_000),
    )
}

pub fn rewrite_to_gategraph(
    s: &str,
    tech: &TechLib,
    n_inputs: usize,
    bundle: Option<&EggRewriteBundle>,
) -> Option<(GateEGraph, egg::Id, Arc<HashMap<String, Cell>>)> {
    rewrite_to_gategraph_with_limits(s, tech, n_inputs, bundle, 0, 0)
}

/// [`rewrite_to_gategraph`] with explicit equality-saturation limits.
///
/// A limit of `0` falls back to the `EMAPPER_EGG_*_LIMIT` environment default. Cut mapping passes
/// small values because a cut AIG is tiny: running the default 30 iterations over 10k nodes can
/// take seconds per cut while producing the same extracted cover.
pub fn rewrite_to_gategraph_with_limits(
    s: &str,
    tech: &TechLib,
    _n_inputs: usize,
    bundle: Option<&EggRewriteBundle>,
    iter_limit: usize,
    node_limit: usize,
) -> Option<(GateEGraph, egg::Id, Arc<HashMap<String, Cell>>)> {
    let profile = std::env::var("EMAPPER_PROFILE_REWRITE")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let t0 = Instant::now();

    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let expr: egg::RecExpr<GateLang> = s.parse().ok()?;
    // `EGraph::add_expr` panics on empty `RecExpr` (`new_ids.last().unwrap()`).
    if expr.as_ref().is_empty() {
        return None;
    }
    let n_inputs = gate_expr_n_inputs(&expr);
    EGG_GATE_TT_N.set(n_inputs);
    struct EggTtNReset;
    impl Drop for EggTtNReset {
        fn drop(&mut self) {
            EGG_GATE_TT_N.set(0);
        }
    }
    let _egg_tt_n_reset = EggTtNReset;

    let t_after_parse = Instant::now();

    let (env_iter, env_node) = egg_runner_limits_from_env();
    let iter_limit = if iter_limit == 0 { env_iter } else { iter_limit };
    let node_limit = if node_limit == 0 { env_node } else { node_limit };
    let mut runner = egg::Runner::<GateLang, GateTtFold, ()>::default()
        .with_iter_limit(iter_limit)
        .with_node_limit(node_limit)
        .with_expr(&expr);
    let root = runner.roots[0];
    let t_after_runner_init = Instant::now();

    const N_STATIC: usize = 8;
    let b = bundle.unwrap_or_else(|| tech.egg_rewrite_bundle().as_ref());
    let t_after_rules = Instant::now();
    let n_dynamic = b.rules().len().saturating_sub(N_STATIC);
    let gate_cells = Arc::clone(&b.gate_cells);

    runner = runner.run(b.rules());
    let t_end = Instant::now();

    if profile {
        let parse_ms = t_after_parse.duration_since(t0).as_secs_f64() * 1000.0;
        let init_ms = t_after_runner_init
            .duration_since(t_after_parse)
            .as_secs_f64()
            * 1000.0;
        let build_rules_ms = t_after_rules
            .duration_since(t_after_runner_init)
            .as_secs_f64()
            * 1000.0;
        let run_ms = t_end.duration_since(t_after_rules).as_secs_f64() * 1000.0;
        let total_ms = t_end.duration_since(t0).as_secs_f64() * 1000.0;
        eprintln!(
            "EMAPPER_PROFILE_REWRITE rewrite_to_gategraph: parse={parse_ms:.3}ms init_runner={init_ms:.3}ms \
             build_rules(static={N_STATIC}+dynamic={n_dynamic})={build_rules_ms:.3}ms \
             runner.run(saturate)={run_ms:.3}ms total={total_ms:.3}ms"
        );
    }

    Some((runner.egraph, root, gate_cells))
}

#[cfg(test)]
mod rewrite_to_gategraph_tests {
    use std::path::{Path, PathBuf};

    use super::rewrite_to_gategraph;
    use crate::techlib::TechLib;

    /// Repo path: `tools/mockturtle/experiments/cell_libraries/asap7.genlib` (relative to workspace root).
    fn asap7_genlib_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tools/mockturtle/experiments/cell_libraries/asap7.genlib")
    }

    fn tt_at_root_u64(egraph: &super::GateEGraph, root: egg::Id) -> u64 {
        let d = egraph[root]
            .data
            .as_ref()
            .expect("root eclass should have TT analysis");
        let n = d.n;
        let mask = if n >= 6 {
            u64::MAX
        } else {
            (1u64 << (1u32 << n)) - 1
        };
        d.tt[0] & mask
    }

    #[test]
    fn rejects_invalid_sexp() {
        let tech = TechLib::default();
        assert!(rewrite_to_gategraph("", &tech, 0, None).is_none());
        assert!(rewrite_to_gategraph("   ", &tech, 0, None).is_none());
        assert!(rewrite_to_gategraph("(((", &tech, 0, None).is_none());
    }

    #[test]
    fn simple_and_has_consistent_tt() {
        let tech = TechLib::default();
        let (egraph, root, _) =
            rewrite_to_gategraph("(& a b)", &tech, 2, None).expect("parse & rewrite");
        let tt = tt_at_root_u64(&egraph, root);
        assert_eq!(tt & 0xF, 0x8, "a&b over 2 minterms: only ab=11 is 1");
    }

    #[test]
    fn prebuilt_bundle_matches_scratch_rules() {
        let tech = TechLib::default();
        let cached = tech.egg_rewrite_bundle();
        let (e1, r1, _) =
            rewrite_to_gategraph("(& a b)", &tech, 2, Some(cached.as_ref())).unwrap();
        let (e2, r2, _) = rewrite_to_gategraph("(& a b)", &tech, 2, None).unwrap();
        assert_eq!(tt_at_root_u64(&e1, r1), tt_at_root_u64(&e2, r2));
    }

    #[test]
    fn structural_rules_commute_and() {
        let tech = TechLib::default();
        let (e1, r1, _) = rewrite_to_gategraph("(& a b)", &tech, 2, None).unwrap();
        let (e2, r2, _) = rewrite_to_gategraph("(& b a)", &tech, 2, None).unwrap();
        assert_eq!(tt_at_root_u64(&e1, r1), tt_at_root_u64(&e2, r2));
    }

    #[test]
    fn structural_rules_cancel_double_negation() {
        let tech = TechLib::default();
        let (e1, r1, _) = rewrite_to_gategraph("a", &tech, 1, None).unwrap();
        let (e2, r2, _) = rewrite_to_gategraph("(~ (~ a))", &tech, 1, None).unwrap();
        assert_eq!(tt_at_root_u64(&e1, r1), tt_at_root_u64(&e2, r2));
    }

    /// `GateTtFold` uses `Tt9` + thread-local `n`; check 7 inputs (`a`..=`g`) match [`sexp_to_tt9`].
    #[test]
    fn seven_inputs_tt_fold_matches_sexp_to_tt9() {
        let tech = TechLib::default();
        let s = "(& a g)";
        let (n, tt_ref) = super::sexp_to_tt9(s).expect("sexp_to_tt9");
        assert_eq!(n, 7);
        let (egraph, root, _) = rewrite_to_gategraph(s, &tech, 7, None).expect("rewrite");
        let d = egraph[root]
            .data
            .as_ref()
            .expect("root should have TT analysis");
        assert_eq!(d.n, 7);
        assert_eq!(d.tt, tt_ref, "egg analysis must match direct sexp TT for 7 inputs");
    }

    /// Same for 9 inputs (`a`..=`i`).
    #[test]
    fn nine_inputs_tt_fold_matches_sexp_to_tt9() {
        let tech = TechLib::default();
        let s = "(& a i)";
        let (n, tt_ref) = super::sexp_to_tt9(s).expect("sexp_to_tt9");
        assert_eq!(n, 9);
        let (egraph, root, _) = rewrite_to_gategraph(s, &tech, 9, None).expect("rewrite");
        let d = egraph[root]
            .data
            .as_ref()
            .expect("root should have TT analysis");
        assert_eq!(d.n, 9);
        assert_eq!(d.tt, tt_ref, "egg analysis must match direct sexp TT for 9 inputs");
    }

    /// Loads the real `asap7.genlib` from the workspace (fast; no egg saturation).
    #[test]
    fn asap7_genlib_loads_and_indexes() {
        let p = asap7_genlib_path();
        assert!(
            p.is_file(),
            "expected asap7.genlib at {} (from repo: tools/mockturtle/experiments/cell_libraries/asap7.genlib)",
            p.display()
        );
        let tech = TechLib::from_genlib(&p, 6).expect("from_genlib");
        assert!(!tech.cells.is_empty(), "asap7 should define cells");
        assert!(!tech.index.is_empty(), "TT index should be non-empty");

        eprintln!(
            "asap7.genlib: {} cells, {} (n_inputs, tt) index buckets\n",
            tech.cells.len(),
            tech.index.len()
        );
        for (i, c) in tech.cells.iter().enumerate() {
            eprintln!(
                "{:4}  {:<40}  area={:>8.4}  in={}  out_pin={}  vars={:?}",
                i + 1,
                c.name,
                c.area,
                c.vars.len(),
                c.out_pin,
                c.vars
            );
        }
    }

    /// Full `rewrite_to_gategraph` on asap7: `dynamic_gate_rewrites` adds a very large rule set; egg can run a long time.
    #[test]
    #[ignore = "slow: can take minutes — run: cargo test -p emapper asap7_rewrite_to_gategraph -- --ignored --nocapture"]
    fn asap7_rewrite_to_gategraph_registers_gates() {
        let p = asap7_genlib_path();
        assert!(p.is_file(), "missing {}", p.display());
        let tech = TechLib::from_genlib(&p, 6).expect("from_genlib");
        let (egraph, root, gate_variants) =
            rewrite_to_gategraph("(& a b)", &tech, 2, None).expect("rewrite");
        assert!(
            !gate_variants.is_empty(),
            "asap7 should contribute gate symbols to the egraph"
        );
        assert!(egraph[root].data.is_some());
    }
}
