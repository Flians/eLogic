use egg::Id;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::techlib::{Cell, Expr, TechLib};

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
/// Variables are expected to be single-letter symbols in `a`..`f` (case-sensitive).
/// Returns `(n_inputs, tt64)` where bit `k` corresponds to assignment `k` (LSB first).
#[must_use]
pub fn sexp_to_tt64(sexp: &str) -> Option<(u8, u64)> {
    let expr: egg::RecExpr<AIG> = sexp.parse().ok()?;
    let mut n_inputs = 0u8;

    // Determine number of inputs used (max letter index + 1).
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
            n_inputs = n_inputs.max((c - b'a' + 1) as u8);
        }
    }

    if n_inputs > 6 {
        return None;
    }

    let mask = match n_inputs {
        0 => 1u64,
        6 => u64::MAX, // 2^64 bits cannot be represented as (1<<64)
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
                var_tt((c - b'a') as u8, n_inputs) & mask
            }
            AIG::Not(a) => (!values[usize::from(*a)]) & mask,
            AIG::And([a, b]) => values[usize::from(*a)] & values[usize::from(*b)],
        };
        values.push(v);
    }

    let root = *values.last()?;
    Some((n_inputs, root & mask))
}

pub type GateLang = egg::SymbolLang;
pub type GateEGraph = egg::EGraph<GateLang, GateTtFold>;
type GateRewrite = egg::Rewrite<GateLang, GateTtFold>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TtData {
    pub tt: u64,
}

#[derive(Default)]
pub struct GateTtFold;
impl egg::Analysis<GateLang> for GateTtFold {
    type Data = Option<TtData>;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> egg::DidMerge {
        egg::merge_option(to, from, |a, b| {
            assert_eq!(a.tt, b.tt, "Merged non-equal truth tables");
            egg::DidMerge(false, false)
        })
    }

    fn make(egraph: &mut GateEGraph, enode: &GateLang, _id: Id) -> Self::Data {
        let x = |i: &egg::Id| egraph[*i].data.as_ref().map(|c| c.tt);
        let op = enode.op.as_str();
        let tt = match (op, enode.children.as_slice()) {
            ("0", []) => 0u64,
            ("1", []) => u64::MAX,
            ("~", [a]) => !x(a)?,
            ("&", [a, b]) => x(a)? & x(b)?,
            _ if op.len() == 1 => {
                let bytes = op.as_bytes();
                if !(b'a'..=b'f').contains(&bytes[0]) || !enode.children.is_empty() {
                    return None;
                }
                gate_var_tt(bytes[0] - b'a')
            }
            _ => {
                let (_, gate_tt) = parse_gate_symbol(op)?;
                let children = enode
                    .children
                    .iter()
                    .map(|id| x(id))
                    .collect::<Option<Vec<_>>>()?;
                compose_tt(gate_tt, &children)
            }
        };
        Some(TtData { tt })
    }
}

fn gate_var_tt(idx: u8) -> u64 {
    let mut tt = 0u64;
    for a in 0..64u64 {
        if ((a >> idx) & 1) == 1 {
            tt |= 1u64 << a;
        }
    }
    tt
}

fn compose_tt(gate_tt: u64, children: &[u64]) -> u64 {
    let mut out = 0u64;
    for assign in 0..64u64 {
        let mut idx = 0u64;
        for (bit, child_tt) in children.iter().enumerate() {
            if ((child_tt >> assign) & 1) == 1 {
                idx |= 1u64 << bit;
            }
        }
        if ((gate_tt >> idx) & 1) == 1 {
            out |= 1u64 << assign;
        }
    }
    out
}

fn make_gate_symbol(cell: &Cell) -> String {
    format!(
        "gate:{}:{:016x}:{}:{}",
        cell.vars.len(),
        cell.tt64,
        cell.name,
        cell.out_pin
    )
}

fn parse_gate_symbol(op: &str) -> Option<(usize, u64)> {
    let rest = op.strip_prefix("gate:")?;
    let mut parts = rest.splitn(3, ':');
    let arity = parts.next()?.parse::<usize>().ok()?;
    let tt = u64::from_str_radix(parts.next()?, 16).ok()?;
    Some((arity, tt))
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

fn expr_to_aig_pattern(expr: &Expr, vars: &HashMap<&str, String>) -> Option<String> {
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

fn dynamic_gate_rewrites(
    tech: &TechLib,
    gate_cells: &mut HashMap<String, Cell>,
) -> Vec<GateRewrite> {
    let mut rules = Vec::<GateRewrite>::new();
    for cell in &tech.cells {
        if cell.vars.is_empty() || cell.vars.len() > 6 {
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

        if let Ok(rw) = egg::Rewrite::new(format!("expand-{gate_symbol}"), lhs.clone(), rhs.clone())
        {
            rules.push(rw);
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
    #[must_use]
    pub fn new(tech: &TechLib) -> Self {
        let mut gate_cells = HashMap::<String, Cell>::new();
        let mut rules = vec![
            gate_double_neg(),
            gate_double_neg_flip(),
            gate_associ_and(),
            gate_comm_and(),
            gate_comp_and(),
            gate_dup_and(),
            gate_and_true(),
            gate_and_false(),
        ];
        rules.extend(dynamic_gate_rewrites(tech, &mut gate_cells));
        Self {
            rules,
            gate_cells: Arc::new(gate_cells),
        }
    }

    pub(crate) fn rules(&self) -> &[GateRewrite] {
        &self.rules
    }
}

impl TechLib {
    /// Lazily builds and caches egg rewrites for this library; safe to call from every path
    /// that needs [`EggRewriteBundle`].
    #[must_use]
    pub fn egg_rewrite_bundle(&self) -> &Arc<EggRewriteBundle> {
        self.egg_bundle_cache.get_or_init(|| Arc::new(EggRewriteBundle::new(self)))
    }
}

pub fn rewrite_to_gategraph(
    s: &str,
    tech: &TechLib,
    _n_inputs: usize,
    bundle: Option<&EggRewriteBundle>,
) -> Option<(GateEGraph, egg::Id, Arc<HashMap<String, Cell>>)> {
    let profile = std::env::var("EMAP_PROFILE_REWRITE")
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
    let t_after_parse = Instant::now();

    let mut runner = egg::Runner::<GateLang, GateTtFold, ()>::default().with_expr(&expr);
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
            "EMAP_PROFILE_REWRITE rewrite_to_gategraph: parse={parse_ms:.3}ms init_runner={init_ms:.3}ms \
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

    fn tt_at_root(egraph: &super::GateEGraph, root: egg::Id) -> u64 {
        egraph[root]
            .data
            .as_ref()
            .expect("root eclass should have TT analysis")
            .tt
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
        let tt = tt_at_root(&egraph, root);
        assert_eq!(tt & 0xF, 0x8, "a&b over 2 minterms: only ab=11 is 1");
    }

    #[test]
    fn prebuilt_bundle_matches_scratch_rules() {
        let tech = TechLib::default();
        let cached = tech.egg_rewrite_bundle();
        let (e1, r1, _) =
            rewrite_to_gategraph("(& a b)", &tech, 2, Some(cached.as_ref())).unwrap();
        let (e2, r2, _) = rewrite_to_gategraph("(& a b)", &tech, 2, None).unwrap();
        assert_eq!(tt_at_root(&e1, r1), tt_at_root(&e2, r2));
    }

    #[test]
    fn structural_rules_commute_and() {
        let tech = TechLib::default();
        let (e1, r1, _) = rewrite_to_gategraph("(& a b)", &tech, 2, None).unwrap();
        let (e2, r2, _) = rewrite_to_gategraph("(& b a)", &tech, 2, None).unwrap();
        assert_eq!(tt_at_root(&e1, r1), tt_at_root(&e2, r2));
    }

    #[test]
    fn structural_rules_cancel_double_negation() {
        let tech = TechLib::default();
        let (e1, r1, _) = rewrite_to_gategraph("a", &tech, 1, None).unwrap();
        let (e2, r2, _) = rewrite_to_gategraph("(~ (~ a))", &tech, 1, None).unwrap();
        assert_eq!(tt_at_root(&e1, r1), tt_at_root(&e2, r2));
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
