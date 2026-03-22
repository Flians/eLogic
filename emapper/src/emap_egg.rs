use egg::Id;
use std::collections::HashMap;

use crate::techlib::{Cell, CellVariant, Expr, TechLib};

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

    fn make(egraph: &GateEGraph, enode: &GateLang) -> Self::Data {
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

fn make_gate_symbol(cell: &CellVariant, tt: u64) -> String {
    let perm = cell
        .perm
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "gate:{}:{:016x}:{}:{}:{}:{}:{}",
        cell.n_inputs,
        tt,
        cell.cell_name,
        cell.out_pin,
        perm,
        cell.neg_mask,
        u8::from(cell.out_neg)
    )
}

fn parse_gate_symbol(op: &str) -> Option<(usize, u64)> {
    let rest = op.strip_prefix("gate:")?;
    let mut parts = rest.splitn(3, ':');
    let arity = parts.next()?.parse::<usize>().ok()?;
    let tt = u64::from_str_radix(parts.next()?, 16).ok()?;
    Some((arity, tt))
}

fn variant_gate_pattern(cell: &CellVariant, tt: u64) -> String {
    let op = make_gate_symbol(cell, tt);
    if cell.n_inputs == 0 {
        op
    } else {
        let args = (0..cell.n_inputs)
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

fn variant_aig_pattern(cell: &CellVariant, base: &Cell) -> Option<String> {
    let mut orig_to_input = vec![0usize; cell.n_inputs as usize];
    for (input_pos, &orig_idx) in cell.perm.iter().enumerate() {
        *orig_to_input.get_mut(orig_idx as usize)? = input_pos;
    }
    let mut vars = HashMap::<&str, String>::new();
    for (orig_idx, name) in base.vars.iter().enumerate() {
        let input_pos = *orig_to_input.get(orig_idx)?;
        let mut atom = format!("?i{input_pos}");
        if ((cell.neg_mask >> input_pos) & 1) == 1 {
            atom = format!("(~ {atom})");
        }
        vars.insert(name.as_str(), atom);
    }
    let mut aig = expr_to_aig_pattern(&base.expr, &vars)?;
    if cell.out_neg {
        aig = format!("(~ {aig})");
    }
    Some(aig)
}

fn dynamic_gate_rewrites(
    tech: &TechLib,
    gate_variants: &mut HashMap<String, CellVariant>,
) -> Vec<GateRewrite> {
    let mut rules = Vec::<GateRewrite>::new();
    for ((_, tt), vars) in &tech.index {
        for cell in vars {
            let Some(base) = tech.cells.iter().find(|c| c.name == cell.cell_name) else {
                continue;
            };
            let gate_symbol = make_gate_symbol(cell, *tt);
            gate_variants
                .entry(gate_symbol.clone())
                .or_insert_with(|| cell.clone());

            let lhs = match variant_gate_pattern(cell, *tt).parse::<egg::Pattern<GateLang>>() {
                Ok(p) => p,
                Err(_) => continue,
            };
            let Some(rhs_str) = variant_aig_pattern(cell, base) else {
                continue;
            };
            let rhs = match rhs_str.parse::<egg::Pattern<GateLang>>() {
                Ok(p) => p,
                Err(_) => continue,
            };

            if let Ok(rw) = egg::Rewrite::new(
                format!("expand-{gate_symbol}"),
                lhs.clone(),
                rhs.clone(),
            ) {
                rules.push(rw);
            }
            if let Ok(rw) = egg::Rewrite::new(
                format!("cover-{gate_symbol}"),
                rhs,
                lhs,
            ) {
                rules.push(rw);
            }
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

pub fn rewrite_to_gategraph(
    s: &str,
    tech: &TechLib,
    _n_inputs: usize,
) -> Option<(GateEGraph, egg::Id, HashMap<String, CellVariant>)> {
    let expr: egg::RecExpr<GateLang> = s.parse().ok()?;
    let mut runner = egg::Runner::<GateLang, GateTtFold, ()>::default().with_expr(&expr);
    let root = runner.roots[0];
    let mut gate_variants = HashMap::<String, CellVariant>::new();
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
    rules.extend(dynamic_gate_rewrites(tech, &mut gate_variants));
    runner = runner.run(&rules);
    Some((runner.egraph, root, gate_variants))
}
