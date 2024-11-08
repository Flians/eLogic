use egg::{Id, Language};
use std::cmp;
use std::collections::HashSet;
use std::ops::Add;

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

pub fn simplify(
    s: &str,
    ori_deps: Option<&[u32]>,
    rewrites: Option<&[CRewrite]>,
) -> (egg::RecExpr<MIG>, CCost, CCost) {
    let all_rules = [
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
    let rules = rewrites.unwrap_or(&all_rules);
    // parse the expression, the type annotation tells it which Language to use
    let expr: egg::RecExpr<MIG> = s.parse().unwrap();

    let vars: [u32; 26] = [0; 26];
    let slice: &[u32] = ori_deps.unwrap_or(&vars);

    // create an e-graph with the given expression
    let mut runner = egg::Runner::default().with_expr(&expr);
    // the Runner knows which e-class the expression given with `with_expr` is in
    let root = runner.roots[0];

    // use an Extractor to pick the best element of the root eclass
    let inital_cost =
        egg::Extractor::new(&runner.egraph, MIGCostFn_dsi::new(&runner.egraph, slice))
            .find_best_cost(root);

    // simplify the expression using a Runner, which runs the given rules over it
    runner = runner.run(rules);

    // use an Extractor to pick the best element of the root eclass
    let (best_cost, best) =
        egg::Extractor::new(&runner.egraph, MIGCostFn_dsi::new(&runner.egraph, slice))
            .find_best(root);
    println!(
        "Simplified {} with cost {:?} to {} with cost {:?}",
        expr, inital_cost, best, best_cost
    );
    (best, inital_cost, best_cost)
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
        */
        simplify("(M 0 (~ (M 0 (~ a) b)) (M 0 c (~ d)))", None, None);
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
