use egg::*;
use pyo3::prelude::*;

define_language! {
    pub enum Prop {
        Bool(bool),
        "M" = Maj([Id; 3]),
        "~" = Not(Id),
        Symbol(Symbol),
    }
}

pub struct MIGCostFn;
impl egg::CostFunction<Prop> for MIGCostFn {
    type Cost = (usize, usize, usize);
    fn cost<C>(&mut self, enode: &Prop, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let op_depth = match enode {
            Prop::Maj(..) => 1 as usize,
            Prop::Not(..) => 0 as usize,
            _ => 0 as usize,
        };
        let op_area = match enode {
            Prop::Maj(..) => 1 as usize,
            Prop::Not(..) => 0 as usize,
            _ => 0 as usize,
        };
        let op_inv = match enode {
            Prop::Maj(..) => 0 as usize,
            Prop::Not(..) => 1 as usize,
            _ => 0 as usize,
        };
        (
            op_depth + enode.fold(0, |max, id| max.max(costs(id).0)),
            enode.fold(op_area, |sum, id| sum + costs(id).1),
            enode.fold(op_inv, |sum, id| sum + costs(id).2),
        )
    }
}

type EGraph = egg::EGraph<Prop, ConstantFold>;
type Rewrite = egg::Rewrite<Prop, ConstantFold>;

#[derive(Default)]
pub struct ConstantFold;
impl Analysis<Prop> for ConstantFold {
    type Data = Option<(bool, PatternAst<Prop>)>;
    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        merge_option(to, from, |a, b| {
            assert_eq!(a.0, b.0, "Merged non-equal constants");
            DidMerge(false, false)
        })
    }

    fn make(egraph: &EGraph, enode: &Prop) -> Self::Data {
        let x = |i: &Id| egraph[*i].data.as_ref().map(|c| c.0);
        let result = match enode {
            Prop::Bool(c) => Some((*c, c.to_string().parse().unwrap())),
            Prop::Symbol(_) => None,
            Prop::Maj([a, b, c]) => Some((
                x(a)? && x(b)? || x(a)? && x(c)? || x(b)? && x(c)?,
                format!("(M {} {} {})", x(a)?, x(b)?, x(c)?)
                    .parse()
                    .unwrap(),
            )),
            Prop::Not(a) => Some((!x(a)?, format!("(~ {})", x(a)?).parse().unwrap())),
        };
        // println!("Make: {:?} -> {:?}", enode, result);
        result
    }

    fn modify(egraph: &mut EGraph, id: Id) {
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

pub struct MIGCostFn_lp;
#[cfg_attr(docsrs, doc(cfg(feature = "lp")))]
impl LpCostFunction<Prop, ConstantFold> for MIGCostFn_lp {
    fn node_cost(&mut self, _egraph: &EGraph, _eclass: Id, _enode: &Prop) -> f64 {
        let op_depth = match _enode {
            Prop::Maj(..) => 1 as f64,
            Prop::Not(..) => 0 as f64,
            _ => 0 as f64,
        };
        op_depth
    }
}

macro_rules! rule {
    ($name:ident, $left:literal, $right:literal) => {
        #[allow(dead_code)]
        fn $name() -> Rewrite {
            rewrite!(stringify!($name); $left => $right)
        }
    };
    ($name:ident, $name2:ident, $left:literal, $right:literal) => {
        rule!($name, $left, $right);
        rule!($name2, $right, $left);
    };
}

rule! {double_neg,     double_neg_flip,         "(~ (~ ?a))",                 "?a"                              }
rule! {neg,            neg_flip,                "(~ (M ?a ?b ?c))",           "(M (~ ?a) (~ ?b) (~ ?c))"        }
rule! {distri,         distri_flip,             "(M ?a ?b (M ?c ?d ?e))",     "(M (M ?a ?b ?c) (M ?a ?b ?d) ?e)"}
rule! {com_associ,     com_associ_flip,         "(M ?a ?b (M ?c (~ ?b) ?d))", "(M ?a ?b (M ?c ?a ?d))"          }
rule! {associ,         "(M ?a ?b (M ?c ?b ?d))","(M ?d ?b (M ?c ?b ?a))"    }
rule! {comm_lm,        "(M ?a ?b ?c)",          "(M ?b ?a ?c)"              }
rule! {comm_lr,        "(M ?a ?b ?c)",          "(M ?c ?b ?a)"              }
rule! {comm_mr,        "(M ?a ?b ?c)",          "(M ?a ?c ?b)"              }
rule! {maj_2_equ,      "(M ?a ?b ?b)",          "?b"                        }
rule! {maj_2_com,      "(M ?a ?b (~ ?b))",      "?a"                        }

fn prove_something(name: &str, start: &str, rewrites: &[Rewrite], goals: &[&str]) {
    println!("Proving {}", name);

    let start_expr: RecExpr<_> = start.parse().unwrap();
    let goal_exprs: Vec<RecExpr<_>> = goals.iter().map(|g| g.parse().unwrap()).collect();

    let mut runner = Runner::default()
        .with_iter_limit(20)
        .with_node_limit(5_000)
        .with_expr(&start_expr);

    // println!("r0: {:?}", runner.roots[0]);

    // we are assume the input expr is true
    // this is needed for the soundness of lem_imply
    let true_id = runner.egraph.add(Prop::Bool(true));
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

/// parse an expression, simplify it using egg, and pretty print it back out
#[pyfunction]
pub fn simplify(s: &str) -> (String, (usize, usize, usize), (usize, usize, usize)) {
    let all_rules: &[Rewrite; 13] = &[
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
        comm_lm(),
        comm_lr(),
        maj_2_equ(),
        maj_2_com(),
        associ(),
    ];
    // parse the expression, the type annotation tells it which Language to use
    let expr: RecExpr<Prop> = s.parse().unwrap();

    // create an e-graph with the given expression
    let mut runner = Runner::default().with_expr(&expr);
    // the Runner knows which e-class the expression given with `with_expr` is in
    let root = runner.roots[0];

    // use an Extractor to pick the best element of the root eclass
    let inital_cost = Extractor::new(&runner.egraph, MIGCostFn).find_best_cost(root);

    // simplify the expression using a Runner, which runs the given rules over it
    runner = runner.run(all_rules);

    // use an Extractor to pick the best element of the root eclass
    let (best_cost, best) = Extractor::new(&runner.egraph, MIGCostFn).find_best(root);
    println!(
        "Simplified {} with cost {:?} to {} with cost {:?}",
        expr, inital_cost, best, best_cost
    );
    (best.to_string(), inital_cost, best_cost)
}

// This function name should be same as your project name
#[pymodule]
fn mig_egg(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simplify, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prove_chain() {
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
            comm_lm(),
            comm_lr(),
            maj_2_equ(),
            maj_2_com(),
            associ(),
        ];
        prove_something(
            "chain",
            "(M x false (M y true (M u false v)))",
            rules,
            &[
                "(M x false (M y x (M u false v)))",
                "(M (M y x false) x (M false u v))",
            ],
        );
        simplify("(M x false (M y true (M u false v)))");
        simplify("(M (M w x (~ z)) x (M z x y))");
        simplify("(M x3 (M x3 x4 (M x5 x6 x7)) x1)");
    }

    #[test]
    fn const_fold() {
        let start = "(M false true false)";
        let start_expr = start.parse().unwrap();
        let end = "false";
        let end_expr = end.parse().unwrap();
        let mut eg = EGraph::default();
        eg.add_expr(&start_expr);
        eg.rebuild();
        assert!(!eg.equivs(&start_expr, &end_expr).is_empty());
    }
}
