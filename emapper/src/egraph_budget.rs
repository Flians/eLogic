//! Lifetime admission accounting, shared by import and saturation. Rebuild and
//! union may shrink resident storage, but never refund admitted e-nodes.
use crate::egraph_sat::AigLang;
use egg::{EGraph, ENodeOrVar, Id, Language, PatternAst, Subst};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
pub(crate) type SharedAdmission = Arc<Mutex<NodeAdmission>>;

#[derive(Debug)]
pub(crate) struct NodeAdmission {
    limit: usize,
    admitted: usize,
}
impl NodeAdmission {
    pub(crate) fn new(limit: usize) -> Self { Self { limit, admitted: 0 } }
    pub(crate) fn admitted(&self) -> usize { self.admitted }
    pub(crate) fn remaining(&self) -> usize { self.limit - self.admitted }
    pub(crate) fn add(&mut self, graph: &mut EGraph<AigLang, ()>, node: AigLang) -> Option<Id> {
        let node = node.map_children(|id| graph.find(id));
        if let Some(id) = graph.lookup(node.clone()) { return Some(id); }
        if self.remaining() == 0 { return None; }
        let id = graph.add(node);
        self.admitted += 1;
        Some(id)
    }
    /// Read-only exact RHS dry run. Variables and hash-consed duplicates cost
    /// zero; repeated new subexpressions count once. No partial RHS is installed.
    pub(crate) fn rhs_cost(&self, graph: &EGraph<AigLang, ()>, ast: &PatternAst<AigLang>, subst: &Subst) -> usize {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        enum Ref { Existing(Id), Pending(usize) }
        let mut ids = Vec::new();
        let mut pending = HashMap::new();
        for entry in ast.as_ref() {
            let id = match entry {
                ENodeOrVar::Var(v) => Ref::Existing(graph.find(subst[*v])),
                ENodeOrVar::ENode(node) => {
                    let children: Vec<_> = node.children().iter().map(|id| ids[usize::from(*id)]).collect();
                    let mut existing = Vec::new();
                    let all_existing = children.iter().all(|id| if let Ref::Existing(id) = id { existing.push(*id); true } else { false });
                    let found = if all_existing {
                        let mut i = 0;
                        graph.lookup(node.clone().map_children(|_| { let id = existing[i]; i += 1; id }))
                    } else { None };
                    if let Some(id) = found { Ref::Existing(graph.find(id)) }
                    else {
                        let key = (node.clone().map_children(|_| Id::from(0)), children);
                        let next = pending.len();
                        Ref::Pending(*pending.entry(key).or_insert(next))
                    }
                }
            };
            ids.push(id);
        }
        pending.len()
    }
    /// RHS application is a sequence of add/union operations without rebuild.
    /// Its resident delta equals actual admissions; the read-only dry run
    /// established the capacity first. Unknown/custom appliers are not accepted.
    pub(crate) fn charge_rhs(&mut self, before: usize, after: usize, reserved: usize) {
        let added = after.checked_sub(before).expect("RHS unexpectedly removed resident nodes");
        assert!(added <= reserved && added <= self.remaining(), "RHS exceeded its admission reservation");
        self.admitted += added;
    }
    pub(crate) fn rejected_import(&self, graph: &EGraph<AigLang, ()>, route: &str) {
        eprintln!("budget: cumulative admission rejected before insertion: route={route}, admitted={}, resident={}, node allowance={}", self.admitted, graph.total_number_of_nodes(), self.limit);
        if let Some(path) = std::env::var_os("EGM_POOL_DIAG_PATH") {
            let path = std::path::Path::new(&path).with_extension("admission.json");
            let _ = std::fs::write(path, serde_json::json!({"schema_version":1,"import_complete":false,"route":route,"cumulative_admitted_enodes":self.admitted,"resident_enodes":graph.total_number_of_nodes(),"node_allowance":self.limit,"rejected_before_insertion":true,"refund_on_rebuild":false}).to_string());
        }
    }
    pub(crate) fn import(&mut self, graph: &mut EGraph<AigLang, ()>, node: AigLang, route: &str) -> Option<Id> {
        let result = self.add(graph, node);
        if result.is_none() { self.rejected_import(graph, route); }
        result
    }
}

/// One lifetime ledger, also used by import. Each RHS is dry-run before
/// mutation; zero-new-node unions remain legal when the allowance is exhausted.
struct ReservedRewrite {
    inner: std::sync::Arc<dyn egg::Applier<AigLang, ()> + Send + Sync>,
    admission: crate::egraph_budget::SharedAdmission,
    skipped: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}
impl egg::Applier<AigLang, ()> for ReservedRewrite {
    fn apply_one(&self, graph: &mut egg::EGraph<AigLang, ()>, id: Id,
        subst: &egg::Subst, _ast: Option<&egg::PatternAst<AigLang>>, name: egg::Symbol) -> Vec<Id> {
        let mut admission = self.admission.lock().unwrap();
        let rhs = self.inner.get_pattern_ast().expect("strict rewrite must expose its exact RHS");
        let needed = admission.rhs_cost(graph, rhs, subst);
        if needed > admission.remaining() {
            self.skipped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return vec![];
        }
        let before = graph.total_number_of_nodes();
        // No explanation mode is enabled in this mapper. Passing no searcher
        // AST avoids re-instantiating the LHS; only the preflighted RHS is added.
        let result = self.inner.apply_one(graph, id, subst, None, name);
        admission.charge_rhs(before, graph.total_number_of_nodes(), needed);
        result
    }
    fn vars(&self) -> Vec<egg::Var> { self.inner.vars() }
    fn get_pattern_ast(&self) -> Option<&egg::PatternAst<AigLang>> { self.inner.get_pattern_ast() }
}
pub(crate) fn reserve_rewrites(rules: &mut [egg::Rewrite<AigLang, ()>],
    admission: crate::egraph_budget::SharedAdmission,
) -> std::sync::Arc<std::sync::atomic::AtomicUsize> {
    let skipped = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    for rule in rules {
        assert!(rule.applier.get_pattern_ast().is_some(), "unaccounted custom strict rewrite: {}", rule.name);
        rule.applier = std::sync::Arc::new(ReservedRewrite {
            inner: rule.applier.clone(), admission: admission.clone(), skipped: skipped.clone(),
        });
    }
    skipped
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cumulative_admission_never_refunds_rebuild_and_allows_duplicates_at_cap() {
        let mut g=EGraph::default(); let mut b=NodeAdmission::new(4);
        let a=b.add(&mut g,AigLang::Pi(0)).unwrap();let c=b.add(&mut g,AigLang::Pi(1)).unwrap();
        b.add(&mut g,AigLang::Not(a)).unwrap();b.add(&mut g,AigLang::Not(c)).unwrap();
        assert!(b.add(&mut g,AigLang::Zero).is_none());assert_eq!(g.total_number_of_nodes(),4);
        g.union(a,c);g.rebuild();assert!(g.total_number_of_nodes()<4);
        assert!(b.add(&mut g,AigLang::Not(c)).is_some());
        assert!(b.add(&mut g,AigLang::Zero).is_none());assert_eq!(b.admitted(),4);
    }
    #[test]
    fn rhs_dry_run_matches_actual_additions_across_unions() {
        use egg::{Applier, Pattern};
        let patterns=["?a","(~ ?a)","(& (~ ?a) (~ ?a))","(& (& ?a ?b) (& ?a ?b))","(& ?b (~ (~ ?a)))"];
        for seed in 0..16 {
            let mut g=EGraph::default();let mut b=NodeAdmission::new(100);
            let a=b.add(&mut g,AigLang::Pi(0)).unwrap();let c=b.add(&mut g,AigLang::Pi(1)).unwrap();
            let n=b.add(&mut g,AigLang::Not(a)).unwrap();
            if seed&1!=0 {b.add(&mut g,AigLang::And([a,c])).unwrap();}
            if seed&2!=0 {b.add(&mut g,AigLang::Not(n)).unwrap();}
            if seed&4!=0 {g.union(a,c);}
            if seed&8!=0 {g.rebuild();}
            let mut subst=Subst::default();subst.insert("?a".parse().unwrap(),a);subst.insert("?b".parse().unwrap(),c);
            for text in patterns {
                let rhs:Pattern<AigLang>=text.parse().unwrap();let expected=b.rhs_cost(&g,&rhs.ast,&subst);
                let mut trial=g.clone();let before=trial.total_number_of_nodes();
                rhs.apply_one(&mut trial,a,&subst,None,"test".into());
                assert_eq!(expected,trial.total_number_of_nodes()-before,"seed{seed}, RHS{text}");
            }
        }
    }

    #[test]
    fn rhs_reservation_counts_new_subterms_not_variables_or_existing_nodes() {
        let mut g=EGraph::default();let mut b=NodeAdmission::new(3);
        let a=b.add(&mut g,AigLang::Pi(0)).unwrap(); let n=b.add(&mut g,AigLang::Not(a)).unwrap();
        let mut s=Subst::default();s.insert("?a".parse().unwrap(),a);s.insert("?n".parse().unwrap(),n);
        for (text,expected) in [("?a",0),("(~ ?a)",0),("(& (~ ?n) (~ ?n))",2)] {
            let p:egg::Pattern<AigLang>=text.parse().unwrap();assert_eq!(b.rhs_cost(&g,&p.ast,&s),expected);
        }
        assert_eq!(b.admitted(),2);assert_eq!(g.total_number_of_nodes(),2);
    }
}
