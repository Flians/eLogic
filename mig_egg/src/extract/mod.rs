use egraph_serialize::{ClassId, Cost, EGraph, Node, NodeId};
use indexmap::IndexMap;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;

pub use crate::*;

pub mod bottom_up;
pub mod faster_bottom_up;
pub mod faster_greedy_dag;
#[cfg(feature = "ilp-cbc")]
pub mod faster_ilp_cbc;
pub mod global_greedy_dag;
pub mod greedy_dag;
#[cfg(feature = "ilp-cbc")]
pub mod ilp_cbc;

// Allowance for floating point values to be considered equal
pub const EPSILON_ALLOWANCE: f64 = 0.00001;
pub const INFINITY: Cost = unsafe { Cost::new_unchecked(std::f64::INFINITY) };

pub trait Extractor: Sync {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult;

    fn boxed(self) -> Box<dyn Extractor>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}

pub trait MapGet<K, V> {
    fn get(&self, key: &K) -> Option<&V>;
}

impl<K, V> MapGet<K, V> for HashMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    fn get(&self, key: &K) -> Option<&V> {
        HashMap::get(self, key)
    }
}

impl<K, V> MapGet<K, V> for FxHashMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    fn get(&self, key: &K) -> Option<&V> {
        FxHashMap::get(self, key)
    }
}

impl<K, V> MapGet<K, V> for IndexMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    fn get(&self, key: &K) -> Option<&V> {
        IndexMap::get(self, key)
    }
}

#[derive(Default, Clone)]
pub struct ExtractionResult {
    pub choices: IndexMap<ClassId, NodeId>,
}

#[derive(Clone, Copy)]
enum Status {
    Doing,
    Done,
}

impl ExtractionResult {
    pub fn print_aft_expr(&self, egraph: &EGraph) -> String {
        let mut result = String::new();
        for root in &egraph.root_eclasses {
            result.push_str(&self.print_aft_term(egraph, root));
        }
        result
    }

    fn print_aft_term(&self, egraph: &EGraph, class_id: &ClassId) -> String {
        let node_id = &self.choices[class_id];
        let node = &egraph[node_id];

        if node.children.is_empty() {
            // Leaf node
            format!("{}", node.op)
        } else {
            // Internal node
            let children: Vec<String> = node
                .children
                .iter()
                .map(|child| self.print_aft_term(egraph, egraph.nid_to_cid(child)))
                .collect();
            format!("({} {})", node.op, children.join(" "))
        }
    }

    pub fn print_extracted_term(
        &self,
        egraph: &EGraph,
        cost_function: &MIGCostFn_dsi,
    ) -> (String, CCost) {
        let mut metrics: CCost = CCost::default();
        let mut result = String::new();
        for root in &egraph.root_eclasses {
            let (term, term_metrics) = self.print_term(egraph, cost_function, root);
            result.push_str(&term);
            metrics = CCost::merge(&metrics, &term_metrics);
        }
        (result, metrics)
    }

    fn print_term(
        &self,
        egraph: &EGraph,
        cost_function: &MIGCostFn_dsi,
        class_id: &ClassId,
    ) -> (String, CCost) {
        let node_id = &self.choices[class_id];
        let node = &egraph[node_id];
        let metrics = cost_function.cal_cur_cost_bystr(node.op.as_str());

        if node.children.is_empty() {
            // Leaf node
            (format!("{}", node.op), metrics)
        } else {
            // Internal node
            let mut children_terms = Vec::new();
            let mut to_cost = CCost::default();
            for child in &node.children {
                let (child_term, child_metrics) =
                    self.print_term(egraph, cost_function, egraph.nid_to_cid(child));
                to_cost = CCost::merge(&to_cost, &child_metrics);
                children_terms.push(child_term);
            }

            (
                format!("({} {})", node.op, children_terms.join(" ")),
                metrics + to_cost,
            )
        }
    }

    pub fn check(&self, egraph: &EGraph) {
        // should be a root
        assert!(!egraph.root_eclasses.is_empty());

        // All roots should be selected.
        for cid in egraph.root_eclasses.iter() {
            assert!(self.choices.contains_key(cid));
        }

        // No cycles
        assert!(self.find_cycles(&egraph, &egraph.root_eclasses).is_empty());

        // Nodes should match the class they are selected into.
        for (cid, nid) in &self.choices {
            let node = &egraph[nid];
            assert!(node.eclass == *cid);
        }

        // All the nodes the roots depend upon should be selected.
        let mut todo: Vec<ClassId> = egraph.root_eclasses.to_vec();
        let mut visited: FxHashSet<ClassId> = Default::default();
        while let Some(cid) = todo.pop() {
            if !visited.insert(cid.clone()) {
                continue;
            }
            assert!(self.choices.contains_key(&cid));

            for child in &egraph[&self.choices[&cid]].children {
                todo.push(egraph.nid_to_cid(child).clone());
            }
        }
    }

    pub fn choose(&mut self, class_id: ClassId, node_id: NodeId) {
        self.choices.insert(class_id, node_id);
    }

    pub fn find_cycles(&self, egraph: &EGraph, roots: &[ClassId]) -> Vec<ClassId> {
        // let mut status = vec![Status::Todo; egraph.classes().len()];
        let mut status = IndexMap::<ClassId, Status>::default();
        let mut cycles = vec![];
        for root in roots {
            // let root_index = egraph.classes().get_index_of(root).unwrap();
            self.cycle_dfs(egraph, root, &mut status, &mut cycles)
        }
        cycles
    }

    fn cycle_dfs(
        &self,
        egraph: &EGraph,
        class_id: &ClassId,
        status: &mut IndexMap<ClassId, Status>,
        cycles: &mut Vec<ClassId>,
    ) {
        match status.get(class_id).cloned() {
            Some(Status::Done) => (),
            Some(Status::Doing) => cycles.push(class_id.clone()),
            None => {
                status.insert(class_id.clone(), Status::Doing);
                let node_id = &self.choices[class_id];
                let node = &egraph[node_id];
                for child in &node.children {
                    let child_cid = egraph.nid_to_cid(child);
                    self.cycle_dfs(egraph, child_cid, status, cycles)
                }
                status.insert(class_id.clone(), Status::Done);
            }
        }
    }

    pub fn tree_cost(&self, egraph: &EGraph, roots: &[ClassId]) -> Cost {
        let node_roots = roots
            .iter()
            .map(|cid| self.choices[cid].clone())
            .collect::<Vec<NodeId>>();
        self.tree_cost_rec(egraph, &node_roots, &mut HashMap::new())
    }

    fn tree_cost_rec(
        &self,
        egraph: &EGraph,
        roots: &[NodeId],
        memo: &mut HashMap<NodeId, Cost>,
    ) -> Cost {
        let mut cost = Cost::default();
        for root in roots {
            if let Some(c) = memo.get(root) {
                cost += *c;
                continue;
            }
            let class = egraph.nid_to_cid(root);
            let node = &egraph[&self.choices[class]];
            let inner = node.cost + self.tree_cost_rec(egraph, &node.children, memo);
            memo.insert(root.clone(), inner);
            cost += inner;
        }
        cost
    }

    // this will loop if there are cycles
    pub fn dag_cost(&self, egraph: &EGraph, roots: &[ClassId]) -> Cost {
        let mut costs: IndexMap<ClassId, Cost> = IndexMap::new();
        let mut todo: Vec<ClassId> = roots.to_vec();
        while let Some(cid) = todo.pop() {
            let node_id = &self.choices[&cid];
            let node = &egraph[node_id];
            if costs.insert(cid.clone(), node.cost).is_some() {
                continue;
            }
            for child in &node.children {
                todo.push(egraph.nid_to_cid(child).clone());
            }
        }
        costs.values().sum()
    }
    pub fn dag_cost_size_from_node(&self, egraph: &EGraph, node_id: &NodeId, visited: &mut FxHashSet<NodeId>) -> u32 {
        // 如果节点已经访问过（环检测），直接返回 0
        if !visited.insert(node_id.clone()) {
            return 0;
        }
    
        let node = &egraph[node_id];
    
        // 递归计算所有子节点的总大小
        let total_child_size: u32 = node
            .children
            .iter()
            .map(|child| self.dag_cost_size_from_node(egraph, child, visited)) // 对每个子节点递归调用
            .sum();
    
        // 当前节点的大小贡献
        total_child_size + CCost::decode(node.cost.into()).aom
    }
    pub fn dag_cost_size(&self, egraph: &EGraph, roots: &[ClassId]) -> u32 {
        let mut costs: IndexMap<ClassId, Cost> = IndexMap::new();
        let mut todo: Vec<ClassId> = roots.to_vec();

        while let Some(cid) = todo.pop() {
            let node_id = &self.choices[&cid];
            let node = &egraph[node_id];
            if costs.insert(cid.clone(), node.cost).is_some() {
                continue;
            }
            for child in &node.children {
                todo.push(egraph.nid_to_cid(child).clone());
            }
        }
        costs
            .into_iter()
            .fold(0, |sum, (_, cost)| sum + CCost::decode(cost.into()).aom)
    }

    pub fn dag_cost_depth(&self, egraph: &EGraph, roots: &[ClassId]) -> u32 {
        let mut depth = CCost::default().dep;
        for root in roots {
            let depth_new = self.calculate_depth(egraph, root);
            depth = cmp::max(depth, depth_new);
        }
        depth
    }
    fn calculate_depth(&self, egraph: &EGraph, roots: &ClassId) -> u32 {
        let node_id = &self.choices[roots];
        let node = &egraph[node_id];
        let max_child_depth = node
            .children
            .iter()
            .map(|child| {
                let child_cid = egraph.nid_to_cid(child);
                self.calculate_depth(egraph, child_cid)
            })
            .max()
            .unwrap_or(0);

        max_child_depth + CCost::decode(node.cost.into()).dep
    }

    pub fn node_sum_cost<M>(&self, egraph: &EGraph, node: &Node, costs: &M) -> Cost
    where
        M: MapGet<ClassId, Cost>,
    {
        node.cost
            + node
                .children
                .iter()
                .map(|n| {
                    let cid = egraph.nid_to_cid(n);
                    costs.get(cid).unwrap_or(&INFINITY)
                })
                .sum::<Cost>()
    }

    // node_depth_cost method calculates the maximum cost among a node and its children
    pub fn node_depth_cost<M>(&self, egraph: &EGraph, node: &Node, costs: &M) -> Cost
    where
        M: MapGet<ClassId, Cost>,
    {
        let child_max_cost = node
            .children
            .iter()
            .map(|n| {
                let cid = egraph.nid_to_cid(n);
                costs.get(cid).unwrap_or(&INFINITY)
            })
            .max()
            .copied()
            .unwrap_or(Cost::default());

        node.cost + child_max_cost
    }
}
