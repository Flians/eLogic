use super::*;
use indexmap::IndexMap;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

#[derive(Clone, Debug, Default)]
struct DagCost {
    area: u32,
    depth: u32,
    inv: u32,
    tree_area: u32,
}

struct CostSet {
    // Map ClassId to (area, inv) of the node picked for that class
    nodes: FxHashMap<ClassId, (u32, u32)>,
    cost: DagCost,
    choice: NodeId,
}

pub struct UniqueNodeExtractor {
    pub first_depth: bool,
}

impl UniqueNodeExtractor {
    pub fn new(f_dep: Option<bool>) -> Self {
        Self {
            first_depth: f_dep.unwrap_or(true),
        }
    }
}

struct UniqueQueue<T> {
    set: FxHashSet<T>,
    queue: VecDeque<T>,
}

impl<T> Default for UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    fn default() -> Self {
        UniqueQueue {
            set: Default::default(),
            queue: VecDeque::new(),
        }
    }
}

impl<T> UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    pub fn insert(&mut self, t: T) {
        if self.set.insert(t.clone()) {
            self.queue.push_back(t);
        }
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for t in iter.into_iter() {
            self.insert(t);
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        let res = self.queue.pop_front();
        res.as_ref().map(|t| self.set.remove(t));
        res
    }
}

impl Extractor for UniqueNodeExtractor {
    fn extract(&self, egraph: &EGraph, _roots: &[ClassId]) -> ExtractionResult {
        let mut parents = IndexMap::<ClassId, Vec<NodeId>>::with_capacity(egraph.classes().len());
        let n2c = |nid: &NodeId| egraph.nid_to_cid(nid);
        let mut analysis_pending = UniqueQueue::default();

        for class in egraph.classes().values() {
            parents.insert(class.id.clone(), Vec::new());
        }

        for class in egraph.classes().values() {
            for node in &class.nodes {
                for c in &egraph[node].children {
                    // compute parents of this enode
                    parents[n2c(c)].push(node.clone());
                }

                // start the analysis from leaves
                if egraph[node].is_leaf() {
                    analysis_pending.insert(node.clone());
                }
            }
        }

        let mut costs = FxHashMap::<ClassId, CostSet>::with_capacity_and_hasher(
            egraph.classes().len(),
            Default::default(),
        );

        while let Some(node_id) = analysis_pending.pop() {
            let class_id = n2c(&node_id);
            let node = &egraph[&node_id];

            if node.children.iter().all(|c| costs.contains_key(n2c(c))) {
                let cid = class_id;

                // Decode node cost
                let cost_val = node.cost.into_inner();
                let ccost = CCost::decode(cost_val);

                let (node_depth, node_area, node_inv) = if self.first_depth {
                    (ccost.dep, ccost.aom, ccost.inv)
                } else {
                    (ccost.aom, ccost.dep, ccost.inv)
                };

                let mut max_child_depth = 0;
                let mut total_tree_area = node_area;
                let mut cycle_detected = false;

                // Find the child with the largest node set to clone from
                let mut largest_child_idx = None;
                let mut largest_size = 0;

                for (i, child) in node.children.iter().enumerate() {
                    let child_cid = egraph.nid_to_cid(child);
                    if let Some(child_cost_set) = costs.get(child_cid) {
                        if child_cost_set.nodes.contains_key(cid) {
                            cycle_detected = true;
                            break;
                        }
                        if child_cost_set.nodes.len() > largest_size {
                            largest_size = child_cost_set.nodes.len();
                            largest_child_idx = Some(i);
                        }
                    } else {
                         cycle_detected = true;
                         break;
                    }
                }

                if cycle_detected {
                    continue;
                }

                let mut current_nodes = if let Some(idx) = largest_child_idx {
                    let child_cid = egraph.nid_to_cid(&node.children[idx]);
                    costs.get(child_cid).unwrap().nodes.clone()
                } else {
                    FxHashMap::default()
                };

                // compute the cost set from the children
                for (i, child) in node.children.iter().enumerate() {
                    if Some(i) == largest_child_idx {
                        let child_cid = egraph.nid_to_cid(child);
                        let child_cost_set = costs.get(child_cid).unwrap();
                        max_child_depth = std::cmp::max(max_child_depth, child_cost_set.cost.depth);
                        total_tree_area += child_cost_set.cost.tree_area;
                        continue;
                    }

                    let child_cid = egraph.nid_to_cid(child);
                    if let Some(child_cost_set) = costs.get(child_cid) {
                        current_nodes.extend(child_cost_set.nodes.clone());
                        max_child_depth = std::cmp::max(max_child_depth, child_cost_set.cost.depth);
                        total_tree_area += child_cost_set.cost.tree_area;
                    }
                }

                // Add current node
                current_nodes.insert(cid.clone(), (node_area, node_inv));

                // Calculate totals
                let total_area: u32 = current_nodes.values().map(|(a, _)| a).sum();
                let total_inv: u32 = current_nodes.values().map(|(_, i)| i).sum();
                let total_depth = max_child_depth + node_depth;

                let new_cost = DagCost {
                    area: total_area,
                    depth: total_depth,
                    inv: total_inv,
                    tree_area: total_tree_area,
                };

                // Update if better
                let update = if let Some(old_cost_set) = costs.get(cid) {
                    if self.first_depth {
                        // Depth first
                        if new_cost.depth < old_cost_set.cost.depth {
                            true
                        } else if new_cost.depth == old_cost_set.cost.depth {
                            if new_cost.area < old_cost_set.cost.area {
                                true
                            } else if new_cost.area == old_cost_set.cost.area {
                                if new_cost.tree_area < old_cost_set.cost.tree_area {
                                    true
                                } else {
                                    new_cost.tree_area == old_cost_set.cost.tree_area
                                        && new_cost.inv < old_cost_set.cost.inv
                                }
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        // Area first
                        if new_cost.area < old_cost_set.cost.area {
                            true
                        } else if new_cost.area == old_cost_set.cost.area {
                            if new_cost.depth < old_cost_set.cost.depth {
                                true
                            } else if new_cost.depth == old_cost_set.cost.depth {
                                if new_cost.tree_area < old_cost_set.cost.tree_area {
                                    true
                                } else {
                                    new_cost.tree_area == old_cost_set.cost.tree_area
                                        && new_cost.inv < old_cost_set.cost.inv
                                }
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    }
                } else {
                    true
                };

                if update {
                    let new_cost_set = CostSet {
                        nodes: current_nodes,
                        cost: new_cost,
                        choice: node_id.clone(),
                    };
                    costs.insert(cid.clone(), new_cost_set);
                    analysis_pending.extend(parents[cid].iter().cloned());
                }
            }
        }

        let mut result = ExtractionResult::default();
        for (cid, cost_set) in costs {
            result.choose(cid, cost_set.choice);
        }
        result
    }
}
