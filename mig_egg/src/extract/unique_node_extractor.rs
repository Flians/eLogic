use super::*;
use rustc_hash::FxHashMap;

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
    pub fn new(first_depth: bool) -> Self {
        Self { first_depth }
    }
}

impl Extractor for UniqueNodeExtractor {
    fn extract(&self, egraph: &EGraph, _roots: &[ClassId]) -> ExtractionResult {
        let mut costs = FxHashMap::<ClassId, CostSet>::with_capacity_and_hasher(
            egraph.classes().len(),
            Default::default(),
        );

        let mut keep_going = true;

        while keep_going {
            keep_going = false;

            'node_loop: for (node_id, node) in &egraph.nodes {
                let cid = egraph.nid_to_cid(node_id);

                // Decode node cost
                let cost_val = node.cost.into_inner();
                let ccost = CCost::decode(cost_val);

                let (node_depth, node_area, node_inv) = if self.first_depth {
                    (ccost.dep, ccost.aom, ccost.inv)
                } else {
                    (ccost.aom, ccost.dep, ccost.inv)
                };

                let mut current_nodes = FxHashMap::default();
                let mut max_child_depth = 0;
                let mut total_tree_area = node_area;

                // compute the cost set from the children
                for child in &node.children {
                    let child_cid = egraph.nid_to_cid(child);
                    if let Some(child_cost_set) = costs.get(child_cid) {
                        // prevent a cycle
                        if child_cost_set.nodes.contains_key(cid) {
                            continue 'node_loop;
                        }
                        current_nodes.extend(child_cost_set.nodes.clone());
                        max_child_depth = std::cmp::max(max_child_depth, child_cost_set.cost.depth);
                        total_tree_area += child_cost_set.cost.tree_area;
                    } else {
                        continue 'node_loop;
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
                    keep_going = true;
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
