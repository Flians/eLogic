use super::ExtractionResult; // 使用统一的 ExtractionResult
use super::*;
use indexmap::IndexMap;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Clone)]
struct CostSet {
    // It's slightly faster if this is an HashMap rather than an FxHashMap.
    costs: HashMap<ClassId, CCost>,
    total: CCost,
    choice: NodeId,
}

pub struct FasterGreedyDagExtractor {
    first_depth: bool,
}
impl Default for FasterGreedyDagExtractor {
    fn default() -> Self {
        FasterGreedyDagExtractor { first_depth: true }
    }
}
impl FasterGreedyDagExtractor {
    pub fn new(f_dep: Option<bool>) -> Self {
        Self {
            first_depth: f_dep.unwrap_or(true),
        }
    }
}

impl FasterGreedyDagExtractor {
    fn calculate_cost_set(
        egraph: &EGraph,
        node_id: NodeId,
        costs: &FxHashMap<ClassId, CostSet>,
        best_cost: CCost,
        extraction_result: &ExtractionResult, // 传入 ExtractionResult
        first_depth: bool,
    ) -> CostSet {
        let node = &egraph[&node_id];
        let cid = egraph.nid_to_cid(&node_id);

        // 处理叶子节点
        if node.children.is_empty() {
            return CostSet {
                costs: HashMap::from([(cid.clone(), CCost::decode(node.cost.into()))]),
                total: CCost::decode(node.cost.into()),
                choice: node_id.clone(),
            };
        }

        // 获取子节点的唯一等价类
        let mut childrens_classes = node
            .children
            .iter()
            .map(|c| egraph.nid_to_cid(&c).clone())
            .collect::<Vec<ClassId>>();
        childrens_classes.sort();
        childrens_classes.dedup();

        let first_cost = costs.get(&childrens_classes[0]).unwrap();

        // 剪枝逻辑
        if childrens_classes.contains(cid)
            || (childrens_classes.len() == 1
                && (CCost::decode(node.cost.into()) + first_cost.total > best_cost))
        {
            return CostSet {
                costs: Default::default(),
                total: CCost::max(),
                choice: node_id.clone(),
            };
        }

        // 合并子节点的成本集合
        let id_of_biggest = childrens_classes
            .iter()
            .max_by_key(|s| costs.get(s).unwrap().costs.len())
            .unwrap();
        let mut result = costs.get(&id_of_biggest).unwrap().costs.clone();
        for child_cid in &childrens_classes {
            if child_cid == id_of_biggest {
                continue;
            }

            let next_cost = &costs.get(child_cid).unwrap().costs;
            for (key, value) in next_cost.iter() {
                result.insert(key.clone(), value.clone());
            }
        }

        let contains = result.contains_key(&cid);
        result.insert(cid.clone(), CCost::decode(node.cost.into()));

        // 使用 ExtractionResult 的 calculate_depth 方法计算深度
        let max_child_depth = node
            .children
            .iter()
            .map(|child| {
                let child_cid = egraph.nid_to_cid(child);
                extraction_result.calculate_depth(egraph, child_cid, Some(first_depth))
                // 调用 ExtractionResult 的递归深度计算
            })
            .max()
            .unwrap_or(0);

        // 如果当前节点的操作符是 "M"，深度加 1
        let max_depth = if node.op == "M" || node.op == "&" {
            max_child_depth + 1
        } else {
            max_child_depth
        };

        // 计算 DAG 的大小
        let total_size: u32 = result
            .values()
            .map(|cost| if first_depth { cost.aom } else { cost.dep })
            .sum();

        // 构造结果
        let result_cost = if contains {
            CCost::max()
        } else {
            CCost {
                aom: total_size, // 总大小
                dep: max_depth,  // 最大深度
                inv: 0,          // 假设 inv 不变
            }
        };

        return CostSet {
            costs: result,
            total: result_cost,
            choice: node_id.clone(),
        };
    }
}

impl Extractor for FasterGreedyDagExtractor {
    fn extract(&self, egraph: &EGraph, _roots: &[ClassId]) -> extract::ExtractionResult {
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

        // 初始化结果数据结构
        let mut result = ExtractionResult::default();
        let mut costs = FxHashMap::<ClassId, CostSet>::with_capacity_and_hasher(
            egraph.classes().len(),
            Default::default(),
        );

        while let Some(node_id) = analysis_pending.pop() {
            let class_id = n2c(&node_id);
            let node = &egraph[&node_id];

            if node.children.iter().all(|c| costs.contains_key(n2c(c))) {
                let lookup = costs.get(class_id);
                let mut prev_cost = CCost::max();
                if let Some(existing) = lookup {
                    prev_cost = existing.total;
                }
                for (cid, cost_set) in costs.clone() {
                    result.choose(cid, cost_set.choice);
                }
                // 调用 calculate_cost_set，传入 ExtractionResult
                let cost_set = FasterGreedyDagExtractor::calculate_cost_set(
                    egraph,
                    node_id.clone(),
                    &costs,
                    prev_cost,
                    &result, // 传递 ExtractionResult 的引用
                    self.first_depth,
                );

                if cost_set.total < prev_cost {
                    costs.insert(class_id.clone(), cost_set);
                    analysis_pending.extend(parents[class_id].iter().cloned());
                }
            }
        }

        /*
        for root in _roots {
            if let Some(cost_set) = costs.get(root) {
                println!("Root {:?} -> CCost: {:?}", root, cost_set.total);
            } else {
                println!("Root {:?} has no cost calculated", root);
            }
        }
        */

        // 将选择的节点加入 ExtractionResult
        for (cid, cost_set) in costs {
            result.choose(cid, cost_set.choice);
        }

        // 返回 ExtractionResult
        result
    }
}

/** A data structure to maintain a queue of unique elements.

Notably, insert/pop operations have O(1) expected amortized runtime complexity.

Thanks @Bastacyclop for the implementation!
*/
#[derive(Clone)]
#[cfg_attr(feature = "serde-1", derive(Serialize, Deserialize))]
pub(crate) struct UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    set: FxHashSet<T>, // hashbrown::
    queue: std::collections::VecDeque<T>,
}

impl<T> Default for UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    fn default() -> Self {
        UniqueQueue {
            set: Default::default(),
            queue: std::collections::VecDeque::new(),
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

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        let r = self.queue.is_empty();
        debug_assert_eq!(r, self.set.is_empty());
        r
    }
}
