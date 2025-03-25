/* An ILP extractor that returns the optimal DAG-extraction.

   If the timeout is reached, it will return the result of the faster-greedy-dag extractor.
*/

use super::*;
use coin_cbc::{Col, Model, Sense};
use indexmap::IndexSet;

struct ClassVars {
    active: Col,
    nodes: Vec<Col>,
}

pub struct CbcExtractor {
    first_depth: bool,
}
impl Default for CbcExtractor {
    fn default() -> Self {
        CbcExtractor { first_depth: true }
    }
}
impl CbcExtractor {
    pub fn new(f_dep: Option<bool>) -> Self {
        Self {
            first_depth: f_dep.unwrap_or(true),
        }
    }
}

impl Extractor for CbcExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        // return extract(egraph, roots, std::u32::MAX);
        return extract(egraph, roots, 30, self.first_depth);
    }
}

fn extract(
    egraph: &EGraph,
    roots: &[ClassId],
    timeout_seconds: u32,
    first_depth: bool,
) -> ExtractionResult {
    println!("extracting with cbc");
    let mut model = Model::default();
    model.set_parameter("log", "0");
    model.set_parameter("seconds", &timeout_seconds.to_string());
    model.set_parameter("allowableGap", "0.2"); // Make more strict
    model.set_parameter("maxSolutions", "10"); // Increase solutions explored
    model.set_parameter("maxNodes", "10000"); // Increase search space
                                              // model.set_parameter("seconds", "60");          // Increase timeout

    // 给每个 eclass 及其 enodes 建立 binary 变量
    let vars: IndexMap<ClassId, ClassVars> = egraph
        .classes()
        .values()
        .map(|class| {
            let cvars = ClassVars {
                active: model.add_binary(),
                nodes: class.nodes.iter().map(|_| model.add_binary()).collect(),
            };
            (class.id.clone(), cvars)
        })
        .collect();

    // 约束：class.active == sum(node.active)
    for (class_id, class) in &vars {
        let row = model.add_row();
        model.set_row_equal(row, 0.0);
        model.set_weight(row, class.active, -1.0);
        for &node_active in &class.nodes {
            model.set_weight(row, node_active, 1.0);
        }
    }

    // 约束：若 node.active，则它所有 child 的 eclass.active 也要激活
    for (class_id, class_vars) in &vars {
        let childrens_classes_var = |nid: NodeId| {
            egraph[&nid]
                .children
                .iter()
                .map(|n| egraph[n].eclass.clone())
                .map(|n| vars[&n].active)
                .collect::<IndexSet<_>>()
        };

        for (node_id, &node_active) in egraph[class_id].nodes.iter().zip(&class_vars.nodes) {
            for child_active in childrens_classes_var(node_id.clone()) {
                let row = model.add_row();
                model.set_row_upper(row, 0.0);
                model.set_weight(row, node_active, 1.0);
                model.set_weight(row, child_active, -1.0);
            }
        }
    }

    // 目标函数：Minimize
    model.set_obj_sense(Sense::Minimize);

    // ------------------【修改部分：decode + 线性加权】------------------ //
    // 在 baseline 中, CCost::encode() = bits => f64, 但非常小(1e-309量级).
    // 这里解码后, 对 dep,aom,inv 分别给定 W_dep=1_000_000, W_aom=1_000, W_inv=1 的加权,
    // 你可以根据需求更改这些权重
    let w_dep = 1_000_000.0;
    let w_aom = 1_000.0;
    let w_inv = 1.0;

    for class in egraph.classes().values() {
        for (node_id, &node_active) in class.nodes.iter().zip(&vars[&class.id].nodes) {
            let node = &egraph[node_id];
            let node_cost_bits = node.cost.into_inner(); // encode后的f64

            if node_cost_bits == 0.0 {
                // 说明 (dep,aom,inv) = (0,0,0) => cost=0, 不需要加到 objective
                continue;
            }

            // 解码
            let c = CCost::decode(node_cost_bits);
            let dep = c.dep as f64;
            let aom = c.aom as f64;
            let inv = c.inv as f64;

            let cost_scalar = dep * w_dep + aom * w_aom + inv * w_inv;
            // cost_scalar > 0 时才设置到目标函数
            if cost_scalar > 0.0 {
                model.set_obj_coeff(node_active, cost_scalar);
            }
        }
    }

    // 对所有 root eclass 设置 active=1 (确保提取)
    for root in roots {
        model.set_col_lower(vars[root].active, 1.0);
    }

    // 防环约束
    block_cycles(&mut model, &vars, egraph);

    // 求解
    let solution = model.solve();
    log::info!(
        "CBC status {:?}, {:?}, obj = {}",
        solution.raw().status(),
        solution.raw().secondary_status(),
        solution.raw().obj_value(),
    );

    // 如果 solver 未完成（例如超时），fallback
    if solution.raw().status() != coin_cbc::raw::Status::Finished {
        println!("Unfinished CBC solution => fallback to FasterGreedyDag");
        assert!(timeout_seconds != std::u32::MAX);
        let initial_result =
            super::faster_greedy_dag::FasterGreedyDagExtractor::new(Some(first_depth))
                .extract(egraph, roots);
        log::info!("Unfinished CBC solution => fallback to FasterGreedyDag");
        return initial_result;
    }

    // 根据解的值, 选出激活的 node
    let mut result = ExtractionResult::default();
    for (id, var) in &vars {
        // eclass 的激活
        let active = solution.col(var.active) > 0.0;
        if active {
            // 找到哪个 node 被选中
            let node_idx = var
                .nodes
                .iter()
                .position(|&n| solution.col(n) > 0.0)
                .unwrap();
            let node_id = egraph[id].nodes[node_idx].clone();
            result.choose(id.clone(), node_id);
        }
    }

    result
}

/*

 To block cycles, we enforce that a topological ordering exists on the extraction.
 Each class is mapped to a variable (called its level).  Then for each node,
 we add a constraint that if a node is active, then the level of the class the node
 belongs to must be less than the level of each of the node's children.

 This ensures no cycles can form.

*/
fn block_cycles(model: &mut Model, vars: &IndexMap<ClassId, ClassVars>, egraph: &EGraph) {
    // 给每个 eclass 定义一个 level 列 (整型或实数都可).
    let mut levels: IndexMap<ClassId, Col> = Default::default();
    for c in vars.keys() {
        let var = model.add_col();
        levels.insert(c.clone(), var);
        // 可以给 level 设下/上界, 也可以留空
    }

    // opposite: node-active 与 node-inactive 互斥
    let mut opposite: IndexMap<Col, Col> = Default::default();
    for c in vars.values() {
        for &n in &c.nodes {
            let opposite_col = model.add_binary();
            opposite.insert(n, opposite_col);

            // node_active + opposite_col = 1
            let row = model.add_row();
            model.set_row_equal(row, 1.0);
            model.set_weight(row, opposite_col, 1.0);
            model.set_weight(row, n, 1.0);
        }
    }

    // 添加 level 约束: active => level(parent) < level(child)
    for (class_id, c) in vars {
        for i in 0..c.nodes.len() {
            let n_id = &egraph[class_id].nodes[i];
            let n = &egraph[n_id];
            let var = c.nodes[i];

            let children_classes = n
                .children
                .iter()
                .map(|ch| egraph[ch].eclass.clone())
                .collect::<IndexSet<_>>();

            // 若出现 self-loop, disable
            if children_classes.contains(class_id) {
                // 强行 node_active = 0
                let row = model.add_row();
                model.set_row_equal(row, 0.0);
                model.set_weight(row, var, 1.0);
                continue;
            }

            // block cycles
            for cc in children_classes {
                let row = model.add_row();
                model.set_row_lower(row, 1.0);

                // level(child) - level(parent) >= 1
                model.set_weight(row, *levels.get(&cc).unwrap(), 1.0);
                model.set_weight(row, *levels.get(class_id).unwrap(), -1.0);

                // 如果 node_active=0, 则此约束无效 => + (opposite var)*(large number)
                model.set_weight(row, *opposite.get(&var).unwrap(), (vars.len() + 1) as f64);
            }
        }
    }
}
