use super::*;
use coin_cbc::{Col, Model};
use indexmap::IndexSet;
use ordered_float::NotNan;

pub struct Config {
    pub pull_up_costs: bool,
    pub remove_self_loops: bool,
    pub remove_high_cost_nodes: bool,
    pub remove_more_expensive_subsumed_nodes: bool,
    pub remove_unreachable_classes: bool,
    pub pull_up_single_parent: bool,
    pub take_intersection_of_children_in_class: bool,
    pub move_min_cost_of_members_to_class: bool,
    pub find_extra_roots: bool,
    pub remove_empty_classes: bool,
    pub return_improved_on_timeout: bool,
    pub remove_single_zero_cost: bool,
}

impl Config {
    pub const fn default() -> Self {
        Self {
            pull_up_costs: true,
            remove_self_loops: true,
            remove_high_cost_nodes: true,
            remove_more_expensive_subsumed_nodes: true,
            remove_unreachable_classes: true,
            pull_up_single_parent: true,
            take_intersection_of_children_in_class: true,
            move_min_cost_of_members_to_class: false,
            find_extra_roots: true,
            remove_empty_classes: true,
            return_improved_on_timeout: true,
            remove_single_zero_cost: true,
        }
    }
}

struct NodeILP {
    variable: Col,
    cost: Cost,
    member: NodeId,
    children_classes: IndexSet<ClassId>,
}

struct ClassILP {
    active: Col,
    members: Vec<NodeId>,
    variables: Vec<Col>,
    costs: Vec<Cost>,
    childrens_classes: Vec<IndexSet<ClassId>>,
}

impl ClassILP {
    fn remove(&mut self, idx: usize) {
        self.variables.remove(idx);
        self.costs.remove(idx);
        self.members.remove(idx);
        self.childrens_classes.remove(idx);
    }

    fn remove_node(&mut self, node_id: &NodeId) {
        if let Some(idx) = self.members.iter().position(|n| n == node_id) {
            self.remove(idx);
        }
    }

    fn members(&self) -> usize {
        self.variables.len()
    }

    fn as_nodes(&self) -> Vec<NodeILP> {
        self.variables
            .iter()
            .zip(&self.costs)
            .zip(&self.members)
            .zip(&self.childrens_classes)
            .map(|(((variable, &cost_), member), children_classes)| NodeILP {
                variable: *variable,
                cost: cost_,
                member: member.clone(),
                children_classes: children_classes.clone(),
            })
            .collect()
    }
}

pub struct FasterCbcExtractorWithTimeout<const TIMEOUT_IN_SECONDS: u32>;

impl<const TIMEOUT_IN_SECONDS: u32> Extractor for FasterCbcExtractorWithTimeout<TIMEOUT_IN_SECONDS> {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        return extract(egraph, roots, &Config::default(), TIMEOUT_IN_SECONDS);
    }
}

#[derive(Default)]
pub struct FasterCbcExtractor;

impl Extractor for FasterCbcExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        return extract(egraph, roots, &Config::default(), 30);
    }
}

fn extract(
    egraph: &EGraph,
    roots_slice: &[ClassId],
    config: &Config,
    timeout: u32,
) -> ExtractionResult {
    let mut roots = roots_slice.to_vec();
    roots.sort();
    roots.dedup();

    let mut model = Model::default();
    model.set_parameter("loglevel", "0");
    model.set_parameter("seconds", &timeout.to_string());
    model.set_parameter("allowableGap", "0.2");
    model.set_parameter("maxSolutions", "10");
    model.set_parameter("maxNodes", "10000");

    let mut vars: IndexMap<ClassId, ClassILP> = egraph
        .classes()
        .values()
        .map(|class| {
            let cvars = ClassILP {
                active: model.add_binary(),
                variables: class.nodes.iter().map(|_| model.add_binary()).collect(),
                costs: class.nodes.iter().map(|n| egraph[n].cost).collect(),
                members: class.nodes.clone(),
                childrens_classes: class
                    .nodes
                    .iter()
                    .map(|n| {
                        egraph[n]
                            .children
                            .iter()
                            .map(|c| egraph.nid_to_cid(c).clone())
                            .collect::<IndexSet<ClassId>>()
                    })
                    .collect(),
            };
            (class.id.clone(), cvars)
        })
        .collect();

    let initial_result = super::faster_greedy_dag::FasterGreedyDagExtractor.extract(egraph, &roots);
    let initial_result_cost = initial_result.dag_cost(egraph, &roots);
    let mut result = ExtractionResult::default();

    // Apply optimizations
    remove_with_loops(&mut vars, &roots, config);
    remove_high_cost(&mut vars, initial_result_cost, &roots, config);
    remove_more_expensive_subsumed_nodes(&mut vars, config);
    remove_unreachable_classes(&mut vars, &roots, config);
    pull_up_with_single_parent(&mut vars, &roots, config);
    pull_up_costs(&mut vars, &roots, config);
    remove_single_zero_cost(&mut vars, &mut result, &roots, config);
    find_extra_roots(&mut vars, &mut roots, config);
    remove_empty_classes(&mut vars, config);

    // Add constraints
    for (classid, class) in &vars {
        if class.members() == 0 {
            if roots.contains(classid) {
                return ExtractionResult::default();
            }
            model.set_col_upper(class.active, 0.0);
            continue;
        }

        let row = model.add_row();
        model.set_row_equal(row, 0.0);
        model.set_weight(row, class.active, -1.0);
        for &node_active in &class.variables {
            model.set_weight(row, node_active, 1.0);
        }

        for (childrens_classes, &node_active) in class.childrens_classes.iter().zip(&class.variables) {
            for child_class in childrens_classes {
                let child_active = vars[child_class].active;
                let row = model.add_row();
                model.set_row_upper(row, 0.0);
                model.set_weight(row, node_active, 1.0);
                model.set_weight(row, child_active, -1.0);
            }
        }
    }

    for root in &roots {
        model.set_col_lower(vars[root].active, 1.0);
    }

    loop {
        if let Ok(difference) = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
        {
            let seconds = timeout.saturating_sub(difference.as_secs().try_into().unwrap());
            model.set_parameter("seconds", &seconds.to_string());
        }

        let solution = model.solve();
        log::info!(
            "CBC status {:?}, {:?}, obj = {}",
            solution.raw().status(),
            solution.raw().secondary_status(),
            solution.raw().obj_value(),
        );

        if solution.raw().is_proven_infeasible() {
            return ExtractionResult::default();
        }

        let stopped_without_finishing = solution.raw().status() != coin_cbc::raw::Status::Finished;

        if stopped_without_finishing {
            if !config.return_improved_on_timeout
                || solution.raw().obj_value() > initial_result_cost.into_inner()
            {
                return initial_result;
            }
        }

        for (id, var) in &vars {
            let active = solution.col(var.active) > 0.0;
            if active {
                let node_idx = var
                    .variables
                    .iter()
                    .position(|&n| solution.col(n) > 0.0)
                    .unwrap();
                let node_id = var.members[node_idx].clone();
                result.choose(id.clone(), node_id);
            }
        }

        let cycles = find_cycles_in_result(&result, &vars, &roots);
        if cycles.is_empty() {
            return result;
        }

        for cycle in &cycles {
            block_cycle(&mut model, cycle, &vars);
        }
    }
}

fn block_cycle(model: &mut Model, cycle: &Vec<ClassId>, vars: &IndexMap<ClassId, ClassILP>) {
    if cycle.is_empty() {
        return;
    }
    let mut blocking = Vec::new();
    for i in 0..cycle.len() {
        let current_class_id = &cycle[i];
        let next_class_id = &cycle[(i + 1) % cycle.len()];

        let mut this_level = Vec::default();
        for node in &vars[current_class_id].as_nodes() {
            if node.children_classes.contains(next_class_id) {
                this_level.push(node.variable);
            }
        }

        if !this_level.is_empty() {
            let blocking_var = if this_level.len() == 1 {
                this_level[0]
            } else {
                let var = model.add_binary();
                for n in this_level {
                    let row = model.add_row();
                    model.set_row_upper(row, 0.0);
                    model.set_weight(row, n, 1.0);
                    model.set_weight(row, var, -1.0);
                }
                var
            };
            blocking.push(blocking_var);
        }
    }

    if !blocking.is_empty() {
        let row = model.add_row();
        model.set_row_upper(row, blocking.len() as f64 - 1.0);
        for b in blocking {
            model.set_weight(row, b, 1.0)
        }
    }
}

fn find_cycles_in_result(
    extraction_result: &ExtractionResult,
    vars: &IndexMap<ClassId, ClassILP>,
    roots: &[ClassId],
) -> Vec<Vec<ClassId>> {
    let mut status = IndexMap::<ClassId, TraverseStatus>::default();
    let mut cycles = vec![];
    for root in roots {
        let mut stack = vec![];
        cycle_dfs(
            extraction_result,
            vars,
            root,
            &mut status,
            &mut cycles,
            &mut stack,
        )
    }
    cycles
}

#[derive(Clone)]
enum TraverseStatus {
    Doing,
    Done,
}

fn cycle_dfs(
    extraction_result: &ExtractionResult,
    vars: &IndexMap<ClassId, ClassILP>,
    class_id: &ClassId,
    status: &mut IndexMap<ClassId, TraverseStatus>,
    cycles: &mut Vec<Vec<ClassId>>,
    stack: &mut Vec<ClassId>,
) {
    match status.get(class_id).cloned() {
        Some(TraverseStatus::Done) => (),
        Some(TraverseStatus::Doing) => {
            if let Some(pos) = stack.iter().position(|id| id == class_id) {
                let mut cycle = vec![];
                cycle.extend_from_slice(&stack[pos..]);
                cycles.push(cycle);
            }
        }
        None => {
            status.insert(class_id.clone(), TraverseStatus::Doing);
            stack.push(class_id.clone());
            let node_id = &extraction_result.choices[class_id];
            for child_cid in vars[class_id]
                .childrens_classes
                .iter()
                .find(|_| true)
                .unwrap_or(&IndexSet::new())
            {
                cycle_dfs(extraction_result, vars, child_cid, status, cycles, stack)
            }
            stack.pop();
            status.insert(class_id.clone(), TraverseStatus::Done);
        }
    }
}

// Helper functions for optimization
fn remove_with_loops(vars: &mut IndexMap<ClassId, ClassILP>, roots: &[ClassId], config: &Config) {
    if config.remove_self_loops {
        let mut removed = 0;
        for (class_id, class_details) in vars.iter_mut() {
            for i in (0..class_details.childrens_classes.len()).rev() {
                if class_details.childrens_classes[i]
                    .iter()
                    .any(|cid| *cid == *class_id || (roots.len() == 1 && roots[0] == *cid))
                {
                    class_details.remove(i);
                    removed += 1;
                }
            }
        }
        log::info!("Omitted looping nodes: {}", removed);
    }
}

fn remove_high_cost(
    vars: &mut IndexMap<ClassId, ClassILP>,
    initial_result_cost: Cost,
    roots: &[ClassId],
    config: &Config,
) {
    if config.remove_high_cost_nodes {
        let lowest_root_cost_sum: Cost = roots
            .iter()
            .filter_map(|root| vars[root].costs.iter().min())
            .sum();

        let mut removed = 0;
        for (class_id, class_details) in vars.iter_mut() {
            for i in (0..class_details.costs.len()).rev() {
                let cost = &class_details.costs[i];
                let this_root: Cost = if roots.contains(class_id) {
                    *class_details.costs.iter().min().unwrap()
                } else {
                    Cost::default()
                };

                if cost > &(initial_result_cost - lowest_root_cost_sum + this_root + EPSILON_ALLOWANCE)
                {
                    class_details.remove(i);
                    removed += 1;
                }
            }
        }
        log::info!("Removed high-cost nodes: {}", removed);
    }
}

fn remove_more_expensive_subsumed_nodes(vars: &mut IndexMap<ClassId, ClassILP>, config: &Config) {
    if config.remove_more_expensive_subsumed_nodes {
        let mut removed = 0;
        for class in vars.values_mut() {
            let mut children = class.as_nodes();
            children.sort_by_key(|e| (e.children_classes.len(), e.cost));

            let mut i = 0;
            while i < children.len() {
                for j in ((i + 1)..children.len()).rev() {
                    let node_b = &children[j];
                    if children[i].cost <= node_b.cost
                        && children[i]
                            .children_classes
                            .is_subset(&node_b.children_classes)
                    {
                        class.remove_node(&node_b.member.clone());
                        children.remove(j);
                        removed += 1;
                    }
                }
                i += 1;
            }
        }
        log::info!("Removed more expensive subsumed nodes: {}", removed);
    }
}

fn remove_unreachable_classes(
    vars: &mut IndexMap<ClassId, ClassILP>,
    roots: &[ClassId],
    config: &Config,
) {
    if config.remove_unreachable_classes {
        let mut reachable_classes: IndexSet<ClassId> = IndexSet::default();
        reachable(&*vars, roots, &mut reachable_classes);
        let initial_size = vars.len();
        vars.retain(|class_id, _| reachable_classes.contains(class_id));
        log::info!("Unreachable classes: {}", initial_size - vars.len());
    }
}

fn reachable(
    vars: &IndexMap<ClassId, ClassILP>,
    classes: &[ClassId],
    is_reachable: &mut IndexSet<ClassId>,
) {
    for class in classes {
        if is_reachable.insert(class.clone()) {
            if let Some(class_vars) = vars.get(class) {
                for kids in &class_vars.childrens_classes {
                    for child_class in kids {
                        reachable(vars, &[child_class.clone()], is_reachable);
                    }
                }
            }
        }
    }
}

fn pull_up_costs(vars: &mut IndexMap<ClassId, ClassILP>, roots: &[ClassId], config: &Config) {
    if config.pull_up_costs {
        let mut count = 0;
        let mut changed = true;
        let child_to_parent = classes_with_single_parent(&*vars);

        while count < 10 && changed {
            changed = false;
            count += 1;
            for (child, parent) in &child_to_parent {
                if child == parent || roots.contains(child) || vars[child].members() == 0 {
                    continue;
                }

                let min_cost = vars[child]
                    .costs
                    .iter()
                    .min()
                    .unwrap_or(&Cost::default())
                    .into_inner();

                if min_cost == 0.0 {
                    continue;
                }
                changed = true;

                for c in &mut vars[child].costs {
                    *c -= min_cost;
                }

                let indices: Vec<_> = vars[parent]
                    .childrens_classes
                    .iter()
                    .enumerate()
                    .filter(|&(_, c)| c.contains(child))
                    .map(|(id, _)| id)
                    .collect();

                for id in indices {
                    vars[parent].costs[id] += min_cost;
                }
            }
        }
    }
}

fn pull_up_with_single_parent(
    vars: &mut IndexMap<ClassId, ClassILP>,
    roots: &[ClassId],
    config: &Config,
) {
    if config.pull_up_single_parent {
        for _ in 0..10 {
            let child_to_parent = classes_with_single_parent(&*vars);
            let mut pull_up_count = 0;

            for (child, parent) in &child_to_parent {
                if child == parent
                    || roots.contains(child)
                    || vars[child].members.len() != 1
                    || vars[child].childrens_classes[0].is_empty()
                {
                    continue;
                }

                let found = vars[parent]
                    .childrens_classes
                    .iter()
                    .filter(|c| c.contains(child))
                    .count();

                if found != 1 {
                    continue;
                }

                let idx = vars[parent]
                    .childrens_classes
                    .iter()
                    .position(|e| e.contains(child))
                    .unwrap();

                let child_descendants = vars[child].childrens_classes[0].clone();
                let parent_descendants = &mut vars[parent].childrens_classes[idx];

                for e in &child_descendants {
                    parent_descendants.insert(e.clone());
                }

                vars[child].childrens_classes[0].clear();
                pull_up_count += 1;
            }

            if pull_up_count == 0 {
                break;
            }
            log::info!("Pull up count: {pull_up_count}");
        }
    }
}

fn remove_single_zero_cost(
    vars: &mut IndexMap<ClassId, ClassILP>,
    extraction_result: &mut ExtractionResult,
    roots: &[ClassId],
    config: &Config,
) {
    if config.remove_single_zero_cost {
        let mut zero = IndexSet::new();
        for (class_id, details) in &*vars {
            if details.childrens_classes.len() == 1
                && details.childrens_classes[0].is_empty()
                && details.costs[0] == 0.0
                && !roots.contains(class_id)
            {
                zero.insert(class_id.clone());
            }
        }

        if !zero.is_empty() {
            let child_to_parents = child_to_parents(&vars);
            let mut removed = 0;

            for e in &zero {
                if let Some(parents) = child_to_parents.get(e) {
                    for parent in parents {
                        for i in (0..vars[parent].childrens_classes.len()).rev() {
                            if vars[parent].childrens_classes[i].contains(e) {
                                vars[parent].childrens_classes[i].remove(e);
                                removed += 1;
                            }
                        }
                    }
                }
                extraction_result.choose(e.clone(), vars[e].members[0].clone());
            }

            vars.retain(|class_id, _| !zero.contains(class_id));
            log::info!("Zero cost & zero children removed: {}, links removed: {}", zero.len(), removed);
        }
    }
}

fn find_extra_roots(
    vars: &mut IndexMap<ClassId, ClassILP>,
    roots: &mut Vec<ClassId>,
    config: &Config,
) {
    if config.find_extra_roots {
        let mut i = 0;
        let mut extra = 0;
        while i < roots.len() {
            let r = &roots[i];
            let details = &vars[r];
            
            if !details.childrens_classes.is_empty() {
                let mut intersection = details.childrens_classes[0].clone();
                for other_children in &details.childrens_classes[1..] {
                    intersection.retain(|x| other_children.contains(x));
                }

                for new_root in &intersection {
                    if !roots.contains(new_root) {
                        roots.push(new_root.clone());
                        extra += 1;
                    }
                }
            }
            i += 1;
        }
        log::info!("Extra roots discovered: {extra}");
    }
}

fn remove_empty_classes(vars: &mut IndexMap<ClassId, ClassILP>, config: &Config) {
    if config.remove_empty_classes {
        let mut empty_classes = std::collections::VecDeque::new();
        for (class_id, detail) in vars.iter() {
            if detail.members() == 0 {
                empty_classes.push_back(class_id.clone());
            }
        }

        let mut removed = 0;
        let child_to_parents = child_to_parents(&vars);
        let mut done = IndexSet::new();

        while let Some(e) = empty_classes.pop_front() {
            if !done.insert(e.clone()) {
                continue;
            }

            if let Some(parents) = child_to_parents.get(&e) {
                for parent in parents {
                    for i in (0..vars[parent].childrens_classes.len()).rev() {
                        if vars[parent].childrens_classes[i].contains(&e) {
                            vars[parent].remove(i);
                            removed += 1;
                        }
                    }

                    if vars[parent].members() == 0 {
                        empty_classes.push_back(parent.clone());
                    }
                }
            }
        }
        log::info!("Nodes removed that point to empty classes: {}", removed);
    }
}

fn child_to_parents(vars: &IndexMap<ClassId, ClassILP>) -> IndexMap<ClassId, IndexSet<ClassId>> {
    let mut result = IndexMap::new();
    for (class_id, class_vars) in vars {
        for kids in &class_vars.childrens_classes {
            for child_class in kids {
                result
                    .entry(child_class.clone())
                    .or_insert_with(IndexSet::new)
                    .insert(class_id.clone());
            }
        }
    }
    result
}

fn classes_with_single_parent(vars: &IndexMap<ClassId, ClassILP>) -> IndexMap<ClassId, ClassId> {
    let mut child_to_parents = IndexMap::new();
    for (class_id, class_vars) in vars {
        for kids in &class_vars.childrens_classes {
            for child_class in kids {
                child_to_parents
                    .entry(child_class.clone())
                    .or_insert_with(IndexSet::new)
                    .insert(class_id.clone());
            }
        }
    }

    child_to_parents
        .into_iter()
        .filter_map(|(child, parents)| {
            if parents.len() == 1 {
                Some((child, parents.into_iter().next().unwrap()))
            } else {
                None
            }
        })
        .collect()
}