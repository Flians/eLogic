//! Bounded multi-realization extraction independent of leaf arrivals.
//! Profiles carry physical DAGs, not e-class references to mutable single winners.
use super::*;
use std::collections::{BTreeSet, HashMap};

pub(crate) fn narrow_enabled() -> bool {
    !std::env::var("MAP_EGG_PROFILE_NARROW").is_ok_and(|v| v == "0")
}
pub(crate) fn phase_keep_enabled() -> bool {
    std::env::var("MAP_EGG_PROFILE_PHASE_KEEP").is_ok_and(|v| v == "1")
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct Limits {
    pub frontier: usize,
    pub combinations: usize,
    pub visits: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            frontier: 8,
            combinations: 20000,
            visits: 4000,
        }
    }
}
#[derive(Clone)]
struct Realization {
    cover: MappedCoverChoice,
    dependencies: BTreeSet<usize>,
    fingerprint: String,
    // Potential savings when the opposite phase of each cut leaf is already physical.
    // Stored only in extraction state: cached covers remain context independent.
    phase_credit: Vec<f32>,
}
fn profile(c: &MappedCoverChoice, i: usize) -> f32 {
    if c.used_leaf_indices.contains(&i) {
        c.pin_to_root_delay[i]
    } else {
        f32::NEG_INFINITY
    }
}
fn dominates(a: &MappedCoverChoice, b: &MappedCoverChoice) -> bool {
    a.area <= b.area && (0..a.pin_to_root_delay.len()).all(|i| profile(a, i) <= profile(b, i))
}
fn order(a: &Realization, b: &Realization) -> std::cmp::Ordering {
    a.cover
        .area
        .total_cmp(&b.cover.area)
        .then(a.cover.delay.total_cmp(&b.cover.delay))
        .then(a.fingerprint.cmp(&b.fingerprint))
}
fn retain(front: &mut Vec<Realization>, c: Realization, cap: usize, root: bool) -> bool {
    if front.iter().any(|x| x.fingerprint == c.fingerprint) {
        return false;
    }
    // Intermediate structures with distinct sharing footprints must survive even if
    // standalone dominated: another parent can reuse their cells. Caps are heuristic.
    let comparable = |a: &Realization, b: &Realization| {
        !phase_keep_enabled() || a.phase_credit == b.phase_credit
    };
    if root && front.iter().any(|x| comparable(x, &c) && dominates(&x.cover, &c.cover)) {
        return false;
    }
    let previous: Vec<String> = front.iter().map(|x| x.fingerprint.clone()).collect();
    if root {
        front.retain(|x| !comparable(&c, x) || !dominates(&c.cover, &x.cover));
    }
    front.push(c);
    front.sort_by(order);
    if front.len() > cap {
        let mut keep = BTreeSet::new();
        keep.insert(0);
        if phase_keep_enabled() && cap > 1 {
            // Keep the optimistic boundary-reuse winner alongside the standalone area anchor.
            let optimistic = |c: &Realization| c.cover.area-c.phase_credit.iter().sum::<f32>();
            let k=(0..front.len()).min_by(|&a,&b| optimistic(&front[a]).total_cmp(&optimistic(&front[b]))
                .then(order(&front[a],&front[b]))).unwrap();
            keep.insert(k);
        }
        let best_delay = (0..front.len())
            .min_by(|&a, &b| {
                front[a]
                    .cover
                    .delay
                    .total_cmp(&front[b].cover.delay)
                    .then(a.cmp(&b))
            })
            .unwrap();
        if keep.len() < cap {
            keep.insert(best_delay);
        }
        for i in 0..front[0].cover.pin_to_root_delay.len() {
            if phase_keep_enabled() && keep.len() < cap {
                let k=(0..front.len()).min_by(|&a,&b| {
                    let score=|c: &Realization| c.cover.area-c.phase_credit[i];
                    score(&front[a]).total_cmp(&score(&front[b])).then(order(&front[a],&front[b]))
                }).unwrap();
                keep.insert(k);
            }
            let k = (0..front.len())
                .min_by(|&a, &b| {
                    profile(&front[a].cover, i)
                        .total_cmp(&profile(&front[b].cover, i))
                        .then(order(&front[a], &front[b]))
                })
                .unwrap();
            if keep.len() < cap {
                keep.insert(k);
            }
        }
        for k in 0..front.len() {
            if keep.len() < cap {
                keep.insert(k);
            }
        }
        *front = front
            .drain(..)
            .enumerate()
            .filter_map(|(i, c)| keep.contains(&i).then_some(c))
            .collect();
    }
    previous
        != front
            .iter()
            .map(|x| x.fingerprint.clone())
            .collect::<Vec<_>>()
}
fn realization(
    graph: MappedCoverGraph,
    dependencies: BTreeSet<usize>,
    tech: &TechLib,
    n: usize,
) -> Option<Realization> {
    let mut used = Vec::new();
    let mut area = 0.;
    for node in &graph.nodes {
        match node {
            MappedCoverNode::Input(i) => used.push(*i),
            MappedCoverNode::Const(_) => (),
            MappedCoverNode::Cell { cell, .. } => {
                area += cell.area
                    + tech.inv_area
                        * (cell.neg_mask.count_ones() as f32 + u32::from(cell.out_neg) as f32)
            }
        }
    }
    used.sort_unstable();
    used.dedup();
    let delays = compute_pin_to_root_delays(&graph, tech, n);
    if !area.is_finite() || delays.iter().any(|d| !d.is_finite()) {
        return None;
    }
    let delay = used.iter().map(|&i| delays[i]).fold(0., f32::max);
    let fingerprint = serde_json::to_string(&graph).ok()?;
    let mut phase_credit=vec![0.;n];
    for node in &graph.nodes {
        if let MappedCoverNode::Cell {cell,fanins} = node {
            // A negative input edge, or an explicit physical INV of a boundary leaf,
            // can reuse that leaf's inverter input. Internal sharing remains heuristic.
            for (k,&fanin) in fanins.iter().enumerate() {
                if let MappedCoverNode::Input(i)=&graph.nodes[fanin] {
                    if (cell.neg_mask>>k)&1 != 0 { phase_credit[*i]+=tech.inv_area; }
                }
            }
            if fanins.len()==1 && tech.inv_variant.as_ref().is_some_and(|inv| inv.cell_name==cell.cell_name)
                && cell.neg_mask==0 && !cell.out_neg {
                if let MappedCoverNode::Input(i)=&graph.nodes[fanins[0]] { phase_credit[*i]+=cell.area; }
            }
        }
    }
    Some(Realization {
        cover: MappedCoverChoice {
            graph,
            area,
            delay,
            used_leaf_indices: used,
            pin_to_root_delay: delays,
        },
        dependencies,
        fingerprint,
        phase_credit,
    })
}
fn compose(
    children: &[Realization],
    cell: &CellVariant,
    key: usize,
    tech: &TechLib,
    n: usize,
) -> Option<Realization> {
    if children.iter().any(|c| c.dependencies.contains(&key)) {
        return None;
    }
    let mut nodes = Vec::new();
    let mut intern = HashMap::<String, usize>::new();
    let mut roots = Vec::new();
    let mut dependencies = BTreeSet::new();
    for child in children {
        dependencies.extend(child.dependencies.iter().copied());
        let mut ids = Vec::new();
        for node in &child.cover.graph.nodes {
            let mapped = match node {
                MappedCoverNode::Cell { cell, fanins } => MappedCoverNode::Cell {
                    cell: cell.clone(),
                    fanins: fanins.iter().map(|&i| ids[i]).collect(),
                },
                _ => node.clone(),
            };
            let symbol = serde_json::to_string(&mapped).ok()?;
            let id = if let Some(&i) = intern.get(&symbol) {
                i
            } else {
                let i = nodes.len();
                nodes.push(mapped);
                intern.insert(symbol, i);
                i
            };
            ids.push(id);
        }
        roots.push(ids[child.cover.graph.root]);
    }
    let root_node = MappedCoverNode::Cell {
        cell: cell.clone(),
        fanins: roots,
    };
    let symbol = serde_json::to_string(&root_node).ok()?;
    let root = if let Some(&i) = intern.get(&symbol) {
        i
    } else {
        let i = nodes.len();
        nodes.push(root_node);
        i
    };
    dependencies.insert(key);
    realization(MappedCoverGraph { nodes, root }, dependencies, tech, n)
}
fn combinations(
    lists: &[Vec<Realization>],
    remaining: &mut usize,
    callback: &mut impl FnMut(&[Realization]),
) {
    fn rec(
        lists: &[Vec<Realization>],
        i: usize,
        selected: &mut Vec<Realization>,
        remaining: &mut usize,
        cb: &mut impl FnMut(&[Realization]),
    ) {
        if *remaining == 0 {
            return;
        }
        if i == lists.len() {
            *remaining -= 1;
            cb(selected);
            return;
        }
        for c in &lists[i] {
            selected.push(c.clone());
            rec(lists, i + 1, selected, remaining, cb);
            selected.pop();
            if *remaining == 0 {
                break;
            }
        }
    }
    rec(lists, 0, &mut Vec::new(), remaining, callback);
}

fn relax_fronts(
    dag: &CoverDag,
    variants: &HashMap<String, Vec<CellVariant>>,
    tech: &TechLib,
    n: usize,
    fronts: &mut HashMap<usize, Vec<Realization>>,
    remaining: &mut usize,
    visits: &mut usize,
    visit_limit: usize,
    frontier: usize,
) {
    let mut queue = UniqueQueue::<usize>::default();
    let mut keys: Vec<usize> = dag.class_nodes.keys().copied().collect();
    keys.sort_unstable();
    queue.extend(keys);
    while let Some(key) = queue.pop() {
        *visits += 1;
        if *visits >= visit_limit || *remaining == 0 {
            break;
        }
        let mut proposed = Vec::new();
        for &node_id in &dag.class_nodes[&key] {
            let node = &dag.nodes[node_id];
            let terminal = match &node.op {
                CoverDagNodeOp::Input(i) => Some(MappedCoverNode::Input(*i)),
                CoverDagNodeOp::Const(v) => Some(MappedCoverNode::Const(*v)),
                _ => None,
            };
            if let Some(node) = terminal {
                if let Some(c) = realization(
                    MappedCoverGraph {
                        nodes: vec![node],
                        root: 0,
                    },
                    BTreeSet::from([key]),
                    tech,
                    n,
                ) {
                    proposed.push(c);
                }
                continue;
            }
            let lists: Vec<Vec<Realization>> = node
                .children
                .iter()
                .map(|id| fronts.get(id).cloned().unwrap_or_default())
                .collect();
            if lists.iter().any(Vec::is_empty) {
                continue;
            }
            let cell_variants = match &node.op {
                CoverDagNodeOp::Not => tech.exact_inv_variant().cloned().into_iter().collect(),
                CoverDagNodeOp::And => tech.exact_and2_variant().cloned().into_iter().collect(),
                CoverDagNodeOp::Gate(s) => variants.get(s).cloned().unwrap_or_default(),
                _ => Vec::new(),
            };
            // Each variant consumes a budget unit. Enumerate across child realizations,
            // not only root alternatives. Retain current front first for stable iteration.
            for cell in &cell_variants {
                combinations(&lists, remaining, &mut |children| {
                    if let Some(c) = compose(children, cell, key, tech, n) {
                        retain(&mut proposed, c, frontier.max(1), false);
                    }
                });
                if *remaining == 0 {
                    break;
                }
            }
            if *remaining == 0 {
                break;
            }
        }
        let front = fronts.entry(key).or_default();
        let mut changed = false;
        for c in proposed {
            changed |= retain(front, c, frontier.max(1), false);
        }
        if changed {
            let mut parents = dag.parents[&key].clone();
            parents.sort_unstable();
            queue.extend(parents);
        }
    }
}

pub(crate) fn extract(
    expr: &str,
    tech: &TechLib,
    n: usize,
    limits: Limits,
) -> Result<Vec<MappedCoverChoice>, String> {
    let ps = cut_mapping_params();
    let rec: egg::RecExpr<AIG> = expr.parse().map_err(|_| "profile_expr_parse")?;
    if rec.as_ref().len() > ps.max_cover_nodes {
        return Err("profile_expr_size".into());
    }
    let bundle = cut_egg_bundle(tech);
    let (egraph, root, cells) = rewrite_to_gategraph_with_limits(
        expr,
        tech,
        n,
        Some(bundle.as_ref()),
        ps.egg_iter_limit,
        ps.egg_node_limit,
    )
    .ok_or("profile_rewrite_failed")?;
    let dag = build_cover_dag(&egraph, root);
    // Build only the pin variants of cells appearing in this bounded e-graph.
    // Disk hits require neither saturation nor the eager whole-library NP index.
    static VARIANTS: std::sync::OnceLock<std::sync::Mutex<HashMap<String, Vec<CellVariant>>>> =
        std::sync::OnceLock::new();
    let cache = VARIANTS.get_or_init(Default::default);
    let mut variants = HashMap::<String, Vec<CellVariant>>::new();
    for node in &dag.nodes {
        if let CoverDagNodeOp::Gate(name) = &node.op {
            if variants.contains_key(name) {
                continue;
            }
            let cell = &cells[name];
            let key = serde_json::to_string(cell).unwrap();
            let mut stored = cache.lock().unwrap();
            let list = stored
                .entry(key)
                .or_insert_with(|| tech.matching_cell_variants(cell));
            variants.insert(name.clone(), list.clone());
        }
    }
    let mut fronts = HashMap::<usize, Vec<Realization>>::new();
    let mut remaining = limits.combinations;
    let mut visits = 0;
    // Establish area anchors before Cartesian products of wider child fronts
    // consume the global budget. Reuse every anchor as a warm start for the
    // wider pass; both passes together obey the original work limits.
    if narrow_enabled() && limits.frontier > 1 && remaining >= 4 {
        let allowance = remaining / 4;
        let mut narrow_remaining = allowance;
        relax_fronts(&dag, &variants, tech, n, &mut fronts, &mut narrow_remaining,
            &mut visits, limits.visits / 2, 1);
        remaining -= allowance - narrow_remaining;
    }
    relax_fronts(&dag, &variants, tech, n, &mut fronts, &mut remaining,
        &mut visits, limits.visits, limits.frontier);
    let mut root_front = Vec::new();
    if std::env::var_os("MAP_EGG_PROFILE_EXTRACT_TRACE").is_some() {
        eprintln!("profile_extract root={} visits={} remaining={} root_nodes={:?} root_front={:?}",
            dag.root_class, visits, remaining, dag.class_nodes[&dag.root_class].iter().map(|&i| &dag.nodes[i]).collect::<Vec<_>>(),
            fronts.get(&dag.root_class).map(|f| f.iter().map(|x| (x.cover.area,x.cover.delay)).collect::<Vec<_>>()));
    }
    for c in fronts.remove(&dag.root_class).unwrap_or_default() {
        retain(&mut root_front, c, limits.frontier.max(1), true);
    }
    if root_front.is_empty() {
        return Err("profile_root_missing".into());
    }
    Ok(root_front.into_iter().map(|c| c.cover).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn boundary_phase_preserves_standalone_dominated_area_winner() {
        let path=std::env::temp_dir().join(format!("phase-retention-{}-{}.genlib",
            std::process::id(),crate::profile_cache::test_serial()));
        std::fs::write(&path,"GATE INV 0.1 Y=!A;\nPIN A INV 1 999 1 0 1 0\nGATE AND 0.2 Y=A*B;\nPIN A NONINV 1 999 2 0 2 0\nPIN B NONINV 1 999 2 0 2 0\nGATE ANDN 0.25 Y=!A*B;\nPIN A INV 1 999 0.5 0 0.5 0\nPIN B NONINV 1 999 0.5 0 0.5 0\n").unwrap();
        let tech=TechLib::from_genlib_unindexed(&path,6).unwrap();std::fs::remove_file(path).unwrap();
        let pool=extract("(& (~ a) b)",&tech,2,Limits::default()).unwrap();
        let best=pool.iter().map(|c| {
            let r=realization(c.graph.clone(),BTreeSet::new(),&tech,2).unwrap();
            c.area-r.phase_credit[0]
        }).min_by(f32::total_cmp).unwrap();
        // With physical a=INV(x), AND(x,b) costs .20; the standalone .25 ANDN
        // dominates its .30 INV+AND representation unless phase demand is preserved.
        if phase_keep_enabled() { assert!(best<=0.20001,"{best}"); }
        else { assert!(best>=0.24999,"{best}"); }
    }
    #[test]
    fn progressive_search_keeps_area_anchor_under_cartesian_budget() {
        let path = std::env::temp_dir().join(format!("profile-budget-{}-{}.genlib",
            std::process::id(), crate::profile_cache::test_serial()));
        std::fs::write(&path, include_str!("../../tests/data/profile_budget.genlib")).unwrap();
        let tech = TechLib::from_genlib_unindexed(&path, 9).unwrap();
        std::fs::remove_file(path).unwrap();
        let expr = "(AOI21x1_ASAP7_75t_R (NAND2x1_ASAP7_75t_R a c) (AO21x1_ASAP7_75t_R b c e) d)";
        let aig = crate::emap_egg::gate_expr_to_aig_sexp(expr, &tech).unwrap();
        let pool = extract(&aig, &tech, 5, Limits::default()).unwrap();
        // Wide Cartesian expansion used to exhaust 20,000 combinations before
        // propagating this area-saving, slower AO32 implementation to the root.
        assert!(pool.iter().any(|c| c.area <= 0.29001), "{:?}",
            pool.iter().map(|c| (c.area,c.delay)).collect::<Vec<_>>());
        assert!(pool.iter().any(|c| c.delay < 50.));
    }
    /// Replay an extracted cut without changing the synthesis or mapping input.
    /// Fixture: {"genlib": path, "expr": AIG sexp, "arrivals": [...], "required": number}.
    #[test]
    #[ignore = "manual cut-level diagnosis; requires MAP_EGG_PROFILE_FIXTURE"]
    fn replay_profile_fixture() {
        let path = std::env::var("MAP_EGG_PROFILE_FIXTURE").unwrap();
        let f: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let tech = TechLib::from_genlib(f["genlib"].as_str().unwrap(), 9).unwrap();
        let arrivals: Vec<f32> = serde_json::from_value(f["arrivals"].clone()).unwrap();
        let expr = f["expr"].as_str().unwrap();
        let aig = crate::emap_egg::gate_expr_to_aig_sexp(expr, &tech).unwrap();
        let old = map_cut_to_cover_cover_only(&aig, &tech, &arrivals, crate::EmapObjective::AreaFirst,
            f["required"].as_f64().map(|x| x as f32));
        let mut limits = Limits::default();
        if let Some(k) = f["frontier"].as_u64() { limits.frontier = k as usize; }
        if let Some(k) = f["combinations"].as_u64() { limits.combinations = k as usize; }
        let new = extract(&aig, &tech, arrivals.len(), limits);
        let mut sweep = Vec::new();
        if f["sweep"].as_bool() == Some(true) {
            for (frontier,combinations) in [(1,20000),(2,20000),(4,20000),(16,20000),(32,20000),(8,100000),(8,1000000)] {
                let profiles = extract(&aig,&tech,arrivals.len(),Limits{frontier,combinations,..limits});
                sweep.push(serde_json::json!({"frontier":frontier,"combinations":combinations,"profiles":profiles}));
            }
        }
        println!("{}", serde_json::json!({"aig": aig, "legacy": old, "profiles": new,"sweep":sweep}));
    }
    fn library() -> TechLib {
        let p = std::env::temp_dir().join(format!(
            "profile-lib-{}-{}.genlib",
            std::process::id(),
            super::super::super::profile_cache::test_serial()
        ));
        std::fs::write(&p,"GATE inv 0.1 Y=!A;\nPIN A INV 1 999 1 0 1 0\nGATE x 1 Y=A*B;\nPIN A NONINV 1 999 10 0 10 0\nPIN B NONINV 1 999 100 0 100 0\nGATE y 1 Y=A*B;\nPIN A NONINV 1 999 90 0 90 0\nPIN B NONINV 1 999 20 0 20 0\n").unwrap();
        let lib = TechLib::from_genlib(&p, 6).unwrap();
        std::fs::remove_file(p).unwrap();
        lib
    }
    #[test]
    fn vector_frontier_preserves_late_input_winner() {
        let tech = library();
        let f = extract("(& a b)", &tech, 2, Limits::default()).unwrap();
        assert!(f.iter().any(|c| c.pin_to_root_delay == vec![10., 100.]));
        let actual = |c: &MappedCoverChoice| {
            c.used_leaf_indices
                .iter()
                .map(|&i| [100., 0.][i] + c.pin_to_root_delay[i])
                .fold(0., f32::max)
        };
        assert_eq!(f.iter().map(actual).min_by(f32::total_cmp), Some(110.));
        assert_eq!(f.iter().map(|c| c.delay).min_by(f32::total_cmp), Some(90.));
    }
    #[test]
    fn child_fronts_reach_root_and_dags_are_acyclic() {
        let tech = library();
        let f = extract(
            "(& (& a b) c)",
            &tech,
            3,
            Limits {
                frontier: 24,
                ..Limits::default()
            },
        )
        .unwrap();
        assert!(f.len() > 1);
        for c in f {
            assert_eq!(c.used_leaf_indices, vec![0, 1, 2]);
            for (i, node) in c.graph.nodes.iter().enumerate() {
                if let MappedCoverNode::Cell { fanins, .. } = node {
                    assert!(fanins.iter().all(|&j| j < i));
                }
            }
        }
    }
    #[test]
    fn unused_inputs_and_repeated_fanins_are_costed_once() {
        let tech = library();
        let input = realization(
            MappedCoverGraph {
                nodes: vec![MappedCoverNode::Input(0)],
                root: 0,
            },
            BTreeSet::from([0]),
            &tech,
            3,
        )
        .unwrap();
        let cell = tech.exact_and2_variant().unwrap();
        let c = compose(&[input.clone(), input], cell, 1, &tech, 3).unwrap();
        assert_eq!(c.cover.graph.nodes.len(), 2);
        assert_eq!(c.cover.used_leaf_indices, vec![0]);
        assert_eq!(c.cover.area, cell.area);
        assert_eq!(c.cover.pin_to_root_delay[1..], [0., 0.]);
        let parent = compose(&[c.clone(), c], cell, 2, &tech, 3).unwrap();
        assert_eq!(parent.cover.graph.nodes.len(), 3);
        assert_eq!(parent.cover.area, 2. * cell.area);
    }
    #[test]
    fn physical_profiles_have_the_original_truth_table() {
        let tech = library();
        for expr in ["(& a b)", "(~ (& a b))", "(& (& a b) c)", "(& a (~ b))"] {
            let (n, expected) = sexp_to_tt9(expr).unwrap();
            for c in extract(expr, &tech, n as usize, Limits::default()).unwrap() {
                for assignment in 0..(1usize << n) {
                    let mut values = Vec::<bool>::new();
                    for node in &c.graph.nodes {
                        let value = match node {
                            MappedCoverNode::Input(i) => (assignment >> i) & 1 == 1,
                            MappedCoverNode::Const(v) => *v,
                            MappedCoverNode::Cell { cell, fanins } => {
                                let bits = fanins
                                    .iter()
                                    .enumerate()
                                    .fold(0, |a, (i, &j)| a | ((values[j] as usize) << i));
                                let tt = tech.variant_tt_wiring(cell).unwrap();
                                crate::tt9::tt9_bit_get(&tt, bits)
                            }
                        };
                        values.push(value);
                    }
                    assert_eq!(
                        values[c.graph.root],
                        crate::tt9::tt9_bit_get(&expected, assignment),
                        "{expr}"
                    );
                }
            }
        }
    }
    #[test]
    fn lazy_variants_match_eager_index_including_pin_order() {
        let tech = library();
        for cell in &tech.cells {
            let expected = tech
                .candidates(cell.vars.len() as u8, cell.tt)
                .unwrap_or(&[])
                .iter()
                .filter(|v| v.cell_name == cell.name && tech.variant_tt_wiring(v) == Some(cell.tt))
                .cloned()
                .collect::<Vec<_>>();
            assert_eq!(
                serde_json::to_string(&expected).unwrap(),
                serde_json::to_string(&tech.matching_cell_variants(cell)).unwrap()
            );
        }
    }
}
