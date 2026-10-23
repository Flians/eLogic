//! Region-owned physical matching on one global egraph and joint phase recovery.
//! Region boundaries retain global class IDs. Neither nodes, candidates, saturation
//! iterations nor recovery time are reset at region boundaries.
use super::*;

/// Import only logically demanded inversion enodes, while retaining every PI
/// and the constant. Physical signal phases remain globally available through
/// library matches; unreferenced Not enodes need not consume the egraph budget.
/// All actually imported nodes and all rewrite nodes still count globally.
pub(super) fn import_demanded(aig: &Aig) -> Option<(egg::EGraph<AigLang, ()>, Vec<(u32, Id)>)> {
    import_demanded_admitted(aig, &mut crate::egraph_budget::NodeAdmission::new(usize::MAX))
}
pub(super) fn import_demanded_admitted(aig: &Aig, admission: &mut crate::egraph_budget::NodeAdmission) -> Option<(egg::EGraph<AigLang, ()>, Vec<(u32, Id)>)> {
    let (expr, roots) = crate::egraph_sat::build_expr(aig)?;
    let nodes = expr.as_ref();
    let mut needed = vec![false; nodes.len()];
    let mut stack: Vec<_> = roots.iter().map(|(_, id)| *id).collect();
    for (i, node) in nodes.iter().enumerate() {
        if matches!(node, AigLang::Pi(_) | AigLang::Zero) {
            stack.push(Id::from(i));
        }
    }
    while let Some(id) = stack.pop() {
        let i = usize::from(id);
        if needed[i] {
            continue;
        }
        needed[i] = true;
        stack.extend(nodes[i].children().iter().copied());
    }
    let mut graph = egg::EGraph::default();
    let mut translated = vec![None; nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        if needed[i] {
            translated[i] = Some(
                admission.import(&mut graph, node.clone()
                        .map_children(|child| translated[usize::from(child)].unwrap()), "demanded")?,
            );
        }
    }
    graph.rebuild();
    let roots = roots
        .into_iter()
        .map(|(pos, id)| (pos, graph.find(translated[usize::from(id)].unwrap())))
        .collect();
    Some((graph, roots))
}

/// Direct phase-edge logical import. Original live AND nodes and every PI/zero
/// are represented, hash-consed and charged globally; only separate NOT enodes
/// on internal edges are replaced by explicit bits on the logical operator.
/// Conservative raw-language preflight. This plans storage without allocating
/// an oversized egraph; all live ANDs/PIs are still imported and charged later.
pub(super) fn ordinary_node_upper(aig:&Aig)->Option<usize> {
    use crate::aig_graph::AigNode;use petgraph::visit::EdgeRef;use petgraph::Direction;
    let g=aig.graph();let mut needed=HashSet::new();let mut stack=Vec::new();
    for n in g.node_indices() {if matches!(aig.node_type(n),AigNode::Input(_)|AigNode::Zero|AigNode::Output(_)){stack.push(n);}}
    while let Some(n)=stack.pop(){if needed.insert(n){stack.extend(g.edges_directed(n,Direction::Incoming).map(|e|e.source()));}}
    let base=needed.iter().filter(|&&n|!matches!(aig.node_type(n),AigNode::Output(_))).count();
    let inverted:HashSet<_>=needed.iter().flat_map(|&n|g.edges_directed(n,Direction::Incoming)).filter(|e|*e.weight()).map(|e|e.source()).collect();
    Some(base+inverted.len())
}

pub(super) fn import_literals(aig: &Aig) -> Option<(egg::EGraph<AigLang, ()>, Vec<(u32, Id)>)> {
    import_literals_admitted(aig, &mut crate::egraph_budget::NodeAdmission::new(usize::MAX))
}
pub(super) fn import_literals_admitted(aig: &Aig, admission: &mut crate::egraph_budget::NodeAdmission) -> Option<(egg::EGraph<AigLang, ()>, Vec<(u32, Id)>)> {
    use crate::aig_graph::AigNode;
    use petgraph::visit::{EdgeRef,Topo,IntoEdgeReferences};
    use petgraph::Direction;
    let g=aig.graph();let mut needed=HashSet::new();let mut stack=Vec::new();
    for n in g.node_indices() {
        if matches!(aig.node_type(n),AigNode::Input(_)|AigNode::Zero|AigNode::Output(_)) {stack.push(n);}
    }
    while let Some(n)=stack.pop() {
        if !needed.insert(n) {continue;}
        stack.extend(g.edges_directed(n,Direction::Incoming).map(|e|e.source()));
    }
    let mut graph=egg::EGraph::default();let mut ids=HashMap::new();let mut topo=Topo::new(g);
    while let Some(n)=topo.next(g) {
        if !needed.contains(&n) {continue;}
        let node=match aig.node_type(n) {
            AigNode::Zero=>AigLang::Zero,
            AigNode::Input(v)=>AigLang::Pi(v),
            AigNode::And=>{
                let (a,b)=aig.try_unwrap_and(n)?;
                let x=*ids.get(&aig.edge_source(a))?;let y=*ids.get(&aig.edge_source(b))?;
                match (aig.is_edge_inverted(a),aig.is_edge_inverted(b)) {
                    (false,false)=>AigLang::And([x,y]),
                    (true,false)=>AigLang::Cell("@egm_and_np".into(),vec![x,y]),
                    (false,true)=>AigLang::Cell("@egm_and_pn".into(),vec![x,y]),
                    (true,true)=>AigLang::Cell("@egm_and_nn".into(),vec![x,y]),
                }
            }
            AigNode::Output(_)=>continue,
        };
        ids.insert(n,admission.import(&mut graph,node,"literal")?);
    }
    let mut roots=Vec::new();
    for n in g.node_indices() {
        if let AigNode::Output(pos)=aig.node_type(n) {
            let edge=g.edges_directed(n,Direction::Incoming).next()?;
            let base=*ids.get(&edge.source())?;
            let root=if *edge.weight() {admission.import(&mut graph,AigLang::Not(base),"literal-output")?} else {base};
            roots.push((pos,root));
        }
    }
    let pre_rebuild_import_enodes=graph.total_number_of_nodes();
 roots.sort_by_key(|(p,_)|*p);graph.rebuild();
    let roots=roots.into_iter().map(|(p,id)|(p,graph.find(id))).collect();
    Some((graph,roots))
}

pub(super) fn ordered_boundaries<'a>(
    boundaries: &'a BTreeMap<Vec<usize>, Vec<Target>>,
    classes: usize,
    width: usize,
    stats: &mut MapStats,
) -> Vec<(&'a Vec<usize>, &'a Vec<Target>)> {
    let mut result: Vec<_> = boundaries.iter().collect();
    if width == 0 {
        return result;
    }
    stats.regions = classes.div_ceil(width);
    // One owner for a complete boundary, including targets in other regions:
    // multioutput sharing cannot be split into separately charged instances.
    result.sort_by_key(|(boundary, targets)| {
        let owner = targets
            .iter()
            .map(|t| t.signal / 2 / width)
            .min()
            .unwrap_or(0);
        (owner, *boundary)
    });
    stats.cross_region_boundaries = result
        .iter()
        .filter(|(boundary, targets)| {
            let owner = targets
                .iter()
                .map(|t| t.signal / 2 / width)
                .min()
                .unwrap_or(0);
            boundary.iter().any(|c| c / width != owner)
                || targets.iter().any(|t| t.signal / 2 / width != owner)
        })
        .count();
    result
}

/// Partition the remaining window without extending or resetting the global deadline.
/// Fraction one exactly preserves the previous scheduling policy.
pub(super) fn phase_deadline(start: Instant, global: Instant, fraction: f64) -> Instant {
    if fraction == 1.0 || global <= start {
        global
    } else {
        start + global.saturating_duration_since(start).mul_f64(fraction)
    }
}

pub(super) fn recover_phases(
    instances: &[Instance],
    chosen: &mut Vec<Option<Binding>>,
    witness: &[Option<Binding>],
    joint_reference: Option<&[Option<Binding>]>,
    free: &[bool],
    roots: &[usize],
    ps: &MapParams,
    stats: &mut MapStats,
    deadline: Instant,
) -> Result<(), String> {
    let mut producers = vec![Vec::new(); free.len()];
    let mut reverse = HashMap::<(usize, usize), Binding>::new();
    // Recognize physical unary INV pins. Use only already admitted bindings;
    // this pass creates no logical equality and never unions SDC classes.
    for (i, m) in instances.iter().enumerate() {
        if Instant::now() >= deadline {
            return Ok(());
        }
        for (pin, o) in m.outputs.iter().enumerate() {
            for &s in &o.targets {
                producers[s].push(Binding { instance: i, pin });
                if m.inputs.len() == 1 && o.function == 1 {
                    reverse
                        .entry((m.inputs[0], s))
                        .or_insert(Binding { instance: i, pin });
                }
            }
        }
    }
    let mut best = metrics(instances, chosen, free, roots);
    stats.phase_initial_delay = best.1;
    stats.phase_final_delay = best.1;
    for t in 0..chosen.len() {
        if Instant::now() >= deadline || stats.phase_trials >= ps.refine_trials {
            break;
        }
        let Some(old) = chosen[t] else {
            continue;
        };
        let inv = &instances[old.instance];
        if inv.inputs.len() != 1 || inv.outputs[old.pin].function != 1 {
            continue;
        }
        let s = inv.inputs[0];
        if free[s] || chosen[s].is_none() {
            continue;
        }
        let Some(&back) = reverse.get(&(t, s)) else {
            continue;
        };
        for &alternative in &producers[t] {
            if Instant::now() >= deadline || stats.phase_trials >= ps.refine_trials {
                break;
            }
            let candidate = &instances[alternative.instance];
            if alternative.instance == old.instance
                || candidate.inputs.contains(&s)
                || candidate.inputs.contains(&t)
            {
                continue;
            }
            stats.phase_trials += 1;
            let mut trial = chosen.clone();
            trial[t] = Some(alternative);
            trial[s] = Some(back);
            // Atomic replacement prevents the temporary s->t->s inverse cycle.
            // Full DAG validation and global unique-instance metric handle
            // fanout sharing and all live pins, including multioutput cells.
            if joint_input_cone(instances, &mut trial, witness, free, roots,joint_reference,chosen,stats,deadline).is_err() {
                continue;
            }
            let score = metrics(instances, &trial, free, roots);
            if better_cover(score, best, ps) {
                *chosen = trial;
                best = score;
                stats.phase_accepted += 1;
                stats.phase_final_delay = score.1;
            }
        }
    }
    // Optional proposals cannot consume the legacy atomic phase trial budget first.
    if ps.introduce_phase_inverters && Instant::now()<deadline && stats.phase_trials<ps.refine_trials {
        // Direct producers of both phases need not contain a selected INV seed.
        // Propose only admitted bindings; rank by estimated reclaim, but accept
        // solely after full DAG pruning and unique physical-instance scoring.
        // These trials share the original phase trial cap and stage deadline.
        let mut proposals=Vec::new();
        for (&(s,t),&b) in &reverse {
            if Instant::now()>=deadline {return prune(instances,chosen,free,roots);}
            if free[t] || chosen[t].is_none() || (!free[s] && chosen[s].is_none()) {continue;}
            let old=chosen[t].unwrap();
            if (old.instance,old.pin)==(b.instance,b.pin) {continue;}
            let gain=instances[old.instance].area-instances[b.instance].area;
            proposals.push((gain,s,t,b));
        }
        proposals.sort_by(|a,b|b.0.total_cmp(&a.0).then(a.2.cmp(&b.2)).then(a.1.cmp(&b.1)).then(a.3.instance.cmp(&b.3.instance)).then(a.3.pin.cmp(&b.3.pin)));
        for (_,_,t,b) in proposals {
            if Instant::now()>=deadline || stats.phase_trials>=ps.refine_trials {break;}
            if chosen[t].is_none() {continue;}
            stats.phase_trials+=1;
            let mut trial=chosen.clone();trial[t]=Some(b);
            if joint_input_cone(instances,&mut trial,witness,free,roots,joint_reference,chosen,stats,deadline).is_err(){continue;}
            let score=metrics(instances,&trial,free,roots);
            if better_cover(score,best,ps){*chosen=trial;best=score;stats.phase_accepted+=1;stats.phase_final_delay=score.1;}
        }
    }
    prune(instances, chosen, free, roots)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unselected_inverse_recovers_slack_shared_area() {
        let mut m=vec![cell("AND",vec![0,1],2,8,1.),cell("NAND",vec![0,1],3,7,1.),cell("BUF",vec![0],4,2,3.),cell("INV",vec![3],2,1,1.)];
        m[0].area=4.;m[3].area=0.5;
        let free=vec![true,true,false,false,false];let roots=vec![2,3,4];
        let original=vec![None,None,Some(Binding{instance:0,pin:0}),Some(Binding{instance:1,pin:0}),Some(Binding{instance:2,pin:0})];
        for (enabled,trials,expired,expected) in [(false,100,false,6.),(true,100,false,2.5),(true,0,false,6.),(true,100,true,6.)] {
            let mut chosen=original.clone();let mut stats=MapStats::default();
            let ps=MapParams{objective:MapObjective::DelayArea,phase_recovery:true,introduce_phase_inverters:enabled,refine_trials:trials,..Default::default()};
            let deadline=if expired{Instant::now()}else{Instant::now()+std::time::Duration::from_secs(1)};
            recover_phases(&m,&mut chosen,&original,None,&free,&roots,&ps,&mut stats,deadline).unwrap();
            assert_eq!(metrics(&m,&chosen,&free,&roots),(expected,3.));
            assert!(stats.phase_trials<=ps.refine_trials);
            // Explicit original truth and introduced negative-to-positive phase.
            for a in [false,true] {for b in [false,true] {
                let mut values=vec![a,b,false,false,false];
                fn eval(s:usize,m:&[Instance],c:&[Option<Binding>],values:&mut[bool],free:&[bool])->bool {
                    if free[s]{return values[s];}let b=c[s].unwrap();let inst=&m[b.instance];let mut idx=0;
                    for (i,&input) in inst.inputs.iter().enumerate(){if eval(input,m,c,values,free){idx|=1<<i;}}
                    let v=((inst.outputs[b.pin].function>>idx)&1)!=0;values[s]=v;v
                }
                assert_eq!(eval(2,&m,&chosen,&mut values,&free),a&&b);
                assert_eq!(eval(3,&m,&chosen,&mut values,&free),!(a&&b));
                assert_eq!(eval(4,&m,&chosen,&mut values,&free),a);
            }}
        }
    }

    #[test]
    fn new_inverse_cannot_starve_legacy_atomic_flip() {
        let m=vec![cell("AND",vec![0,1],2,8,10.),cell("INV",vec![2],3,1,3.),cell("NAND",vec![0,1],3,7,4.),cell("INV",vec![3],2,1,3.)];
        let free=vec![true,true,false,false];let roots=vec![2,3];
        for enabled in [false,true] {
            let mut chosen=vec![None,None,Some(Binding{instance:0,pin:0}),Some(Binding{instance:1,pin:0})];let witness=chosen.clone();let mut stats=MapStats::default();
            let ps=MapParams{objective:MapObjective::DelayArea,phase_recovery:true,introduce_phase_inverters:enabled,refine_trials:1,..Default::default()};
            recover_phases(&m,&mut chosen,&witness,None,&free,&roots,&ps,&mut stats,Instant::now()+std::time::Duration::from_secs(1)).unwrap();
            assert_eq!(metrics(&m,&chosen,&free,&roots),(2.,7.),"optional cyclic reverse INV must not consume sole legacy phase trial");
            assert!(stats.phase_trials<=1);
        }
    }

    #[test]
    fn phase_allocation_preserves_global_deadline() {
        let start = Instant::now();
        let global = start + std::time::Duration::from_secs(2);
        assert_eq!(phase_deadline(start, global, 1.0), global);
        assert_eq!(phase_deadline(start, global, 0.0), start);
        assert_eq!(phase_deadline(start, global, 0.25), start + std::time::Duration::from_millis(500));
        let expired = start - std::time::Duration::from_millis(1);
        assert_eq!(phase_deadline(start, expired, 1.0), expired);
        assert_eq!(phase_deadline(start, expired, 0.25), expired);
    }

    fn cell(name: &str, inputs: Vec<usize>, target: usize, function: u64, delay: f32) -> Instance {
        Instance {
            cell: name.into(),
            area: 1.,
            input_pins: inputs.iter().map(|_| "A".into()).collect(),
            outputs: vec![OutputPin {sources:Vec::new(),
                name: "Y".into(),
                delays: vec![delay; inputs.len()],
                function,
                targets: vec![target],
            }],
            inputs,
        }
    }
    #[test]
    fn atomic_phase_flip_preserves_shared_positive_root() {
        let m = vec![
            cell("AND", vec![0, 1], 2, 8, 10.),
            cell("INV", vec![2], 3, 1, 3.),
            cell("NAND", vec![0, 1], 3, 7, 4.),
            cell("INV", vec![3], 2, 1, 3.),
        ];
        let free = vec![true, true, false, false];
        let roots = vec![2, 3];
        for objective in [MapObjective::AreaDelay, MapObjective::DelayArea] {
            let mut chosen = vec![
                None,
                None,
                Some(Binding {
                    instance: 0,
                    pin: 0,
                }),
                Some(Binding {
                    instance: 1,
                    pin: 0,
                }),
            ];
            let witness = chosen.clone();
            let ps = MapParams {
                objective,
                phase_recovery: true,
                ..Default::default()
            };
            let mut stats = MapStats::default();
            recover_phases(
                &m,
                &mut chosen,
                &witness,
                None,
                &free,
                &roots,
                &ps,
                &mut stats,
                Instant::now() + std::time::Duration::from_secs(1),
            )
            .unwrap();
            assert_eq!(metrics(&m, &chosen, &free, &roots), (2., 7.));
            assert_eq!(stats.phase_accepted, 1);
        }
    }
    #[test]
    fn expired_phase_deadline_preserves_incumbent() {
        let m = vec![
            cell("AND", vec![0, 1], 2, 8, 10.),
            cell("INV", vec![2], 3, 1, 3.),
        ];
        let free = vec![true, true, false, false];
        let roots = vec![2, 3];
        let mut chosen = vec![
            None,
            None,
            Some(Binding {
                instance: 0,
                pin: 0,
            }),
            Some(Binding {
                instance: 1,
                pin: 0,
            }),
        ];
        let witness = chosen.clone();
        let mut stats = MapStats::default();
        recover_phases(
            &m,
            &mut chosen,
            &witness,
            None,
            &free,
            &roots,
            &MapParams::default(),
            &mut stats,
            Instant::now(),
        )
        .unwrap();
        assert_eq!(stats.phase_trials, 0);
        assert_eq!(metrics(&m, &chosen, &free, &roots), (2., 13.));
    }

    #[test]
    fn region_ownership_keeps_cross_region_joint_targets() {
        let b = BTreeMap::from([(
            vec![0, 3],
            vec![
                Target {
                    signal: 4,
                    function: 8,
                },
                Target {
                    signal: 12,
                    function: 7,
                },
            ],
        )]);
        let mut stats = MapStats::default();
        let owned = ordered_boundaries(&b, 8, 2, &mut stats);
        assert_eq!(owned.len(), 1);
        assert_eq!(owned[0].1.len(), 2);
        assert_eq!(stats.regions, 4);
        assert_eq!(stats.cross_region_boundaries, 1);
    }
}

pub(super) use crate::egraph_budget::reserve_rewrites;

/// Fixed exact XOR pattern contraction BEFORE import. Original intermediate AND
/// functions replaced by a two-input XOR are not materialized as hidden eclasses.
/// Every realized XOR/AND/PI/Not/constant is globally charged; no retirement/reset.
#[derive(Clone,Copy)]
pub(super) struct SourceAnd{pub root:Id,pub left:Id,pub right:Id,pub left_neg:bool,pub right_neg:bool}

pub(super) fn import_xors(aig:&Aig)->Option<(egg::EGraph<AigLang,()>,Vec<(u32,Id)>)>{import_with_source(aig,false).map(|(g,r,_)|(g,r))}

/// All helper functions are actual imported nodes. The source AND certificate
/// only authorizes physical cut/match records in the ordinary bounded cut/pool
/// bank; it adds no egraph alternative or conditional union. No omitted function
/// may appear in any certificate or physical boundary.
pub(super) fn import_with_source(aig:&Aig,keep_helpers:bool)->Option<(egg::EGraph<AigLang,()>,Vec<(u32,Id)>,Vec<SourceAnd>)>{
 import_with_helpers(aig,if keep_helpers{usize::MAX}else{0},usize::MAX)
}
/// floor(9n/10), without overflowing on an arbitrary caller's node allowance.
pub(super) fn initial_node_ceiling(n:usize)->usize{n-n/10-usize::from(n%10!=0)}

pub(super) fn import_with_helpers(aig:&Aig,helper_quota:usize,node_allowance:usize)->Option<(egg::EGraph<AigLang,()>,Vec<(u32,Id)>,Vec<SourceAnd>)>{import_with_canonical_helpers(aig,helper_quota,node_allowance,false)}
pub(super) fn import_with_canonical_helpers(aig:&Aig,helper_quota:usize,node_allowance:usize,canonical:bool)->Option<(egg::EGraph<AigLang,()>,Vec<(u32,Id)>,Vec<SourceAnd>)>{import_with_helper_allocation(aig,helper_quota,node_allowance,canonical,false)}
pub(super) fn import_with_helper_allocation(aig:&Aig,helper_quota:usize,node_allowance:usize,canonical:bool,fill_unused:bool)->Option<(egg::EGraph<AigLang,()>,Vec<(u32,Id)>,Vec<SourceAnd>)>{import_helpers_admitted(aig,helper_quota,node_allowance,canonical,fill_unused,&mut crate::egraph_budget::NodeAdmission::new(node_allowance))}
pub(super) fn import_helpers_admitted(aig:&Aig,helper_quota:usize,node_allowance:usize,canonical:bool,fill_unused:bool,admission:&mut crate::egraph_budget::NodeAdmission)->Option<(egg::EGraph<AigLang,()>,Vec<(u32,Id)>,Vec<SourceAnd>)>{
 use crate::aig_graph::AigNode;use petgraph::visit::{EdgeRef,Topo,IntoEdgeReferences};use petgraph::Direction;
 let g=aig.graph();let pair=|n|{let (a,b)=aig.try_unwrap_and(n)?;let mut v=vec![(aig.edge_source(a),aig.is_edge_inverted(a)),(aig.edge_source(b),aig.is_edge_inverted(b))];v.sort();Some(v)};
 let mut macros=HashMap::new();
 for n in g.node_indices(){if !matches!(aig.node_type(n),AigNode::And){continue}let outer=pair(n)?;if !outer.iter().all(|(n,p)|*p && matches!(aig.node_type(*n),AigNode::And)){continue}let x=pair(outer[0].0)?;let y=pair(outer[1].0)?;let mut complement:Vec<_>=y.iter().map(|&(n,p)|(n,!p)).collect();complement.sort();if x==complement{macros.insert(n,x);}}
 let mut uses=HashMap::<_,usize>::new();for e in g.edge_references(){*uses.entry(e.source()).or_default()+=1;}
 let mut triples=HashMap::new();
 for n in g.node_indices(){if !matches!(aig.node_type(n),AigNode::And)||macros.contains_key(&n){continue}let outer=pair(n)?;
  for which in 0..2{let child=outer[which].0;if !matches!(aig.node_type(child),AigNode::And)||uses[&child]!=1||macros.contains_key(&child){continue}let inner=pair(child)?;let leaves:Vec<_>=[outer[1-which].0,inner[0].0,inner[1].0].into_iter().collect::<BTreeSet<_>>().into_iter().collect();if leaves.len()!=3{continue}let mut tt=0;for a in 0..8{let get=|node|((a>>leaves.binary_search(&node).unwrap())&1)!=0;let c=(get(inner[0].0)^inner[0].1)&(get(inner[1].0)^inner[1].1);let other=get(outer[1-which].0)^outer[1-which].1;tt|=u64::from((c^outer[which].1)&other)<<a;}triples.insert(n,(leaves,tt));break;}
 }
 let mut needed=HashSet::new();let mut stack:Vec<_>=g.node_indices().filter(|&n|matches!(aig.node_type(n),AigNode::Input(_)|AigNode::Zero|AigNode::Output(_))).collect();
 while let Some(n)=stack.pop(){if !needed.insert(n){continue}if let Some((leaves,_))=triples.get(&n){stack.extend(leaves);}else if let Some(v)=macros.get(&n){stack.extend(v.iter().map(|(n,_)|*n));}else{stack.extend(g.edges_directed(n,Direction::Incoming).map(|e|e.source()));}}
 let mut helpers=HashSet::new();let mut old_helper_sources=Vec::new();let mut planned_import_upper=0usize;
 if helper_quota==usize::MAX {for n in g.node_indices(){if matches!(aig.node_type(n),AigNode::And)&&!needed.contains(&n){helpers.insert(n);needed.insert(n);}}}
 else if helper_quota>0 {
  let mut topo=Topo::new(g);let mut order=Vec::new();let mut depth=HashMap::<_,usize>::new();while let Some(n)=topo.next(g){let d=if matches!(aig.node_type(n),AigNode::And){let v=pair(n)?;depth[&v[0].0].max(depth[&v[1].0])+1usize}else if matches!(aig.node_type(n),AigNode::Output(_)){let e=g.edges_directed(n,Direction::Incoming).next()?;depth[&e.source()]}else{0};depth.insert(n,d);order.push(n);}
  let maximum=order.iter().filter(|&&n|matches!(aig.node_type(n),AigNode::Output(_))).map(|n|depth[n]).max().unwrap_or(0);let mut required=HashMap::<_,usize>::new();for &n in &order{if matches!(aig.node_type(n),AigNode::Output(_)){required.insert(n,maximum);}}
  for &n in order.iter().rev(){if let Some(&r)=required.get(&n){let child_r=r.saturating_sub(usize::from(matches!(aig.node_type(n),AigNode::And)));for e in g.edges_directed(n,Direction::Incoming){required.entry(e.source()).and_modify(|v|*v=(*v).min(child_r)).or_insert(child_r);}}}
  let mut groups=Vec::new();for &n in &needed{if !matches!(aig.node_type(n),AigNode::And)||(!macros.contains_key(&n)&&!triples.contains_key(&n)){continue}let original=pair(n)?;let missing:Vec<_>=original.iter().map(|v|v.0).filter(|n|!needed.contains(n)).collect::<BTreeSet<_>>().into_iter().collect();if missing.is_empty(){continue}if missing.iter().all(|n|matches!(aig.node_type(*n),AigNode::And)&&pair(*n).is_some_and(|v|v.iter().all(|(c,_)|needed.contains(c)))){groups.push((required.get(&n).copied().unwrap_or(usize::MAX).saturating_sub(depth[&n]),std::cmp::Reverse(depth[&n]),n,missing));}}
  groups.sort_by_key(|(slack,d,n,_)|(*slack,*d,*n));let output_not:HashSet<_>=g.node_indices().filter(|&n|matches!(aig.node_type(n),AigNode::Output(_))).flat_map(|n|g.edges_directed(n,Direction::Incoming)).filter(|e|*e.weight()).map(|e|e.source()).collect();let base=needed.iter().filter(|&&n|!matches!(aig.node_type(n),AigNode::Output(_))).count()+output_not.len();let ceiling=initial_node_ceiling(node_allowance);
  // First preserve the exact original mandatory helper set and its ordering.
  for (_,_,_,missing) in &groups{let novel:Vec<_>=missing.iter().copied().filter(|n|!needed.contains(n)).collect();if helpers.len()+novel.len()>helper_quota||base+helpers.len()+novel.len()>ceiling{continue}for n in novel{needed.insert(n);helpers.insert(n);}}
  old_helper_sources=helpers.iter().map(|n|n.index()).collect();old_helper_sources.sort_unstable();
  // Allocate unused INITIAL-node capacity, not additional global nodes. Reserve
  // the same10% for later rewriting; all old helpers remain actual graph nodes.
  if fill_unused {for (_,_,_,missing) in &groups{let novel:Vec<_>=missing.iter().copied().filter(|n|!needed.contains(n)).collect();if base+helpers.len()+novel.len()>ceiling{continue}for n in novel{needed.insert(n);helpers.insert(n);}}}
  planned_import_upper=base+helpers.len();
 }
 if let Some(path)=std::env::var_os("EGM_POOL_DIAG_PATH"){let mut all:Vec<_>=helpers.iter().map(|n|n.index()).collect();all.sort_unstable();let _=std::fs::write(std::path::Path::new(&path).with_extension("helper-allocation.json"),serde_json::json!({"enabled":fill_unused,"old_helper_quota":helper_quota,"old_helper_sources":old_helper_sources,"actual_helper_sources":all,"planned_import_upper":planned_import_upper,"initial_node_ceiling":initial_node_ceiling(node_allowance),"global_node_allowance":node_allowance,"scope":"originalmandatoryhelpers retained; extrahelpersconsumeunusedsameglobalinitialcapacity, no20kplushelpers/retirement/reset; actual nodes still checkedbeforeeveryadd"}).to_string());}
 let mut graph=egg::EGraph::default();let mut ids=HashMap::new();let mut topo=Topo::new(g);let mut actual_xor=0;let mut actual_tt3=0;let mut canonical_examined=0;let mut canonical_merges=0;
 while let Some(n)=topo.next(g){if !needed.contains(&n){continue}let node=match aig.node_type(n){
 AigNode::Zero=>AigLang::Zero,AigNode::Input(v)=>AigLang::Pi(v),AigNode::Output(_)=>continue,
 AigNode::And=>{if let Some((leaves,tt))=triples.get(&n).filter(|_|!helpers.contains(&n)){actual_tt3+=1;AigLang::Cell(format!("@egm_tt3_{tt:02x}").into(),leaves.iter().map(|n|ids[n]).collect())}else if let Some(v)=macros.get(&n).filter(|_|!helpers.contains(&n)){actual_xor+=1;let odd=v[0].1^v[1].1;AigLang::Cell(if odd{"@egm_xnor"}else{"@egm_xor"}.into(),v.iter().map(|(n,_)|ids[n]).collect())}else{let v=pair(n)?;let x=ids[&v[0].0];let y=ids[&v[1].0];match(v[0].1,v[1].1){(false,false)=>AigLang::And([x,y]),(true,false)=>AigLang::Cell("@egm_and_np".into(),vec![x,y]),(false,true)=>AigLang::Cell("@egm_and_pn".into(),vec![x,y]),(true,true)=>AigLang::Cell("@egm_and_nn".into(),vec![x,y])}}}};
 let node=if canonical{canonical_source_node(node)}else{node};
 canonical_examined+=1;if graph.lookup(node.clone()).is_some(){canonical_merges+=1;}
 ids.insert(n,admission.import(&mut graph,node,"cone-helper")?);}
 let mut roots=Vec::new();for n in g.node_indices(){if let AigNode::Output(pos)=aig.node_type(n){let e=g.edges_directed(n,Direction::Incoming).next()?;let base=ids[&e.source()];let root=if *e.weight(){let node=AigLang::Not(base);admission.import(&mut graph,node,"cone-output")?}else{base};roots.push((pos,root));}}
 canonical_source_audit(canonical,true,canonical_examined,canonical_merges,graph.total_number_of_nodes(),node_allowance);
 let pre_rebuild_import_enodes=graph.total_number_of_nodes();
 roots.sort_by_key(|(p,_)|*p);graph.rebuild();let roots=roots.into_iter().map(|(p,id)|(p,graph.find(id))).collect();
 let mut source=Vec::new();for n in g.node_indices(){if !ids.contains_key(&n)||helpers.contains(&n)||(!macros.contains_key(&n)&&!triples.contains_key(&n)){continue}let inputs=pair(n)?;if let(Some(&a),Some(&b))=(ids.get(&inputs[0].0),ids.get(&inputs[1].0)){source.push(SourceAnd{root:graph.find(ids[&n]),left:graph.find(a),right:graph.find(b),left_neg:inputs[0].1,right_neg:inputs[1].1});}}

 let helper_classes:BTreeSet<_>=helpers.iter().map(|n|graph.find(ids[n])).collect();
 let unique_xor=graph.classes().flat_map(|c|&c.nodes).filter(|n|crate::egraph_sat::xor_inputs(n).is_some()).count();
 let unique_tt3=graph.classes().flat_map(|c|&c.nodes).filter(|n|crate::egraph_sat::tt3_inputs(n).is_some()).count();
 if let Some(path)=std::env::var_os("EGM_POOL_DIAG_PATH"){let path=std::path::Path::new(&path).with_extension("import.json");let _=std::fs::write(path,serde_json::json!({"schema_version":2,"import_complete":true,"pre_rebuild_import_enodes":pre_rebuild_import_enodes,"actual_imported_enodes":graph.total_number_of_nodes(),"source_xor_occurrences":actual_xor,"source_tt3_occurrences":actual_tt3,"actual_xor_enodes":unique_xor,"actual_tt3_enodes":unique_tt3,"retained_xor_operators":unique_xor,"retained_tt3_operators":unique_tt3,"all_original_helpers_retained":helper_quota==usize::MAX,"helper_quota":helper_quota,"helper_quota_scope":"Original mandatory subset; explicit global-capacity reallocation may add helpers within unchanged cumulative node limit.","helper_allocation_enabled":fill_unused,"mandatory_old_helpers":old_helper_sources.len(),"selected_source_helpers":helpers.len(),"actual_helpers":helper_classes.len(),"helper_node_reserve_fraction":0.1,"source_physical_witnesses":source.len(),"original_source_ands":g.node_indices().filter(|&n|matches!(aig.node_type(n),AigNode::And)).count(),"representation":"Exact pre-import XOR and unshared two-AND TT3 contraction; omitted internal functions not retained as hidden classes; all actual enodes count cumulatively; no regional reset. Not an unchanged expanded-AIG-node budget comparison."}).to_string());}
 Some((graph,roots,source))
}

pub(super) fn compact_node_upper(aig:&Aig)->Option<usize>{
 use crate::aig_graph::AigNode;use petgraph::visit::EdgeRef;use petgraph::Direction;
 let g=aig.graph();let mut live=HashSet::new();let mut stack:Vec<_>=g.node_indices().filter(|&n|matches!(aig.node_type(n),AigNode::Input(_)|AigNode::Zero|AigNode::Output(_))).collect();
 while let Some(n)=stack.pop(){if live.insert(n){stack.extend(g.edges_directed(n,Direction::Incoming).map(|e|e.source()));}}
 let output_not:HashSet<_>=g.node_indices().filter(|&n|matches!(aig.node_type(n),AigNode::Output(_))).flat_map(|n|g.edges_directed(n,Direction::Incoming)).filter(|e|*e.weight()).map(|e|e.source()).collect();
 Some(live.iter().filter(|&&n|!matches!(aig.node_type(n),AigNode::Output(_))).count()+output_not.len())
}

fn admission_failure(limit:usize,admitted:usize){
 eprintln!("budget: cumulative import would exceed node allowance {limit}; {admitted} admitted, incomplete input");
 if let Some(path)=std::env::var_os("EGM_POOL_DIAG_PATH"){let path=std::path::Path::new(&path).with_extension("import.json");let _=std::fs::write(path,serde_json::json!({"schema_version":3,"import_complete":false,"cumulative_import_admitted_enodes":admitted,"node_allowance":limit,"no_egraph_retirement":true,"error":"extra distinct operator or PO phase rejected before admission; incomplete prefix is not a valid cover"}).to_string());}
}


fn canonical_source_node(node:AigLang)->AigLang {
 if let Some((args,tt))=crate::egraph_sat::tt3_inputs(&node){let mut order:Vec<_>=(0..args.len()).collect();order.sort_by_key(|&p|(args[p],p));let new_args=order.iter().map(|&p|args[p]).collect();let mut new_tt=0u64;for a in 0..(1usize<<args.len()){let old=order.iter().enumerate().map(|(p,&old)|((a>>p)&1)<<old).sum::<usize>();new_tt|=((tt>>old)&1)<<a;}return AigLang::Cell(format!("@egm_tt3_{new_tt:02x}").into(),new_args)}
 match node {
  AigLang::And(mut args)=>{args.sort_unstable();AigLang::And(args)},
  AigLang::Cell(sym,mut args) if args.len()==2 && ["@egm_xor","@egm_xnor","@egm_and_np","@egm_and_pn","@egm_and_nn"].contains(&sym.as_str())=>{
   if args[0]>args[1]{args.swap(0,1);let name=match sym.as_str(){"@egm_and_np"=>"@egm_and_pn","@egm_and_pn"=>"@egm_and_np",name=>name};AigLang::Cell(name.into(),args)}else{AigLang::Cell(sym,args)}
  },node=>node,
 }
}
#[cfg(test)]mod source_permutation_tests{use super::*;
 #[test]fn all_tt3_permutations_share_actual_counted_node(){
  for tt in 0u64..256 {let mut g=egg::EGraph::<AigLang,()>::default();let vars:Vec<_>=(0..3).map(|n|g.add(AigLang::Pi(n))).collect();let root=g.add(canonical_source_node(AigLang::Cell(format!("@egm_tt3_{tt:02x}").into(),vars.clone())));
   for order in [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]{let args=order.iter().map(|&p|vars[p]).collect();let mut perm=0;for a in 0..8{let old=order.iter().enumerate().map(|(p,&old)|((a>>p)&1)<<old).sum::<usize>();perm|=((tt>>old)&1)<<a;}
    let normalized=canonical_source_node(AigLang::Cell(format!("@egm_tt3_{perm:02x}").into(),args));assert_eq!(g.add(normalized),root);
   }assert_eq!(g.total_number_of_nodes(),4);
  }
 }
 #[test]fn canonical_literals_keep_negated_operand_and_xor_phase(){
  let a=Id::from(1);let b=Id::from(2);for (name,swapped) in [("@egm_and_np","@egm_and_pn"),("@egm_and_pn","@egm_and_np"),("@egm_and_nn","@egm_and_nn"),("@egm_xor","@egm_xor"),("@egm_xnor","@egm_xnor")]{assert_eq!(canonical_source_node(AigLang::Cell(name.into(),vec![b,a])),AigLang::Cell(swapped.into(),vec![a,b]));}
  for tt in 0u64..256{let node=AigLang::Cell(format!("@egm_tt3_{tt:02x}").into(),vec![b,a,a]);let next=canonical_source_node(node);let (args,nt)=crate::egraph_sat::tt3_inputs(&next).unwrap();for bits in 0..4{let get=|id:Id|if id==a{bits&1}else{(bits>>1)&1};let old=get(b)|(get(a)<<1)|(get(a)<<2);let new=args.iter().enumerate().map(|(p,&id)|get(id)<<p).sum::<usize>();assert_eq!((tt>>old)&1,(nt>>new)&1);}}
 }
}

fn canonical_source_audit(enabled:bool,complete:bool,examined:usize,merges:usize,nodes:usize,limit:usize){if let Some(p)=std::env::var_os("EGM_POOL_DIAG_PATH"){let _=std::fs::write(std::path::Path::new(&p).with_extension("canonical-import.json"),serde_json::json!({"canonical_source_inputs":enabled,"import_complete":complete,"source_operators_examined":examined,"source_alias_hits":merges,"actual_nodes":nodes,"node_allowance":limit,"scope":"exact argument/TT permutation before hash-consing; failed prefix is not full input, mapped cover or feasibility proof; no node retirement/reset"}).to_string());}}

#[cfg(test)]mod helper_allocation_tests{use super::*;use crate::aig_graph::AigNode;use egg::Language;
 fn circuit()->Aig{let mut a=Aig::default();a.add_node(AigNode::Zero);let p:Vec<_>=(1..=18).map(|v|a.add_node(AigNode::Input(v))).collect();for k in 0..6{let u=a.add_node(AigNode::And);a.add_edge(p[k*3],u,false);a.add_edge(p[k*3+1],u,false);let r=a.add_node(AigNode::And);a.add_edge(u,r,false);a.add_edge(p[k*3+2],r,false);let o=a.add_node(AigNode::Output(k as u32));a.add_edge(r,o,false);}a}
 fn signatures(g:&egg::EGraph<AigLang,()>)->BTreeSet<String>{fn visit(g:&egg::EGraph<AigLang,()>,id:Id,m:&mut HashMap<Id,String>)->String{let id=g.find(id);if let Some(s)=m.get(&id){return s.clone()}let n=&g[id].nodes[0];let name=match n{AigLang::Pi(v)=>format!("pi{v}"),AigLang::Zero=>"zero".into(),AigLang::One=>"one".into(),AigLang::Cell(s,_)=>format!("cell{s}"),AigLang::And(_)=>"and".into(),AigLang::Not(_)=>"not".into()};let args:Vec<_>=n.children().iter().map(|&c|visit(g,c,m)).collect();let text=format!("{name}({})",args.join(","));m.insert(id,text.clone());text}let mut memo=HashMap::new();g.classes().map(|c|visit(g,c.id,&mut memo)).collect()}
 #[test]fn unused_global_capacity_preserves_every_old_function_and_all_inputs(){
  let a=circuit();let(old,_,old_cert)=import_with_helper_allocation(&a,2,40,false,false).unwrap();let(new,_,new_cert)=import_with_helper_allocation(&a,2,40,false,true).unwrap();assert_eq!(old.total_number_of_nodes(),27);assert_eq!(new.total_number_of_nodes(),31);assert!(signatures(&old).is_subset(&signatures(&new)));assert!(new_cert.len()>old_cert.len());assert_eq!(new.classes().filter(|c|matches!(c.nodes[0],AigLang::Pi(_))).count(),18);assert!(new.total_number_of_nodes()<=initial_node_ceiling(40));
  let (tight,_,_)=import_with_helper_allocation(&a,2,27,false,true).unwrap();assert!(tight.total_number_of_nodes()<=27);assert!(import_with_helper_allocation(&a,2,24,false,true).is_none());
 }
}
