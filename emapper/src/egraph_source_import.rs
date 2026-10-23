//! Exact bounded source planning; isolated from physical phase recovery.
use super::*;
use super::regions::SourceAnd;
use super::regions::initial_node_ceiling;
pub(super) fn import_with_shared_helpers(aig:&Aig,quota:usize,limit:usize)->Option<(egg::EGraph<AigLang,()>,Vec<(u32,Id)>,Vec<SourceAnd>)>{import_helpers_impl(aig,quota,limit,true,3,None)}
pub(super) fn import_with_helpers(aig:&Aig,helper_quota:usize,node_allowance:usize)->Option<(egg::EGraph<AigLang,()>,Vec<(u32,Id)>,Vec<SourceAnd>)>{
 import_helpers_impl(aig,helper_quota,node_allowance,false,3,None)
}
pub(super) fn import_with_library_source(aig:&Aig,quota:usize,limit:usize,width:usize,tech:&TechLib)->Option<(egg::EGraph<AigLang,()>,Vec<(u32,Id)>,Vec<SourceAnd>)>{import_helpers_impl(aig,quota,limit,true,width.clamp(3,4),Some(tech))}
fn source_table(tt:u64,args:Vec<Id>)->AigLang {AigLang::Cell(if args.len()==4{format!("@egm_tt4_{tt:04x}")}else{format!("@egm_tt3_{tt:02x}")}.into(),args)}
fn import_helpers_impl(aig:&Aig,helper_quota:usize,node_allowance:usize,shared_majority:bool,source_width:usize,source_tech:Option<&TechLib>)->Option<(egg::EGraph<AigLang,()>,Vec<(u32,Id)>,Vec<SourceAnd>)>{import_source_admitted(aig,helper_quota,node_allowance,shared_majority,source_width,source_tech,&mut crate::egraph_budget::NodeAdmission::new(node_allowance))}
pub(super) fn import_source_admitted(aig:&Aig,helper_quota:usize,node_allowance:usize,shared_majority:bool,source_width:usize,source_tech:Option<&TechLib>,admission:&mut crate::egraph_budget::NodeAdmission)->Option<(egg::EGraph<AigLang,()>,Vec<(u32,Id)>,Vec<SourceAnd>)>{
 use crate::aig_graph::AigNode;use petgraph::visit::{EdgeRef,Topo,IntoEdgeReferences};use petgraph::Direction;
 let g=aig.graph();let pair=|n|{let (a,b)=aig.try_unwrap_and(n)?;let mut v=vec![(aig.edge_source(a),aig.is_edge_inverted(a)),(aig.edge_source(b),aig.is_edge_inverted(b))];v.sort();Some(v)};
 let mut macros=HashMap::new();
 for n in g.node_indices(){if !matches!(aig.node_type(n),AigNode::And){continue}let outer=pair(n)?;if !outer.iter().all(|(n,p)|*p && matches!(aig.node_type(*n),AigNode::And)){continue}let x=pair(outer[0].0)?;let y=pair(outer[1].0)?;let mut complement:Vec<_>=y.iter().map(|&(n,p)|(n,!p)).collect();complement.sort();if x==complement{macros.insert(n,x);}}
 let mut uses=HashMap::<_,usize>::new();for e in g.edge_references(){*uses.entry(e.source()).or_default()+=1;}
 let mut triples=HashMap::new();
 for n in g.node_indices(){if !matches!(aig.node_type(n),AigNode::And)||macros.contains_key(&n){continue}let outer=pair(n)?;
  for which in 0..2{let child=outer[which].0;if !matches!(aig.node_type(child),AigNode::And)||uses[&child]!=1||macros.contains_key(&child){continue}let inner=pair(child)?;let leaves:Vec<_>=[outer[1-which].0,inner[0].0,inner[1].0].into_iter().collect::<BTreeSet<_>>().into_iter().collect();if leaves.len()!=3{continue}let mut tt=0;for a in 0..8{let get=|node|((a>>leaves.binary_search(&node).unwrap())&1)!=0;let c=(get(inner[0].0)^inner[0].1)&(get(inner[1].0)^inner[1].1);let other=get(outer[1-which].0)^outer[1-which].1;tt|=u64::from((c^outer[which].1)&other)<<a;}triples.insert(n,(leaves,tt));break;}
 }
 let mut raw_source_upper=0usize;
 if shared_majority {
  let mut raw=HashSet::new();let mut work:Vec<_>=g.node_indices().filter(|&n|matches!(aig.node_type(n),AigNode::Input(_)|AigNode::Zero|AigNode::Output(_))).collect();
  while let Some(n)=work.pop(){if !raw.insert(n){continue}if let Some((leaves,_))=triples.get(&n){work.extend(leaves);}else if let Some(v)=macros.get(&n){work.extend(v.iter().map(|x|x.0));}else{work.extend(g.edges_directed(n,Direction::Incoming).map(|e|e.source()));}}
  let inversions:HashSet<_>=g.node_indices().filter(|&n|matches!(aig.node_type(n),AigNode::Output(_))).flat_map(|n|g.edges_directed(n,Direction::Incoming)).filter(|e|*e.weight()).map(|e|e.source()).collect();
  raw_source_upper=raw.iter().filter(|&&n|!matches!(aig.node_type(n),AigNode::Output(_))).count()+inversions.len();
 }
 let compact_primitives=shared_majority && raw_source_upper>node_allowance;
 let mut original_triples=if shared_majority {Some(triples.clone())}else{None};
 let mut majority_folds=0usize;let mut source_choices=HashMap::<_,Vec<(Vec<petgraph::graph::NodeIndex>,u64)>>::new();let mut source_depth=HashMap::new();let mut source_truth_queries=0usize;let mut source_truth_queries_max=0usize;let mut library_queries=HashMap::<(usize,u64),bool>::new();
 if shared_majority && (compact_primitives || raw_source_upper<initial_node_ceiling(node_allowance)) {
  // Reuse the existing16-state source banks; support joins choose at most16 complete truth queries.
  // This plans a representation, not an extra retained physical-cut bank.
  let majority_tables:BTreeSet<u64>=(0..8).map(|phase|(0u32..8).fold(0u64,|tt,a|tt|(u64::from((a^phase).count_ones()>=2)<<a))).collect();
  let mut majority_tables=majority_tables;
  if compact_primitives {for phase in 0..8 {for neg in [false,true] {for last in 0..3 {
   let others:Vec<_>=(0..3).filter(|p|*p!=last).collect();
   majority_tables.insert((0..8).fold(0u64,|tt,a|{let v=a^phase;let bit=(((v>>others[0])&1!=0)&&((v>>others[1])&1!=0))||((v>>last)&1!=0);tt|(u64::from(bit^neg)<<a)}));
  }majority_tables.insert((0..8).fold(0u64,|tt,a|tt|(u64::from(((a^phase)==7)^neg)<<a)));}}}

  let mut topo=Topo::new(g);let depth=&mut source_depth;let mut query_rank=vec![0usize;g.node_indices().map(|n|n.index()+1).max().unwrap_or(0)];
  while let Some(n)=topo.next(g){
   let level=g.edges_directed(n,Direction::Incoming).map(|e|depth[&e.source()]).max().unwrap_or(0usize)+usize::from(matches!(aig.node_type(n),AigNode::And));depth.insert(n,level);query_rank[n.index()]=level;
   if !matches!(aig.node_type(n),AigNode::And)||macros.contains_key(&n){continue}
   let original=pair(n)?;
   let fringe=|node,triples:&HashMap<petgraph::graph::NodeIndex,(Vec<petgraph::graph::NodeIndex>,u64)>,bank:&HashMap<petgraph::graph::NodeIndex,Vec<(Vec<petgraph::graph::NodeIndex>,u64)>>|{
    let mut cuts=vec![(vec![node],2u64)];let mut extra=Vec::new();
    if let Some(v)=pair(node){let leaves:Vec<_>=v.iter().map(|x|x.0).collect();let tt=(0..4).fold(0u64,|tt,a|tt|(u64::from((((a&1)!=0)^v[0].1)&&(((a&2)!=0)^v[1].1))<<a));cuts.push((leaves,tt));
     for which in 0..2 {if let Some(x)=macros.get(&v[which].0){
      let leaves:Vec<_>=x.iter().map(|x|x.0).chain(std::iter::once(v[1-which].0)).collect::<BTreeSet<_>>().into_iter().collect();if leaves.len()>3{continue}
      let tt=(0..1usize<<leaves.len()).fold(0u64,|tt,bits|{let get=|n|bits&(1<<leaves.binary_search(&n).unwrap())!=0;let parity=(get(x[0].0)^get(x[1].0))^(x[0].1^x[1].1);let other=get(v[1-which].0)^v[1-which].1;tt|(u64::from((parity^v[which].1)&other)<<bits)});extra.push((leaves,tt));
     }}
    }
    if let Some(cut)=triples.get(&node){extra.push(cut.clone());}
    if let Some(v)=macros.get(&node){extra.push((v.iter().map(|x|x.0).collect(),if v[0].1^v[1].1{9}else{6}));}
    if let Some(saved)=bank.get(&node){extra.extend(saved.iter().cloned());}
    extra.sort_by_key(|(leaves,tt)|(leaves.iter().map(|n|depth[n]).sum::<usize>(),leaves.clone(),*tt));extra.dedup();cuts.extend(extra.into_iter().take(16-cuts.len()));
    cuts
   };
   let left=fringe(original[0].0,&triples,&source_choices);let right=fringe(original[1].0,&triples,&source_choices);let mut best=None;
   let left_support:Vec<_>=left.iter().map(|(leaves,_)|leaves.iter().map(|n|n.index()).collect()).collect();let right_support:Vec<_>=right.iter().map(|(leaves,_)|leaves.iter().map(|n|n.index()).collect()).collect();
   let query_pairs=crate::egraph_source_plan::source_pairs(&left_support,&right_support,&query_rank,source_width,16);source_truth_queries+=query_pairs.len();source_truth_queries_max=source_truth_queries_max.max(query_pairs.len());
   for (left_index,right_index) in query_pairs {let (a,at)=&left[left_index];let (b,bt)=&right[right_index];
    let leaves:Vec<_>=a.iter().chain(b).copied().collect::<BTreeSet<_>>().into_iter().collect();if leaves.len()<3||leaves.len()>source_width{continue}
    let value=|bits:usize,cut:&Vec<petgraph::graph::NodeIndex>,tt:u64|{let row=cut.iter().enumerate().map(|(p,x)|((bits>>leaves.binary_search(x).unwrap())&1)<<p).sum::<usize>();(tt>>row)&1!=0};
    let tt=(0..1usize<<leaves.len()).fold(0u64,|tt,bits|tt|(u64::from((value(bits,a,*at)^original[0].1)&(value(bits,b,*bt)^original[1].1))<<bits));
    let supported=if let Some(tech)=source_tech {let matched=*library_queries.entry((leaves.len(),tt)).or_insert_with(||tech.candidates(leaves.len() as u8,tt9_from_u64(tt)).is_some_and(|v|!v.is_empty()));matched && (compact_primitives || source_width==4 || (leaves.len()==3 && majority_tables.contains(&tt)))}else{leaves.len()==3 && majority_tables.contains(&tt)};if !supported{continue}
    source_choices.entry(n).or_default().push((leaves.clone(),tt));
    let score=if !compact_primitives {if let Some(tech)=source_tech {
      let unit=tech.and2_delay.iter().copied().fold(0.,f32::max) as f64;
      tech.candidates(leaves.len() as u8,tt9_from_u64(tt)).into_iter().flatten().map(|v|leaves.iter().enumerate().map(|(p,n)|depth[n] as f64*unit+f64::from(v.pin_delay[p])+if v.neg_mask&(1<<p)!=0{f64::from(tech.inv_delay)}else{0.}).fold(0.,f64::max)+if v.out_neg{f64::from(tech.inv_delay)}else{0.}).fold(f64::INFINITY,f64::min)
    }else{leaves.iter().map(|n|depth[n]).sum::<usize>() as f64}}else{leaves.iter().map(|n|depth[n]).sum::<usize>() as f64};
    let rank=(score,leaves.clone(),tt);
    if best.as_ref().is_none_or(|old:&(f64,Vec<petgraph::graph::NodeIndex>,u64)|rank.0.total_cmp(&old.0).then(rank.1.cmp(&old.1)).then(rank.2.cmp(&old.2)).is_lt()){best=Some(rank);}
   }
   if let Some((_,leaves,tt))=best {if triples.get(&n)!=Some(&(leaves.clone(),tt)){majority_folds+=1;triples.insert(n,(leaves,tt));}}
  }
 }
 let mut source_plan_before=0usize;let mut source_plan_after=0usize;let mut source_plan_swaps=0usize;
 if compact_primitives {
  let original=original_triples.take().unwrap();let size=g.node_indices().map(|n|n.index()+1).max().unwrap_or(0);
  let mut parents=vec![Vec::new();size];let mut roots=Vec::new();let mut rank=vec![0usize;size];
  for n in g.node_indices(){rank[n.index()]=source_depth.get(&n).copied().unwrap_or(0);match aig.node_type(n){
   AigNode::Input(_)|AigNode::Zero=>roots.push(n.index()),
   AigNode::Output(_)=>{let e=g.edges_directed(n,Direction::Incoming).next()?;roots.push(e.source().index());},
   AigNode::And=>{parents[n.index()]=if let Some((leaves,_))=original.get(&n){leaves.iter().map(|n|n.index()).collect()}else if let Some(v)=macros.get(&n){v.iter().map(|x|x.0.index()).collect()}else{pair(n)?.iter().map(|x|x.0.index()).collect()};}
  }}
  let mut truth_banks=vec![Vec::new();size];let mut choices=vec![Vec::new();size];
  for (n,mut candidates) in source_choices {
   candidates.sort_by_key(|(leaves,tt)|(leaves.iter().map(|n|source_depth[n]).sum::<usize>(),leaves.clone(),*tt));candidates.dedup();candidates.truncate(15);
   choices[n.index()]=candidates.iter().map(|(leaves,_)|leaves.iter().map(|n|n.index()).collect()).collect();truth_banks[n.index()]=candidates;
  }
  let mut topo=Topo::new(g);let mut order=Vec::new();while let Some(n)=topo.next(g){order.push(n.index());}order.reverse();
  let (selected,before,after)=crate::egraph_source_plan::select(&parents,&roots,&choices,&rank,&order);source_plan_before=before;source_plan_after=after;
  triples=original;
  for (n,choice) in selected.into_iter().enumerate(){if let Some(i)=choice{triples.insert(petgraph::graph::NodeIndex::new(n),truth_banks[n][i].clone());source_plan_swaps+=1;}}
 }
 let mut alternative_plan=Vec::new();
 if let Some(original_triples)=original_triples {
  for (n,cut) in &triples {if original_triples.get(n)!=Some(cut){alternative_plan.push((*n,cut.clone()));}}
  let maximum=g.node_indices().filter(|&n|matches!(aig.node_type(n),AigNode::Output(_))).map(|n|source_depth.get(&n).copied().unwrap_or(0)).max().unwrap_or(0);
  let mut required=HashMap::new();let mut topo=Topo::new(g);let mut order=Vec::new();while let Some(n)=topo.next(g){if matches!(aig.node_type(n),AigNode::Output(_)){required.insert(n,maximum);}order.push(n);}
  for n in order.into_iter().rev(){if let Some(&r)=required.get(&n){let child_r=r.saturating_sub(usize::from(matches!(aig.node_type(n),AigNode::And)));for e in g.edges_directed(n,Direction::Incoming){required.entry(e.source()).and_modify(|v|*v=(*v).min(child_r)).or_insert(child_r);}}}
  alternative_plan.sort_by_key(|(n,_)|(required.get(n).copied().unwrap_or(usize::MAX).saturating_sub(source_depth.get(n).copied().unwrap_or(0)),std::cmp::Reverse(source_depth.get(n).copied().unwrap_or(0)),*n));triples=original_triples;
 }
 let mut needed=HashSet::new();let mut stack:Vec<_>=g.node_indices().filter(|&n|matches!(aig.node_type(n),AigNode::Input(_)|AigNode::Zero|AigNode::Output(_))).collect();
 while let Some(n)=stack.pop(){if !needed.insert(n){continue}if let Some((leaves,_))=triples.get(&n){stack.extend(leaves);}else if let Some(v)=macros.get(&n){stack.extend(v.iter().map(|(n,_)|*n));}else{stack.extend(g.edges_directed(n,Direction::Incoming).map(|e|e.source()));}}
 let mut helpers=HashSet::new();
 if helper_quota==usize::MAX {for n in g.node_indices(){if matches!(aig.node_type(n),AigNode::And)&&!needed.contains(&n){helpers.insert(n);needed.insert(n);}}}
 else if helper_quota>0 {
  let mut topo=Topo::new(g);let mut order=Vec::new();let mut depth=HashMap::<_,usize>::new();while let Some(n)=topo.next(g){let d=if matches!(aig.node_type(n),AigNode::And){let v=pair(n)?;depth[&v[0].0].max(depth[&v[1].0])+1usize}else if matches!(aig.node_type(n),AigNode::Output(_)){let e=g.edges_directed(n,Direction::Incoming).next()?;depth[&e.source()]}else{0};depth.insert(n,d);order.push(n);}
  let maximum=order.iter().filter(|&&n|matches!(aig.node_type(n),AigNode::Output(_))).map(|n|depth[n]).max().unwrap_or(0);let mut required=HashMap::<_,usize>::new();for &n in &order{if matches!(aig.node_type(n),AigNode::Output(_)){required.insert(n,maximum);}}
  for &n in order.iter().rev(){if let Some(&r)=required.get(&n){let child_r=r.saturating_sub(usize::from(matches!(aig.node_type(n),AigNode::And)));for e in g.edges_directed(n,Direction::Incoming){required.entry(e.source()).and_modify(|v|*v=(*v).min(child_r)).or_insert(child_r);}}}
  let mut groups=Vec::new();for &n in &needed{if !matches!(aig.node_type(n),AigNode::And)||(!macros.contains_key(&n)&&!triples.contains_key(&n)){continue}let original=pair(n)?;let missing:Vec<_>=original.iter().map(|v|v.0).filter(|n|!needed.contains(n)).collect::<BTreeSet<_>>().into_iter().collect();if missing.is_empty(){continue}if missing.iter().all(|n|matches!(aig.node_type(*n),AigNode::And)&&pair(*n).is_some_and(|v|v.iter().all(|(c,_)|needed.contains(c)))){groups.push((required.get(&n).copied().unwrap_or(usize::MAX).saturating_sub(depth[&n]),std::cmp::Reverse(depth[&n]),n,missing));}}
  groups.sort_by_key(|(slack,d,n,_)|(*slack,*d,*n));let output_not:HashSet<_>=g.node_indices().filter(|&n|matches!(aig.node_type(n),AigNode::Output(_))).flat_map(|n|g.edges_directed(n,Direction::Incoming)).filter(|e|*e.weight()).map(|e|e.source()).collect();let base=needed.iter().filter(|&&n|!matches!(aig.node_type(n),AigNode::Output(_))).count()+output_not.len();let ceiling=initial_node_ceiling(node_allowance);
  for (_,_,_,missing) in groups{let novel:Vec<_>=missing.into_iter().filter(|n|!needed.contains(n)).collect();if helpers.len()+novel.len()>helper_quota||base+helpers.len()+novel.len()>ceiling{continue}for n in novel{needed.insert(n);helpers.insert(n);}}
 }
 let planned_not:HashSet<_>=g.node_indices().filter(|&n|matches!(aig.node_type(n),AigNode::Output(_))).flat_map(|n|g.edges_directed(n,Direction::Incoming)).filter(|e|*e.weight()).map(|e|e.source()).collect();
 let planned_folded_source_upper=needed.iter().filter(|&&n|!matches!(aig.node_type(n),AigNode::Output(_))).count()+planned_not.len();
 let mut graph=egg::EGraph::default();let mut ids=HashMap::new();let mut topo=Topo::new(g);let mut actual_xor=0;let mut actual_tt3=0;let mut actual_tt4=0;
 while let Some(n)=topo.next(g){if !needed.contains(&n){continue}let node=match aig.node_type(n){
 AigNode::Zero=>AigLang::Zero,AigNode::Input(v)=>AigLang::Pi(v),AigNode::Output(_)=>continue,
 AigNode::And=>{if let Some((leaves,tt))=triples.get(&n).filter(|_|!helpers.contains(&n)){if leaves.len()==4{actual_tt4+=1}else{actual_tt3+=1};source_table(*tt,leaves.iter().map(|n|ids[n]).collect())}else if let Some(v)=macros.get(&n).filter(|_|!helpers.contains(&n)){actual_xor+=1;let odd=v[0].1^v[1].1;AigLang::Cell(if odd{"@egm_xnor"}else{"@egm_xor"}.into(),v.iter().map(|(n,_)|ids[n]).collect())}else{let v=pair(n)?;let x=ids[&v[0].0];let y=ids[&v[1].0];match(v[0].1,v[1].1){(false,false)=>AigLang::And([x,y]),(true,false)=>AigLang::Cell("@egm_and_np".into(),vec![x,y]),(false,true)=>AigLang::Cell("@egm_and_pn".into(),vec![x,y]),(true,true)=>AigLang::Cell("@egm_and_nn".into(),vec![x,y])}}}};
 ids.insert(n,admission.import(&mut graph,node,"library-source")?);}
 let initial_base_enodes=graph.total_number_of_nodes();let mut alternative_enodes=0usize;let mut alternatives=0usize;let mut skipped_alternatives=0usize;
 let po_reserve:HashSet<_>=g.node_indices().filter(|&n|matches!(aig.node_type(n),AigNode::Output(_))).flat_map(|n|g.edges_directed(n,Direction::Incoming)).filter(|e|*e.weight()).map(|e|e.source()).collect();
 for (root,(leaves,tt)) in alternative_plan {
  if !ids.contains_key(&root)||leaves.iter().any(|n|!ids.contains_key(n)){skipped_alternatives+=1;continue}
  if initial_base_enodes.saturating_add(alternative_enodes).saturating_add(po_reserve.len()).saturating_add(1)>initial_node_ceiling(node_allowance){skipped_alternatives+=1;continue}
  let node=source_table(tt,leaves.iter().map(|n|ids[n]).collect());let before=graph.total_number_of_nodes();let Some(new)=admission.add(&mut graph,node) else {skipped_alternatives+=1;continue};alternative_enodes+=graph.total_number_of_nodes()-before;graph.union(ids[&root],new);alternatives+=1;
 }
 let mut roots=Vec::new();for n in g.node_indices(){if let AigNode::Output(pos)=aig.node_type(n){let e=g.edges_directed(n,Direction::Incoming).next()?;let base=ids[&e.source()];let root=if *e.weight(){let node=AigLang::Not(base);admission.import(&mut graph,node,"source-output")?}else{base};roots.push((pos,root));}}
 let pre_rebuild_import_enodes=graph.total_number_of_nodes();
 roots.sort_by_key(|(p,_)|*p);graph.rebuild();let roots=roots.into_iter().map(|(p,id)|(p,graph.find(id))).collect();
 let mut source=Vec::new();for n in g.node_indices(){if !ids.contains_key(&n)||helpers.contains(&n)||(!macros.contains_key(&n)&&!triples.contains_key(&n)){continue}let inputs=pair(n)?;if let(Some(&a),Some(&b))=(ids.get(&inputs[0].0),ids.get(&inputs[1].0)){source.push(SourceAnd{root:graph.find(ids[&n]),left:graph.find(a),right:graph.find(b),left_neg:inputs[0].1,right_neg:inputs[1].1});}}

 let helper_classes:BTreeSet<_>=helpers.iter().map(|n|graph.find(ids[n])).collect();
 let unique_xor=graph.classes().flat_map(|c|&c.nodes).filter(|n|crate::egraph_sat::xor_inputs(n).is_some()).count();
 let unique_tt3=graph.classes().flat_map(|c|&c.nodes).filter(|n|crate::egraph_sat::tt3_inputs(n).is_some_and(|(a,_)|a.len()==3)).count();
 let unique_tt4=graph.classes().flat_map(|c|&c.nodes).filter(|n|crate::egraph_sat::tt3_inputs(n).is_some_and(|(a,_)|a.len()==4)).count();
 if let Some(path)=std::env::var_os("EGM_POOL_DIAG_PATH"){let path=std::path::Path::new(&path).with_extension("import.json");let mut audit=serde_json::json!({"schema_version":5,"source_plan_before":source_plan_before,"source_plan_after":source_plan_after,"source_plan_swaps":source_plan_swaps,"source_planning_choices_per_node":16,"source_planning_has_no_egraph_retirement":true,"import_complete":true,"raw_source_upper":raw_source_upper,"planned_folded_source_upper":planned_folded_source_upper,"compaction_threshold":node_allowance,"compact_primitives":compact_primitives,"source_fringe_cap":16,"source_support_pair_checks_per_node_limit":256,"source_truth_queries_per_node_limit":16,"source_truth_queries_per_node_actual_max":source_truth_queries_max,"source_truth_queries_total":source_truth_queries,"majority_alternatives":alternatives,"alternative_enodes_admitted":alternative_enodes,"skipped_alternatives":skipped_alternatives,"pre_rebuild_import_enodes":pre_rebuild_import_enodes,"shared_majority_enabled":shared_majority,"source_primitive_folds":majority_folds,"legacy_majority_fold_counter_is_primitive_fold_alias":true,"shared_majority_folds":majority_folds,"actual_imported_enodes":graph.total_number_of_nodes(),"source_xor_occurrences":actual_xor,"source_tt3_occurrences":actual_tt3,"actual_xor_enodes":unique_xor,"actual_tt3_enodes":unique_tt3,"retained_xor_operators":unique_xor,"retained_tt3_operators":unique_tt3,"all_original_helpers_retained":helper_quota==usize::MAX,"helper_quota":helper_quota,"selected_source_helpers":helpers.len(),"actual_helpers":helper_classes.len(),"helper_node_reserve_fraction":0.1,"source_physical_witnesses":source.len(),"original_source_ands":g.node_indices().filter(|&n|matches!(aig.node_type(n),AigNode::And)).count(),"representation":if shared_majority{"Exact XOR plus bounded scalar TT3/TT4 operators admitted under original library K3/K4; full truth tables and operands are stored. All retained helpers and alternatives count; no hidden functions or egraph retirement/reset. This richer node unit is not expanded-AND work units."}else{"Exact pre-import XOR and unshared two-AND TT3 contraction; all retained operands/helpers count. No regional reset; richer node unit is not expanded-AND work units."}});audit.as_object_mut().unwrap().extend(serde_json::json!({"source_arity_limit":source_width,"library_truth_filter":source_tech.is_some(),"optional_rhs_ranking":"structural slack, estimated dependent-pin arrival; no PPA winner dispatch","source_tt4_occurrences":actual_tt4,"actual_tt4_enodes":unique_tt4}).as_object().unwrap().clone());let _=std::fs::write(path,audit.to_string());}
 Some((graph,roots,source))
}

fn import_prefix_failure(allowance:usize,source_upper:usize,planned:usize,admitted:usize){
 eprintln!("budget: cumulative import would exceed node allowance {allowance}; {admitted} admitted, complete input not imported");
 if let Some(p)=std::env::var_os("EGM_POOL_DIAG_PATH"){let path=std::path::Path::new(&p).with_extension("import.json");let _=std::fs::write(path,serde_json::json!({"schema_version":5,"source_planning_has_no_egraph_retirement":true,"import_complete":false,"cumulative_import_admitted_enodes":admitted,"raw_source_upper":source_upper,"planned_folded_source_upper":planned,"node_allowance":allowance,"error":"additional distinct enode rejected before admission; incomplete import is not a legal cover"}).to_string());}
}

pub(super) fn compact_node_upper(aig:&Aig)->Option<usize>{
 use crate::aig_graph::AigNode;use petgraph::visit::EdgeRef;use petgraph::Direction;
 let g=aig.graph();let mut live=HashSet::new();let mut stack:Vec<_>=g.node_indices().filter(|&n|matches!(aig.node_type(n),AigNode::Input(_)|AigNode::Zero|AigNode::Output(_))).collect();
 while let Some(n)=stack.pop(){if live.insert(n){stack.extend(g.edges_directed(n,Direction::Incoming).map(|e|e.source()));}}
 let output_not:HashSet<_>=g.node_indices().filter(|&n|matches!(aig.node_type(n),AigNode::Output(_))).flat_map(|n|g.edges_directed(n,Direction::Incoming)).filter(|e|*e.weight()).map(|e|e.source()).collect();
 Some(live.iter().filter(|&&n|!matches!(aig.node_type(n),AigNode::Output(_))).count()+output_not.len())
}
