//! Plan source cuts before constructing an egraph. Never retire admitted egraph state.
use std::collections::{BTreeSet,HashMap};
#[cfg(test)] fn reachable(parents:&[Vec<usize>],roots:&[usize])->usize {let mut todo=roots.to_vec();let mut seen=BTreeSet::new();while let Some(n)=todo.pop(){if seen.insert(n){todo.extend(&parents[n]);}}seen.len()}
fn unique(v:&[usize])->Vec<usize>{v.iter().copied().collect::<BTreeSet<_>>().into_iter().collect()}
fn transaction(parents:&[Vec<usize>],refs:&[usize],old:&[usize],new:&[usize])->(isize,HashMap<usize,usize>){
 transaction_from(parents,refs,old,new,HashMap::new())
}
fn transaction_from(parents:&[Vec<usize>],refs:&[usize],old:&[usize],new:&[usize],mut changes:HashMap<usize,usize>)->(isize,HashMap<usize,usize>){
 let mut delta=0isize;
 for (inputs,add) in [(new,true),(old,false)] {
  let mut todo:Vec<_>=unique(inputs);
  while let Some(n)=todo.pop(){let before=changes.get(&n).copied().unwrap_or(refs[n]);
   let after=if add{before+1}else{before.checked_sub(1).expect("source-plan reference underflow")};changes.insert(n,after);
   if add&&before==0{delta+=1;todo.extend(&parents[n]);}
   if !add&&after==0{delta-=1;todo.extend(&parents[n]);}
  }
 }
 (delta,changes)
}
pub(crate) fn select(parents:&[Vec<usize>],roots:&[usize],choices:&[Vec<Vec<usize>>],rank:&[usize],order:&[usize])->(Vec<Option<usize>>,usize,usize){
 select_with_pairs(parents,roots,choices,rank,order,std::env::var("EGM_JOINT_SOURCE").as_deref()==Ok("1"))
}
fn select_with_pairs(parents:&[Vec<usize>],roots:&[usize],choices:&[Vec<Vec<usize>>],rank:&[usize],order:&[usize],paired:bool)->(Vec<Option<usize>>,usize,usize){
 select_with_zero_pairs(parents,roots,choices,rank,order,paired,std::env::var("EGM_ZERO_COST_SOURCE").as_deref()==Ok("1"))
}
fn select_with_zero_pairs(parents:&[Vec<usize>],roots:&[usize],choices:&[Vec<Vec<usize>>],rank:&[usize],order:&[usize],paired:bool,allow_zero:bool)->(Vec<Option<usize>>,usize,usize){
 select_with_groups(parents,roots,choices,rank,order,paired,allow_zero,std::env::var("EGM_SOURCE_GROUPS").as_deref()==Ok("1"))
}
fn select_with_groups(parents:&[Vec<usize>],roots:&[usize],choices:&[Vec<Vec<usize>>],rank:&[usize],order:&[usize],paired:bool,allow_zero:bool,groups:bool)->(Vec<Option<usize>>,usize,usize){
 select_with_closure_filter(parents,roots,choices,rank,order,paired,allow_zero,groups,std::env::var("EGM_SOURCE_CLOSURE_FILTER").as_deref()==Ok("1"))
}
// Conservative bounded support inspection, never an extra Boolean query.
// A truncated walk is treated as still dependent, preserving the old source.
#[derive(Default)]
struct DependencyWalk { dependent:bool, expanded:usize, rank_pruned:usize, truncated:bool }
fn depends_on_source(parents:&[Vec<usize>],inputs:&[usize],shared:usize)->bool {
 dependency_walk(parents,inputs,shared,None).dependent
}
// `rank` is supplied only after verifying strict decrease on EVERY original edge.
// All accepted source choices already satisfy the same strict decrease.
fn dependency_walk(parents:&[Vec<usize>],inputs:&[usize],shared:usize,rank:Option<&[usize]>)->DependencyWalk {
 let mut stack=inputs.to_vec();let mut seen=BTreeSet::new();let mut result=DependencyWalk::default();
 while let Some(n)=stack.pop(){
  if n==shared{result.dependent=true;return result;}
  if rank.is_some_and(|r|r[n]<=r[shared]){result.rank_pruned+=1;continue;}
  if seen.insert(n){if result.expanded==256{result.dependent=true;result.truncated=true;return result;}
   result.expanded+=1;stack.extend(&parents[n]);
  }
 }
 result
}
fn select_with_closure_filter(parents:&[Vec<usize>],roots:&[usize],choices:&[Vec<Vec<usize>>],rank:&[usize],order:&[usize],paired:bool,allow_zero:bool,groups:bool,closure_filter:bool)->(Vec<Option<usize>>,usize,usize){
 select_with_terminal(parents,roots,choices,rank,order,paired,allow_zero,groups,closure_filter,std::env::var("EGM_SOURCE_TERMINAL_GROUPS").as_deref()==Ok("1"))
}
fn select_with_terminal(parents:&[Vec<usize>],roots:&[usize],choices:&[Vec<Vec<usize>>],rank:&[usize],order:&[usize],paired:bool,allow_zero:bool,groups:bool,closure_filter:bool,terminal:bool)->(Vec<Option<usize>>,usize,usize){
 let rank_pruning_requested=std::env::var("EGM_SOURCE_RANK_PRUNING").as_deref()==Ok("1");
 let rank_invariant=rank.len()==parents.len() && parents.iter().enumerate().all(|(n,p)|p.iter().all(|&i|i<rank.len() && rank[i]<rank[n]));
 let ranked=if rank_pruning_requested && rank_invariant {Some(rank)}else{None};
 let(mut dependency_walks,mut dependency_expansions,mut rank_pruned,mut dependency_unknown)=(0usize,0usize,0usize,0usize);
 let mut parents:Vec<_>=parents.iter().map(|p|unique(p)).collect();let mut refs=vec![0usize;parents.len()];let mut before=0usize;
 for &root in roots {let mut todo=vec![root];while let Some(n)=todo.pop(){refs[n]+=1;if refs[n]==1{before+=1;todo.extend(&parents[n]);}}}
 let mut count=before;let mut selected=vec![None;parents.len()];
 for &n in order {if refs[n]==0||choices[n].is_empty(){continue}
  let old_rank=parents[n].iter().map(|&p|rank[p]).sum::<usize>();let mut best_key=(0isize,old_rank,parents[n].clone());let mut winner=None;
  for (i,inputs) in choices[n].iter().enumerate(){let inputs=unique(inputs);
   if inputs==parents[n]||inputs.iter().any(|&p|rank[p]>=rank[n]){continue}
   let (delta,changes)=transaction(&parents,&refs,&parents[n],&inputs);if delta>0{continue}
   let key=(delta,inputs.iter().map(|&p|rank[p]).sum::<usize>(),inputs.clone());
   if key<best_key{best_key=key;winner=Some((i,inputs,changes));}
  }
  if let Some((i,inputs,changes))=winner{count=(count as isize+best_key.0) as usize;for(p,r)in changes{refs[p]=r;}parents[n]=inputs;selected[n]=Some(i);}
 }
 let mut group_trials=0usize;let mut accepted_groups=0usize;
 let mut pair_trials=0;let mut accepted_pairs=0;let single_count=count;
 let mut used_pair_trials=vec![0usize;parents.len()];let mut zero_trials=0usize;let mut zero_accepted=0usize;let mut after_positive=count;
 if paired {
 for zero_mode in [false,true]{if zero_mode&&!allow_zero{break;}
 // Joint source changes use only existing truth-bank choices. Limit16 TOTAL
 // paired transactions per root; no new truth query, cut capacity or node allowance.
 let mut users=vec![Vec::new();parents.len()];for(n,inputs)in parents.iter().enumerate(){if refs[n]>0{for &p in inputs{users[p].push(n);}}}
 for &n in order{if refs[n]==0||choices[n].is_empty(){continue}let mut partners=BTreeSet::new();for &p in &parents[n]{for &m in &users[p]{if m>n&&refs[m]>0&&rank[m]==rank[n]&&!choices[m].is_empty()&&parents[m].contains(&p){partners.insert(m);}}}
  // Reallocate the SAME sixteen total transactions per root, not truth calls.
  // A bounded same-rank dependency closure can retire a cone that no pair frees.
  if groups && !zero_mode {
   for shared in parents[n].clone() {
    if used_pair_trials[n]>=16 {break;}
    let closure:Vec<_>=std::iter::once(n).chain(users[shared].iter().copied().filter(|&m|m>n && refs[m]>0 && rank[m]==rank[n] && parents[m].contains(&shared))).collect::<BTreeSet<_>>().into_iter().take(4).collect();
    if closure.len()<3 {continue;}
    let proposals:Vec<_>=closure.iter().filter_map(|&m|choices[m].iter().enumerate().map(|(i,p)|(i,unique(p))).find(|(_,p)|*p!=parents[m] && p.iter().all(|&p|rank[p]<rank[m]) && (!closure_filter || {let w=dependency_walk(&parents,p,shared,ranked);dependency_walks+=1;dependency_expansions+=w.expanded;rank_pruned+=w.rank_pruned;dependency_unknown+=usize::from(w.truncated);!w.dependent})).map(|(i,p)|(m,i,p))).collect();
    if proposals.len()!=closure.len(){continue;}
    used_pair_trials[n]+=1;pair_trials+=1;group_trials+=1;
    let mut changes=HashMap::new();let mut delta=0;
    for (m,_,inputs) in &proposals {let(d,c)=transaction_from(&parents,&refs,&parents[*m],inputs,changes);delta+=d;changes=c;}
    if delta<0 {
     count=(count as isize+delta)as usize;for(p,r)in changes{refs[p]=r;}
     for(m,i,inputs)in proposals{parents[m]=inputs;selected[m]=Some(i);for &p in &parents[m]{users[p].push(m);}}
     accepted_groups+=1;break;
    }
   }
  }
  let mut tried=used_pair_trials[n];let mut winner=None;let mut best_delta=0;
  'pairs:for(a,inputs)in choices[n].iter().enumerate(){let inputs=unique(inputs);if inputs==parents[n]||inputs.iter().any(|&p|rank[p]>=rank[n]){continue}
   let(d1,c1)=transaction(&parents,&refs,&parents[n],&inputs);if if zero_mode{d1!=0}else{d1<=0}{continue}
   for &m in &partners{for(b,other)in choices[m].iter().enumerate(){let other=unique(other);if other==parents[m]||other.iter().any(|&p|rank[p]>=rank[m]){continue}if tried==16{break 'pairs}tried+=1;used_pair_trials[n]=tried;pair_trials+=1;if zero_mode{zero_trials+=1;}
    // Same-rank roots cannot occur inside one another's strictly lower-rank
    // replacement cones, so shared descendant refs can be transacted in sequence.
    let(d2,c2)=transaction_from(&parents,&refs,&parents[m],&other,c1.clone());let delta=d1+d2;
    if delta<best_delta{best_delta=delta;winner=Some((a,inputs.clone(),m,b,other,c2));}
   }}
  }
  if let Some((a,inputs,m,b,other,changes))=winner{accepted_pairs+=1;if zero_mode{zero_accepted+=1;}count=(count as isize+best_delta)as usize;for(p,r)in changes{refs[p]=r;}parents[n]=inputs;parents[m]=other;selected[n]=Some(a);selected[m]=Some(b);for &p in &parents[n]{users[p].push(n);}for &p in &parents[m]{users[p].push(m);}}
 }
 if !zero_mode{after_positive=count;}
 }
 debug_assert!(count<=after_positive);
 }
 // Preserve the COMPLETE old selected source plan. Spend only transactions
 // that the original group/pair/zero passes left unused at each root.
 let before_terminal=count;let(mut terminal_trials,mut terminal_accepted)=(0usize,0usize);
 if terminal && paired && groups {
  let mut users=vec![Vec::new();parents.len()];
  for(n,inputs)in parents.iter().enumerate(){if refs[n]>0{for &p in inputs{users[p].push(n);}}}
  for &n in order {
   if refs[n]==0 || choices[n].is_empty(){continue;}
   for shared in parents[n].clone(){
    if used_pair_trials[n]>=16{break;}
    let closure:Vec<_>=std::iter::once(n).chain(users[shared].iter().copied().filter(|&m|m>n && refs[m]>0 && rank[m]==rank[n] && parents[m].contains(&shared))).collect::<BTreeSet<_>>().into_iter().take(4).collect();
    if closure.len()<3{continue;}
    let proposals:Vec<_>=closure.iter().filter_map(|&m|choices[m].iter().enumerate().map(|(i,p)|(i,unique(p))).find(|(_,p)|*p!=parents[m] && p.iter().all(|&p|rank[p]<rank[m]) && {let w=dependency_walk(&parents,p,shared,ranked);dependency_walks+=1;dependency_expansions+=w.expanded;rank_pruned+=w.rank_pruned;dependency_unknown+=usize::from(w.truncated);!w.dependent}).map(|(i,p)|(m,i,p))).collect();
    if proposals.len()!=closure.len(){continue;}
    used_pair_trials[n]+=1;pair_trials+=1;group_trials+=1;terminal_trials+=1;
    let mut changes=HashMap::new();let mut delta=0;
    for (m,_,inputs) in &proposals{let(d,c)=transaction_from(&parents,&refs,&parents[*m],inputs,changes);delta+=d;changes=c;}
    if delta<0{
     count=(count as isize+delta)as usize;for(p,r)in changes{refs[p]=r;}
     for(m,i,inputs)in proposals{parents[m]=inputs;selected[m]=Some(i);for &p in &parents[m]{users[p].push(m);}}
     terminal_accepted+=1;break;
    }
   }
  }
 }
 assert!(count<=before_terminal,"terminal source transactions must preserve the full old source count");
 if let Some(p)=std::env::var_os("EGM_POOL_DIAG_PATH"){let _=std::fs::write(std::path::Path::new(&p).with_extension("joint-source.json"),serde_json::json!({"enabled":paired,"terminal_enabled":terminal,"before_terminal":before_terminal,"terminal_trials":terminal_trials,"terminal_accepted":terminal_accepted,"groups_enabled":groups,"closure_filter":closure_filter,"dependency_scan_cap":256,"rank_pruning_requested":rank_pruning_requested,"rank_invariant":rank_invariant,"rank_pruning_enabled":ranked.is_some(),"dependency_walks":dependency_walks,"dependency_expansions":dependency_expansions,"rank_pruned_branches":rank_pruned,"dependency_unknown":dependency_unknown,"group_trials":group_trials,"accepted_groups":accepted_groups,"max_root_transactions":used_pair_trials.iter().copied().max().unwrap_or(0),"zero_cost_enabled":allow_zero,"zero_trials":zero_trials,"zero_accepted":zero_accepted,"after_positive":after_positive,"before":before,"after_single":single_count,"after_pairs":count,"pair_trials":pair_trials,"accepted_pairs":accepted_pairs,"pair_trial_cap_per_root":16,"scope":"Exact existing source choices only; bounded same-rank roots updated transactionally before global egraph admission. No extra truth queries, hidden functions, egraph retirement or counter reset. Counts are source representation estimates; actual cumulative enodes and physical CEC remain required."}).to_string());}
 debug_assert_eq!(refs.iter().filter(|&&r|r>0).count(),count);debug_assert!(count<=before);(selected,before,count)
}

/// Inspect only supports before choosing at most `limit` complete truth compositions.
/// Both input banks fit the existing16-cut allowance; no extra Boolean query budget.
pub(crate) fn source_pairs(left:&[Vec<usize>],right:&[Vec<usize>],rank:&[usize],support:usize,limit:usize)->Vec<(usize,usize)>{
 if limit==0{return Vec::new()}
 let mut compatible=Vec::new();
 for(i,a)in left.iter().take(16).enumerate(){for(j,b)in right.iter().take(16).enumerate(){let mut leaves=a.clone();leaves.extend(b);leaves.sort_unstable();leaves.dedup();if leaves.len()<3||leaves.len()>support{continue}let depth=leaves.iter().map(|&n|rank[n]).sum::<usize>();compatible.push((depth,leaves,i,j));}}
 compatible.sort_unstable();let mut seen=BTreeSet::new();let mut unique=Vec::new();for entry in compatible{if seen.insert(entry.1.clone()){unique.push(entry);}}
 let mut out=Vec::new();let mut used=BTreeSet::new();
 if support==4 {for width in [3,4] {for (index,(_,leaves,i,j)) in unique.iter().enumerate().filter(|(_,x)|x.1.len()==width).take(limit/2){out.push((*i,*j));used.insert(index);}}}
 for(index,(_,_,i,j))in unique.into_iter().enumerate(){if out.len()>=limit{break}if used.insert(index){out.push((i,j));}}out
}
#[cfg(test)] mod tests {
 use super::*;
 #[test] fn egraph_source_plan_preserves_shared_refs_and_joint_reclamation(){
  let parents=vec![vec![],vec![],vec![0,1],vec![0,1],vec![0,1],vec![0,1],vec![2,3],vec![2,3]];let roots=vec![0,1,6,7];let rank:Vec<_>=(0..8).collect();let order:Vec<_>=(0..8).rev().collect();
  let mut choices=vec![vec![];8];choices[6]=vec![vec![4,5]];let (s,b,a)=select(&parents,&roots,&choices,&rank,&order);assert_eq!((b,a),(6,6));assert_eq!(s[6],None,"do not abandon live shared dependencies and add two private functions");
  choices[6]=vec![vec![0,1]];choices[7]=vec![vec![0,1]];let (s,b,a)=select(&parents,&roots,&choices,&rank,&order);assert_eq!((b,a),(6,4),"zero-cost shared-ref transitions must enable joint reclamation without an intermediate node-count increase");let mut actual=parents.clone();for (n,i) in s.iter().enumerate(){if let Some(i)=i{actual[n]=choices[n][*i].clone();}}assert_eq!(reachable(&actual,&roots),a);
 }
 #[test] fn egraph_source_plan_matches_independent_reachability_on_reconvergent_dags(){
  let mut state=0x12345678u64;for _ in 0..128 {let mut next=|n|{state=state.wrapping_mul(6364136223846793005).wrapping_add(1);(state as usize)%n};let n=24;let mut parents=vec![vec![];n];let mut choices=vec![vec![];n];
   for i in 3..n {parents[i]=vec![next(i),next(i)];for _ in 0..4{choices[i].push(vec![next(i),next(i),next(i)]);}}
   let roots=vec![0,1,2,n-1,n-2];let rank:Vec<_>=(0..n).collect();let order:Vec<_>=(0..n).rev().collect();let (selected,before,after)=select(&parents,&roots,&choices,&rank,&order);assert_eq!(before,reachable(&parents,&roots));let mut actual=parents.clone();for (i,c) in selected.iter().enumerate(){if let Some(c)=c{actual[i]=choices[i][*c].clone();}}assert_eq!(after,reachable(&actual,&roots));assert!(after<=before);
  }
 }
 #[test] fn egraph_source_support_join_uses_existing_bank_without_extra_truth_queries(){
  let left=vec![vec![3,4,5],vec![3,4,6],vec![3,4,7],vec![3,4,8],vec![0,1]];let right=vec![vec![9,10,11],vec![9,10,12],vec![9,10,13],vec![9,10,14],vec![1,2]];let rank:Vec<_>=(0..15).collect();let pairs=source_pairs(&left,&right,&rank,3,16);assert!(pairs.contains(&(4,4)),"a compatible existing source boundary is beyond the arbitrary first-four fringe; use support joins within the same16 truth-query cap");assert!(pairs.len()<=16);assert!(source_pairs(&left,&right,&rank,3,0).is_empty());
 }
 #[test] fn egraph_library_source_query_quota_preserves_small_and_wide_boundaries(){let left=vec![vec![0,1],vec![0,1,2]];let right=vec![vec![2],vec![3]];let rank=vec![0,1,2,3];let p=source_pairs(&left,&right,&rank,4,16);assert!(p.len()<=16);assert!(p.iter().any(|&(a,b)|unique(&[left[a].as_slice(),right[b].as_slice()].concat()).len()==3));assert!(p.iter().any(|&(a,b)|unique(&[left[a].as_slice(),right[b].as_slice()].concat()).len()==4));}
}

#[cfg(test)]mod pair_counterexample{use super::*;
 #[test]fn jointly_reclaim_shared_sources_without_increasing_final_nodes(){
  let p=vec![vec![],vec![],vec![0,1],vec![0,1],vec![0,1],vec![2,3],vec![2,3]];let roots=[0,1,5,6];let rank=[0,0,1,1,1,2,2];let order=[6,5,4,3,2,1,0];let mut choices=vec![vec![];7];choices[5]=vec![vec![4]];choices[6]=vec![vec![4]];
  let(s,b,a)=select_with_pairs(&p,&roots,&choices,&rank,&order,true);assert_eq!(b,6);assert_eq!(a,5,"pair must escape individually +1 changes and reclaim two shared sources");let mut actual=p.clone();for(n,i)in s.iter().enumerate(){if let Some(i)=i{actual[n]=choices[n][*i].clone();}}assert_eq!(reachable(&actual,&roots),a);
 }
}

#[cfg(test)]mod joint_property{use super::*;
 #[test]fn layered_joint_transactions_match_full_independent_reachability(){let mut state=1967u64;for _ in 0..256{let mut rnd=|n|{state=state.wrapping_mul(6364136223846793005).wrapping_add(1);(state as usize)%n};let n=42;let mut p=vec![vec![];n];let mut choices=vec![vec![];n];let mut rank=vec![0;n];for i in 3..n{let level=(i-3)/6+1;rank[i]=level;let bound=3+(level-1)*6;for _ in 0..2{p[i].push(rnd(bound));}for _ in 0..4{choices[i].push(vec![rnd(bound),rnd(bound)]);}}let roots=[0,1,2,36,37,38,39,40,41];let order:Vec<_>=(0..n).rev().collect();let(s,b,a)=select_with_zero_pairs(&p,&roots,&choices,&rank,&order,true,true);let mut actual=p.clone();for(n,i)in s.iter().enumerate(){if let Some(i)=i{actual[n]=choices[n][*i].clone();}}assert_eq!(b,reachable(&p,&roots));assert_eq!(a,reachable(&actual,&roots));assert!(a<=b);}}
}
#[cfg(test)]mod bounded_source_boundary_case{use super::*;
 #[test]fn paired_three_leaf_banks_escape_single_point_local_minimum(){let p=vec![vec![],vec![],vec![],vec![],vec![0,1,2],vec![0,1,3],vec![2,3],vec![4,5],vec![4,5]];let roots=[0,1,2,3,7,8];let rank=[0,0,0,0,1,1,1,2,2];let order=[8,7,6,5,4,3,2,1,0];let mut choices=vec![vec![];9];choices[7]=vec![vec![0,1,6]];choices[8]=vec![vec![0,1,6]];let(_,b,a)=select_with_pairs(&p,&roots,&choices,&rank,&order,true);assert_eq!((b,a),(8,7));}
}

#[cfg(test)]mod zero_cost_pair_test{use super::*;
 #[test]fn zero_cost_rank_worsening_first_move_can_enable_shared_release(){let p=vec![vec![],vec![],vec![],vec![],vec![0,1],vec![0,1],vec![2,3,4],vec![2,3,4]];let roots=[0,1,2,3,5,6,7];let rank=[0,0,0,0,1,2,3,3];let order=[7,6,5,4,3,2,1,0];let mut choices=vec![vec![];8];choices[6]=vec![vec![2,3,5]];choices[7]=vec![vec![2,3,5]];let(_,before,after)=select_with_zero_pairs(&p,&roots,&choices,&rank,&order,true,true);assert_eq!(before,8);assert_eq!(after,7,"equal-cost intermediate is necessary before both users can release source4");}
}

#[cfg(test)]mod group_counterexample{use super::*;
 #[test]fn three_consumer_reclamation_escapes_positive_single_and_pair_moves(){
  let mut p=vec![vec![];10];p[4]=vec![3];p[5]=vec![4];p[6]=vec![5];for n in 7..10{p[n]=vec![6];}
  let roots=vec![7,8,9];let mut choices=vec![vec![];10];for n in 7..10{choices[n]=vec![vec![n-7]];}
  let rank=vec![0,0,0,0,1,2,3,4,4,4];let order=vec![7,8,9,6,5,4,3,2,1,0];
  let(s,b,a)=select_with_groups(&p,&roots,&choices,&rank,&order,true,true,true);assert_eq!(b,7);assert_eq!(a,6,"all three consumers must switch before the shared four-node cone is reclaimed");
  let mut actual=p.clone();for(n,i)in s.iter().enumerate(){if let Some(i)=i{actual[n]=choices[n][*i].clone();}}assert_eq!(reachable(&actual,&roots),a);
 }
}

#[cfg(test)]mod closure_filter_tests {use super::*;
#[test]fn group_generation_skips_choices_that_still_reach_shared_cone(){
 let mut p=vec![vec![];10];p[4]=vec![3];p[5]=vec![4];p[6]=vec![5];for n in 7..10{p[n]=vec![6];}
 let roots=vec![7,8,9];let mut choices=vec![vec![];10];for n in 7..10{choices[n]=vec![vec![6,n-7],vec![n-7]];}
 let rank=vec![0,0,0,0,1,2,3,4,4,4];let order=vec![7,8,9,6,5,4,3,2,1,0];
 let(_,old_b,old_a)=select_with_closure_filter(&p,&roots,&choices,&rank,&order,true,true,true,false);assert_eq!((old_b,old_a),(7,7));
 let(s,b,a)=select_with_closure_filter(&p,&roots,&choices,&rank,&order,true,true,true,true);assert_eq!((b,a),(7,6),"first eligible group choices retain the shared cone; later existing choices must retire it");
 let mut actual=p.clone();for(n,i)in s.iter().enumerate(){if let Some(i)=i{actual[n]=choices[n][*i].clone();}}assert_eq!(reachable(&actual,&roots),a);
}
}

#[cfg(test)]mod closure_dependency_tests{use super::*;
#[test]fn transitive_dependency_and_unknown_fail_closed(){let mut p=vec![vec![];300];for n in 1..300{p[n]=vec![n-1];}assert!(depends_on_source(&p,&[10],3));assert!(!depends_on_source(&p,&[10],20));assert!(depends_on_source(&p,&[299],301),"unknown/truncated walk must reject candidate");}
#[test]fn filtered_groups_match_recomputed_global_reachability(){let mut state=91u64;for _ in 0..128{let mut next=|n|{state=state.wrapping_mul(6364136223846793005).wrapping_add(1);(state as usize)%n};let mut p=vec![vec![];32];let mut choices=vec![vec![];32];for n in 4..32{p[n]=vec![next(n),next(n)];for _ in 0..4{choices[n].push(vec![next(n),next(n),next(n)]);}}let rank:Vec<_>=(0..32).map(|n|if n>=28{32}else{n}).collect();for n in 28..32{p[n].retain(|&x|x<28);choices[n].retain(|c|c.iter().all(|&x|x<28));}let roots=[28,29,30,31];let order:Vec<_>=(0..32).rev().collect();let(s,b,a)=select_with_closure_filter(&p,&roots,&choices,&rank,&order,true,true,true,true);let mut actual=p.clone();for(n,i)in s.iter().enumerate(){if let Some(i)=i{actual[n]=choices[n][*i].clone();}}assert_eq!(reachable(&actual,&roots),a);assert!(a<=b);}}
}

#[cfg(test)] mod ranked_source_regression { use super::*;
 #[test] fn low_rank_unrelated_cone_must_not_exhaust_dependency_scan() {
  let mut p=vec![vec![];303];for n in 1..300{p[n]=vec![n-1];}
  let mut ranks:Vec<_>=(0..303).collect(); ranks[300]=400; ranks[301]=401;ranks[302]=402;
  assert!(!dependency_walk(&p,&[299],300,Some(&ranks)).dependent,"an unrelated cone wholly below the shared rank cannot reach it; bounded DFS currently rejects after256 expansions");
 }
}

#[cfg(test)] mod rank_walk_properties {use super::*;
 #[test]fn equal_rank_distinct_node_is_also_unreachable(){let p=vec![vec![],vec![],vec![0],vec![1]];let r=[0,0,1,1];let w=dependency_walk(&p,&[3],2,Some(&r));assert!(!w.dependent);assert_eq!(w.expanded,0);assert_eq!(w.rank_pruned,1);assert!(dependency_walk(&p,&[2],2,Some(&r)).dependent);}
 #[test]fn rank_is_not_node_index_and_matches_exhaustive_dependencies(){
  let mut state=31u64;for _ in 0..256{let mut p=vec![vec![];80];let permutation:Vec<_>=(0..80).map(|i|(i*37)%80).collect();let mut rank=vec![0;80];
   for depth in 0..80{let n=permutation[depth];rank[n]=depth;for _ in 0..2{if depth>0{state=state.wrapping_mul(6364136223846793005).wrapping_add(1);p[n].push(permutation[(state as usize)%depth]);}}}
   for shared in 0..80{let input=permutation[(shared*17)%80];let mut stack=vec![input];let mut seen=BTreeSet::new();while let Some(n)=stack.pop(){if seen.insert(n){stack.extend(&p[n]);}}
    let w=dependency_walk(&p,&[input],shared,Some(&rank));assert_eq!(w.dependent,seen.contains(&shared));assert!(w.expanded<=256);assert!(!w.truncated);
   }
  }
 }
 #[test]fn higher_rank_unknown_still_fails_closed(){let mut p=vec![vec![];303];let mut r=vec![0;303];for n in 1..300{p[n]=vec![n-1];r[n]=n+10;}r[300]=1;let w=dependency_walk(&p,&[299],300,Some(&r));assert!(w.dependent && w.truncated);assert_eq!(w.expanded,256);}
}

#[cfg(test)] mod terminal_source_tests {use super::*;
 #[test] fn terminal_groups_use_remaining_transactions_and_preserve_old_plan(){
  let mut p=vec![vec![];10];p[4]=vec![3];p[5]=vec![4];p[6]=vec![5];for n in 7..10{p[n]=vec![6];}
  let roots=[7,8,9];let mut choices=vec![vec![];10];for n in 7..10{choices[n]=vec![vec![6,n-7],vec![n-7]];}
  let rank=[0,0,0,0,1,2,3,4,4,4];let order=[7,8,9,6,5,4,3,2,1,0];
  let(old,b,a)=select_with_terminal(&p,&roots,&choices,&rank,&order,true,true,true,false,false);assert_eq!((b,a),(7,7));
  let(new,b,a)=select_with_terminal(&p,&roots,&choices,&rank,&order,true,true,true,false,true);assert_eq!((b,a),(7,6),"terminal source recovery must use only unused original transactions after the full old plan");
  let mut actual=p.clone();for(n,i)in new.iter().enumerate(){if let Some(i)=i{actual[n]=choices[n][*i].clone();}}assert_eq!(reachable(&actual,&roots),a);assert_ne!(old,new);
 }
}

#[cfg(test)]mod terminal_source_property{use super::*;
 #[test]fn complete_legacy_source_count_cannot_regress_on_reconvergent_dags(){let mut state=907u64;for _ in 0..256{let mut rnd=|n|{state=state.wrapping_mul(6364136223846793005).wrapping_add(1);(state as usize)%n};let n=42;let mut p=vec![vec![];n];let mut choices=vec![vec![];n];let mut rank=vec![0;n];for i in 3..n{let level=(i-3)/6+1;rank[i]=level;let bound=3+(level-1)*6;for _ in 0..2{p[i].push(rnd(bound));}for _ in 0..4{choices[i].push(vec![rnd(bound),rnd(bound)]);}}let roots=[0,1,2,36,37,38,39,40,41];let order:Vec<_>=(0..n).rev().collect();let(_,b,old)=select_with_terminal(&p,&roots,&choices,&rank,&order,true,true,true,false,false);let(s,b2,new)=select_with_terminal(&p,&roots,&choices,&rank,&order,true,true,true,false,true);assert_eq!(b,b2);assert!(new<=old);let mut actual=p.clone();for(n,i)in s.iter().enumerate(){if let Some(i)=i{actual[n]=choices[n][*i].clone();}}assert_eq!(reachable(&actual,&roots),new);}}
}
