//! Generator evidence and transactions over complete, already-visited covers.
//! Evidence belongs to this graph/library invocation; it is never loaded from
//! archived netlists. Conditional care permits a physical match, not a union.
use super::*;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Assignment {
    key: InstanceKey,
    area_bits: u32,
    raw_inputs: Vec<usize>,
    pins: Vec<String>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingSource {
    pub(super) target: usize,
    raw_target: usize,
    boundary: Arc<Vec<usize>>,
    function: u64,
    care: u64,
    inverter: bool,
    assignment: Arc<Assignment>,
}
impl BindingSource {
    fn cut(&self) -> Option<(usize,Cut)> {
        (!self.inverter).then(||(self.raw_target/2,Cut{leaves:self.boundary.as_ref().clone(),function:self.function ^ if self.raw_target%2==1 {mask(self.boundary.len())} else {0}}))
    }
    fn valid(&self,m:&Instance,output:usize,cuts:&[BTreeSet<Cut>])->bool {
        if self.assignment.key!=instance_key(m) || self.assignment.area_bits!=m.area.to_bits() || !m.outputs[output].targets.contains(&self.target) {return false;}
        if self.inverter {
            return self.assignment.raw_inputs==[self.raw_target^1] && m.outputs[output].function==1;
        }
        self.cut().is_some_and(|(class,cut)|cuts.get(class).is_some_and(|bank|bank.contains(&cut)))
    }
    pub(super) fn diagnostic(&self)->serde_json::Value {
        serde_json::json!({"target":self.target,"raw_target":self.raw_target,"class":self.raw_target/2,"phase":self.raw_target%2,"boundary":self.boundary.as_ref(),"phased_function":self.function,"care":self.care,"kind":if self.inverter{"exact_phase_complement"}else if self.care==mask(self.boundary.len()){"full_care_cut"}else{"fixed_boundary_sdc"},"signed_physical_inputs":self.assignment.raw_inputs,"named_pins":self.assignment.pins,"normalized_instance_key":format!("{:?}",self.assignment.key)})
    }
}
fn assignment(m:&Instance,aliases:Option<(&[usize],&[bool])>)->Arc<Assignment>{
    let key=if let Some((a,_))=aliases {
        let mut order:Vec<_>=(0..m.inputs.len()).collect();order.sort_by_key(|&i|a[m.inputs[i]]);
        let inputs=order.iter().map(|&i|a[m.inputs[i]]).collect();
        let outputs=m.outputs.iter().map(|o|{
            let mut function=0;for bits in 0..1usize<<order.len(){let raw=order.iter().enumerate().fold(0,|v,(i,p)|v|(((bits>>i)&1)<<p));function|=((o.function>>raw)&1)<<bits;}
            (o.name.clone(),function,order.iter().map(|&i|o.delays[i].to_bits()).collect())
        }).collect();(m.cell.clone(),inputs,outputs)
    }else{instance_key(m)};
    Arc::new(Assignment{key,area_bits:m.area.to_bits(),raw_inputs:m.inputs.clone(),pins:m.input_pins.clone()})
}
pub(super) fn recorded_key(m:&Instance)->Option<InstanceKey>{m.outputs.iter().flat_map(|o|&o.sources).next().map(|s|s.assignment.key.clone())}

pub(super) fn record_matches(m:&mut Instance,boundary:&[usize],targets:&[Target],care:u64,aliases:Option<(&[usize],&[bool])>){
    let assignment=assignment(m,aliases);let boundary=Arc::new(boundary.to_vec());
    for output in &mut m.outputs {
        let mut actual=0;
        for a in 0..1usize<<boundary.len(){
            let tuple=m.inputs.iter().enumerate().fold(0,|v,(p,&s)|v|(((((a>>boundary.binary_search(&(s/2)).expect("physical input outside matched boundary"))&1)^(s%2)))<<p));
            actual|=((output.function>>tuple)&1)<<a;
        }
        for t in targets.iter().filter(|t|output.targets.contains(&t.signal) && (actual^t.function)&care==0){
            output.sources.push(BindingSource{target:t.signal,raw_target:t.signal,boundary:boundary.clone(),function:t.function,care,inverter:false,assignment:assignment.clone()});
        }
    }
}
pub(super) fn record_inverter(m:&mut Instance,aliases:Option<(&[usize],&[bool])>){
    let assignment=assignment(m,aliases);
    for o in &mut m.outputs {for &s in &o.targets {o.sources.push(BindingSource{target:s,raw_target:s,boundary:Arc::new(vec![]),function:1,care:1,inverter:true,assignment:assignment.clone()});}}
}
pub(super) fn normalize_sources(output:&mut OutputPin,aliases:&[usize]) {
    for source in &mut output.sources {source.target=aliases[source.target];}
    output.sources.retain(|s|output.targets.contains(&s.target));
}

#[derive(Debug)]
pub(super) struct ProtectedCover {
    pub chosen: Vec<Option<Binding>>,
    pub value: (f64,f64),
    pub pins: BTreeMap<usize,BTreeSet<Cut>>,
    pub live: BTreeSet<usize>,
}
impl ProtectedCover {
    pub(super) fn capture(instances:&[Instance],chosen:&[Option<Binding>],free:&[bool],roots:&[usize],cuts:&[BTreeSet<Cut>])->Result<Self,String>{
        let mut chosen=chosen.to_vec();prune(instances,&mut chosen,free,roots)?;
        let value=metrics(instances,&chosen,free,roots);if !value.0.is_finite()||!value.1.is_finite(){return Err("nonfinite protected cover".into());}
        let mut pins:BTreeMap<usize,BTreeSet<Cut>>=BTreeMap::new();let mut live=BTreeSet::new();
        for (s,b) in chosen.iter().enumerate(){if let Some(b)=b {
            let m=&instances[b.instance];let o=&m.outputs[b.pin];
            let source=o.sources.iter().find(|p|p.target==s&&p.valid(m,b.pin,cuts)).ok_or_else(||format!("binding {s} lacks generator cut/care/phase/pin evidence"))?;
            if let Some((class,cut))=source.cut(){pins.entry(class).or_default().insert(cut);}
            live.insert(b.instance);
        }}
        Ok(Self{chosen,value,pins,live})
    }
    pub(super) fn check_bank(&self,cuts:&[BTreeSet<Cut>],limit:usize)->Result<(),String>{
        if cuts.iter().any(|b|b.len()>limit){return Err("exchange exceeds cut allowance".into());}
        for (&c,pins)in &self.pins {if !pins.is_subset(&cuts[c]){return Err("exchange removes a protected binding's last selected source".into());}}
        Ok(())
    }
}

#[cfg(test)]
mod tests {
 use super::*;
 fn cell(name:&str,inputs:Vec<usize>,target:usize)->Instance {Instance{cell:name.into(),area:1.,input_pins:inputs.iter().enumerate().map(|(i,_)|format!("I{i}")).collect(),inputs,outputs:vec![OutputPin{sources:Vec::new(),name:"Y".into(),delays:vec![1.,1.],function:8,targets:vec![target]}]}}
 #[test]fn dedup_keeps_each_generator_source(){
  let boundary=[0,1];let mut m=cell("AND",vec![0,2],4);record_matches(&mut m,&boundary,&[Target{signal:4,function:8}],15,None);
  let mut n=m.clone();n.outputs[0].sources[0].care=7;
  let mut pool=Vec::new();let mut index=HashMap::new();insert_instance(&mut pool,&mut index,m);insert_instance(&mut pool,&mut index,n);
  assert_eq!(pool.len(),1);assert_eq!(pool[0].outputs[0].sources.len(),2,"dedup silently discards a distinct care certificate");
 }
 fn fixture()->(Vec<Instance>,Vec<Option<Binding>>,Vec<BTreeSet<Cut>>,Vec<bool>){
  let mut m=cell("AND",vec![0,2],4);record_matches(&mut m,&[0,1],&[Target{signal:4,function:8}],15,None);
  let mut n=cell("TOP",vec![4,0],6);n.outputs[0].delays=vec![1.,10.];record_matches(&mut n,&[0,2],&[Target{signal:6,function:8}],15,None);
  let bank=vec![BTreeSet::new(),BTreeSet::new(),BTreeSet::from([Cut{leaves:vec![0,1],function:8}]),BTreeSet::from([Cut{leaves:vec![0,2],function:8}])];
  (vec![m,n],vec![None,None,None,None,Some(Binding{instance:0,pin:0}),None,Some(Binding{instance:1,pin:0})],bank,vec![true,false,true,false,false,false,false])
 }
 #[test]fn complete_cover_requires_noncritical_dependency_sources(){
  let(mut m,c,bank,f)=fixture();let saved=ProtectedCover::capture(&m,&c,&f,&[6],&bank).unwrap();assert_eq!(saved.live.len(),2);assert_eq!(saved.pins.len(),2);assert!(!critical_arrivals(&m,&c,&f,&[6]).1[4],"test must protect a noncritical shared dependency");
  m[0].outputs[0].sources.clear();assert!(ProtectedCover::capture(&m,&c,&f,&[6],&bank).unwrap_err().contains("binding 4"));
 }
 #[test]fn transaction_preserves_all_shared_bindings_or_rejects(){
  let(m,c,bank,f)=fixture();let saved=ProtectedCover::capture(&m,&c,&f,&[6,4],&bank).unwrap();assert_eq!(saved.value,(2.,10.));
  let deadline=Instant::now()+std::time::Duration::from_secs(1);
  let(out,chosen)=merge_transaction(&m,vec![],&saved,&bank,&f,&[6,4],2,deadline).unwrap();assert_eq!(metrics(&out,&chosen,&f,&[6,4]),saved.value);
  assert!(merge_transaction(&m,vec![],&saved,&bank,&f,&[6,4],1,deadline).is_err());
  let mut missing=bank.clone();missing[2].clear();assert!(merge_transaction(&m,vec![],&saved,&missing,&f,&[6,4],2,deadline).is_err());
  assert!(merge_transaction(&m,vec![],&saved,&bank,&f,&[6,4],2,Instant::now()).is_err());
  assert_eq!(metrics(&m,&c,&f,&[6,4]),saved.value);
 }

 #[test]fn new_equal_area_composite_requires_post_exchange_delay_flow(){
  let make=|d|Instance{cell:"BUF".into(),area:1.,inputs:vec![0],input_pins:vec!["A".into()],outputs:vec![OutputPin{sources:vec![],name:"Y".into(),delays:vec![d],function:2,targets:vec![1]}]};
  let m=vec![make(2.),make(1.)];let f=[true,false];let c=vec![None,Some(Binding{instance:0,pin:0})];let p=MapParams{objective:MapObjective::DelayArea,..Default::default()};
  let area=MapParams{objective:MapObjective::AreaDelay,online_area:true,refine_rounds:1,refine_trials:1,..p.clone()};
  let (_,old)=slack_area_pass(&m,&f,&[1],&c,c.clone(),(1.,2.),&area,&mut MapStats::default(),Instant::now()+std::time::Duration::from_millis(50),true);assert_eq!(old,(1.,2.),"area-only proposals omit an equal-area faster composite");
  let out=recover_new_pool_delay(&m,c,&f,&[1],&p,Instant::now()+std::time::Duration::from_millis(50));assert_eq!(metrics(&m,&out,&f,&[1]),(1.,1.));
 }

 #[test]fn deadline_prefix_is_normalized_before_cached_key_insertion(){
  let aliases=[0,1,0,3,4,5];let free=[true,false,true,false,false,false];let mut m=cell("AND",vec![0,2],4);
  record_matches(&mut m,&[0,1],&[Target{signal:4,function:8}],15,Some((&aliases,&free)));
  assert_ne!(instance_key(&m),m.outputs[0].sources[0].assignment.key);
  let mut prefix=vec![m];normalize_exchange_prefix(&mut prefix,&aliases,&free);assert_eq!(prefix.len(),1);
  assert_eq!(instance_key(&prefix[0]),prefix[0].outputs[0].sources[0].assignment.key);assert_eq!(prefix[0].inputs,vec![0,0]);
 }

 #[test]fn direct_aliased_assignment_key_equals_complete_physical_normalization(){
  let f=[true,true,false,false,true,true,false,false];
  for aliases in [[0,1,0,1,4,5,6,7],[0,1,4,5,4,5,6,7],[0,1,2,3,4,5,6,7]] {for bits in 0..256 {
   let mut m=cell("TEST",vec![bits&3,(bits>>2)&3],6);m.outputs[0].function=(bits>>4)as u64;m.outputs[0].delays=vec![0.5,1.25];
   let cached=assignment(&m,Some((&aliases,&f)));assert!(normalize_physical_instance(&mut m,&aliases,&f));assert_eq!(cached.key,instance_key(&m));
  }}
 }

 #[test]fn expensive_fast_point_cannot_discard_a_legal_intermediate_shared_cover(){
  let p=MapParams{objective:MapObjective::DelayArea,..Default::default()};assert_eq!(protected_point((5.,8.),(12.,5.),(10.,10.),&p),0);
 }

 #[test]fn four_pin_aliased_key_preserves_truth_and_distinct_timing_arcs(){
  let aliases=[0,1,6,7,4,5,6,7,8,9];let f=[true,false,false,false,true,false,true,false,false,false];
  for seed in 0..128u64 {
   let inputs=vec![0,2,4,6];let mut m=cell("FOUR",inputs,8);m.outputs[0].function=seed.wrapping_mul(104729)&65535;m.outputs[0].delays=vec![1.,2.,3.,4.];
   let cached=assignment(&m,Some((&aliases,&f)));assert!(normalize_physical_instance(&mut m,&aliases,&f));assert_eq!(cached.key,instance_key(&m));
  }
 }

 #[test]fn sparse_transaction_moves_unchanged_physical_records(){
  let(mut m,c,bank,f)=fixture();let saved=ProtectedCover::capture(&m,&c,&f,&[6,4],&bank).unwrap();let pointers:Vec<_>=m.iter().map(|x|x.inputs.as_ptr()).collect();
  let mut extra=cell("ALT",vec![0,2],4);record_matches(&mut extra,&[0,1],&[Target{signal:4,function:8}],15,None);
  let out=merge_transaction_sparse(&mut m,vec![extra],&saved,&bank,&bank,&f,&[6,4],3,Instant::now()+std::time::Duration::from_secs(1)).unwrap();assert_eq!(m.len(),3);assert_eq!(metrics(&m,&out,&f,&[6,4]),saved.value);
  for i in 0..2{assert_eq!(m[i].inputs.as_ptr(),pointers[i],"unchanged physical records were deep-cloned");}
 }

 #[test]fn sparse_transaction_matches_reference_retention_and_dedup(){
  for seed in 0..32{let(mut m,c,bank,f)=fixture();for j in 0..5{let mut a=cell(&format!("ALT{j}"),vec![0,2],4);a.area=(j+1)as f32;a.outputs[0].delays=vec![0.5+j as f32,1.];record_matches(&mut a,&[0,1],&[Target{signal:4,function:8}],15,None);m.push(a);}
   let saved=ProtectedCover::capture(&m,&c,&f,&[6,4],&bank).unwrap();let mut extra=Vec::new();for j in 0..5{let id=(j+seed)%7;let mut a=cell(&format!("ALT{id}"),vec![0,2],4);a.area=(id+1)as f32;a.outputs[0].delays=vec![0.5+id as f32,1.];record_matches(&mut a,&[0,1],&[Target{signal:4,function:8}],if j%2==0{15}else{7},None);extra.push(a);}
   let limit=7+seed%3;let deadline=Instant::now()+std::time::Duration::from_secs(2);let(ref_pool,ref_choice)=merge_transaction(&m,extra.clone(),&saved,&bank,&f,&[6,4],limit,deadline).unwrap();let choice=merge_transaction_sparse(&mut m,extra,&saved,&bank,&bank,&f,&[6,4],limit,deadline).unwrap();
   let signature=|v:&[Instance]|v.iter().map(|m|(instance_key(m),(m.area.to_bits(),m.outputs.iter().map(|o|(o.targets.clone(),o.sources.clone())).collect::<Vec<_>>()))).collect::<BTreeMap<_,_>>();assert_eq!(signature(&m),signature(&ref_pool));assert_eq!(metrics(&m,&choice,&f,&[6,4]),metrics(&ref_pool,&ref_choice,&f,&[6,4]));
  }
 }
 #[test]fn sparse_transaction_failures_keep_complete_original_pool(){
  let(mut m,c,bank,f)=fixture();let saved=ProtectedCover::capture(&m,&c,&f,&[6,4],&bank).unwrap();let before=format!("{:?}",m);let pointers:Vec<_>=m.iter().map(|x|x.inputs.as_ptr()).collect();let mut missing=bank.clone();missing[2].clear();
  assert!(merge_transaction_sparse(&mut m,vec![],&saved,&bank,&missing,&f,&[6,4],2,Instant::now()+std::time::Duration::from_secs(1)).is_err());assert!(merge_transaction_sparse(&mut m,vec![],&saved,&bank,&bank,&f,&[6,4],1,Instant::now()+std::time::Duration::from_secs(1)).is_err());assert!(merge_transaction_sparse(&mut m,vec![],&saved,&bank,&bank,&f,&[6,4],2,Instant::now()).is_err());assert_eq!(format!("{:?}",m),before);assert_eq!(m.iter().map(|x|x.inputs.as_ptr()).collect::<Vec<_>>(),pointers);
 }
 #[test]fn sparse_transaction_filters_only_changed_noncritical_proofs(){
  let(mut m,c,mut bank,f)=fixture();let alt=Cut{leaves:vec![0,3],function:8};bank[2].insert(alt.clone());let mut back=cell("BACKEDGE",vec![0,6],4);record_matches(&mut back,&[0,3],&[Target{signal:4,function:8}],15,None);m.push(back);let saved=ProtectedCover::capture(&m,&c,&f,&[6,4],&bank).unwrap();let mut next=bank.clone();next[2].remove(&alt);let pointers:Vec<_>=m[..2].iter().map(|x|x.inputs.as_ptr()).collect();let choice=merge_transaction_sparse(&mut m,vec![],&saved,&bank,&next,&f,&[6,4],3,Instant::now()+std::time::Duration::from_secs(1)).unwrap();assert_eq!(m.len(),2);assert_eq!(metrics(&m,&choice,&f,&[6,4]),saved.value);assert_eq!(m.iter().map(|x|x.inputs.as_ptr()).collect::<Vec<_>>(),pointers);
 }
 #[test]fn sparse_transaction_touched_cap_rejects_before_any_mutation(){
  let(mut m,c,mut bank,f)=fixture();let alt=Cut{leaves:vec![0,3],function:8};bank[2].insert(alt.clone());for j in 0..2049{let mut back=cell(&format!("BACK{j}"),vec![0,6],4);record_matches(&mut back,&[0,3],&[Target{signal:4,function:8}],15,None);m.push(back);}let saved=ProtectedCover::capture(&m,&c,&f,&[6,4],&bank).unwrap();let mut next=bank.clone();next[2].remove(&alt);let pointers:Vec<_>=m.iter().map(|x|x.inputs.as_ptr()).collect();let before=format!("{:?}",m);let error=merge_transaction_sparse(&mut m,vec![],&saved,&bank,&next,&f,&[6,4],3000,Instant::now()+std::time::Duration::from_secs(2)).unwrap_err();assert!(error.contains("touched-record"));assert_eq!(format!("{:?}",m),before);assert_eq!(m.iter().map(|x|x.inputs.as_ptr()).collect::<Vec<_>>(),pointers);
 }

 #[test]fn sparse_transaction_rejects_an_overbudget_input_pool(){let(mut m,c,bank,f)=fixture();let mut a=cell("ALT",vec![0,2],4);record_matches(&mut a,&[0,1],&[Target{signal:4,function:8}],15,None);m.push(a);let saved=ProtectedCover::capture(&m,&c,&f,&[6,4],&bank).unwrap();let before=format!("{:?}",m);assert!(merge_transaction_sparse(&mut m,vec![],&saved,&bank,&bank,&f,&[6,4],2,Instant::now()+std::time::Duration::from_secs(1)).unwrap_err().contains("pre-transaction retained pool"));assert_eq!(format!("{:?}",m),before);}

}

fn evidence_key(m:&Instance)->Result<&InstanceKey,String>{m.outputs.iter().flat_map(|o|&o.sources).next().map(|p|&p.assignment.key).ok_or_else(||"transaction candidate has no generator identity".into())}
fn merge_outputs(existing:&mut Instance,mut offered:Instance)->Result<(),String>{
 if existing.area.to_bits()!=offered.area.to_bits() || evidence_key(existing)?!=evidence_key(&offered)? {return Err("transaction physical identity/area mismatch".into());}
 for(a,b)in existing.outputs.iter_mut().zip(&mut offered.outputs){a.targets.extend(b.targets.iter().copied());a.targets.sort_unstable();a.targets.dedup();for source in b.sources.drain(..){if !a.sources.contains(&source){a.sources.push(source);}}}Ok(())
}

// Prepare a borrowed pool and clone only records whose proofs/targets change.
// Until the final ownership transfer, every failure leaves `old` untouched.
fn merge_transaction_sparse(old:&mut Vec<Instance>,additional:Vec<Instance>,saved:&ProtectedCover,old_cuts:&[BTreeSet<Cut>],next:&[BTreeSet<Cut>],free:&[bool],roots:&[usize],limit:usize,deadline:Instant)->Result<Vec<Option<Binding>>,String>{
 const TOUCHED_LIMIT:usize=2048;
 let start=Instant::now();let expired=||Instant::now()>=deadline;
 if expired(){return Err("exchange preparation expired".into());}saved.check_bank(next,usize::MAX)?;
 if saved.live.len()>limit{return Err("complete protected cover exceeds retained pool budget".into());}
 if old.len()>limit{return Err("pre-transaction retained pool exceeds budget".into());}
 if old_cuts.len()!=next.len() || additional.len()>2048{return Err("exchange bank/offered allowance mismatch".into());}
 let changed:Vec<_>=old_cuts.iter().zip(next).map(|(a,b)|a!=b).collect();
 let invalid=|p:&BindingSource|!p.inverter && changed.get(p.raw_target/2).copied().unwrap_or(true) && !p.cut().is_some_and(|(c,cut)|next.get(c).is_some_and(|bank|bank.contains(&cut)));
 let active=|m:&Instance|m.outputs.iter().any(|o|!o.targets.is_empty());
 let mut edits:BTreeMap<usize,Instance>=BTreeMap::new();
 for(i,m)in old.iter().enumerate(){
  if i%128==0 && expired(){return Err("exchange proof filter expired".into());}
  if m.outputs.iter().flat_map(|o|&o.sources).any(&invalid){
   if edits.len()>=TOUCHED_LIMIT{return Err("exchange touched-record allowance exceeded".into());}
   let mut n=m.clone();for o in &mut n.outputs{o.sources.retain(|p|!invalid(p));o.targets.retain(|s|o.sources.iter().any(|p|p.target==*s));}edits.insert(i,n);
  }
 }
 // Keys borrow immutable generator identities; no key or full old record copy.
 let mut index:HashMap<&InstanceKey,(bool,usize)>=HashMap::new();
 for(i,m)in old.iter().enumerate(){
  if i%128==0 && expired(){return Err("exchange identity index expired".into());}
  if active(edits.get(&i).unwrap_or(m)){if index.insert(evidence_key(m)?,(true,i)).is_some(){return Err("nonunique old physical identity".into());}}
 }
 let mut plan=Vec::with_capacity(additional.len());let mut fresh_count=0;
 for m in &additional{
  if expired(){return Err("exchange offer planning expired".into());}
  if !active(m) || evidence_key(m)?!=&instance_key(m){return Err("offered match identity is not normalized".into());}
  let key=evidence_key(m)?;
  let target=if let Some(&id)=index.get(key){id}else{let id=(false,fresh_count);fresh_count+=1;index.insert(key,id);id};plan.push(target);
 }drop(index);
 let mut fresh:Vec<Option<Instance>>=(0..fresh_count).map(|_|None).collect();
 for(m,(existing,id))in additional.into_iter().zip(plan){
  if expired(){return Err("exchange offer merge expired".into());}
  if existing{let current=edits.get(&id).unwrap_or(&old[id]);if current.area.to_bits()==m.area.to_bits() && evidence_key(current)?==evidence_key(&m)? && current.outputs.iter().zip(&m.outputs).all(|(a,b)|b.targets.iter().all(|x|a.targets.contains(x))&&b.sources.iter().all(|p|a.sources.contains(p))){continue;}if !edits.contains_key(&id){if edits.len()+fresh_count>=TOUCHED_LIMIT{return Err("exchange touched-record allowance exceeded".into());}edits.insert(id,old[id].clone());}merge_outputs(edits.get_mut(&id).unwrap(),m)?;}
  else if let Some(slot)=fresh[id].as_mut(){merge_outputs(slot,m)?;}else{fresh[id]=Some(m);}
 }
 if edits.len()+fresh_count>TOUCHED_LIMIT{return Err("exchange touched-record allowance exceeded".into());}
 let mut view=Vec::new();let mut origins=Vec::new();let mut old_to_view=vec![None;old.len()];
 for(i,m)in old.iter().enumerate(){let m=edits.get(&i).unwrap_or(m);if active(m){old_to_view[i]=Some(view.len());view.push(m);origins.push((true,i));}}
 let old_count=view.len();
 for(i,m)in fresh.iter().enumerate(){let m=m.as_ref().ok_or("missing planned physical offer")?;view.push(m);origins.push((false,i));}
 let mut chosen=saved.chosen.clone();
 for(s,b)in chosen.iter_mut().enumerate(){if let Some(binding)=b{
  if s%128==0 && expired(){return Err("exchange protected proof validation expired".into());}
  let original=&old[binding.instance];let mapped=old_to_view[binding.instance].ok_or("protected binding lost during proof filtering")?;let m=view[mapped];
  // Inputs, pin arcs and area are unchanged: the saved acyclic circuit is
  // preserved by an injective producer renaming, with the same summation order.
  if m.area.to_bits()!=original.area.to_bits() || m.inputs!=original.inputs || m.input_pins!=original.input_pins || instance_key(m)!=instance_key(original){return Err("protected physical implementation changed".into());}
  if !m.outputs[binding.pin].targets.contains(&s) || !m.outputs[binding.pin].sources.iter().any(|p|p.target==s&&p.valid(m,binding.pin,next)){return Err("protected binding lost its generator proof".into());}
  binding.instance=mapped;
 }}
 let must:BTreeSet<_>=chosen.iter().flatten().map(|b|b.instance).collect();let mut keep=must.clone();
 if view.len()>limit{
  if expired(){return Err("exchange ranking expired".into());}
  let(arrival,critical)=critical_arrivals(old,&saved.chosen,free,roots);
  let mut ranked=Vec::new();for(i,m)in view.iter().enumerate(){
   if i%128==0 && expired(){return Err("exchange optional scoring expired".into());}if must.contains(&i){continue;}
   let gain=m.outputs.iter().flat_map(|o|o.targets.iter().filter(|&&s|critical[s]).map(move|&s|(o,s))).map(|(o,s)|{let time=m.inputs.iter().enumerate().filter(|(p,_)|o.has_arc(*p)).map(|(p,&a)|arrival[a].unwrap_or(f64::INFINITY)+f64::from(o.delays[p])).fold(0.,f64::max);time-arrival[s].unwrap_or(0.)}).fold(f64::INFINITY,f64::min);
   ranked.push((gain,m.area,i<old_count,i));
  }ranked.sort_by(|a,b|a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)).then(a.3.cmp(&b.3)));keep.extend(ranked.into_iter().take(limit-must.len()).map(|x|x.3));
 }else{keep.extend(0..view.len());}
 let mut remap=vec![None;view.len()];let mut keep_old=vec![false;old.len()];let mut keep_new=vec![false;fresh.len()];
 for(i,&slot)in keep.iter().enumerate(){remap[slot]=Some(i);let(origin,id)=origins[slot];if origin{keep_old[id]=true;}else{keep_new[id]=true;}}
 for b in chosen.iter_mut().flatten(){b.instance=remap[b.instance].ok_or("protected binding evicted by retained cap")?;}
 let touched=edits.len();let raw_view=view.len();drop(view);
 // Reserve disposal/ownership-transfer time within the EXISTING exchange window.
 if Instant::now()+std::time::Duration::from_millis(50)>=deadline{return Err("exchange commit/disposal reserve unavailable".into());}
 let mut committed=Vec::with_capacity(keep.len());
 for(i,m)in std::mem::take(old).into_iter().enumerate(){if keep_old[i]{committed.push(edits.remove(&i).unwrap_or(m));}}
 for(i,m)in fresh.into_iter().enumerate(){if keep_new[i]{committed.push(m.unwrap());}}
 drop(edits);*old=committed;
 recovery_event("exchange_transaction_sparse",start,serde_json::json!({"retained_pool":old.len(),"protected_physical_instances":saved.live.len(),"old_area":saved.value.0,"old_delay":saved.value.1,"new_area":saved.value.0,"new_delay":saved.value.1,"deep_cloned_records":touched,"whole_pool_deep_clone":false,"touched_record_limit":TOUCHED_LIMIT,"raw_borrowed_view":raw_view,"commit_disposal_reserve_secs":0.05,"protected_dag_preserved_by_injective_renaming":true,"ownership_transfer_and_disposal_in_global_clock":true,"stage_checks_nonpreemptive":true}));
 Ok(chosen)
}

fn merge_transaction(old:&[Instance],additional:Vec<Instance>,saved:&ProtectedCover,next:&[BTreeSet<Cut>],free:&[bool],roots:&[usize],limit:usize,deadline:Instant)->Result<(Vec<Instance>,Vec<Option<Binding>>),String>{
    saved.check_bank(next,usize::MAX)?;
    let transaction_start=Instant::now();
    let key_for=|m:&Instance|m.outputs.iter().flat_map(|o|&o.sources).next().map(|p|p.assignment.key.clone()).unwrap_or_else(||instance_key(m));
    let mut pool=Vec::new();let mut index=HashMap::new();let mut remap=vec![None;old.len()];
    for (i,m)in old.iter().enumerate(){
        if i%128==0 && Instant::now()>=deadline{return Err("exchange stage expired before commit".into());}
        let mut m=m.clone();
        for o in &mut m.outputs {o.sources.retain(|p|p.inverter||p.cut().is_some_and(|(c,cut)|next[c].contains(&cut)));o.targets.retain(|s|o.sources.iter().any(|p|p.target==*s));}
        if m.outputs.iter().all(|o|o.targets.is_empty()){continue;}
        let key=key_for(&m);remap[i]=Some(insert_instance_key(&mut pool,&mut index,m,key));
    }
    let mut chosen=saved.chosen.clone();
    for b in chosen.iter_mut().flatten(){b.instance=remap[b.instance].ok_or("protected physical binding lost during exchange")?;}
    let old_count=pool.len();
    for (i,m) in additional.into_iter().enumerate(){
        if i%128==0 && Instant::now()>=deadline{return Err("exchange insertion expired before commit".into());}
        let key=key_for(&m);insert_instance_key(&mut pool,&mut index,m,key);
    }
    let must:BTreeSet<_>=chosen.iter().flatten().map(|b|b.instance).collect();
    if must.len()>limit{return Err("complete protected cover exceeds retained pool budget".into());}
    if pool.len()>limit {
        // The protected closure is mandatory; optional ranking uses the actual
        // saved arrival labels. No extra flow extraction or recovery invocation.
        let (arrival,critical)=critical_arrivals(&pool,&chosen,free,roots);
        let mut ranked:Vec<_>=pool.iter().enumerate().filter(|(i,_)|!must.contains(i)).map(|(i,m)|{
            let gain=m.outputs.iter().flat_map(|o|o.targets.iter().filter(|&&s|critical[s]).map(move |&s|(o,s))).map(|(o,s)|{
                let time=m.inputs.iter().enumerate().filter(|(p,_)|o.has_arc(*p)).map(|(p,&a)|arrival[a].unwrap_or(f64::INFINITY)+f64::from(o.delays[p])).fold(0.,f64::max);
                time-arrival[s].unwrap_or(0.)
            }).fold(f64::INFINITY,f64::min);
            (gain,m.area,i<old_count,i)
        }).collect();
        ranked.sort_by(|a,b|a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)).then(a.3.cmp(&b.3)));
        let mut keep=must;let remaining=limit-keep.len();keep.extend(ranked.into_iter().take(remaining).map(|x|x.3));
        let mut compact=Vec::with_capacity(keep.len());let mut ids=vec![None;pool.len()];
        for (i,m)in pool.into_iter().enumerate(){if keep.contains(&i){ids[i]=Some(compact.len());compact.push(m);}}
        for b in chosen.iter_mut().flatten(){b.instance=ids[b.instance].ok_or("protected binding evicted by physical cap")?;}pool=compact;
    }
    let verified=ProtectedCover::capture(&pool,&chosen,free,roots,next)?;
    if verified.value!=saved.value{return Err("complete protected physical area/delay changed on remap".into());}
    chosen=verified.chosen;
    recovery_event("exchange_transaction",transaction_start,serde_json::json!({"retained_pool":pool.len(),"protected_physical_instances":verified.live.len(),"old_area":saved.value.0,"old_delay":saved.value.1,"new_area":verified.value.0,"new_delay":verified.value.1,"no_flow_replay":true}));
    if Instant::now()>=deadline{return Err("exchange validation expired before commit".into());}
    Ok((pool,chosen))
}

fn protected_point(candidate:(f64,f64),fast:(f64,f64),old:(f64,f64),ps:&MapParams)->usize{
    let mut value=candidate;let mut selected=0;
    let feasible=|v:(f64,f64)|v.0<=old.0 && v.1<=old.1;
    if feasible(fast) && (!feasible(value)||better_cover(fast,value,ps)){value=fast;selected=1;}
    if value.0>old.0 || value.1>old.1 || !better_cover(value,old,ps){selected=2;}
    selected
}

// Removing a metric guard does not remove cycle, source-proof or pool checks.
fn recovery_point(candidate:(f64,f64),fast:(f64,f64),old:(f64,f64),ps:&MapParams)->usize{
    if ps.ablate_incumbent_guard {usize::from(better_cover(fast,candidate,ps))}
    else {protected_point(candidate,fast,old,ps)}
}

fn normalize_exchange_prefix(instances:&mut Vec<Instance>,aliases:&[usize],free:&[bool]){
    // generate() can return a raw legal prefix when its deadline fires before
    // its ordinary normalization/dedup tail. Canonicalize before cached keys.
    instances.retain_mut(|m|normalize_physical_instance(m,aliases,free));
}

fn recover_new_pool_delay(instances:&[Instance],chosen:Vec<Option<Binding>>,free:&[bool],roots:&[usize],ps:&MapParams,deadline:Instant)->Vec<Option<Binding>> {
    if Instant::now()>=deadline{return chosen;}
    let before=metrics(instances,&chosen,free,roots);
    if let Some((mut candidate,_))=delay_frontier_seed(instances,free,deadline){
        if prune(instances,&mut candidate,free,roots).is_ok() && better_cover(metrics(instances,&candidate,free,roots),before,ps){return candidate;}
    }
    chosen
}

fn exchange_cuts(cuts:&[BTreeSet<Cut>],saved:&ProtectedCover,instances:&[Instance],witness:&[Option<Binding>],free:&[bool],roots:&[usize],aliases:&[usize],tech:&TechLib,ps:&MapParams,stats:&mut MapStats,deadline:Instant)->Option<Vec<BTreeSet<Cut>>>{
    if ps.ablate_functional_cuts {return Some(cuts.to_vec());}
    let index=functional_support_index_until(cuts,Some(deadline))?;
    if Instant::now()>=deadline{return None;}
    let mut next=cuts.to_vec();let batch_start=stats.exchange_cuts;
    let(arrival,critical)=critical_arrivals(instances,&saved.chosen,free,roots);
    if Instant::now()>=deadline{return None;}
    let witness_roots:Vec<_>=witness.iter().enumerate().filter_map(|(s,b)|b.map(|_|s)).collect();
    let fallback=critical_arrivals(instances,witness,free,&witness_roots).0;
    if Instant::now()>=deadline{return None;}
    let times:Vec<_>=arrival.iter().zip(&fallback).map(|(a,b)|a.or(*b).unwrap_or(f64::INFINITY)).collect();
    let mut order:Vec<_>=(0..cuts.len()).collect();order.sort_by_key(|&c|(![0,1].iter().any(|p|critical[aliases[c*2+p]]),std::cmp::Reverse([0,1].iter().filter_map(|p|arrival[aliases[c*2+p]]).fold(0.,f64::max).to_bits()),c));
    for c in order {
        if Instant::now()>=deadline || stats.exchange_cuts-batch_start>=64{break;}
        let before=stats.functional_cut_trials;let proposals=functional_proposals(c,&cuts[c],&index,tech,stats);stats.exchange_trials+=stats.functional_cut_trials-before;
        if proposals.is_empty(){continue;}
        let mut pinned=saved.pins.get(&c).cloned().unwrap_or_default();
        // The self boundary remains available for ordinary wires and INV matching.
        for cut in &cuts[c]{if cut.leaves==[c]{pinned.insert(cut.clone());}}
        if pinned.len()>ps.cuts_per_class{continue;}
        let mut optional:Vec<_>=proposals.iter().chain(cuts[c].iter()).filter(|p|!pinned.contains(*p)).cloned().collect::<BTreeSet<_>>().into_iter().collect();
        let score=|cut:&Cut|{
            let k=cut.leaves.len();if k==0{return 0.;}
            let mut demanded=Vec::new();for phase in 0..2 {let s=aliases[c*2+phase];if saved.chosen[s].is_some(){demanded.push(phase);}}
            if demanded.is_empty(){demanded.push(0);}
            demanded.into_iter().map(|phase|tech.candidates(k as u8,tt9_from_u64(cut.function^if phase==1{mask(k)}else{0})).into_iter().flatten().filter(|v|!v.out_neg).map(|v|cut.leaves.iter().enumerate().map(|(p,&leaf)|times[aliases[leaf*2+usize::from((v.neg_mask>>p)&1)]]+f64::from(v.pin_delay[p])).fold(0.,f64::max)).fold(f64::INFINITY,f64::min)-times[aliases[c*2+phase]]).fold(f64::NEG_INFINITY,f64::max)
        };
        let mut ranked:Vec<_>=optional.drain(..).map(|cut|{let score=score(&cut);(if score.is_nan(){f64::INFINITY}else{score},cut)}).collect();
        ranked.sort_by(|a,b|a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut added=0;
        for (_,cut) in ranked {
            if pinned.len()==ps.cuts_per_class{break;}
            if !cuts[c].contains(&cut){if stats.exchange_cuts-batch_start+added>=64{continue;}added+=1;}
            pinned.insert(cut);
        }
        stats.exchange_cuts+=pinned.difference(&cuts[c]).count();next[c]=pinned;
    }
    Some(next)
}

/// A new opt-in schedule, not a replay of a previous pipeline's two-second
/// final result: 31 ordinary cut rounds + one quotient exchange, one recovery
/// clock. The protected reference is the complete seed visited before exchange.
pub(super) fn recover(instances:&mut Vec<Instance>,cuts:&mut Vec<BTreeSet<Cut>>,free:&[bool],roots:&[usize],interface_free:&[bool],constants:&[Option<bool>],physical_aliases:Option<(&[usize],&[bool])>,tech:&TechLib,reach:&mut Reachability,ps:&MapParams,stats:&mut MapStats,generation_frontier:Option<GenerationFrontier>)->Result<Vec<Option<Binding>>,String>{
    let _trace=RecoveryTrace;RECOVERY_EVENTS.with(|x|x.borrow_mut().clear());let start=Instant::now();
    let deadline=start+std::time::Duration::from_secs_f64(ps.refine_time_secs);
    let search=deadline-std::time::Duration::from_secs_f64(ps.return_reserve.min(ps.refine_time_secs));
    recovery_event("component_ablation",start,serde_json::json!({"functional_cuts_disabled":ps.ablate_functional_cuts,"shared_area_disabled":ps.ablate_shared_area,"incumbent_metric_guard_disabled":ps.ablate_incumbent_guard,"structural_legality_and_proofs_retained":true,"same_global_deadline":true}));
    let delay_seed_deadline=regions::phase_deadline(start,search,0.15);
    let seed_deadline=regions::phase_deadline(start,search,0.35);
    // Resolve and validate already necessary generation data inside this clock.
    // Uncapped pools provide no cache and keep the ordinary in-window setup.
    let mut retained=match generation_frontier {Some(f)=>Some(f.resolve(instances,free,roots)?),None=>None};
    let reused=retained.is_some();
    let witness=if let Some((_,w,_))=&mut retained{std::mem::take(w)}else{finite_choices_queued(instances,free)};
    let mut chosen=witness.clone();prune(instances,&mut chosen,free,roots)?;
    let mut value=metrics(instances,&chosen,free,roots);stats.initial_area=value.0;stats.initial_delay=value.1;
    if let Some((candidate,_,origin))=retained {
        // Index remapping, prune and metric validation all consume this same
        // recovery window. No extra retention or diagnostic frontier is run.
        let v=metrics(instances,&candidate,free,roots);let adopted=better_cover(v,value,ps);
        if adopted{chosen=candidate;value=v;}
        recovery_event("generation_frontier_reference",start,serde_json::json!({"origin":origin,"area":v.0,"delay":v.1,"adopted":adopted,"from_required_retention":true,"reporting_only":false,"additional_frontier_calls":0,"additional_finite_witness_calls":0,"area_witness_reused":true,"remap_and_validate_inside_recovery":true}));
    }
    if !reused {if let Some((mut fast,_))=delay_frontier_seed(instances,free,delay_seed_deadline){if prune(instances,&mut fast,free,roots).is_ok(){let v=metrics(instances,&fast,free,roots);if better_cover(v,value,ps){chosen=fast;value=v;}}}}
    if Instant::now()<seed_deadline {if let Some(mut area)=required_area_dag(instances,free,roots,&chosen,&witness,seed_deadline){if prune(instances,&mut area,free,roots).is_ok(){let v=metrics(instances,&area,free,roots);if v.1<=value.1 && better_cover(v,value,ps){chosen=area;value=v;}}}}
    let saved=ProtectedCover::capture(instances,&chosen,free,roots,cuts)?;stats.protected_bindings=saved.chosen.iter().flatten().count();stats.seed_delay=value.1;
    recovery_event("protected_reference",start,serde_json::json!({"area":value.0,"delay":value.1,"bindings":stats.protected_bindings,"physical_instances":saved.live.len(),"scope":"complete same-run pre-exchange seed; NOT a historical full-budget final cover","one_global_clock":true}));
    let mut pool_changed=false;
    let exchange_deadline=regions::phase_deadline(start,search,0.70);
    let exchange_start=Instant::now();
    let proposal_deadline=regions::phase_deadline(exchange_start,exchange_deadline,0.35);
    let generation_deadline=regions::phase_deadline(exchange_start,exchange_deadline,0.65);
    let aliases=physical_aliases.expect("checked raw alias wires").0;
    if Instant::now()<exchange_deadline {
        stats.cut_rounds_started+=1;
        let next=match exchange_cuts(cuts,&saved,instances,&witness,free,roots,aliases,tech,ps,stats,proposal_deadline){Some(bank)=>bank,None=>{recovery_event("exchange_setup_aborted",start,serde_json::json!({"incomplete_index_not_used":true,"unchanged_pool_and_bank":true,"same_global_budget":true,"available_witness_reused":true}));cuts.clone()}};
        saved.check_bank(&next,ps.cuts_per_class)?;
        if next==*cuts {
            // No physical change exists to commit. Rebuilding the whole pool
            // would consume the area stage without changing a single match.
            recovery_event("protected_exchange",start,serde_json::json!({"trials":stats.exchange_trials,"cuts":0,"committed":false,"no_op":true,"raw_additional":0,"retained_pool":instances.len(),"cut_rounds_started":stats.cut_rounds_started,"original_cut_round_limit":ps.cut_rounds,"no_fresh_recovery_budget":true,"functional_initial_bank":ps.exchange_functional_seed}));
        }else{
        let delta:Vec<_>=next.iter().zip(cuts.iter()).map(|(new,old)|new.difference(old).cloned().collect()).collect();
        let mut additional=generate(&delta,interface_free,constants,tech,ps,reach,stats,physical_aliases,Some(generation_deadline));
        let raw=additional.len();stats.generated_instances+=raw;
        additional.truncate(2048);
        normalize_exchange_prefix(&mut additional,aliases,free);
        let offered=additional.len();
        let failure=if additional.is_empty(){stats.exchange_fallback=true;Some("no new normalized physical match; old bank/pool retained".to_string())}else{match merge_transaction_sparse(instances,additional,&saved,cuts,&next,free,roots,ps.max_instances,exchange_deadline){Ok(remapped)=>{*cuts=next;chosen=remapped;pool_changed=true;None},Err(error)=>{stats.exchange_fallback=true;Some(error)}}};
        recovery_event("protected_exchange",start,serde_json::json!({"trials":stats.exchange_trials,"cuts":stats.exchange_cuts,"committed":!stats.exchange_fallback,"raw_additional":raw,"offered_additional":offered,"match_prefix_limit":2048,"new_cut_batch_limit":64,"functional_initial_bank":ps.exchange_functional_seed,"retained_pool":instances.len(),"cut_rounds_started":stats.cut_rounds_started,"original_cut_round_limit":ps.cut_rounds,"phase_fraction_of_global_search":0.70,"failure":failure,"proposal_deadline_secs":proposal_deadline.duration_since(start).as_secs_f64(),"generation_deadline_secs":generation_deadline.duration_since(start).as_secs_f64(),"commit_deadline_secs":exchange_deadline.duration_since(start).as_secs_f64(),"matching_and_validation_reserved":true,"no_fresh_recovery_budget":true}));
        }
    }
    let protected=chosen.clone();let protected_value=metrics(instances,&protected,free,roots);
    let witness=if ps.retain_generation_frontier && !pool_changed {
        recovery_event("unchanged_pool_witness_reuse",start,serde_json::json!({"pool_changed":false,"new_witness_calls":0}));witness
    }else{finite_choices_queued(instances,free)};
    if !stats.exchange_fallback && stats.exchange_cuts>0 && Instant::now()<search {
        let flow_start=Instant::now();let flow_end=regions::phase_deadline(flow_start,search,0.25);
        chosen=recover_new_pool_delay(instances,chosen,free,roots,ps,flow_end);
        let value=metrics(instances,&chosen,free,roots);
        recovery_event("post_exchange_delay_flow",start,serde_json::json!({"area":value.0,"delay":value.1,"elapsed":flow_start.elapsed().as_secs_f64(),"no_new_global_deadline":true}));
    }
    if ps.phase_recovery && Instant::now()<search {
        let phase_end=regions::phase_deadline(Instant::now(),search,ps.phase_time_fraction);
        let phase_start=Instant::now();
        regions::recover_phases(instances,&mut chosen,&witness,if ps.ablate_incumbent_guard {None}else{Some(&protected)},free,roots,ps,stats,phase_end)?;
        stats.phase_secs=phase_start.elapsed().as_secs_f64();
    }
    let phase_value=metrics(instances,&chosen,free,roots);
    let mut fast=if ps.ablate_incumbent_guard || (phase_value.0<=protected_value.0 && phase_value.1<=protected_value.1) {chosen.clone()}else{protected.clone()};
    let mut fast_value=metrics(instances,&fast,free,roots);let fixed_ref=fast_value.1;
    let mut area=ps.clone();area.objective=MapObjective::AreaDelay;area.max_delay=Some(fixed_ref);area.protected_area_limit=if ps.ablate_incumbent_guard {None}else{Some(protected_value.0)};area.refresh_area_timing=true;area.online_area=true;area.terminal_area=false;
    if !ps.ablate_shared_area && Instant::now()<search {
        stats.slack_attempts+=1;let flow_start=Instant::now();let flow_end=regions::phase_deadline(flow_start,search,0.30);
        if let Some(mut candidate)=required_area_dag(instances,free,roots,&fast,&witness,flow_end){
            if prune(instances,&mut candidate,free,roots).is_ok(){
                let v=metrics(instances,&candidate,free,roots);stats.slack_valid+=1;
                if (ps.ablate_incumbent_guard || v.0<=protected_value.0) && v.1<=fixed_ref && better_cover(v,fast_value,ps){fast=candidate.clone();fast_value=v;}
                if v.1<=fixed_ref && better_cover(v,metrics(instances,&chosen,free,roots),&area){chosen=candidate;stats.slack_accepted+=1;}
            }
        }
        recovery_event("fixed_bound_area_flow",start,serde_json::json!({"fixed_delay_limit":fixed_ref,"protected_area_limit":protected_value.0,"elapsed":flow_start.elapsed().as_secs_f64(),"shared_global_deadline":true}));
    }
    if !ps.ablate_shared_area && Instant::now()<search {
        let current=metrics(instances,&chosen,free,roots);
        (chosen,_)=slack_area_pass(instances,free,roots,&witness,chosen,current,&area,stats,search,true);
    }
    // The area stage can explore slack, but never discards the faster DA point.
    let mut best=metrics(instances,&chosen,free,roots);
    match recovery_point(best,fast_value,protected_value,ps){1=>{chosen=fast;best=fast_value;},2=>{chosen=protected;best=protected_value;},_=>{}}
    prune(instances,&mut chosen,free,roots)?;ProtectedCover::capture(instances,&chosen,free,roots,cuts)?;
    stats.heuristic_area=best.0;stats.heuristic_secs=start.elapsed().as_secs_f64();stats.used_fallback=true;stats.solver_status="protected-exchange-heuristic".into();
    recovery_event("global_total",start,serde_json::json!({"budget":ps.refine_time_secs}));
    if Instant::now()>deadline{return Err("budget: protected exchange recovery exceeded global inclusive deadline".into());}
    Ok(chosen)
}

#[cfg(test)] mod independent_component_tests {
 use super::*;
 #[test] fn metric_guard_ablation_allows_faster_cover_with_area_tradeoff(){
  let p=MapParams{objective:MapObjective::DelayArea,..Default::default()};
  assert_eq!(recovery_point((12.,5.),(10.,10.),(10.,10.),&p),2);
  let ablated=MapParams{ablate_incumbent_guard:true,..p};
  assert_eq!(recovery_point((12.,5.),(10.,10.),(10.,10.),&ablated),0);
 }
 #[test] fn disabling_area_or_quotients_does_not_disable_the_seed_guard(){
  let p=MapParams{objective:MapObjective::DelayArea,ablate_shared_area:true,ablate_functional_cuts:true,..Default::default()};
  assert_eq!(recovery_point((12.,5.),(12.,5.),(10.,10.),&p),2);
  assert_eq!(p.refine_time_secs,MapParams::default().refine_time_secs);
 }
 #[test] fn disabled_quotient_exchange_returns_identical_bank_without_trials(){
  let cuts=vec![BTreeSet::from([Cut{leaves:vec![0],function:2}])];
  let saved=ProtectedCover{chosen:vec![None],value:(0.,0.),live:BTreeSet::new(),pins:BTreeMap::new()};
  let p=MapParams{ablate_functional_cuts:true,..Default::default()};let mut stats=MapStats::default();
  let result=exchange_cuts(&cuts,&saved,&[],&[],&[true],&[0],&[0],&TechLib::default(),&p,&mut stats,Instant::now());
  assert_eq!(result.unwrap(),cuts);assert_eq!(stats.functional_cut_trials,0);assert_eq!(stats.exchange_trials,0);
 }
}
