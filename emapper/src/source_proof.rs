//! Bounded exact ROBDD proof of unconditional source aliases. No sampled union.
use crate::aig_graph::{Aig,AigNode};
use petgraph::{prelude::NodeIndex,visit::{Topo,EdgeRef,IntoEdgeReferences},Direction};
use std::collections::{HashMap,HashSet};
use std::time::Instant;
#[derive(Clone,Copy)]struct Decision{variable:u32,low:u32,high:u32}
struct Proof{nodes:Vec<Decision>,unique:HashMap<(u32,u32,u32),u32>,memo:HashMap<(u32,u32),u32>,calls:usize,start:Instant}
impl Proof{
 fn new()->Self{Self{nodes:vec![Decision{variable:u32::MAX,low:0,high:0}],unique:HashMap::new(),memo:HashMap::new(),calls:0,start:Instant::now()}}
 fn make(&mut self,var:u32,mut low:u32,mut high:u32)->Option<u32>{
  if low==high{return Some(low);}
  let phase=low&1;low^=phase;high^=phase;
  if let Some(&id)=self.unique.get(&(var,low,high)){return Some(id^phase);}
  if self.nodes.len()>=4096 || self.start.elapsed().as_secs_f64()>=0.2{return None;}
  let id=(self.nodes.len()as u32)*2;self.nodes.push(Decision{variable:var,low,high});self.unique.insert((var,low,high),id);Some(id^phase)
 }
 fn and(&mut self,mut a:u32,mut b:u32)->Option<u32>{
  if self.calls>=20000 || self.start.elapsed().as_secs_f64()>=0.2{return None;}
  self.calls+=1;
  if a>b{std::mem::swap(&mut a,&mut b);}
  if a==0 || a==(b^1){return Some(0);}if a==1{return Some(b);}if a==b{return Some(a);}
  if let Some(&v)=self.memo.get(&(a,b)){return Some(v);}
  let x=self.nodes[(a/2)as usize];let y=self.nodes[(b/2)as usize];let top=x.variable.min(y.variable);
  let split=|v:u32,d:Decision|if d.variable==top{(d.low^(v&1),d.high^(v&1))}else{(v,v)};
  let(al,ah)=split(a,x);let(bl,bh)=split(b,y);let low=self.and(al,bl)?;let high=self.and(ah,bh)?;let v=self.make(top,low,high)?;self.memo.insert((a,b),v);Some(v)
 }
}
pub(crate) fn compress(aig:&Aig)->Aig{compress_with_propagation(aig,false)}
pub(crate) fn compress_with_propagation(aig:&Aig,propagate:bool)->Aig{
 let start=Instant::now();let g=aig.graph();let mut proof=Proof::new();let mut values=HashMap::new();let mut representative=HashMap::new();let mut alias=HashMap::new();let mut topo=Topo::new(g);let mut examined=0;let mut unknown=0;let before=g.node_indices().filter(|&n|matches!(aig.node_type(n),AigNode::And)).count();
 // Equality of canonical ROBDD handles is an exact all-input identity.
 // Keep the first topological representative, so rewiring cannot introduce cycles.
 while let Some(n)=topo.next(g){
  let value=match aig.node_type(n){
   AigNode::Zero=>Some(0),AigNode::Input(p)=>proof.make(p,0,1),AigNode::Output(_)=>continue,
   AigNode::And=>{examined+=1;let mut e=g.edges_directed(n,Direction::Incoming);let x=e.next().expect("binary source");let y=e.next().expect("binary source");assert!(e.next().is_none());match (values.get(&x.source()).copied(),values.get(&y.source()).copied()){(Some(a),Some(b))=>proof.and(a^u32::from(*x.weight()),b^u32::from(*y.weight())),_=>None}}
  };
  if let Some(v)=value{
   values.insert(n,v);
   if let Some(&(old,phase))=representative.get(&(v/2)) {if matches!(aig.node_type(n),AigNode::And){alias.insert(n,(old,(v&1)!=phase));}}
   else{representative.insert(v/2,(n,v&1));}
  }else{unknown+=1;}
 }
 let mut result=aig.clone();
 // Redirect only edges with individually proved source identity. The original
 // PI and PO vertices/symbols remain, including unused primary inputs.
 let edges:Vec<_>=g.edge_references().map(|e|(e.id(),e.source(),e.target(),*e.weight())).collect();
 for (id,source,target,phase) in edges{if let Some(&(rep,inverted))=alias.get(&source){result.remove_edge(id);result.add_edge(rep,target,phase^inverted);}}
 if propagate {result=propagate_structural(&result);}
 let mut live=HashSet::new();let mut todo:Vec<_>=result.node_indices().filter(|&n|matches!(result.node_type(n),AigNode::Output(_)|AigNode::Input(_)|AigNode::Zero)).collect();
 while let Some(n)=todo.pop(){if live.insert(n){todo.extend(result.graph().edges_directed(n,Direction::Incoming).map(|e|e.source()));}}
 let dead:Vec<_>=result.node_indices().filter(|&n|matches!(result.node_type(n),AigNode::And)&&!live.contains(&n)).collect();for n in dead{result.remove_node(n);}
 let after=result.node_indices().filter(|&n|matches!(result.node_type(n),AigNode::And)).count();
 if let Some(path)=std::env::var_os("EGM_POOL_DIAG_PATH"){let _=std::fs::write(std::path::Path::new(&path).with_extension("source-proof.json"),serde_json::json!({"enabled":true,"structural_propagation":propagate,"before_source_and":before,"after_source_and":after,"proved_aliases":alias.len(),"examined_and":examined,"unknown":unknown,"proof_decisions":proof.nodes.len()-1,"proof_calls":proof.calls,"proof_decision_cap":4095,"proof_call_cap":20000,"proof_search_window_secs":0.2,"total_secs":start.elapsed().as_secs_f64(),"scope":"exact unconditional signed ROBDD handles; unknown retains source; all original PI/PO symbols retained. Temporary proof decisions are not egraph functions/candidates and cannot appear in cuts. Proof/search/rewiring charged to original native60/mapping300 guards, no offline cache. Actual imported functions still globally charged, no node retirement/reset. These source counts do not imply complete import or mapped PPA."}).to_string());}
 result
}

#[cfg(test)]mod tests{
 use super::*;
 fn eval(a:&Aig,bits:usize)->Vec<bool>{let mut v=HashMap::new();let mut t=Topo::new(a.graph());let mut out=vec![];while let Some(n)=t.next(a.graph()) {let get=|e:petgraph::stable_graph::EdgeReference<bool>|v[&e.source()]^*e.weight();let value=match a.node_type(n){AigNode::Zero=>false,AigNode::Input(p)=>(bits>>(p-1))&1!=0,AigNode::And=>a.graph().edges_directed(n,Direction::Incoming).all(get),AigNode::Output(p)=>{let value=get(a.graph().edges_directed(n,Direction::Incoming).next().unwrap());out.push((p,value));value}};v.insert(n,value);}out.sort();out.into_iter().map(|(_,v)|v).collect()}
 fn gate(a:&mut Aig,x:NodeIndex,xn:bool,y:NodeIndex,yn:bool)->NodeIndex{let n=a.add_node(AigNode::And);a.add_edge(x,n,xn);a.add_edge(y,n,yn);n}
 #[test]fn exact_absorption_releases_unique_cone_and_preserves_ports(){let mut a=Aig::default();a.add_node(AigNode::Zero);let x=a.add_node(AigNode::Input(1));let y=a.add_node(AigNode::Input(2));a.set_symbol_input(1,"a[0]".into());let or=gate(&mut a,x,true,y,true);let root=gate(&mut a,x,false,or,true);let o=a.add_node(AigNode::Output(0));a.set_symbol_output(0,"y[0]".into());a.add_edge(root,o,false);let c=compress(&a);assert_eq!(c.node_indices().filter(|&n|matches!(c.node_type(n),AigNode::And)).count(),0,"exact x&(x|y)=x should remove both ANDs");assert_eq!(c.symbol_input(1),a.symbol_input(1));assert_eq!(c.symbol_output(0),a.symbol_output(0));for b in 0..4{assert_eq!(eval(&a,b),eval(&c,b));}}
}
#[cfg(test)]mod generated_tests{
 use super::*;
 pub(super) fn evaluate(a:&Aig,bits:usize)->Vec<bool>{let mut values=HashMap::new();let mut t=Topo::new(a.graph());let mut out=vec![];while let Some(n)=t.next(a.graph()){let mut edges=a.graph().edges_directed(n,Direction::Incoming);let value=match a.node_type(n){AigNode::Zero=>false,AigNode::Input(v)=>(bits>>(v-1))&1!=0,AigNode::And=>edges.all(|e|values[&e.source()]^*e.weight()),AigNode::Output(v)=>{let e=edges.next().unwrap();let value=values[&e.source()]^*e.weight();out.push((v,value));value}};values.insert(n,value);}out.sort();out.into_iter().map(|(_,v)|v).collect()}
 #[test]fn signed_aliases_and_complements_preserve_all_primary_assignments(){
  for seed in 0..64 {let mut a=Aig::default();let mut nodes=vec![a.add_node(AigNode::Zero)];for p in 1..=6{nodes.push(a.add_node(AigNode::Input(p)));}for j in 0..40{let l=(seed*7+j*11)%nodes.len();let r=(seed*13+j*3)%nodes.len();let n=a.add_node(AigNode::And);a.add_edge(nodes[l],n,(j+seed)%3==0);a.add_edge(nodes[r],n,(j*5+seed)%4==0);nodes.push(n);}for p in 0..4{let n=a.add_node(AigNode::Output(p));a.add_edge(nodes[nodes.len()-1-p as usize],n,(seed+p as usize)%2==0);}let c=compress(&a);assert_eq!(a.input_nodes().len(),c.input_nodes().len());for bits in 0..64{assert_eq!(evaluate(&a,bits),evaluate(&c,bits),"seed{seed} bits{bits}");}}
 }
}

/// One topological structural pass; no new semantic proof queries.
fn propagate_structural(aig:&Aig)->Aig {
 let g=aig.graph();let mut result=aig.clone();
 let zero=g.node_indices().find(|&n|matches!(aig.node_type(n),AigNode::Zero)).unwrap_or_else(||result.add_node(AigNode::Zero));
 let mut translated:HashMap<NodeIndex,(NodeIndex,bool)>=HashMap::new();let mut unique=HashMap::new();let mut topo=Topo::new(g);
 while let Some(n)=topo.next(g){
  let value=match aig.node_type(n){
   AigNode::Zero=>(zero,false),AigNode::Input(_)=>(n,false),AigNode::Output(_)=>continue,
   AigNode::And=>{
    let mut edges=g.edges_directed(n,Direction::Incoming);let x=edges.next().expect("binary source");let y=edges.next().expect("binary source");assert!(edges.next().is_none());
    let get=|e:petgraph::stable_graph::EdgeReference<bool>|{let (r,p)=translated[&e.source()];(r,p^*e.weight())};
    let mut pair=[get(x),get(y)];pair.sort_unstable();let [a,b]=pair;
    if a==(zero,false)||b==(zero,false)||(a.0==b.0&&a.1!=b.1){(zero,false)}
    else if a==(zero,true){b}else if b==(zero,true)||a==b{a}
    else {(*unique.entry(pair).or_insert(n),false)}
   }
  };translated.insert(n,value);
 }
 for e in g.edge_references(){if let Some(&(rep,phase))=translated.get(&e.source()){
  if rep!=e.source()||phase {result.remove_edge(e.id());result.add_edge(rep,e.target(),*e.weight()^phase);}
 }}
 let mut live=HashSet::new();let mut todo:Vec<_>=result.node_indices().filter(|&n|!matches!(result.node_type(n),AigNode::And)).collect();
 while let Some(n)=todo.pop(){if live.insert(n){todo.extend(result.graph().edges_directed(n,Direction::Incoming).map(|e|e.source()));}}
 let dead:Vec<_>=result.node_indices().filter(|&n|matches!(result.node_type(n),AigNode::And)&&!live.contains(&n)).collect();for n in dead{result.remove_node(n);}
 result
}
#[cfg(test)]mod propagation_tests{
 use super::*;
 #[test]fn structural_propagation_preserves_signed_graphs_and_names(){
  for seed in 0..64{let mut a=Aig::default();let mut ns=vec![a.add_node(AigNode::Zero)];
   for v in 1..=6{ns.push(a.add_node(AigNode::Input(v)));a.set_symbol_input(v,format!("a[{v}]"));}
   for i in 0..40{let x=ns[(i*11+seed)%ns.len()];let y=ns[(i*7+seed*3)%ns.len()];let n=a.add_node(AigNode::And);a.add_edge(x,n,(i+seed)%3==0);a.add_edge(y,n,(i+seed)%4==0);ns.push(n);}
   for v in 0..4{let o=a.add_node(AigNode::Output(v));a.add_edge(ns[ns.len()-1-v as usize],o,(seed+v as usize)%2==0);a.set_symbol_output(v,format!("y[{v}]"));}
   let r=propagate_structural(&a);for bits in 0..64{assert_eq!(super::generated_tests::evaluate(&a,bits),super::generated_tests::evaluate(&r,bits),"seed{seed},bits{bits}");}
   assert_eq!(a.input_nodes().len(),r.input_nodes().len());for v in 1..=6{assert_eq!(a.symbol_input(v),r.symbol_input(v));}for v in 0..4{assert_eq!(a.symbol_output(v),r.symbol_output(v));}
  }
 }
 #[test]fn equal_input_alias_exposes_parent_duplicate(){
  let mut a=Aig::default();a.add_node(AigNode::Zero);let x=a.add_node(AigNode::Input(1));let y=a.add_node(AigNode::Input(2));
  let dup=a.add_node(AigNode::And);a.add_edge(x,dup,false);a.add_edge(x,dup,false);
  let p=a.add_node(AigNode::And);a.add_edge(dup,p,false);a.add_edge(y,p,false);
  let q=a.add_node(AigNode::And);a.add_edge(x,q,false);a.add_edge(y,q,false);
  for (id,n) in [(0,p),(1,q)]{let o=a.add_node(AigNode::Output(id));a.add_edge(n,o,false);}
  let r=propagate_structural(&a);
  assert_eq!(r.node_indices().filter(|&n|matches!(r.node_type(n),AigNode::And)).count(),1);
 }
}
