//! Construction-only probe using a real selected-evidence JSON fixture.
//! This does not validate a mapping or measure mapper runtime.
use std::{alloc::{GlobalAlloc,Layout,System},sync::atomic::{AtomicBool,AtomicUsize,Ordering}};
static ACTIVE:AtomicBool=AtomicBool::new(false);
static ALLOCS:AtomicUsize=AtomicUsize::new(0);
static BYTES:AtomicUsize=AtomicUsize::new(0);
struct Count;
unsafe impl GlobalAlloc for Count {
 unsafe fn alloc(&self,l:Layout)->*mut u8{if ACTIVE.load(Ordering::Relaxed){ALLOCS.fetch_add(1,Ordering::Relaxed);BYTES.fetch_add(l.size(),Ordering::Relaxed);}unsafe{System.alloc(l)}}
 unsafe fn dealloc(&self,p:*mut u8,l:Layout){unsafe{System.dealloc(p,l)}}
 unsafe fn realloc(&self,p:*mut u8,l:Layout,n:usize)->*mut u8{if ACTIVE.load(Ordering::Relaxed){ALLOCS.fetch_add(1,Ordering::Relaxed);BYTES.fetch_add(n,Ordering::Relaxed);}unsafe{System.realloc(p,l,n)}}
}
#[global_allocator]static ALLOCATOR:Count=Count;
fn build(input:&serde_json::Value,moving:bool)->(serde_json::Value,usize,usize,bool){
 let mut object=input.as_object().unwrap().clone();
 let selected=object.remove("selected").unwrap().as_array().unwrap().clone();
 let instances=object.remove("instances").unwrap().as_array().unwrap().clone();
 let pointer=selected.as_ptr();ALLOCS.store(0,Ordering::SeqCst);BYTES.store(0,Ordering::SeqCst);ACTIVE.store(true,Ordering::SeqCst);
 // json!(vec) follows the SAME to_value(&vec) path used by object-valued json!.
 let selected=if moving{serde_json::Value::Array(selected)}else{serde_json::json!(selected)};
 let instances=if moving{serde_json::Value::Array(instances)}else{serde_json::json!(instances)};
 let kept=selected.as_array().unwrap().as_ptr()==pointer;
 object.insert("selected".into(),selected);object.insert("instances".into(),instances);
 ACTIVE.store(false,Ordering::SeqCst);
 (serde_json::Value::Object(object),ALLOCS.load(Ordering::SeqCst),BYTES.load(Ordering::SeqCst),kept)
}
fn main(){
 let path=std::env::args().nth(1).expect("selected-evidence JSON fixture required");
 let input:serde_json::Value=serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
 let(a,alloc_a,bytes_a,ptr_a)=build(&input,false);let(b,alloc_b,bytes_b,ptr_b)=build(&input,true);
 assert_eq!(a,b);assert_eq!(a,input);assert_eq!(serde_json::to_vec(&a).unwrap(),serde_json::to_vec(&b).unwrap());
 assert!(!ptr_a);assert!(ptr_b);assert!(alloc_b<alloc_a);
 println!("{}",serde_json::json!({"scope":"array assembly only; real archived fixture; no mapper runtime or PPA claim","selected":input["selected"].as_array().unwrap().len(),"instances":input["instances"].as_array().unwrap().len(),"equal_JSON_value":true,"equal_serialized_bytes":true,"original_array_storage_retained":[ptr_a,ptr_b],"construct_allocations":[alloc_a,alloc_b],"requested_allocation_bytes":[bytes_a,bytes_b],"allocator_bytes_are_not_peak_RSS":true}));
}
