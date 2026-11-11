//! Write-trait granularity only, not a filesystem or mapper speed benchmark.
use std::io::{self,Write};
#[derive(Default)]struct Counter{calls:usize,bytes:Vec<u8>}
impl Write for Counter{
 fn write(&mut self,b:&[u8])->io::Result<usize>{self.calls+=1;self.bytes.extend_from_slice(b);Ok(b.len())}
 fn flush(&mut self)->io::Result<()>{Ok(())}
}
fn main(){
 let row:serde_json::Value=serde_json::from_str(&std::env::args().nth(1).unwrap()).unwrap();
 let mut old=Counter::default();writeln!(&mut old,"{}",row).unwrap();
 let mut new=Counter::default();let mut bytes=serde_json::to_vec(&row).unwrap();bytes.push(b'\n');new.write_all(&bytes).unwrap();
 assert_eq!(old.bytes,new.bytes);assert!(new.calls<old.calls);
 println!("{}",serde_json::json!({"equal_bytes":true,"write_trait_calls":[old.calls,new.calls],"bytes":old.bytes.len(),"scope":"write granularity probe; not native syscall timing or a mapper speedup"}));
}
