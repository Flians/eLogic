//! Explicit experimental native fast entry; never calls an external mapper.
use emapper::egraph_map::fast_native::{Params,run_files};
use std::path::Path;
fn main()->Result<(),Box<dyn std::error::Error>>{
 let a:Vec<_>=std::env::args().skip(1).collect();if !(4..=7).contains(&a.len()){return Err("usage: emapper_fast library.genlib input.aig mapped.v report.json [cut_frontier=16] [critical_quotients=1] [area_passes=2]".into())}
 let p=Params{cut_frontier:a.get(4).map(|s|s.parse()).transpose()?.unwrap_or(16),critical_quotients:match a.get(5).map(String::as_str).unwrap_or("1"){"0"=>false,"1"=>true,_=>return Err("critical_quotients must be0 or1".into())},area_passes:a.get(6).map(|s|s.parse()).transpose()?.unwrap_or(2),..Params::default()};
 run_files(Path::new(&a[0]),Path::new(&a[1]),Path::new(&a[2]),Path::new(&a[3]),&p).map_err(Into::into)
}
