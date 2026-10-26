use emapper::techlib::TechLib;
use emapper::netlist_metrics::measure_json;
fn main()->Result<(),Box<dyn std::error::Error>>{let a:Vec<_>=std::env::args().skip(1).collect();if a.len()!=2{return Err("expected genlib mapped.json".into());}let t=TechLib::from_genlib_directed(&a[0])?;let m=measure_json(&std::fs::read_to_string(&a[1])?,&t)?;println!("{}",serde_json_placeholder(m));Ok(())}
fn serde_json_placeholder(m:emapper::netlist_metrics::NetlistMetrics)->String{format!("{{\"area\":{},\"delay\":{},\"instances\":{},\"multioutput_instances\":{}}}",m.area,m.delay,m.instances,m.multioutput_instances)}
