//! Library models, common netlist metrics, and strict named-port CEC for baseline experiments.
use emapper::egraph_map::{verify_with_models, verify_with_verilog_models, write_library_verilog};
use emapper::netlist_metrics::measure_json;
use emapper::techlib::TechLib;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("Usage: bench_rs_netlist_eval <genlib> <golden.aig> <mapped.v> <mapped.json>\nOr: bench_rs_netlist_eval <genlib> --models <cells.v> <library.json>\nOr: bench_rs_netlist_eval --cec-models <models.v> <golden.aig> <mapped.v>".into());
    }
    if args[0] == "--cec-models" {
        let cec = verify_with_verilog_models(
            &std::fs::read_to_string(&args[1])?,
            Path::new(&args[2]),
            Path::new(&args[3]),
        )?;
        println!("{{\"cec\":{}}}", u8::from(cec));
        if !cec {
            return Err("non-equivalent modeled netlist".into());
        }
        return Ok(());
    }
    let tech = TechLib::from_genlib(&args[0], 6)?;
    if args[1] == "--models" {
        write_library_verilog(&tech, std::fs::File::create(&args[2])?)?;
        // Export authoritative parser results without making Python independently parse genlib.
        std::fs::write(&args[3], emapper::netlist_metrics::library_json(&tech)?)?;
        return Ok(());
    }
    let measured = measure_json(&std::fs::read_to_string(&args[3])?, &tech)?;
    let cec = verify_with_models(&tech, Path::new(&args[1]), Path::new(&args[2]))?;
    println!(
        "{{\"area\":{},\"delay\":{},\"instances\":{},\"multioutput_instances\":{},\"cec\":{}}}",
        measured.area,
        measured.delay,
        measured.instances,
        measured.multioutput_instances,
        u8::from(cec)
    );
    if !cec {
        return Err("non-equivalent baseline netlist".into());
    }
    Ok(())
}
