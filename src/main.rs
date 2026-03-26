//! Example: AIG technology mapping with `emapper`.

use std::path::PathBuf;

use emapper::aig::Aig;
use emapper::mapping::{
    EmapParams, NodeBindingKind, binding_as_instance, emap_with_params, write_mapped_verilog,
};
use emapper::techlib::{MappingObjective, TechLib};

fn main() {
    let aig_path = "tools/mockturtle/experiments/benchmarks/adder.aig";
    let genlib_path = "tools/mockturtle/experiments/cell_libraries/asap7.genlib";

    let mut aig = Aig::from_aiger(aig_path);
    aig.cleanup_graph();

    let tech = TechLib::from_genlib(genlib_path, 6).expect("failed to load genlib");

    aig.to_graphviz("before.dot").expect("graphviz before");

    let res = emap_with_params(
        &aig,
        &tech,
        EmapParams {
            max_cuts: 16,
            max_inputs: 6,
            wire_delay: 0.0,
            use_egg: true,
            cache_egg: true,
            cache_tt: true,
            objective: MappingObjective::DelayFirst,
            critical_path_mixed_objective: false,
            area_flow_passes: 3,
            exact_area_passes: 2,
            max_cover_nodes: 64,
            use_dual_phase_delay: true,
            mapper_name: "egraph-based-mapper",
            abc_genlib_path: Some(PathBuf::from(genlib_path)),
            verilog_module_name: "top",
        },
    );

    let (p_cut, p_egg, p_tt, p_match) = res.perf.ratio_percent();
    println!("mapper: {}", res.mapper_name);
    println!(
        "total area = {:.2}, max delay = {:.2}, gates (ABC nd) = {}, depth (ABC lev) = {}, bindings = {}",
        res.total_area,
        res.max_delay,
        res.gate_count,
        res.logic_depth,
        res.bindings.len()
    );
    println!(
        "profile: cut_gen={:.3}% egg={:.3}% tt={:.3}% tech_match={:.3}%",
        p_cut, p_egg, p_tt, p_match
    );

    let mut bindings = res.bindings.clone();
    bindings.sort_by_key(|b| b.node);
    for b in &bindings {
        match &b.kind {
            NodeBindingKind::Direct { cell } => {
                let (ins, out) = binding_as_instance(b).expect("direct binding");
                println!(
                    "n{}: {} ({} inputs) area {:.2} delay {:.2} in={:?} out={:?}",
                    b.node, cell.cell_name, cell.n_inputs, b.area, b.delay, ins, out
                );
            }
            NodeBindingKind::MappedCover { graph } => {
                println!(
                    "n{}: mapped-cover area {:.2} delay {:.2} gates {}",
                    b.node,
                    b.area,
                    b.delay,
                    graph.nodes.len()
                );
            }
        }
    }

    let f = std::fs::File::create("mapped.v").expect("create mapped.v");
    write_mapped_verilog(&aig, &tech, &res, f, "top").expect("write verilog");
    println!("Wrote mapped Verilog to mapped.v");

    aig.to_graphviz("after.dot").expect("graphviz after");
}
