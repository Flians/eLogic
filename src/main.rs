//! Example: AIG technology mapping with `emapper`.
//!
//! Environment (optional):
//! - `EMAPPER_GENLIB`: path to `.genlib` (default: `asap7.genlib` under mockturtle experiments).
//! - `EMAPPER`: path to `.aig` (default: `adder.aig`). For **comparing delay** between settings,
//!   use `tools/mockturtle/experiments/benchmarks/c432.aig` — small circuits barely move `max_delay`.
//! - `EMAPPER_MAX_INPUTS`: genlib + mapper cut width (default **9**; lower for a faster demo —
//!   `from_genlib` index build can take a long time on large libs).
//! - `EMAPPER_MAX_CUTS`, `EMAPPER_AREA_FLOW_PASSES`, `EMAPPER_EXACT_AREA_PASSES`
//! - `EMAPPER_MAPPING_AREA_COST`: `flow` (default) or `physical`
//! - `EMAPPER_OBJECTIVE`: `delay` (default) or `area`
//! - `EMAPPER_MIXED_OBJECTIVE`: `0`/`1` — if `1`, Stage 1b rematch uses delay-first on critical ANDs
//!   and area-first off-critical; if `0`, Stage 1b still runs (same cut pool + critical path for supergate)
//!   but uses delay-first objective everywhere (`critical_path_mixed_objective` in `EmapParams`).
//! - `EMAPPER_TECHLIB_CACHE` or `MIGBALANCE_TECHLIB_CACHE`: explicit JSON cache path; if unset, a
//!   sidecar `*.m{max_inputs}.ttcache.*` next to the genlib is used.
//! - `EMAPPER_EGG_ITER_LIMIT`, `EMAPPER_EGG_NODE_LIMIT`: cap egg saturation (defaults 30 / 10000);
//!   lower for faster runs.

use std::io::Write;
use std::path::Path;

use emapper::aig::Aig;
use emapper::mapping::{
    emap_params_from_env, emap_with_params,
    techlib_cache_path_near_genlib, write_mapped_verilog,
};
use emapper::run_abc_cec;
use emapper::run_abc_print_stats;
use emapper::techlib::TechLib;

fn main() {
    let aig_path = std::env::var("EMAPPER")
        .unwrap_or_else(|_| "tools/mockturtle/experiments/benchmarks/adder.aig".to_string());
    let genlib_path = std::env::var("EMAPPER_GENLIB")
        .unwrap_or_else(|_| "tools/mockturtle/experiments/cell_libraries/asap7.genlib".to_string());
    let mapped_v_path = "mapped.v";
    let params = emap_params_from_env();

    eprintln!("EMAPPER: loading AIG from {aig_path}");
    let mut aig = Aig::from_aiger(&aig_path);
    aig.cleanup_graph();
    eprintln!(
        "EMAPPER: loaded ({} nodes in graph); reading genlib {}",
        aig.node_count(),
        genlib_path
    );

    eprintln!(
        "EMAPPER: parsing genlib + TT index (max_inputs={}; first run with a cold cache can be very slow on asap7)…",
        params.max_inputs
    );
    let _ = std::io::stderr().flush();
    let tech_cache = std::env::var("EMAPPER_TECHLIB_CACHE")
        .or_else(|_| std::env::var("MIGBALANCE_TECHLIB_CACHE"))
        .ok()
        .filter(|s| !s.is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            techlib_cache_path_near_genlib(Path::new(&genlib_path), params.max_inputs)
        });
    let tech = TechLib::from_genlib_cached(&genlib_path, params.max_inputs, &tech_cache)
        .expect("failed to load genlib");
    eprintln!("EMAPPER: technology mapping (egg + cuts; may take a bit on larger designs)…");

    // aig.to_graphviz("before.dot").expect("graphviz before");

    let res = emap_with_params(&aig, &tech, params);

    let (p_cut, p_egg, p_tt, p_match) = res.perf.ratio_percent();
    println!("mapper: {}", res.mapper_name);
    println!(
        "mapper internal QoR: total_area = {:.2}, max_delay = {:.2}, bindings = {}",
        res.total_area,
        res.max_delay,
        res.bindings.len(),
    );
    println!(
        "profile: cut_gen={:.3}% egg={:.3}% tt={:.3}% tech_match={:.3}%",
        p_cut, p_egg, p_tt, p_match
    );

    /*
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
            NodeBindingKind::MappedCover { graph, .. } => {
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
    */

    let f = std::fs::File::create(mapped_v_path).expect("create mapped.v");
    write_mapped_verilog(&aig, &tech, &res, f, "top").expect("write verilog");
    println!("Wrote mapped Verilog to {mapped_v_path}");

    match run_abc_print_stats(Path::new(&genlib_path), Path::new(mapped_v_path)) {
        Some(abc) => println!(
            "ABC print_stats (same genlib + mapped .v): nd = {}, area = {:.2}, delay = {:.2}, lev = {}",
            abc.nd, abc.area, abc.delay, abc.lev
        ),
        None => eprintln!(
            "ABC print_stats: unavailable (set PATH to yosys-abc or abc, or BENCH_RS_EMAP_ABC)"
        ),
    }

    let nec = run_abc_cec(
        Path::new(&genlib_path),
        Path::new(&aig_path),
        Path::new(mapped_v_path),
    );
    match nec {
        1.0 => println!("CEC: golden AIG and mapped Verilog are combinatorially equivalent."),
        0.0 => {
            eprintln!("CEC: networks are NOT equivalent.");
            std::process::exit(1);
        }
        _ => {
            eprintln!(
                "CEC: could not verify (nec={nec}). Install ABC (yosys-abc/abc) or set BENCH_RS_EMAP_ABC; use BENCH_RS_EMAP_CEC_DEBUG=1 for details."
            );
            std::process::exit(2);
        }
    }

    // aig.to_graphviz("after.dot").expect("graphviz after");
}
