//! Benchmark mapper QoR. Use **`tools/mockturtle/experiments/benchmarks/c432.aig`** (or larger) to
//! compare **delay** — tiny benchmarks show almost no `max_delay` spread.
//!
//! Relevant env (see `EmapParams` in `emapper`; all **`EMAPPER_*`**):
//! - `EMAPPER_AREA_FLOW_PASSES`, `EMAPPER_EXACT_AREA_PASSES` — area recovery (0 = delay-first).
//!   If **unset**, this binary defaults to **1** / **1** (lighter than `main`).
//! - `EMAPPER_MAX_CUTS` — more cuts can improve delay (slower)
//! - `EMAPPER_MAPPING_AREA_COST` — `flow` (default) or `physical` for cut ranking
//! - `EMAPPER_MIXED_OBJECTIVE` — `1` = Stage 1b delay/area split by critical path; `0` = Stage 1b still runs, uniform objective
//! - `EMAPPER_USE_EGG`, `EMAPPER_OBJECTIVE`, `EMAPPER_MAX_COVER_NODES`, `EMAPPER_USE_DUAL_PHASE_DELAY`
//! - `EMAPPER_TECHLIB_CACHE` or `MIGBALANCE_TECHLIB_CACHE` — JSON cache path; if unset, a sidecar
//!   `*.m{max_inputs}.ttcache.*` next to the genlib is used.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use emapper::aig::Aig;
use emapper::emap::EmapResult;
use emapper::mapping::{
    NodeBindingKind, emap_params_from_bench_env, emap_with_params, mapped_depth,
    techlib_cache_path_near_genlib, write_mapped_verilog,
};
use emapper::run_abc_cec;
use emapper::techlib::TechLib;

fn main() {
    let mut a = std::env::args().skip(1);
    let genlib: PathBuf = a
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| "tools/mockturtle/experiments/cell_libraries/asap7.genlib".into());
    let aig: PathBuf = a
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| "tests/adder2.aig".into());
    let verilog_out = a.next().map(PathBuf::from);

    let mut aig_graph = Aig::from_aiger(aig.to_string_lossy().as_ref());
    aig_graph.cleanup_graph();
    let max_inputs = 9usize;
    let tech_cache = std::env::var("EMAPPER_TECHLIB_CACHE")
        .or_else(|_| std::env::var("MIGBALANCE_TECHLIB_CACHE"))
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| techlib_cache_path_near_genlib(&genlib, max_inputs));
    let tech =
        TechLib::from_genlib_cached(&genlib, max_inputs, &tech_cache).expect("failed to load genlib");

    let t0 = Instant::now();
    let res = emap_with_params(&aig_graph, &tech, emap_params_from_bench_env());
    let runtime = t0.elapsed().as_secs_f64();
    let (p_cut, p_egg, p_tt, p_match) = res.perf.ratio_percent();
    /* `depth`: binding-graph depth of the emitted bindings (historical metric, ignores `out_neg`).
       `depth_netlist`: cell depth of the netlist the writer actually emits, which is what ABC's
       `lev` and mockturtle's `depth_view` report. */
    let max_depth = max_binding_depth(&res);
    let netlist_depth = mapped_depth(&aig_graph, &res);

    let (vpath, cleanup_tmp) = verilog_out.map_or_else(
        || {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "bench_rs_emap_{}_{}.v",
                std::process::id(),
                nanos_stamp()
            ));
            (p, true)
        },
        |p| (p, false),
    );
    let f = std::fs::File::create(&vpath).expect("create verilog");
    write_mapped_verilog(&aig_graph, &tech, &res, f, "top").expect("write verilog");
    let nec = run_abc_cec(&genlib, &aig, &vpath);
    if cleanup_tmp {
        let _ = std::fs::remove_file(&vpath);
    }

    println!(
        "{{\"mapper\":\"{}\",\"area\": {:.6}, \"gates\": {:.6}, \"delay\": {:.6}, \"depth\": {:.6}, \"depth_netlist\": {:.6}, \"runtime\": {:.6}, \"nec\": {:.6}}}",
        res.mapper_name,
        res.total_area as f64,
        res.bindings.len() as f64,
        res.max_delay as f64,
        max_depth as f64,
        netlist_depth as f64,
        runtime,
        nec
    );
    eprintln!(
        "profile: cut_gen={:.3}% egg={:.3}% tt={:.3}% tech_match={:.3}% (accounted {:.6}s, direct_tt_hit={}, direct_variant_reject={}, egg_attempt={}, egg_build_ok={}, egg_coverable_hit={}, egg_proxy_fallback={}, egg_reject_non_improving={}, egg_miss={}, supergate_hit={}, supergate_store={})",
        p_cut,
        p_egg,
        p_tt,
        p_match,
        res.perf.total_accounted().as_secs_f64(),
        res.perf.direct_tt_hit,
        res.perf.direct_variant_reject,
        res.perf.egg_attempt,
        res.perf.egg_build_ok,
        res.perf.egg_coverable_hit,
        res.perf.egg_proxy_fallback,
        res.perf.egg_reject_non_improving,
        res.perf.egg_miss,
        res.perf.supergate_hit,
        res.perf.supergate_store
    );
}

fn max_binding_depth(res: &EmapResult) -> u32 {
    type BindMap<'a> = HashMap<(usize, bool), &'a emapper::mapping::NodeBinding>;
    let bind_map: BindMap = res
        .bindings
        .iter()
        .map(|b| ((b.node, b.phase_neg), b))
        .collect();
    let bind_nodes: HashSet<usize> = res.bindings.iter().map(|b| b.node).collect();
    let mut memo = HashMap::<(usize, bool), u32>::new();

    fn depth(
        node: usize,
        phase_neg: bool,
        binds: &BindMap<'_>,
        bind_nodes: &HashSet<usize>,
        memo: &mut HashMap<(usize, bool), u32>,
    ) -> u32 {
        if let Some(&d) = memo.get(&(node, phase_neg)) {
            return d;
        }
        let d = if let Some(b) = binds.get(&(node, phase_neg)) {
            let mut best = 0u32;
            match &b.kind {
                NodeBindingKind::Direct { cell } => {
                    for (i, &leaf) in b.leaves.iter().take(cell.n_inputs as usize).enumerate() {
                        if leaf != 0 && bind_nodes.contains(&leaf) {
                            let neg = ((cell.neg_mask >> i) & 1) == 1;
                            best = best.max(depth(leaf, neg, binds, bind_nodes, memo));
                        }
                    }
                }
                NodeBindingKind::MappedCover { .. } => {
                    for &leaf in &b.leaves {
                        if leaf != 0 && bind_nodes.contains(&leaf) {
                            best = best.max(depth(leaf, false, binds, bind_nodes, memo));
                        }
                    }
                }
            }
            best + 1
        } else if phase_neg && bind_nodes.contains(&node) {
            depth(node, false, binds, bind_nodes, memo) + 1
        } else {
            0
        };
        memo.insert((node, phase_neg), d);
        d
    }

    res.bindings
        .iter()
        .map(|b| depth(b.node, b.phase_neg, &bind_map, &bind_nodes, &mut memo))
        .max()
        .unwrap_or(0)
}

fn nanos_stamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos().to_string())
        .unwrap_or_else(|_| "0".into())
}
