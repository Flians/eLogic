use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use emapper::aig::Aig;
use emapper::mapping::{emap_with_params, EmapParams};
use emapper::techlib::{MappingObjective, TechLib};

fn main() {
    let mut args = std::env::args().skip(1);
    let genlib = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| usage_exit());
    let aig = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| usage_exit());
    let verilog_out = args.next().map(PathBuf::from);

    let mut aig_graph = Aig::from_aiger(aig.to_string_lossy().as_ref());
    aig_graph.cleanup_graph();

    let tech = TechLib::from_genlib(&genlib, 6).expect("failed to load genlib");

    let use_egg = std::env::var("EMAP_USE_EGG")
        .ok()
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(true);
    let objective = match std::env::var("EMAP_OBJECTIVE")
        .unwrap_or_else(|_| "delay".to_string())
        .to_lowercase()
        .as_str()
    {
        "area" => MappingObjective::AreaFirst,
        _ => MappingObjective::DelayFirst,
    };
    let mixed_obj = std::env::var("EMAP_MIXED_OBJECTIVE")
        .ok()
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
    let t0 = Instant::now();
    let res = emap_with_params(
        &aig_graph,
        &tech,
        EmapParams {
            max_cuts: 16,
            max_inputs: 6,
            wire_delay: 0.0,
            use_egg,
            cache_egg: use_egg,
            cache_tt: true,
            objective,
            critical_path_mixed_objective: mixed_obj,
            area_flow_passes: std::env::var("EMAP_AREA_FLOW_PASSES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1),
            exact_area_passes: std::env::var("EMAP_EXACT_AREA_PASSES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1),
            max_cover_nodes: std::env::var("EMAP_MAX_COVER_NODES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(64),
            mapper_name: "egraph-based-mapper",
        },
    );
    let runtime = t0.elapsed().as_secs_f64();
    let (p_cut, p_egg, p_tt, p_match) = res.perf.ratio_percent();

    // Depth proxy: LUT/cut depth (levels of mapped nodes)
    // depth(node) = 1 + max depth(leaf) over mapped leaves; PIs/consts have depth 0.
    let mut bind_set = std::collections::HashSet::<usize>::new();
    for b in &res.bindings {
        bind_set.insert(b.node);
    }
    let mut depth_memo: HashMap<usize, u32> = HashMap::new();
    fn depth_of(
        node: usize,
        binds: &Vec<emapper::mapping::NodeBinding>,
        bind_set: &std::collections::HashSet<usize>,
        memo: &mut HashMap<usize, u32>,
    ) -> u32 {
        if let Some(&d) = memo.get(&node) {
            return d;
        }
        let b = binds.iter().find(|b| b.node == node).expect("binding");
        let mut best = 0u32;
        for &leaf in &b.leaves {
            if leaf == 0 {
                continue;
            }
            if bind_set.contains(&leaf) {
                best = best.max(depth_of(leaf, binds, bind_set, memo));
            }
        }
        let d = best + 1;
        memo.insert(node, d);
        d
    }
    let mut max_depth = 0u32;
    for b in &res.bindings {
        max_depth = max_depth.max(depth_of(b.node, &res.bindings, &bind_set, &mut depth_memo));
    }

    // Always produce a netlist for CEC; use user path when provided.
    let (vpath, cleanup_tmp) = if let Some(vout) = verilog_out {
        (vout, false)
    } else {
        let mut tmp = std::env::temp_dir();
        tmp.push(format!("bench_rs_emap_{}_{}.v", std::process::id(), chrono_like_stamp()));
        (tmp, true)
    };
    let f = std::fs::File::create(&vpath).expect("create verilog");
    emapper::mapping::write_mapped_verilog(&aig_graph, &tech, &res, f, "top")
        .expect("write verilog");
    let nec = run_abc_cec(&genlib, &aig, &vpath);
    if cleanup_tmp {
        let _ = std::fs::remove_file(&vpath);
    }

    // JSON benchmark output for the egraph-based mapper.
    println!(
        "{{\"mapper\":\"{}\",\"area\": {:.6}, \"gates\": {:.6}, \"delay\": {:.6}, \"depth\": {:.6}, \"runtime\": {:.6}, \"nec\": {:.6}}}",
        res.mapper_name,
        res.total_area as f64,
        res.bindings.len() as f64,
        res.max_delay as f64,
        max_depth as f64,
        runtime,
        nec
    );
    eprintln!(
        "profile: cut_gen={:.3}% egg={:.3}% tt={:.3}% tech_match={:.3}% (accounted {:.6}s, direct_tt_hit={}, direct_variant_reject={}, egg_attempt={}, egg_build_ok={}, egg_coverable_hit={}, egg_proxy_fallback={}, egg_reject_non_improving={}, egg_miss={})",
        p_cut,
        p_egg,
        p_tt,
        p_match,
        res.perf.total_accounted().as_secs_f64()
        ,
        res.perf.direct_tt_hit,
        res.perf.direct_variant_reject,
        res.perf.egg_attempt,
        res.perf.egg_build_ok,
        res.perf.egg_coverable_hit,
        res.perf.egg_proxy_fallback,
        res.perf.egg_reject_non_improving,
        res.perf.egg_miss
    );
}

fn usage_exit() -> ! {
    eprintln!("Usage: bench_rs_emap genlibPath aigPath [verilogPath]");
    std::process::exit(1);
}

fn run_abc_cec(genlib: &PathBuf, aig: &PathBuf, vpath: &PathBuf) -> f64 {
    let mut impl_aig = std::env::temp_dir();
    impl_aig.push(format!(
        "bench_rs_emap_impl_{}_{}.aig",
        std::process::id(),
        chrono_like_stamp()
    ));

    let cmd_build = format!(
        "read_library {}; read_verilog -m {}; strash; write_aiger {}",
        genlib.display(),
        vpath.display(),
        impl_aig.display()
    );
    let out_build = Command::new("abc").args(["-c", &cmd_build]).output();
    let Ok(out_build) = out_build else {
        return -1.0;
    };
    if !out_build.status.success() || !impl_aig.exists() {
        let _ = std::fs::remove_file(&impl_aig);
        return -1.0;
    }

    let cmd_cec = format!("cec -n {} {}", aig.display(), impl_aig.display());
    let out = Command::new("abc").args(["-c", &cmd_cec]).output();
    let _ = std::fs::remove_file(&impl_aig);
    let Ok(out) = out else {
        return -1.0;
    };
    let mut txt = String::new();
    txt.push_str(&String::from_utf8_lossy(&out.stdout));
    txt.push_str(&String::from_utf8_lossy(&out.stderr));
    if txt.contains("Networks are equivalent") || txt.contains("UNSAT") {
        1.0
    } else if txt.contains("Networks are NOT EQUIVALENT") || txt.contains("SAT") {
        0.0
    } else {
        -1.0
    }
}

fn chrono_like_stamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{ns}")
}

