use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Instant;

use emapper::aig::Aig;
use emapper::mapping::{EmapParams, emap_with_params};
use emapper::techlib::{MappingObjective, TechLib};

fn main() {
    let mut a = std::env::args().skip(1);
    let genlib: PathBuf = a
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| "tools/mockturtle/experiments/cell_libraries/asap7.genlib".into());
    let aig: PathBuf = a
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| "tools/mockturtle/experiments/benchmarks/adder.aig".into());
    let verilog_out = a.next().map(PathBuf::from);

    let mut aig_graph = Aig::from_aiger(aig.to_string_lossy().as_ref());
    aig_graph.cleanup_graph();
    let tech = TechLib::from_genlib(&genlib, 6).expect("failed to load genlib");

    let mut ps = emap_params_from_env();
    ps.abc_genlib_path = Some(genlib.clone());
    let t0 = Instant::now();
    let res = emap_with_params(&aig_graph, &tech, ps);
    let runtime = t0.elapsed().as_secs_f64();
    let (p_cut, p_egg, p_tt, p_match) = res.perf.ratio_percent();

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
    emapper::mapping::write_mapped_verilog(&aig_graph, &tech, &res, f, "top")
        .expect("write verilog");
    let nec = run_abc_cec(&genlib, &aig, &vpath);
    if cleanup_tmp {
        let _ = std::fs::remove_file(&vpath);
    }

    println!(
        "{{\"mapper\":\"{}\",\"area\": {:.6}, \"gates\": {:.6}, \"delay\": {:.6}, \"depth\": {:.6}, \"runtime\": {:.6}, \"nec\": {:.6}}}",
        res.mapper_name,
        res.total_area as f64,
        res.gate_count as f64,
        res.max_delay as f64,
        res.logic_depth as f64,
        runtime,
        nec
    );
    eprintln!(
        "profile: cut_gen={:.3}% egg={:.3}% tt={:.3}% tech_match={:.3}% (accounted {:.6}s, direct_tt_hit={}, direct_variant_reject={}, egg_attempt={}, egg_build_ok={}, egg_coverable_hit={}, egg_proxy_fallback={}, egg_reject_non_improving={}, egg_miss={})",
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
        res.perf.egg_miss
    );
}

fn emap_params_from_env() -> EmapParams {
    let use_egg = match std::env::var("EMAP_USE_EGG") {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => true,
    };
    let objective = match std::env::var("EMAP_OBJECTIVE")
        .unwrap_or_else(|_| "delay".to_string())
        .to_lowercase()
    {
        ref s if s == "area" => MappingObjective::AreaFirst,
        _ => MappingObjective::DelayFirst,
    };
    let mixed = std::env::var("EMAP_MIXED_OBJECTIVE")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let env_u = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let dual = std::env::var("EMAP_SINGLE_PHASE_DELAY")
        .ok()
        .is_none_or(|v| v != "1" && !v.eq_ignore_ascii_case("true"));

    EmapParams {
        max_cuts: 16,
        max_inputs: 6,
        wire_delay: 0.0,
        use_egg,
        cache_egg: use_egg,
        cache_tt: true,
        objective,
        critical_path_mixed_objective: mixed,
        area_flow_passes: env_u("EMAP_AREA_FLOW_PASSES", 3),
        exact_area_passes: env_u("EMAP_EXACT_AREA_PASSES", 2),
        max_cover_nodes: env_u("EMAP_MAX_COVER_NODES", 64),
        use_dual_phase_delay: dual,
        mapper_name: "egraph-based-mapper",
        abc_genlib_path: None,
        verilog_module_name: "top",
    }
}

fn usage_exit() -> ! {
    eprintln!("Usage: bench_rs_emap genlibPath aigPath [verilogPath]");
    eprintln!("Env: BENCH_RS_EMAP_ABC=…  BENCH_RS_EMAP_CEC_DEBUG=1");
    std::process::exit(1);
}

static ABC_FOR_CEC: OnceLock<Option<PathBuf>> = OnceLock::new();

fn abc_for_cec() -> Option<PathBuf> {
    ABC_FOR_CEC
        .get_or_init(|| {
            if let Ok(p) = std::env::var("BENCH_RS_EMAP_ABC") {
                let pb = PathBuf::from(p.trim());
                return if pb.as_os_str().is_empty() || !pb.is_file() {
                    None
                } else {
                    Some(pb)
                };
            }
            ["yosys-abc", "abc"]
                .into_iter()
                .find(|n| Command::new(n).arg("-h").output().is_ok())
                .map(PathBuf::from)
        })
        .clone()
}

fn parse_cec_outcome(txt: &str) -> Option<f64> {
    let lower = txt.to_ascii_lowercase();
    if lower.contains("networks are equivalent") || txt.contains("UNSAT") {
        return Some(1.0);
    }
    if lower.contains("networks are not equivalent") {
        return Some(0.0);
    }
    if txt.contains("SAT") && !txt.contains("UNSAT") {
        return Some(0.0);
    }
    None
}

fn shell_path(p: &Path) -> String {
    let s = p.to_string_lossy();
    if s.contains(' ') {
        format!("\"{s}\"")
    } else {
        s.into_owned()
    }
}

fn abs(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

fn cec_debug(msg: &str) {
    if std::env::var("BENCH_RS_EMAP_CEC_DEBUG").as_deref() == Ok("1") {
        eprintln!("{msg}");
    }
}

fn run_abc_cec(genlib: &Path, aig: &Path, vpath: &Path) -> f64 {
    let Some(exe) = abc_for_cec() else {
        cec_debug("BENCH_RS_EMAP_CEC_DEBUG: no ABC (set BENCH_RS_EMAP_ABC or PATH yosys-abc/abc)");
        return -1.0;
    };
    let g = abs(genlib);
    let a = abs(aig);
    let v = abs(vpath);
    let mut impl_aig = std::env::temp_dir();
    impl_aig.push(format!(
        "bench_rs_emap_impl_{}_{}.aig",
        std::process::id(),
        nanos_stamp()
    ));
    let impl_s = shell_path(&impl_aig);

    let build = format!(
        "read_library {}; read_verilog -m {}; strash; write_aiger {}",
        shell_path(&g),
        shell_path(&v),
        impl_s
    );
    let out_b = match Command::new(&exe).args(["-c", &build]).output() {
        Ok(o) => o,
        Err(_) => {
            cec_debug("BENCH_RS_EMAP_CEC_DEBUG: spawn failed (read_library/read_verilog)");
            return -1.0;
        }
    };
    if !out_b.status.success() || !impl_aig.exists() {
        cec_debug(&format!(
            "BENCH_RS_EMAP_CEC_DEBUG: ABC build failed (exit {:?})\n--- stdout ---\n{}\n--- stderr ---\n{}",
            out_b.status.code(),
            String::from_utf8_lossy(&out_b.stdout),
            String::from_utf8_lossy(&out_b.stderr)
        ));
        let _ = std::fs::remove_file(&impl_aig);
        return -1.0;
    }

    let cec = format!("cec -n {} {}", shell_path(&a), impl_s);
    let out = match Command::new(&exe).args(["-c", &cec]).output() {
        Ok(o) => o,
        Err(_) => {
            cec_debug("BENCH_RS_EMAP_CEC_DEBUG: spawn failed (cec)");
            let _ = std::fs::remove_file(&impl_aig);
            return -1.0;
        }
    };
    let _ = std::fs::remove_file(&impl_aig);

    let txt = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if let Some(x) = parse_cec_outcome(&txt) {
        return x;
    }
    cec_debug(&format!(
        "BENCH_RS_EMAP_CEC_DEBUG: unrecognized cec output (exit {:?})\n{txt}",
        out.status.code()
    ));
    -1.0
}

fn nanos_stamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos().to_string())
        .unwrap_or_else(|_| "0".into())
}
