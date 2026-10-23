//! **P0 decision experiment** — joint DAG-aware ILP covering on the *plain AIG cut space*
//! (no e-graph), measured against `emap`.
//!
//! Usage:
//!   bench_rs_ilp <genlib> <aig> [mapped.v]
//!
//! Env:
//!   ILP_WEIGHTS   comma-separated arrival weights to sweep (default `0`)
//!   ILP_TIME      per-solve CBC time limit in seconds (default 60)
//!   ILP_K         max cut leaves, <= 6 (default 6)
//!   ILP_CUTS      max cuts per node (default 64)
//!   ILP_MATCHES   max matches per class after pruning (default 24)
//!   ILP_GAP       CBC ratio gap (default 0.001)
//!   ILP_NOCEC     set to skip ABC CEC (faster sweeps)
//!
//! Prints one JSON object per weight on stdout.

use std::path::PathBuf;
use std::time::Instant;

use emapper::aig::Aig;
use emapper::ilp_cover::{required_classes, solve, write_verilog, IlpParams};
use emapper::mapping::techlib_cache_path_near_genlib;
use emapper::techlib::TechLib;
use emapper::{run_abc_cec, run_abc_print_stats};

fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn nanos_stamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos().to_string())
        .unwrap_or_else(|_| "0".into())
}

fn main() {
    let mut a = std::env::args().skip(1);
    let genlib: PathBuf = a
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| "tools/mockturtle/experiments/cell_libraries/asap7.genlib".into());
    let aig_path: PathBuf = a
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| "tools/mockturtle/experiments/benchmarks/c432.aig".into());
    let verilog_out = a.next().map(PathBuf::from);

    let weights: Vec<f64> = std::env::var("ILP_WEIGHTS")
        .unwrap_or_else(|_| "0".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    /* `ILP_SLACK` accepts a comma-separated list of arrival-target relaxations, which is what traces
       the area–delay Pareto front. The token `area` (or `-1`) selects the plain area-only cover. */
    let slacks: Vec<Option<f32>> = match std::env::var("ILP_SLACK") {
        Ok(v) if !v.trim().is_empty() => v
            .split(',')
            .map(|t| {
                let t = t.trim();
                if t.eq_ignore_ascii_case("area") || t == "-1" {
                    None
                } else {
                    t.parse().ok()
                }
            })
            .collect(),
        _ => vec![None],
    };
    let do_cec = std::env::var_os("ILP_NOCEC").is_none();

    let mut aig = Aig::from_aiger(aig_path.to_string_lossy().as_ref());
    aig.cleanup_graph();

    /* `ILP_EGRAPH=1` inserts stage S1+S2: saturate the AIG in an e-graph and map the restructured
       AIG instead. Everything downstream is unchanged, so any delta is attributable to the
       e-graph's realisation space alone. */
    let mut sat: Option<emapper::egraph_sat::SatStats> = None;
    if std::env::var_os("ILP_EGRAPH").is_some() {
        let sp = emapper::egraph_sat::sat_params_from_env();
        match emapper::egraph_sat::restructure(&aig, &sp) {
            Some((r, st)) => {
                eprintln!(
                    "egraph: nodes {} -> {}  eclasses={} enodes={} iters={} stop={} ({:.2}s)",
                    st.nodes_before, st.nodes_after, st.eclasses, st.enodes, st.iters,
                    st.stop_reason, st.secs
                );
                sat = Some(st);
                aig = r;
            }
            None => eprintln!("egraph: saturation failed, mapping the original AIG"),
        }
    }

    let max_inputs = 9usize;
    let tech_cache = std::env::var("EMAPPER_TECHLIB_CACHE")
        .or_else(|_| std::env::var("MIGBALANCE_TECHLIB_CACHE"))
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| techlib_cache_path_near_genlib(&genlib, max_inputs));
    let tech = TechLib::from_genlib_cached(&genlib, max_inputs, &tech_cache)
        .expect("failed to load genlib");

    let (required, _ports) = required_classes(&aig);

    let base = IlpParams {
        max_cut_leaves: env_usize("ILP_K", 6).min(6),
        max_cuts_per_node: env_usize("ILP_CUTS", 64),
        max_matches_per_class: env_usize("ILP_MATCHES", 24),
        time_limit_secs: env_f64("ILP_TIME", 60.0),
        ratio_gap: env_f64("ILP_GAP", 0.001),
        use_levels: std::env::var_os("ILP_LEVELS").is_some(),
        greedy_rounds: env_usize("ILP_ROUNDS", 12),
        timed_slack: std::env::var("ILP_SLACK").ok().and_then(|s| s.parse().ok()),
        delay_weight: env_f64("ILP_DW", 0.0),
        exact_rounds: env_usize("ILP_EXACT", 2),
        use_multioutput: std::env::var_os("ILP_MULTIOUTPUT").is_some(),
        mo_max_sets: env_usize("ILP_MO_MAXSETS", 16384),
        mo_max_partners: env_usize("ILP_MO_PARTNERS", 8),
        mo_max_cands: env_usize("ILP_MO_MAXCANDS", 200_000),
        arrival_weight: 0.0,
    };

    /* `ILP_DC_STATS=1` only measures how much don't-care exists over the cuts, then exits. */
    if let Ok(v) = std::env::var("ILP_DC_STATS") {
        if v != "0" {
            let limit: u32 = std::env::var("ILP_DC_SUPPORT")
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(10);
            let cone: usize = std::env::var("ILP_DC_CONE")
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(4000);
            let st = emapper::ilp_cover::measure_cut_dc(&aig, &tech, &base, limit, cone);
            println!(
                "{{\"bench\":\"x\",\"dc_cuts_total\":{},\"dc_cuts_measured\":{},\"dc_cuts_with_dc\":{},\
\"dc_unreachable\":{},\"dc_total_points\":{},\"dc_limit\":{},\"dc_helps\":{},\"dc_area_exact\":{:.4},\"dc_area_with_dc\":{:.4}}}",
                st.cuts_total,
                st.cuts_measured,
                st.cuts_with_dc,
                st.unreachable_points,
                st.total_points,
                limit,
                st.cuts_dc_helps,
                st.area_exact,
                st.area_with_dc
            );
            let mut hist: Vec<(usize, usize)> = st
                .support_hist
                .iter()
                .enumerate()
                .filter(|(_, c)| **c > 0)
                .map(|(m, c)| (m, *c))
                .collect();
            hist.sort_unstable();
            eprintln!("support histogram (popcount: cuts): {hist:?}");
            return;
        }
    }

    /* `ILP_S3=1` maps with cell matches **inside** the e-graph (EqMap's architecture). */
    if std::env::var_os("ILP_S3").is_some() {
        let sp = emapper::egraph_sat::sat_params_from_env();
        let Some(cc) = emapper::egraph_sat::map_with_cells(&aig, &tech, &sp) else {
            println!("{{\"mapper\":\"s3_cells\",\"error\":\"s3 failed\"}}");
            return;
        };
        eprintln!(
            "s3: eclasses={} enodes={} iters={} stop={} model_area={:.3} cells={} ({:.2}s)",
            cc.stats.eclasses, cc.stats.enodes, cc.stats.iters, cc.stats.stop_reason,
            cc.model_area, cc.cells, cc.stats.secs
        );
        let vpath = verilog_out.clone().unwrap_or_else(|| {
            let mut p = std::env::temp_dir();
            p.push(format!("bench_rs_s3_{}_{}.v", std::process::id(), nanos_stamp()));
            p
        });
        std::fs::write(&vpath, &cc.verilog).expect("write verilog");
        let stats = run_abc_print_stats(&genlib, &vpath);
        let nec = if do_cec { run_abc_cec(&genlib, &aig_path, &vpath) } else { f64::NAN };
        let (a, dl, lv, nd) = stats
            .as_ref()
            .map_or((f32::NAN, f32::NAN, 0, 0), |s| (s.area, s.delay, s.lev, s.nd));
        println!(
            "{{\"bench\":\"{}\",\"mapper\":\"s3_cells\",\"slack\":-1,\"weight\":0,\
\"abc_area\":{a:.4},\"abc_delay\":{dl:.4},\"abc_lev\":{lv},\"abc_nd\":{nd},\
\"model_area\":{:.4},\"model_delay\":0.0,\"model_depth\":0,\"nec\":{nec},\"runtime\":{:.3},\
\"eg_classes\":{},\"eg_secs\":{:.3}}}",
            aig_path.file_stem().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default(),
            cc.model_area,
            cc.stats.secs,
            cc.stats.eclasses,
            cc.stats.secs
        );
        if verilog_out.is_none() {
            let _ = std::fs::remove_file(&vpath);
        }
        return;
    }

    /* `ILP_MO_STATS=1` measures the multi-output payload, then exits. */
    if std::env::var("ILP_MO_STATS").ok().as_deref().is_some_and(|v| v != "0") {
        let k: usize = std::env::var("ILP_MO_K").ok().and_then(|x| x.parse().ok()).unwrap_or(4);
        let st = emapper::ilp_cover::measure_multioutput_payload(&aig, &tech, &base, k, 400);
        println!(
            "{{\"mo_nodes\":{},\"mo_leafsets\":{},\"mo_pairs\":{},\"mo_avail\":{},\"mo_cheaper\":{},\
\"mo_unreal\":{},\"mo_saving\":{:.4},\"mo_best\":{:.4}}}",
            st.nodes, st.leaf_sets_with_pairs, st.pairs, st.pairs_available,
            st.pairs_cheaper, st.pairs_with_unrealisable_half, st.saving_total, st.best_saving
        );
        return;
    }

    let bench = aig_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "top".into());

    /* The best cut-size limit is circuit-dependent — a carry chain wants 3-input cells while a
       multiplier's compound cells want wide cuts — so this is swept like the slack/weight axes. Each
       setting is a full independent solve; the caller keeps the smallest area. */
    let ks: Vec<usize> = std::env::var("ILP_KS")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(|v| {
            v.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .map(|k| k.min(6).max(2))
                .collect()
        })
        .unwrap_or_else(|| vec![base.max_cut_leaves]);

    let cut_caps: Vec<usize> = std::env::var("ILP_CUTS")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(|v| {
            v.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![base.max_cuts_per_node]);

    for k in &ks {
      for cut_cap in &cut_caps {
      for slack in &slacks {
      for w in &weights {
        let w = *w;
        let ps = IlpParams {
            arrival_weight: w,
            timed_slack: *slack,
            max_cut_leaves: *k,
            max_cuts_per_node: *cut_cap,
            ..base.clone()
        };
        let t0 = Instant::now();
        let cov = match solve(&aig, &tech, &required, &ps) {
            Ok(c) => c,
            Err(e) => {
                println!(
                    "{{\"bench\":\"{bench}\",\"weight\":{w},\"slack\":{},\"error\":\"{}\"}}",
                    slack.map_or(-1.0, f64::from),
                    e.to_string().replace('"', "'")
                );
                continue;
            }
        };
        let runtime = t0.elapsed().as_secs_f64();

        let vpath = verilog_out.clone().unwrap_or_else(|| {
            let mut p = std::env::temp_dir();
            p.push(format!("bench_rs_ilp_{}_{}.v", std::process::id(), nanos_stamp()));
            p
        });
        {
            let f = std::fs::File::create(&vpath).expect("create verilog");
            write_verilog(&aig, &cov, f, "top").expect("write verilog");
        }

        let stats = run_abc_print_stats(&genlib, &vpath);
        let nec = if do_cec {
            run_abc_cec(&genlib, &aig_path, &vpath)
        } else {
            f64::NAN
        };

        let (abc_area, abc_delay, abc_lev, abc_nd) = stats.as_ref().map_or((f32::NAN, f32::NAN, 0, 0), |s| {
            (s.area, s.delay, s.lev, s.nd)
        });

        println!(
            "{{\"bench\":\"{bench}\",\"mapper\":\"ilp_cover\",\"weight\":{w},\"slack\":{},\
\"abc_area\":{abc_area:.4},\"abc_delay\":{abc_delay:.4},\"abc_lev\":{abc_lev},\"abc_nd\":{abc_nd},\
\"model_area\":{:.4},\"model_delay\":{:.4},\"model_depth\":{},\
\"runtime\":{runtime:.4},\"nec\":{nec},\"k\":{},\"cut_cap\":{},\"status\":\"{}\",\
\"matches\":{},\"matches_raw\":{},\"cuts\":{},\"inverters\":{},\"multiouts\":{},\
\"mo_selected\":{},\"mo_cuts_indexed\":{},\"mo_seeds\":{},\"mo_partner_edges\":{},\
\"mo_hit_cap\":{},\"mo_bucket_overflows\":{},\"mo_identical_pairs\":{},\"mo_subset_pairs\":{},\
\"greedy_area\":{:.4},\"used_greedy\":{},\
\"build_secs\":{:.3},\"solve_secs\":{:.3},\"obj\":{:.4},\
\"eg_before\":{},\"eg_after\":{},\"eg_ands_before\":{},\"eg_ands_after\":{},\"eg_cyclic\":{},\"eg_classes\":{},\"eg_secs\":{:.3}}}",
            slack.map_or(-1.0, f64::from),
            cov.area_model,
            cov.arrival_model,
            cov.depth_model,
            ps.max_cut_leaves, // `"k"` sits after `model_depth` in the format string
            ps.max_cuts_per_node,
            cov.stats.status,
            cov.stats.matches_kept,
            cov.stats.matches_generated,
            cov.stats.cuts,
            cov.stats.inverters,
            cov.matches.iter().filter(|m| !m.extra_outs.is_empty()).count(),
            cov.stats.mo_selected,
            cov.stats.mo.cuts_indexed,
            cov.stats.mo.seeds,
            cov.stats.mo.partner_edges,
            cov.stats.mo.hit_total_cap,
            cov.stats.mo.bucket_overflows,
            cov.stats.mo.identical_pairs,
            cov.stats.mo.subset_pairs,
            cov.stats.greedy_area,
            cov.stats.used_greedy,
            cov.stats.build_secs,
            cov.stats.solve_secs,
            cov.stats.objective,
            sat.as_ref().map_or(0, |s| s.nodes_before),
            sat.as_ref().map_or(0, |s| s.nodes_after),
            sat.as_ref().map_or(0, |s| s.ands_before),
            sat.as_ref().map_or(0, |s| s.ands_after),
            sat.as_ref().map_or(0, |s| s.cyclic_classes),
            sat.as_ref().map_or(0, |s| s.eclasses),
            sat.as_ref().map_or(0.0, |s| s.secs)
        );

        if verilog_out.is_none() {
            let _ = std::fs::remove_file(&vpath);
        }
      }
      }
      }
    }
}
