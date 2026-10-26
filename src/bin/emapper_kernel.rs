//! Direct e-graph mapping, with optional multi-output cells and fixed-boundary SDC.
use emapper::aig::Aig;
use emapper::egraph_map::{map, verify_with_models, MapObjective, MapParams};
use emapper::techlib::TechLib;
use std::path::PathBuf;

fn number<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _total_window = emapper::egraph_time_policy::Window::from_env()?;
    eprintln!("configuration: EGM_TOTAL_TIME_ONLY={}", u8::from(emapper::egraph_time_policy::enabled()));
    eprintln!("configuration: EGM_EXACT_CUT_OPS={}", std::env::var("EGM_EXACT_CUT_OPS").unwrap_or_else(|_| "0".into()));
    eprintln!("configuration: EGM_EXCHANGE_TIME_FRACTION={}", std::env::var("EGM_EXCHANGE_TIME_FRACTION").unwrap_or_else(|_| "0.70".into()));
    eprintln!("configuration: EGM_WORD_REDUCE={}", std::env::var("EGM_WORD_REDUCE").unwrap_or_else(|_| "0".into()));
    eprintln!("configuration: EGM_ADMISSION_SHADOW={}", std::env::var("EGM_ADMISSION_SHADOW").unwrap_or_else(|_| "0".into()));
    eprintln!("configuration: EGM_FAST_ADMISSION_DELTA={}", std::env::var("EGM_FAST_ADMISSION_DELTA").unwrap_or_else(|_| "0".into()));
    eprintln!("configuration: EGM_WORD_TRUTH={}", std::env::var("EGM_WORD_TRUTH").unwrap_or_else(|_| "0".into()));
    eprintln!("configuration: EGM_SHARE_GENERATOR_KEYS={}", std::env::var("EGM_SHARE_GENERATOR_KEYS").unwrap_or_else(|_| "0".into()));
    eprintln!("configuration: EGM_PENDING_FRONTIER={}", std::env::var("EGM_PENDING_FRONTIER").unwrap_or_else(|_|"0".into()));
    eprintln!("configuration: EGM_FOLDED_PIN_MATCH={}", std::env::var("EGM_FOLDED_PIN_MATCH").unwrap_or_else(|_|"0".into()));
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 2 || args.len() > 3 {
        eprintln!("Low-level kernel: emapper_kernel <genlib> <aig> [mapped.v]");
        eprintln!("For the fixed 600-second single-output (D,A) configuration, use scripts/emapper/run.py.");
        eprintln!("Bare kernel parameters use module defaults; they are not the delivery profile.");
        std::process::exit(2);
    }
    // Opt-in preset for the full-scope tested coupled recovery configuration.
    // Individual budget flags still retain their ordinary override semantics.
    let queue_profile = match std::env::var("EGM_RECOVERY_PROFILE").as_deref().unwrap_or("legacy") {
        "legacy" => false,
        "queue-delay" => true,
        other => return Err(format!("invalid EGM_RECOVERY_PROFILE: {other}").into()),
    };
    let library = PathBuf::from(&args[0]);
    let golden = PathBuf::from(&args[1]);
    let output = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("egraph_mapped.v"));
    let mut aig = Aig::from_aiger(golden.to_str().ok_or("invalid AIG path")?);
    aig.cleanup_graph();
    // The functional index stays small; directed library loading retains wider cells for rules.
    let mut ps = MapParams {
        max_inputs: number("EGM_K", 6),
        cuts_per_class: number("EGM_CUTS", if queue_profile {16} else {32}),
        cut_rounds: number("EGM_CUT_ROUNDS", 32),
        max_instances: number("EGM_INSTANCES", 100_000),
        multioutput: number("EGM_MULTIOUTPUT", 1) != 0,
        multioutput_cumulative: number("EGM_MULTI_CUMULATIVE", 1) != 0,
        dc_support: number("EGM_DC_SUPPORT", if queue_profile {4} else {0}),
        dc_cone: number("EGM_DC_CONE", 1024),
        single_structure_cuts: number("EGM_SINGLE_STRUCTURE_CUTS",0)!=0,
        no_input_phase_matches: number("EGM_NO_INPUT_PHASE_MATCHES",0)!=0,
        canonical_pin_matching: number("EGM_CANONICAL_PIN_MATCHING",0)!=0,
        no_actual_area_tail: number("EGM_NO_ACTUAL_AREA_TAIL",0)!=0,
        ablate_functional_cuts: number("EGM_ABLATE_FUNCTIONAL_CUTS",0)!=0,
        ablate_shared_area: number("EGM_ABLATE_SHARED_AREA",0)!=0,
        ablate_incumbent_guard: number("EGM_ABLATE_INCUMBENT_GUARD",0)!=0,
        time_limit_secs: number("EGM_TIME", if queue_profile {0.0} else {30.0}),
        delay_weight: number("EGM_DELAY_WEIGHT", 0.0),
        objective: match std::env::var("EGM_OBJECTIVE")
            .as_deref()
            .unwrap_or(if queue_profile {"delay-area"} else {"weighted"})
        {
            "weighted" => MapObjective::Weighted,
            "area-delay" => MapObjective::AreaDelay,
            "delay-area" => MapObjective::DelayArea,
            other => return Err(format!("invalid EGM_OBJECTIVE: {other}").into()),
        },
        refine_rounds: number("EGM_REFINE_ROUNDS", 4),
        refine_trials: number("EGM_REFINE_TRIALS", 2000),
        boolean_rules: number("EGM_BOOLEAN", 0),
        flow_delay_ties: number("EGM_FLOW_DELAY_TIES", 0) != 0,
        structural_refs: number("EGM_STRUCTURAL_REFS", 0) != 0,
        flow_worklist: number("EGM_FLOW_WORKLIST", u8::from(queue_profile)) != 0,
        fast_delay_seed: number("EGM_FAST_DELAY_SEED", 0) != 0,
        required_area_dag: number("EGM_REQUIRED_AREA_DAG", 0) != 0,
        defer_area_dag: number("EGM_DEFER_AREA_DAG", 0) != 0,
        preserve_legacy_seeds: number("EGM_PRESERVE_LEGACY_SEEDS", 0) != 0,
        metric_epochs: number("EGM_METRIC_EPOCHS", 0) != 0,
        bounded_proposals: number("EGM_BOUNDED_PROPOSALS", 0) != 0,
        seed_window_fraction: number("EGM_SEED_WINDOW_FRACTION", 1.0),
        flow_compare: number("EGM_FLOW_COMPARE", 0) != 0,
        critical_recovery: number("EGM_CRITICAL_RECOVERY", 0) != 0,
        cut_depth_priority: number("EGM_CUT_DEPTH", 0) != 0,
        cut_depth_quota: number("EGM_CUT_DEPTH_QUOTA", 0),
        library_timing: number("EGM_LIBRARY_TIMING", 0) != 0,
        slack_flow: number("EGM_SLACK_FLOW", 0) != 0,
        timing_cap: number("EGM_TIMING_CAP", 0) != 0,
        strict_budgets: number("EGM_STRICT_BUDGETS", u8::from(queue_profile)) != 0,
        literal_auto: number("EGM_LITERAL_AUTO",0)!=0,
        cone_import: number("EGM_CONE_IMPORT",0)!=0,
        shared_majority: number("EGM_SHARED_MAJORITY",0)!=0,
        proof_source: number("EGM_PROOF_SOURCE",0)!=0,
        source_propagation: number("EGM_SOURCE_PROPAGATION",0)!=0,
        scored_incumbent: number("EGM_SCORED_INCUMBENT",0)!=0,
        joint_cones: number("EGM_JOINT_CONES",0)!=0,
        timing_witness_pool: number("EGM_TIMING_WITNESS_POOL",0)!=0,
        online_area: number("EGM_ONLINE_AREA",0)!=0,
        fused_cover_scoring: number("EGM_FUSED_COVER_SCORING",0)!=0,
        shared_area_bound: number("EGM_SHARED_AREA_BOUND",0)!=0,
        cut_pin_flow: number("EGM_CUT_PIN_FLOW",0)!=0,
        delta_area_filter: number("EGM_DELTA_AREA_FILTER",0)!=0,
        online_shared_bound: number("EGM_ONLINE_SHARED_BOUND",0)!=0,
        refresh_area_timing: number("EGM_REFRESH_AREA_TIMING",0)!=0,
        tail_shared_area: number("EGM_TAIL_SHARED_AREA",0)!=0,
        terminal_area: number("EGM_TERMINAL_AREA",0)!=0,
        retain_generation_frontier: number("EGM_RETAIN_GENERATION_FRONTIER",0)!=0,
        exchange_functional_seed: number("EGM_EXCHANGE_FUNCTIONAL_SEED",0)!=0,
        protected_exchange: number("EGM_PROTECTED_EXCHANGE",0)!=0,
        fixed_area_bound: number("EGM_FIXED_AREA_BOUND",0)!=0,
        functional_cuts: number("EGM_FUNCTIONAL_CUTS",0)!=0,
        functional_phase_screen: number("EGM_FUNCTIONAL_PHASE_SCREEN",0)!=0,
        canonical_source_inputs: number("EGM_CANONICAL_SOURCE_INPUTS",0)!=0,
        import_helper_fill: number("EGM_IMPORT_HELPER_FILL",0)!=0,
        introduce_phase_inverters: number("EGM_INTRODUCE_PHASE_INVERTERS",0)!=0,
        preserve_shared_trajectory: number("EGM_PRESERVE_SHARED_TRAJECTORY",0)!=0,
        activate_deferred: number("EGM_ACTIVATE_DEFERRED",0)!=0,
        literal_import: number("EGM_LITERAL_IMPORT",0)!=0,
        fast_area_scan: number("EGM_FAST_AREA",0)!=0,
        return_reserve: number("EGM_RETURN_RESERVE",0.05),
        exchange_time_fraction:number("EGM_EXCHANGE_TIME_FRACTION",0.70),
        area_time_fraction: number("EGM_AREA_TIME_FRACTION",0.0),
        canonical_phases: number("EGM_CANONICAL_PHASE",0)!=0,
        canonical_wire_only: number("EGM_CANONICAL_WIRES",0)!=0,
        region_classes: number("EGM_REGION_CLASSES", if queue_profile {256} else {0}),
        phase_recovery: number("EGM_PHASE_RECOVERY", u8::from(queue_profile)) != 0,
        phase_time_fraction: number("EGM_PHASE_TIME_FRACTION", if queue_profile {0.25} else {1.0}),
        recover_live_refs: number("EGM_LIVE_REFS", 1) != 0,
        recover_cones: number("EGM_CONE_RECOVERY", 1) != 0,
        refine_time_secs: number("EGM_REFINE_TIME", 2.0),
        max_delay: std::env::var("EGM_MAX_DELAY")
            .ok()
            .map(|v| v.parse())
            .transpose()?,
        ..Default::default()
    };
    if queue_profile && (!ps.strict_budgets || ps.objective != MapObjective::DelayArea) {
        return Err("queue-delay profile requires strict budgets and delay-area objective".into());
    }
    ps.saturation.iter_limit = number("EGM_ITER", 16);
    ps.saturation.node_limit = number("EGM_NODES", 20_000);
    ps.wide_cells=number("EGM_WIDE_MATCH",0)!=0;
    ps.share_generator_keys=number("EGM_SHARE_GENERATOR_KEYS",0)!=0;
    ps.wide_index_filter=number("EGM_WIDE_INDEX_FILTER",0)!=0;
    let tech = TechLib::from_genlib_directed(&library)?;
    let result = map(&aig, &tech, &ps)?;
    result.write_verilog(std::fs::File::create(&output)?, "mapped")?;
    let cec = if number("EGM_NOCEC", 0) != 0 {
        -1.0
    } else {
        if verify_with_models(&tech, &golden, &output)? {
            1.0
        } else {
            0.0
        }
    };
    let s = &result.stats;
    if let Some(path)=std::env::var_os("EGM_POOL_DIAG_PATH") {let path=std::path::Path::new(&path).with_extension("wide.json");let _=std::fs::write(path,format!("{{\"enabled\":{},\"trials\":{},\"candidates\":{},\"truncated\":{}}}",ps.wide_cells,s.wide_trials,s.wide_candidates,s.wide_work_limit_hit));}

    println!(
        "{{\"area\":{},\"delay\":{},\"cec\":{},\"regions\":{},\"cross_region_boundaries\":{},\"online_area_updates\":{},\"joint_attempts\":{},\"joint_valid\":{},\"phase_trials\":{},\"phase_accepted\":{},\"phase_secs\":{},\"saturation_budget_skips\":{},\"imported_enodes\":{},\"cumulative_imported_enodes\":{},\"cumulative_admitted_enodes\":{},\"phase_initial_delay\":{},\"phase_final_delay\":{},\"eclasses\":{},\"enodes\":{},\
        \"cut_rounds_started\":{},\"exchange_trials\":{},\"exchange_cuts\":{},\"protected_bindings\":{},\"exchange_source_bindings\":{},\"exchange_fallback\":{},\"cuts\":{},\"max_cuts_per_class\":{},\"instances\":{},\"generated_instances\":{},\"instance_budget_pruned\":{},\"bindings\":{},\"dc_boundaries\":{},\
        \"dc_with_unreachable\":{},\"dc_skipped\":{},\"dc_bindings\":{},\
        \"selected_multioutput\":{},\"selected_joint_multioutput\":{},\
        \"functional_cut_trials\":{},\"functional_cut_proofs\":{},\"functional_cut_admitted\":{},\"cut_depth_swaps\":{},\"cut_limit_hit\":{},\"instance_limit_hit\":{},\"fallback\":{},\
        \"proven_optimal\":{},\"status\":{:?},\"saturation_stop\":{:?},\"build_secs\":{},\"solve_secs\":{},\
        \"slack_attempts\":{},\"slack_valid\":{},\"slack_accepted\":{},\"slack_rounds_started\":{},\"terminal_area_attempted\":{},\"terminal_area_accepted\":{},\"slack_local_scored\":{},\"slack_local_trials\":{},\"slack_local_accepted\":{},\"initial_delay\":{},\"seed_delay\":{},\"initial_area\":{},\"heuristic_area\":{},\"heuristic_secs\":{},\"refinement_trials\":{},\"refinement_accepted\":{},\"flow_compare_attempted\":{},\"flow_compare_accepted\":{}}}",
        result.area,
        result.delay,
        cec,
        s.regions,
        s.cross_region_boundaries,
        s.online_area_updates,
        s.joint_attempts,
        s.joint_valid,
        s.phase_trials,
        s.phase_accepted,
        s.phase_secs,
        s.saturation_budget_skips,
        s.imported_enodes,
        s.cumulative_imported_enodes,
        s.cumulative_admitted_enodes,
        s.phase_initial_delay,
        s.phase_final_delay,
        s.eclasses,
        s.enodes,
        s.cut_rounds_started,
        s.exchange_trials,
        s.exchange_cuts,
        s.protected_bindings,
        s.exchange_source_bindings,
        s.exchange_fallback,
        s.cuts,
        s.max_cuts_per_class,
        s.instances,
        s.generated_instances,
        s.instance_budget_pruned,
        s.bindings,
        s.dc_boundaries,
        s.dc_with_unreachable,
        s.dc_skipped,
        s.dc_bindings,
        s.selected_multioutput,
        s.selected_joint_multioutput,
        s.functional_cut_trials,
        s.functional_cut_proofs,
        s.functional_cut_admitted,
        s.cut_depth_swaps,
        s.cut_limit_hit,
        s.instance_limit_hit,
        s.used_fallback,
        s.proven_optimal,
        s.solver_status,
        s.saturation_stop,
        s.build_secs,
        s.solve_secs,
        s.slack_attempts,
        s.slack_valid,
        s.slack_accepted,
        s.slack_rounds_started,
        s.terminal_area_attempted,
        s.terminal_area_accepted,
        s.slack_local_scored,
        s.slack_local_trials,
        s.slack_local_accepted,
        s.initial_delay,
        s.seed_delay,
        s.initial_area,
        s.heuristic_area,
        s.heuristic_secs,
        s.refinement_trials,
        s.refinement_accepted,
        s.flow_compare_attempted,
        s.flow_compare_accepted
    );
    if cec == 0.0 {
        return Err("ABC found a non-equivalent mapped netlist".into());
    }
    if cec < 0.0 && number("EGM_NOCEC", 0) == 0 {
        return Err("ABC could not verify the emitted netlist".into());
    }
    Ok(())
}
