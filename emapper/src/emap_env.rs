//! Shared `EmapParams` loading from process environment (CLI binaries).
//!
//! All mapper-related keys use the **`EMAPPER_*`** prefix.
//!
//! - [`emap_params_from_env`] — full set of knobs (defaults tuned for the interactive `MIGBalance` example).
//! - [`emap_params_from_bench_env`] — same keys; if `EMAPPER_AREA_FLOW_PASSES` / `EMAPPER_EXACT_AREA_PASSES`
//!   are **unset**, uses lighter defaults (**1** / **1**) like the original `bench_rs_emap` binary.
//! - Egg equality saturation (see `emap_egg::rewrite_to_gategraph`): `EMAPPER_EGG_ITER_LIMIT`,
//!   `EMAPPER_EGG_NODE_LIMIT` (defaults **30** / **10_000**, matching egg 0.11).
//! - Root cover ties: `EMAPPER_EGG_COVER_ALTERNATIVES` (default **1**); up to **64** distinct optimal
//!   root enode covers are materialized and the best binding for the current objective is chosen.
//! - Supergate cache file: `EMAPPER_SUPERGATE_DB`, default **`supergates.jsonl`** (cwd).

use std::path::{Path, PathBuf};

use crate::emap::EmapParams;
use crate::techlib::{MappingAreaCost, MappingObjective};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
        .unwrap_or(default)
}

fn mapping_area_cost_from_env(key: &str) -> MappingAreaCost {
    match std::env::var(key)
        .unwrap_or_else(|_| "flow".to_string())
        .to_lowercase()
        .as_str()
    {
        "physical" | "phys" => MappingAreaCost::Physical,
        _ => MappingAreaCost::Flow,
    }
}

fn objective_from_env(key: &str) -> MappingObjective {
    match std::env::var(key)
        .unwrap_or_else(|_| "delay".to_string())
        .to_lowercase()
        .as_str()
    {
        "area" => MappingObjective::AreaFirst,
        _ => MappingObjective::DelayFirst,
    }
}

/// Sidecar JSON path for [`TechLib::from_genlib_cached`] next to the genlib file.
#[must_use]
pub fn techlib_cache_path_near_genlib(genlib: &Path, max_inputs: usize) -> PathBuf {
    let mut p = genlib.to_path_buf();
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .map_or_else(|| "json".to_string(), |e| format!("{e}.json"));
    p.set_extension(format!("m{max_inputs}.ttcache.{ext}"));
    p
}

/// `EMAPPER_*` — defaults tuned for the interactive `MIGBalance` example (stronger area recovery).
#[must_use]
pub fn emap_params_from_env() -> EmapParams {
    let use_egg = env_bool("EMAPPER_USE_EGG", true);
    let cache_egg = env_bool("EMAPPER_CACHE_EGG", use_egg);
    EmapParams {
        // Area-oriented runs benefit from a wider cut pool (measured: c880 22.34 -> 21.99,
        // c5315 129.0 -> 129.0, c432 15.91 -> 15.72) at some delay cost; delay-oriented runs do not.
        max_cuts: env_usize(
            "EMAPPER_MAX_CUTS",
            if objective_from_env("EMAPPER_OBJECTIVE") == MappingObjective::AreaFirst {
                24
            } else {
                16
            },
        ),
        max_inputs: env_usize("EMAPPER_MAX_INPUTS", 9),
        wire_delay: 0.0,
        use_egg,
        cache_egg,
        cache_tt: env_bool("EMAPPER_CACHE_TT", true),
        objective: objective_from_env("EMAPPER_OBJECTIVE"),
        critical_path_mixed_objective: env_bool("EMAPPER_MIXED_OBJECTIVE", false),
        area_flow_passes: env_usize("EMAPPER_AREA_FLOW_PASSES", 3),
        exact_area_passes: env_usize("EMAPPER_EXACT_AREA_PASSES", 2),
        max_cover_nodes: env_usize("EMAPPER_MAX_COVER_NODES", 64),
        use_dual_phase_delay: env_bool("EMAPPER_USE_DUAL_PHASE_DELAY", true),
        mapper_name: "egraph-based-mapper",
        mapping_area_cost: mapping_area_cost_from_env("EMAPPER_MAPPING_AREA_COST"),
        max_egg_cover_alternatives: env_usize("EMAPPER_EGG_COVER_ALTERNATIVES", 1),
        egg_iter_limit: 0,
        egg_node_limit: 0,
        depth_recovery_passes: env_usize("EMAPPER_DEPTH_RECOVERY", 1),
        use_two_level_cover: env_bool("EMAPPER_TWO_LEVEL", true),
        // Left to the phase portfolio in `emap_with_params`; `EMAPPER_PESSIMISTIC_PHASE` still
        // forces one model for the single-model path.
        pessimistic_phase: None,
    }
}

/// Back-compat alias for [`emap_params_from_env`].
#[inline]
#[must_use]
pub fn emap_params_from_migbalance_env() -> EmapParams {
    emap_params_from_env()
}

/// Same `EMAPPER_*` keys as [`emap_params_from_env`], but if **`EMAPPER_AREA_FLOW_PASSES`** or
/// **`EMAPPER_EXACT_AREA_PASSES`** are unset, uses **1** / **1** (lighter area recovery) for the bench binary.
#[must_use]
pub fn emap_params_from_bench_env() -> EmapParams {
    let mut p = emap_params_from_env();
    if std::env::var_os("EMAPPER_AREA_FLOW_PASSES").is_none() {
        p.area_flow_passes = 1;
    }
    if std::env::var_os("EMAPPER_EXACT_AREA_PASSES").is_none() {
        p.exact_area_passes = 1;
    }
    p
}
