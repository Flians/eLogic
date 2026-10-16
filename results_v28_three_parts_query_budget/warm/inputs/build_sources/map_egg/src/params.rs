//! Mapper parameters shared by the cover generators.
//!
//! Like [`crate::binding`], this used to live in the legacy standalone mapper (`emap.rs`); the cover
//! generators only need the parameters themselves, not that mapper.

use crate::techlib::{MappingAreaCost, MappingObjective};

#[derive(Clone, Debug)]
pub struct EmapParams {
    pub max_cuts: usize,
    pub max_inputs: usize,
    pub wire_delay: f32,
    pub use_egg: bool,
    pub cache_egg: bool,
    pub cache_tt: bool,
    pub objective: MappingObjective,
    /// If true, Stage 1b rematch uses delay-first on critical ANDs and area-first elsewhere.
    /// Stage 1b always runs after Stage 1a; if false, Stage 1b uses [`MappingObjective::DelayFirst`] for every node.
    pub critical_path_mixed_objective: bool,
    /// Number of area-flow style recovery passes after delay pass.
    pub area_flow_passes: usize,
    /// Number of exact-area style recovery passes after area-flow pass.
    pub exact_area_passes: usize,
    /// Reject oversized extracted AIG covers to avoid QoR collapse from expression blow-up.
    pub max_cover_nodes: usize,
    /// Use dual-phase delay flow (considers both pos and neg bindings per node) in Stage 1.
    /// Gives lower delay than the single-phase flow at similar runtime.  Defaults to true.
    pub use_dual_phase_delay: bool,
    /// Distinguish mapper intent from mockturtle-style emap naming.
    /// This implementation targets an egraph-based mapper flow.
    pub mapper_name: &'static str,
    /// Area term used to **compare** candidate cells during mapping (mockturtle-style flow vs
    /// physical). [`NodeBinding::area`] is always **physical** for reporting.
    pub mapping_area_cost: MappingAreaCost,
    /// After egg saturation, collect up to this many **root** cover DAG enodes that tie on
    /// `(delay, area)` for the current objective, materialize each, and pick the best binding downstream.
    pub max_egg_cover_alternatives: usize,
    /// Egg saturation iteration limit for cover extraction. `0` uses the `EMAPPER_EGG_ITER_LIMIT`
    /// environment default (30). Cut mapping uses a small value: a cut AIG is tiny, and the
    /// default 30 iterations can take seconds per cut without changing the extracted cover.
    pub egg_iter_limit: usize,
    /// Egg saturation node limit for cover extraction. `0` uses the `EMAPPER_EGG_NODE_LIMIT`
    /// environment default (10_000).
    pub egg_node_limit: usize,
}

impl Default for EmapParams {
    fn default() -> Self {
        Self {
            max_cuts: 16,
            max_inputs: 9,
            wire_delay: 0.0,
            use_egg: true,
            cache_egg: true,
            cache_tt: true,
            objective: MappingObjective::DelayFirst,
            critical_path_mixed_objective: false,
            area_flow_passes: 1,
            exact_area_passes: 1,
            max_cover_nodes: 64,
            use_dual_phase_delay: true,
            mapper_name: "egraph-based-mapper",
            mapping_area_cost: MappingAreaCost::Flow,
            max_egg_cover_alternatives: 1,
            egg_iter_limit: 0,
            egg_node_limit: 0,
        }
    }
}
