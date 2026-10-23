#![forbid(unsafe_code)]
#![warn(clippy::pedantic, clippy::nursery)]

mod tt9;
pub use tt9::Tt9;

mod abc_cec;
mod abc_stats;
pub use abc_cec::run_abc_cec;
pub use abc_stats::{run_abc_print_stats, AbcPrintStats};
pub mod aig_graph;
pub mod emap;
mod emap_env;
pub use emap_env::{
    emap_params_from_bench_env, emap_params_from_env, emap_params_from_migbalance_env,
    techlib_cache_path_near_genlib,
};
mod emap_cover;
pub mod supergate_db;
pub use emap_cover::{
    compute_pin_to_root_delays, map_cut_to_aig_sexp, map_cut_to_cover,
    map_cut_to_cover_alternatives, map_cut_to_cover_cover_only, map_cut_to_cover_with_reason,
    map_cut_to_library, map_tt_to_library, MappedCoverChoice, MappedCoverGraph, MappedCoverNode,
};
pub mod egraph_map;
pub mod egraph_sat;
pub mod emap_cut;
pub mod emap_egg;
pub mod ilp_cover;
pub mod netlist_metrics;
pub mod supergate;
pub mod synth;
pub use synth::{
    graph_tt, synthesize_cut_choice, synthesize_cut_choice_for_objective, synthesize_cut_cover,
};
pub mod techlib;

/// AIG graph/representation namespace.
///
/// Keep data-structure code separate from technology mapping.
pub mod aig {
    pub use crate::aig_graph::{Aig, AigNode};
}

/// Technology mapping namespace.
pub mod mapping {
    pub use crate::emap::{
        binding_as_instance, emap, emap_with_params, inverter_breakdown, mapped_depth,
        materialized_area, write_mapped_verilog, EmapParams, EmapResult, NodeBinding,
        NodeBindingKind,
    };
    pub use crate::emap_cut;
    pub use crate::emap_egg;
    pub use crate::emap_env::{
        emap_params_from_bench_env, emap_params_from_env, emap_params_from_migbalance_env,
        techlib_cache_path_near_genlib,
    };
    pub use crate::techlib::{MappingAreaCost, MappingObjective, TechLib};
    pub use crate::{
        compute_pin_to_root_delays, map_cut_to_aig_sexp, map_cut_to_cover,
        map_cut_to_cover_with_reason, MappedCoverChoice, MappedCoverGraph, MappedCoverNode,
    };
    pub use crate::{run_abc_print_stats, AbcPrintStats};
}

mod egraph_source_plan;
mod source_proof;

mod egraph_budget;
