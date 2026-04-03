#![forbid(unsafe_code)]
#![warn(clippy::pedantic, clippy::nursery)]

mod abc_stats;
pub mod aig_graph;
pub mod emap;
mod emap_cover;
pub use emap_cover::{
    map_cut_to_aig_sexp, map_cut_to_cover, MappedCoverChoice, MappedCoverGraph, MappedCoverNode,
};
pub mod emap_cut;
pub mod emap_egg;
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
        binding_as_instance, emap, emap_with_params, write_mapped_verilog, EmapParams, EmapResult,
        NodeBinding, NodeBindingKind,
    };
    pub use crate::emap_cut;
    pub use crate::emap_egg;
    pub use crate::techlib::{MappingObjective, TechLib};
    pub use crate::{
        map_cut_to_aig_sexp, map_cut_to_cover, MappedCoverChoice, MappedCoverGraph, MappedCoverNode,
    };
}
