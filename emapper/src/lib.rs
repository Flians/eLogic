#![forbid(unsafe_code)]
#![warn(clippy::pedantic, clippy::nursery)]

pub mod aig_graph;
mod abc_stats;
pub mod emap;
mod emap_cover;
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
}
