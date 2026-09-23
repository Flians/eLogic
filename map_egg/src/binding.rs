//! Cut-to-node bindings shared by the cover generators.
//!
//! These two types used to live in the legacy standalone mapper (`emap.rs`); they are what
//! [`crate::emap_cover`] produces and what the C++ engine consumes through `map_egg`.

use crate::emap_cut::Cut;
use crate::emap_cover::MappedCoverGraph;
use crate::techlib::CellVariant;

#[derive(Clone, Debug)]
pub enum NodeBindingKind {
    Direct {
        cell: CellVariant,
    },
    /// Gate-level mapped cover extracted from the rewritten cut.
    MappedCover {
        graph: MappedCoverGraph,
        /// Per entry in [`NodeBinding::leaves`] (same order as [`MappedCoverChoice::used_leaf_indices`]):
        /// mockturtle-style delay from that leaf through the cover to the root output.
        pin_to_root_delay: Vec<f32>,
    },
}

#[derive(Clone, Debug)]
pub struct NodeBinding {
    pub node: usize,
    pub cut: Cut,
    /// Leaf nodes in variable order a,b,c,... (constant 0 is omitted).
    pub leaves: Vec<usize>,
    pub kind: NodeBindingKind,
    /// Whether the binding directly realizes the complemented node phase.
    pub phase_neg: bool,
    /// Arrival of the phase directly realized by this binding.
    pub native_delay: f32,
    /// Arrival of the positive logical node phase.
    pub delay: f32,
    /// Physical area (matches emitted netlist / [`TechLib::variant_area_physical_invs`]).
    pub area: f32,
    /// Area used only when **ranking** cuts (see [`EmapParams::mapping_area_cost`]).
    /// Equals [`NodeBinding::area`] when comparing with physical cost.
    pub mapping_cost_area: f32,
}
