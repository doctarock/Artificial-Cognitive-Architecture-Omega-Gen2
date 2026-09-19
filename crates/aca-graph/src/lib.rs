//! The ACT-R activation graph and GWT ranking math: base-level decay,
//! spreading activation, Hebbian associative-edge reinforcement, and
//! capacity-limited broadcast admission. Deliberately has **no `tokio`
//! dependency at all** — every function here is synchronous, pure
//! computation over in-memory data, so a blocking call sneaking into this
//! crate is a compile error, not a code-review miss. This is what lets
//! `aca-engine`'s cognitive cycle steps 1-6 run constantly without ever
//! risking a stall on model-tier I/O.

mod activation;
mod edges;
mod graph;
mod ranking;
mod recall;

pub use activation::{
    clears_retrieval_threshold, compute_base_level, compute_spreading_activation,
    leak_dynamics, recompute_activation, record_reference, sample_noise, stimulate_dynamics,
    try_fire_dynamics,
};
pub use edges::{
    effective_strength, reinforce_edge, DEFAULT_HEBBIAN_INCREMENT, DEFAULT_MAX_EDGE_STRENGTH,
};
pub use graph::Graph;
pub use ranking::{admit_top_n, rank_candidates};
pub use recall::spread_activation_multi_hop;
