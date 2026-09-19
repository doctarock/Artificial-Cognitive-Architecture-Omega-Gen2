//! The shared currency of Omega ACA's cognitive cycle: `MentalObject` and
//! its supporting types. Every candidate, memory, goal, prediction,
//! coalition, and operator output is a `MentalObject` — this crate is pure
//! data with zero I/O, depended on by every other `aca-*` crate.

mod enums;
mod ids;
mod mental_object;

pub use enums::{EdgeKind, GoalStatus, MemoryRole, MentalObjectKind, ObjectStatus, Tier};
pub use ids::{EdgeId, GoalStackId, MentalObjectId};
pub use mental_object::{
    ActivationState, AssociativeEdge, GoalStackMembership, MentalObject, MentalObjectDynamics,
    PredictionState, PromotionState, PromotionStatus, WorkspaceState, DEFAULT_REFERENCE_LOG_CAPACITY,
};
