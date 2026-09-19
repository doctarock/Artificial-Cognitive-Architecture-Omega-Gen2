//! The ten-step cognitive cycle, one module per step (or closely related
//! pair of steps). Built incrementally: each module lands with its own
//! tests proving that step's behavior in isolation; full end-to-end wiring
//! into a single `CognitiveLoopActor` tick lands once enough steps exist to
//! make that meaningful.

pub mod act;
pub mod affect;
pub mod agenda;
pub mod arbitrate;
pub mod boredom;
pub mod broadcast;
pub mod coalition;
pub mod communicative_intent;
pub mod compare;
pub mod confidence_revision;
pub mod displacement;
pub mod drives;
pub mod eligibility;
pub mod executive;
pub mod interlocutor;
pub mod knowledge_library;
pub mod known_answers;
pub mod learn;
pub mod memory_formation;
pub mod metacognition;
pub mod observe;
pub mod orient;
pub mod orient_outcome;
pub mod predict;
pub mod procedural;
pub mod spikes;
pub mod recall;
pub mod social_interface;
pub mod synthesize;
pub mod tools;
pub mod tool_intent;
