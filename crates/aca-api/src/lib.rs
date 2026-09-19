//! The local HTTP+WebSocket API surface: a pure consumer of a
//! `CognitiveLoopActor`'s channels (input submission, the live
//! `cycle_events` feed, and the always-latest Working Memory snapshot).
//! Never a second path into the graph.

mod server;
#[cfg(feature = "dev-tools")]
mod test_status;

pub use server::{build_router, ApiState};
