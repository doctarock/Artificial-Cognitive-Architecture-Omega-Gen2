//! Network-first cognitive cluster substrate for Omega milestone 1.
//!
//! This crate deliberately contains only infrastructure: typed envelopes,
//! shared cognitive state, a processor registry/client, mock processors, and
//! a tiny runtime loop that demonstrates state mutation through services.

pub mod ocl;
pub mod processor;
pub mod protocol;
pub mod registry;
pub mod runtime;
pub mod state;
pub mod store;
pub mod workspace;
