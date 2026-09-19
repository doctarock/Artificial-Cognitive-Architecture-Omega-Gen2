//! Generic, dependency-light helpers shared by every other `aca-*` crate:
//! a deterministic time source, clamping helpers for confidence/precision
//! values, and a bounded ring buffer for ACT-R's reference log and rolling
//! precision windows. Nothing here does I/O.

mod clamp;
mod ring_buffer;
mod text;
mod time;
mod vector;

pub use clamp::{clamp01_or_default, clamp_or_default};
pub use ring_buffer::RingBuffer;
pub use text::chunk_text;
pub use time::{Clock, EpochMillis, ManualClock, SystemClock};
pub use vector::{cosine_error, cosine_similarity, weighted_blend};
