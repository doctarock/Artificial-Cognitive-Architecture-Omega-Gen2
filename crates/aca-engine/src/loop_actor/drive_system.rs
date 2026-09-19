use aca_util::EpochMillis;

use crate::steps::drives::DriveState;

/// The continuous cognitive pressures `steps::agenda` reads to decide
/// whether a new intention is worth spawning, and which drive it should be
/// attributed to - see `steps::drives`'s own doc comment.
#[derive(Default)]
pub(crate) struct DriveSystem {
    pub(crate) drive_state: DriveState,
    /// When a real conversational turn (`SourceChannel::ConversationInput`)
    /// last arrived - `steps::drives::social_connection_pressure`'s input.
    /// `None` until the very first one ever arrives (never assumed to be
    /// "just now" at cold start - see that function's own doc comment).
    pub(crate) last_conversation_input_at: Option<EpochMillis>,
}
