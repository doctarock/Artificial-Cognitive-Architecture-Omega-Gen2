use std::collections::VecDeque;

use aca_util::EpochMillis;
use tokio::sync::mpsc;

use super::{CuratedAnswerCommand, ExternalAgentInput, RoomInput, SensorInput, SensorSignal};

/// Every raw-input channel this actor listens on, the room-audio backlog
/// buffer, the Act/Knowledge-Library self-send reentry loopbacks, and the
/// boredom/idle timers that gate when the actor generates its own input
/// rather than waiting for real input to arrive.
pub(crate) struct PerceptionState {
    pub(crate) input_rx: mpsc::Receiver<String>,
    /// Filled by `CognitiveScheduler`'s `select!` when `input_rx` is the
    /// channel that wakes a sleeping `run()` - `select!`'s winning branch
    /// actually consumes that message, so it has to go somewhere `tick()`'s
    /// own `try_recv`-based drain will still find it, or it's silently
    /// lost rather than merely delayed a tick. `take_input` checks this
    /// before the channel itself; ticks driven by direct `tick()` calls
    /// (as in nearly every test) never populate it, so behavior there is
    /// unchanged.
    pub(crate) input_primed: Option<String>,
    pub(crate) external_agent_input_rx: mpsc::Receiver<ExternalAgentInput>,
    /// Same reentry-priming role as `input_primed`.
    pub(crate) external_agent_primed: Option<ExternalAgentInput>,
    pub(crate) sensor_input_rx: mpsc::Receiver<SensorInput>,
    /// Same reentry-priming role as `input_primed`.
    pub(crate) sensor_primed: Option<SensorInput>,
    pub(crate) sensor_signal_rx: mpsc::Receiver<SensorSignal>,
    pub(crate) sensor_signal_primed: Option<SensorSignal>,
    pub(crate) curated_answer_rx: mpsc::Receiver<CuratedAnswerCommand>,
    pub(crate) curated_answer_primed: Option<CuratedAnswerCommand>,
    pub(crate) room_input_rx: mpsc::Receiver<RoomInput>,
    /// Room-audio backlog already drained from `room_input_rx` this run but
    /// not yet processed - `tick()` only ever coalesces a same-`stream_id`
    /// run off the *front* of this buffer, leaving anything from a
    /// different stream here for a later tick rather than blending it in. A
    /// plain `mpsc::Receiver` has no peek, so this is what makes "look at
    /// what's next without necessarily consuming it across stream
    /// boundaries" possible at all.
    pub(crate) pending_room_input: VecDeque<RoomInput>,
    /// A successful Knowledge Library consult result loops back through here
    /// to become an Observation next tick, exactly like any other input -
    /// never exposed via `LoopHandles`, since the actor is both the sole
    /// producer and sole consumer (see `tick()`'s KL re-entry send site for
    /// why this is `try_send`, not an awaited send).
    pub(crate) kl_reentry_tx: mpsc::Sender<String>,
    pub(crate) kl_reentry_rx: mpsc::Receiver<String>,
    /// Same reentry-priming role as `input_primed`.
    pub(crate) kl_reentry_primed: Option<String>,
    /// A tool invocation's result loops back through here to become an
    /// Observation next tick - same actor-internal-only, self-send shape as
    /// `kl_reentry_tx`/`kl_reentry_rx`.
    pub(crate) act_reentry_tx: mpsc::Sender<String>,
    pub(crate) act_reentry_rx: mpsc::Receiver<String>,
    /// Same reentry-priming role as `input_primed`.
    pub(crate) act_reentry_primed: Option<String>,
    /// The instant Working Memory most recently became empty, cleared back
    /// to `None` the instant it's non-empty again - `steps::boredom`'s idle
    /// gate ("has WM genuinely had nothing in it for a while," not just "no
    /// raw input arrived this exact tick") is measured from this, not from
    /// `cycle_seq` or wall-clock ticks.
    pub(crate) wm_empty_since: Option<EpochMillis>,
    /// When a boredom stimulus was last generated, success or failure - the
    /// same cost-control-backstop role `last_synthesis_at` plays for
    /// synthesis, gating `steps::boredom::BoredomConfig::min_interval_ms`.
    pub(crate) last_boredom_at: Option<EpochMillis>,
    /// When the standing self-status duty (`steps::boredom`) last actually
    /// fired - distinct from `last_boredom_at` because the two have
    /// independent cadences (a boredom stimulus firing doesn't necessarily
    /// mean the self-status duty was the one that fired).
    pub(crate) last_self_status_at: Option<EpochMillis>,
}

impl PerceptionState {
    /// Checks the primed slot (see `input_primed`'s doc comment) before the
    /// channel itself - non-blocking either way, same as a plain
    /// `try_recv()` from `tick()`'s point of view.
    pub(crate) fn take_input(&mut self) -> Option<String> {
        self.input_primed.take().or_else(|| self.input_rx.try_recv().ok())
    }

    pub(crate) fn take_external_agent_input(&mut self) -> Option<ExternalAgentInput> {
        self.external_agent_primed.take().or_else(|| self.external_agent_input_rx.try_recv().ok())
    }

    pub(crate) fn take_sensor_input(&mut self) -> Option<SensorInput> {
        self.sensor_primed.take().or_else(|| self.sensor_input_rx.try_recv().ok())
    }

    pub(crate) fn take_sensor_signal(&mut self) -> Option<SensorSignal> {
        self.sensor_signal_primed.take().or_else(|| self.sensor_signal_rx.try_recv().ok())
    }

    pub(crate) fn take_curated_answer_command(&mut self) -> Option<CuratedAnswerCommand> {
        self.curated_answer_primed.take().or_else(|| self.curated_answer_rx.try_recv().ok())
    }

    pub(crate) fn take_kl_reentry(&mut self) -> Option<String> {
        self.kl_reentry_primed.take().or_else(|| self.kl_reentry_rx.try_recv().ok())
    }

    pub(crate) fn take_act_reentry(&mut self) -> Option<String> {
        self.act_reentry_primed.take().or_else(|| self.act_reentry_rx.try_recv().ok())
    }
}
