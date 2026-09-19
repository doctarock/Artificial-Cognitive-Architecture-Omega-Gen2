use std::time::Duration;

use super::CognitiveLoopActor;

/// The earliest of several candidate wake deadlines - all `wait_for_next_tick`
/// ever needs from its handful of coarse cognitive timers (idle maintenance,
/// boredom, synthesis, each preconscious trace's reevaluation/expiry, spike
/// propagation). Previously a `BinaryHeap`-backed priority queue keyed by an
/// enum of deadline "kinds", rebuilt fresh on every wait: nothing outside
/// tests ever read *which* kind won, only the instant itself, so this
/// replaces that heap/enum machinery with a running minimum computed with no
/// allocation at all - a real cost the old version paid on every idle wake
/// (up to `O(2 * preconscious_traces.len())` heap pushes) to answer a
/// question a plain fold already answers. If a later feature genuinely needs
/// to know which specific deadline fired outside tests, that's the moment to
/// bring the kind-tracking machinery back, not before.
#[derive(Debug, Default)]
struct EarliestDeadline(Option<i64>);

impl EarliestDeadline {
    fn consider(&mut self, due_at: i64) {
        self.0 = Some(self.0.map_or(due_at, |current| current.min(due_at)));
    }
}

/// Replaces the old `run()`'s unconditional `tick(); yield_now();` busy-poll
/// with a real wait: sleep until whichever wake condition is soonest, or
/// return immediately if one is already due. Stateless by design - every
/// input it reads and every primed slot it fills lives on the actor's own
/// sub-components, so this is a named place for the wait logic to live, not
/// an owner of anything itself.
///
/// Agenda commitment decay/floor grace, drive smoothing, the write-behind
/// flush cadence, and `steps::metacognition`'s outcome-feedback window all
/// use elapsed wall time rather than a tick count, so none of them erode
/// when `LoopConfig::max_idle_interval_ms` is raised - see that field's own
/// doc comment for the one tick-counted knob left by design
/// (`attention_reconciliation_interval`, which bounds a count of state
/// transitions, not a duration) and for the self-status interrupt's own
/// dedicated wake (`self_status_interrupt_ready`, below).
pub(crate) struct CognitiveScheduler;

impl CognitiveScheduler {
    /// Waits for the next reason to tick, then returns so `run()` can call
    /// `tick()` once. Does not drain anything beyond stashing the one
    /// message (if any) that actually woke it into the matching primed slot
    /// (see `PerceptionState::input_primed`'s doc comment for why) -
    /// `tick()`'s own `try_recv`-based drain still does the real work,
    /// unchanged, so a timer-woken tick still opportunistically picks up
    /// anything else that happens to be queued, exactly as before this
    /// existed.
    pub(crate) async fn wait_for_next_tick(actor: &mut CognitiveLoopActor) {
        // All of these are "already due, right now" conditions with no
        // gating check of their own once they're non-empty - see
        // `MemoryCoordinator::pending_admission` and
        // `PerceptionState::pending_room_input`'s own doc comments, and
        // every `*_primed` slot's own doc comment (a message already
        // sitting in one, e.g. because a higher-priority `select!` arm won
        // a previous wake, is real unconsumed work for the very next
        // `tick()`, not something worth an extra sleep before noticing).
        // Every primed slot across `perception`/`prediction`/
        // `procedure_feedback` is listed here, not just a subset - missing
        // one here doesn't lose the message (the `select!` below and
        // `tick()`'s own drain still see it eventually), it only adds up to
        // `max_idle_interval_ms` of avoidable latency each time that slot's
        // channel loses the race to a busier one.
        if !actor.memory.pending_admission.is_empty()
            || !actor.perception.pending_room_input.is_empty()
            || actor.perception.input_primed.is_some()
            || actor.perception.external_agent_primed.is_some()
            || actor.perception.sensor_primed.is_some()
            || actor.perception.sensor_signal_primed.is_some()
            || actor.perception.curated_answer_primed.is_some()
            || actor.perception.kl_reentry_primed.is_some()
            || actor.perception.act_reentry_primed.is_some()
            || actor.prediction.embedding_reentry_primed.is_some()
            || actor.procedure_feedback.feedback_primed.is_some()
            || actor.procedure_feedback.outcome_primed.is_some()
            || self_status_interrupt_ready(actor)
        {
            // A cooperative hand-off point, not a delay - same role the old
            // busy-poll's own `yield_now` played, kept here so a run of
            // back-to-back immediate-tick conditions (e.g. a sustained
            // input flood keeping `pending_room_input` non-empty) can't
            // starve the API server or in-flight tier calls of the
            // executor.
            tokio::task::yield_now().await;
            return;
        }

        let now = actor.clock.now();
        let boredom_config = &actor.config.boredom_config;
        let mut deadline = EarliestDeadline::default();
        deadline.consider(now.0.saturating_add(actor.config.max_idle_interval_ms as i64));
        if !actor.config.ablation_config.disable_boredom {
            if let Some(next_boredom) = boredom_deadline(
                actor.perception.wm_empty_since.map(|t| t.0),
                actor.perception.last_boredom_at.map(|t| t.0),
                boredom_config.idle_threshold_ms,
                boredom_config.min_interval_ms,
            ) {
                deadline.consider(next_boredom);
            }
        }
        let synthesis_config = &actor.config.synthesis_config;
        if !actor.config.ablation_config.disable_synthesis {
            if let Some(next_synthesis) = synthesis_deadline(
                actor.memory.episodic_since_last_synthesis.len(),
                synthesis_config.min_new_episodic,
                // Mirrors `tick()`'s own `synthesis_backlog_pressure` exactly
                // - see `synthesis_deadline`'s doc comment for why only this
                // threshold (not `min_new_episodic` alone) is safe to treat
                // as an immediate scheduler deadline.
                synthesis_config.max_cluster_size * 2,
                actor.memory.last_synthesis_at.map(|t| t.0),
                synthesis_config.min_interval_ms,
                now.0,
            ) {
                deadline.consider(next_synthesis);
            }
        }
        if !actor.config.ablation_config.disable_preconscious_traces {
            for trace in actor.memory.preconscious_traces.values() {
                deadline.consider(trace.next_evaluation_at.0);
                deadline.consider(trace.expires_at.0);
            }
        }
        if !actor.config.ablation_config.disable_spike_propagation {
            if let Some(due_at) = actor.memory.spike_events.next_deadline() {
                deadline.consider(due_at.0);
            }
        }
        let deadline = deadline.0.expect("idle maintenance always supplies a deadline");
        let wait_ms = deadline.saturating_sub(now.0).max(0) as u64;
        let sleep = tokio::time::sleep(Duration::from_millis(wait_ms));
        tokio::pin!(sleep);

        tokio::select! {
            _ = &mut sleep => {}
            msg = actor.perception.input_rx.recv(), if !actor.perception.input_rx.is_closed() => actor.perception.input_primed = msg,
            msg = actor.perception.external_agent_input_rx.recv(), if !actor.perception.external_agent_input_rx.is_closed() => actor.perception.external_agent_primed = msg,
            msg = actor.perception.sensor_input_rx.recv(), if !actor.perception.sensor_input_rx.is_closed() => actor.perception.sensor_primed = msg,
            msg = actor.perception.sensor_signal_rx.recv(), if !actor.perception.sensor_signal_rx.is_closed() => actor.perception.sensor_signal_primed = msg,
            msg = actor.perception.curated_answer_rx.recv(), if !actor.perception.curated_answer_rx.is_closed() => actor.perception.curated_answer_primed = msg,
            msg = actor.procedure_feedback.feedback_rx.recv(), if !actor.procedure_feedback.feedback_rx.is_closed() => actor.procedure_feedback.feedback_primed = msg,
            msg = actor.procedure_feedback.outcome_rx.recv(), if !actor.procedure_feedback.outcome_rx.is_closed() => actor.procedure_feedback.outcome_primed = msg,
            msg = actor.perception.room_input_rx.recv(), if !actor.perception.room_input_rx.is_closed() => {
                if let Some(item) = msg {
                    actor.perception.pending_room_input.push_back(item);
                }
            }
            msg = actor.perception.kl_reentry_rx.recv(), if !actor.perception.kl_reentry_rx.is_closed() => actor.perception.kl_reentry_primed = msg,
            msg = actor.perception.act_reentry_rx.recv(), if !actor.perception.act_reentry_rx.is_closed() => actor.perception.act_reentry_primed = msg,
            msg = actor.prediction.embedding_reentry_rx.recv(), if !actor.prediction.embedding_reentry_rx.is_closed() => actor.prediction.embedding_reentry_primed = msg,
        }
    }
}

/// Whether `mod.rs`'s `self_status_interrupt_due` gate (minus `incoming.is_none()`,
/// which only resolves during `tick()` itself) is currently satisfied - the
/// self-status interrupt's *own* dedicated wake condition, decoupled from
/// `max_idle_interval_ms` entirely (see that field's doc comment).
///
/// Not modeled as a `CognitiveEventQueue` deadline like `boredom_deadline`/
/// `synthesis_deadline`: unlike a wall-clock timer, "has `self_monitoring_pressure`
/// crossed the threshold" isn't something this scheduler can forecast a future
/// instant for - drive pressure only changes once per actual tick. Checking
/// the *current* reading directly, right alongside `pending_admission` in the
/// fast path above, needs no forecast and carries none of `synthesis_deadline`'s
/// former hot-loop risk: this can only ever return `true` when a tick's own
/// gate would also see the same threshold already crossed, so a tick this
/// causes either fires the interrupt (resetting `last_boredom_at`, clearing
/// the condition) or finds real competing input (`incoming.is_some()`) that
/// tick's own busy-ness already accounts for.
fn self_status_interrupt_ready(actor: &CognitiveLoopActor) -> bool {
    let boredom_config = &actor.config.boredom_config;
    !actor.config.ablation_config.disable_boredom
        // Same "unreachable disables this path" convention the threshold's
        // own doc comment documents (every drive is clamped to `[0, 1]`).
        && boredom_config.self_status_interrupt_drive_threshold <= 1.0
        && actor.tool_registry.available().iter().any(|(name, _)| *name == crate::steps::tools::SelfStatusTool::NAME)
        && actor.drives.drive_state.self_monitoring_pressure() >= boredom_config.self_status_interrupt_drive_threshold
        && actor.perception.last_boredom_at.is_none_or(|t| actor.clock.now().0.saturating_sub(t.0) >= boredom_config.min_interval_ms)
}

/// Boredom requires both continuous idleness and the retry interval to have
/// elapsed. The later deadline is therefore the first useful wake-up; taking
/// the earlier one makes an already-past idle threshold permanently hot.
fn boredom_deadline(
    wm_empty_since: Option<i64>,
    last_boredom_at: Option<i64>,
    idle_threshold_ms: i64,
    min_interval_ms: i64,
) -> Option<i64> {
    let idle_ready = wm_empty_since?.saturating_add(idle_threshold_ms);
    let retry_ready =
        last_boredom_at.map_or(idle_ready, |last| last.saturating_add(min_interval_ms));
    Some(idle_ready.max(retry_ready))
}

/// A partial consolidation buffer is not actionable. Once the buffer crosses
/// `backlog_pressure_threshold` (`tick()`'s own `synthesis_backlog_pressure`
/// - the *unconditional* trigger, independent of whether this tick happens
/// to be idle), a never-attempted synthesis is due now; subsequent attempts
/// observe the configured wall-clock interval.
///
/// Deliberately does *not* mint an eager deadline the moment
/// `pending_count` merely reaches `min_new_episodic` (below backlog
/// pressure): `tick()` only actually attempts synthesis there when this
/// tick happens to be idle, a fact this scheduler cannot see ahead of
/// time (`idle_now` depends on whether *this* tick's incoming stimulus
/// produces a new object). An eager deadline was tried and reverted here -
/// confirmed live as a real hot loop: with `last_synthesis_at` never set
/// (nothing ever idle enough to attempt it), this function returned `now`
/// on every single call, so `wait_for_next_tick` stopped sleeping at all
/// under a sustained non-idle trickle - exactly the busy-poll regression
/// the scheduler exists to prevent. Below backlog pressure, an eventual
/// genuinely idle tick is instead left to the ordinary channel-wake/
/// `IdleMaintenance` cadence.
fn synthesis_deadline(
    pending_count: usize,
    min_new_episodic: usize,
    backlog_pressure_threshold: usize,
    last_synthesis_at: Option<i64>,
    min_interval_ms: i64,
    now: i64,
) -> Option<i64> {
    if pending_count < min_new_episodic || pending_count < backlog_pressure_threshold {
        return None;
    }
    Some(last_synthesis_at.map_or(now, |last| last.saturating_add(min_interval_ms)))
}

#[cfg(test)]
mod tests {
    use super::{boredom_deadline, synthesis_deadline, EarliestDeadline};

    #[test]
    fn earliest_deadline_tracks_the_minimum_regardless_of_consideration_order() {
        let mut deadline = EarliestDeadline::default();
        assert_eq!(deadline.0, None, "no candidate considered yet");
        deadline.consider(250);
        deadline.consider(80);
        deadline.consider(10);
        deadline.consider(20);
        assert_eq!(deadline.0, Some(10), "the smallest due_at considered so far should win regardless of order");
        deadline.consider(500);
        assert_eq!(deadline.0, Some(10), "a later, larger candidate must not displace an earlier, smaller one");
    }

    #[test]
    fn boredom_waits_for_both_idle_and_retry_gates() {
        assert_eq!(
            boredom_deadline(Some(1_000), None, 45_000, 120_000),
            Some(46_000)
        );
        assert_eq!(
            boredom_deadline(Some(1_000), Some(40_000), 45_000, 120_000),
            Some(160_000)
        );
        assert_eq!(
            boredom_deadline(Some(100_000), Some(1_000), 45_000, 120_000),
            Some(145_000)
        );
        assert_eq!(boredom_deadline(None, Some(40_000), 45_000, 120_000), None);
    }

    #[test]
    fn synthesis_ignores_partial_buffers_and_observes_retry_interval() {
        // Below `min_new_episodic`: never due.
        assert_eq!(synthesis_deadline(2, 3, 6, None, 60_000, 10_000), None);
        // At/above `min_new_episodic` but below backlog pressure: not
        // actionable by the scheduler (see this function's own doc comment
        // on the hot-loop this used to cause) - left to ordinary wakes.
        assert_eq!(synthesis_deadline(3, 3, 6, None, 60_000, 10_000), None);
        assert_eq!(synthesis_deadline(5, 3, 6, Some(8_000), 60_000, 10_000), None);
        // At/above backlog pressure: due now (first attempt) or at the
        // configured retry interval after the last attempt.
        assert_eq!(synthesis_deadline(6, 3, 6, None, 60_000, 10_000), Some(10_000));
        assert_eq!(
            synthesis_deadline(6, 3, 6, Some(8_000), 60_000, 10_000),
            Some(68_000)
        );
    }
}
