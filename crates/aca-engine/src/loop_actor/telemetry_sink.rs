use std::collections::HashMap;

use aca_store::{CycleEvent, CycleEventKind, CyclePhase};
use aca_util::Clock;
use tokio::sync::{broadcast, watch};

use crate::snapshot::EngineSnapshot;

/// This actor's own event bus and `tracing` mirror - every `CyclePhase`
/// emits through `record`, which both appends to the outward-facing
/// `pending_events`/`events_tx` (unthrottled - every consumer still gets
/// every cycle event) and mirrors to `tracing` through `log_cycle_event`
/// (throttled - see that method's own doc comment for the flood this
/// prevents).
pub(crate) struct TelemetrySink {
    pub(crate) pending_events: Vec<CycleEvent>,
    /// Throttles the `tracing` mirror only (not `pending_events`/
    /// `events_tx`, which every consumer still gets every tick, unthrottled)
    /// - `(coarse signature, consecutive repeat count)` per `CyclePhase`,
    /// tracked independently per phase rather than as one single "most
    /// recently logged event" slot. That distinction is load-bearing: a
    /// single tick fires Coalition, then Executive, then Act, then Learn in
    /// sequence, so comparing only against the immediately-preceding event
    /// (regardless of phase) meant the "is this a repeat" check almost
    /// never matched at all - Executive's signature is never equal to
    /// Coalition's, by construction, no matter how steady the actual state
    /// is. Confirmed live: with the single-slot version, a fully-settled
    /// Working Memory item spinning at ~1000 ticks/sec still wrote every
    /// single Coalition/Executive/Act/Learn line unthrottled, because each
    /// one only ever got compared against a *different* phase's event.
    /// Per-phase tracking is what actually lets a steady state collapse:
    /// this tick's Coalition compares against *last tick's* Coalition, not
    /// this tick's own Learn.
    ///
    /// Before the cognitive scheduler was introduced, the loop re-settled on
    /// the same terminal operator hundreds of times a second. Logging every
    /// identical repeat was a confirmed flood: a 435 MB log in twenty
    /// minutes with no throttle, then 632,000 lines with the single-slot
    /// version above. Scheduling now prevents that idle spin, while this
    /// throttle remains useful for genuinely repeated event-driven states. The
    /// signature deliberately ignores noisy numeric fields (activation
    /// scores carry small per-tick jitter even when nothing meaningful
    /// changed) so a genuinely steady state is recognized as one, not
    /// treated as constant novelty.
    pub(crate) log_repeat: HashMap<CyclePhase, (String, u32)>,
    pub(crate) events_tx: broadcast::Sender<CycleEvent>,
    pub(crate) snapshot_tx: watch::Sender<EngineSnapshot>,
}

impl TelemetrySink {
    /// How many consecutive identical-signature repeats of the same
    /// `Normal`-kind event `log_cycle_event` lets pass before logging one
    /// more as a heartbeat. It is deliberately count-based rather than tied
    /// to an assumed tick rate now that the actor is event-driven.
    pub(crate) const LOG_HEARTBEAT_EVERY: u32 = 500;

    pub(crate) fn record(
        &mut self,
        cycle_seq: u64,
        clock: &dyn Clock,
        phase: CyclePhase,
        kind: CycleEventKind,
        tier: Option<aca_types::Tier>,
        payload: serde_json::Value,
    ) {
        let event = CycleEvent::new(cycle_seq, clock.now(), phase, kind, tier, payload);
        // Every cycle event, mirrored to `tracing` at `debug` level - opt-in
        // only (`RUST_LOG=aca_engine::loop_actor=debug` or similar), silent
        // under this daemon's default `info` filter, so a released binary
        // prints nothing extra by default. `/events`/`pending_events` alone
        // made a stall like a never-tagged failed Act/ContinueReflecting
        // reconstructible only after the fact from cycle-count gaps (see
        // `steps::act`'s `Operator::Act`/`ContinueReflecting` arms' own doc
        // comments on that exact failure mode) - enabling this makes it
        // visible line-by-line, live, for whoever's actually debugging it.
        // Throttled through `log_cycle_event` regardless of level - see
        // that method's own doc comment for why an unthrottled mirror
        // floods output once the loop settles into its documented
        // no-idle-backoff spin.
        self.log_cycle_event(&event);
        // Skips the clone entirely when nothing is subscribed (the common
        // case for a headless run) - `emit_event` fires up to ~8 times per
        // tick, and `CycleEvent`'s `serde_json::Value` payload makes that a
        // real, avoidable allocation otherwise.
        if self.events_tx.receiver_count() > 0 {
            let _ = self.events_tx.send(event.clone());
        }
        self.pending_events.push(event);
    }

    /// See `log_repeat`'s own doc comment for the flood this exists to
    /// prevent. `Impasse`/`Escalation`/`Error` events bypass the throttle
    /// entirely (and reset it) - they're rare enough on their own that
    /// collapsing them would only ever hide something worth seeing, never
    /// save meaningful volume.
    ///
    /// Level is picked per event, not fixed at `debug` for everything: the
    /// routine per-tick churn (Coalition candidate scores, a repeated
    /// Ignore/Silent settling on an already-handled object) is genuinely
    /// debug-only noise, silent by default - but `Impasse`/`Escalation`/
    /// `Error` are always worth seeing, and so is the actual conversational
    /// substance: `Compare` carries whatever text was just perceived (the
    /// "question"), and a `speak` outcome from `Act` carries what was
    /// actually said (the "response", alongside the separate `println!` in
    /// `omega-acad::main` that already prints it unconditionally).
    /// Demoting *everything* to `debug` here once already hid this pair by
    /// accident - confirmed live, this is what fixes that back.
    fn log_cycle_event(&mut self, event: &CycleEvent) {
        let always_visible = event.event_type != CycleEventKind::Normal
            || event.phase == CyclePhase::Compare
            || (event.phase == CyclePhase::Act && event.payload.get("operator").and_then(|v| v.as_str()) == Some("speak"));

        if event.event_type != CycleEventKind::Normal {
            self.log_repeat.remove(&event.phase);
            if always_visible {
                tracing::info!(cycle_seq = event.cycle_seq, phase = ?event.phase, kind = ?event.event_type, tier = ?event.tier_used, payload = %event.payload, "cycle event");
            } else {
                tracing::debug!(cycle_seq = event.cycle_seq, phase = ?event.phase, kind = ?event.event_type, tier = ?event.tier_used, payload = %event.payload, "cycle event");
            }
            return;
        }
        let signature = log_signature(&event.payload);
        match self.log_repeat.get_mut(&event.phase) {
            Some((last_signature, count)) if *last_signature == signature => {
                *count += 1;
                if *count % Self::LOG_HEARTBEAT_EVERY == 0 {
                    if always_visible {
                        tracing::info!(cycle_seq = event.cycle_seq, phase = ?event.phase, kind = ?event.event_type, tier = ?event.tier_used, payload = %event.payload, repeated = *count, "cycle event (still repeating)");
                    } else {
                        tracing::debug!(cycle_seq = event.cycle_seq, phase = ?event.phase, kind = ?event.event_type, tier = ?event.tier_used, payload = %event.payload, repeated = *count, "cycle event (still repeating)");
                    }
                }
            }
            _ => {
                self.log_repeat.insert(event.phase, (signature, 1));
                if always_visible {
                    tracing::info!(cycle_seq = event.cycle_seq, phase = ?event.phase, kind = ?event.event_type, tier = ?event.tier_used, payload = %event.payload, "cycle event");
                } else {
                    tracing::debug!(cycle_seq = event.cycle_seq, phase = ?event.phase, kind = ?event.event_type, tier = ?event.tier_used, payload = %event.payload, "cycle event");
                }
            }
        }
    }
}

/// A coarse, noise-stripped stand-in for a cycle event's payload, used only
/// to decide whether `log_cycle_event` should treat two consecutive events
/// as "the same thing happening again" - never persisted, never sent to a
/// consumer. Floating-point fields (activation/preference/confidence
/// scores, all of which carry small per-tick jitter even when nothing
/// meaningful changed - `ACT-R` base-level activation recomputes every
/// tick, `sample_noise` adds fresh jitter every tick) are normalized away so
/// that jitter alone doesn't defeat the repeat detection; integers, strings,
/// and bools - the fields that actually carry a real state change - are
/// kept as-is.
pub(crate) fn log_signature(payload: &serde_json::Value) -> String {
    fn strip_floats(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Number(n) if n.is_f64() => serde_json::Value::Null,
            serde_json::Value::Object(map) => serde_json::Value::Object(map.iter().map(|(k, v)| (k.clone(), strip_floats(v))).collect()),
            serde_json::Value::Array(items) => serde_json::Value::Array(items.iter().map(strip_floats).collect()),
            other => other.clone(),
        }
    }
    strip_floats(payload).to_string()
}
