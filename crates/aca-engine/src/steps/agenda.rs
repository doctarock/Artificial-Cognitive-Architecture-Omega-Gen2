use std::collections::HashSet;

use aca_graph::Graph;
use aca_types::{GoalStackId, GoalStackMembership, GoalStatus, MemoryRole, MentalObject, MentalObjectId, MentalObjectKind};
use aca_util::EpochMillis;
use serde::{Deserialize, Serialize};

use super::drives::DriveState;

/// Persistent, competing-for-attention `Intention` Mental Objects - Stage 1
/// of a "will" subsystem: drives (`steps::drives`) and unresolved
/// discrepancies exert pressure, at most one genuinely new Intention forms
/// per tick when that pressure is strong enough, and every existing Active
/// intention decays with real inertia rather than being regenerated from
/// scratch each tick.
///
/// Reuses `MentalObjectKind::Intention` and `MentalObject.goal:
/// Option<GoalStackMembership>` as-is (both were fully modeled but had no
/// production writer before this module) - `GoalStatus` is the intention's
/// lifecycle, `GoalStackMembership::priority` is its commitment score.
/// Intention-specific bookkeeping (why it matters, what's expected, what to
/// do next, which drive it came from, replan cadence, decay bookkeeping)
/// lives in `MentalObject.data` as `IntentionPayload` - that field's own
/// doc comment already calls out "tighten once real usage settles the
/// shapes"; this is that real usage.
///
/// Two entry points, run at two different points in `loop_actor::tick()`:
/// `surface_active_intentions` early (before Predict, using last tick's
/// commitment scores) gives existing intentions a real shot at this tick's
/// attention; `revise_agenda` late (after Act/Learn, using this tick's
/// fresh signals) is where commitment actually changes, plan-derived
/// Reflections get folded back into their parent intention, status
/// transitions happen, and - at most once per tick - a new intention may be
/// spawned. See each function's own doc comment for why they're split this
/// way and can't be merged into one pass.
#[derive(Debug, Clone, Copy)]
pub struct AgendaConfig {
    /// How many of the highest-commitment Active intentions get a fresh
    /// `record_reference` + `pending_admission` shot at attention each
    /// tick. Small and constant-cost by design - this runs unconditionally
    /// every tick, so it must never scale with graph size.
    pub top_k_surfaced: usize,
    /// Minimum wall-clock gap between re-nominating a dormant intention.
    /// Resident intentions already compete in Working Memory and need no
    /// artificial reference pulse on maintenance wakes.
    pub surface_interval_ms: i64,
    /// Below this commitment, an intention is considered to have
    /// effectively lost the competition for pursuit - not deleted (nothing
    /// in this architecture deletes), but no longer worth surfacing.
    pub commitment_floor: f32,
    /// Number of nominal 250ms maintenance intervals an intention must stay below
    /// `commitment_floor` before it's actually transitioned to `Suspended`
    /// (or `Abandoned`, if already `Suspended` once before) - a brief dip
    /// shouldn't end an intention outright; a sustained one should. See
    /// `revise_agenda`'s guard against an intention lingering at the floor
    /// forever.
    pub floor_grace_ticks: u32,
    /// Minimum wall-clock gap between an Active intention's `Plan`
    /// proposals - `steps::executive::propose_operators`'s gate. Intentions
    /// replan periodically, not exactly once ever (unlike
    /// `executive::has_reflection_for`'s "exactly once" dedup for ordinary
    /// Reflections), so this needs its own cadence rather than reusing that
    /// pattern.
    pub replan_interval_ms: i64,
    /// Minimum wall-clock gap between spawning new intentions from the same
    /// drive, even if that drive's pressure looks like it clears
    /// `spawn_threshold` on every intervening tick - a cost-control
    /// backstop independent of whether the trigger condition looks
    /// satisfied, same role `SynthesisConfig::min_interval_ms` plays for
    /// pattern synthesis.
    pub min_spawn_interval_ms: i64,
    /// The strongest current drive pressure (`DriveState::strongest`) must
    /// clear this before a new intention is worth spawning at all.
    pub spawn_threshold: f32,
    /// A small fixed cost subtracted from every commitment computation -
    /// real cost modeling (estimating what pursuing an intention actually
    /// costs) is out of scope for this stage; this just keeps commitment
    /// from being pure upside.
    pub estimated_cost: f32,
    /// How much of the *sum* of every other Active intention's commitment
    /// discounts a fresh commitment computation - competing intentions
    /// genuinely compete, so more of them (or stronger ones) makes each
    /// individual one's own commitment harder to sustain.
    pub competing_intention_discount: f32,
    /// Multiplicative decay per nominal 250ms maintenance interval applied to every Active intention's
    /// commitment in `revise_agenda`'s unconditional decay pass, unless
    /// that same tick also folds in genuine fresh evidence (a completed
    /// plan) for it. `0.98` loses about 45% of an unreinforced intention's
    /// commitment over 25 seconds - noticeable within a normal session,
    /// never instant.
    pub decay_factor: f32,
}

impl Default for AgendaConfig {
    fn default() -> Self {
        Self {
            top_k_surfaced: 2,
            surface_interval_ms: 10_000,
            commitment_floor: 0.15,
            floor_grace_ticks: 20,
            replan_interval_ms: 2 * 60 * 1000,
            min_spawn_interval_ms: 3 * 60 * 1000,
            spawn_threshold: 0.6,
            estimated_cost: 0.1,
            competing_intention_discount: 0.15,
            decay_factor: 0.98,
        }
    }
}

/// Intention-specific bookkeeping stored in `MentalObject.data` - never
/// duplicates commitment itself (that's `GoalStackMembership::priority`,
/// the one score that actually needs to be read by `predict`/`coalition`-
/// adjacent code outside this module).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct IntentionPayload {
    /// Why this intention was worth forming - the drive/discrepancy that
    /// justified it, in words a model prompt can use.
    reason: String,
    /// What pursuing this intention is expected to produce.
    expected_outcome: String,
    /// The concrete next step, refreshed each time `revise_agenda` folds in
    /// a fresh plan-derived Reflection.
    next_action: String,
    /// Which `DriveState` field this intention originated from
    /// (`"uncertainty"`, `"curiosity"`, etc.) - the dedup key
    /// `try_spawn_intention` uses to refuse a second intention for a drive
    /// that already has one Active or Suspended.
    source_drive: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_planned_at: Option<EpochMillis>,
    /// Missing on old memories means the first surface is immediately due.
    #[serde(default)]
    last_surfaced_at: Option<EpochMillis>,
    /// Last maintenance instant. Missing on old memories: the first read
    /// applies one nominal interval, then timestamps control subsequent decay.
    #[serde(default)]
    last_revised_at: Option<EpochMillis>,
    #[serde(default)]
    below_floor_since: Option<EpochMillis>,
    /// Whether this intention has already been `Suspended` once before -
    /// a second trip to the floor after that abandons it outright rather
    /// than suspending it again indefinitely.
    #[serde(default)]
    previously_suspended: bool,
}

impl IntentionPayload {
    fn from_object(object: &MentalObject) -> Option<Self> {
        serde_json::from_value(object.data.clone()).ok()
    }

    fn write(&self, object: &mut MentalObject) {
        object.data = serde_json::to_value(self).expect("IntentionPayload always serializes");
    }
}

/// Whether `object` is a plan-derived Reflection - `steps::act::act`'s
/// `Operator::Plan` success arm tags a freshly-created Reflection
/// `data: {"source": "plan", "for_intention": <id>}` before inserting it.
/// Checked by provenance, not text or confidence, same discipline as every
/// other "don't mistake my own output for new content" guard already in
/// this engine (e.g. `executive::propose_operators`'s
/// `is_a_tools_own_result`).
pub fn is_plan_reflection(object: &MentalObject) -> bool {
    object.kind == MentalObjectKind::Reflection && object.data.get("source").and_then(|v| v.as_str()) == Some("plan")
}

/// The parent intention id a plan-derived Reflection was produced for -
/// `None` for anything `is_plan_reflection` doesn't already agree is one.
fn plan_reflection_target(object: &MentalObject) -> Option<MentalObjectId> {
    if !is_plan_reflection(object) {
        return None;
    }
    object.data.get("for_intention").and_then(|v| v.as_str()).and_then(|s| s.parse().ok())
}

/// Whether `object` is an Active Intention due for a fresh `Operator::Plan`
/// - `steps::executive::propose_operators`'s gate. A freshly-spawned
/// intention with no `last_planned_at` yet always counts as due; anything
/// that isn't an Active Intention at all is never due (`false`, not an
/// error - callers already branch on `object.kind` separately).
pub fn intention_due_to_replan(object: &MentalObject, now: EpochMillis, replan_interval_ms: i64) -> bool {
    if object.kind != MentalObjectKind::Intention {
        return false;
    }
    if !matches!(&object.goal, Some(goal) if goal.status == GoalStatus::Active) {
        return false;
    }
    let last_planned_at = IntentionPayload::from_object(object).and_then(|payload| payload.last_planned_at);
    last_planned_at.is_none_or(|t| now.0 - t.0 >= replan_interval_ms)
}

/// Gives up to `config.top_k_surfaced` highest-commitment dormant Active
/// intentions a cadence-bounded shot at attention: a fresh
/// `aca_graph::record_reference` (the same lever `steps::synthesize`'s
/// reinforce-existing-pattern path already uses to keep an object's ACT-R
/// activation alive without minting a duplicate) plus insertion into
/// `pending_admission`, the "one honest shot at admission" mechanism
/// `loop_actor` already uses for a freshly-synthesized pattern and a
/// mid-tick Reflection.
///
/// `record_reference` alone is not sufficient: `loop_actor::tick`'s
/// `raw_candidates` is built only from this tick's fresh input, current
/// `working_memory` members, and `pending_admission` - once an object has
/// ever dropped out of Working Memory, nothing re-nominates it on its own,
/// no matter how high its activation would score if it were reconsidered.
/// Without this second step, a persistent intention's commitment would
/// never actually compete for attention again after its first admission.
///
/// The check runs every tick, but a maintenance wake is not fresh evidence
/// and cannot mint another reference pulse inside `surface_interval_ms`.
/// Resident intentions are already nominated by Working Memory. The sparse
/// goal index avoids scanning unrelated autobiographical objects.
pub fn surface_active_intentions(graph: &mut Graph, working_memory: &HashSet<MentalObjectId>, pending_admission: &mut HashSet<MentalObjectId>, config: &AgendaConfig, now: EpochMillis) -> Vec<MentalObjectId> {
    let mut active: Vec<(MentalObjectId, f32)> = graph
        .goal_objects()
        .filter(|object| object.kind == MentalObjectKind::Intention)
        .filter_map(|object| object.goal.as_ref().map(|goal| (object.id, goal)))
        .filter(|(_, goal)| goal.status == GoalStatus::Active)
        .map(|(id, goal)| (id, goal.priority))
        .collect();
    active.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut newly_surfaced = Vec::new();
    for (id, _commitment) in active {
        if newly_surfaced.len() >= config.top_k_surfaced {
            break;
        }
        if working_memory.contains(&id) || pending_admission.contains(&id) {
            continue;
        }
        let Some(object) = graph.get_mut(&id) else { continue };
        // Old goal-carrying Intention memories may predate `IntentionPayload`.
        // They still deserve their first pulse; initialize the missing
        // cadence field rather than silently making them unreachable.
        let mut payload = IntentionPayload::from_object(object).unwrap_or_default();
        if payload.last_surfaced_at.is_some_and(|last| now.0.saturating_sub(last.0) < config.surface_interval_ms.max(1)) {
            continue;
        }
        aca_graph::record_reference(&mut object.activation, now);
        payload.last_surfaced_at = Some(now);
        payload.write(object);
        pending_admission.insert(id);
        newly_surfaced.push(id);
    }
    newly_surfaced
}

/// What `revise_agenda` actually did this tick - purely for the caller to
/// decide what's worth a cycle event and which ids need `dirty_ids`
/// entries; `revise_agenda` itself never touches `loop_actor`'s state
/// directly.
#[derive(Debug, Default)]
pub struct AgendaOutcome {
    /// Every intention (and consumed plan-Reflection) id mutated this tick.
    pub dirty_ids: Vec<MentalObjectId>,
    /// The newly-spawned intention's id, if this tick's drive pressure
    /// cleared `spawn_threshold` and no existing intention already covers
    /// that drive.
    pub spawned: Option<MentalObjectId>,
    /// `(intention_id, new_status)` for every status transition this tick.
    pub transitions: Vec<(MentalObjectId, GoalStatus)>,
}

/// The late-phase half of `steps::agenda`: given this tick's fresh drive
/// readings, (1) folds any plan-derived Reflections created this tick back
/// into their parent intention, (2) unconditionally decays every Active
/// intention's commitment, transitioning status once it's spent enough
/// consecutive ticks at the floor, and (3) spawns at most one new intention
/// if the current strongest drive pressure clears `config.spawn_threshold`
/// and no existing Active/Suspended intention already covers that drive.
///
/// Runs unconditionally every tick, same reasoning as
/// `surface_active_intentions` - an intention's decay must not depend on
/// the system happening to be idle, or a busy system would let commitment
/// silently freeze instead of decaying.
pub fn revise_agenda(graph: &mut Graph, dirty_ids_this_tick: &HashSet<MentalObjectId>, drives: &DriveState, config: &AgendaConfig, now: EpochMillis, decay_d: f32) -> AgendaOutcome {
    let mut outcome = AgendaOutcome::default();

    // Step 1: fold in this tick's plan-derived Reflections. Consumed
    // (`produced_by_operator` set) so a later tick's `dirty_ids` scan never
    // double-folds the same Reflection twice, and so
    // `executive::propose_operators`'s own `is_plan_reflection` early
    // return is redundant-but-harmless on any later tick it happens to be
    // re-nominated (it already never gets a real proposal either way).
    let mut folded_plans: Vec<(MentalObjectId, MentalObjectId, String)> = Vec::new();
    for &id in dirty_ids_this_tick {
        let Some(reflection) = graph.get(&id) else { continue };
        if reflection.produced_by_operator.is_some() {
            continue;
        }
        if let Some(intention_id) = plan_reflection_target(reflection) {
            folded_plans.push((id, intention_id, reflection.text.clone()));
        }
    }
    for (reflection_id, intention_id, plan_text) in folded_plans {
        if let Some(reflection) = graph.get_mut(&reflection_id) {
            reflection.produced_by_operator = Some("Plan".to_string());
        }
        outcome.dirty_ids.push(reflection_id);
        if let Some(intention) = graph.get_mut(&intention_id) {
            let mut payload = IntentionPayload::from_object(intention).unwrap_or_default();
            // `expected_outcome` ("what pursuing this intention is expected
            // to produce") is intentionally left alone here - `plan_text` is
            // the next concrete step, not the intention's outcome, and
            // nothing else populates `expected_outcome` yet (it's set at
            // spawn to an honest empty string; see the constructor below).
            payload.next_action = plan_text;
            payload.last_planned_at = Some(now);
            payload.write(intention);
            // Real evidence, not attention alone (see `AgendaConfig::
            // decay_factor`'s doc comment on why commitment must not rise
            // just from being nominated/attended to): a plan actually
            // completing counts as `expected_information_gain` realized,
            // so it earns a genuine, bounded commitment bump rather than
            // being left to decay like an unreinforced tick.
            if let Some(goal) = intention.goal.as_mut() {
                goal.priority = (goal.priority + 0.1).clamp(0.0, 1.0);
            }
            outcome.dirty_ids.push(intention_id);
        }
    }

    // Step 2: unconditional decay + status transitions for every Active
    // intention. Collected as owned ids first (not mutated while iterating
    // `graph.iter()`) since the loop body needs `graph.get_mut`.
    let active_ids: Vec<MentalObjectId> = graph.goal_objects().filter(|object| object.kind == MentalObjectKind::Intention).filter_map(|object| object.goal.as_ref().map(|_| object.id)).collect();
    for id in active_ids {
        let Some(object) = graph.get_mut(&id) else { continue };
        if object.goal.as_ref().is_none_or(|goal| goal.status != GoalStatus::Active) {
            continue;
        }
        // A commitment bumped by this tick's plan-folding step above
        // should not *also* be decayed this same tick - it just received
        // genuine fresh evidence.
        let already_reinforced_this_tick = outcome.dirty_ids.contains(&id);
        let mut payload = IntentionPayload::from_object(object).unwrap_or_default();
        let elapsed_ms = payload.last_revised_at.map_or(250,
            |last| now.0.saturating_sub(last.0).max(0));
        payload.last_revised_at = Some(now);
        if !already_reinforced_this_tick {
            if let Some(goal) = object.goal.as_mut() {
                goal.priority *= config.decay_factor.clamp(0.0, 1.0).powf(elapsed_ms as f32 / 250.0);
            }
        }
        let below_floor = object.goal.as_ref().is_some_and(|goal| goal.priority <= config.commitment_floor);
        if below_floor {
            payload.below_floor_since.get_or_insert(now);
        } else {
            payload.below_floor_since = None;
        }
        let grace_ms = (config.floor_grace_ticks.max(1) as i64).saturating_mul(250);
        let should_transition = payload.below_floor_since.is_some_and(|since|
            now.0.saturating_sub(since.0) >= grace_ms);
        if should_transition {
            let new_status = if payload.previously_suspended { GoalStatus::Abandoned } else { GoalStatus::Suspended };
            payload.previously_suspended = true;
            payload.below_floor_since = None;
            payload.write(object);
            if let Some(goal) = object.goal.as_mut() {
                goal.status = new_status;
            }
            outcome.transitions.push((id, new_status));
        } else {
            payload.write(object);
        }
        outcome.dirty_ids.push(id);
    }

    // Step 3: at most one new intention, only when genuinely warranted.
    let (drive_name, drive_value) = drives.strongest();
    if drive_value >= config.spawn_threshold {
        // Every intention ever spawned for this drive, any status -
        // Abandoned/Suspended ones are never deleted (nothing in this
        // architecture deletes), so they remain real evidence of *when*
        // this drive last got a fresh intention, not just *whether* one is
        // currently active.
        let existing_for_drive: Vec<&MentalObject> = graph
            .iter()
            .filter(|object| object.kind == MentalObjectKind::Intention)
            .filter(|object| IntentionPayload::from_object(object).is_some_and(|payload| payload.source_drive == drive_name))
            .collect();
        // Primary gate: one Active/Suspended intention per drive at a time.
        let already_covered = existing_for_drive.iter().any(|object| matches!(&object.goal, Some(goal) if matches!(goal.status, GoalStatus::Active | GoalStatus::Suspended)));
        // Backstop gate, independent of status: even an intention that
        // already resolved (Abandoned) for this drive must not be
        // immediately replaced by a fresh one - without this, a
        // chronically-above-threshold drive whose intentions happen to
        // abandon quickly (e.g. a low initial commitment that decays past
        // `commitment_floor` within `floor_grace_ticks`) would spawn a new
        // one practically every tick, exactly the runaway-repetition
        // failure class this engine has hit before (see `AgendaConfig`'s
        // own module-level doc comment). Mirrors `SynthesisConfig::
        // min_interval_ms`'s identical role for pattern synthesis.
        let spawned_too_recently = existing_for_drive.iter().any(|object| now.0 - object.created_at.0 < config.min_spawn_interval_ms);
        if !already_covered && !spawned_too_recently {
            let competing_commitment: f32 = graph
                .iter()
                .filter(|object| object.kind == MentalObjectKind::Intention)
                .filter_map(|object| object.goal.as_ref())
                .filter(|goal| goal.status == GoalStatus::Active)
                .map(|goal| goal.priority)
                .sum();
            // spec §7's formula, Stage 1's real-signal approximation: no
            // independent "epistemic value at spawn" reading exists yet
            // for every drive (only `uncertainty` literally *is* one) -
            // `drive_value` stands in for both `intrinsic_value` (how
            // strong the pressure itself is) and `expected_information_gain`
            // (how much resolving it is expected to help), a documented
            // simplification rather than two genuinely independent signals.
            let intrinsic_value = drive_value;
            let expected_information_gain = drive_value;
            let commitment = (intrinsic_value + expected_information_gain - config.estimated_cost - config.competing_intention_discount * competing_commitment).clamp(0.0, 1.0);

            let mut intention = MentalObject::new_observation(format!("a pressure worth acting on: {drive_name}"), now, decay_d);
            intention.kind = MentalObjectKind::Intention;
            intention.confidence = drive_value;
            intention.goal = Some(GoalStackMembership { stack_id: GoalStackId::new(), parent_goal_id: None, status: GoalStatus::Active, priority: commitment });
            IntentionPayload {
                reason: format!("{drive_name} pressure reached {drive_value:.2}, above the {:.2} spawn threshold", config.spawn_threshold),
                expected_outcome: String::new(),
                next_action: String::new(),
                source_drive: drive_name.to_string(),
                last_planned_at: None,
                last_surfaced_at: None,
                last_revised_at: None,
                below_floor_since: None,
                previously_suspended: false,
            }
            .write(&mut intention);
            intention.memory_roles.push(MemoryRole::Working);
            let new_id = intention.id;
            graph.insert(intention);
            outcome.dirty_ids.push(new_id);
            outcome.spawned = Some(new_id);
        }
    }

    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    fn active_intention(text: &str, priority: f32, source_drive: &str, now: EpochMillis, decay_d: f32) -> MentalObject {
        let mut intention = MentalObject::new_observation(text, now, decay_d);
        intention.kind = MentalObjectKind::Intention;
        intention.goal = Some(GoalStackMembership { stack_id: GoalStackId::new(), parent_goal_id: None, status: GoalStatus::Active, priority });
        IntentionPayload { source_drive: source_drive.to_string(), ..Default::default() }.write(&mut intention);
        intention
    }

    #[test]
    fn is_plan_reflection_matches_only_the_tagged_shape() {
        let mut reflection = MentalObject::new_observation("a plan", EpochMillis(0), 0.5);
        reflection.kind = MentalObjectKind::Reflection;
        assert!(!is_plan_reflection(&reflection));
        reflection.data = serde_json::json!({"source": "plan", "for_intention": MentalObjectId::new().to_string()});
        assert!(is_plan_reflection(&reflection));

        let mut other_source = MentalObject::new_observation("a tool result", EpochMillis(0), 0.5);
        other_source.kind = MentalObjectKind::Reflection;
        other_source.data = serde_json::json!({"source": "tool"});
        assert!(!is_plan_reflection(&other_source));
    }

    #[test]
    fn intention_due_to_replan_is_true_for_a_never_planned_active_intention() {
        let intention = active_intention("do something", 0.5, "curiosity", EpochMillis(0), 0.5);
        assert!(intention_due_to_replan(&intention, EpochMillis(1_000_000), 60_000));
    }

    #[test]
    fn intention_due_to_replan_respects_the_cadence() {
        let mut intention = active_intention("do something", 0.5, "curiosity", EpochMillis(0), 0.5);
        let mut payload = IntentionPayload::from_object(&intention).unwrap();
        payload.last_planned_at = Some(EpochMillis(1_000));
        payload.write(&mut intention);

        assert!(!intention_due_to_replan(&intention, EpochMillis(1_500), 60_000), "well within the cadence, should not be due yet");
        assert!(intention_due_to_replan(&intention, EpochMillis(1_000 + 60_000), 60_000), "at exactly the cadence, should be due again");
    }

    #[test]
    fn intention_due_to_replan_is_false_for_non_intentions_and_non_active_ones() {
        let observation = MentalObject::new_observation("just an observation", EpochMillis(0), 0.5);
        assert!(!intention_due_to_replan(&observation, EpochMillis(1_000_000), 60_000));

        let mut suspended = active_intention("paused", 0.5, "curiosity", EpochMillis(0), 0.5);
        suspended.goal.as_mut().unwrap().status = GoalStatus::Suspended;
        assert!(!intention_due_to_replan(&suspended, EpochMillis(1_000_000), 60_000));
    }

    #[test]
    fn surface_active_intentions_surfaces_only_the_top_k_by_commitment() {
        let mut graph = Graph::new();
        let low = active_intention("low commitment", 0.2, "curiosity", EpochMillis(0), 0.5);
        let mid = active_intention("mid commitment", 0.5, "uncertainty", EpochMillis(0), 0.5);
        let high = active_intention("high commitment", 0.9, "competence", EpochMillis(0), 0.5);
        let (low_id, mid_id, high_id) = (low.id, mid.id, high.id);
        graph.insert(low);
        graph.insert(mid);
        graph.insert(high);

        let config = AgendaConfig { top_k_surfaced: 2, ..AgendaConfig::default() };
        let mut pending_admission = HashSet::new();
        let surfaced = surface_active_intentions(&mut graph, &HashSet::new(), &mut pending_admission, &config, EpochMillis(1_000));

        assert_eq!(surfaced.len(), 2);
        assert!(surfaced.contains(&high_id));
        assert!(surfaced.contains(&mid_id));
        assert!(!surfaced.contains(&low_id));
        assert!(pending_admission.contains(&high_id));
        assert!(pending_admission.contains(&mid_id));
    }

    #[test]
    fn surface_active_intentions_skips_ids_already_in_working_memory() {
        let mut graph = Graph::new();
        let intention = active_intention("already resident", 0.9, "curiosity", EpochMillis(0), 0.5);
        let id = intention.id;
        graph.insert(intention);

        let mut working_memory = HashSet::new();
        working_memory.insert(id);
        let mut pending_admission = HashSet::new();
        let surfaced = surface_active_intentions(&mut graph, &working_memory, &mut pending_admission, &AgendaConfig::default(), EpochMillis(1_000));

        assert!(surfaced.is_empty(), "already-resident intentions don't need a pending_admission entry");
        assert!(pending_admission.is_empty());
    }

    #[test]
    fn resident_intentions_do_not_get_fake_reference_pulses_on_idle_wakes() {
        let mut graph = Graph::new();
        let intention = active_intention("already resident", 0.9, "curiosity", EpochMillis(0), 0.5);
        let id = intention.id;
        let initial_references = intention.activation.reference_log.len();
        graph.insert(intention);

        let mut working_memory = HashSet::new();
        working_memory.insert(id);
        let mut pending_admission = HashSet::new();
        surface_active_intentions(&mut graph, &working_memory, &mut pending_admission, &AgendaConfig::default(), EpochMillis(5_000));

        assert_eq!(graph.get(&id).unwrap().activation.reference_log.len(), initial_references);
        assert!(pending_admission.is_empty());
    }

    #[test]
    fn dormant_intention_is_surfaced_once_per_wall_clock_cadence() {
        let mut graph = Graph::new();
        let intention = active_intention("a dormant goal", 0.9, "curiosity", EpochMillis(0), 0.5);
        let id = intention.id;
        graph.insert(intention);
        let mut pending = HashSet::new();
        let config = AgendaConfig { surface_interval_ms: 10_000, ..AgendaConfig::default() };

        assert_eq!(surface_active_intentions(&mut graph, &HashSet::new(), &mut pending, &config, EpochMillis(1_000)), vec![id]);
        pending.clear(); // lost admission this tick
        let references_after_first = graph.get(&id).unwrap().activation.reference_log.len();
        assert!(surface_active_intentions(&mut graph, &HashSet::new(), &mut pending, &config, EpochMillis(1_250)).is_empty());
        assert!(surface_active_intentions(&mut graph, &HashSet::new(), &mut pending, &config, EpochMillis(10_999)).is_empty());
        assert_eq!(graph.get(&id).unwrap().activation.reference_log.len(), references_after_first);
        assert_eq!(surface_active_intentions(&mut graph, &HashSet::new(), &mut pending, &config, EpochMillis(11_000)), vec![id]);
        assert_eq!(graph.get(&id).unwrap().activation.reference_log.len(), references_after_first + 1);
    }

    #[test]
    fn revise_agenda_decays_commitment_when_nothing_reinforces_it() {
        let mut graph = Graph::new();
        let intention = active_intention("fading", 0.5, "curiosity", EpochMillis(0), 0.5);
        let id = intention.id;
        graph.insert(intention);

        let drives = DriveState::default();
        let config = AgendaConfig::default();
        revise_agenda(&mut graph, &HashSet::new(), &drives, &config, EpochMillis(1_000), 0.5);

        let after = graph.get(&id).unwrap().goal.as_ref().unwrap().priority;
        assert!((after - 0.5 * config.decay_factor).abs() < 1e-6, "got {after}");
    }

    #[test]
    fn intention_decay_tracks_elapsed_time_not_number_of_scheduler_wakes() {
        let intention = active_intention("fading", 0.8, "curiosity", EpochMillis(0), 0.5);
        let id = intention.id;
        let mut sparse = Graph::new();
        let mut frequent = Graph::new();
        sparse.insert(intention.clone());
        frequent.insert(intention);
        let config = AgendaConfig::default();
        for graph in [&mut sparse, &mut frequent] {
            revise_agenda(graph, &HashSet::new(), &DriveState::default(), &config, EpochMillis(0), 0.5);
        }
        revise_agenda(&mut sparse, &HashSet::new(), &DriveState::default(), &config, EpochMillis(1_000), 0.5);
        for at in [250, 500, 750, 1_000] {
            revise_agenda(&mut frequent, &HashSet::new(), &DriveState::default(), &config, EpochMillis(at), 0.5);
        }
        let a = sparse.get(&id).unwrap().goal.as_ref().unwrap().priority;
        let b = frequent.get(&id).unwrap().goal.as_ref().unwrap().priority;
        assert!((a - b).abs() < 1e-6, "same wall time should yield the same commitment: {a} vs {b}");
    }

    #[test]
    fn revise_agenda_folds_a_plan_reflection_back_into_its_parent_intention() {
        let mut graph = Graph::new();
        let intention = active_intention("find out about the weather", 0.5, "curiosity", EpochMillis(0), 0.5);
        let intention_id = intention.id;
        let initial_priority = intention.goal.as_ref().unwrap().priority;
        graph.insert(intention);

        let mut plan_reflection = MentalObject::new_observation("check a weather tool", EpochMillis(1_000), 0.5);
        plan_reflection.kind = MentalObjectKind::Reflection;
        plan_reflection.data = serde_json::json!({"source": "plan", "for_intention": intention_id.to_string()});
        let reflection_id = plan_reflection.id;
        graph.insert(plan_reflection);

        let mut dirty = HashSet::new();
        dirty.insert(reflection_id);

        let outcome = revise_agenda(&mut graph, &dirty, &DriveState::default(), &AgendaConfig::default(), EpochMillis(2_000), 0.5);

        assert!(outcome.dirty_ids.contains(&reflection_id));
        assert!(outcome.dirty_ids.contains(&intention_id));
        let consumed = graph.get(&reflection_id).unwrap();
        assert_eq!(consumed.produced_by_operator.as_deref(), Some("Plan"));

        let intention = graph.get(&intention_id).unwrap();
        assert_eq!(intention.goal.as_ref().unwrap().priority, (initial_priority + 0.1).clamp(0.0, 1.0), "a completed plan should bump commitment, not decay it");
        let payload = IntentionPayload::from_object(intention).unwrap();
        assert_eq!(payload.next_action, "check a weather tool");
        assert!(payload.last_planned_at.is_some());
    }

    #[test]
    fn revise_agenda_does_not_refold_an_already_consumed_plan_reflection() {
        let mut graph = Graph::new();
        let intention = active_intention("already planned", 0.5, "curiosity", EpochMillis(0), 0.5);
        let intention_id = intention.id;
        graph.insert(intention);

        let mut consumed_reflection = MentalObject::new_observation("an old plan", EpochMillis(0), 0.5);
        consumed_reflection.kind = MentalObjectKind::Reflection;
        consumed_reflection.data = serde_json::json!({"source": "plan", "for_intention": intention_id.to_string()});
        consumed_reflection.produced_by_operator = Some("Plan".to_string());
        let reflection_id = consumed_reflection.id;
        graph.insert(consumed_reflection);

        let mut dirty = HashSet::new();
        dirty.insert(reflection_id);
        let outcome = revise_agenda(&mut graph, &dirty, &DriveState::default(), &AgendaConfig::default(), EpochMillis(1_000), 0.5);

        assert!(!outcome.dirty_ids.contains(&reflection_id), "an already-consumed plan reflection must not be folded again");
    }

    #[test]
    fn revise_agenda_suspends_then_abandons_an_intention_that_stays_below_the_floor() {
        let mut graph = Graph::new();
        let intention = active_intention("losing steam", 0.1, "curiosity", EpochMillis(0), 0.5);
        let id = intention.id;
        graph.insert(intention);

        let config = AgendaConfig { floor_grace_ticks: 2, decay_factor: 1.0, ..AgendaConfig::default() };
        let drives = DriveState::default();

        // Tick 1: below floor, grace counter starts at 1 - not yet transitioned.
        let outcome1 = revise_agenda(&mut graph, &HashSet::new(), &drives, &config, EpochMillis(1_000), 0.5);
        assert!(outcome1.transitions.is_empty());
        assert_eq!(graph.get(&id).unwrap().goal.as_ref().unwrap().status, GoalStatus::Active);

        // Tick 2: grace counter reaches 2 - suspended.
        let outcome2 = revise_agenda(&mut graph, &HashSet::new(), &drives, &config, EpochMillis(2_000), 0.5);
        assert_eq!(outcome2.transitions, vec![(id, GoalStatus::Suspended)]);
        assert_eq!(graph.get(&id).unwrap().goal.as_ref().unwrap().status, GoalStatus::Suspended);
    }

    #[test]
    fn revise_agenda_never_transitions_an_intention_that_recovers_above_the_floor() {
        let mut graph = Graph::new();
        // Starts comfortably above the floor and never decays below it
        // (decay_factor = 1.0), so it should never accumulate any
        // below-floor ticks at all.
        let intention = active_intention("holding steady", 0.9, "curiosity", EpochMillis(0), 0.5);
        let id = intention.id;
        graph.insert(intention);

        let config = AgendaConfig { floor_grace_ticks: 2, decay_factor: 1.0, commitment_floor: 0.15, ..AgendaConfig::default() };
        for i in 0..10 {
            let outcome = revise_agenda(&mut graph, &HashSet::new(), &DriveState::default(), &config, EpochMillis(1_000 * i), 0.5);
            assert!(outcome.transitions.is_empty());
        }
        assert_eq!(graph.get(&id).unwrap().goal.as_ref().unwrap().status, GoalStatus::Active);
    }

    #[test]
    fn revise_agenda_spawns_a_new_intention_when_drive_pressure_clears_the_threshold() {
        let mut graph = Graph::new();
        let mut drives = DriveState::default();
        for _ in 0..50 {
            drives.update(None, 1.0, None, None, 0.0, 0.0); // drives curiosity toward 1.0
        }

        let config = AgendaConfig::default();
        let outcome = revise_agenda(&mut graph, &HashSet::new(), &drives, &config, EpochMillis(1_000), 0.5);

        let spawned_id = outcome.spawned.expect("strong curiosity pressure should spawn a new intention");
        let spawned = graph.get(&spawned_id).unwrap();
        assert_eq!(spawned.kind, MentalObjectKind::Intention);
        assert_eq!(spawned.goal.as_ref().unwrap().status, GoalStatus::Active);
        let payload = IntentionPayload::from_object(spawned).unwrap();
        assert_eq!(payload.source_drive, "curiosity");
    }

    #[test]
    fn revise_agenda_never_spawns_a_second_intention_for_a_drive_already_covered() {
        let mut graph = Graph::new();
        graph.insert(active_intention("already pursuing curiosity", 0.5, "curiosity", EpochMillis(0), 0.5));

        let mut drives = DriveState::default();
        for _ in 0..50 {
            drives.update(None, 1.0, None, None, 0.0, 0.0);
        }

        let outcome = revise_agenda(&mut graph, &HashSet::new(), &drives, &AgendaConfig::default(), EpochMillis(1_000), 0.5);
        assert!(outcome.spawned.is_none(), "a drive that already has an Active intention must not spawn a second one");
    }

    #[test]
    fn revise_agenda_spawns_nothing_when_no_drive_clears_the_threshold() {
        let mut graph = Graph::new();
        let outcome = revise_agenda(&mut graph, &HashSet::new(), &DriveState::default(), &AgendaConfig::default(), EpochMillis(1_000), 0.5);
        assert!(outcome.spawned.is_none());
        assert_eq!(graph.len(), 0);
    }
}
