use aca_types::{ActivationState, AssociativeEdge, EdgeKind, MentalObjectDynamics, MentalObjectId};
use aca_util::{Clock, EpochMillis, RingBuffer};
use rand::Rng;

/// ACT-R's base-level activation: `ln(Σ_j (t - t_j)^-d)` over past reference
/// timestamps `t_j`, decay `d`. Recent/frequent references dominate the sum
/// (power-law forgetting) — this is the "Memory Decay" mechanism from
/// specs.md, applied to activation, never to storage.
///
/// Returns `f32::NEG_INFINITY` for an empty reference log (never referenced,
/// never retrievable) so the retrieval-threshold gate excludes it naturally
/// without a special case at the call site.
pub fn compute_base_level(reference_log: &RingBuffer<EpochMillis>, decay_d: f32, now: EpochMillis) -> f32 {
    if reference_log.is_empty() {
        return f32::NEG_INFINITY;
    }
    let sum: f64 = reference_log
        .iter()
        .map(|&t_j| {
            // Guard the singularity at delta=0 (a reference at the exact
            // current instant) by flooring the elapsed time at 1ms — ACT-R's
            // formula is undefined at delta=0, not just large.
            let delta_ms = t_j.elapsed_ms_until(now).max(1) as f64;
            let delta_units = delta_ms / 1000.0; // seconds, an arbitrary but fixed unit
            delta_units.powf(-(decay_d as f64))
        })
        .sum();
    (sum.ln()) as f32
}

/// ACT-R's spreading activation: `Σ_k W_k * S_ki` — for candidate `target_id`,
/// sum over every currently-active context source `k` the learned
/// associative strength `S_ki` of any edge from `k` to `target_id`, weighted
/// by `k`'s attentional weight. v1 uses fixed uniform weights (`1 /
/// |active_sources|`) rather than a formally-derived attentional-capacity
/// allocation (see the plan's named simplifications). `S_ki` is read via
/// `edges::effective_strength` (time-decayed from `last_coactivated_at`),
/// never the raw stored `strength` — `decay_rate_per_ms = 0.0` is a true
/// no-op (`exp(0) = 1.0`) for callers that don't want decay.
///
/// Takes each source's outgoing edges directly rather than the whole
/// `MentalObject` (its edges are the only field this ever reads) - lets a
/// caller pass a lightweight per-tick snapshot instead of cloning every
/// Working Memory member's full text/embedding/JSON payload just to spread
/// activation from it.
pub fn compute_spreading_activation(
    target_id: MentalObjectId,
    active_sources: &[&[AssociativeEdge]],
    now: EpochMillis,
    decay_rate_per_ms: f64,
) -> f32 {
    if active_sources.is_empty() {
        return 0.0;
    }
    let uniform_weight = 1.0 / active_sources.len() as f32;
    active_sources
        .iter()
        .map(|edges| {
            let s_ki: f32 = edges
                .iter()
                .filter(|edge| edge.target_id == target_id)
                .map(|edge| {
                    let sign = if matches!(edge.kind, EdgeKind::Inhibitory | EdgeKind::Contradicts) { -1.0 } else { 1.0 };
                    sign * crate::edges::effective_strength(edge, now, decay_rate_per_ms)
                })
                .sum();
            uniform_weight * s_ki
        })
        .sum()
}

/// A small tie-breaking jitter, uniform in `[-max_noise, max_noise]`. Named
/// simplification: this is not empirically calibrated to human retrieval-
/// time distributions (ACT-R's own logistic noise term) — it exists purely
/// so ranking never has to fall back to an arbitrary/unstable tie-break
/// among otherwise-identical activation values.
pub fn sample_noise(rng: &mut impl Rng, max_noise: f32) -> f32 {
    if max_noise <= 0.0 {
        return 0.0;
    }
    rng.gen_range(-max_noise..=max_noise)
}

/// Recomputes and writes back `base_level`, `spreading`, `noise`, `total`,
/// and `last_computed_at` for a single activation state, given the current
/// set of active context sources it should spread from.
pub fn recompute_activation(
    state: &mut ActivationState,
    target_id: MentalObjectId,
    active_sources: &[&[AssociativeEdge]],
    max_noise: f32,
    edge_decay_rate_per_ms: f64,
    rng: &mut impl Rng,
    clock: &dyn Clock,
) {
    let now = clock.now();
    state.base_level = compute_base_level(&state.reference_log, state.decay_d, now);
    state.spreading = compute_spreading_activation(target_id, active_sources, now, edge_decay_rate_per_ms);
    state.noise = sample_noise(rng, max_noise);
    state.total = state.base_level + state.spreading + state.noise;
    state.last_computed_at = now;
}

/// Only objects whose total activation clears this threshold are eligible
/// recall/coalition candidates.
pub fn clears_retrieval_threshold(total_activation: f32, threshold: f32) -> bool {
    total_activation > threshold
}

/// Records a fresh reference (creation, recall, or reinforcement) by pushing
/// `at` onto the bounded reference log — this is what "reinforcement
/// restores activation" means concretely: a new entry in the base-level sum.
pub fn record_reference(state: &mut ActivationState, at: EpochMillis) {
    state.reference_log.push(at);
}

/// Leaks short-lived potential and firing adaptation forward to `now`.
/// Nothing is scheduled merely to perform this decay: it is evaluated lazily
/// when an event next touches the object, keeping dormant objects at zero
/// runtime cost.
pub fn leak_dynamics(state: &mut MentalObjectDynamics, now: EpochMillis, potential_tau_ms: f32, adaptation_tau_ms: f32) {
    let elapsed_ms = state.last_updated_at.elapsed_ms_until(now).max(0) as f32;
    if elapsed_ms > 0.0 {
        state.potential *= (-elapsed_ms / potential_tau_ms.max(1.0)).exp();
        state.adaptation *= (-elapsed_ms / adaptation_tau_ms.max(1.0)).exp();
        state.last_updated_at = now;
    }
}

/// Applies one local input pulse after lazy leakage. Negative pulses are
/// inhibition and cannot push potential below zero.
pub fn stimulate_dynamics(
    state: &mut MentalObjectDynamics,
    input: f32,
    now: EpochMillis,
    potential_tau_ms: f32,
    adaptation_tau_ms: f32,
) {
    leak_dynamics(state, now, potential_tau_ms, adaptation_tau_ms);
    state.potential = (state.potential + input).max(0.0);
}

/// Emits a sparse firing event when potential clears the adapted threshold
/// outside the refractory window. Excess potential is retained, while firing
/// raises adaptation and installs a short object-local cooldown.
pub fn try_fire_dynamics(state: &mut MentalObjectDynamics, now: EpochMillis, refractory_ms: i64, adaptation_increment: f32) -> bool {
    if state.refractory_until.is_some_and(|until| now < until) || state.potential < state.threshold + state.adaptation {
        return false;
    }
    state.potential = (state.potential - state.threshold).max(0.0);
    state.adaptation += adaptation_increment.max(0.0);
    state.last_fired_at = Some(now);
    state.refractory_until = Some(EpochMillis(now.0.saturating_add(refractory_ms.max(0))));
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_util::ManualClock;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn log_with(refs: &[i64]) -> RingBuffer<EpochMillis> {
        let mut log = RingBuffer::new(64);
        for &t in refs {
            log.push(EpochMillis(t));
        }
        log
    }

    #[test]
    fn empty_reference_log_is_negative_infinity() {
        let log = log_with(&[]);
        let base = compute_base_level(&log, 0.5, EpochMillis(10_000));
        assert_eq!(base, f32::NEG_INFINITY);
    }

    #[test]
    fn more_recent_references_yield_higher_base_level() {
        let recent = log_with(&[9_000]);
        let stale = log_with(&[1_000]);
        let now = EpochMillis(10_000);
        let recent_base = compute_base_level(&recent, 0.5, now);
        let stale_base = compute_base_level(&stale, 0.5, now);
        assert!(
            recent_base > stale_base,
            "recent={recent_base} should exceed stale={stale_base}"
        );
    }

    #[test]
    fn more_references_yield_higher_base_level_than_one() {
        let many = log_with(&[1_000, 3_000, 5_000, 7_000, 9_000]);
        let one = log_with(&[9_000]);
        let now = EpochMillis(10_000);
        assert!(compute_base_level(&many, 0.5, now) > compute_base_level(&one, 0.5, now));
    }

    #[test]
    fn same_instant_reference_does_not_produce_nan_or_infinity() {
        let log = log_with(&[10_000]);
        let base = compute_base_level(&log, 0.5, EpochMillis(10_000));
        assert!(base.is_finite(), "expected finite, got {base}");
    }

    #[test]
    fn spreading_activation_sums_weighted_edge_strengths() {
        use aca_types::{AssociativeEdge, EdgeKind, MentalObject};

        let target = MentalObjectId::new();
        let mut source_a = MentalObject::new_observation("a", EpochMillis(0), 0.5);
        source_a.edges.push(AssociativeEdge {
            target_id: target,
            kind: EdgeKind::Associative,
            strength: 0.8,
            last_coactivated_at: EpochMillis(0),
        });
        let mut source_b = MentalObject::new_observation("b", EpochMillis(0), 0.5);
        source_b.edges.push(AssociativeEdge {
            target_id: target,
            kind: EdgeKind::Associative,
            strength: 0.4,
            last_coactivated_at: EpochMillis(0),
        });

        let sources: Vec<&[AssociativeEdge]> = vec![&source_a.edges, &source_b.edges];
        let spreading = compute_spreading_activation(target, &sources, EpochMillis(0), 0.0);
        // uniform weight = 1/2 each: 0.5*0.8 + 0.5*0.4 = 0.6
        assert!((spreading - 0.6).abs() < 1e-6, "got {spreading}");
    }

    #[test]
    fn spreading_activation_is_zero_with_no_sources() {
        let target = MentalObjectId::new();
        assert_eq!(compute_spreading_activation(target, &[], EpochMillis(0), 0.0), 0.0);
    }

    #[test]
    fn spreading_activation_decays_with_edge_age() {
        use aca_types::{AssociativeEdge, EdgeKind, MentalObject};

        let target = MentalObjectId::new();
        let mut source = MentalObject::new_observation("a", EpochMillis(0), 0.5);
        source.edges.push(AssociativeEdge {
            target_id: target,
            kind: EdgeKind::Associative,
            strength: 1.0,
            last_coactivated_at: EpochMillis(0),
        });
        let sources: Vec<&[AssociativeEdge]> = vec![&source.edges];

        let no_decay = compute_spreading_activation(target, &sources, EpochMillis(1_000_000), 0.0);
        let decayed = compute_spreading_activation(target, &sources, EpochMillis(1_000_000), 0.001);
        assert!((no_decay - 1.0).abs() < 1e-6, "decay_rate=0.0 must be a true no-op");
        assert!(decayed < no_decay, "an aged edge should contribute less spreading activation");
    }

    #[test]
    fn retrieval_threshold_gate_is_strict_greater_than() {
        assert!(!clears_retrieval_threshold(1.0, 1.0));
        assert!(clears_retrieval_threshold(1.01, 1.0));
        assert!(!clears_retrieval_threshold(f32::NEG_INFINITY, -100.0));
    }

    #[test]
    fn recompute_activation_writes_back_all_fields() {
        let clock = ManualClock::new(EpochMillis(10_000));
        let mut rng = StdRng::seed_from_u64(42);
        let target = MentalObjectId::new();
        let mut state = ActivationState::new_at(EpochMillis(9_000), 0.5);
        recompute_activation(&mut state, target, &[], 0.05, 0.0, &mut rng, &clock);
        assert!(state.base_level.is_finite());
        assert_eq!(state.spreading, 0.0);
        assert!(state.noise.abs() <= 0.05);
        assert_eq!(state.last_computed_at, EpochMillis(10_000));
        assert!((state.total - (state.base_level + state.spreading + state.noise)).abs() < 1e-6);
    }

    #[test]
    fn inhibitory_and_contradictory_edges_subtract_from_excitation() {
        use aca_types::{AssociativeEdge, MentalObject};

        let target = MentalObjectId::new();
        let mut source = MentalObject::new_observation("source", EpochMillis(0), 0.5);
        for (kind, strength) in [(EdgeKind::Associative, 0.8), (EdgeKind::Inhibitory, 0.5), (EdgeKind::Contradicts, 0.1)] {
            source.edges.push(AssociativeEdge { target_id: target, kind, strength, last_coactivated_at: EpochMillis(0) });
        }
        let result = compute_spreading_activation(target, &[&source.edges], EpochMillis(0), 0.0);
        assert!((result - 0.2).abs() < 1e-6);
    }

    #[test]
    fn dynamics_leak_accumulate_fire_and_refract() {
        let mut state = MentalObjectDynamics::new_at(EpochMillis(0));
        stimulate_dynamics(&mut state, 0.6, EpochMillis(0), 1_000.0, 5_000.0);
        assert!(!try_fire_dynamics(&mut state, EpochMillis(0), 400, 0.2));
        stimulate_dynamics(&mut state, 0.6, EpochMillis(100), 1_000.0, 5_000.0);
        assert!(try_fire_dynamics(&mut state, EpochMillis(100), 400, 0.2));
        stimulate_dynamics(&mut state, 2.0, EpochMillis(200), 1_000.0, 5_000.0);
        assert!(!try_fire_dynamics(&mut state, EpochMillis(200), 400, 0.2), "the refractory window should suppress immediate re-firing");
        assert!(try_fire_dynamics(&mut state, EpochMillis(500), 400, 0.2));
    }

    #[test]
    fn inhibitory_pulses_cannot_make_potential_negative() {
        let mut state = MentalObjectDynamics::new_at(EpochMillis(0));
        stimulate_dynamics(&mut state, -3.0, EpochMillis(0), 1_000.0, 5_000.0);
        assert_eq!(state.potential, 0.0);
    }
}
