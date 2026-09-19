use aca_graph::Graph;
use aca_types::MentalObjectId;
use aca_util::EpochMillis;
use serde_json::json;

/// The candidate-count and per-candidate `activation_total` ranges
/// `training/attention-v0/generate_scenarios_v0_2.py` samples
/// from (candidate count 1-9; activation clamped to
/// `[-10.0, 10.0]` — see `SCHEMA_V0_2.md`). The deterministic
/// `rank_candidates`/`admit_top_n` algorithm generalizes to any input by
/// construction (it's just arithmetic and a sort); the trained model has
/// no such guarantee outside what it was shown. These bounds are the one
/// concrete, checkable proxy for "is this tick's workspace shaped like
/// reproducible training data". The generator restored in this checkout is
/// not the exact historical training recipe, so this is not proof that the
/// installed model saw every in-range value or judges it well.
const TRAINING_CANDIDATE_COUNT_RANGE: std::ops::RangeInclusive<usize> = 1..=9;
const TRAINING_ACTIVATION_RANGE: std::ops::RangeInclusive<f32> = -10.0..=10.0;

/// Best-effort out-of-distribution flag for this tick's candidate set,
/// checked *before* the attention model is ever consulted - see `tick()`'s
/// Step 5/6 boundary. `false` means the workspace falls outside what
/// `generate_scenarios_v0_2.py` samples (candidate count, activation,
/// and at most one finite surprise), so the model is skipped in favor of
/// the deterministic algorithm for that tick rather than trusting an
/// extrapolation with no principled confidence signal behind it, and the
/// result is surfaced in `attention_out_of_distribution` telemetry so real-world
/// distribution shift over a long-running graph is observable, not silent.
pub(crate) fn workspace_in_distribution(raw_candidates: &[(MentalObjectId, f32, Option<f32>)]) -> bool {
    TRAINING_CANDIDATE_COUNT_RANGE.contains(&raw_candidates.len())
        && raw_candidates.iter().all(|(_, activation_total, _)| TRAINING_ACTIVATION_RANGE.contains(activation_total))
        && raw_candidates.iter().filter(|(_, _, surprise)| surprise.is_some()).count() <= 1
        && raw_candidates.iter().all(|(_, _, surprise)| surprise.is_none_or(f32::is_finite))
}

/// Builds the `omega-attention-workspace/v0.2` JSON the attention model
/// expects — see `training/attention-v0/SCHEMA_V0_2.md`, the current
/// contract both implementations (this and the Python generator)
/// must stay faithful to. Every field maps 1:1 onto a real value already
/// computed elsewhere in the tick; none are fabricated proxies.
///
/// `raw_candidates` is the same `(MentalObjectId, activation_total,
/// surprise)` tuple Step 5 Coalition already scores from — built once in
/// `loop_actor::tick`, before either Coalition or this call runs.
pub(crate) fn build_attention_workspace_json(
    graph: &Graph,
    raw_candidates: &[(MentalObjectId, f32, Option<f32>)],
    current_focus: Option<MentalObjectId>,
    working_memory_capacity: usize,
    attention_threshold: f32,
    now: EpochMillis,
) -> String {
    let candidates: Vec<_> = raw_candidates
        .iter()
        .filter_map(|(id, activation_total, surprise)| {
            let object = graph.get(id)?;
            let goal_priority = object.goal.as_ref().map(|g| g.priority);
            let age_ms = (now.0 - object.created_at.0).max(0);
            Some(json!({
                "id": id.to_string(),
                "kind": object.kind,
                "activation_total": activation_total,
                "surprise": surprise,
                "confidence": object.confidence,
                "goal_priority": goal_priority,
                "in_working_memory": object.workspace.in_working_memory,
                "broadcast_count": object.workspace.broadcast_count,
                "age_ms": age_ms,
            }))
        })
        .collect();

    let workspace = json!({
        "protocol": "omega-attention-workspace/v0.2",
        "attention_threshold": attention_threshold,
        "working_memory_capacity": working_memory_capacity,
        "current_focus": current_focus.map(|id| id.to_string()),
        "candidates": candidates,
    });
    serde_json::to_string(&workspace).expect("a json! Value built entirely from primitives and MentalObjectKind's own Serialize impl is always serializable")
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_types::{GoalStackId, GoalStackMembership, GoalStatus, MentalObject, MentalObjectKind};

    #[test]
    fn empty_candidates_are_out_of_distribution() {
        // Zero candidates is below the training range's floor of 1 - the
        // generator never sampled an empty workspace, so this predicate
        // says so honestly. In practice `loop_actor::tick` never actually
        // asks: it bypasses the whole attention-arbitration path (this
        // predicate included) before ever calling it whenever
        // `raw_candidates` is empty - see that fast path's own doc comment
        // (confirmed live: without it, an idle loop firing ~1000
        // ticks/sec would otherwise flood the event log calling the model,
        // or this predicate, on every single one).
        assert!(!workspace_in_distribution(&[]));
    }

    #[test]
    fn a_typical_candidate_set_is_in_distribution() {
        let raw_candidates = vec![(MentalObjectId::new(), 2.5, None), (MentalObjectId::new(), -1.0, Some(0.6))];
        assert!(workspace_in_distribution(&raw_candidates));
    }

    #[test]
    fn more_than_nine_candidates_is_out_of_distribution() {
        let raw_candidates: Vec<_> = (0..10).map(|_| (MentalObjectId::new(), 1.0, None)).collect();
        assert!(!workspace_in_distribution(&raw_candidates));
    }

    #[test]
    fn an_activation_value_far_outside_the_trained_range_is_out_of_distribution() {
        let raw_candidates = vec![(MentalObjectId::new(), 47.0, None)];
        assert!(!workspace_in_distribution(&raw_candidates));
    }

    #[test]
    fn boundary_activation_values_are_still_in_distribution() {
        let raw_candidates = vec![(MentalObjectId::new(), -10.0, None), (MentalObjectId::new(), 10.0, None)];
        assert!(workspace_in_distribution(&raw_candidates));
    }

    #[test]
    fn multiple_surprises_are_outside_the_generated_workspace_shape() {
        let raw_candidates = vec![(MentalObjectId::new(), 1.0, Some(0.3)), (MentalObjectId::new(), 2.0, Some(0.4))];
        assert!(!workspace_in_distribution(&raw_candidates));
    }

    #[test]
    fn nonfinite_surprise_cannot_be_silently_serialized_as_null() {
        let raw_candidates = vec![(MentalObjectId::new(), 1.0, Some(f32::NAN))];
        assert!(!workspace_in_distribution(&raw_candidates));
    }

    #[test]
    fn builds_the_v0_2_schema_shape_from_real_fields() {
        let mut graph = Graph::new();
        let mut object = MentalObject::new_observation("hello", EpochMillis(10_000), 0.5);
        object.confidence = 0.62;
        object.workspace.in_working_memory = true;
        object.workspace.broadcast_count = 7;
        let id = object.id;
        graph.insert(object);

        let raw_candidates = vec![(id, 3.598, None)];
        let json_text = build_attention_workspace_json(&graph, &raw_candidates, None, 4, -2.0, EpochMillis(13_479));
        let parsed: serde_json::Value = serde_json::from_str(&json_text).unwrap();

        assert_eq!(parsed["protocol"], "omega-attention-workspace/v0.2");
        assert_eq!(parsed["attention_threshold"], -2.0);
        assert_eq!(parsed["working_memory_capacity"], 4);
        assert!(parsed["current_focus"].is_null());
        let candidate = &parsed["candidates"][0];
        assert_eq!(candidate["id"], id.to_string());
        assert_eq!(candidate["kind"], "observation");
        assert!((candidate["activation_total"].as_f64().unwrap() - 3.598).abs() < 1e-5, "f32->f64 JSON round-trip should be close, not bit-identical");
        assert!(candidate["surprise"].is_null());
        assert!((candidate["confidence"].as_f64().unwrap() - 0.62).abs() < 1e-5);
        assert!(candidate["goal_priority"].is_null());
        assert_eq!(candidate["in_working_memory"], true);
        assert_eq!(candidate["broadcast_count"], 7);
        assert_eq!(candidate["age_ms"], 3479);
    }

    #[test]
    fn goal_priority_is_populated_only_for_goal_stack_members() {
        let mut graph = Graph::new();
        let mut object = MentalObject::new_observation("a goal", EpochMillis(0), 0.5);
        object.kind = MentalObjectKind::Goal;
        object.goal = Some(GoalStackMembership {
            stack_id: GoalStackId::new(),
            parent_goal_id: None,
            status: GoalStatus::Active,
            priority: 0.77,
        });
        let id = object.id;
        graph.insert(object);

        let raw_candidates = vec![(id, 1.0, Some(0.5))];
        let json_text = build_attention_workspace_json(&graph, &raw_candidates, Some(id), 4, -2.0, EpochMillis(0));
        let parsed: serde_json::Value = serde_json::from_str(&json_text).unwrap();

        assert_eq!(parsed["current_focus"], id.to_string());
        assert!((parsed["candidates"][0]["goal_priority"].as_f64().unwrap() - 0.77).abs() < 1e-5);
        assert!((parsed["candidates"][0]["surprise"].as_f64().unwrap() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn a_raw_candidate_missing_from_the_graph_is_skipped_not_a_panic() {
        let graph = Graph::new();
        let raw_candidates = vec![(MentalObjectId::new(), 1.0, None)];
        let json_text = build_attention_workspace_json(&graph, &raw_candidates, None, 4, -2.0, EpochMillis(0));
        let parsed: serde_json::Value = serde_json::from_str(&json_text).unwrap();
        assert!(parsed["candidates"].as_array().unwrap().is_empty());
    }
}
