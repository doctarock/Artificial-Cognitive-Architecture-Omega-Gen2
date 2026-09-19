use std::collections::HashMap;

use aca_graph::{Graph, record_reference};
use aca_types::{MemoryRole, MentalObject, MentalObjectKind, ObjectStatus};
use aca_util::EpochMillis;

const SCHEMA: &str = "omega-curated-static-answer/v1";

#[derive(Debug, Clone, PartialEq)]
pub struct CuratedAnswer {
    pub answer: String,
    pub question_embedding: Vec<f32>,
}

/// Exact key only: no approximate match may silently turn one static fact
/// into an answer to a different or time-sensitive question.
pub fn static_question_key(question: &str) -> Option<String> {
    let trimmed = question.trim();
    if !trimmed.ends_with('?') || trimmed.chars().any(|c| c.is_numeric()) {
        return None;
    }
    let key = trimmed
        .trim_end_matches('?')
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let words: Vec<_> = key.split_whitespace().collect();
    if words.is_empty()
        || words.len() > 16
        || !matches!(words[0], "what" | "where" | "which")
        || words.iter().any(|word| {
            matches!(
                *word,
                "now"
                    | "today"
                    | "tomorrow"
                    | "yesterday"
                    | "latest"
                    | "current"
                    | "weather"
                    | "time"
                    | "price"
                    | "cost"
                    | "stock"
                    | "market"
                    | "president"
                    | "prime"
                    | "minister"
                    | "ceo"
                    | "open"
                    | "available"
            )
        })
    {
        return None;
    }
    Some(key)
}

fn safe_static_answer(answer: &str) -> bool {
    let trimmed = answer.trim();
    !trimmed.is_empty()
        && !trimmed.contains('\n')
        && !trimmed.to_ascii_lowercase().contains("http")
        && !trimmed.chars().any(|c| c.is_numeric())
        && trimmed.split_whitespace().count() <= 32
}

/// Loads only explicit curated-answer memories with a valid cached question
/// embedding. Other Semantic memories and generated Reflections never enter
/// this index merely because their text resembles a factual answer.
pub fn curated_answer_index(graph: &Graph) -> HashMap<String, CuratedAnswer> {
    let mut latest = HashMap::<String, &MentalObject>::new();
    for object in graph.iter() {
        if object.status != ObjectStatus::Active
            || object.kind != MentalObjectKind::Memory
            || !object.memory_roles.contains(&MemoryRole::Semantic)
            || object.data.get("schema").and_then(|value| value.as_str()) != Some(SCHEMA)
        {
            continue;
        }
        let Some(key) = object
            .data
            .get("question")
            .and_then(|value| value.as_str())
            .and_then(static_question_key)
        else {
            continue;
        };
        if !object
            .data
            .get("answer")
            .and_then(|value| value.as_str())
            .is_some_and(safe_static_answer)
            || !object.embedding.as_ref().is_some_and(|embedding| {
                !embedding.is_empty() && embedding.iter().all(|value| value.is_finite())
            })
        {
            continue;
        }
        latest
            .entry(key)
            .and_modify(|prior| {
                if (object.created_at, object.id) > (prior.created_at, prior.id) {
                    *prior = object;
                }
            })
            .or_insert(object);
    }
    latest
        .into_iter()
        .map(|(key, object)| {
            (
                key,
                CuratedAnswer {
                    answer: object.data["answer"]
                        .as_str()
                        .expect("validated answer")
                        .to_string(),
                    question_embedding: object.embedding.clone().expect("validated embedding"),
                },
            )
        })
        .collect()
}

/// Explicit trusted ingestion, not automatic compilation from a model
/// response. The caller is responsible for curating the static fact.
pub fn record_curated_answer(
    graph: &mut Graph,
    question: &str,
    answer: &str,
    question_embedding: &[f32],
    now: EpochMillis,
    decay_d: f32,
) -> Option<aca_types::MentalObjectId> {
    let key = static_question_key(question)?;
    if !safe_static_answer(answer)
        || question_embedding.is_empty()
        || !question_embedding.iter().all(|value| value.is_finite())
    {
        return None;
    }
    let existing_id = graph
        .iter()
        .filter(|object| {
            object.status == ObjectStatus::Active
                && object.data.get("schema").and_then(|value| value.as_str()) == Some(SCHEMA)
                && object
                    .data
                    .get("question")
                    .and_then(|value| value.as_str())
                    .and_then(static_question_key)
                    .as_deref()
                    == Some(key.as_str())
        })
        .max_by_key(|object| (object.created_at, object.id))
        .map(|object| object.id);
    if let Some(id) = existing_id {
        let object = graph.get_mut(&id)?;
        object.data["question"] = serde_json::json!(format!("{key}?"));
        object.data["answer"] = serde_json::json!(answer.trim());
        object.embedding = Some(question_embedding.to_vec());
        object.created_at = now;
        record_reference(&mut object.activation, now);
        return Some(id);
    }
    let mut object =
        MentalObject::new_observation(format!("curated static answer: {key}"), now, decay_d);
    object.kind = MentalObjectKind::Memory;
    object.memory_roles.push(MemoryRole::Semantic);
    object.embedding = Some(question_embedding.to_vec());
    object.data = serde_json::json!({
        "schema": SCHEMA,
        "question": format!("{key}?"),
        "answer": answer.trim(),
    });
    let id = object.id;
    graph.insert(object);
    Some(id)
}

/// Revokes every active version of one exact curated question. Memories are
/// marked Discarded (durable and recoverable), never physically deleted.
pub fn revoke_curated_answer(graph: &mut Graph, question: &str) -> Vec<aca_types::MentalObjectId> {
    let Some(key) = static_question_key(question) else {
        return Vec::new();
    };
    let ids: Vec<_> = graph
        .iter()
        .filter(|object| {
            object.status == ObjectStatus::Active
                && object.data.get("schema").and_then(|value| value.as_str()) == Some(SCHEMA)
                && object
                    .data
                    .get("question")
                    .and_then(|value| value.as_str())
                    .and_then(static_question_key)
                    .as_deref()
                    == Some(key.as_str())
        })
        .map(|object| object.id)
        .collect();
    for id in &ids {
        if let Some(object) = graph.get_mut(id) {
            object.status = ObjectStatus::Discarded;
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_explicit_static_answers_enter_the_exact_index() {
        let mut graph = Graph::new();
        assert!(
            record_curated_answer(
                &mut graph,
                "What is the capital of France?",
                "Paris.",
                &[1.0, 0.0],
                EpochMillis(0),
                0.5
            )
            .is_some()
        );
        for unsafe_question in [
            "What is the weather today?",
            "What is the price now?",
            "Who is the CEO?",
            "When is it open?",
            "What is two plus 2?",
        ] {
            assert!(
                record_curated_answer(
                    &mut graph,
                    unsafe_question,
                    "A stale answer.",
                    &[1.0],
                    EpochMillis(0),
                    0.5
                )
                .is_none()
            );
        }
        let index = curated_answer_index(&graph);
        assert_eq!(index.len(), 1);
        assert_eq!(
            index
                .get(&static_question_key("  WHAT is the capital of France ?").unwrap())
                .unwrap()
                .answer,
            "Paris."
        );
        assert!(static_question_key("What is the capital of Germany?").is_some());
        assert!(
            !index.contains_key(&static_question_key("What is the capital of Germany?").unwrap())
        );
        record_curated_answer(
            &mut graph,
            "What is the capital of France?",
            "Paris, France.",
            &[0.0, 1.0],
            EpochMillis(1),
            0.5,
        )
        .unwrap();
        let revised = curated_answer_index(&graph);
        assert_eq!(
            revised
                .get(&static_question_key("What is the capital of France?").unwrap())
                .unwrap()
                .answer,
            "Paris, France."
        );
        assert_eq!(
            revoke_curated_answer(&mut graph, "What is the capital of France?").len(),
            1
        );
        assert!(
            !curated_answer_index(&graph)
                .contains_key(&static_question_key("What is the capital of France?").unwrap())
        );
    }
}
