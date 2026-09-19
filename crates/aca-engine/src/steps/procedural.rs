use aca_graph::{Graph, record_reference};
use aca_types::{MemoryRole, MentalObject, MentalObjectId, MentalObjectKind, ObjectStatus};
use aca_util::EpochMillis;

const SCHEMA: &str = "omega-routine-response/v1";
const VERIFIED_SEQUENCE_SCHEMA: &str = "omega-verified-operator-sequence/v2";
const COMPILATION_THRESHOLD: u64 = 3;
const MAX_STIMULUS_WORDS: usize = 6;
const MAX_RESPONSE_WORDS: usize = 24;

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledProcedure {
    pub condition: CompiledCondition,
    pub steps: Vec<CompiledOperatorStep>,
    pub expected_consequence: ExpectedConsequence,
    pub stimulus_embedding: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompiledCondition {
    ExactNormalizedRoutineObservation { stimulus: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompiledOperatorStep {
    ContinueReflecting,
    SpeakExact { text: String },
    AskExact { text: String },
    Ignore,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpectedConsequence {
    SpokeExact { text: String },
    AskedExact { text: String },
    Ignored,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompiledExecution<'a> {
    SpeakExact(&'a str),
    AskExact(&'a str),
    Ignore,
}

impl CompiledProcedure {
    /// The only optimized execution currently supported. Both typed steps
    /// and the expected consequence must agree; callers cannot replay just
    /// a convenient response string while ignoring the program contract.
    pub fn direct_spoken_response(&self, stimulus: &str) -> Option<&str> {
        match self.execution(stimulus)? {
            CompiledExecution::SpeakExact(text) => Some(text),
            CompiledExecution::AskExact(_) => None,
            CompiledExecution::Ignore => None,
        }
    }

    pub fn execution(&self, stimulus: &str) -> Option<CompiledExecution<'_>> {
        let CompiledCondition::ExactNormalizedRoutineObservation { stimulus: expected_stimulus } = &self.condition;
        if routine_key(stimulus).as_deref() != Some(expected_stimulus.as_str()) { return None; }
        match (self.steps.as_slice(), &self.expected_consequence) {
            ([CompiledOperatorStep::ContinueReflecting, CompiledOperatorStep::SpeakExact { text }],
                ExpectedConsequence::SpokeExact { text: expected }) if expected == text => Some(CompiledExecution::SpeakExact(text)),
            ([CompiledOperatorStep::ContinueReflecting, CompiledOperatorStep::AskExact { text }],
                ExpectedConsequence::AskedExact { text: expected }) if expected == text => Some(CompiledExecution::AskExact(text)),
            ([CompiledOperatorStep::ContinueReflecting, CompiledOperatorStep::Ignore],
                ExpectedConsequence::Ignored) => Some(CompiledExecution::Ignore),
            _ => None,
        }
    }
}

fn compiled_ignore_program(stimulus: &str, embedding: &[f32]) -> CompiledProcedure {
    CompiledProcedure {
        condition: CompiledCondition::ExactNormalizedRoutineObservation { stimulus: normalize(stimulus) },
        steps: vec![CompiledOperatorStep::ContinueReflecting, CompiledOperatorStep::Ignore],
        expected_consequence: ExpectedConsequence::Ignored,
        stimulus_embedding: embedding.to_vec(),
    }
}

fn compiled_program(stimulus: &str, response: &str, embedding: &[f32]) -> CompiledProcedure {
    CompiledProcedure {
        condition: CompiledCondition::ExactNormalizedRoutineObservation { stimulus: normalize(stimulus) },
        steps: vec![CompiledOperatorStep::ContinueReflecting, CompiledOperatorStep::SpeakExact { text: response.to_string() }],
        expected_consequence: ExpectedConsequence::SpokeExact { text: response.to_string() },
        stimulus_embedding: embedding.to_vec(),
    }
}

fn compiled_ask_program(stimulus: &str, question: &str, embedding: &[f32]) -> CompiledProcedure {
    CompiledProcedure {
        condition: CompiledCondition::ExactNormalizedRoutineObservation { stimulus: normalize(stimulus) },
        steps: vec![CompiledOperatorStep::ContinueReflecting, CompiledOperatorStep::AskExact { text: question.to_string() }],
        expected_consequence: ExpectedConsequence::AskedExact { text: question.to_string() },
        stimulus_embedding: embedding.to_vec(),
    }
}

fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|c: char| matches!(c, '.' | ',' | '!' | ';' | ':'))
        .to_lowercase()
}

pub fn routine_key(text: &str) -> Option<String> {
    is_routine_stimulus(text).then(|| normalize(text))
}

fn valid_response(text: &str) -> bool {
    !text.trim().is_empty() && text.split_whitespace().count() <= MAX_RESPONSE_WORDS
}

/// An intentionally tiny innate social reflex. It handles only an isolated
/// greeting, never a question, command, or longer conversational turn.
pub fn innate_social_response(text: &str) -> Option<&'static str> {
    matches!(
        normalize(text).as_str(),
        "hi" | "hello" | "hey" | "hi omega" | "hello omega" | "hey omega"
    )
    .then_some("Hello.")
}

/// Build the actor's O(1) compiled-skill lookup once when durable memories
/// are loaded. The public graph-scan lookup remains available to pure tests
/// and tooling, but is not performed on every live familiar stimulus.
pub fn compiled_index(graph: &Graph) -> std::collections::HashMap<String, CompiledProcedure> {
    let mut index = std::collections::HashMap::new();
    let mut seen_verified = std::collections::HashSet::new();
    let mut ambiguous_keys = std::collections::HashSet::new();
    // Repeated speech is only a candidate observation, never evidence of a
    // successful outcome. Only the explicitly verified two-step sequence may
    // collapse into direct speech, including for pre-existing legacy records.
    for object in graph.iter().filter(|object| object.status == ObjectStatus::Active) {
        let Some(data) = object.data.as_object() else { continue };
        if data.get("schema").and_then(|value| value.as_str()) != Some(VERIFIED_SEQUENCE_SCHEMA) {
            continue;
        }
        let Some(stimulus) = data.get("stimulus").and_then(|value| value.as_str()) else { continue };
        let Some(key) = routine_key(stimulus) else { continue };
        if !seen_verified.insert(key.clone()) {
            ambiguous_keys.insert(key.clone());
            index.remove(&key);
            continue;
        }
        if data.get("verified_successes").and_then(|value| value.as_u64()).unwrap_or(0) < COMPILATION_THRESHOLD { 
            continue;
        }
        let Some(embedding) = object.embedding.as_ref().filter(|embedding| !embedding.is_empty() && embedding.iter().all(|value| value.is_finite())) else { continue };
        let Some(kind) = verified_kind(data) else { continue };
        index.insert(key, match kind {
            VerifiedKind::Speak(response) => compiled_program(stimulus, &response, embedding),
            VerifiedKind::Ask(question) => compiled_ask_program(stimulus, &question, embedding),
            VerifiedKind::Ignore => compiled_ignore_program(stimulus, embedding),
        });
    }
    for key in ambiguous_keys { index.remove(&key); }
    index
}

fn verified_speak_steps(value: Option<&serde_json::Value>) -> bool {
    value.and_then(|value| value.as_array()).is_some_and(|steps| {
        steps.len() == 2
            && steps[0] == "ContinueReflecting"
            && steps[1] == "Speak"
    })
}

fn verified_ask_steps(value: Option<&serde_json::Value>) -> bool {
    value.and_then(|value| value.as_array()).is_some_and(|steps| {
        steps.len() == 2
            && steps[0] == "ContinueReflecting"
            && steps[1] == "Ask"
    })
}

fn verified_conditions(data: &serde_json::Map<String, serde_json::Value>) -> bool {
    data.get("conditions") == Some(&serde_json::json!({
            "source_kind": "observation",
            "stimulus_policy": "routine_non_command",
            "match": "exact_normalized",
        }))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum VerifiedKind { Speak(String), Ask(String), Ignore }

fn verified_kind(data: &serde_json::Map<String, serde_json::Value>) -> Option<VerifiedKind> {
    if !verified_conditions(data) { return None; }
    if verified_speak_steps(data.get("steps")) {
        let response = data.get("expected_spoken_response")?.as_str()?.to_string();
        if !valid_response(&response) || data.get("expected_consequence") != Some(&serde_json::json!({
            "kind": "spoke",
            "text": &response,
        })) { return None; }
        return Some(VerifiedKind::Speak(response));
    }
    if verified_ask_steps(data.get("steps")) {
        let question = data.get("expected_asked_question")?.as_str()?.to_string();
        if !valid_response(&question) || data.get("expected_consequence") != Some(&serde_json::json!({
            "kind": "asked",
            "text": &question,
        })) { return None; }
        return Some(VerifiedKind::Ask(question));
    }
    let steps = data.get("steps")?.as_array()?;
    (steps.len() == 2 && steps[0] == "ContinueReflecting" && steps[1] == "Ignore"
        && data.get("expected_consequence") == Some(&serde_json::json!({"kind": "ignored"})))
        .then_some(VerifiedKind::Ignore)
}

fn find_verified(graph: &Graph, stimulus: &str) -> Option<(MentalObjectId, VerifiedKind, u64, Vec<String>)> {
    let normalized = normalize(stimulus);
    let mut found = None;
    for object in graph.iter().filter(|object| object.status == ObjectStatus::Active) {
        let Some(data) = object.data.as_object() else { continue };
        if data.get("schema").and_then(|value| value.as_str()) != Some(VERIFIED_SEQUENCE_SCHEMA)
            || data.get("stimulus").and_then(|value| value.as_str()) != Some(normalized.as_str())
        { continue; }
        if found.is_some() { return None; }
        let kind = verified_kind(data)?;
        let successes = data.get("verified_successes")?.as_u64()?;
        let credits = data.get("credited_observation_ids")?.as_array()?
            .iter().filter_map(|value| value.as_str().map(str::to_string)).collect();
        found = Some((object.id, kind, successes, credits));
    }
    found
}

/// Credit a *distinct* externally verified successful outcome for the
/// observed `ContinueReflecting -> Speak` sequence. The caller must tie the
/// observation id to an actual spoken turn; this function never treats a
/// model's own output or repeated speech as success. Questions, commands,
/// time-sensitive content, unbounded responses, and non-finite embeddings
/// are ineligible for a direct-action macro.
pub fn record_verified_sequence_success(
    graph: &mut Graph,
    observation_id: MentalObjectId,
    stimulus: &str,
    response: &str,
    stimulus_embedding: &[f32],
    now: EpochMillis,
    decay_d: f32,
) -> Option<MentalObjectId> {
    if !is_routine_stimulus(stimulus) || !valid_response(response)
        || stimulus_embedding.is_empty()
        || stimulus_embedding.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    let normalized = normalize(stimulus);
    if let Some((id, kind, successes, mut credited_ids)) = find_verified(graph, stimulus) {
        let VerifiedKind::Speak(prior_response) = kind else { return None; };
        let credit = observation_id.to_string();
        if credited_ids.iter().any(|prior| prior == &credit) {
            return Some(id); // one observed turn cannot count twice
        }
        let next_successes = if prior_response == response { successes.saturating_add(1) } else { 1 };
        if prior_response != response { credited_ids.clear(); }
        credited_ids.push(credit);
        // Keep only enough provenance to prevent duplicate credit during
        // the maturity window; the count itself is durable.
        if credited_ids.len() > 16 { credited_ids.remove(0); }
        let object = graph.get_mut(&id)?;
        object.data = serde_json::json!({
            "schema": VERIFIED_SEQUENCE_SCHEMA,
            "stimulus": normalized,
            "steps": ["ContinueReflecting", "Speak"],
            "conditions": {"source_kind": "observation", "stimulus_policy": "routine_non_command", "match": "exact_normalized"},
            "expected_spoken_response": response,
            "expected_consequence": {"kind": "spoke", "text": response},
            "verified_successes": next_successes,
            "credited_observation_ids": credited_ids,
        });
        object.embedding = Some(stimulus_embedding.to_vec());
        record_reference(&mut object.activation, now);
        return Some(id);
    }

    let mut macro_object = MentalObject::new_observation(format!("verified sequence: {normalized}"), now, decay_d);
    macro_object.kind = MentalObjectKind::Memory;
    macro_object.memory_roles.push(MemoryRole::Semantic);
    macro_object.embedding = Some(stimulus_embedding.to_vec());
    macro_object.data = serde_json::json!({
        "schema": VERIFIED_SEQUENCE_SCHEMA,
        "stimulus": normalized,
        "steps": ["ContinueReflecting", "Speak"],
        "conditions": {"source_kind": "observation", "stimulus_policy": "routine_non_command", "match": "exact_normalized"},
        "expected_spoken_response": response,
        "expected_consequence": {"kind": "spoke", "text": response},
        "verified_successes": 1,
        "credited_observation_ids": [observation_id.to_string()],
    });
    let id = macro_object.id;
    graph.insert(macro_object);
    Some(id)
}

/// Credit a distinct externally verified successful outcome for an observed
/// `ContinueReflecting -> Ask` sequence. The question is preserved exactly;
/// compilation never invents or reformulates it and cannot invoke a tool.
pub fn record_verified_ask_success(
    graph: &mut Graph,
    observation_id: MentalObjectId,
    stimulus: &str,
    question: &str,
    stimulus_embedding: &[f32],
    now: EpochMillis,
    decay_d: f32,
) -> Option<MentalObjectId> {
    if !is_routine_stimulus(stimulus) || !valid_response(question)
        || stimulus_embedding.is_empty()
        || stimulus_embedding.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    let normalized = normalize(stimulus);
    if let Some((id, kind, successes, mut credited_ids)) = find_verified(graph, stimulus) {
        let VerifiedKind::Ask(prior_question) = kind else { return None; };
        let credit = observation_id.to_string();
        if credited_ids.iter().any(|prior| prior == &credit) { return Some(id); }
        let next_successes = if prior_question == question { successes.saturating_add(1) } else { 1 };
        if prior_question != question { credited_ids.clear(); }
        credited_ids.push(credit);
        if credited_ids.len() > 16 { credited_ids.remove(0); }
        let object = graph.get_mut(&id)?;
        object.data = serde_json::json!({
            "schema": VERIFIED_SEQUENCE_SCHEMA,
            "stimulus": normalized,
            "steps": ["ContinueReflecting", "Ask"],
            "conditions": {"source_kind": "observation", "stimulus_policy": "routine_non_command", "match": "exact_normalized"},
            "expected_asked_question": question,
            "expected_consequence": {"kind": "asked", "text": question},
            "verified_successes": next_successes,
            "credited_observation_ids": credited_ids,
        });
        object.embedding = Some(stimulus_embedding.to_vec());
        record_reference(&mut object.activation, now);
        return Some(id);
    }

    let mut macro_object = MentalObject::new_observation(format!("verified sequence: {normalized}"), now, decay_d);
    macro_object.kind = MentalObjectKind::Memory;
    macro_object.memory_roles.push(MemoryRole::Semantic);
    macro_object.embedding = Some(stimulus_embedding.to_vec());
    macro_object.data = serde_json::json!({
        "schema": VERIFIED_SEQUENCE_SCHEMA,
        "stimulus": normalized,
        "steps": ["ContinueReflecting", "Ask"],
        "conditions": {"source_kind": "observation", "stimulus_policy": "routine_non_command", "match": "exact_normalized"},
        "expected_asked_question": question,
        "expected_consequence": {"kind": "asked", "text": question},
        "verified_successes": 1,
        "credited_observation_ids": [observation_id.to_string()],
    });
    let id = macro_object.id;
    graph.insert(macro_object);
    Some(id)
}

/// Credit a distinct successful outcome for an actually observed
/// `ContinueReflecting -> Ignore` sequence. Ignoring is the only silent
/// compiled consequence admitted: it is internal, reversible by a later
/// observation, and has no tool/network/world side effect.
pub fn record_verified_ignore_success(
    graph: &mut Graph,
    observation_id: MentalObjectId,
    stimulus: &str,
    stimulus_embedding: &[f32],
    now: EpochMillis,
    decay_d: f32,
) -> Option<MentalObjectId> {
    if !is_routine_stimulus(stimulus) || stimulus_embedding.is_empty()
        || stimulus_embedding.iter().any(|value| !value.is_finite()) { return None; }
    let normalized = normalize(stimulus);
    if let Some((id, kind, successes, mut credited_ids)) = find_verified(graph, stimulus) {
        if kind != VerifiedKind::Ignore { return None; }
        let credit = observation_id.to_string();
        if credited_ids.iter().any(|prior| prior == &credit) { return Some(id); }
        credited_ids.push(credit);
        if credited_ids.len() > 16 { credited_ids.remove(0); }
        let object = graph.get_mut(&id)?;
        object.data["verified_successes"] = serde_json::json!(successes.saturating_add(1));
        object.data["credited_observation_ids"] = serde_json::json!(credited_ids);
        object.embedding = Some(stimulus_embedding.to_vec());
        record_reference(&mut object.activation, now);
        return Some(id);
    }
    let mut macro_object = MentalObject::new_observation(format!("verified ignore sequence: {normalized}"), now, decay_d);
    macro_object.kind = MentalObjectKind::Memory;
    macro_object.memory_roles.push(MemoryRole::Semantic);
    macro_object.embedding = Some(stimulus_embedding.to_vec());
    macro_object.data = serde_json::json!({
        "schema": VERIFIED_SEQUENCE_SCHEMA,
        "stimulus": normalized,
        "steps": ["ContinueReflecting", "Ignore"],
        "conditions": {"source_kind": "observation", "stimulus_policy": "routine_non_command", "match": "exact_normalized"},
        "expected_consequence": {"kind": "ignored"},
        "verified_successes": 1,
        "credited_observation_ids": [observation_id.to_string()],
    });
    let id = macro_object.id;
    graph.insert(macro_object);
    Some(id)
}

/// Negative outcome feedback immediately demotes a verified macro; its
/// memory remains in the graph for analysis, but it ceases to bypass models.
pub fn record_verified_sequence_failure(graph: &mut Graph, stimulus: &str, now: EpochMillis) -> Option<MentalObjectId> {
    if let Some((id, _, _, _)) = find_verified(graph, stimulus) {
        let object = graph.get_mut(&id)?;
        object.data["verified_successes"] = serde_json::json!(0);
        object.data["credited_observation_ids"] = serde_json::json!([]);
        record_reference(&mut object.activation, now);
        return Some(id);
    }
    // A negative observed outcome for a legacy compiled routine must also
    // revoke its direct path, not merely fail to find a verified record.
    let legacy_id = find(graph, stimulus)?.0;
    graph.discard(&legacy_id, now);
    Some(legacy_id)
}

/// Conservative admission for automatic stimulus-response compilation.
/// Questions, digits, and interrogative/request prefixes stay on the normal
/// semantic path so time-sensitive or action-bearing answers cannot become a
/// stale reflex accidentally.
pub fn is_routine_stimulus(text: &str) -> bool {
    let normalized = normalize(text);
    let first = normalized.split_whitespace().next().unwrap_or_default();
    let time_sensitive = normalized.split_whitespace().any(|word| {
        matches!(
            word,
            "today" | "tomorrow" | "yesterday" | "now" | "weather" | "current" | "latest"
        )
    });
    !normalized.is_empty()
        && normalized.split_whitespace().count() <= MAX_STIMULUS_WORDS
        && !text.contains('?')
        && !text.chars().any(|c| c.is_ascii_digit())
        && !time_sensitive
        && !matches!(
            first,
            "what"
                | "when"
                | "where"
                | "who"
                | "why"
                | "how"
                | "can"
                | "could"
                | "would"
                | "will"
                | "please"
                | "is"
                | "are"
                | "do"
                | "does"
                | "did"
                | "have"
                | "has"
                | "open"
                | "close"
                | "turn"
                | "set"
                | "send"
                | "delete"
                | "remove"
                | "stop"
                | "start"
                | "call"
                | "remind"
                | "schedule"
                | "book"
                | "buy"
                | "pay"
                | "search"
                | "check"
                | "update"
                | "change"
                | "run"
        )
}

fn find(graph: &Graph, stimulus: &str) -> Option<(MentalObjectId, String, u64)> {
    let normalized = normalize(stimulus);
    graph.iter().find_map(|object| {
        if object.status != ObjectStatus::Active {
            return None;
        }
        let data = object.data.as_object()?;
        (data.get("schema")?.as_str()? == SCHEMA && data.get("stimulus")?.as_str()? == normalized)
            .then(|| {
                (
                    object.id,
                    data.get("response")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    data.get("consecutive_hits")
                        .and_then(|value| value.as_u64())
                        .unwrap_or(0),
                )
            })
    })
}

pub fn compiled_response(graph: &Graph, stimulus: &str) -> Option<String> {
    compiled_procedure(graph, stimulus)?.direct_spoken_response(stimulus).map(str::to_string)
}

pub fn compiled_procedure(graph: &Graph, stimulus: &str) -> Option<CompiledProcedure> {
    if !is_routine_stimulus(stimulus) {
        return None;
    }
    let (id, kind, successes, _) = find_verified(graph, stimulus)?;
    let embedding = graph.get(&id).and_then(|object| object.embedding.clone()).unwrap_or_default();
    if successes < COMPILATION_THRESHOLD || embedding.is_empty() || !embedding.iter().all(|value| value.is_finite()) {
        return None;
    }
    match kind {
        VerifiedKind::Speak(response) => valid_response(&response).then(|| compiled_program(stimulus, &response, &embedding)),
        VerifiedKind::Ask(question) => valid_response(&question).then(|| compiled_ask_program(stimulus, &question, &embedding)),
        VerifiedKind::Ignore => Some(compiled_ignore_program(stimulus, &embedding)),
    }
}

/// Records a repeated stimulus-to-spoken-response candidate. Speech alone is
/// not independent evidence that the answer was appropriate, so this record
/// never becomes an executable shortcut without separately verified outcomes.
pub fn record_response(
    graph: &mut Graph,
    stimulus: &str,
    response: &str,
    now: EpochMillis,
    decay_d: f32,
) -> Option<MentalObjectId> {
    record_response_with_embedding(graph, stimulus, response, None, now, decay_d)
}

pub fn record_response_with_embedding(
    graph: &mut Graph,
    stimulus: &str,
    response: &str,
    stimulus_embedding: Option<&[f32]>,
    now: EpochMillis,
    decay_d: f32,
) -> Option<MentalObjectId> {
    if !is_routine_stimulus(stimulus)
        || !valid_response(response)
        || stimulus_embedding.is_some_and(|embedding| embedding.iter().any(|value| !value.is_finite()))
    {
        return None;
    }
    let normalized = normalize(stimulus);
    if let Some((id, prior_response, hits)) = find(graph, stimulus) {
        let object = graph.get_mut(&id)?;
        let next_hits = if prior_response == response {
            hits + 1
        } else {
            1
        };
        object.data = serde_json::json!({
            "schema": SCHEMA,
            "stimulus": normalized,
            "response": response,
            "consecutive_hits": next_hits,
        });
        if let Some(embedding) = stimulus_embedding {
            object.embedding = Some(embedding.to_vec());
        }
        record_reference(&mut object.activation, now);
        return Some(id);
    }

    let mut skill =
        MentalObject::new_observation(format!("routine response: {normalized}"), now, decay_d);
    skill.kind = MentalObjectKind::Memory;
    skill.embedding = Some(stimulus_embedding.unwrap_or_default().to_vec());
    skill.memory_roles.push(MemoryRole::Semantic);
    skill.data = serde_json::json!({
        "schema": SCHEMA,
        "stimulus": normalized,
        "response": response,
        "consecutive_hits": 1,
    });
    let id = skill.id;
    graph.insert(skill);
    Some(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_speech_remains_a_candidate_without_verified_success() {
        let mut graph = Graph::new();
        for i in 0..2 {
            record_response(
                &mut graph,
                "Hello Omega!",
                "Hello Derek.",
                EpochMillis(i),
                0.5,
            );
        }
        assert_eq!(compiled_response(&graph, "hello omega"), None);
        record_response(
            &mut graph,
            "hello   OMEGA.",
            "Hello Derek.",
            EpochMillis(2),
            0.5,
        );
        assert_eq!(find(&graph, "Hello Omega!").unwrap().2, 3);
        assert_eq!(compiled_response(&graph, "Hello Omega!"), None);
    }

    #[test]
    fn a_changed_response_resets_candidate_streak() {
        let mut graph = Graph::new();
        for i in 0..3 {
            record_response(&mut graph, "hello", "Hello.", EpochMillis(i), 0.5);
        }
        assert_eq!(find(&graph, "hello").unwrap().2, 3);
        record_response(&mut graph, "hello", "Hi.", EpochMillis(4), 0.5);
        assert_eq!(find(&graph, "hello").unwrap().2, 1);
        assert_eq!(compiled_response(&graph, "hello"), None);
    }

    #[test]
    fn questions_and_time_sensitive_digits_never_compile() {
        let mut graph = Graph::new();
        for i in 0..5 {
            assert_eq!(
                record_response(&mut graph, "what time is it?", "Noon", EpochMillis(i), 0.5),
                None
            );
            assert_eq!(
                record_response(&mut graph, "remind me at 5", "Okay", EpochMillis(i), 0.5),
                None
            );
            assert_eq!(
                record_response(&mut graph, "open the door", "Okay", EpochMillis(i), 0.5),
                None
            );
            assert_eq!(
                record_response(&mut graph, "weather today", "Sunny", EpochMillis(i), 0.5),
                None
            );
        }
        assert!(graph.is_empty());
    }

    #[test]
    fn compiled_index_excludes_legacy_speech_only_routines() {
        let mut graph = Graph::new();
        for i in 0..3 {
            record_response_with_embedding(
                &mut graph,
                "hello",
                "Hi.",
                Some(&[1.0, 0.0]),
                EpochMillis(i),
                0.5,
            );
        }
        assert_eq!(find(&graph, "hello").unwrap().2, 3);
        assert!(compiled_index(&graph).is_empty());
        let skill_id = find(&graph, "hello").unwrap().0;
        graph.discard(&skill_id, EpochMillis(10));
        assert!(compiled_index(&graph).is_empty());
        assert!(compiled_procedure(&graph, "hello").is_none());
    }

    #[test]
    fn loaded_routine_index_rejects_corrupt_response_or_embedding() {
        let mut graph = Graph::new();
        for i in 0..3 {
            record_response_with_embedding(&mut graph, "hello", "Hi.", Some(&[1.0, 0.0]), EpochMillis(i), 0.5);
        }
        let id = find(&graph, "hello").unwrap().0;
        graph.get_mut(&id).unwrap().data["response"] = serde_json::json!("word ".repeat(25));
        assert!(compiled_index(&graph).is_empty());
        assert!(compiled_procedure(&graph, "hello").is_none());

        graph.get_mut(&id).unwrap().data["response"] = serde_json::json!("Hi.");
        graph.get_mut(&id).unwrap().embedding = Some(vec![f32::NAN]);
        assert!(compiled_index(&graph).is_empty());
        assert!(compiled_procedure(&graph, "hello").is_none());
        assert!(record_response_with_embedding(&mut graph, "hello", "Hi.", Some(&[f32::NAN]), EpochMillis(4), 0.5).is_none());
    }

    #[test]
    fn verified_two_step_macro_requires_three_distinct_successful_outcomes() {
        let mut graph = Graph::new();
        let first = MentalObjectId::new();
        let second = MentalObjectId::new();
        let third = MentalObjectId::new();
        let record = |graph: &mut Graph, observation_id, at| {
            record_verified_sequence_success(graph, observation_id, "hello omega", "Hello Derek.", &[1.0, 0.0], EpochMillis(at), 0.5).unwrap()
        };
        let macro_id = record(&mut graph, first, 1);
        assert!(compiled_index(&graph).is_empty());
        assert_eq!(record(&mut graph, first, 2), macro_id);
        assert_eq!(graph.get(&macro_id).unwrap().data["verified_successes"], 1);
        record(&mut graph, second, 3);
        assert!(compiled_index(&graph).is_empty());
        record(&mut graph, third, 4);
        let program = compiled_procedure(&graph, "Hello Omega!").unwrap();
        assert_eq!(program.direct_spoken_response("Hello Omega!"), Some("Hello Derek."));
        assert_eq!(program.direct_spoken_response("hey omega"), None,
            "typed program must recheck its exact normalized input condition");
        assert_eq!(program.steps, vec![CompiledOperatorStep::ContinueReflecting,
            CompiledOperatorStep::SpeakExact { text: "Hello Derek.".to_string() }]);
        assert_eq!(graph.get(&macro_id).unwrap().data["steps"], serde_json::json!(["ContinueReflecting", "Speak"]));
        assert_eq!(graph.get(&macro_id).unwrap().data["expected_spoken_response"], "Hello Derek.");
        assert_eq!(graph.get(&macro_id).unwrap().data["conditions"]["match"], "exact_normalized");
        assert_eq!(graph.get(&macro_id).unwrap().data["expected_consequence"]["text"], "Hello Derek.");

        record_verified_sequence_failure(&mut graph, "hello omega", EpochMillis(5)).unwrap();
        assert!(compiled_index(&graph).is_empty());
        assert!(compiled_procedure(&graph, "hello omega").is_none());
    }

    #[test]
    fn verified_failure_prevents_legacy_speech_count_fallback() {
        let mut graph = Graph::new();
        for i in 0..3 {
            record_response_with_embedding(&mut graph, "hello omega", "Hello Derek.", Some(&[1.0, 0.0]), EpochMillis(i), 0.5);
        }
        assert!(compiled_index(&graph).is_empty());
        record_verified_sequence_success(&mut graph, MentalObjectId::new(), "hello omega", "Hello Derek.", &[1.0, 0.0], EpochMillis(10), 0.5);
        assert!(compiled_index(&graph).is_empty(), "one verified outcome must not inherit three unverified speech counts");
        assert!(compiled_procedure(&graph, "hello omega").is_none());
    }

    #[test]
    fn duplicate_verified_records_fail_closed_on_load_and_lookup() {
        let mut graph = Graph::new();
        for i in 0..3 {
            record_verified_sequence_success(&mut graph, MentalObjectId::new(), "hello omega", "Hello Derek.", &[1.0], EpochMillis(i), 0.5);
        }
        let original = graph.iter().next().unwrap().clone();
        let mut duplicate = original.clone();
        duplicate.id = MentalObjectId::new();
        duplicate.data["verified_successes"] = serde_json::json!(0);
        graph.insert(duplicate);
        assert!(compiled_index(&graph).is_empty());
        assert!(compiled_procedure(&graph, "hello omega").is_none());
    }

    #[test]
    fn corrupt_condition_or_consequence_never_executes_on_reload() {
        let mut graph = Graph::new();
        for i in 0..3 {
            record_verified_sequence_success(&mut graph, MentalObjectId::new(), "hello omega", "Hello Derek.", &[1.0], EpochMillis(i), 0.5);
        }
        let id = graph.iter().next().unwrap().id;
        graph.get_mut(&id).unwrap().data["expected_consequence"]["text"] = serde_json::json!("different output");
        assert!(compiled_index(&graph).is_empty());
        assert!(compiled_procedure(&graph, "hello omega").is_none());
        graph.get_mut(&id).unwrap().data["expected_consequence"]["text"] = serde_json::json!("Hello Derek.");
        graph.get_mut(&id).unwrap().data["conditions"]["match"] = serde_json::json!("approximate");
        assert!(compiled_index(&graph).is_empty());
        assert!(compiled_procedure(&graph, "hello omega").is_none());
    }

    #[test]
    fn old_uncontracted_v1_sequence_is_retained_but_not_executable() {
        let mut graph = Graph::new();
        for i in 0..3 {
            record_verified_sequence_success(&mut graph, MentalObjectId::new(), "hello omega", "Hello Derek.", &[1.0], EpochMillis(i), 0.5);
        }
        let id = graph.iter().next().unwrap().id;
        graph.get_mut(&id).unwrap().data["schema"] = serde_json::json!("omega-verified-operator-sequence/v1");
        assert!(compiled_index(&graph).is_empty());
        assert!(compiled_procedure(&graph, "hello omega").is_none());
        assert!(graph.get(&id).is_some(), "old record remains available for audit");
    }

    #[test]
    fn verified_reflect_then_ignore_is_a_distinct_safe_program() {
        let mut graph = Graph::new();
        let ids = [MentalObjectId::new(), MentalObjectId::new(), MentalObjectId::new()];
        for (i, id) in ids.into_iter().enumerate() {
            record_verified_ignore_success(&mut graph, id, "background hum", &[0.2, 0.8], EpochMillis(i as i64), 0.5).unwrap();
        }
        let program = compiled_procedure(&graph, "background hum").unwrap();
        assert_eq!(program.execution("background hum"), Some(CompiledExecution::Ignore));
        assert_eq!(program.direct_spoken_response("background hum"), None);
        assert_eq!(program.steps, vec![CompiledOperatorStep::ContinueReflecting, CompiledOperatorStep::Ignore]);
        assert_eq!(compiled_response(&graph, "background hum"), None);
        assert!(record_verified_sequence_success(&mut graph, MentalObjectId::new(), "background hum", "Okay.", &[0.2, 0.8], EpochMillis(4), 0.5).is_none(),
            "one stimulus cannot silently change its verified consequence kind");
        record_verified_sequence_failure(&mut graph, "background hum", EpochMillis(5)).unwrap();
        assert!(compiled_procedure(&graph, "background hum").is_none());
    }

    #[test]
    fn verified_reflect_then_ask_is_typed_and_requires_distinct_outcomes() {
        let mut graph = Graph::new();
        let ids = [MentalObjectId::new(), MentalObjectId::new(), MentalObjectId::new()];
        let first = record_verified_ask_success(
            &mut graph, ids[0], "status update", "Which project do you mean?",
            &[0.4, 0.6], EpochMillis(0), 0.5,
        ).unwrap();
        assert!(compiled_procedure(&graph, "status update").is_none());
        assert_eq!(record_verified_ask_success(
            &mut graph, ids[0], "status update", "Which project do you mean?",
            &[0.4, 0.6], EpochMillis(1), 0.5,
        ), Some(first), "one observation cannot be credited twice");
        for (i, id) in ids.into_iter().enumerate().skip(1) {
            record_verified_ask_success(
                &mut graph, id, "status update", "Which project do you mean?",
                &[0.4, 0.6], EpochMillis(i as i64 + 1), 0.5,
            ).unwrap();
        }
        let program = compiled_procedure(&graph, "status update").unwrap();
        assert_eq!(program.execution("status update"), Some(CompiledExecution::AskExact("Which project do you mean?")));
        assert_eq!(program.steps, vec![CompiledOperatorStep::ContinueReflecting,
            CompiledOperatorStep::AskExact { text: "Which project do you mean?".to_string() }]);
        assert_eq!(program.expected_consequence,
            ExpectedConsequence::AskedExact { text: "Which project do you mean?".to_string() });
        assert!(record_verified_ignore_success(
            &mut graph, MentalObjectId::new(), "status update", &[0.4, 0.6], EpochMillis(5), 0.5,
        ).is_none(), "a stimulus cannot silently change consequence kind");

        graph.get_mut(&first).unwrap().data["expected_consequence"]["text"] =
            serde_json::json!("a different question");
        assert!(compiled_index(&graph).is_empty());
        assert!(compiled_procedure(&graph, "status update").is_none(),
            "a mismatched Ask consequence must fail closed on reload");
    }

    #[test]
    fn negative_outcome_revokes_a_legacy_shortcut_without_deleting_memory() {
        let mut graph = Graph::new();
        for i in 0..3 {
            record_response_with_embedding(&mut graph, "hello", "Hi.", Some(&[1.0]), EpochMillis(i), 0.5);
        }
        let legacy_id = find(&graph, "hello").unwrap().0;
        assert!(compiled_index(&graph).is_empty());
        assert_eq!(record_verified_sequence_failure(&mut graph, "hello", EpochMillis(10)), Some(legacy_id));
        assert_eq!(graph.get(&legacy_id).unwrap().status, ObjectStatus::Discarded);
        assert!(compiled_index(&graph).is_empty());
    }

    #[test]
    fn verified_macro_rejects_unsafe_stimuli_and_invalid_embeddings() {
        let mut graph = Graph::new();
        let id = MentalObjectId::new();
        for stimulus in ["what time is it?", "open the door", "weather today"] {
            assert!(record_verified_sequence_success(&mut graph, id, stimulus, "Okay.", &[1.0], EpochMillis(1), 0.5).is_none());
        }
        assert!(record_verified_sequence_success(&mut graph, id, "hello", "Hi.", &[f32::NAN], EpochMillis(1), 0.5).is_none());
        assert!(graph.is_empty());
    }

    #[test]
    fn innate_greeting_reflex_is_narrow_and_not_a_general_text_shortcut() {
        assert_eq!(innate_social_response("Hi Omega!"), Some("Hello."));
        assert_eq!(innate_social_response("hello"), Some("Hello."));
        assert_eq!(innate_social_response("hello, can you help?"), None);
        assert_eq!(innate_social_response("open the door"), None);
    }
}
