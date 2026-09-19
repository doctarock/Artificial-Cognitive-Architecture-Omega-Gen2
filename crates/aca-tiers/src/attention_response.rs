use std::str::FromStr;

use aca_types::MentalObjectId;
use aca_util::clamp01_or_default;
use serde::Deserialize;

use crate::attention::{AttentionDecision, AttentionError, AttentionOp};
use crate::response::extract_json_candidate;

/// The attention model's own fixed, trained-time 4-key envelope — distinct
/// from `response::TierEnvelope`'s generic `{"confidence","response"}`
/// shape, since this model was never trained on that contract. Every field
/// optional: malformed/missing input must degrade to `MalformedResponse`,
/// never panic.
#[derive(Debug, Deserialize)]
struct AttentionEnvelope {
    #[serde(default)]
    operation: Option<String>,
    #[serde(default)]
    target: Option<String>,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    reason_code: Option<String>,
}

/// Parses a raw model completion into an `AttentionDecision`. Reuses
/// `response::extract_json_candidate`'s `<think>`-stripping/brace-balanced
/// extraction (shape-agnostic, already hardened against context bleed —
/// see that function's doc comment) rather than reimplementing it.
///
/// Unlike `parse_tier_response`, a malformed completion here is a real
/// `Err`, not a silent fallback: the caller (`loop_actor`) already treats
/// any `Err` as "run the deterministic algorithm this tick instead," so
/// there is no meaningful raw-text fallback to return — a broken JSON
/// completion can't be treated as a valid operation.
pub(crate) fn parse_attention_response(raw_completion: &str) -> Result<AttentionDecision, AttentionError> {
    let candidate = extract_json_candidate(raw_completion);
    let envelope: AttentionEnvelope = serde_json::from_str(candidate)
        .map_err(|err| AttentionError::MalformedResponse { reason: err.to_string() })?;
    let operation = envelope
        .operation
        .as_deref()
        .and_then(AttentionOp::parse)
        .ok_or_else(|| AttentionError::MalformedResponse {
            reason: "missing or unrecognized operation".to_string(),
        })?;
    let target = envelope.target.as_deref().and_then(|s| MentalObjectId::from_str(s).ok());
    if matches!(operation, AttentionOp::Attend | AttentionOp::Switch | AttentionOp::Suppress) && target.is_none() {
        return Err(AttentionError::MalformedResponse {
            reason: "actionable attention operation needs a valid target UUID".to_string(),
        });
    }
    let confidence = envelope.confidence.map(clamp01_or_default).unwrap_or(0.5);
    let reason_code = envelope.reason_code.unwrap_or_default();
    Ok(AttentionDecision { operation, target, confidence, reason_code })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_well_formed_suppress_decision() {
        let raw = r#"{"operation":"SUPPRESS","target":"550e8400-e29b-41d4-a716-446655440000","confidence":0.86,"reason_code":"LOW_CONFIDENCE_HIGH_ACTIVATION"}"#;
        let decision = parse_attention_response(raw).unwrap();
        assert_eq!(decision.operation, AttentionOp::Suppress);
        assert_eq!(decision.target, MentalObjectId::from_str("550e8400-e29b-41d4-a716-446655440000").ok());
        assert!(decision.target.is_some());
        assert!((decision.confidence - 0.86).abs() < 1e-6);
        assert_eq!(decision.reason_code, "LOW_CONFIDENCE_HIGH_ACTIVATION");
    }

    #[test]
    fn missing_confidence_defaults_to_half() {
        let raw = r#"{"operation":"IGNORE","target":null,"reason_code":"NO_ACTIONABLE_ATTENTION_SIGNAL"}"#;
        let decision = parse_attention_response(raw).unwrap();
        assert_eq!(decision.confidence, 0.5);
        assert_eq!(decision.target, None);
    }

    #[test]
    fn out_of_range_confidence_clamps() {
        let raw = r#"{"operation":"ATTEND","target":"550e8400-e29b-41d4-a716-446655440000","confidence":5.0,"reason_code":"NO_CURRENT_FOCUS"}"#;
        let decision = parse_attention_response(raw).unwrap();
        assert_eq!(decision.confidence, 1.0);
    }

    #[test]
    fn unrecognized_operation_is_malformed() {
        let raw = r#"{"operation":"RECONSIDER","target":"x","confidence":0.5,"reason_code":"?"}"#;
        let result = parse_attention_response(raw);
        assert!(matches!(result, Err(AttentionError::MalformedResponse { .. })));
    }

    #[test]
    fn non_json_completion_is_malformed() {
        let result = parse_attention_response("this is not json at all");
        assert!(matches!(result, Err(AttentionError::MalformedResponse { .. })));
    }

    #[test]
    fn actionable_operation_with_malformed_target_falls_back_instead_of_becoming_a_trusted_no_op() {
        let raw = r#"{"operation":"SWITCH","target":"not-a-uuid","confidence":0.7,"reason_code":"HIGHER_PRIORITY_INTERRUPT"}"#;
        assert!(matches!(parse_attention_response(raw), Err(AttentionError::MalformedResponse { .. })));
    }

    #[test]
    fn maintain_can_abstain_from_a_specific_target() {
        let raw = r#"{"operation":"MAINTAIN","target":"not-a-uuid","confidence":0.7,"reason_code":"CURRENT_FOCUS_STILL_DOMINANT"}"#;
        assert_eq!(parse_attention_response(raw).unwrap().target, None);
    }

    #[test]
    fn extracts_the_envelope_past_a_thinking_models_reasoning_block() {
        let raw = "\n\n<think>\nweighing candidates\n</think>\n\n{\"operation\":\"MAINTAIN\",\"target\":\"belief_2\",\"confidence\":0.75,\"reason_code\":\"CURRENT_FOCUS_STILL_DOMINANT\"}";
        let decision = parse_attention_response(raw).unwrap();
        assert_eq!(decision.operation, AttentionOp::Maintain);
        assert_eq!(decision.reason_code, "CURRENT_FOCUS_STILL_DOMINANT");
    }
}
