use aca_types::Tier;
use aca_util::clamp01_or_default;
use serde::Deserialize;

use crate::client::TierResponse;

/// The envelope a tier is prompted to respond in: a `confidence` alongside
/// its primary content. Every field is optional — a small local model
/// routinely fails to format this perfectly, and per the confidence
/// contract, malformed/missing input must never fail the turn.
#[derive(Debug, Deserialize)]
struct TierEnvelope {
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    response: Option<String>,
}

/// Parses a raw model completion into a `TierResponse`: extracts and clamps
/// `confidence` (default `0.5` if missing, malformed, or out of range), and
/// unwraps the `response` field if the completion parsed as the expected
/// envelope — falling back to the full raw completion text otherwise, so a
/// non-JSON-formatting model still produces something usable rather than an
/// error.
pub fn parse_tier_response(raw_completion: &str, tier: Tier) -> TierResponse {
    match serde_json::from_str::<TierEnvelope>(extract_json_candidate(raw_completion)) {
        Ok(envelope) => TierResponse {
            raw_text: envelope.response.unwrap_or_else(|| raw_completion.to_string()),
            confidence: envelope
                .confidence
                .map(clamp01_or_default)
                .unwrap_or(0.5),
            tier,
        },
        Err(_) => TierResponse {
            raw_text: raw_completion.to_string(),
            confidence: 0.5,
            tier,
        },
    }
}

/// Some models (Qwen3/DeepSeek-R1-style "thinking" models — confirmed live
/// with `richardyoung/qwen3.6-27b-abliterated`) prefix their actual
/// completion with a `<think>...</think>` reasoning block, and/or trail it
/// with stray control tokens (`<|endoftext|>...`) — neither of which is
/// valid JSON, so a strict whole-string parse treats the *entire* raw
/// completion (reasoning trace included) as unparseable and falls back to
/// speaking it verbatim. This narrows the parse attempt to the first
/// balanced `{...}` object found after dropping any `<think>...</think>`
/// block — exactly where the strict-JSON envelope the prompt asked for
/// actually lives — while leaving already-clean completions (no object
/// found, or the whole string already valid) exactly as they were.
pub(crate) fn extract_json_candidate(raw_completion: &str) -> &str {
    let after_think = match raw_completion.rfind("</think>") {
        Some(end) => &raw_completion[end + "</think>".len()..],
        None => raw_completion,
    };
    match find_balanced_json_object(after_think) {
        Some((start, end)) => &after_think[start..=end],
        None => raw_completion,
    }
}

/// Finds the byte range (inclusive) of the *first* balanced `{...}` object
/// in `text`, scanning with JSON string/escape awareness so a brace
/// appearing inside a quoted string value is never counted as structural.
///
/// Replaces a naive "first `{` to last `}`" scan, which broke on a raw
/// completion confirmed live from `richardyoung/qwen3.6-27b-abliterated`:
/// the model's completion, after its `<think>` block and the real JSON
/// answer, went on to echo an entire *second* prompt verbatim (context
/// bleed from the model server) — and that echoed prompt's own instruction
/// text contains a literal `{"confidence": ..., "response": ...}` template.
/// `rfind('}')` grabbed that far-later brace instead of the real answer's
/// own closing one, so the "extracted" span covered the real JSON, the
/// closing code fence, and the entire echoed second prompt - not valid JSON
/// as a whole, so parsing failed and the fallback stored the *entire* raw
/// completion (reasoning trace, prompt echo, and all) as the Reflection's
/// text. Depth-counting from the first `{` and stopping the instant it
/// returns to zero can't be fooled by unrelated braces arbitrarily far
/// later in the string, no matter why they're there.
fn find_balanced_json_object(text: &str) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let start = text.find('{')?;
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((start, i));
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_well_formed_envelope() {
        let raw = r#"{"confidence": 0.8, "response": "hello"}"#;
        let parsed = parse_tier_response(raw, Tier::T3);
        assert_eq!(parsed.raw_text, "hello");
        assert!((parsed.confidence - 0.8).abs() < 1e-6);
        assert_eq!(parsed.tier, Tier::T3);
    }

    #[test]
    fn defaults_confidence_when_field_missing() {
        let raw = r#"{"response": "hello"}"#;
        let parsed = parse_tier_response(raw, Tier::T1);
        assert_eq!(parsed.confidence, 0.5);
        assert_eq!(parsed.raw_text, "hello");
    }

    #[test]
    fn clamps_out_of_range_confidence() {
        let raw = r#"{"confidence": 5.0, "response": "hi"}"#;
        let parsed = parse_tier_response(raw, Tier::T2);
        assert_eq!(parsed.confidence, 1.0);
    }

    #[test]
    fn falls_back_to_raw_text_on_non_json_completion() {
        let raw = "this is not json at all";
        let parsed = parse_tier_response(raw, Tier::T4);
        assert_eq!(parsed.raw_text, raw);
        assert_eq!(parsed.confidence, 0.5);
    }

    #[test]
    fn falls_back_to_full_text_when_response_field_absent() {
        let raw = r#"{"confidence": 0.9}"#;
        let parsed = parse_tier_response(raw, Tier::T3);
        assert_eq!(parsed.raw_text, raw);
        assert!((parsed.confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn extracts_the_envelope_past_a_thinking_models_reasoning_block() {
        // Reproduces exactly what richardyoung/qwen3.6-27b-abliterated sent
        // live: a <think>...</think> reasoning trace, then the actual JSON
        // envelope, then trailing control tokens - none of which is valid
        // JSON as a whole string.
        let raw = "\n\n<think>\nSome internal reasoning that isn't JSON at all.\n</think>\n\n{\"confidence\": 0.95, \"response\": \"the real answer\"}<|endoftext|><|im_start|>user\n";
        let parsed = parse_tier_response(raw, Tier::T3);
        assert_eq!(parsed.raw_text, "the real answer");
        assert!((parsed.confidence - 0.95).abs() < 1e-6);
    }

    #[test]
    fn a_thinking_block_containing_braces_does_not_confuse_extraction() {
        let raw = "<think>reasoning that mentions a {curly} aside</think>{\"confidence\": 0.7, \"response\": \"ok\"}";
        let parsed = parse_tier_response(raw, Tier::T3);
        assert_eq!(parsed.raw_text, "ok");
        assert!((parsed.confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn trailing_content_with_its_own_braces_does_not_swallow_the_real_answer() {
        // Reproduces exactly what richardyoung/qwen3.6-27b-abliterated sent
        // live: after `</think>` and the real JSON answer, the completion
        // went on to echo an entire second prompt verbatim - and that
        // echoed prompt's own instruction text contains a literal
        // `{"confidence": ..., "response": ...}` template, well past the
        // real answer's own closing brace. A naive "first `{` to last `}`"
        // scan grabbed all the way to that far-later brace, producing an
        // unparseable span and falling back to storing the *entire* raw
        // completion (reasoning trace and echoed prompt included) as the
        // Reflection's text - confirmed live in the running engine.
        let raw = "\n\n<think>\nsome reasoning\n</think>\n\n```json\n{\"confidence\": 0.98, \"response\": \"the real answer\"}\n```<|endoftext|><|im_start|>user\n\nReflect on this. Respond with ONLY a JSON object of the form {\"confidence\": <0.0-1.0>, \"response\": \"<your reflection>\"}.";
        let parsed = parse_tier_response(raw, Tier::T3);
        assert_eq!(parsed.raw_text, "the real answer");
        assert!((parsed.confidence - 0.98).abs() < 1e-6);
    }

    #[test]
    fn a_response_value_containing_a_literal_brace_does_not_end_extraction_early() {
        // String-awareness must cut both ways: a brace *inside* the
        // response text's own quoted value must not be mistaken for the
        // object's structural closing brace either.
        let raw = "{\"confidence\": 0.6, \"response\": \"note: {see above}\"}";
        let parsed = parse_tier_response(raw, Tier::T3);
        assert_eq!(parsed.raw_text, "note: {see above}");
        assert!((parsed.confidence - 0.6).abs() < 1e-6);
    }
}
