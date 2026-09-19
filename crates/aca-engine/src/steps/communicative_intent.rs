use std::collections::HashMap;

use serde::Deserialize;

const SCHEMA: &str = "omega-communicative-intent-softmax/v1";
pub(super) const HASHED_TEXT_FEATURE_SCHEMA: &str = "fnv1a-word-bigram-char3-5-l2/v1";
const LABELS: [&str; 3] = ["speak", "ask", "ignore"];
pub(super) const HASHED_TEXT_DIMENSIONS: usize = 1024;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CommunicativeIntentModelError {
    #[error("invalid JSON: {0}")]
    Json(String),
    #[error("unsupported or unsafe artifact: {0}")]
    Invalid(&'static str),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CommunicativeIntentPrediction {
    pub label: &'static str,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct CommunicativeIntentSpecialist {
    weights: Vec<Vec<f64>>,
    bias: [f64; 3],
}

#[derive(Deserialize)]
struct HeldOut {
    samples: usize,
    accuracy: f64,
    class_recall: HashMap<String, f64>,
}

#[derive(Deserialize)]
struct Artifact {
    schema: String,
    status: String,
    labels: Vec<String>,
    dimensions: usize,
    feature_schema: String,
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
    held_out: HeldOut,
}

impl CommunicativeIntentSpecialist {
    /// Loads only an explicitly shadow-only artifact whose held-out metrics,
    /// feature schema, dimensions, and parameters independently pass the
    /// runtime gate. The type exposes prediction but no control operation.
    pub fn from_json(text: &str) -> Result<Self, CommunicativeIntentModelError> {
        let artifact: Artifact = serde_json::from_str(text)
            .map_err(|error| CommunicativeIntentModelError::Json(error.to_string()))?;
        if artifact.schema != SCHEMA { return Err(CommunicativeIntentModelError::Invalid("schema")); }
        if artifact.status != "shadow_only" { return Err(CommunicativeIntentModelError::Invalid("status must be shadow_only")); }
        if artifact.labels.iter().map(String::as_str).ne(LABELS) { return Err(CommunicativeIntentModelError::Invalid("label order")); }
        if artifact.dimensions != HASHED_TEXT_DIMENSIONS || artifact.feature_schema != HASHED_TEXT_FEATURE_SCHEMA {
            return Err(CommunicativeIntentModelError::Invalid("feature schema or width"));
        }
        if artifact.held_out.samples < 30
            || artifact.held_out.accuracy < 0.90
            || !artifact.held_out.accuracy.is_finite()
            || LABELS.iter().any(|label| artifact.held_out.class_recall.get(*label)
                .is_none_or(|recall| !recall.is_finite() || *recall < 0.80))
        {
            return Err(CommunicativeIntentModelError::Invalid("held-out promotion metrics"));
        }
        if artifact.weights.len() != LABELS.len()
            || artifact.weights.iter().any(|row| row.len() != HASHED_TEXT_DIMENSIONS || row.iter().any(|value| !value.is_finite()))
        {
            return Err(CommunicativeIntentModelError::Invalid("weight shape or value"));
        }
        let bias: [f64; 3] = artifact.bias.try_into()
            .map_err(|_| CommunicativeIntentModelError::Invalid("bias width"))?;
        if bias.iter().any(|value| !value.is_finite()) {
            return Err(CommunicativeIntentModelError::Invalid("bias value"));
        }
        Ok(Self { weights: artifact.weights, bias })
    }

    pub fn predict(&self, text: &str) -> CommunicativeIntentPrediction {
        let features = hashed_text_vector(text);
        let logits: [f64; 3] = std::array::from_fn(|label| {
            self.bias[label] + features.iter().map(|(index, value)| self.weights[label][*index] * value).sum::<f64>()
        });
        let peak = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exponentials = logits.map(|value| (value - peak).exp());
        let total = exponentials.iter().sum::<f64>();
        let probabilities = exponentials.map(|value| value / total);
        let winner = (0..LABELS.len()).max_by(|a, b| probabilities[*a].total_cmp(&probabilities[*b])).unwrap_or(0);
        CommunicativeIntentPrediction { label: LABELS[winner], confidence: probabilities[winner] as f32 }
    }
}

fn fnv1a(text: &str) -> u64 {
    text.as_bytes().iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn feature_strings(text: &str) -> Vec<String> {
    let normalized = text.to_ascii_lowercase().chars()
        .map(|character| if character.is_ascii_alphanumeric() || character == '\'' { character } else { ' ' })
        .collect::<String>()
        .split_whitespace().collect::<Vec<_>>().join(" ");
    let words: Vec<&str> = normalized.split_whitespace().collect();
    let mut features: Vec<String> = words.iter().map(|word| format!("w:{word}")).collect();
    features.extend(words.windows(2).map(|pair| format!("b:{}_{}", pair[0], pair[1])));
    let padded = format!("  {normalized}  ");
    for width in [3, 4, 5] {
        if padded.len() >= width {
            features.extend((0..=padded.len() - width).map(|index| format!("c:{}", &padded[index..index + width])));
        }
    }
    features
}

pub(super) fn hashed_text_vector(text: &str) -> Vec<(usize, f64)> {
    let mut values = HashMap::<usize, f64>::new();
    for feature in feature_strings(text) {
        let hashed = fnv1a(&feature);
        let index = hashed as usize % HASHED_TEXT_DIMENSIONS;
        let sign = if hashed >> 63 == 0 { 1.0 } else { -1.0 };
        *values.entry(index).or_default() += sign;
    }
    let norm = values.values().map(|value| value * value).sum::<f64>().sqrt().max(1.0);
    values.into_iter().map(|(index, value)| (index, value / norm)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRAINED: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../training/specialists/outputs/communicative_intent.json"));

    #[test]
    fn trained_artifact_loads_and_matches_representative_held_out_phrases() {
        let model = CommunicativeIntentSpecialist::from_json(TRAINED).unwrap();
        assert_eq!(model.predict("Report that the requested calculation produced 42.").label, "speak");
        assert_eq!(model.predict("The destination folder is missing; ask which folder to use.").label, "ask");
        assert_eq!(model.predict("This private implementation note does not need to be voiced.").label, "ignore");
    }

    #[test]
    fn loader_rejects_control_status_and_claimed_metrics_below_gate() {
        let mut artifact: serde_json::Value = serde_json::from_str(TRAINED).unwrap();
        artifact["status"] = serde_json::json!("active");
        assert!(CommunicativeIntentSpecialist::from_json(&artifact.to_string()).is_err());
        artifact["status"] = serde_json::json!("shadow_only");
        artifact["held_out"]["class_recall"]["ask"] = serde_json::json!(0.79);
        assert!(CommunicativeIntentSpecialist::from_json(&artifact.to_string()).is_err());
    }

    #[test]
    #[ignore = "local microbenchmark; timings vary with host load"]
    fn communicative_intent_inference_latency_probe() {
        let model = CommunicativeIntentSpecialist::from_json(TRAINED).unwrap();
        for _ in 0..100 { let _ = model.predict("The destination is missing; ask which folder to use."); }
        let mut samples_us = Vec::with_capacity(10_000);
        for _ in 0..10_000 {
            let started = std::time::Instant::now();
            let _ = model.predict("The destination is missing; ask which folder to use.");
            samples_us.push(started.elapsed().as_micros() as u64);
        }
        samples_us.sort_unstable();
        println!("communicative-intent shadow inference: n=10000, p50_us={}, p95_us={}, max_us={}",
            samples_us[4_999], samples_us[9_499], samples_us[9_999]);
        assert!(samples_us[9_499] < 10_000, "a local classifier must not restore multi-second latency");
    }
}
