use std::collections::HashMap;

use serde::Deserialize;

use super::communicative_intent::{hashed_text_vector, HASHED_TEXT_DIMENSIONS, HASHED_TEXT_FEATURE_SCHEMA};

const SCHEMA: &str = "omega-tool-intent-softmax/v1";
const LABELS: [&str; 4] = ["none", "current_time", "self_status", "abstraction_status"];

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ToolIntentModelError {
    #[error("invalid JSON: {0}")]
    Json(String),
    #[error("unsupported or unsafe artifact: {0}")]
    Invalid(&'static str),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolIntentPrediction {
    pub label: &'static str,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct ToolIntentSpecialist {
    weights: Vec<Vec<f64>>,
    bias: [f64; 4],
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

impl ToolIntentSpecialist {
    pub fn from_json(text: &str) -> Result<Self, ToolIntentModelError> {
        let artifact: Artifact = serde_json::from_str(text)
            .map_err(|error| ToolIntentModelError::Json(error.to_string()))?;
        if artifact.schema != SCHEMA { return Err(ToolIntentModelError::Invalid("schema")); }
        if artifact.status != "shadow_only" { return Err(ToolIntentModelError::Invalid("status must be shadow_only")); }
        if artifact.labels.iter().map(String::as_str).ne(LABELS) { return Err(ToolIntentModelError::Invalid("label order")); }
        if artifact.dimensions != HASHED_TEXT_DIMENSIONS || artifact.feature_schema != HASHED_TEXT_FEATURE_SCHEMA {
            return Err(ToolIntentModelError::Invalid("feature schema or width"));
        }
        if artifact.held_out.samples < 40
            || artifact.held_out.accuracy < 0.90
            || !artifact.held_out.accuracy.is_finite()
            || LABELS.iter().any(|label| artifact.held_out.class_recall.get(*label)
                .is_none_or(|recall| !recall.is_finite() || *recall < 0.80))
        {
            return Err(ToolIntentModelError::Invalid("held-out promotion metrics"));
        }
        if artifact.weights.len() != LABELS.len()
            || artifact.weights.iter().any(|row| row.len() != HASHED_TEXT_DIMENSIONS || row.iter().any(|value| !value.is_finite()))
        {
            return Err(ToolIntentModelError::Invalid("weight shape or value"));
        }
        let bias: [f64; 4] = artifact.bias.try_into().map_err(|_| ToolIntentModelError::Invalid("bias width"))?;
        if bias.iter().any(|value| !value.is_finite()) { return Err(ToolIntentModelError::Invalid("bias value")); }
        Ok(Self { weights: artifact.weights, bias })
    }

    pub fn predict(&self, text: &str) -> ToolIntentPrediction {
        let features = hashed_text_vector(text);
        let logits: [f64; 4] = std::array::from_fn(|label| {
            self.bias[label] + features.iter().map(|(index, value)| self.weights[label][*index] * value).sum::<f64>()
        });
        let peak = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exponentials = logits.map(|value| (value - peak).exp());
        let total = exponentials.iter().sum::<f64>();
        let probabilities = exponentials.map(|value| value / total);
        let winner = (0..LABELS.len()).max_by(|a, b| probabilities[*a].total_cmp(&probabilities[*b])).unwrap_or(0);
        ToolIntentPrediction { label: LABELS[winner], confidence: probabilities[winner] as f32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRAINED: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../training/specialists/outputs/tool_intent.json"));

    #[test]
    fn trained_artifact_loads_and_handles_requests_and_topical_negatives() {
        let model = ToolIntentSpecialist::from_json(TRAINED).unwrap();
        assert_eq!(model.predict("Could you tell me what time it is right now?").label, "current_time");
        assert_eq!(model.predict("What currently has your attention?").label, "self_status");
        assert_eq!(model.predict("Show me your current provisional abstraction.").label, "abstraction_status");
        assert_eq!(model.predict("The status page has a repeating pattern.").label, "none");
    }

    #[test]
    fn loader_rejects_control_status_and_weak_negative_recall() {
        let mut artifact: serde_json::Value = serde_json::from_str(TRAINED).unwrap();
        artifact["status"] = serde_json::json!("active");
        assert!(ToolIntentSpecialist::from_json(&artifact.to_string()).is_err());
        artifact["status"] = serde_json::json!("shadow_only");
        artifact["held_out"]["class_recall"]["none"] = serde_json::json!(0.79);
        assert!(ToolIntentSpecialist::from_json(&artifact.to_string()).is_err());
    }

    #[test]
    #[ignore = "local microbenchmark; timings vary with host load"]
    fn tool_intent_inference_latency_probe() {
        let model = ToolIntentSpecialist::from_json(TRAINED).unwrap();
        for _ in 0..100 { let _ = model.predict("Could you report the current local time?"); }
        let mut samples_us = Vec::with_capacity(10_000);
        for _ in 0..10_000 {
            let started = std::time::Instant::now();
            let _ = model.predict("Could you report the current local time?");
            samples_us.push(started.elapsed().as_micros() as u64);
        }
        samples_us.sort_unstable();
        println!("tool-intent shadow inference: n=10000, p50_us={}, p95_us={}, max_us={}",
            samples_us[4_999], samples_us[9_499], samples_us[9_999]);
        assert!(samples_us[9_499] < 10_000);
    }
}
