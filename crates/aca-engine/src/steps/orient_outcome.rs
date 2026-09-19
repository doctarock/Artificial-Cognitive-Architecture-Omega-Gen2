use serde::Deserialize;

use super::orient::OrientingResult;

const SCHEMA: &str = "omega-orient-observed-outcome-logistic/v1";
const FEATURES: [&str; 8] = ["novelty", "prediction_error", "goal_relevance",
    "affective_salience", "social_relevance", "threat", "urgency", "ignited"];

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum OrientOutcomeModelError {
    #[error("invalid JSON: {0}")]
    Json(String),
    #[error("unsupported or unsafe artifact: {0}")]
    Invalid(&'static str),
}

#[derive(Debug, Clone)]
pub struct OrientOutcomeSpecialist {
    means: [f64; 8],
    scales: [f64; 8],
    weights: [f64; 8],
    bias: f64,
}

#[derive(Deserialize)]
struct Metrics { sample_count: usize, balanced_accuracy: f64, brier_score: f64 }

#[derive(Deserialize)]
struct Artifact {
    schema: String,
    status: String,
    features: Vec<String>,
    means: Vec<f64>,
    scales: Vec<f64>,
    weights: Vec<f64>,
    bias: f64,
    held_out_metrics: Metrics,
}

impl OrientOutcomeSpecialist {
    /// Loads only a held-out-gated, explicitly shadow-only artifact. This
    /// specialist forecasts observed outcome; it never controls attention.
    pub fn from_json(text: &str) -> Result<Self, OrientOutcomeModelError> {
        let artifact: Artifact = serde_json::from_str(text)
            .map_err(|error| OrientOutcomeModelError::Json(error.to_string()))?;
        if artifact.schema != SCHEMA { return Err(OrientOutcomeModelError::Invalid("schema")); }
        if artifact.status != "shadow_only" { return Err(OrientOutcomeModelError::Invalid("status must be shadow_only")); }
        if artifact.features.iter().map(String::as_str).ne(FEATURES) {
            return Err(OrientOutcomeModelError::Invalid("feature order"));
        }
        if artifact.held_out_metrics.sample_count < 200
            || artifact.held_out_metrics.balanced_accuracy < 0.65
            || artifact.held_out_metrics.brier_score > 0.22
            || !artifact.held_out_metrics.balanced_accuracy.is_finite()
            || !artifact.held_out_metrics.brier_score.is_finite()
        { return Err(OrientOutcomeModelError::Invalid("held-out promotion metrics")); }
        let means: [f64; 8] = artifact.means.try_into().map_err(|_| OrientOutcomeModelError::Invalid("mean width"))?;
        let scales: [f64; 8] = artifact.scales.try_into().map_err(|_| OrientOutcomeModelError::Invalid("scale width"))?;
        let weights: [f64; 8] = artifact.weights.try_into().map_err(|_| OrientOutcomeModelError::Invalid("weight width"))?;
        if !artifact.bias.is_finite() || means.iter().chain(scales.iter()).chain(weights.iter()).any(|value| !value.is_finite())
            || scales.iter().any(|scale| *scale <= 0.0)
        { return Err(OrientOutcomeModelError::Invalid("non-finite parameter or nonpositive scale")); }
        Ok(Self { means, scales, weights, bias: artifact.bias })
    }

    pub fn predict_observed_success(&self, orienting: &OrientingResult, ignited: bool) -> f32 {
        let values = [orienting.novelty, orienting.prediction_error, orienting.goal_relevance,
            orienting.affective_salience, orienting.social_relevance, orienting.threat,
            orienting.urgency, f32::from(ignited)].map(f64::from);
        let score = self.bias + (0..8).map(|index|
            self.weights[index] * (values[index] - self.means[index]) / self.scales[index]).sum::<f64>();
        (1.0 / (1.0 + (-score.clamp(-30.0, 30.0)).exp())) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(status: &str, balanced_accuracy: f64) -> String {
        serde_json::json!({
            "schema": SCHEMA, "status": status, "features": FEATURES,
            "means": vec![0.0; 8], "scales": vec![1.0; 8],
            "weights": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "bias": -1.0,
            "held_out_metrics": {"sample_count": 200, "balanced_accuracy": balanced_accuracy, "brier_score": 0.1}
        }).to_string()
    }

    #[test]
    fn loads_only_gated_shadow_artifacts_and_predicts_finitely() {
        let model = OrientOutcomeSpecialist::from_json(&artifact("shadow_only", 0.8)).unwrap();
        let orienting = OrientingResult { score: 0.0, fired: false, novelty: 1.0,
            prediction_error: 0.2, goal_relevance: 0.0, affective_salience: 0.0,
            social_relevance: 0.0, threat: 0.0, urgency: 0.0 };
        assert!(model.predict_observed_success(&orienting, true) > model.predict_observed_success(&orienting, false));
        assert!(OrientOutcomeSpecialist::from_json(&artifact("active", 0.8)).is_err());
        assert!(OrientOutcomeSpecialist::from_json(&artifact("shadow_only", 0.6)).is_err());
    }
}
