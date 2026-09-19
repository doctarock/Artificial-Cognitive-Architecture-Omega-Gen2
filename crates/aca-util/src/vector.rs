/// Cosine similarity between two equal-length vectors, in `[-1.0, 1.0]`.
/// Returns `0.0` for a zero-length or mismatched-length input rather than
/// dividing by zero/panicking — an object with no embedding yet should
/// never reach this function (see the embedding-is-I/O wrinkle), but the
/// math itself stays total.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// `1 - cosine_similarity`, the prediction-error magnitude used throughout
/// the Compare step. Ranges `0.0` (identical direction) to `2.0` (opposite
/// direction).
pub fn cosine_error(expected: &[f32], actual: &[f32]) -> f32 {
    1.0 - cosine_similarity(expected, actual)
}

/// Element-wise weighted blend of same-length vectors, normalizing weights
/// to sum to 1 so callers don't have to pre-normalize. Empty input yields
/// an empty vector; a weight of `0.0` (or all-zero weights) is handled
/// without dividing by zero.
pub fn weighted_blend(vectors_and_weights: &[(&[f32], f32)]) -> Vec<f32> {
    let total_weight: f32 = vectors_and_weights.iter().map(|(_, w)| w).sum();
    let Some(dim) = vectors_and_weights.iter().map(|(v, _)| v.len()).max() else {
        return Vec::new();
    };
    if total_weight <= 0.0 || dim == 0 {
        return vec![0.0; dim];
    }
    let mut blended = vec![0.0f32; dim];
    for (vector, weight) in vectors_and_weights {
        let normalized_weight = weight / total_weight;
        for (i, value) in vector.iter().enumerate() {
            blended[i] += value * normalized_weight;
        }
    }
    blended
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_similarity_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn opposite_vectors_have_similarity_negative_one() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - -1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_have_similarity_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn mismatched_or_empty_inputs_return_zero_not_panic() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn cosine_error_is_zero_for_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        assert!(cosine_error(&v, &v).abs() < 1e-6);
    }

    #[test]
    fn weighted_blend_respects_relative_weights() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let blended = weighted_blend(&[(&a, 3.0), (&b, 1.0)]);
        // normalized weights: 0.75, 0.25
        assert!((blended[0] - 0.75).abs() < 1e-6);
        assert!((blended[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn weighted_blend_handles_empty_input() {
        assert_eq!(weighted_blend(&[]), Vec::<f32>::new());
    }

    #[test]
    fn weighted_blend_handles_zero_total_weight() {
        let a = vec![1.0, 2.0];
        let blended = weighted_blend(&[(&a, 0.0)]);
        assert_eq!(blended, vec![0.0, 0.0]);
    }
}
