/// Clamp a value into `0.0..=1.0`, defaulting to `0.5` if the value is not a
/// finite number (NaN/infinite) — the same "never fail the turn on a
/// malformed/missing signal" contract used throughout the tier-response and
/// confidence handling.
pub fn clamp01_or_default(value: f32) -> f32 {
    if !value.is_finite() {
        return 0.5;
    }
    value.clamp(0.0, 1.0)
}

/// Clamp a value into an arbitrary closed range, defaulting to the range's
/// midpoint if the value is not finite.
pub fn clamp_or_default(value: f32, min: f32, max: f32) -> f32 {
    if !value.is_finite() {
        return min + (max - min) / 2.0;
    }
    value.clamp(min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_in_range_values_unchanged() {
        assert_eq!(clamp01_or_default(0.7), 0.7);
        assert_eq!(clamp01_or_default(0.0), 0.0);
        assert_eq!(clamp01_or_default(1.0), 1.0);
    }

    #[test]
    fn clamps_out_of_range_values() {
        assert_eq!(clamp01_or_default(1.5), 1.0);
        assert_eq!(clamp01_or_default(-0.3), 0.0);
    }

    #[test]
    fn defaults_non_finite_values() {
        assert_eq!(clamp01_or_default(f32::NAN), 0.5);
        assert_eq!(clamp01_or_default(f32::INFINITY), 0.5);
        assert_eq!(clamp01_or_default(f32::NEG_INFINITY), 0.5);
    }

    #[test]
    fn clamp_or_default_uses_range_midpoint() {
        assert_eq!(clamp_or_default(f32::NAN, 10.0, 20.0), 15.0);
        assert_eq!(clamp_or_default(25.0, 10.0, 20.0), 20.0);
        assert_eq!(clamp_or_default(5.0, 10.0, 20.0), 10.0);
    }
}
