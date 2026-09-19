use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Milliseconds since the Unix epoch. The one timestamp representation used
/// throughout the engine: persisted, transmitted, and compared as a plain
/// integer rather than `std::time::Instant` (not serializable) or
/// `SystemTime` (not directly orderable/arithmetic-friendly for the ACT-R
/// decay math).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct EpochMillis(pub i64);

impl EpochMillis {
    pub fn as_millis(self) -> i64 {
        self.0
    }

    /// Elapsed time from `self` to `later`, in milliseconds. Negative if
    /// `later` is actually earlier than `self`.
    pub fn elapsed_ms_until(self, later: EpochMillis) -> i64 {
        later.0 - self.0
    }
}

/// Injectable time source so activation decay/spreading math is
/// deterministic and testable without real wall-clock sleeps.
pub trait Clock: Send + Sync {
    fn now(&self) -> EpochMillis;
}

/// The real wall-clock, used in production.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> EpochMillis {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is before the Unix epoch")
            .as_millis() as i64;
        EpochMillis(millis)
    }
}

/// A manually-advanced clock for tests: starts at a fixed instant and only
/// moves when `advance`/`set` is called, so decay/activation math can be
/// asserted against exact, reproducible time deltas.
#[derive(Debug)]
pub struct ManualClock {
    now: AtomicI64,
}

impl ManualClock {
    pub fn new(start: EpochMillis) -> Self {
        Self {
            now: AtomicI64::new(start.0),
        }
    }

    pub fn advance(&self, delta_ms: i64) {
        self.now.fetch_add(delta_ms, Ordering::SeqCst);
    }

    pub fn set(&self, at: EpochMillis) {
        self.now.store(at.0, Ordering::SeqCst);
    }
}

impl Clock for ManualClock {
    fn now(&self) -> EpochMillis {
        EpochMillis(self.now.load(Ordering::SeqCst))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_clock_advances_deterministically() {
        let clock = ManualClock::new(EpochMillis(1_000));
        assert_eq!(clock.now(), EpochMillis(1_000));
        clock.advance(500);
        assert_eq!(clock.now(), EpochMillis(1_500));
        clock.set(EpochMillis(9_999));
        assert_eq!(clock.now(), EpochMillis(9_999));
    }

    #[test]
    fn elapsed_ms_until_computes_delta() {
        let a = EpochMillis(1_000);
        let b = EpochMillis(1_750);
        assert_eq!(a.elapsed_ms_until(b), 750);
        assert_eq!(b.elapsed_ms_until(a), -750);
    }
}
