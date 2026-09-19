use std::collections::VecDeque;

/// A fixed-capacity FIFO buffer: pushing past capacity evicts the oldest
/// entry. Used for ACT-R's `reference_log` (bounded history of past
/// reference timestamps feeding the base-level activation sum) and for the
/// rolling per-source-channel precision windows — both need "the last K
/// values," never an unbounded history.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RingBuffer<T> {
    capacity: usize,
    items: VecDeque<T>,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be positive");
        Self {
            capacity,
            items: VecDeque::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, item: T) {
        if self.items.len() == self.capacity {
            self.items.pop_front();
        }
        self.items.push_back(item);
    }

    /// Same eviction as `push`, but returns the evicted item instead of
    /// discarding it - for a caller that needs to know *which* item aged
    /// out, e.g. to remove a matching entry from a companion lookup
    /// structure (a bounded FIFO cache keyed by this buffer's own order).
    pub fn push_evicting(&mut self, item: T) -> Option<T> {
        let evicted = if self.items.len() == self.capacity { self.items.pop_front() } else { None };
        self.items.push_back(item);
        evicted
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_oldest_when_over_capacity() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(3);
        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert_eq!(buf.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3]);
        buf.push(4);
        assert_eq!(buf.iter().copied().collect::<Vec<_>>(), vec![2, 3, 4]);
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn push_evicting_returns_the_evicted_item_only_once_over_capacity() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(2);
        assert_eq!(buf.push_evicting(1), None);
        assert_eq!(buf.push_evicting(2), None);
        assert_eq!(buf.push_evicting(3), Some(1));
        assert_eq!(buf.iter().copied().collect::<Vec<_>>(), vec![2, 3]);
    }

    #[test]
    fn empty_buffer_reports_empty() {
        let buf: RingBuffer<i32> = RingBuffer::new(2);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    #[should_panic(expected = "capacity must be positive")]
    fn zero_capacity_panics() {
        RingBuffer::<i32>::new(0);
    }
}
