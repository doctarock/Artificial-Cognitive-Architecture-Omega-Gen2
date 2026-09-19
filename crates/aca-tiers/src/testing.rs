use async_trait::async_trait;

use crate::client::{EmbeddingClient, TierError};

/// A deterministic, network-free `EmbeddingClient` for tests: the same text
/// always yields the same unit-length vector, different text yields an
/// uncorrelated one. Not semantically meaningful (this is a hash, not a
/// real embedding model) — tests that need specific similarity
/// relationships should construct `MentalObject::embedding` directly rather
/// than relying on this client's output to "mean" anything.
pub struct FakeEmbeddingClient {
    pub dimension: usize,
}

impl FakeEmbeddingClient {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Default for FakeEmbeddingClient {
    fn default() -> Self {
        Self::new(8)
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[async_trait]
impl EmbeddingClient for FakeEmbeddingClient {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, TierError> {
        let mut seed = fnv1a(text.as_bytes());
        let mut values = Vec::with_capacity(self.dimension);
        for _ in 0..self.dimension {
            // xorshift64
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            // map to roughly [-1, 1]
            let value = (seed % 2000) as f32 / 1000.0 - 1.0;
            values.push(value);
        }
        let norm: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut values {
                *v /= norm;
            }
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn same_text_yields_the_same_vector() {
        let client = FakeEmbeddingClient::default();
        let a = client.embed("hello world").await.unwrap();
        let b = client.embed("hello world").await.unwrap();
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn different_text_yields_different_vectors() {
        let client = FakeEmbeddingClient::default();
        let a = client.embed("hello").await.unwrap();
        let b = client.embed("goodbye").await.unwrap();
        assert_ne!(a, b);
    }

    #[tokio::test]
    async fn vectors_are_unit_length() {
        let client = FakeEmbeddingClient::default();
        let v = client.embed("anything").await.unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3);
    }
}
