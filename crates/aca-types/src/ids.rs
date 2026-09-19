use std::fmt;
use std::str::FromStr;
use uuid::Uuid;

/// Defines a UUID-backed newtype id. Each Mental Object id-namespace gets its
/// own distinct type (`MentalObjectId`, `EdgeId`, `GoalStackId`) so that
/// "passed the wrong kind of id" is a compile error rather than a runtime
/// bug — the one class of mistake the original TypeScript design (bare
/// `string` ids) had no static defense against.
macro_rules! uuid_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub Uuid);

        impl $name {
            /// Time-orderable v7 UUID — pairs naturally with ACT-R's
            /// recency-based activation reasoning (newer ids sort later).
            pub fn new() -> Self {
                Self(Uuid::now_v7())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }

        impl FromStr for $name {
            type Err = uuid::Error;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Ok(Self(Uuid::from_str(s)?))
            }
        }
    };
}

uuid_id!(MentalObjectId);
uuid_id!(EdgeId);
uuid_id!(GoalStackId);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_id_types_have_distinct_values() {
        let a = MentalObjectId::new();
        let b = MentalObjectId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn round_trips_through_display_and_from_str() {
        let id = MentalObjectId::new();
        let text = id.to_string();
        let parsed: MentalObjectId = text.parse().expect("valid uuid text");
        assert_eq!(id, parsed);
    }

    #[test]
    fn serializes_as_a_plain_string() {
        let id = MentalObjectId::new();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, format!("\"{}\"", id.0));
    }
}
