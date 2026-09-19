use std::time::Duration;

/// Builds the one shared `reqwest::Client` used by every adapter.
/// `reqwest`'s default is the *opposite* footgun from the old JS system's
/// (undici's 300s `headersTimeout` silently killing legitimate long
/// generations): reqwest has **no default timeout at all**, so a truly
/// hung connection would block its task forever. Rather than lean on a
/// client-level timeout (which would apply uniformly to every tier
/// regardless of how long that tier's generations legitimately take), every
/// call site wraps its own future in an explicit `tokio::time::timeout`
/// sized per tier (see `pool.rs`) — this client only sets a short
/// *connect* timeout (a dead/unreachable host should fail fast) and TCP
/// keepalive (guards against silent idle-socket reaping during a long
/// silent-until-done non-streaming generation).
pub fn build_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .tcp_keepalive(Duration::from_secs(30))
        .build()
        .expect("reqwest client construction should never fail with static config")
}
