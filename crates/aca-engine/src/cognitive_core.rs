use aca_tiers::{ChatClient, DivergentPool, EmbeddingClient, GenerateRequest, TierError, TierPool, TierResponse};
use aca_types::{MentalObject, MentalObjectKind};
use aca_util::Clock;

use crate::prompt_templates::{reflect_prompt, ContextEntry};
use crate::steps::arbitrate::arbitrate_by_agreement;

/// How many independent samples a cheap tier needs before its answer can be
/// trusted at all - see `try_tier_via_agreement`. When fewer clients are
/// configured than this, `DivergentPool::sample` makes up the shortfall by
/// repeat-calling the clients it does have, so a solo-model tier still gets
/// a real cross-check instead of being permanently unconfirmable.
const MIN_SAMPLES: usize = 3;

/// Below this cross-candidate agreement (mean cosine similarity between
/// independently-generated answers - see `steps::arbitrate`), Tier 1's
/// answer isn't trusted enough to stop at - escalate to Tier 2. Stricter
/// than Tier 2's threshold: small models need tighter consensus before
/// their combined answer is trusted with anything Omega might say.
const TIER1_AGREEMENT_THRESHOLD: f32 = 0.92;

/// Same idea as `TIER1_AGREEMENT_THRESHOLD`, one rung up. Looser than
/// Tier 1's bar - ~9B models are more reliable, so somewhat less mutual
/// agreement is still meaningful corroboration.
const TIER2_AGREEMENT_THRESHOLD: f32 = 0.80;

/// The Cognitive Core's output: a new Reflection Mental Object plus the raw
/// tier response it was built from (kept for logging/`cycle_events`, not
/// because callers need to re-derive anything from it).
pub struct CognitiveCoreOutput {
    pub reflection: MentalObject,
    pub tier_response: TierResponse,
}

/// Runs reflection/planning/reasoning (specs.md's Cognitive Core) on
/// whatever won GWT broadcast and was selected by the Executive for
/// elaboration. Tries Tier 1 (many cheap, concurrent) first, then Tier 2 (up
/// to three ~9B, concurrent), escalating to Tier 3 (the single-flight seat)
/// only when neither cheap tier produced answers that actually agree with
/// each other - "escalation is confidence-driven" (specs.md's tiering
/// section), made literal, except the confidence is computed (Tier 0:
/// cross-candidate embedding agreement, see `steps::arbitrate`), never a
/// tier's own self-reported number. Tier 3 is always the final backstop, so
/// a real answer is still attempted even if Tier 1/2 are unconfigured or
/// their candidates didn't agree; either pool being empty (nothing
/// configured) is treated as "skip this tier," not an error.
///
/// This replaces an earlier design that trusted each tier's own
/// self-reported `confidence` field. Live testing falsified that for Tier 1
/// directly: a sub-4B model free-associated "Omega" into a Super Mario
/// character while claiming `confidence: 1.0` in response to "hi, can you
/// hear me?" - a threshold on a number the model itself produces can't
/// defend against a model whose self-report carries no real signal at that
/// size. The fix is structural, not a smarter threshold: confidence is now
/// *earned* by independent samples actually landing near the same place in
/// embedding space (see `try_tier_via_agreement`), which is what makes it
/// safe to give Tier 1 a real role in the ladder at all instead of excluding
/// it outright.
///
/// Escalation to Tier 4 is a separate, caller-level concern, not something
/// this function decides on its own. Never communicates directly with the
/// outside world — its output is a Mental Object, handed to the Social
/// Interface only if the Executive later selects a "speak" operator on it.
///
/// Takes a `clock` rather than a pre-fetched `EpochMillis` deliberately: the
/// tier call this awaits can genuinely take tens of seconds on a real
/// model, and the Reflection's own activation is seeded from a single
/// reference at its creation instant (see `ActivationState::new_at`). Timing
/// it against whatever "now" was captured *before* the await backdates that
/// reference by the entire generation time — observed directly in a live
/// run: a 70-second CPU-bound Tier-3 call meant the Reflection's activation
/// had already decayed below the Coalition admission threshold by the time
/// it got its one shot at broadcast, so it was created but never spoken.
///
/// Also resolves the Reflection's own embedding via `embedding_client`
/// before returning — without this, `form_memory` discards it forever (no
/// embedding to compare/store), which means `already_remembered` can never
/// become true for it, which means `propose_operators` proposes `Remember`
/// for the same Reflection on *every subsequent tick indefinitely* — a real
/// runaway loop observed live (the "unreadable, fast-scrolling" symptom).
/// A failed embed here degrades gracefully (`embedding` stays `None`, same
/// one-shot Discarded outcome as before) rather than failing the whole
/// reflection - the loop-prevention fix in `propose_operators`/`act` handles
/// the remaining "permanently unembeddable" edge case, this just makes that
/// edge case rare instead of universal.
#[allow(clippy::too_many_arguments)]
pub async fn reflect(
    tier1_pool: &DivergentPool,
    tier2_pool: &DivergentPool,
    tier3_pool: &TierPool,
    tier3_client: &dyn ChatClient,
    embedding_client: &dyn EmbeddingClient,
    broadcast_object: &MentalObject,
    working_memory_entries: &[ContextEntry<'_>],
    self_summary: &str,
    presence: f32,
    clock: &dyn Clock,
    decay_d: f32,
    tier3_hedge_delay: std::time::Duration,
    temperature: f32,
    // Grounds `prompt_templates::reflect_prompt`'s displacement claim
    // (`steps::displacement::explain_release`'s verified verdict) - the
    // caller (`steps::act`, forwarding what `loop_actor` computed at
    // Broadcast) is responsible for only ever passing `Some` when
    // `broadcast_object` is itself this tick's real, counterfactually-
    // confirmed displacement entrant, never as an ambient "something was
    // displaced somewhere" fact unrelated to what's actually being
    // reflected on here.
    displacement_note: Option<&str>,
) -> Result<CognitiveCoreOutput, TierError> {
    let prompt = reflect_prompt(self_summary, presence, &broadcast_object.text, working_memory_entries, displacement_note);
    let req = GenerateRequest { prompt, temperature };
    let response = resolve_via_tier_ladder(tier1_pool, tier2_pool, tier3_pool, tier3_client, embedding_client, req, tier3_hedge_delay).await?;

    let now = clock.now();
    let mut reflection = MentalObject::new_observation(response.raw_text.clone(), now, decay_d);
    reflection.kind = MentalObjectKind::Reflection;
    reflection.confidence = response.confidence;
    reflection.tier_used = Some(response.tier);
    reflection.source_object_ids = vec![broadcast_object.id];
    reflection.embedding = embedding_client.embed(&response.raw_text).await.ok();

    Ok(CognitiveCoreOutput {
        reflection,
        tier_response: response,
    })
}

/// Starts Tier 1 and Tier 2 concurrently (rather than waiting out Tier 1's
/// full attempt before Tier 2 even begins) and accepts Tier 1's answer the
/// instant it clears its own agreement threshold - dropping Tier 2's
/// still-in-flight attempt at that point, since nothing downstream needs it
/// once Tier 1 has already earned trust. If Tier 1 doesn't clear its bar,
/// Tier 2's attempt has been running the whole time Tier 1 was, so there's
/// nothing left to wait out but whatever's left of Tier 2's own call -
/// otherwise falls through to Tier 3. This preserves Tier 1's priority over
/// Tier 2 exactly as before (only ever accepted for *not* clearing its own
/// threshold, never for losing a race), it just stops paying Tier 1's full
/// latency before Tier 2 gets a chance to start - the worst case (both cheap
/// tiers fail) used to sum both tiers' latency; now it pays roughly
/// `max(tier1, tier2)`. A cheap-tier failure (including "not configured,"
/// i.e. an empty pool) is swallowed, not propagated - only Tier 3 failing
/// can fail the whole reflection, since it's the guaranteed-always-configured
/// backstop both cheap tiers escalate toward.
///
/// `pub(crate)` rather than private: `steps::synthesize` reuses this
/// directly for pattern synthesis rather than re-deriving the same
/// escalation/agreement/empty-completion-handling logic a second time -
/// callers there typically pass an empty `tier1_pool` (this function's own
/// `is_empty()` guard skips it cleanly) since synthesis is deliberately
/// elaborative work, not fast triage.
///
/// `tier3_hedge_delay` adds one more layer on top of the Tier 1/Tier 2 race
/// described above: if neither cheap tier has produced a decisive answer
/// within that long, this non-blockingly tries to acquire Tier 3's single
/// seat (`TierPool::try_acquire`) and, if free, starts it *concurrently*
/// with whatever's left of the cheap tiers' race, rather than only starting
/// it after both are fully exhausted. This is what turns the worst case
/// from `tier1 + tier2 + tier3` latency (paid sequentially, and the direct
/// cause of the documented 70s reflection-decay incident - see `reflect`'s
/// own doc comment) into roughly `max(remaining tier1/tier2, tier3)`. If
/// Tier 3 is busy (someone else's `resolve_confidence_impasse` escalation
/// holding the seat), this is simply skipped - no queueing, no change from
/// the prior sequential-fallback behavior. A cheap tier winning after the
/// speculative Tier 3 call has started drops that call, releasing the
/// permit immediately (see `TierPool::run_chat_with_permit`'s own doc
/// comment on permit-drop semantics). A speculative Tier 3 call that itself
/// completes with a genuinely empty completion is treated as a non-win
/// (same rule as the ordinary retry path below) and simply discarded rather
/// than threaded into that retry's own state - a doubly-rare combination
/// (hedge fired *and* an empty completion) that at worst costs one extra
/// wasted Tier 3 call, not a correctness issue.
pub(crate) async fn resolve_via_tier_ladder(
    tier1_pool: &DivergentPool,
    tier2_pool: &DivergentPool,
    tier3_pool: &TierPool,
    tier3_client: &dyn ChatClient,
    embedding_client: &dyn EmbeddingClient,
    req: GenerateRequest,
    tier3_hedge_delay: std::time::Duration,
) -> Result<TierResponse, TierError> {
    // Each future owns its own clone of `req` (rather than borrowing the
    // local below) so its lifetime is independent of the original `req`,
    // which is still needed by value for the Tier 3 fallback call at the
    // bottom of this function.
    let tier1_req = req.clone();
    let tier2_req = req.clone();
    let tier1_fut = async move {
        if tier1_pool.is_empty() {
            None
        } else {
            try_tier_via_agreement(tier1_pool, embedding_client, &tier1_req, TIER1_AGREEMENT_THRESHOLD).await
        }
    };
    let tier2_fut = async move {
        if tier2_pool.is_empty() {
            None
        } else {
            try_tier_via_agreement(tier2_pool, embedding_client, &tier2_req, TIER2_AGREEMENT_THRESHOLD).await
        }
    };
    tokio::pin!(tier1_fut);
    tokio::pin!(tier2_fut);

    // The Tier 1/Tier 2 race, exactly as before, wrapped as its own future
    // so the hedge below can race it against a timer without duplicating
    // its internal priority rules (Tier 1 must definitively resolve before
    // a Tier 2 win is ever accepted - see the loop's own comments).
    let cheap_tiers_fut = async {
        let mut tier1_done = false;
        let mut tier2_done = false;
        let mut tier2_outcome: Option<TierResponse> = None;

        // Runs until Tier 1 has definitively resolved (accepted, or
        // exhausted with Tier 2's own outcome already in hand too) - never
        // longer, per `tier1_done && tier2_done` below breaking the loop
        // the instant both sides have reported in with neither accepted.
        while !tier1_done {
            tokio::select! {
                result = &mut tier1_fut, if !tier1_done => {
                    tier1_done = true;
                    if let Some(response) = result {
                        // Tier 1 earned it - Tier 2's still-in-flight
                        // attempt (if any) is dropped here, cancelling its
                        // underlying call rather than paying to wait out an
                        // answer nothing will ever use.
                        return Some(response);
                    }
                }
                result = &mut tier2_fut, if !tier2_done => {
                    tier2_done = true;
                    tier2_outcome = result;
                }
            }
        }
        if tier2_done {
            if let Some(response) = tier2_outcome {
                return Some(response);
            }
        } else if let Some(response) = tier2_fut.await {
            // Tier 1 resolved (without clearing its bar) before Tier 2
            // did - Tier 2 was already running the whole time, so this
            // only ever waits out whatever's left of its call, never a
            // fresh one started from zero.
            return Some(response);
        }
        None
    };
    tokio::pin!(cheap_tiers_fut);

    tokio::select! {
        cheap = &mut cheap_tiers_fut => {
            if let Some(response) = cheap {
                return Ok(response);
            }
            // Both cheap tiers exhausted before the hedge ever fired - fall
            // straight through to the unchanged sequential path below.
        }
        _ = tokio::time::sleep(tier3_hedge_delay) => {
            if let Some(permit) = tier3_pool.try_acquire() {
                let tier3_req = req.clone();
                tokio::select! {
                    cheap = &mut cheap_tiers_fut => {
                        // A cheap tier won the race - the speculative Tier
                        // 3 branch above is dropped by `select!` here,
                        // releasing its permit immediately.
                        if let Some(response) = cheap {
                            return Ok(response);
                        }
                    }
                    tier3_result = tier3_pool.run_chat_with_permit(permit, tier3_client, tier3_req) => {
                        match &tier3_result {
                            Ok(response) if !response.raw_text.trim().is_empty() => return tier3_result,
                            Err(_) => return tier3_result,
                            // Empty completion: not a usable answer (same
                            // rule as the retry path below) - the cheap
                            // tiers are still racing, so give them the
                            // remainder of their own attempt rather than
                            // treating this as a hard failure.
                            Ok(_) => {}
                        }
                        if let Some(response) = cheap_tiers_fut.await {
                            return Ok(response);
                        }
                    }
                }
            } else if let Some(response) = cheap_tiers_fut.await {
                // Tier 3 busy right now - no speculation possible, just
                // wait out the cheap tiers exactly as before.
                return Ok(response);
            }
        }
    }

    let mut response = tier3_pool.run_chat(tier3_client, req.clone()).await?;
    // Tier 3 is the guaranteed backstop - nothing in this function escalates
    // beyond it - so its own answer is otherwise trusted unconditionally.
    // But a genuinely empty completion is not an answer, and letting one
    // through meant a hollow Reflection got created, admitted to Working
    // Memory, and eventually spoken as literal silence (confirmed live: a
    // `qwen3.5:9b` call returned "" and Omega "said" nothing while still
    // registering it as a successful Speak).
    //
    // A single retry first, though - also confirmed live (a real user
    // question, see `steps::act`'s `SilentReason::ReflectionFailed` doc
    // comment): an empty completion from a real local model is often a
    // one-off flake in the generation itself, not a systemic problem with
    // the prompt or the model, and immediately trying the exact same request
    // again routinely succeeds. This is deliberately narrow - only the
    // "answered, but with nothing" case retries; a genuine transport/timeout
    // `TierError` from `run_chat` still propagates via `?` immediately,
    // unretried, since a retry is far less likely to fix a connection that's
    // actually down and this is the guaranteed-backstop tier - piling a
    // retry loop onto an already-failing call risks compounding the exact
    // latency spike `reflect`'s own doc comment already warns can decay a
    // Reflection out of Working Memory before it ever gets a shot at
    // broadcast.
    if response.raw_text.trim().is_empty() {
        response = tier3_pool.run_chat(tier3_client, req).await?;
    }
    // Still empty after the retry: treating this as the same
    // `MalformedResponse` any other unusable completion already produces
    // means the caller's existing `Err(_) => ActOutcome::Silent` handles it
    // for free - no new error-handling path needed.
    if response.raw_text.trim().is_empty() {
        return Err(TierError::MalformedResponse { tier: response.tier, reason: "completion was empty (after one retry)".to_string() });
    }
    Ok(response)
}

/// Samples `pool` (repeat-sampling up to `MIN_SAMPLES` if fewer clients are
/// configured - see `DivergentPool::sample`), embeds every successful
/// candidate, and arbitrates between them by agreement (`steps::arbitrate`).
/// Returns `Some` only when at least two candidates had a usable embedding
/// *and* their agreement clears `threshold` - a lone surviving candidate
/// (every repeat-sample but one failed, or embedding failed for the rest)
/// is never trusted on its own, since there's nothing to corroborate it
/// against. Any failure along the way (pool exhausted, nothing embeddable)
/// is `None`, meaning "escalate," not an error - this tier simply didn't
/// earn the answer.
async fn try_tier_via_agreement(
    pool: &DivergentPool,
    embedding_client: &dyn EmbeddingClient,
    req: &GenerateRequest,
    threshold: f32,
) -> Option<TierResponse> {
    let candidates = pool.sample(req.clone(), MIN_SAMPLES).await.ok()?;

    // An empty completion is never usable corroboration - without this,
    // several models all independently returning "" would embed identically
    // and register as *maximal* agreement, the exact opposite of what
    // agreement is supposed to mean here.
    let non_empty: Vec<TierResponse> = candidates.into_iter().filter(|c| !c.raw_text.trim().is_empty()).collect();
    // Embedding each candidate is otherwise-independent I/O - fanning these
    // out (instead of one sequential `.await` per candidate in a `for` loop)
    // saves up to `MIN_SAMPLES - 1` round trips to the embedding server on
    // every tier rung this function is invoked for.
    let embed_results = futures::future::join_all(non_empty.iter().map(|c| embedding_client.embed(&c.raw_text))).await;

    let mut usable = Vec::new();
    let mut embeddings = Vec::new();
    for (candidate, embed_result) in non_empty.into_iter().zip(embed_results) {
        if let Ok(embedding) = embed_result {
            embeddings.push(embedding);
            usable.push(candidate);
        }
    }

    let outcome = arbitrate_by_agreement(usable, &embeddings)?;
    if outcome.sample_count >= 2 && outcome.agreement >= threshold {
        // Overwrite the winning candidate's own self-reported confidence
        // with the real, cross-validated agreement score - this is the
        // whole reason this function exists (see this module's top-level
        // doc comment), so the confidence that ultimately lands on the
        // stored Reflection (`reflect`'s `reflection.confidence =
        // response.confidence`) should be the honest signal too, not the
        // self-report this function was built specifically to stop
        // trusting. `steps::executive::propose_operators` depends on this
        // being trustworthy to decide whether a Reflection is solid enough
        // to actually speak.
        let mut chosen = outcome.chosen;
        chosen.confidence = outcome.agreement;
        Some(chosen)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aca_util::{EpochMillis, ManualClock};
    use async_trait::async_trait;
    use std::time::Duration;

    struct FixedChatClient {
        raw_text: String,
        confidence: f32,
        tier: aca_types::Tier,
    }

    #[async_trait]
    impl ChatClient for FixedChatClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            // A real adapter (ollama.rs/openai_compat.rs) already ran the
            // completion through `parse_tier_response` before returning -
            // this test double returns the already-parsed fields directly,
            // matching the trait's actual contract.
            Ok(TierResponse {
                raw_text: self.raw_text.clone(),
                confidence: self.confidence,
                tier: self.tier,
            })
        }
    }

    fn empty_tier1() -> DivergentPool {
        DivergentPool::new(aca_types::Tier::T1, Duration::from_secs(5), vec![])
    }

    fn empty_tier2() -> DivergentPool {
        DivergentPool::new(aca_types::Tier::T2, Duration::from_secs(5), vec![])
    }

    /// `n` clients that all return the same `raw_text` - simulates
    /// independent samples that actually agree with each other.
    fn agreeing_pool(tier: aca_types::Tier, raw_text: &str, n: usize) -> DivergentPool {
        let clients = (0..n)
            .map(|_| std::sync::Arc::new(FixedChatClient { raw_text: raw_text.to_string(), confidence: 0.99, tier }) as std::sync::Arc<dyn ChatClient>)
            .collect();
        DivergentPool::new(tier, Duration::from_secs(5), clients)
    }

    /// `texts.len()` clients, each returning different text - simulates the
    /// observed real failure: independent samples that don't corroborate
    /// each other, no matter how confident each one claims to be.
    fn disagreeing_pool(tier: aca_types::Tier, texts: &[&str]) -> DivergentPool {
        let clients = texts
            .iter()
            .map(|t| std::sync::Arc::new(FixedChatClient { raw_text: t.to_string(), confidence: 0.99, tier }) as std::sync::Arc<dyn ChatClient>)
            .collect();
        DivergentPool::new(tier, Duration::from_secs(5), clients)
    }

    fn tier3_client(raw_text: &str, confidence: f32) -> FixedChatClient {
        FixedChatClient { raw_text: raw_text.to_string(), confidence, tier: aca_types::Tier::T3 }
    }

    /// Empty on its first call, `recovered_text` on every call after -
    /// simulates the documented live flake this retry exists for (a single
    /// bad generation, not a systemically broken model/prompt).
    struct FlakyThenRecoversChatClient {
        recovered_text: &'static str,
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl ChatClient for FlakyThenRecoversChatClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            let call_number = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let raw_text = if call_number == 0 { String::new() } else { self.recovered_text.to_string() };
            Ok(TierResponse { raw_text, confidence: 0.8, tier: aca_types::Tier::T3 })
        }
    }

    #[tokio::test]
    async fn reflection_carries_provenance_and_tier_used() {
        let tier1 = empty_tier1();
        let tier2 = empty_tier2();
        let tier3 = TierPool::new(aca_types::Tier::T3, 1, Duration::from_secs(5));
        let client = tier3_client("that seems significant", 0.85);
        let broadcast_object = MentalObject::new_observation("something surprising", EpochMillis(0), 0.5);
        let broadcast_id = broadcast_object.id;

        let clock = ManualClock::new(EpochMillis(1_000));
        let embedding_client = aca_tiers::testing::FakeEmbeddingClient::default();
        let output = reflect(&tier1, &tier2, &tier3, &client, &embedding_client, &broadcast_object, &[], "", 0.0, &clock, 0.5, Duration::from_secs(5), 0.5, None)
            .await
            .unwrap();

        assert_eq!(output.reflection.text, "that seems significant");
        assert_eq!(output.reflection.kind, MentalObjectKind::Reflection);
        assert!((output.reflection.confidence - 0.85).abs() < 1e-6);
        assert_eq!(output.reflection.tier_used, Some(aca_types::Tier::T3));
        assert_eq!(output.reflection.source_object_ids, vec![broadcast_id]);
    }

    /// Captures the exact `GenerateRequest.prompt` Tier 3 actually received -
    /// the only way to prove `displacement_note` genuinely reaches the real
    /// prompt through the whole tier-ladder call, rather than merely being
    /// accepted as a parameter and silently dropped somewhere on the way to
    /// `reflect_prompt`.
    struct PromptCapturingChatClient {
        raw_text: String,
        captured_prompt: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    }

    #[async_trait]
    impl ChatClient for PromptCapturingChatClient {
        async fn generate(&self, req: GenerateRequest) -> Result<TierResponse, TierError> {
            *self.captured_prompt.lock().unwrap() = Some(req.prompt);
            Ok(TierResponse { raw_text: self.raw_text.clone(), confidence: 0.9, tier: aca_types::Tier::T3 })
        }
    }

    #[tokio::test]
    async fn a_real_displacement_note_reaches_the_actual_tier3_prompt() {
        let tier1 = empty_tier1();
        let tier2 = empty_tier2();
        let tier3 = TierPool::new(aca_types::Tier::T3, 1, Duration::from_secs(5));
        let captured_prompt = std::sync::Arc::new(std::sync::Mutex::new(None));
        let client = PromptCapturingChatClient { raw_text: "noted".to_string(), captured_prompt: captured_prompt.clone() };
        let broadcast_object = MentalObject::new_observation("someone is at the front door", EpochMillis(0), 0.5);
        let clock = ManualClock::new(EpochMillis(1_000));
        let embedding_client = aca_tiers::testing::FakeEmbeddingClient::default();

        reflect(
            &tier1,
            &tier2,
            &tier3,
            &client,
            &embedding_client,
            &broadcast_object,
            &[],
            "",
            0.0,
            &clock,
            0.5,
            Duration::from_secs(5),
            0.5,
            Some("the kettle is boiling"),
        )
        .await
        .unwrap();

        let prompt = captured_prompt.lock().unwrap().clone().expect("Tier 3 should have been called and its prompt captured");
        assert!(prompt.contains("the kettle is boiling"), "the real displacement note should reach the actual prompt sent to the model, not just be accepted and dropped");
    }

    #[tokio::test]
    async fn an_empty_tier3_completion_fails_reflection_instead_of_being_spoken_as_silence() {
        // Regression test for a real live failure: qwen3.5:9b occasionally
        // returned an empty completion, which previously became a hollow
        // Reflection, got selected for Speak, and was rendered/printed as
        // literal silence - a "successful" turn that said nothing.
        let tier1 = empty_tier1();
        let tier2 = empty_tier2();
        let tier3 = TierPool::new(aca_types::Tier::T3, 1, Duration::from_secs(5));
        let client = tier3_client("", 0.5);
        let broadcast_object = MentalObject::new_observation("what time is it?", EpochMillis(0), 0.5);
        let clock = ManualClock::new(EpochMillis(1_000));
        let embedding_client = aca_tiers::testing::FakeEmbeddingClient::default();

        let result = reflect(&tier1, &tier2, &tier3, &client, &embedding_client, &broadcast_object, &[], "", 0.0, &clock, 0.5, Duration::from_secs(5), 0.5, None).await;

        let Err(err) = result else { panic!("an empty Tier 3 completion should fail reflection, not produce a hollow Reflection") };
        assert!(err.to_string().contains("after one retry"), "an always-empty client should still fail only after the retry has actually been attempted, got: {err}");
    }

    #[tokio::test]
    async fn a_tier3_completion_that_is_empty_only_once_recovers_on_retry() {
        // The actual point of the retry: a real local model's single blank
        // generation (confirmed live, not hypothetical - see `steps::act`'s
        // `SilentReason::ReflectionFailed` doc comment for the live incident
        // this whole visibility+resilience pair of fixes was built for)
        // should not fail the whole reflection when trying again would have
        // worked fine.
        let tier1 = empty_tier1();
        let tier2 = empty_tier2();
        let tier3 = TierPool::new(aca_types::Tier::T3, 1, Duration::from_secs(5));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let client = FlakyThenRecoversChatClient { recovered_text: "a real answer on the second try", calls: calls.clone() };
        let broadcast_object = MentalObject::new_observation("are you there?", EpochMillis(0), 0.5);
        let clock = ManualClock::new(EpochMillis(1_000));
        let embedding_client = aca_tiers::testing::FakeEmbeddingClient::default();

        let output = reflect(&tier1, &tier2, &tier3, &client, &embedding_client, &broadcast_object, &[], "", 0.0, &clock, 0.5, Duration::from_secs(5), 0.5, None)
            .await
            .expect("a completion that recovers on retry should succeed, not fail the whole reflection");

        assert_eq!(output.reflection.text, "a real answer on the second try");
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2, "should have retried exactly once after the empty first attempt");
    }

    #[tokio::test]
    async fn a_whitespace_only_tier3_completion_also_fails_reflection() {
        let tier1 = empty_tier1();
        let tier2 = empty_tier2();
        let tier3 = TierPool::new(aca_types::Tier::T3, 1, Duration::from_secs(5));
        let client = tier3_client("   \n  ", 0.5);
        let broadcast_object = MentalObject::new_observation("hello?", EpochMillis(0), 0.5);
        let clock = ManualClock::new(EpochMillis(1_000));
        let embedding_client = aca_tiers::testing::FakeEmbeddingClient::default();

        let result = reflect(&tier1, &tier2, &tier3, &client, &embedding_client, &broadcast_object, &[], "", 0.0, &clock, 0.5, Duration::from_secs(5), 0.5, None).await;

        assert!(result.is_err(), "whitespace-only counts as empty, not real content");
    }

    #[tokio::test]
    async fn unanimous_empty_tier1_candidates_do_not_count_as_agreement() {
        // Without filtering empty completions before arbitration, several
        // models all independently returning "" would embed identically and
        // register as *maximal* agreement - the opposite of what agreement
        // is supposed to mean. This should fall through to Tier 3 instead.
        let tier1 = agreeing_pool(aca_types::Tier::T1, "", 3);
        let tier2 = empty_tier2();
        let tier3 = TierPool::new(aca_types::Tier::T3, 1, Duration::from_secs(5));
        let client = tier3_client("a real answer from the backstop", 0.8);
        let broadcast_object = MentalObject::new_observation("something", EpochMillis(0), 0.5);
        let clock = ManualClock::new(EpochMillis(1_000));
        let embedding_client = aca_tiers::testing::FakeEmbeddingClient::default();

        let output = reflect(&tier1, &tier2, &tier3, &client, &embedding_client, &broadcast_object, &[], "", 0.0, &clock, 0.5, Duration::from_secs(5), 0.5, None)
            .await
            .unwrap();

        assert_eq!(output.reflection.text, "a real answer from the backstop");
        assert_eq!(output.reflection.tier_used, Some(aca_types::Tier::T3));
    }

    #[tokio::test]
    async fn reflection_gets_a_resolved_embedding_so_it_can_later_be_remembered() {
        // Regression test for a real runaway loop found live: a Reflection
        // with no embedding is discarded forever by `form_memory`, which
        // means `already_remembered` never becomes true, which means
        // `propose_operators` proposes `Remember` for it on every
        // subsequent tick indefinitely.
        let tier1 = empty_tier1();
        let tier2 = empty_tier2();
        let tier3 = TierPool::new(aca_types::Tier::T3, 1, Duration::from_secs(5));
        let client = tier3_client("a memorable reflection", 0.8);
        let broadcast_object = MentalObject::new_observation("something", EpochMillis(0), 0.5);
        let clock = ManualClock::new(EpochMillis(1_000));
        let embedding_client = aca_tiers::testing::FakeEmbeddingClient::default();

        let output = reflect(&tier1, &tier2, &tier3, &client, &embedding_client, &broadcast_object, &[], "", 0.0, &clock, 0.5, Duration::from_secs(5), 0.5, None)
            .await
            .unwrap();

        assert!(output.reflection.embedding.is_some(), "a Reflection should be born with a resolved embedding");
    }

    #[tokio::test]
    async fn tier2_candidates_that_agree_short_circuit_tier3() {
        let tier1 = empty_tier1();
        let tier2 = agreeing_pool(aca_types::Tier::T2, "confident enough", 3);
        let tier3 = TierPool::new(aca_types::Tier::T3, 1, Duration::from_secs(5));
        // If the ladder ever fell through to Tier 3 despite Tier 2 agreeing
        // with itself, this would be what got spoken instead.
        let client = tier3_client("expensive fallback that should never be used", 0.99);
        let broadcast_object = MentalObject::new_observation("something", EpochMillis(0), 0.5);
        let clock = ManualClock::new(EpochMillis(1_000));
        let embedding_client = aca_tiers::testing::FakeEmbeddingClient::default();

        let output = reflect(&tier1, &tier2, &tier3, &client, &embedding_client, &broadcast_object, &[], "", 0.0, &clock, 0.5, Duration::from_secs(5), 0.5, None)
            .await
            .unwrap();

        assert_eq!(output.reflection.text, "confident enough");
        assert_eq!(output.reflection.tier_used, Some(aca_types::Tier::T2));
    }

    #[tokio::test]
    async fn a_tier2_agreement_win_reports_agreement_as_confidence_not_the_self_report() {
        // agreeing_pool's clients all self-report confidence 0.99 (see its
        // own doc comment) - a deterministic embedding client makes three
        // identical candidates agree at exactly 1.0, a value that could
        // never come from copying a self-reported 0.99 through unchanged.
        let tier1 = empty_tier1();
        let tier2 = agreeing_pool(aca_types::Tier::T2, "confident enough", 3);
        let tier3 = TierPool::new(aca_types::Tier::T3, 1, Duration::from_secs(5));
        let client = tier3_client("should never be used", 0.5);
        let broadcast_object = MentalObject::new_observation("something", EpochMillis(0), 0.5);
        let clock = ManualClock::new(EpochMillis(1_000));
        let embedding_client = aca_tiers::testing::FakeEmbeddingClient::default();

        let output = reflect(&tier1, &tier2, &tier3, &client, &embedding_client, &broadcast_object, &[], "", 0.0, &clock, 0.5, Duration::from_secs(5), 0.5, None)
            .await
            .unwrap();

        assert!((output.reflection.confidence - 1.0).abs() < 1e-5, "identical candidates should agree at ~1.0, got {}", output.reflection.confidence);
    }

    #[tokio::test]
    async fn tier2_candidates_that_disagree_fall_through_to_tier3() {
        let tier1 = empty_tier1();
        // Same shape as the real Tier 1 failure this whole mechanism exists
        // to catch: every candidate confidently claims 0.99, but they don't
        // actually agree on anything.
        let tier2 = disagreeing_pool(aca_types::Tier::T2, &["I am a Super Mario character", "the weather is nice today", "purple elephants dance slowly"]);
        let tier3 = TierPool::new(aca_types::Tier::T3, 1, Duration::from_secs(5));
        let client = tier3_client("the deliberate answer", 0.9);
        let broadcast_object = MentalObject::new_observation("something", EpochMillis(0), 0.5);
        let clock = ManualClock::new(EpochMillis(1_000));
        let embedding_client = aca_tiers::testing::FakeEmbeddingClient::default();

        let output = reflect(&tier1, &tier2, &tier3, &client, &embedding_client, &broadcast_object, &[], "", 0.0, &clock, 0.5, Duration::from_secs(5), 0.5, None)
            .await
            .unwrap();

        assert_eq!(output.reflection.text, "the deliberate answer");
        assert_eq!(output.reflection.tier_used, Some(aca_types::Tier::T3));
    }

    /// A single client whose successive calls cycle through different
    /// canned responses - simulates a real model, sampled repeatedly at
    /// temperature > 0, that doesn't actually say the same thing twice.
    /// `FixedChatClient` (deterministic, always identical output) can't
    /// represent this: repeat-sampling it trivially "agrees with itself,"
    /// which is a meaningfully different scenario from a real model that
    /// genuinely diverges across resamples.
    struct SequenceClient {
        responses: Vec<&'static str>,
        confidence: f32,
        tier: aca_types::Tier,
        next: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl ChatClient for SequenceClient {
        async fn generate(&self, _req: GenerateRequest) -> Result<TierResponse, TierError> {
            let i = self.next.fetch_add(1, std::sync::atomic::Ordering::SeqCst) % self.responses.len();
            Ok(TierResponse { raw_text: self.responses[i].to_string(), confidence: self.confidence, tier: self.tier })
        }
    }

    #[tokio::test]
    async fn a_single_models_divergent_resamples_still_escalate() {
        // Regression test for the exact original failure mode: one small
        // model, self-reporting maximal confidence, with nothing to
        // corroborate it - now exercised through repeat-sampling (the only
        // client configured at this tier gets called `MIN_SAMPLES` times)
        // rather than multiple distinct clients. Divergent resamples must
        // still fail to earn trust no matter how confident each one claims
        // to be.
        let tier1 = empty_tier1();
        let tier2 = DivergentPool::new(
            aca_types::Tier::T2,
            Duration::from_secs(5),
            vec![std::sync::Arc::new(SequenceClient {
                responses: vec!["I am a Super Mario character", "the weather is nice today", "purple elephants dance slowly"],
                confidence: 1.0,
                tier: aca_types::Tier::T2,
                next: std::sync::atomic::AtomicUsize::new(0),
            })],
        );
        let tier3 = TierPool::new(aca_types::Tier::T3, 1, Duration::from_secs(5));
        let client = tier3_client("the deliberate answer", 0.9);
        let broadcast_object = MentalObject::new_observation("hi, can you hear me?", EpochMillis(0), 0.5);
        let clock = ManualClock::new(EpochMillis(1_000));
        let embedding_client = aca_tiers::testing::FakeEmbeddingClient::default();

        let output = reflect(&tier1, &tier2, &tier3, &client, &embedding_client, &broadcast_object, &[], "", 0.0, &clock, 0.5, Duration::from_secs(5), 0.5, None)
            .await
            .unwrap();

        assert_eq!(output.reflection.text, "the deliberate answer");
        assert_eq!(output.reflection.tier_used, Some(aca_types::Tier::T3));
    }

    #[tokio::test]
    async fn a_single_models_consistent_resamples_are_trusted() {
        // The other side of repeat-sampling: a solo-model tier isn't
        // permanently unconfirmable. If its repeated samples genuinely
        // agree, that's real corroboration and the tier resolves without
        // escalating - exactly the tradeoff the repeat-sample design makes.
        let tier1 = agreeing_pool(aca_types::Tier::T1, "hi, yes, I can hear you", 1);
        let tier2 = empty_tier2();
        let tier3 = TierPool::new(aca_types::Tier::T3, 1, Duration::from_secs(5));
        let client = tier3_client("should never be reached", 0.99);
        let broadcast_object = MentalObject::new_observation("hi, can you hear me?", EpochMillis(0), 0.5);
        let clock = ManualClock::new(EpochMillis(1_000));
        let embedding_client = aca_tiers::testing::FakeEmbeddingClient::default();

        let output = reflect(&tier1, &tier2, &tier3, &client, &embedding_client, &broadcast_object, &[], "", 0.0, &clock, 0.5, Duration::from_secs(5), 0.5, None)
            .await
            .unwrap();

        assert_eq!(output.reflection.text, "hi, yes, I can hear you");
        assert_eq!(output.reflection.tier_used, Some(aca_types::Tier::T1));
    }

    #[tokio::test]
    async fn tier1_candidates_that_agree_resolve_without_ever_reaching_tier2_or_3() {
        let tier1 = agreeing_pool(aca_types::Tier::T1, "hi, yes, I can hear you", 3);
        let tier2 = DivergentPool::new(
            aca_types::Tier::T2,
            Duration::from_secs(5),
            vec![std::sync::Arc::new(FixedChatClient { raw_text: "should never be reached".to_string(), confidence: 0.99, tier: aca_types::Tier::T2 })],
        );
        let tier3 = TierPool::new(aca_types::Tier::T3, 1, Duration::from_secs(5));
        let client = tier3_client("should also never be reached", 0.99);
        let broadcast_object = MentalObject::new_observation("hi, can you hear me?", EpochMillis(0), 0.5);
        let clock = ManualClock::new(EpochMillis(1_000));
        let embedding_client = aca_tiers::testing::FakeEmbeddingClient::default();

        let output = reflect(&tier1, &tier2, &tier3, &client, &embedding_client, &broadcast_object, &[], "", 0.0, &clock, 0.5, Duration::from_secs(5), 0.5, None)
            .await
            .unwrap();

        assert_eq!(output.reflection.text, "hi, yes, I can hear you");
        assert_eq!(output.reflection.tier_used, Some(aca_types::Tier::T1));
    }

    #[tokio::test]
    async fn unconfigured_cheap_tiers_are_skipped_not_treated_as_a_failure() {
        let tier1 = empty_tier1();
        let tier2 = empty_tier2();
        let tier3 = TierPool::new(aca_types::Tier::T3, 1, Duration::from_secs(5));
        let client = tier3_client("only tier 3 is configured", 0.5);
        let broadcast_object = MentalObject::new_observation("something", EpochMillis(0), 0.5);
        let clock = ManualClock::new(EpochMillis(1_000));
        let embedding_client = aca_tiers::testing::FakeEmbeddingClient::default();

        let output = reflect(&tier1, &tier2, &tier3, &client, &embedding_client, &broadcast_object, &[], "", 0.0, &clock, 0.5, Duration::from_secs(5), 0.5, None)
            .await
            .unwrap();

        assert_eq!(output.reflection.text, "only tier 3 is configured");
    }
}
