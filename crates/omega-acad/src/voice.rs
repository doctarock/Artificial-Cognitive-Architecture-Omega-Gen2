//! A thin client for the separate "Voice Interaction" service
//! (`E:\AI\Voice Interaction`), which owns real audio capture (sounddevice),
//! ASR (whisper.cpp), TTS synthesis (Piper), and all playback-package/
//! transcript bookkeeping - this module only ever calls its existing HTTP
//! API, never reimplements or manages any of that. Deliberately a separate
//! process, not something this daemon spawns/manages - same philosophy as
//! the Godot visualization being its own process: the cognitive loop must
//! keep running whether or not anything downstream (a viewer, a voice) is
//! actually attached.
//!
//! Two loops, both just HTTP calls to the same service:
//! - Output: `POST /room/speak` synthesizes and returns audio for text
//!   Omega already decided to say, without running that service's own
//!   ASR/LLM pipeline (which would be redundant - Omega already did the
//!   thinking).
//! - Input: `GET /sync?after={cursor}` is polled for the `speech.utterance`
//!   events the service's own capture/VAD/Whisper pipeline produces on its
//!   own, with no client needed to drive it; anything not tagged as Omega's
//!   own voice gets forwarded into the cognitive loop's input channel. This
//!   module never touches `/capture/*` or `/ingest/capture` - capture is
//!   entirely the voice service's own concern now, not something a client
//!   starts, stops, or paces.

use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// `None` fields mean "voice output isn't configured" - callers treat a
/// missing `VoiceClient` as a no-op, not an error, mirroring how every
/// other optional integration in this daemon degrades (no Tier 3 endpoint,
/// no embedding endpoint, etc.).
pub struct VoiceClient {
    http: reqwest::Client,
    base_url: String,
    speaker: String,
    conversation_id: String,
    /// Wall-clock (Unix epoch ms) `(start, end)` of the most recent `speak`
    /// call - `poll_utterances` drops any utterance whose event timestamp
    /// falls near this window, regardless of `speaker_label`. Necessary,
    /// not just belt-and-suspenders: confirmed live, the voice service's
    /// own diarization gives Omega's own TTS bleeding back into the mic a
    /// *new* fake "Unknown speaker" identity each time rather than ever
    /// labeling it "Omega," so a `speaker_label`-only filter alone lets a
    /// feedback loop through (an utterance paraphrasing what Omega had just
    /// said arrived moments later as if a person said it, and degraded into
    /// nonsense within a couple of exchanges before this existed). Capture
    /// itself is entirely the voice service's own concern now - this is
    /// purely post-hoc filtering of what it independently produces, not
    /// anything that mutes or paces its capture.
    last_speak_window: Mutex<Option<(u64, u64)>>,
}

impl VoiceClient {
    /// Guard window added after `last_speak_window`'s end when filtering
    /// echo out of `/sync` - room echo/reverb tail and the mic's own
    /// settling time mean bleed-through can arrive a beat after playback
    /// actually ends, not just up to the exact millisecond it does.
    const POST_SPEECH_COOLDOWN: Duration = Duration::from_millis(900);

    pub fn new(base_url: String, speaker: String, conversation_id: String) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("reqwest client builds with only a timeout configured"),
            base_url,
            speaker,
            conversation_id,
            last_speak_window: Mutex::new(None),
        }
    }

    fn epoch_ms_now() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
    }

    /// Synthesizes `text` via the voice service and plays it synchronously
    /// (relative to the caller) - callers that don't want to block on
    /// playback should spawn this themselves. Deliberately sequential, not
    /// fire-and-forget-per-call: Omega speaking two things at once over
    /// itself would be worse than a short queue delay.
    ///
    /// Records the actual start/end of this call as `last_speak_window`
    /// regardless of success or failure, so `poll_utterances` can filter
    /// out any mic bleed-through of this playback from `/sync` afterward -
    /// see `last_speak_window`'s doc comment for why that's needed.
    pub async fn speak(&self, text: &str) -> anyhow::Result<()> {
        let speak_start_ms = Self::epoch_ms_now();
        let result = self.speak_and_play(text).await;
        let speak_end_ms = Self::epoch_ms_now();
        *self.last_speak_window.lock().expect("last_speak_window mutex is never poisoned") = Some((speak_start_ms, speak_end_ms));
        result
    }

    async fn speak_and_play(&self, text: &str) -> anyhow::Result<()> {
        let body = serde_json::json!({
            "text": text,
            "speaker": self.speaker,
            "conversation_id": self.conversation_id,
        });
        let response = self
            .http
            .post(format!("{}/room/speak", self.base_url))
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        let response: serde_json::Value = response.json().await?;

        let package = response
            .get("playback_package")
            .ok_or_else(|| anyhow::anyhow!("voice service response missing playback_package"))?;
        let hex = package
            .pointer("/pcm_payload/value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("voice service response missing pcm_payload.value"))?;
        let sample_rate_hz = package.get("sample_rate_hz").and_then(|v| v.as_u64()).unwrap_or(16_000) as u32;

        let pcm_bytes = decode_hex(hex)?;
        let samples: Vec<i16> = pcm_bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();

        // rodio's output stream is not meant to cross an .await point - the
        // entire play-and-block-until-done sequence runs on one blocking
        // thread instead, keeping this async fn's own executor thread free.
        tokio::task::spawn_blocking(move || play_pcm(samples, sample_rate_hz)).await??;
        Ok(())
    }

    /// Spawns a detached background task that polls the voice service's
    /// `GET /sync?after={cursor}` feed for new `speech.utterance` events and
    /// forwards anything *not* spoken by Omega itself into `room_input_tx`
    /// as a `RoomInput` - unlike the plain-text `input_tx` stdin/the local
    /// HTTP API feed, this preserves per-utterance stream/speaker
    /// provenance (see `RoomInput`'s own doc comment for why: this is a
    /// shared room feed, not a single-party channel, and the cognitive loop
    /// needs to know which utterances belong together to avoid blending
    /// different speakers into one Observation). Filtering on
    /// `speaker_label == self.speaker` is what stops Omega hearing its own
    /// TTS output (which the same service also records as a
    /// "speech.utterance" event, tagged `transcript_engine:
    /// "speaker-output"`) and re-ingesting its own words as if a person had
    /// said them.
    pub fn spawn_listener(self: std::sync::Arc<Self>, room_input_tx: tokio::sync::mpsc::Sender<aca_engine::RoomInput>) {
        tokio::spawn(async move {
            // Starts from the service's *current* tip, never from 0 - a
            // fresh restart used to unconditionally replay its entire
            // historical event log (every utterance from every previous
            // session) straight into the cognitive loop. Confirmed live:
            // each replayed utterance needs its own embedding resolved,
            // and a large backlog against a busy/shared embedding server
            // can take many multiples of `OMEGA_EMBEDDING_TIMEOUT_SECS` to
            // grind through sequentially - `cycle_seq` visibly crawled
            // forward roughly one tick per timeout window, looking
            // indistinguishable from a genuine deadlock, with fresh input
            // stuck behind the entire queue. `fetch_current_cursor`'s own
            // retry handles the ordinary startup race (this daemon coming
            // up slightly before the voice service does) without falling
            // back to the history-replaying `0` for what's really just a
            // few seconds of impatience.
            let mut cursor: u64 = match self.fetch_current_cursor().await {
                Ok(cursor) => {
                    tracing::info!(cursor, "voice listener starting from the service's current tip, not replaying its history");
                    cursor
                }
                Err(err) => {
                    tracing::warn!(error = %err, "could not learn the voice service's current cursor after retrying - falling back to 0, which WILL replay its full history");
                    0
                }
            };
            loop {
                match self.poll_utterances(cursor).await {
                    Ok((next_cursor, utterances)) => {
                        cursor = next_cursor;
                        for utterance in utterances {
                            tracing::info!(text = %utterance.text, speaker_label = ?utterance.speaker_label, "heard speech, forwarding to the cognitive loop");
                            if room_input_tx.send(utterance).await.is_err() {
                                tracing::error!("cognitive loop is no longer accepting input; stopping voice input listener");
                                return;
                            }
                        }
                    }
                    Err(err) => tracing::warn!(error = %err, "voice input poll failed, retrying"),
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        });
    }

    /// Retry/backoff for `fetch_current_cursor` - covers the ordinary
    /// startup race (this daemon coming up slightly before the voice
    /// service does) without waiting so long that a genuinely unreachable
    /// or misconfigured service holds up the listener forever. ~10 seconds
    /// total, comfortably past the couple of seconds a normal co-started
    /// service takes to bind its port.
    const INITIAL_CURSOR_FETCH_ATTEMPTS: u32 = 10;
    const INITIAL_CURSOR_FETCH_RETRY_DELAY: Duration = Duration::from_secs(1);

    /// Learns the voice service's current event-log tip *without*
    /// forwarding any of its history into the cognitive loop - see
    /// `spawn_listener`'s own doc comment for the flood this exists to
    /// prevent. `compact=true` (unlike every other `/sync` call this client
    /// makes) since only `next_event` is read here; the events array itself
    /// is discarded either way, so there's no reason to pay for the fuller
    /// payload shape.
    async fn fetch_current_cursor(&self) -> anyhow::Result<u64> {
        let mut last_err = None;
        for attempt in 0..Self::INITIAL_CURSOR_FETCH_ATTEMPTS {
            if attempt > 0 {
                tokio::time::sleep(Self::INITIAL_CURSOR_FETCH_RETRY_DELAY).await;
            }
            let response: anyhow::Result<serde_json::Value> = async {
                Ok(self
                    .http
                    .get(format!("{}/sync", self.base_url))
                    .query(&[("after", "0"), ("compact", "true")])
                    .send()
                    .await?
                    .error_for_status()?
                    .json()
                    .await?)
            }
            .await;
            match response.and_then(|body| body.get("next_event").and_then(|v| v.as_u64()).ok_or_else(|| anyhow::anyhow!("response missing next_event"))) {
                Ok(cursor) => return Ok(cursor),
                Err(err) => last_err = Some(err),
            }
        }
        Err(last_err.expect("the attempts loop runs at least once, always setting last_err on failure"))
    }

    /// Generous margin for how late a `speech.utterance` event tagged to a
    /// chunk that started before `speak` began can still arrive - the voice
    /// service's own capture/VAD paces itself now, so this is a fixed
    /// estimate rather than derived from any client-side chunk timing.
    const ECHO_GUARD_BEFORE_MS: u64 = 10_000;

    /// Whether `label` is a real, named/enrolled speaker identity rather
    /// than the voice service's own default for a not-yet-enrolled voice.
    /// Verified live against this service's actual behavior
    /// (`voice_interaction/core.py`'s `label = ... or "Unknown speaker"`
    /// default, `services.py`'s `rename_track`): an
    /// enrolled label like "alex" maps to exactly one underlying
    /// `speaker_track_id` for an entire session, while every "Unknown
    /// speaker N" label maps to several *different* `speaker_track_id`s
    /// over the same window - it's a rotating bucket assigned to
    /// not-yet-enrolled voices, reused across different real people, never
    /// a stable identity. `RoomInput::speaker_label` must never carry one
    /// of these - see that field's doc comment.
    fn is_enrolled_label(label: &str) -> bool {
        let trimmed = label.trim();
        !trimmed.is_empty() && !trimmed.to_lowercase().starts_with("unknown speaker")
    }

    /// Whisper's own combined hallucination heuristic (community-standard,
    /// the same one `openai-whisper`'s own decoder uses to flag a segment
    /// worth discarding): a segment predicted with near-certainty that no
    /// one was even speaking (`no_speech_prob`), AND whose token sequence
    /// the model itself was unconfident about (`avg_logprob`), is very
    /// unlikely to be real transcribed speech - see `docs/room-audio-
    /// quality-gate.md` for the live gibberish (`"and malicious but it can
    /// take its course with a view that"`) this exists to keep out of
    /// Working Memory before it ever competes for Broadcast.
    const NO_SPEECH_PROB_THRESHOLD: f64 = 0.6;
    const AVG_LOGPROB_THRESHOLD: f64 = -1.0;
    /// A second, independent tell: whisper.cpp's own repetition-loop
    /// failure mode (the decoder gets stuck re-emitting near-identical
    /// text) drives the ratio of raw text length to its zlib-compressed
    /// length way up, since compression collapses the repetition - high
    /// text entropy from real speech doesn't. Same threshold OpenAI's own
    /// Whisper uses for this.
    const COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;

    /// Whether this utterance's own Whisper-reported signals (when the
    /// voice service actually provides them - see `TranscriptionResult`'s
    /// sibling fields in that service) mark it as a likely hallucination
    /// rather than real speech. `false` (never filtered) whenever a signal
    /// is missing - an older voice-service build, or a transcription path
    /// that never had segment-level data to report (the CLI/subprocess
    /// adapter) - fails open exactly like every other optional integration
    /// in this daemon, rather than silently dropping real utterances a
    /// service update just hasn't reached yet.
    fn looks_hallucinated(payload: &serde_json::Value) -> bool {
        let no_speech_prob = payload.get("no_speech_prob").and_then(|v| v.as_f64());
        let avg_logprob = payload.get("avg_logprob").and_then(|v| v.as_f64());
        let compression_ratio = payload.get("compression_ratio").and_then(|v| v.as_f64());
        if let (Some(no_speech_prob), Some(avg_logprob)) = (no_speech_prob, avg_logprob)
            && no_speech_prob > Self::NO_SPEECH_PROB_THRESHOLD
            && avg_logprob < Self::AVG_LOGPROB_THRESHOLD
        {
            return true;
        }
        if let Some(compression_ratio) = compression_ratio
            && compression_ratio > Self::COMPRESSION_RATIO_THRESHOLD
        {
            return true;
        }
        false
    }

    async fn poll_utterances(&self, cursor: u64) -> anyhow::Result<(u64, Vec<aca_engine::RoomInput>)> {
        let response: serde_json::Value = self
            .http
            .get(format!("{}/sync", self.base_url))
            .query(&[("after", cursor.to_string()), ("compact", "false".to_string())])
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        let next_cursor = response.get("next_event").and_then(|v| v.as_u64()).unwrap_or(cursor);
        let events = response.get("events").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let speak_window = *self.last_speak_window.lock().expect("last_speak_window mutex is never poisoned");

        let utterances = events
            .into_iter()
            .filter(|event| event.get("topic").and_then(|v| v.as_str()) == Some("speech.utterance"))
            .filter(|event| !Self::overlaps_recent_speech(event, speak_window))
            .filter_map(|event| {
                let payload = event.get("payload")?;
                let speaker_label = payload.get("speaker_label").and_then(|v| v.as_str()).unwrap_or("");
                if speaker_label.eq_ignore_ascii_case(&self.speaker) {
                    return None; // Omega hearing itself - not real input.
                }
                let text = payload.get("text").and_then(|v| v.as_str())?.trim();
                if text.is_empty() {
                    return None;
                }
                if Self::looks_hallucinated(payload) {
                    tracing::debug!(text, "dropping a likely-hallucinated transcript before it reaches the cognitive loop");
                    return None;
                }
                // `speaker_track_id` groups this utterance with others from
                // the same acoustic source regardless of whether that
                // source has a real name yet - the raw diarization track,
                // not an identity (see `RoomInput::stream_id`'s doc
                // comment). Falls back to the label itself only if the
                // service ever omits a track id, so coalescing still has
                // *something* stable to key on for this poll.
                let stream_id = payload.get("speaker_track_id").and_then(|v| v.as_str()).unwrap_or(speaker_label).to_string();
                let speaker_label = Self::is_enrolled_label(speaker_label).then(|| speaker_label.to_string());
                Some(aca_engine::RoomInput { text: text.to_string(), speaker_label, stream_id })
            })
            .collect();

        Ok((next_cursor, utterances))
    }

    /// Whether `event`'s own emission timestamp falls close enough to
    /// `speak_window` (the most recent span Omega was actually speaking)
    /// that it's plausibly mic bleed-through rather than real input -
    /// independent of `speaker_label`. Confirmed live: an utterance
    /// paraphrasing a sentence Omega had just spoken ("...focused on the
    /// characters and their experience, with an emphasis on survival and
    /// emotional impact.") arrived moments later labeled "Unknown speaker
    /// 3" with the text "We expect us on the characters and their
    /// experience." - the voice service's own diarization gives echo a
    /// *new* fake identity each time rather than ever recognizing it as
    /// Omega, so a label-only filter can't catch this. The margin extends
    /// backward by `ECHO_GUARD_BEFORE_MS` (a chunk of the service's own,
    /// independently-paced capture can already be in flight when `speak`
    /// starts, and there's no client-side mute to stop it) and forward by
    /// the post-speech cooldown plus a reverb-tail buffer.
    fn overlaps_recent_speech(event: &serde_json::Value, speak_window: Option<(u64, u64)>) -> bool {
        let Some((start_ms, end_ms)) = speak_window else { return false };
        let Some(event_ms) = event.get("timestamp_ms").and_then(|v| v.as_u64()) else { return false };
        let guard_before_ms = Self::ECHO_GUARD_BEFORE_MS;
        let guard_after_ms = Self::POST_SPEECH_COOLDOWN.as_millis() as u64 + 2_000;
        let window_start = start_ms.saturating_sub(guard_before_ms);
        let window_end = end_ms.saturating_add(guard_after_ms);
        event_ms >= window_start && event_ms <= window_end
    }
}

fn decode_hex(hex: &str) -> anyhow::Result<Vec<u8>> {
    if hex.len() % 2 != 0 {
        anyhow::bail!("odd-length hex string ({} chars)", hex.len());
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).map_err(anyhow::Error::from))
        .collect()
}

fn play_pcm(samples: Vec<i16>, sample_rate_hz: u32) -> anyhow::Result<()> {
    let (_stream, stream_handle) = rodio::OutputStream::try_default()?;
    let sink = rodio::Sink::try_new(&stream_handle)?;
    sink.append(rodio::buffer::SamplesBuffer::new(1, sample_rate_hz, samples));
    sink.sleep_until_end();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_enrolled_label_accepts_a_real_name() {
        assert!(VoiceClient::is_enrolled_label("alex"));
    }

    #[test]
    fn looks_hallucinated_flags_high_no_speech_with_low_confidence() {
        let payload = serde_json::json!({"no_speech_prob": 0.85, "avg_logprob": -1.4});
        assert!(VoiceClient::looks_hallucinated(&payload));
    }

    #[test]
    fn looks_hallucinated_flags_a_repetition_loop_via_compression_ratio() {
        let payload = serde_json::json!({"compression_ratio": 3.1});
        assert!(VoiceClient::looks_hallucinated(&payload));
    }

    #[test]
    fn looks_hallucinated_is_false_for_confident_real_speech() {
        let payload = serde_json::json!({"no_speech_prob": 0.05, "avg_logprob": -0.2, "compression_ratio": 1.3});
        assert!(!VoiceClient::looks_hallucinated(&payload));
    }

    #[test]
    fn looks_hallucinated_needs_both_no_speech_and_low_confidence_together() {
        // High no_speech_prob alone (e.g. a brief pause the VAD still
        // chunked) shouldn't condemn a transcript the model was otherwise
        // confident about, and vice versa.
        let high_no_speech_only = serde_json::json!({"no_speech_prob": 0.9, "avg_logprob": -0.1});
        let low_confidence_only = serde_json::json!({"no_speech_prob": 0.1, "avg_logprob": -1.5});
        assert!(!VoiceClient::looks_hallucinated(&high_no_speech_only));
        assert!(!VoiceClient::looks_hallucinated(&low_confidence_only));
    }

    #[test]
    fn looks_hallucinated_fails_open_when_signals_are_missing() {
        // An older voice-service build (or the CLI/subprocess transcription
        // path, which never has segment-level data) simply won't report
        // these fields - that must never be read as "definitely
        // hallucinated," or every utterance from an unupgraded service
        // would get silently dropped.
        assert!(!VoiceClient::looks_hallucinated(&serde_json::json!({})));
    }

    #[test]
    fn is_enrolled_label_rejects_the_default_unenrolled_placeholder() {
        // Verified live: this exact label (with or without a trailing
        // number) maps to several *different* `speaker_track_id`s over a
        // session - it is a rotating bucket, never a stable identity. See
        // `is_enrolled_label`'s own doc comment.
        assert!(!VoiceClient::is_enrolled_label("Unknown speaker"));
        assert!(!VoiceClient::is_enrolled_label("Unknown speaker 11"));
        assert!(!VoiceClient::is_enrolled_label("unknown speaker 3"));
    }

    #[test]
    fn is_enrolled_label_rejects_empty_or_missing_labels() {
        assert!(!VoiceClient::is_enrolled_label(""));
        assert!(!VoiceClient::is_enrolled_label("   "));
    }
}
