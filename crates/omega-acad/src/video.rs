//! A thin client for the separate "Video Interaction" service
//! (`E:\AI\Video Interaction`), which owns camera capture, cheap
//! change/motion detection, stable visual tracks, optional local face
//! recognition, and triggered VLM captioning - this module only ever calls
//! its existing HTTP API, never reimplements or manages any of that.
//! Deliberately a separate process, not something this daemon spawns/
//! manages - same philosophy as `voice::VoiceClient` and the Godot
//! visualization both being their own processes: the cognitive loop must
//! keep running whether or not a camera is actually attached.
//!
//! One loop, just HTTP polling: `GET /sync?after={cursor}` for `scene.change`
//! / `entity.enter` / `entity.exit` events the service's own capture/
//! detection/tracking pipeline produces on its own, forwarded into the
//! cognitive loop as `SensorInput` (`SourceChannel::Environment`,
//! `source: "camera"`) - the exact path `loop_actor.rs`'s own
//! `an_environment_sensor_observation_flows_through_the_same_pipeline_as_conversation`
//! test proves already exists and needs no engine change to use.
//!
//! A recognized `entity_label` (the video service's own face-gallery match)
//! is forwarded as `SensorInput::entity_label`, which folds into the exact
//! same `data_tag["speaker_label"]` key `RoomInput::speaker_label` already
//! uses - see that field's doc comment in `loop_actor.rs`. A person
//! recognized by voice and by camera under the same enrolled name lands on
//! one shared interlocutor node, proven live by
//! `loop_actor::tests::a_camera_recognized_entity_label_resolves_to_the_same_interlocutor_a_voice_enrollment_already_built`.
//! Because a bad identity claim here contaminates a *persistent, shared*
//! memory node rather than one throwaway observation (video-interaction-
//! plan.md section 8.4), this client applies its own confidence floor -
//! `LABEL_TRUST_THRESHOLD` - on top of whatever bar the video service's own
//! gallery match already cleared, and only ever trusts a label sourced from
//! `entity.enter` (the one topic that actually carries a fresh
//! `recognition_confidence`), never `entity.exit`'s echoed-but-unverified
//! copy of it - see `event_observation`'s doc comment.

use std::collections::HashSet;
use std::time::Duration;

use aca_engine::{SensorInput, SourceChannel};
use tokio::sync::mpsc;

pub struct VideoClient {
    http: reqwest::Client,
    base_url: String,
    /// Minimum spacing between forwarded `SensorInput`s, from
    /// `OMEGA_VIDEO_MIN_INTERVAL_MS` - also this client's poll interval
    /// (see `spawn_listener`). A camera can fire far more
    /// `entity.enter`/`entity.exit`/`scene.change` events per minute than a
    /// person could ever speak turns (ordinary foot traffic, a pet, a
    /// flickering light defeating the service's own debounce), and
    /// `loop_actor.rs`'s own test confirms a `SensorInput` competes for
    /// Working Memory admission exactly like a conversational turn, with no
    /// discount for being non-conversational - see plan section 8.1. Rather
    /// than a separate token-bucket, this client simply polls no faster
    /// than this interval and joins everything collected in one poll into a
    /// single `SensorInput`, so the rate limit and the coalescing are the
    /// same mechanism.
    min_interval: Duration,
}

/// Safety cap on how many events one poll's worth of text gets joined
/// into a single `SensorInput` - a pathological burst (well past what
/// `min_interval` coalescing is meant to absorb, e.g. the video service
/// restarting and a large backlog arriving at once) should still produce
/// one bounded observation, not one enormous run-on sentence. Keeps only
/// the most recent events, the same "most temporally relevant" bias
/// `voice.rs`'s cursor-from-tip behavior already applies at reconnect.
const MAX_EVENTS_PER_FORWARD: usize = 8;

/// The confidence floor this client requires before trusting an
/// `entity.enter`'s `recognition_confidence` enough to treat its
/// `entity_label` as a real identity claim (i.e. let it reach
/// `SensorInput::entity_label`). Deliberately higher than the video
/// service's own default gallery-match threshold
/// (`VIDEO_INTERACTION_FACE_MATCH_THRESHOLD`, 0.55) - that threshold only
/// decides whether a *single observation* gets narrated with a name; this
/// one decides whether to reinforce a *persistent, shared* interlocutor
/// node that a voice enrollment may already anchor real history to. Getting
/// this wrong is a materially worse mistake than a wrongly-labeled one-off
/// sighting - see video-interaction-plan.md section 8.4. Start here, then
/// retune against a real gallery's actual false-match rate once one exists.
const LABEL_TRUST_THRESHOLD: f64 = 0.75;

/// One event's narration plus (for `entity.enter` only, above
/// `LABEL_TRUST_THRESHOLD`) a trusted identity claim. Anything else from a
/// newer Video Interaction build than this client knows about is skipped
/// rather than forwarded as raw JSON, the same "fail open, never forward
/// garbage" discipline `voice.rs::looks_hallucinated` applies to ASR
/// output.
///
/// `entity.exit`'s `entity_label` is deliberately never trusted here, even
/// though it echoes whatever label `entity.enter` assigned that track: this
/// client is stateless across polls (no per-`track_id` memory of whether
/// that earlier label actually cleared `LABEL_TRUST_THRESHOLD`), and
/// re-verifying a claim this client never itself confirmed is worse than
/// just not making it - the exit's *text* still says who left, it just
/// doesn't reinforce the interlocutor graph on that basis.
fn event_observation(topic: &str, payload: &serde_json::Value) -> Option<(String, Option<String>)> {
    let text = payload.get("text").and_then(|v| v.as_str())?.to_string();
    let label = match topic {
        "entity.enter" => {
            let confidence = payload.get("recognition_confidence").and_then(|v| v.as_f64());
            let label = payload.get("entity_label").and_then(|v| v.as_str());
            match (label, confidence) {
                (Some(label), Some(confidence)) if confidence >= LABEL_TRUST_THRESHOLD => Some(label.to_string()),
                _ => None,
            }
        }
        "scene.change" | "entity.exit" => None,
        _ => return None,
    };
    Some((text, label))
}

/// Joins up to `MAX_EVENTS_PER_FORWARD` event texts (most recent first is
/// never desired here - chronological order is what makes "X entered" then
/// "X left" read sensibly) into one `SensorInput` body, mirroring
/// `loop_actor.rs`'s own same-`stream_id` `RoomInput` coalescing (space-
/// joined runs of text), applied here to a time window instead of a shared
/// stream identity.
///
/// The combined identity claim is "unanimous or none": if every trusted
/// label among the kept events agrees, it's forwarded; if a poll window
/// happens to catch two *different* trusted identities (e.g. Derek and
/// Alice both entered within one `min_interval`), attaching either one to
/// the merged text would misattribute the other person's presence onto the
/// wrong interlocutor node, which is a worse outcome than just not claiming
/// an identity for that forward - see video-interaction-plan.md section
/// 8.4. The text itself is never dropped either way.
fn coalesce(observations: &[(String, Option<String>)]) -> Option<(String, Option<String>)> {
    if observations.is_empty() {
        return None;
    }
    let kept = if observations.len() > MAX_EVENTS_PER_FORWARD {
        tracing::warn!(total = observations.len(), kept = MAX_EVENTS_PER_FORWARD, "video event burst exceeded the per-forward cap; keeping only the most recent events");
        &observations[observations.len() - MAX_EVENTS_PER_FORWARD..]
    } else {
        observations
    };
    let text = kept.iter().map(|(text, _)| text.as_str()).collect::<Vec<_>>().join(" ");
    let distinct_labels: HashSet<&str> = kept.iter().filter_map(|(_, label)| label.as_deref()).collect();
    let label = match distinct_labels.len() {
        1 => distinct_labels.into_iter().next().map(str::to_string),
        0 => None,
        _ => {
            tracing::debug!(labels = ?distinct_labels, "more than one distinct recognized identity in one forward window; dropping identity linking for this batch, keeping the text");
            None
        }
    };
    Some((text, label))
}

impl VideoClient {
    pub fn new(base_url: String, min_interval: Duration) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("reqwest client builds with only a timeout configured"),
            base_url,
            min_interval,
        }
    }

    /// Retry/backoff for the initial cursor fetch - identical reasoning to
    /// `voice.rs::fetch_current_cursor`: covers the ordinary startup race
    /// (this daemon coming up slightly before the camera service does)
    /// without waiting so long a genuinely unreachable service holds up the
    /// listener forever, and specifically avoids ever falling back to
    /// replaying the service's full historical event log on a fresh start.
    const INITIAL_CURSOR_FETCH_ATTEMPTS: u32 = 10;
    const INITIAL_CURSOR_FETCH_RETRY_DELAY: Duration = Duration::from_secs(1);

    async fn fetch_current_cursor(&self) -> anyhow::Result<u64> {
        let mut last_err = None;
        for attempt in 0..Self::INITIAL_CURSOR_FETCH_ATTEMPTS {
            if attempt > 0 {
                tokio::time::sleep(Self::INITIAL_CURSOR_FETCH_RETRY_DELAY).await;
            }
            let response: anyhow::Result<serde_json::Value> = async {
                Ok(self.http.get(format!("{}/sync", self.base_url)).query(&[("after", "0")]).send().await?.error_for_status()?.json().await?)
            }
            .await;
            match response.and_then(|body| body.get("next_event").and_then(|v| v.as_u64()).ok_or_else(|| anyhow::anyhow!("response missing next_event"))) {
                Ok(cursor) => return Ok(cursor),
                Err(err) => last_err = Some(err),
            }
        }
        Err(last_err.expect("the attempts loop runs at least once, always setting last_err on failure"))
    }

    async fn poll_events(&self, cursor: u64) -> anyhow::Result<(u64, Vec<(String, Option<String>)>)> {
        let response: serde_json::Value = self.http.get(format!("{}/sync", self.base_url)).query(&[("after", cursor.to_string())]).send().await?.error_for_status()?.json().await?;
        let next_cursor = response.get("next_event").and_then(|v| v.as_u64()).unwrap_or(cursor);
        let events = response.get("events").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let observations = events
            .into_iter()
            .filter_map(|event| {
                let topic = event.get("topic").and_then(|v| v.as_str())?;
                let payload = event.get("payload")?;
                event_observation(topic, payload)
            })
            .collect();
        Ok((next_cursor, observations))
    }

    /// Spawns a detached background task that polls the video service's
    /// `GET /sync?after={cursor}` feed at `min_interval` cadence and
    /// forwards a coalesced `SensorInput` into `sensor_input_tx` for
    /// whatever `scene.change`/`entity.enter`/`entity.exit` events arrived
    /// since the last poll. Starts from the service's *current* tip via
    /// `fetch_current_cursor`, never from 0 - see that method's doc comment,
    /// and `voice.rs::spawn_listener`'s own for the flood a fresh restart
    /// replaying full history would otherwise cause.
    pub fn spawn_listener(self: std::sync::Arc<Self>, sensor_input_tx: mpsc::Sender<SensorInput>) {
        tokio::spawn(async move {
            let mut cursor: u64 = match self.fetch_current_cursor().await {
                Ok(cursor) => {
                    tracing::info!(cursor, "video listener starting from the service's current tip, not replaying its history");
                    cursor
                }
                Err(err) => {
                    tracing::warn!(error = %err, "could not learn the video service's current cursor after retrying - falling back to 0, which WILL replay its full history");
                    0
                }
            };
            loop {
                match self.poll_events(cursor).await {
                    Ok((next_cursor, observations)) => {
                        cursor = next_cursor;
                        if let Some((text, entity_label)) = coalesce(&observations) {
                            tracing::info!(text = %text, entity_label = ?entity_label, "observed camera activity, forwarding to the cognitive loop");
                            if sensor_input_tx.send(SensorInput { text, channel: SourceChannel::Environment, source: "camera", entity_label }).await.is_err() {
                                tracing::error!("cognitive loop is no longer accepting input; stopping video input listener");
                                return;
                            }
                        }
                    }
                    Err(err) => tracing::warn!(error = %err, "video input poll failed, retrying"),
                }
                tokio::time::sleep(self.min_interval).await;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_observation_narrates_known_topics_without_a_trusted_label_by_default() {
        let payload = serde_json::json!({"text": "A person entered the camera view."});
        assert_eq!(event_observation("entity.enter", &payload), Some(("A person entered the camera view.".to_string(), None)));
        assert_eq!(event_observation("scene.change", &payload), Some(("A person entered the camera view.".to_string(), None)));
        assert_eq!(event_observation("entity.exit", &payload), Some(("A person entered the camera view.".to_string(), None)));
    }

    #[test]
    fn event_observation_skips_unrecognized_topics() {
        let payload = serde_json::json!({"text": "irrelevant"});
        assert_eq!(event_observation("some.future.topic", &payload), None);
    }

    #[test]
    fn event_observation_is_none_without_a_text_field() {
        let payload = serde_json::json!({"track_id": "t1"});
        assert_eq!(event_observation("entity.enter", &payload), None);
    }

    #[test]
    fn event_observation_trusts_an_entity_enter_label_above_the_threshold() {
        let payload = serde_json::json!({"text": "Derek entered the camera view.", "entity_label": "derek", "recognition_confidence": 0.9});
        assert_eq!(event_observation("entity.enter", &payload), Some(("Derek entered the camera view.".to_string(), Some("derek".to_string()))));
    }

    #[test]
    fn event_observation_drops_an_entity_enter_label_below_the_threshold() {
        // Below LABEL_TRUST_THRESHOLD (0.75) even though it cleared the video
        // service's own default gallery-match threshold (0.55) - this
        // client's own, deliberately higher bar for identity-linking claims.
        let payload = serde_json::json!({"text": "Someone entered the camera view.", "entity_label": "derek", "recognition_confidence": 0.6});
        assert_eq!(event_observation("entity.enter", &payload), Some(("Someone entered the camera view.".to_string(), None)));
    }

    #[test]
    fn event_observation_never_trusts_an_entity_exit_label() {
        // Echoes the enter-time label, but this client never verified it
        // cleared the threshold at enter time - see this fn's own doc comment.
        let payload = serde_json::json!({"text": "Derek left the camera view.", "entity_label": "derek"});
        assert_eq!(event_observation("entity.exit", &payload), Some(("Derek left the camera view.".to_string(), None)));
    }

    #[test]
    fn coalesce_joins_multiple_texts_in_order() {
        let observations = vec![("A entered.".to_string(), None), ("B entered.".to_string(), None)];
        assert_eq!(coalesce(&observations), Some(("A entered. B entered.".to_string(), None)));
    }

    #[test]
    fn coalesce_is_none_for_an_empty_batch() {
        assert_eq!(coalesce(&[]), None);
    }

    #[test]
    fn coalesce_caps_a_burst_to_the_most_recent_events() {
        let observations: Vec<(String, Option<String>)> = (0..20).map(|i| (format!("event {i}"), None)).collect();
        let (joined, _) = coalesce(&observations).unwrap();
        assert!(!joined.contains("event 0"), "the oldest events in an oversized burst should be dropped, not the newest");
        assert!(joined.contains("event 19"), "the most recent event must survive the cap");
        assert_eq!(joined.split(' ').filter(|s| *s == "event").count(), MAX_EVENTS_PER_FORWARD);
    }

    #[test]
    fn coalesce_carries_a_unanimous_trusted_label_through() {
        let observations = vec![("Derek entered.".to_string(), Some("derek".to_string())), ("Derek picked something up.".to_string(), Some("derek".to_string()))];
        let (text, label) = coalesce(&observations).unwrap();
        assert_eq!(text, "Derek entered. Derek picked something up.");
        assert_eq!(label, Some("derek".to_string()));
    }

    #[test]
    fn coalesce_drops_the_label_when_two_different_identities_appear_in_one_window() {
        // Attaching either name to the merged text would misattribute the
        // other person's presence onto the wrong interlocutor node - see
        // this fn's own doc comment.
        let observations = vec![("Derek entered.".to_string(), Some("derek".to_string())), ("Alice entered.".to_string(), Some("alice".to_string()))];
        let (text, label) = coalesce(&observations).unwrap();
        assert_eq!(text, "Derek entered. Alice entered.", "the text itself must still carry both observations");
        assert_eq!(label, None, "an ambiguous batch must not link either identity");
    }

    #[test]
    fn coalesce_carries_a_label_through_even_when_mixed_with_unlabeled_events() {
        let observations = vec![("The camera view changed materially.".to_string(), None), ("Derek entered.".to_string(), Some("derek".to_string()))];
        let (_, label) = coalesce(&observations).unwrap();
        assert_eq!(label, Some("derek".to_string()));
    }
}
