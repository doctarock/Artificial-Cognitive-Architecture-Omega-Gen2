# Room-Audio Quality Gate

The first piece of the "handle noisy multi-party rooms" problem: don't let hallucinated ASR
transcripts into the cognitive loop at all. Confirmed live — a room-audio session produced
Working Memory entries like `"and malicious but it can take its course with a view that"` and
`"I'm not a racist."`, neither of which is anything a real person plausibly said; these are
Whisper hallucinating on ambient noise/silence, not genuine speech that the cognitive
architecture's own relevance filtering should be expected to make sense of.

This is step one of a larger effort: filter irrelevant chatter, track discourse/topic context,
maintain a mental model of a multi-party conversation, and contribute meaningfully to it.
Addressee/relevance detection (distinguishing "said near Omega" from "said to Omega" — today,
both hit the same `SourceChannel::ConversationInput` with nothing to tell them apart) and actual
discourse-thread state are separate, larger pieces still to come; this only stops garbage
transcripts from ever reaching Working Memory in the first place.

## How It Works

Two repos, because the signal has to be produced before it can be filtered on.

**`E:\AI\Voice Interaction` (Python, the ASR/VAD/TTS service)**

- `WhisperAdapter._transcribe_endpoint` (`workers.py`) now requests `response_format=verbose_json`
  instead of `json` from the whisper.cpp server — the only way to get Whisper's own per-segment
  hallucination signals (`avg_logprob`, `no_speech_prob`, `compression_ratio`) back at all; plain
  `json` only ever returns the concatenated text.
- `_worst_case_segment_signals` reduces a possibly-multi-segment response to one worst-case triple
  (lowest `avg_logprob`, highest `no_speech_prob`, highest `compression_ratio`) — any one segment
  reading as hallucinated is reason enough to flag the whole utterance.
- These three fields are threaded through `TranscriptionResult` → the worker-RPC payload
  (`worker_host.py`/`WorkerRoutedWhisperAdapter`, so the worker-routed transcription path carries
  them too) → `Utterance` (`models.py`) → the `speech.utterance` event published over `/sync`.
  All three are `Optional[float]`, defaulting to `None` — an older client, a transcription path
  with no segment-level data (the CLI/subprocess adapter), or restoring an old persisted snapshot
  from before this change all degrade safely rather than erroring.
- This service makes no filtering decision itself — it only ever exposes the raw signal.
  Deliberate: other consumers of the same event stream may still want the raw text even when
  Omega wants to drop it, and *what counts as good enough* is a policy call that belongs to the
  consumer, not the transport.

**This repo (Rust, `crates/omega-acad/src/voice.rs`)**

- `VoiceClient::looks_hallucinated` applies the actual filter, using the same combined heuristic
  Whisper's own decoder uses to flag a segment for discard: `no_speech_prob > 0.6 AND
  avg_logprob < -1.0` (both together — a pause the VAD chunked, or ordinary text the model was
  just moderately unsure about, shouldn't alone condemn a transcript), OR `compression_ratio >
  2.4` (whisper.cpp's separate repetition-loop failure mode, where the decoder gets stuck
  re-emitting near-identical text — compression collapses the repetition, so the ratio spikes in
  a way real speech's entropy doesn't).
- Applied in `poll_utterances`, right where echo/self-hearing filtering already happens — a
  hallucinated utterance is dropped before it's ever built into a `RoomInput` and sent into the
  cognitive loop's input channel, the same "never even nominated for Coalition" treatment
  self-echo already gets.
- Fails open: any utterance missing one of these fields (an unupgraded Voice Interaction
  instance, or a transcription path that never had segment data) is never filtered on that basis
  — same "optional integration degrades to a no-op, never a silent drop of real input" discipline
  every other adapter in this daemon follows.

## Tuning

Thresholds are the community-standard Whisper hallucination-detection values (also OpenAI's own
Whisper decoder's defaults), not something re-derived for this deployment — start there before
adjusting. If real quiet/uncertain speech starts getting dropped, loosen `AVG_LOGPROB_THRESHOLD`
first (make it more negative) before touching `NO_SPEECH_PROB_THRESHOLD`, since a false drop from
the logprob side is more common than from the no-speech side in practice.

## Read The Results

- `speech.utterance` events over `/sync` should now carry `avg_logprob`/`no_speech_prob`/
  `compression_ratio` (non-`null`) whenever the Voice Interaction service's Whisper endpoint is
  configured — verify with the same event log used for other debugging.
- Fragments that were previously landing in Working Memory as nonsense (garbled/repetitive text
  with no plausible speaker intent) should no longer arrive as Observations at all.
- `VoiceClient::tests::looks_hallucinated_*` (`voice.rs`) covers the threshold logic;
  `tests/test_adapters.py`/`tests/test_worker_routing.py` (Voice Interaction repo) cover the
  signal threading through both transcription paths.
