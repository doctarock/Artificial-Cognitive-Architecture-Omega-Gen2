extends Control
## The UI overlay: a scrolling live event log, a text input box wired to
## POST /input, and a connection status indicator. This is the "text"
## interface layered on top of the 3D view - typing here and watching the
## mind map react is the whole point of this v0.

const MAX_LOG_LINES := 200

@onready var _log: RichTextLabel = %EventLog
@onready var _input: TextEdit = %InputBox
@onready var _status: Label = %StatusLabel

var _line_count := 0


func _ready() -> void:
	ApiClient.event_received.connect(_on_event_received)
	ApiClient.connection_status_changed.connect(_on_connection_status_changed)
	ApiClient.input_submit_failed.connect(_on_input_submit_failed)
	_input.gui_input.connect(_on_input_gui_input)
	_status.text = "connecting..."


func _on_connection_status_changed(connected: bool) -> void:
	_status.text = "connected" if connected else "disconnected - retrying"
	_status.modulate = Color(0.4, 1.0, 0.5) if connected else Color(1.0, 0.5, 0.4)


## `TextEdit` (unlike `LineEdit`) has no `text_submitted` signal and no
## single-line limitation - which is exactly why this switched from
## `LineEdit` in the first place: confirmed live, pasting a real multi-
## paragraph block (blank lines between paragraphs) into a `LineEdit` could
## silently fail to land in the box at all, so Enter afterward submitted
## nothing and nothing ever showed up anywhere, not even an error - the
## box was just empty. A single-line control was never the right tool for
## "paste an AI prompt here." Enter still sends (matching the old
## behavior and every chat UI's convention); Shift+Enter inserts an actual
## newline instead, so multi-line composition by hand still works too.
func _on_input_gui_input(event: InputEvent) -> void:
	if event is InputEventKey:
		var key_event := event as InputEventKey
		if not key_event.pressed or key_event.echo:
			return
		var is_enter := key_event.keycode == KEY_ENTER or key_event.keycode == KEY_KP_ENTER
		if is_enter and not key_event.shift_pressed:
			_input.accept_event()
			_try_submit()


## How much of a sent message to echo into the log directly - long enough
## to recognize a multi-paragraph submission at a glance, short enough that
## one message can't fill a meaningful fraction of the whole scrollback.
const ECHO_PREVIEW_MAX_CHARS := 200


func _try_submit() -> void:
	var text := _input.text
	if text.strip_edges().is_empty():
		return
	# Only clear the box on an actually-started submission - confirmed live,
	# clearing unconditionally made a rejected large paste (413 from the
	# API's body limit, or any other failure) look identical to a
	# successful one: the text vanished from the box and nothing ever
	# appeared in the log. `_on_input_submit_failed` reports why when this
	# returns false, and the user's pasted text is left in place instead of
	# silently discarded.
	if ApiClient.submit_input(text):
		_input.text = ""
		# Confirms the send *immediately*, on the HTTP request actually
		# starting - not on the engine getting around to processing it.
		# Confirmed live: a slow/contended embedding call can leave a real
		# gap of minutes between "sent" and the engine's own "heard: ..."
		# line - with nothing in between, a message that was accepted fine
		# looked identical to one silently lost. This line is the
		# difference between the two.
		var preview := text.strip_edges().replace("\n", " / ")
		if preview.length() > ECHO_PREVIEW_MAX_CHARS:
			preview = preview.substr(0, ECHO_PREVIEW_MAX_CHARS) + "…"
		_append_log_line("[color=#66ff99][sent] %s[/color]" % preview)


func _on_input_submit_failed(reason: String) -> void:
	_append_log_line("[color=#ff2222][input not sent] %s[/color]" % reason)


## Shared by every source of a log line (engine events, send confirmations,
## submit failures) so the scrollback cap is enforced consistently - before
## this was factored out, `_on_input_submit_failed` didn't apply it at all.
func _append_log_line(bbcode_line: String) -> void:
	_log.append_text(bbcode_line + "\n")
	_line_count += 1
	if _line_count > MAX_LOG_LINES:
		_log.clear()
		_line_count = 0


func _on_event_received(data: Dictionary) -> void:
	var phase: String = data.get("phase", "?")
	var event_type: String = data.get("event_type", "?")
	var payload: Dictionary = data.get("payload", {})
	var cycle_seq = data.get("cycle_seq", "?")

	if not _is_worth_logging(phase, event_type, payload):
		return

	var color := "aaaaaa"
	match event_type:
		"impasse":
			color = "ff9944"
		"escalation":
			color = "ff5566"
		"error":
			color = "ff2222"
		"normal":
			color = "88bbff"

	var summary := _summarize_payload(phase, payload)
	_append_log_line("[color=#%s][%s] %s: %s[/color]" % [color, cycle_seq, phase, summary])


## An allowlist, not a denylist, deliberately: Coalition and routine Predict/
## Learn events fire on essentially every idle tick for as long as anything
## sits in Working Memory (real activity, but a candidate-score dump like
## `{"candidate_count":1,"top_score":-1.99}` every few milliseconds is not a
## readable log - it's what made this feel like "a lot of numbers scrolling
## fast"). The pipeline view's station pulses already show that rhythm
## visually; this text log is for the comparatively rare, meaningful beats -
## a new input arriving, something actually said, an impasse, a chunk
## learned. A denylist would need updating every time a new noisy event
## shape appeared; this way, anything new defaults to hidden.
func _is_worth_logging(phase: String, event_type: String, payload: Dictionary) -> bool:
	if event_type == "impasse" or event_type == "escalation" or event_type == "error":
		return true
	match phase:
		"compare", "broadcast":
			return true
		"executive":
			return str(payload.get("operator", "")) != "Ignore"
		"act":
			return str(payload.get("operator", "")) != "silent"
		"learn":
			return payload.has("chunked_operator")
		"synthesize":
			# Gated (needs 3+ new episodic memories, minutes apart per
			# SynthesisConfig) and self-triggered - rare and meaningful by
			# construction, same bar as an impasse or a learned chunk.
			return true
		_:
			return false


func _summarize_payload(phase: String, payload: Dictionary) -> String:
	if phase == "compare" and payload.has("text"):
		return "heard: " + str(payload.get("text"))
	if payload.has("text"):
		return str(payload.get("operator", "")) + ": " + str(payload.get("text"))
	# Checked ahead of the generic `payload.has("reason")` fallback below,
	# which this would otherwise also match (any `act`-phase Silent outcome
	# now carries a `reason` field - see `SilentReason` in `steps::act`) but
	# renders uselessly (`kind` is absent from this payload shape). This is
	# the one Silent reason that ever actually reaches this function at all:
	# every other reason stays Normal severity and phase="act" operator=
	# "silent" is filtered out by `_is_worth_logging` before summarizing is
	# ever attempted - `reflection-failed` is deliberately promoted to Error
	# severity precisely so it survives that filter (see `act_outcome_
	# is_failure` on the engine side) and lands here instead of vanishing.
	if payload.get("reason", "") == "reflection-failed":
		return "reflection failed (was going to %s): %s" % [str(payload.get("attempted_operator", "")), str(payload.get("error", ""))]
	if payload.has("reason"):
		return str(payload.get("kind", "")) + " - " + str(payload.get("reason"))
	if payload.has("tool"):
		if payload.get("ok", false):
			return "used tool '%s'" % str(payload.get("tool"))
		return "tool '%s' failed: %s" % [str(payload.get("tool")), str(payload.get("error", ""))]
	if str(payload.get("operator", "")) == "consult-knowledge-library":
		return "consulted knowledge library: %s" % ("found a match" if payload.get("found", false) else "no match found")
	if payload.has("source_count"):
		return "synthesized a pattern from %s episodic memories" % str(payload.get("source_count"))
	if payload.has("operator"):
		return str(payload.get("operator"))
	if payload.has("admitted") or payload.has("released"):
		return "admitted %s, released %s" % [payload.get("admitted", 0), payload.get("released", 0)]
	if payload.has("chunked_operator"):
		return "learned: prefer " + str(payload.get("chunked_operator"))
	return JSON.stringify(payload)
