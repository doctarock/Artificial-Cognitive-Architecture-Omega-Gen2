extends Node
## Autoloaded singleton: the visualization's one connection to the Rust
## cognitive engine (`omega-acad`). A pure consumer of aca-api's HTTP+WS
## surface - this script never invents state, it only mirrors what the
## engine reports.

signal snapshot_updated(data: Dictionary)
signal event_received(data: Dictionary)
signal connection_status_changed(connected: bool)
signal test_run_started()
signal test_status_updated(data: Dictionary)
signal test_run_failed(reason: String)
signal book_delivered(data: Dictionary)
signal book_delivery_failed(reason: String)
signal input_submit_failed(reason: String)

const BASE_URL := "http://127.0.0.1:8787"
const WS_URL := "ws://127.0.0.1:8787/events"
const SNAPSHOT_POLL_INTERVAL_SEC := 0.5
const RECONNECT_DELAY_SEC := 2.0
# `cargo test --workspace` can legitimately take well over a minute on a
# cold build (see aca-api::test_status's own timeout, 180s) - HTTPRequest's
# default 60s Godot-side timeout would cut that off first and report a
# spurious failure for a run that was still genuinely in progress server-side.
const TEST_STATUS_TIMEOUT_SEC := 190.0
# Godot's HTTPRequest default timeout is 0 (never times out). Every ordinary
# local-daemon call except the test-status poll above is bounded by this
# instead: if the daemon accepts the connection but stalls (e.g. wedged
# behind a long synchronous Tier 3/4 call), the matching `_*_in_flight` flag
# would otherwise stay true forever - permanently wedging snapshot polling
# or input submission with no way to recover short of restarting the viz.
const REQUEST_TIMEOUT_SEC := 10.0

var _socket := WebSocketPeer.new()
var _was_connected := false
var _reconnect_timer := 0.0

@onready var _snapshot_request := HTTPRequest.new()
@onready var _input_request := HTTPRequest.new()
@onready var _test_status_request := HTTPRequest.new()
@onready var _deliver_book_request := HTTPRequest.new()

# get_http_client_status() proved unreliable as a "is this node free"
# check across repeated .request() calls on the same HTTPRequest node (the
# snapshot view stayed empty forever in a live test, even though the
# daemon genuinely had Working Memory members) - an explicit flag toggled
# by the request/response pair is simple and unambiguous.
var _snapshot_in_flight := false
var _input_in_flight := false
var _test_status_in_flight := false
var _deliver_book_in_flight := false


func _ready() -> void:
	add_child(_snapshot_request)
	add_child(_input_request)
	add_child(_test_status_request)
	add_child(_deliver_book_request)
	_snapshot_request.request_completed.connect(_on_snapshot_response)
	_snapshot_request.timeout = REQUEST_TIMEOUT_SEC
	_input_request.request_completed.connect(_on_input_response)
	_input_request.timeout = REQUEST_TIMEOUT_SEC
	_test_status_request.request_completed.connect(_on_test_status_response)
	_test_status_request.timeout = TEST_STATUS_TIMEOUT_SEC
	_deliver_book_request.request_completed.connect(_on_deliver_book_response)
	_deliver_book_request.timeout = REQUEST_TIMEOUT_SEC

	var timer := Timer.new()
	add_child(timer)
	timer.wait_time = SNAPSHOT_POLL_INTERVAL_SEC
	timer.timeout.connect(_poll_snapshot)
	# `autostart` only takes effect when the node enters the tree, which
	# already happened via add_child() above by the time this line would
	# have set it - the timer silently never started. An explicit start()
	# has no such ordering trap. This was the actual root cause of the
	# mind map staying empty: not the camera, not emission, not HTTP -
	# the poll timer that should have driven all of it never ticked.
	timer.start()

	_connect_socket()
	_poll_snapshot()  # also do an immediate first poll, don't wait 0.5s


func _connect_socket() -> void:
	var err := _socket.connect_to_url(WS_URL)
	if err != OK:
		push_warning("ApiClient: failed to start websocket connection: %s" % err)


func _process(delta: float) -> void:
	_socket.poll()
	var state := _socket.get_ready_state()

	match state:
		WebSocketPeer.STATE_OPEN:
			if not _was_connected:
				_was_connected = true
				connection_status_changed.emit(true)
			while _socket.get_available_packet_count() > 0:
				var packet := _socket.get_packet()
				_handle_event_packet(packet.get_string_from_utf8())
		WebSocketPeer.STATE_CLOSED:
			if _was_connected:
				_was_connected = false
				connection_status_changed.emit(false)
			_reconnect_timer += delta
			if _reconnect_timer >= RECONNECT_DELAY_SEC:
				_reconnect_timer = 0.0
				_connect_socket()


func _handle_event_packet(text: String) -> void:
	var parsed = JSON.parse_string(text)
	if typeof(parsed) == TYPE_DICTIONARY:
		event_received.emit(parsed)


func _poll_snapshot() -> void:
	if _snapshot_in_flight:
		return
	_snapshot_in_flight = true
	var err := _snapshot_request.request(BASE_URL + "/snapshot/overview")
	if err != OK:
		push_warning("ApiClient: snapshot request failed to start: %s" % err)
		_snapshot_in_flight = false


func _on_snapshot_response(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	_snapshot_in_flight = false
	if response_code != 200:
		push_warning("ApiClient: snapshot request returned HTTP %s" % response_code)
		return
	var parsed = JSON.parse_string(body.get_string_from_utf8())
	if typeof(parsed) == TYPE_DICTIONARY:
		snapshot_updated.emit(parsed)


## Submits text input to the cognitive loop, the same channel the CLI and
## any other client use - never a side path into the engine. Returns `true`
## if the request actually started - `false` means it was dropped outright
## (still in flight, or failed to start) and the caller should not treat the
## text as sent (see `event_log.gd`'s `_on_text_submitted`, which used to
## clear the input box unconditionally regardless of this).
func submit_input(text: String) -> bool:
	if _input_in_flight:
		input_submit_failed.emit("a previous submission is still in flight")
		return false
	_input_in_flight = true
	var body := JSON.stringify({"text": text})
	var headers := ["Content-Type: application/json"]
	var err := _input_request.request(BASE_URL + "/input", headers, HTTPClient.METHOD_POST, body)
	if err != OK:
		_input_in_flight = false
		input_submit_failed.emit("failed to start request: %s" % err)
		return false
	return true


## Confirmed live: a large paste (a document, a log excerpt) silently never
## appeared in the console at all - this handler used to discard
## `response_code` entirely (`_result`/`_body` too), so a non-2xx response
## (413 from a body-size limit, 503 from the actor being unavailable, or
## anything else) produced zero feedback anywhere. The input box had
## already been cleared by `_on_text_submitted` the instant Enter was
## pressed, so the paste looked like it had simply vanished.
func _on_input_response(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	_input_in_flight = false
	if response_code < 200 or response_code >= 300:
		var detail := body.get_string_from_utf8()
		input_submit_failed.emit("HTTP %s%s" % [response_code, (": " + detail) if not detail.is_empty() else ""])


## Triggers a fresh `cargo test --workspace` run on the daemon and reports
## the result via `test_status_updated`. Deliberately caller-triggered, not
## polled the way `_poll_snapshot` is - a full workspace run takes real
## seconds (see aca-api::test_status's own doc comment), so hammering this
## every 0.5s the way the snapshot is would mean a near-permanent backlog
## of test runs. `_test_status_in_flight` guards against starting a second
## run while one is already in progress.
func run_tests() -> void:
	if _test_status_in_flight:
		return
	_test_status_in_flight = true
	test_run_started.emit()
	var err := _test_status_request.request(BASE_URL + "/tests/status")
	if err != OK:
		_test_status_in_flight = false
		test_run_failed.emit("failed to start request: %s" % err)


func _on_test_status_response(result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	_test_status_in_flight = false
	if result == HTTPRequest.RESULT_TIMEOUT:
		test_run_failed.emit("timed out waiting for the test run to finish")
		return
	if response_code != 200:
		test_run_failed.emit("HTTP %s" % response_code)
		return
	var parsed = JSON.parse_string(body.get_string_from_utf8())
	if typeof(parsed) == TYPE_DICTIONARY:
		test_status_updated.emit(parsed)
	else:
		test_run_failed.emit("response was not valid JSON")


## Manually advances the library's "currently reading" cursor by exactly one
## chunk (`omega_acad::library::deliver_next_chunk`) - the replacement for
## the old unattended drip timer. Caller-triggered only, like `run_tests()`.
func deliver_book() -> void:
	if _deliver_book_in_flight:
		return
	_deliver_book_in_flight = true
	var err := _deliver_book_request.request(BASE_URL + "/library/deliver", [], HTTPClient.METHOD_POST)
	if err != OK:
		_deliver_book_in_flight = false
		book_delivery_failed.emit("failed to start request: %s" % err)


func _on_deliver_book_response(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	_deliver_book_in_flight = false
	if response_code != 200:
		book_delivery_failed.emit("HTTP %s (is OMEGA_LIBRARY_PATH configured?)" % response_code)
		return
	var parsed = JSON.parse_string(body.get_string_from_utf8())
	if typeof(parsed) == TYPE_DICTIONARY:
		book_delivered.emit(parsed)
	else:
		book_delivery_failed.emit("response was not valid JSON")
