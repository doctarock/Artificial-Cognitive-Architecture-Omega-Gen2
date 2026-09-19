extends Control
## A caller-triggered test-suite panel: press "Run Tests" to fire
## `cargo test --workspace` on the daemon (see aca-api::test_status), then
## show per-crate pass/fail with color coding. Not auto-polled - a full
## workspace run takes real seconds, not the ~0ms `/snapshot/overview`
## already gets polled at, so running it on every tick the way the
## snapshot view does would mean a permanent backlog of test runs.

@onready var _run_button: Button = %RunTestsButton
@onready var _summary_label: Label = %TestSummaryLabel
@onready var _details_label: RichTextLabel = %TestDetailsLabel

const COLOR_PASSING := Color(0.4, 1.0, 0.5)
const COLOR_FAILING := Color(1.0, 0.35, 0.35)
const COLOR_RUNNING := Color(0.95, 0.85, 0.4)
const COLOR_IDLE := Color(0.7, 0.7, 0.75)


func _ready() -> void:
	_run_button.pressed.connect(_on_run_pressed)
	ApiClient.test_run_started.connect(_on_test_run_started)
	ApiClient.test_status_updated.connect(_on_test_status_updated)
	ApiClient.test_run_failed.connect(_on_test_run_failed)
	_summary_label.text = "Tests: not run yet"
	_summary_label.modulate = COLOR_IDLE


func _on_run_pressed() -> void:
	ApiClient.run_tests()


func _on_test_run_started() -> void:
	_run_button.disabled = true
	_summary_label.text = "Running cargo test --workspace..."
	_summary_label.modulate = COLOR_RUNNING
	_details_label.clear()


func _on_test_run_failed(reason: String) -> void:
	_run_button.disabled = false
	_summary_label.text = "Test run failed: %s" % reason
	_summary_label.modulate = COLOR_FAILING


func _on_test_status_updated(data: Dictionary) -> void:
	_run_button.disabled = false
	var all_passing: bool = data.get("all_passing", false)
	var total_passed: int = int(data.get("total_passed", 0))
	var total_failed: int = int(data.get("total_failed", 0))
	var duration_sec: float = float(data.get("duration_ms", 0)) / 1000.0

	if all_passing:
		_summary_label.text = "✓ all %d tests passing (%.1fs)" % [total_passed, duration_sec]
		_summary_label.modulate = COLOR_PASSING
	else:
		_summary_label.text = "✗ %d passed, %d failed (%.1fs)" % [total_passed, total_failed, duration_sec]
		_summary_label.modulate = COLOR_FAILING

	_details_label.clear()
	var binaries: Array = data.get("binaries", [])
	for entry in binaries:
		var binary: Dictionary = entry
		var name: String = str(binary.get("name", "?"))
		var passed: int = int(binary.get("passed", 0))
		var failed: int = int(binary.get("failed", 0))
		# Doctest binaries with nothing to run are real (this workspace has
		# no doc examples yet) but add no information - skip the noise.
		if passed == 0 and failed == 0:
			continue
		var color := "88ff99" if failed == 0 else "ff5555"
		_details_label.append_text("[color=#%s]%s: %d passed, %d failed[/color]\n" % [color, name, passed, failed])
