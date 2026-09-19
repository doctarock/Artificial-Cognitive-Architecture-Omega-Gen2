extends Node3D
## The SOAR goal stack, as floating text near the Executive station: every
## active or impassed goal currently on it. Empty most of the time in this
## MVP (goals are mostly born from impasses) - showing "(empty)" honestly
## rather than hiding the label is the point; an empty stack is real
## information, not a placeholder to disguise. Flashes and pops when the
## stack actually changes (a goal appears, resolves, or changes status)
## rather than silently swapping text - the one purely textual view in this
## scene otherwise had no activity cue at all.

const PULSE_COLOR := Color(1.0, 0.85, 0.4)
const REST_COLOR := Color(1, 1, 1)
const PULSE_UP_SEC := 0.12
const PULSE_DOWN_SEC := 0.5

var _label: Label3D
## `false` until the first snapshot lands - that first population is just
## establishing the baseline, not a "change" to pulse for.
var _seen_before := false
var _previous_goals: Dictionary = {}  # goal id (String) -> status (String)


func _ready() -> void:
	ApiClient.snapshot_updated.connect(_on_snapshot_updated)

	_label = Label3D.new()
	_label.pixel_size = 0.01
	_label.font_size = 30
	_label.outline_size = 6
	_label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
	_label.modulate = REST_COLOR
	_label.text = "Goal Stack: (empty)"
	add_child(_label)


func _on_snapshot_updated(data: Dictionary) -> void:
	var goals: Array = data.get("goal_stack", [])
	# `steps::agenda`'s persistent-intention count - not part of goal_stack
	# itself (Intentions are a separate MentalObjectKind), but close enough
	# in concept (both are SOAR goal-stack-adjacent standing commitments)
	# that it rides on this same label rather than a dedicated view.
	var active_intentions: int = int(data.get("active_intention_count", 0))

	var new_goals: Dictionary = {}
	for g in goals:
		var goal: Dictionary = g
		new_goals[str(goal.get("id", ""))] = str(goal.get("status", ""))

	var lines: Array = []
	if goals.is_empty():
		lines.append("Goal Stack: (empty)")
	else:
		lines.append("Goal Stack:")
		for i in min(goals.size(), 5):
			var goal: Dictionary = goals[i]
			lines.append("- [%s] %s" % [str(goal.get("status", "")), str(goal.get("text", ""))])
	lines.append("Active intentions: %d" % active_intentions)
	_label.text = "\n".join(lines)

	if _seen_before and _goals_changed(new_goals):
		_pulse()
	_seen_before = true
	_previous_goals = new_goals


func _goals_changed(new_goals: Dictionary) -> bool:
	if new_goals.size() != _previous_goals.size():
		return true
	for id in new_goals.keys():
		if not _previous_goals.has(id) or _previous_goals[id] != new_goals[id]:
			return true
	return false


func _pulse() -> void:
	var color_tween := create_tween()
	color_tween.tween_property(_label, "modulate", PULSE_COLOR, PULSE_UP_SEC)
	color_tween.tween_property(_label, "modulate", REST_COLOR, PULSE_DOWN_SEC)

	var scale_tween := create_tween()
	scale_tween.tween_property(_label, "scale", Vector3.ONE * 1.15, PULSE_UP_SEC)
	scale_tween.tween_property(_label, "scale", Vector3.ONE, PULSE_DOWN_SEC)
