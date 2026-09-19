extends Node3D
## The SOAR executive's most recently *selected* operator
## (`EngineSnapshot.last_operator_proposal` - `steps::executive::
## propose_operators`'s winner, one of 8 closed variants: Speak/Remember/
## ContinueReflecting/Ignore/ConsultKnowledgeLibrary/Plan/Ask/Act, published
## as Rust Debug so e.g. "ConsultKnowledgeLibrary" arrives PascalCase).
## Only the winner is ever published here, never rejected candidates. A
## direct structural clone of goal_stack_view.gd's label/diff/pulse pattern
## - same station, same idiom, root-positioned above the existing Goal
## Stack / Knowledge Library pair rather than a third flanking point.

const PULSE_COLOR := Color(1.0, 0.85, 0.4)
const REST_COLOR := Color(1, 1, 1)
const PULSE_UP_SEC := 0.12
const PULSE_DOWN_SEC := 0.5

var _label: Label3D
var _seen_before := false
var _previous_at: int = -1


func _ready() -> void:
	ApiClient.snapshot_updated.connect(_on_snapshot_updated)

	_label = Label3D.new()
	_label.pixel_size = 0.01
	_label.font_size = 28
	_label.outline_size = 6
	_label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_label.modulate = REST_COLOR
	_label.text = "Decision: (none yet)"
	add_child(_label)


func _on_snapshot_updated(data: Dictionary) -> void:
	var proposal = data.get("last_operator_proposal", null)

	if proposal == null:
		_label.text = "Decision: (none yet)"
		_seen_before = true
		return

	var proposal_dict: Dictionary = proposal
	var operator_name: String = str(proposal_dict.get("operator", "?"))
	var preference: float = proposal_dict.get("preference", 0.0)
	var confidence: float = proposal_dict.get("confidence", 0.0)
	_label.text = "Decision: %s\n(pref %.2f, conf %.2f)" % [operator_name, preference, confidence]

	var at: int = int(proposal_dict.get("at", 0))
	if _seen_before and at != _previous_at:
		_pulse()
	_seen_before = true
	_previous_at = at


func _pulse() -> void:
	var color_tween := create_tween()
	color_tween.tween_property(_label, "modulate", PULSE_COLOR, PULSE_UP_SEC)
	color_tween.tween_property(_label, "modulate", REST_COLOR, PULSE_DOWN_SEC)

	var scale_tween := create_tween()
	scale_tween.tween_property(_label, "scale", Vector3.ONE * 1.15, PULSE_UP_SEC)
	scale_tween.tween_property(_label, "scale", Vector3.ONE, PULSE_DOWN_SEC)
