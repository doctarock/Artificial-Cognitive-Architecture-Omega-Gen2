extends Node3D
## The Knowledge Library's corpus size, as floating text near the Executive
## station (the ConsultKnowledgeLibrary operator lives there) - the one
## piece of this session's Knowledge Library feature that otherwise has no
## visual representation at all. specs.md frames the Knowledge Library as
## external, not Memory (see EngineSnapshot::kl_doc_count's own doc
## comment): other household agents write into it directly over MCP,
## bypassing the cognitive loop entirely, so a live count here is the only
## way a viewer sees that growth happen - individual consult results
## already show up in the event log, but this is the standing record of
## what's accumulated, mirroring memory_silos.gd's role for actual Memory.
##
## Two distinct activity cues, since they're two distinct kinds of event:
## a consult query (something the cognitive loop itself did, at Act -
## `ActOutcome::ConsultedKnowledgeLibrary`) gets a packet traveling from Act
## to this label; the corpus growing (another household agent writing in
## over MCP, invisible to the cognitive loop entirely) gets a plain local
## pulse, since there's no in-loop station it could travel from.

const QUERY_COLOR := Color(0.85, 0.7, 1.0)
const GROWTH_COLOR := Color(0.5, 1.0, 0.7)
const PULSE_UP_SEC := 0.12
const PULSE_DOWN_SEC := 0.5
const PACKET_TRAVEL_SEC := 0.45
const PACKET_FADE_SEC := 0.3
const PACKET_Y := 0.55

@onready var _pipeline: Node3D = get_parent().get_node("PipelineView")

var _label: Label3D
## `-1` means "no snapshot seen yet" - the first real snapshot just
## establishes the baseline, it never pulses (a fresh corpus count would
## otherwise always read as "growth" from zero on startup).
var _previous_count: int = -1


func _ready() -> void:
	ApiClient.snapshot_updated.connect(_on_snapshot_updated)
	ApiClient.event_received.connect(_on_event_received)

	_label = Label3D.new()
	_label.pixel_size = 0.01
	_label.font_size = 30
	_label.outline_size = 6
	_label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
	_label.modulate = QUERY_COLOR
	_label.text = "Knowledge Library: (no data yet)"
	add_child(_label)


func _on_snapshot_updated(data: Dictionary) -> void:
	var count: int = int(data.get("kl_doc_count", 0))
	var noun := "document" if count == 1 else "documents"
	_label.text = "Knowledge Library:\n%d %s" % [count, noun]

	if _previous_count >= 0 and count > _previous_count:
		_pulse(GROWTH_COLOR)
	_previous_count = count


func _on_event_received(data: Dictionary) -> void:
	if str(data.get("phase", "")) != "act":
		return
	var payload: Dictionary = data.get("payload", {})
	if str(payload.get("operator", "")) != "consult-knowledge-library":
		return
	_pulse(QUERY_COLOR)
	if _pipeline.has_station("act"):
		_spawn_packet(Vector3(_pipeline.station_x("act"), PACKET_Y, 0))


func _pulse(color: Color) -> void:
	var tween := create_tween()
	tween.tween_property(_label, "modulate", color.lightened(0.4), PULSE_UP_SEC)
	tween.tween_property(_label, "modulate", QUERY_COLOR, PULSE_DOWN_SEC)


func _spawn_packet(from_pos: Vector3) -> void:
	var packet := MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = 0.1
	sphere.height = 0.2
	packet.mesh = sphere
	packet.position = from_pos
	var material := StandardMaterial3D.new()
	material.emission_enabled = true
	material.albedo_color = QUERY_COLOR
	material.emission = QUERY_COLOR
	material.emission_energy_multiplier = 3.0
	packet.set_surface_override_material(0, material)
	# Parented under the shared root (this node's own parent), not under
	# `self` - `self` carries architecture_root.gd's station-adjacent
	# offset, so a child of `self` would apply that offset twice against
	# `global_position` below.
	get_parent().add_child(packet)

	var move_tween := create_tween()
	move_tween.tween_property(packet, "position", global_position, PACKET_TRAVEL_SEC)
	move_tween.finished.connect(func():
		var fade_tween := create_tween()
		fade_tween.tween_property(packet, "scale", Vector3.ZERO, PACKET_FADE_SEC)
		fade_tween.finished.connect(packet.queue_free)
	)
