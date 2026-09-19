extends Node3D
## Visualizes actual flow through the cycle: each time the active phase
## changes, a small glowing packet - carrying a truncated preview of the
## event's text/content when there is one - travels from the previously-
## active station to the newly-active one, then fades. Stations pulsing in
## place shows *that* something happened; a packet visibly moving from
## Compare to Coalition to Broadcast to Executive to Act shows *what*, and
## in what order - this is the direct answer to "watch the whole process."

const TRAVEL_DURATION_SEC := 0.45
const FADE_DURATION_SEC := 0.35
const STATION_Y := 0.55
const MAX_PREVIEW_CHARS := 26

const EVENT_COLORS := {
	"normal": Color(0.4, 0.8, 1.0),
	"impasse": Color(1.0, 0.6, 0.15),
	"escalation": Color(1.0, 0.2, 0.2),
	"error": Color(1.0, 0.0, 0.0),
}

@onready var _pipeline: Node3D = get_parent().get_node("PipelineView")

var _last_phase: String = ""


func _ready() -> void:
	ApiClient.event_received.connect(_on_event_received)


func _on_event_received(data: Dictionary) -> void:
	var phase: String = data.get("phase", "")
	if not _pipeline.has_station(phase):
		return

	if _last_phase != "" and _last_phase != phase and _pipeline.has_station(_last_phase):
		var from_pos := Vector3(_pipeline.station_x(_last_phase), STATION_Y, 0)
		var to_pos := Vector3(_pipeline.station_x(phase), STATION_Y, 0)
		var event_type: String = data.get("event_type", "normal")
		var preview: String = _extract_preview(data.get("payload", {}))
		_spawn_packet(from_pos, to_pos, preview, event_type)

	_last_phase = phase


func _extract_preview(payload: Dictionary) -> String:
	var raw: String = ""
	if payload.has("text"):
		raw = str(payload.get("text", ""))
	elif payload.has("operator"):
		raw = str(payload.get("operator", ""))
	elif payload.has("reason"):
		raw = str(payload.get("reason", ""))
	if raw.length() > MAX_PREVIEW_CHARS:
		raw = raw.substr(0, MAX_PREVIEW_CHARS) + "..."
	return raw


func _spawn_packet(from_pos: Vector3, to_pos: Vector3, preview: String, event_type: String) -> void:
	var color: Color = EVENT_COLORS.get(event_type, EVENT_COLORS["normal"])

	var packet := MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = 0.16
	sphere.height = 0.32
	packet.mesh = sphere
	packet.position = from_pos
	var material := StandardMaterial3D.new()
	material.emission_enabled = true
	material.albedo_color = color
	material.emission = color
	material.emission_energy_multiplier = 3.0
	packet.set_surface_override_material(0, material)
	add_child(packet)

	var label: Label3D = null
	if not preview.is_empty():
		label = Label3D.new()
		label.text = preview
		label.pixel_size = 0.009
		label.font_size = 26
		label.outline_size = 6
		label.modulate = color
		label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
		label.position = from_pos + Vector3(0, 0.32, 0)
		add_child(label)

	var move_tween := create_tween()
	move_tween.tween_property(packet, "position", to_pos, TRAVEL_DURATION_SEC)
	if label != null:
		var label_move_tween := create_tween()
		label_move_tween.tween_property(label, "position", to_pos + Vector3(0, 0.32, 0), TRAVEL_DURATION_SEC)

	move_tween.finished.connect(func():
		var fade_tween := create_tween()
		fade_tween.tween_property(packet, "scale", Vector3.ZERO, FADE_DURATION_SEC)
		fade_tween.finished.connect(packet.queue_free)
		if label != null:
			var label_fade_tween := create_tween()
			label_fade_tween.tween_property(label, "modulate:a", 0.0, FADE_DURATION_SEC)
			label_fade_tween.finished.connect(label.queue_free)
	)
