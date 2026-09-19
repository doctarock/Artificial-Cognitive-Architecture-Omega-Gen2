extends Node3D
## The spine of the whole visualization: nine fixed "stations," one per
## phase that appears in cycle_events - eight from the actual cognitive
## cycle (specs.md's ten steps, condensed to the eight that appear there),
## laid out in a straight line and connected by pipes, plus a ninth,
## visually set apart, for idle-time pattern synthesis (`steps::synthesize`),
## which is self-triggered rather than a fixed step in every tick's
## sequence. Each station pulses live as its phase's events arrive -
## watching stations light up in sequence *is* watching the cycle run,
## which is the whole point of this view.
##
## Deliberately laid out along a single axis (not a curve) so every
## connecting pipe is a simple axis-aligned cylinder - no general 3D
## rotation math, which is exactly what caused a real, hard-to-spot sign
## error in an earlier hand-computed camera transform.

const PHASES := ["predict", "observe", "compare", "coalition", "broadcast", "executive", "act", "learn"]
const PHASE_LABELS := {
	"predict": "Predict",
	"observe": "Observe",
	"compare": "Compare",
	"coalition": "Coalition",
	"broadcast": "Broadcast",
	"executive": "Executive\n(SOAR)",
	"act": "Act",
	"learn": "Learn",
	"synthesize": "Synthesize\n(idle-time)",
}
## Not part of the fixed per-tick sequence - offset below the main line
## rather than appended to it, so its station doesn't visually imply "runs
## after Learn every tick" the way the other eight genuinely do.
const SIDE_PHASE := "synthesize"
const SIDE_PHASE_DROP := 1.6
const SPACING := 3.0
const PULSE_UP_SEC := 0.08
const PULSE_DOWN_SEC := 0.55
const IDLE_ENERGY := 0.25

const EVENT_COLORS := {
	"normal": Color(0.3, 0.65, 1.0),
	"impasse": Color(1.0, 0.6, 0.15),
	"escalation": Color(1.0, 0.2, 0.2),
	"error": Color(1.0, 0.0, 0.0),
}

var _stations: Dictionary = {}  # phase (String) -> MeshInstance3D
var _attention_indicator: MeshInstance3D = null
var _attention_in_flight: bool = false


func _ready() -> void:
	ApiClient.event_received.connect(_on_event_received)
	ApiClient.snapshot_updated.connect(_on_snapshot_updated)
	_build_pipeline()

	var title := Label3D.new()
	title.text = "The Cognitive Cycle"
	title.position = Vector3(0, 1.3, 0)
	title.pixel_size = 0.014
	title.font_size = 40
	title.outline_size = 8
	title.modulate = Color(0.6, 0.75, 1.0)
	title.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	add_child(title)


func has_station(phase: String) -> bool:
	return _stations.has(phase)


## The X position of a named station - other views (Working Memory ring,
## the Tier 3 indicator, the traveling thought-flow packets) hang directly
## below/between specific stations, so this is exposed rather than
## duplicated.
func station_x(phase: String) -> float:
	var index: int = PHASES.find(phase)
	if index < 0:
		return 0.0
	return _start_x() + index * SPACING


func _start_x() -> float:
	return -(PHASES.size() - 1) * SPACING / 2.0


func _build_pipeline() -> void:
	var previous_pos: Vector3 = Vector3.ZERO
	var has_previous := false
	for i in PHASES.size():
		var phase: String = PHASES[i]
		var pos := Vector3(_start_x() + i * SPACING, 0.0, 0.0)
		_stations[phase] = _create_station(phase, pos)
		if has_previous:
			_create_horizontal_pipe(previous_pos, pos)
		previous_pos = pos
		has_previous = true

	# Synthesize hangs directly below Learn (same X) rather than joining the
	# main line, so its connecting pipe is a plain vertical `connect_down` -
	# no new rotation math - while still reading as visually distinct from
	# the fixed eight-step sequence.
	var learn_pos := Vector3(station_x("learn"), 0.0, 0.0)
	_stations[SIDE_PHASE] = _create_station(SIDE_PHASE, learn_pos + Vector3(0, -SIDE_PHASE_DROP, 0))
	connect_down("learn", SIDE_PHASE_DROP)


## A low-poly faceted orb rather than a flat box - few enough segments to
## still read as a cut gem (matches the "glowing node" language every other
## view in this scene uses - Working Memory's spheres, the LLM chain, the
## memory motes) instead of a smooth, characterless ball.
const STATION_RADIUS := 0.68
const STATION_RADIAL_SEGMENTS := 8
const STATION_RINGS := 5


func _create_station(phase: String, pos: Vector3) -> MeshInstance3D:
	var mesh_instance := MeshInstance3D.new()
	mesh_instance.name = "station_%s" % phase
	var gem := SphereMesh.new()
	gem.radius = STATION_RADIUS
	gem.height = STATION_RADIUS * 2.0
	gem.radial_segments = STATION_RADIAL_SEGMENTS
	gem.rings = STATION_RINGS
	mesh_instance.mesh = gem
	mesh_instance.position = pos

	var material := StandardMaterial3D.new()
	material.shading_mode = BaseMaterial3D.SHADING_MODE_PER_PIXEL
	material.emission_enabled = true
	material.albedo_color = Color(0.12, 0.12, 0.16)
	material.emission = EVENT_COLORS["normal"]
	material.emission_energy_multiplier = IDLE_ENERGY
	material.metallic = 0.6
	material.roughness = 0.3
	material.rim_enabled = true
	material.rim = 0.35
	mesh_instance.set_surface_override_material(0, material)
	add_child(mesh_instance)

	var label := Label3D.new()
	label.text = PHASE_LABELS.get(phase, phase)
	label.position = pos + Vector3(0, STATION_RADIUS + 0.35, 0)
	label.pixel_size = 0.01
	label.font_size = 40
	label.outline_size = 8
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	add_child(label)

	return mesh_instance


## Stations are laid out purely along X, so every pipe between them is a
## straight horizontal segment - a default Y-aligned CylinderMesh only
## needs a fixed 90-degree roll onto its side, never a general rotation.
func _create_horizontal_pipe(a: Vector3, b: Vector3) -> void:
	var mesh_instance := MeshInstance3D.new()
	var cylinder := CylinderMesh.new()
	cylinder.height = a.distance_to(b)
	cylinder.top_radius = 0.045
	cylinder.bottom_radius = 0.045
	mesh_instance.mesh = cylinder
	mesh_instance.position = (a + b) / 2.0
	mesh_instance.rotation_degrees = Vector3(0, 0, 90)

	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.25, 0.25, 0.3)
	mesh_instance.set_surface_override_material(0, material)
	add_child(mesh_instance)


## Connects a station straight down to a child view hanging below it (the
## Working Memory ring under Broadcast, the Tier 3 indicator under Act).
## Also purely axis-aligned - a default vertical CylinderMesh needs no
## rotation at all.
func connect_down(phase: String, drop_height: float) -> void:
	var station: MeshInstance3D = _stations.get(phase)
	if station == null:
		return
	var mesh_instance := MeshInstance3D.new()
	var cylinder := CylinderMesh.new()
	cylinder.height = drop_height
	cylinder.top_radius = 0.045
	cylinder.bottom_radius = 0.045
	mesh_instance.mesh = cylinder
	mesh_instance.position = station.position + Vector3(0, -drop_height / 2.0, 0)
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.25, 0.25, 0.3)
	mesh_instance.set_surface_override_material(0, material)
	add_child(mesh_instance)


func _on_event_received(data: Dictionary) -> void:
	var phase: String = data.get("phase", "")
	if not _stations.has(phase):
		return
	var event_type: String = data.get("event_type", "normal")
	var color: Color = EVENT_COLORS.get(event_type, EVENT_COLORS["normal"])
	_pulse_station(_stations[phase], color)


## Fades the pulse's *brightness* back to idle and its *color* back to
## normal, in parallel, over the same down-tween - without the color half,
## a station that ever received a non-normal (impasse/escalation/error)
## event stayed visibly tinted that color forever, since only
## `emission_energy_multiplier` was ever reset, never `emission` itself.
## Confirmed live: Observe only ever emits `error`-kind events (its
## successful work shows up as Compare's own event instead, never a
## "normal" Observe one), so a single historical embedding hiccup left it
## permanently red with no future event of any other color ever able to
## clear it - this fix doesn't depend on Observe ever emitting a different
## event type, it just makes the pulse actually finish fading instead of
## stopping halfway.
func _pulse_station(station: MeshInstance3D, color: Color) -> void:
	var material: StandardMaterial3D = station.get_surface_override_material(0)
	material.emission = color
	var tween := create_tween()
	tween.tween_property(material, "emission_energy_multiplier", 3.5, PULSE_UP_SEC)
	tween.set_parallel(true)
	tween.tween_property(material, "emission_energy_multiplier", IDLE_ENERGY, PULSE_DOWN_SEC)
	tween.tween_property(material, "emission", EVENT_COLORS["normal"], PULSE_DOWN_SEC)


func _on_snapshot_updated(data: Dictionary) -> void:
	var attention_in_flight: bool = data.get("attention_in_flight", false)
	if _attention_in_flight != attention_in_flight:
		_attention_in_flight = attention_in_flight
		_update_attention_indicator()


func _update_attention_indicator() -> void:
	if not _attention_indicator:
		_create_attention_indicator()
	
	if _attention_indicator:
		var material: StandardMaterial3D = _attention_indicator.get_surface_override_material(0)
		var label: Label3D = _attention_indicator.get_meta("label")
		
		if _attention_in_flight:
			material.emission_energy_multiplier = 3.0
			material.emission = Color(0.8, 0.4, 1.0)  # Purple glow for the attention model's live call
			if label:
				label.visible = true
				_pulse_attention_indicator()
		else:
			material.emission_energy_multiplier = IDLE_ENERGY
			material.emission = Color(0.3, 0.3, 0.4)
			if label:
				label.visible = false


func _pulse_attention_indicator() -> void:
	# Stops the recursive chain the instant the model call finishes, rather
	# than pulsing forever - `_on_snapshot_updated` only calls this once,
	# on the false->true transition, so nothing else would ever stop it.
	if not _attention_in_flight or not _attention_indicator:
		return
	var material: StandardMaterial3D = _attention_indicator.get_surface_override_material(0)
	var tween := create_tween()
	# Sequential by default (no set_parallel) - up then down then recurse.
	# `set_parallel(true)` applies to every tweener added afterward, not
	# just the next one, so putting it here made the down-tween and the
	# recursive callback both start alongside the up-tween instead of
	# after it, effectively re-triggering the "pulse" immediately instead
	# of every ~0.8s.
	tween.tween_property(material, "emission_energy_multiplier", 4.0, 0.3)
	tween.tween_property(material, "emission_energy_multiplier", 2.0, 0.5)
	tween.tween_callback(_pulse_attention_indicator)


func _create_attention_indicator() -> void:
	# Create an indicator near the Broadcast station (where attention happens)
	var broadcast_x: float = station_x("broadcast")
	var pos := Vector3(broadcast_x, 1.5, 0)
	
	var mesh_instance := MeshInstance3D.new()
	mesh_instance.name = "attention_indicator"
	var gem := SphereMesh.new()
	gem.radius = 0.35
	gem.height = 0.7
	gem.radial_segments = 8
	gem.rings = 4
	mesh_instance.mesh = gem
	mesh_instance.position = pos
	
	var material := StandardMaterial3D.new()
	material.shading_mode = BaseMaterial3D.SHADING_MODE_PER_PIXEL
	material.emission_enabled = true
	material.albedo_color = Color(0.12, 0.12, 0.16)
	material.emission = Color(0.3, 0.3, 0.4)
	material.emission_energy_multiplier = IDLE_ENERGY
	material.metallic = 0.6
	material.roughness = 0.3
	material.rim_enabled = true
	material.rim = 0.35
	mesh_instance.set_surface_override_material(0, material)
	add_child(mesh_instance)
	
	_attention_indicator = mesh_instance
	
	# Add a label
	var label := Label3D.new()
	label.text = "Attention"
	label.position = pos + Vector3(0, 0.6, 0)
	label.pixel_size = 0.01
	label.font_size = 36
	label.outline_size = 6
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	label.modulate = Color(0.8, 0.4, 1.0)
	label.visible = false
	add_child(label)
	mesh_instance.set_meta("label", label)
