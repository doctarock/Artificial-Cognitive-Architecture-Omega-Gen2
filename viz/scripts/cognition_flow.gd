extends Node3D
## The one *sustained* effect in the visualization, as opposed to every other
## view's instant pulses/packets: while the engine is genuinely blocked
## awaiting the embedding client's HTTP round trip (Observe, between Predict
## and Compare - the single real I/O stall in an otherwise-instant tick), a
## field of violet motes streams inward from the surrounding space and
## converges on the Predict station. A warm call reads as a brief shimmer; a
## slow/cold-start Ollama model load reads as a sustained, visibly converging
## flow - the "it's spinning something up" cue nothing else in this scene can
## show, since no discrete event fires for the entire span of that wait (see
## `loop_actor.rs`'s `embedding_in_flight` field, published true right before
## the await and false right after).

const RADIUS := 4.5
const STATION_Y := 0.55
const FLOW_COLOR := Color(0.78, 0.35, 1.0)
const GLOW_RAMP_SEC := 0.25
const GLOW_PEAK_ENERGY := 3.0

@onready var _pipeline: Node3D = get_parent().get_node("PipelineView")

var _particles: GPUParticles3D
var _glow: OmniLight3D
var _active := false


func _ready() -> void:
	ApiClient.snapshot_updated.connect(_on_snapshot_updated)

	var predict_pos := Vector3(_pipeline.station_x("predict"), STATION_Y, 0)
	_build_particles(predict_pos)
	_build_glow(predict_pos)


func _build_particles(predict_pos: Vector3) -> void:
	_particles = GPUParticles3D.new()
	_particles.position = predict_pos
	_particles.amount = 90
	_particles.lifetime = 1.6
	_particles.emitting = false
	_particles.one_shot = false

	var mesh := SphereMesh.new()
	mesh.radius = 0.035
	mesh.height = 0.07
	var mote_material := StandardMaterial3D.new()
	mote_material.emission_enabled = true
	mote_material.emission = FLOW_COLOR
	mote_material.emission_energy_multiplier = 2.2
	mote_material.vertex_color_use_as_albedo = true
	mesh.surface_set_material(0, mote_material)
	_particles.draw_pass_1 = mesh

	var process_material := ParticleProcessMaterial.new()
	process_material.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_SPHERE_SURFACE
	process_material.emission_sphere_radius = RADIUS
	process_material.direction = Vector3(0, 0, 1)
	process_material.spread = 180.0
	process_material.initial_velocity_min = 0.05
	process_material.initial_velocity_max = 0.2
	# Acceleration toward the emission center (negative = inward), not a
	# fixed inward velocity - motes drift lazily off the shell at first and
	# visibly speed up as they close in on Predict, which reads far more
	# like "being pulled in" than a uniform inbound drift would.
	process_material.radial_accel_min = -2.5
	process_material.radial_accel_max = -4.0
	process_material.gravity = Vector3.ZERO
	process_material.scale_min = 0.6
	process_material.scale_max = 1.4
	process_material.color = FLOW_COLOR
	_particles.process_material = process_material
	add_child(_particles)


func _build_glow(predict_pos: Vector3) -> void:
	_glow = OmniLight3D.new()
	_glow.position = predict_pos
	_glow.light_color = FLOW_COLOR
	_glow.light_energy = 0.0
	_glow.omni_range = 6.0
	add_child(_glow)


func _on_snapshot_updated(data: Dictionary) -> void:
	var in_flight: bool = data.get("embedding_in_flight", false)
	if in_flight == _active:
		return
	_active = in_flight
	_particles.emitting = in_flight

	var glow_tween := create_tween()
	glow_tween.tween_property(_glow, "light_energy", GLOW_PEAK_ENERGY if in_flight else 0.0, GLOW_RAMP_SEC)
