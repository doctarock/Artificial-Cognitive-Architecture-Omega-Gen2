extends Node3D
## Every individually-configured LLM in the reasoning chain, made visible as
## a fanned constellation rather than a flat row: Tier 1's several divergent
## candidates arc wide above Coalition (where they're actually sampled -
## see `cognitive_core::reflect`), Tier 2's tighter, closer arc nests inside
## them, and Tier 3/4's lone escalation seats hang above Executive, Tier 4
## highest/most set-apart since it's the rarer, larger escalation. The shape
## itself tells the ladder's real story - many parallel first-pass takes
## narrowing down to one deciding call - not just decoration. Sphere size
## follows the same tier-prominence story, matching the Working Memory
## ring's "size follows a meaningful property" idiom rather than uniform
## nodes.
##
## Deliberately above the pipeline, not below it like every other station-
## anchored sub-view - Working Memory already occupies the space below
## Broadcast, close enough to Coalition/Executive that this cluster's wide
## arcs would otherwise visually overlap it.
##
## Tier 0 (the embedding model) is the one exception: it hangs *below*
## Learn, deeper than the Long-Term Memory clusters (see memory_silos.gd),
## with its beam passing right through that neighborhood on its way up -
## the same physical client backs Observe's per-tick embedding too (see
## `cognition_flow.gd`), but this is deliberately the one place it's shown
## with a real, identified node rather than an anonymous particle effect,
## specifically because `steps::synthesize` calls it directly to embed
## every newly-formed memory - the literal "encoding LLM" for memory.
##
## Each node is a real configured model (real name, live `in_flight` state
## from `EngineSnapshot.active_models`). A thin beam from each node to its
## home station brightens for exactly as long as that specific model is
## mid-call, so the *interaction* with that phase is visible, not just the
## node's own color - oriented with the same cross-product `Basis(axis,
## angle)` construction Working Memory's associative-edge lines use (see
## `working_memory_ring.gd::_orient_between`'s doc comment for why: a
## fixed, well-defined rotation formula, not a hand-derived one, after a
## past hand-derived camera transform produced a real sign-error bug).

## Per-tier arc placement: which station the group hangs below, how wide
## (radius) and how far around (spread_deg) its fan spreads, its vertical
## drop, and its sphere radius (bigger/rarer models read as bigger nodes).
const TIER_GROUPS := {
	"t0": {"station": "learn", "radius": 0.0, "spread_deg": 0.0, "y": -3.2, "node_radius": 0.26},
	"t1": {"station": "coalition", "radius": 2.6, "spread_deg": 150.0, "y": 2.4, "node_radius": 0.22},
	"t2": {"station": "coalition", "radius": 1.5, "spread_deg": 100.0, "y": 1.6, "node_radius": 0.28},
	"t3": {"station": "executive", "radius": 0.9, "spread_deg": 40.0, "y": 2.6, "node_radius": 0.34},
	"t4": {"station": "executive", "radius": 0.9, "spread_deg": 40.0, "y": 3.4, "node_radius": 0.42},
}
const FALLBACK_GROUP := {"station": "executive", "radius": 1.5, "spread_deg": 100.0, "y": 2.0, "node_radius": 0.3}
const STATION_Y := 0.0

const IDLE_COLOR := Color(0.3, 1.0, 0.4)
const ACTIVE_COLOR := Color(1.0, 0.25, 0.25)
const BEAM_IDLE_COLOR := Color(0.25, 0.25, 0.3)
const BEAM_ACTIVE_COLOR := Color(1.0, 0.75, 0.25)
const BEAM_IDLE_ENERGY := 0.15
const BEAM_ACTIVE_ENERGY := 3.0
const BEAM_TWEEN_SEC := 0.3

## Per-socket converging-motes + glow, the same sustained-effect idiom
## `cognition_flow.gd` uses for the embedding call - a color/brightness
## change alone reads as "slightly different," not "actively working,"
## which is exactly the complaint this exists to fix. Unlike that view's
## single global effect anchored on one station, every in-flight model here
## gets its own small local field, since several tiers' sockets can be
## mid-call at once.
const PARTICLE_RADIUS := 0.9
const PARTICLE_COLOR := Color(1.0, 0.6, 0.2)
const PARTICLE_AMOUNT := 28
const PARTICLE_LIFETIME := 1.0
const GLOW_PEAK_ENERGY := 2.2
const GLOW_TWEEN_SEC := 0.25

@onready var _pipeline: Node3D = get_parent().get_node("PipelineView")

var _order: Array = []  # model id (String), in first-seen order
var _sockets: Dictionary = {}  # model id (String) -> {socket, label, beam, beam_material, tier}


func _ready() -> void:
	ApiClient.snapshot_updated.connect(_on_snapshot_updated)


func _on_snapshot_updated(data: Dictionary) -> void:
	var active_models: Array = data.get("active_models", [])
	var saw_new_model := false
	for status_variant in active_models:
		var status: Dictionary = status_variant
		var id: String = str(status.get("id", ""))
		if id.is_empty():
			continue
		if not _sockets.has(id):
			_sockets[id] = _create_socket(status)
			_order.append(id)
			saw_new_model = true
		_update_socket(_sockets[id], status)
	if saw_new_model:
		_relayout()


func _create_socket(status: Dictionary) -> Dictionary:
	var tier: String = str(status.get("tier", ""))
	var group_config: Dictionary = TIER_GROUPS.get(tier, FALLBACK_GROUP)

	var socket := MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = group_config["node_radius"]
	sphere.height = sphere.radius * 2.0
	socket.mesh = sphere
	var material := StandardMaterial3D.new()
	material.shading_mode = BaseMaterial3D.SHADING_MODE_PER_PIXEL
	material.emission_enabled = true
	socket.set_surface_override_material(0, material)
	add_child(socket)

	var label := Label3D.new()
	label.pixel_size = 0.01
	label.font_size = 28
	label.outline_size = 8
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	add_child(label)

	var beam := MeshInstance3D.new()
	var cylinder := CylinderMesh.new()
	cylinder.top_radius = 0.035
	cylinder.bottom_radius = 0.035
	beam.mesh = cylinder
	var beam_material := StandardMaterial3D.new()
	beam_material.emission_enabled = true
	beam_material.albedo_color = BEAM_IDLE_COLOR
	beam_material.emission = BEAM_IDLE_COLOR
	beam_material.emission_energy_multiplier = BEAM_IDLE_ENERGY
	beam.set_surface_override_material(0, beam_material)
	add_child(beam)

	var particles := _build_particles()
	var glow := _build_glow()

	return {
		"socket": socket, "label": label, "beam": beam, "beam_material": beam_material,
		"tier": tier, "particles": particles, "glow": glow, "was_in_flight": false,
	}


func _build_particles() -> GPUParticles3D:
	var particles := GPUParticles3D.new()
	particles.amount = PARTICLE_AMOUNT
	particles.lifetime = PARTICLE_LIFETIME
	particles.emitting = false
	particles.one_shot = false

	var mesh := SphereMesh.new()
	mesh.radius = 0.025
	mesh.height = 0.05
	var mote_material := StandardMaterial3D.new()
	mote_material.emission_enabled = true
	mote_material.emission = PARTICLE_COLOR
	mote_material.emission_energy_multiplier = 2.2
	mote_material.vertex_color_use_as_albedo = true
	mesh.surface_set_material(0, mote_material)
	particles.draw_pass_1 = mesh

	var process_material := ParticleProcessMaterial.new()
	process_material.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_SPHERE_SURFACE
	process_material.emission_sphere_radius = PARTICLE_RADIUS
	process_material.direction = Vector3(0, 0, 1)
	process_material.spread = 180.0
	process_material.initial_velocity_min = 0.05
	process_material.initial_velocity_max = 0.2
	process_material.radial_accel_min = -2.5
	process_material.radial_accel_max = -4.0
	process_material.gravity = Vector3.ZERO
	process_material.scale_min = 0.6
	process_material.scale_max = 1.4
	process_material.color = PARTICLE_COLOR
	particles.process_material = process_material
	add_child(particles)
	return particles


func _build_glow() -> OmniLight3D:
	var glow := OmniLight3D.new()
	glow.light_color = PARTICLE_COLOR
	glow.light_energy = 0.0
	glow.omni_range = 2.5
	add_child(glow)
	return glow


## Repositions every known socket/label/beam from scratch - run once per
## snapshot that introduces a previously-unseen model, since a new arrival
## can change how many nodes share its tier's arc (and therefore where
## every sibling in that arc belongs), not just where it belongs itself.
func _relayout() -> void:
	var groups: Dictionary = {}  # tier -> Array of ids, in first-seen order
	for id in _order:
		var tier: String = _sockets[id]["tier"]
		if not groups.has(tier):
			groups[tier] = []
		groups[tier].append(id)

	for tier in groups.keys():
		var group_config: Dictionary = TIER_GROUPS.get(tier, FALLBACK_GROUP)
		var station: String = group_config["station"]
		var anchor_x: float = _pipeline.station_x(station) if _pipeline.has_station(station) else 0.0
		var radius: float = group_config["radius"]
		var spread_deg: float = group_config["spread_deg"]
		var y: float = group_config["y"]
		var target := Vector3(anchor_x, STATION_Y, 0)

		var ids: Array = groups[tier]
		var n: int = ids.size()
		for i in n:
			var angle_deg: float = 0.0 if n == 1 else -spread_deg / 2.0 + spread_deg * i / float(n - 1)
			var angle_rad: float = deg_to_rad(angle_deg)
			var pos := Vector3(anchor_x + sin(angle_rad) * radius, y, cos(angle_rad) * radius * 0.5)

			var entry: Dictionary = _sockets[ids[i]]
			(entry["socket"] as MeshInstance3D).position = pos
			(entry["label"] as Label3D).position = pos + Vector3(0, 0.55, 0)
			(entry["particles"] as GPUParticles3D).position = pos
			(entry["glow"] as OmniLight3D).position = pos
			_position_beam(entry["beam"], pos, target)


func _position_beam(beam: MeshInstance3D, from_pos: Vector3, to_pos: Vector3) -> void:
	var direction := to_pos - from_pos
	var length := direction.length()
	var cylinder: CylinderMesh = beam.mesh
	cylinder.height = max(length, 0.01)
	beam.position = (from_pos + to_pos) / 2.0
	_orient_between(beam, direction, length)


## Rotates a default Y-aligned cylinder to point along `direction` - copied
## from `working_memory_ring.gd::_orient_between` rather than reinvented,
## since these beams have the exact same "arbitrary point in space to
## another arbitrary point" shape its associative-edge lines already solve
## correctly. See that function's doc comment for why cross-product/angle-
## axis, not a hand-derived basis.
func _orient_between(mesh_instance: Node3D, direction: Vector3, length: float) -> void:
	if length < 0.0001:
		mesh_instance.transform.basis = Basis.IDENTITY
		return
	var dir_norm := direction / length
	var up := Vector3.UP
	if absf(dir_norm.dot(up)) > 0.999:
		up = Vector3.FORWARD
	var axis := up.cross(dir_norm)
	if axis.length() < 0.0001:
		mesh_instance.transform.basis = Basis.IDENTITY
		return
	axis = axis.normalized()
	var angle := acos(clampf(up.dot(dir_norm), -1.0, 1.0))
	mesh_instance.transform.basis = Basis(axis, angle)


func _update_socket(entry: Dictionary, status: Dictionary) -> void:
	var in_flight: bool = bool(status.get("in_flight", false))
	var model_label: String = str(status.get("label", "?"))

	var socket: MeshInstance3D = entry["socket"]
	var material: StandardMaterial3D = socket.get_surface_override_material(0)
	var socket_color := ACTIVE_COLOR if in_flight else IDLE_COLOR
	material.emission = socket_color
	material.emission_energy_multiplier = 2.5 if in_flight else 1.0

	var label: Label3D = entry["label"]
	label.text = "%s (active)" % model_label if in_flight else model_label

	var beam_material: StandardMaterial3D = entry["beam_material"]
	var beam_color := BEAM_ACTIVE_COLOR if in_flight else BEAM_IDLE_COLOR
	beam_material.albedo_color = beam_color
	beam_material.emission = beam_color
	var energy_tween := create_tween()
	energy_tween.tween_property(beam_material, "emission_energy_multiplier", BEAM_ACTIVE_ENERGY if in_flight else BEAM_IDLE_ENERGY, BEAM_TWEEN_SEC)

	# Gated on change, unlike the beam tween above - `emitting` should only
	# flip at the in-flight edges, not get reasserted every snapshot while a
	# long call is still running.
	if in_flight != entry["was_in_flight"]:
		entry["was_in_flight"] = in_flight
		(entry["particles"] as GPUParticles3D).emitting = in_flight
		var glow_tween := create_tween()
		glow_tween.tween_property(entry["glow"], "light_energy", GLOW_PEAK_ENERGY if in_flight else 0.0, GLOW_TWEEN_SEC)
