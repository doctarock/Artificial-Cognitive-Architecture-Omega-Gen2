extends Node3D
## Long-term memory as three living mote-clouds rather than growing bar
## columns: Episodic, Semantic, and Self each get a small orbiting cluster
## of particles, density (not height) driven by how many graph objects
## currently carry that `MemoryRole` tag (log-scaled so growth stays
## readable indefinitely rather than needing a rescale). Arced below Learn -
## where new episodic memories and reinforced edges actually get written,
## see `loop_actor.rs`'s `CyclePhase::Learn` handling - and beamed to it,
## the same "arc near the relevant station, connected by a beam" language
## `tier_indicator.gd` uses for the reasoning chain, so this reads as part
## of the same architecture rather than a satellite floating off to the
## side. A visible, standing record of what the engine has accumulated,
## distinct from Working Memory's transient ring.

const ROLES := ["episodic", "semantic", "self_memory", "candidates"]
const ROLE_LABELS := {"episodic": "Episodic", "semantic": "Semantic", "self_memory": "Self", "candidates": "Unconfirmed"}
const ROLE_COLORS := {
	"episodic": Color(0.3, 0.9, 0.5),
	"semantic": Color(1.0, 0.85, 0.3),
	"self_memory": Color(0.4, 0.8, 1.0),
	# Deliberately duller than the confirmed-memory colors above - reads
	# as "provisional," mirroring the desaturation `working_memory_ring.gd`
	# applies to Candidate nodes still awaiting the promotion gate.
	"candidates": Color(0.75, 0.75, 0.4),
}

const STATION := "learn"
const STATION_Y := 0.0
const ARC_RADIUS := 1.6
const ARC_SPREAD_DEG := 110.0
const CLUSTER_Y := -2.2
const MAX_PARTICLES := 50
## Reference count a cluster's density is scaled against - not a hard cap
## (density keeps creeping toward 1.0, never resets), just where the log
## curve is calibrated to read as "nearly full."
const REFERENCE_COUNT := 500
const MIN_RATIO := 0.05
const BEAM_COLOR := Color(0.5, 0.5, 0.55)
const BURST_FLASH_ENERGY := 4.0
const BURST_FLASH_SEC := 0.6
const PACKET_TRAVEL_SEC := 0.5
const PACKET_FADE_SEC := 0.3

@onready var _pipeline: Node3D = get_parent().get_node("PipelineView")

var _clusters: Dictionary = {}  # role -> {particles, core, label, pos}
var _learn_pos: Vector3 = Vector3.ZERO
## `-1` means "no snapshot seen yet" - the first real snapshot just
## establishes the baseline count per role, it never bursts (every role
## would otherwise burst on startup from 0 to whatever's already accumulated).
var _previous_counts: Dictionary = {}  # role -> int


func _ready() -> void:
	ApiClient.snapshot_updated.connect(_on_snapshot_updated)
	_build_clusters()

	var anchor_x: float = _pipeline.station_x(STATION) if _pipeline.has_station(STATION) else 0.0
	var title := Label3D.new()
	title.text = "Long-Term Memory"
	title.position = Vector3(anchor_x, 1.2, 0)
	title.pixel_size = 0.013
	title.font_size = 38
	title.outline_size = 8
	title.modulate = Color(0.7, 1.0, 0.8)
	title.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	add_child(title)


func _build_clusters() -> void:
	var anchor_x: float = _pipeline.station_x(STATION) if _pipeline.has_station(STATION) else 0.0
	var target := Vector3(anchor_x, STATION_Y, 0)
	_learn_pos = target
	var n: int = ROLES.size()
	for i in n:
		var role: String = ROLES[i]
		var angle_deg: float = -ARC_SPREAD_DEG / 2.0 + ARC_SPREAD_DEG * i / float(n - 1)
		var angle_rad: float = deg_to_rad(angle_deg)
		var pos := Vector3(anchor_x + sin(angle_rad) * ARC_RADIUS, CLUSTER_Y, cos(angle_rad) * ARC_RADIUS * 0.5)
		_clusters[role] = _create_cluster(role, pos, target)


func _create_cluster(role: String, pos: Vector3, target: Vector3) -> Dictionary:
	var color: Color = ROLE_COLORS.get(role, Color.WHITE)

	var core := MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = 0.18
	sphere.height = 0.36
	core.mesh = sphere
	core.position = pos
	var core_material := StandardMaterial3D.new()
	core_material.shading_mode = BaseMaterial3D.SHADING_MODE_PER_PIXEL
	core_material.emission_enabled = true
	core_material.albedo_color = color
	core_material.emission = color
	core_material.emission_energy_multiplier = 1.2
	core.set_surface_override_material(0, core_material)
	add_child(core)

	var particles := GPUParticles3D.new()
	particles.position = pos
	particles.amount = MAX_PARTICLES
	particles.lifetime = 3.0
	particles.amount_ratio = MIN_RATIO
	add_child(particles)

	var mote_mesh := SphereMesh.new()
	mote_mesh.radius = 0.045
	mote_mesh.height = 0.09
	var mote_material := StandardMaterial3D.new()
	mote_material.emission_enabled = true
	mote_material.emission = color
	mote_material.emission_energy_multiplier = 2.0
	mote_mesh.surface_set_material(0, mote_material)
	particles.draw_pass_1 = mote_mesh

	var process_material := ParticleProcessMaterial.new()
	process_material.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_SPHERE_SURFACE
	process_material.emission_sphere_radius = 0.5
	process_material.direction = Vector3(0, 1, 0)
	process_material.spread = 180.0
	process_material.initial_velocity_min = 0.05
	process_material.initial_velocity_max = 0.15
	process_material.gravity = Vector3.ZERO
	process_material.orbit_velocity_min = 0.08
	process_material.orbit_velocity_max = 0.18
	process_material.scale_min = 0.6
	process_material.scale_max = 1.4
	process_material.color = color
	particles.process_material = process_material

	var label := Label3D.new()
	label.text = "%s: 0" % ROLE_LABELS.get(role, role)
	label.position = pos + Vector3(0, 0.55, 0)
	label.pixel_size = 0.01
	label.font_size = 30
	label.outline_size = 8
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	add_child(label)

	var beam := MeshInstance3D.new()
	var cylinder := CylinderMesh.new()
	cylinder.top_radius = 0.03
	cylinder.bottom_radius = 0.03
	beam.mesh = cylinder
	var beam_material := StandardMaterial3D.new()
	beam_material.emission_enabled = true
	beam_material.albedo_color = BEAM_COLOR
	beam_material.emission = BEAM_COLOR
	beam_material.emission_energy_multiplier = 0.6
	beam.set_surface_override_material(0, beam_material)
	add_child(beam)
	_position_beam(beam, pos, target)

	return {"particles": particles, "core": core, "label": label, "pos": pos}


func _position_beam(beam: MeshInstance3D, from_pos: Vector3, to_pos: Vector3) -> void:
	var direction := to_pos - from_pos
	var length := direction.length()
	var cylinder: CylinderMesh = beam.mesh
	cylinder.height = max(length, 0.01)
	beam.position = (from_pos + to_pos) / 2.0
	_orient_between(beam, direction, length)


## Copied from `working_memory_ring.gd::_orient_between` (also reused by
## `tier_indicator.gd`) rather than reinvented - see that function's doc
## comment for why cross-product/angle-axis, not a hand-derived basis.
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


func _on_snapshot_updated(data: Dictionary) -> void:
	var counts: Dictionary = data.get("memory_counts", {})
	for role in ROLES:
		if not _clusters.has(role):
			continue
		var count: int = int(counts.get(role, 0))
		var ratio: float = clampf(log(count + 1) / log(REFERENCE_COUNT + 1), MIN_RATIO, 1.0)
		var resting_energy := 1.0 + ratio * 2.0

		var entry: Dictionary = _clusters[role]
		var particles: GPUParticles3D = entry["particles"]
		particles.amount_ratio = ratio
		particles.emitting = true

		var core: MeshInstance3D = entry["core"]
		var core_material: StandardMaterial3D = core.get_surface_override_material(0)
		core_material.emission_energy_multiplier = resting_energy

		var label: Label3D = entry["label"]
		label.text = "%s: %d" % [ROLE_LABELS.get(role, role), count]

		if _previous_counts.has(role) and count > _previous_counts[role]:
			_burst(core, resting_energy, entry["pos"], ROLE_COLORS.get(role, Color.WHITE))
		_previous_counts[role] = count


## Fires exactly when a role's count actually grew this snapshot (a new
## memory just got written) - a core flash plus a small packet traveling
## from Learn to the cluster, matching `thought_flow.gd`'s "a packet moving
## shows what happened, not just that something happened" idiom, applied to
## the one kind of activity that isn't a discrete engine event (memory
## formation is only ever visible as a count delta between snapshots).
func _burst(core: MeshInstance3D, resting_energy: float, cluster_pos: Vector3, color: Color) -> void:
	var material: StandardMaterial3D = core.get_surface_override_material(0)
	var tween := create_tween()
	tween.tween_property(material, "emission_energy_multiplier", resting_energy + BURST_FLASH_ENERGY, 0.15)
	tween.tween_property(material, "emission_energy_multiplier", resting_energy, BURST_FLASH_SEC)

	var packet := MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = 0.1
	sphere.height = 0.2
	packet.mesh = sphere
	packet.position = _learn_pos
	var packet_material := StandardMaterial3D.new()
	packet_material.emission_enabled = true
	packet_material.albedo_color = color
	packet_material.emission = color
	packet_material.emission_energy_multiplier = 3.0
	packet.set_surface_override_material(0, packet_material)
	add_child(packet)

	var move_tween := create_tween()
	move_tween.tween_property(packet, "position", cluster_pos, PACKET_TRAVEL_SEC)
	move_tween.finished.connect(func():
		var fade_tween := create_tween()
		fade_tween.tween_property(packet, "scale", Vector3.ZERO, PACKET_FADE_SEC)
		fade_tween.finished.connect(packet.queue_free)
	)
