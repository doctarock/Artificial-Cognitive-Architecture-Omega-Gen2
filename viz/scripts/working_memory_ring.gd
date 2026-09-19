extends Node3D
## Renders the current Working Memory snapshot as glowing spheres arranged
## in a circle, positioned below the pipeline's "Broadcast" station (GWT's
## admission point - this ring *is* what broadcast just let through). Size
## follows activation, glow follows attention_score (meaningful here
## specifically because Working Memory only ever holds objects that already
## won the broadcast competition), color follows kind.

const RADIUS := 1.6
const MIN_SCALE := 0.3
const MAX_SCALE := 1.1
const PULSE_DURATION_SEC := 0.6

const KIND_COLORS := {
	"observation": Color(0.35, 0.55, 1.0),
	"thought": Color(0.6, 0.4, 1.0),
	"reflection": Color(0.75, 0.35, 0.95),
	"question": Color(1.0, 0.65, 0.25),
	"idea": Color(1.0, 0.85, 0.3),
	"goal": Color(0.3, 1.0, 0.6),
	"belief": Color(0.4, 0.8, 1.0),
	"intention": Color(1.0, 0.5, 0.5),
	"decision": Color(1.0, 0.3, 0.3),
	"memory": Color(0.3, 0.9, 0.5),
	"hypothesis": Color(0.8, 0.8, 0.4),
}
const DEFAULT_COLOR := Color(0.7, 0.7, 0.7)
const CANDIDATE_GRAY := Color(0.55, 0.55, 0.55)
const CANDIDATE_BLEND := 0.55
const MIN_CONFIDENCE_ALPHA := 0.15

const EDGE_KIND_COLORS := {
	"associative": Color(0.6, 0.9, 1.0),
	"contradicts": Color(1.0, 0.25, 0.25),
	"supports": Color(0.4, 1.0, 0.5),
	"causal": Color(0.7, 0.5, 1.0),
	"derived-from": Color(1.0, 0.85, 0.3),
	"subgoal-of": Color(0.3, 1.0, 0.6),
	"inhibitory": Color(0.9, 0.45, 0.15),
}
const DEFAULT_EDGE_COLOR := Color(0.6, 0.9, 1.0)

## GNW's attention-without-ignition set (`attended_not_ignited`) - Coalition
## candidates real enough to have competed but that never crossed
## `ignition_threshold`/lost out on capacity. Rendered just outside the real
## ring, dim and unlit, so they read as "present but not conscious" rather
## than as ordinary Working Memory members.
const GHOST_RADIUS := RADIUS + 0.6
const GHOST_ALPHA := 0.3
const GHOST_SCALE := 0.5

const DISPLACEMENT_TOAST_RISE := 0.8
const DISPLACEMENT_TOAST_FADE_SEC := 3.0

var _nodes: Dictionary = {}  # id (String) -> MeshInstance3D
var _edge_lines: Dictionary = {}  # "source_id->target_id" -> MeshInstance3D
var _ghost_nodes: Array = []  # MeshInstance3D, fully rebuilt every snapshot
## `EpochMillis` of the last `last_displacement` a toast was shown for -
## `-1` means "none shown yet." Unlike `goal_stack_view.gd`'s baseline
## suppression, the very first real displacement this panel ever sees is
## still worth announcing (it's a genuine one-time event, not pre-existing
## state from before this viewer connected).
var _previous_displacement_at: int = -1


func _ready() -> void:
	ApiClient.snapshot_updated.connect(_on_snapshot_updated)
	ApiClient.event_received.connect(_on_event_received)

	var title := Label3D.new()
	title.text = "Working Memory"
	title.position = Vector3(0, 2.0, 0)
	title.pixel_size = 0.012
	title.font_size = 34
	title.outline_size = 8
	title.modulate = Color(0.7, 0.85, 1.0)
	title.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	add_child(title)


func _on_snapshot_updated(data: Dictionary) -> void:
	var members: Array = data.get("working_memory", [])
	var seen_ids := {}

	for i in members.size():
		var member: Dictionary = members[i]
		var id: String = member.get("id", "")
		if id.is_empty():
			continue
		seen_ids[id] = true
		_upsert_node(id, member, i, members.size())

	for id in _nodes.keys():
		if not seen_ids.has(id):
			_nodes[id].queue_free()
			_nodes.erase(id)

	_update_edges(data.get("associative_edges", []))
	_update_ghost_nodes(data.get("attended_not_ignited", []))
	_maybe_show_displacement(data.get("last_displacement", null))


func _upsert_node(id: String, member: Dictionary, index: int, total: int) -> void:
	var mesh_instance: MeshInstance3D
	if _nodes.has(id):
		mesh_instance = _nodes[id]
	else:
		mesh_instance = _create_node_visual(id)
		_nodes[id] = mesh_instance

	var activation: float = member.get("activation_total", 0.0)
	var attention = member.get("attention_score", null)
	var kind: String = member.get("kind", "")
	var promotion_status: String = member.get("promotion_status", "confirmed")
	var confidence: float = member.get("confidence", 1.0)

	var scale_amount := clampf(remap(activation, 0.0, 3.0, MIN_SCALE, MAX_SCALE), MIN_SCALE, MAX_SCALE)
	mesh_instance.scale = Vector3.ONE * scale_amount

	var angle := (float(index) / float(max(total, 1))) * TAU
	mesh_instance.position = Vector3(cos(angle) * RADIUS, activation * 0.15, sin(angle) * RADIUS)

	var material: StandardMaterial3D = mesh_instance.get_surface_override_material(0)
	var base_color: Color = KIND_COLORS.get(kind, DEFAULT_COLOR)
	# An unconfirmed Candidate (still awaiting the promotion gate - see
	# steps::memory_formation) is deliberately desaturated toward gray
	# rather than given its full kind color, so "not fully formed yet"
	# reads at a glance without hiding which kind it is.
	if promotion_status == "candidate":
		base_color = base_color.lerp(CANDIDATE_GRAY, CANDIDATE_BLEND)
	var alpha := clampf(confidence, MIN_CONFIDENCE_ALPHA, 1.0)
	material.albedo_color = Color(base_color.r, base_color.g, base_color.b, alpha)
	var glow: float = 0.0 if attention == null else clampf(float(attention), 0.0, 2.0)
	material.emission = base_color
	material.emission_energy_multiplier = 0.5 + glow * 2.0

	mesh_instance.set_meta("text", member.get("text", ""))


func _create_node_visual(id: String) -> MeshInstance3D:
	var mesh_instance := MeshInstance3D.new()
	mesh_instance.name = "wm_%s" % id
	mesh_instance.mesh = SphereMesh.new()
	var material := StandardMaterial3D.new()
	material.shading_mode = BaseMaterial3D.SHADING_MODE_PER_PIXEL
	material.emission_enabled = true
	material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mesh_instance.set_surface_override_material(0, material)
	add_child(mesh_instance)
	return mesh_instance


## `attended_not_ignited` has no persistent identity from tick to tick (it's
## fully recomputed by Coalition every cycle, see the field's own doc
## comment on `snapshot.rs`) - so unlike `_nodes` above, there's no id-keyed
## upsert here: every snapshot just tears down last tick's ghosts and
## rebuilds fresh ones. Realistically single-digit counts, so the churn is
## cheap.
func _update_ghost_nodes(candidates: Array) -> void:
	for ghost in _ghost_nodes:
		ghost.queue_free()
	_ghost_nodes.clear()

	for i in candidates.size():
		var candidate: Dictionary = candidates[i]
		var kind: String = candidate.get("kind", "")
		var angle := (float(i) / float(max(candidates.size(), 1))) * TAU

		var mesh_instance := MeshInstance3D.new()
		mesh_instance.mesh = SphereMesh.new()
		mesh_instance.scale = Vector3.ONE * GHOST_SCALE
		mesh_instance.position = Vector3(cos(angle) * GHOST_RADIUS, 0, sin(angle) * GHOST_RADIUS)

		var material := StandardMaterial3D.new()
		material.shading_mode = BaseMaterial3D.SHADING_MODE_PER_PIXEL
		material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		var base_color: Color = KIND_COLORS.get(kind, DEFAULT_COLOR)
		material.albedo_color = Color(base_color.r, base_color.g, base_color.b, GHOST_ALPHA)
		mesh_instance.set_surface_override_material(0, material)
		mesh_instance.set_meta("text", candidate.get("text", ""))

		add_child(mesh_instance)
		_ghost_nodes.append(mesh_instance)


## GWT's rare, verified "X entered my awareness because it displaced Y"
## claim (`DisplacementSummary` - see `snapshot.rs`'s doc comment for what
## backs it). Event-like, not standing state, so a fading toast rather than
## a persistent element - reconstructs `DisplacementSummary::claim_text()`'s
## exact wording client-side from the `entrant_text`/`evicted_text` already
## on the wire.
func _maybe_show_displacement(displacement) -> void:
	if displacement == null:
		return
	var displacement_dict: Dictionary = displacement
	var at: int = int(displacement_dict.get("at", 0))
	if at == _previous_displacement_at:
		return
	_previous_displacement_at = at

	var entrant_text: String = displacement_dict.get("entrant_text", "")
	var evicted_text: String = displacement_dict.get("evicted_text", "")

	var toast := Label3D.new()
	toast.text = "%s entered my awareness because it displaced %s" % [entrant_text, evicted_text]
	toast.position = Vector3(0, 2.6, 0)
	toast.pixel_size = 0.009
	toast.font_size = 26
	toast.outline_size = 6
	toast.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	toast.modulate = Color(1.0, 0.9, 0.6)
	add_child(toast)

	var tween := create_tween()
	tween.tween_property(toast, "position", toast.position + Vector3(0, DISPLACEMENT_TOAST_RISE, 0), DISPLACEMENT_TOAST_FADE_SEC)
	tween.parallel().tween_property(toast, "modulate:a", 0.0, DISPLACEMENT_TOAST_FADE_SEC)
	tween.finished.connect(toast.queue_free)


func _on_event_received(data: Dictionary) -> void:
	var event_type: String = data.get("event_type", "")
	if event_type == "impasse" or event_type == "escalation":
		_pulse_all_nodes()


func _pulse_all_nodes() -> void:
	for mesh_instance in _nodes.values():
		var material: StandardMaterial3D = mesh_instance.get_surface_override_material(0)
		if material == null:
			continue
		var tween := create_tween()
		var original_energy := material.emission_energy_multiplier
		tween.tween_property(material, "emission_energy_multiplier", original_energy + 4.0, PULSE_DURATION_SEC * 0.3)
		tween.tween_property(material, "emission_energy_multiplier", original_energy, PULSE_DURATION_SEC * 0.7)


## ACT-R's associative strength (S_ki), made visible: a thin glowing rod
## between every pair of currently-broadcast objects that have a learned
## connection, brightness scaled by strength. Only ever connects two
## spheres already rendered above - the engine already scopes
## associative_edges to Working-Memory-to-Working-Memory pairs.
func _update_edges(edges: Array) -> void:
	var seen := {}
	for e in edges:
		var edge: Dictionary = e
		var source_id: String = edge.get("source_id", "")
		var target_id: String = edge.get("target_id", "")
		if not _nodes.has(source_id) or not _nodes.has(target_id):
			continue
		var key := "%s->%s" % [source_id, target_id]
		seen[key] = true
		var is_new := not _edge_lines.has(key)
		var strength: float = edge.get("strength", 0.0)
		var kind: String = edge.get("kind", "associative")
		_upsert_edge_line(key, _nodes[source_id].position, _nodes[target_id].position, strength, kind, is_new)

	for key in _edge_lines.keys():
		if not seen.has(key):
			_edge_lines[key].queue_free()
			_edge_lines.erase(key)


## `is_new` (true only the snapshot a key first appears - see
## `_update_edges`) makes a freshly-formed edge flare bright before settling
## to its strength-based resting brightness, instead of just appearing
## already at rest - the same "show the moment it happened, not only the
## resulting state" idiom as `_pulse_all_nodes` below, applied to edges.
func _upsert_edge_line(key: String, a: Vector3, b: Vector3, strength: float, kind: String, is_new: bool) -> void:
	var mesh_instance: MeshInstance3D
	if _edge_lines.has(key):
		mesh_instance = _edge_lines[key]
	else:
		mesh_instance = MeshInstance3D.new()
		var cylinder := CylinderMesh.new()
		cylinder.top_radius = 0.02
		cylinder.bottom_radius = 0.02
		mesh_instance.mesh = cylinder
		var material := StandardMaterial3D.new()
		material.emission_enabled = true
		material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		mesh_instance.set_surface_override_material(0, material)
		add_child(mesh_instance)
		_edge_lines[key] = mesh_instance

	var direction := b - a
	var length := direction.length()
	var cylinder: CylinderMesh = mesh_instance.mesh
	cylinder.height = max(length, 0.01)
	mesh_instance.position = (a + b) / 2.0
	_orient_between(mesh_instance, direction, length)

	var brightness := clampf(strength, 0.0, 1.0)
	var material: StandardMaterial3D = mesh_instance.get_surface_override_material(0)
	var edge_color: Color = EDGE_KIND_COLORS.get(kind, DEFAULT_EDGE_COLOR)
	material.albedo_color = Color(edge_color.r, edge_color.g, edge_color.b, 0.35 + brightness * 0.5)
	material.emission = edge_color
	var resting_energy := 0.3 + brightness * 1.5
	if is_new:
		material.emission_energy_multiplier = resting_energy + 3.0
		var tween := create_tween()
		tween.tween_property(material, "emission_energy_multiplier", resting_energy, 0.6)
	else:
		material.emission_energy_multiplier = resting_energy


## Rotates a default Y-aligned cylinder to point from `a` to `b` (only the
## direction is passed in, already as `b - a`). Unlike the pipeline's
## pipes, Working Memory spheres sit at arbitrary points on a circle, so
## this can't rely on a fixed axis-aligned roll - it uses Godot's angle-
## axis Basis constructor (a single well-defined rotation formula) rather
## than hand-deriving basis columns, which is exactly what produced a real
## sign-error bug in an earlier camera transform.
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
