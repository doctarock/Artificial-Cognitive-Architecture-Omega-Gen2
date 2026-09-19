extends Node3D
## `steps::drives::DriveState`'s five tracked pressures, made visible - the
## standing motivational backdrop for the whole cognitive cycle (not any
## single phase), so it self-positions as a satellite beamed to Predict the
## same way `memory_silos.gd` beams long-term memory to Learn, rather than
## flanking one station's label. Each drive is EMA-smoothed and individually
## clamped [0,1] (see drives.rs) - bars, not particles, since a bounded
## scalar reads more honestly as a fill level than as a density.

const DRIVES := ["uncertainty", "curiosity", "competence", "social_connection", "resource_pressure"]
const DRIVE_LABELS := {
	"uncertainty": "Uncertainty",
	"curiosity": "Curiosity",
	"competence": "Competence",
	"social_connection": "Social",
	"resource_pressure": "Resources",
}
const DRIVE_COLORS := {
	"uncertainty": Color(0.6, 0.4, 1.0),
	"curiosity": Color(1.0, 0.85, 0.3),
	"competence": Color(0.4, 0.9, 1.0),
	"social_connection": Color(1.0, 0.5, 0.7),
	"resource_pressure": Color(1.0, 0.45, 0.2),
}

const BAR_SPACING := 0.3
const MAX_BAR_HEIGHT := 1.0
const BAR_WIDTH := 0.12
const CLUSTER_Y := -1.6
const BEAM_COLOR := Color(0.5, 0.5, 0.55)

@onready var _pipeline: Node3D = get_parent().get_node("PipelineView")

var _bars: Dictionary = {}  # drive name -> MeshInstance3D
var _labels: Dictionary = {}  # drive name -> Label3D
var _base_pos: Vector3 = Vector3.ZERO


func _ready() -> void:
	ApiClient.snapshot_updated.connect(_on_snapshot_updated)

	var anchor_x: float = _pipeline.station_x("predict") if _pipeline.has_station("predict") else 0.0
	_base_pos = Vector3(anchor_x, CLUSTER_Y, 0)

	var n: int = DRIVES.size()
	var start_x: float = -(n - 1) * BAR_SPACING / 2.0

	for i in n:
		var drive: String = DRIVES[i]
		var bar_x: float = start_x + i * BAR_SPACING
		_create_bar(drive, _base_pos + Vector3(bar_x, 0, 0))

	var beam := MeshInstance3D.new()
	var cylinder := CylinderMesh.new()
	cylinder.top_radius = 0.03
	cylinder.bottom_radius = 0.03
	var station_pos := Vector3(anchor_x, 0, 0)
	var direction := station_pos - _base_pos
	cylinder.height = max(direction.length(), 0.01)
	beam.mesh = cylinder
	beam.position = (_base_pos + station_pos) / 2.0
	_orient_between(beam, direction, direction.length())
	var beam_material := StandardMaterial3D.new()
	beam_material.emission_enabled = true
	beam_material.albedo_color = BEAM_COLOR
	beam_material.emission = BEAM_COLOR
	beam_material.emission_energy_multiplier = 0.6
	beam.set_surface_override_material(0, beam_material)
	add_child(beam)

	var title := Label3D.new()
	title.text = "Drives"
	title.position = _base_pos + Vector3(0, MAX_BAR_HEIGHT + 0.45, 0)
	title.pixel_size = 0.01
	title.font_size = 26
	title.outline_size = 6
	title.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	title.modulate = Color(0.8, 0.8, 0.9)
	add_child(title)


func _create_bar(drive: String, base_pos: Vector3) -> void:
	var color: Color = DRIVE_COLORS.get(drive, Color.WHITE)

	var bar := MeshInstance3D.new()
	var cylinder := CylinderMesh.new()
	cylinder.top_radius = BAR_WIDTH / 2.0
	cylinder.bottom_radius = BAR_WIDTH / 2.0
	cylinder.height = MAX_BAR_HEIGHT
	bar.mesh = cylinder
	bar.scale.y = 0.02
	bar.position = base_pos
	var material := StandardMaterial3D.new()
	material.shading_mode = BaseMaterial3D.SHADING_MODE_PER_PIXEL
	material.emission_enabled = true
	material.albedo_color = color
	material.emission = color
	material.emission_energy_multiplier = 1.0
	bar.set_surface_override_material(0, material)
	add_child(bar)
	_bars[drive] = bar

	var label := Label3D.new()
	label.text = "%s\n0.00" % DRIVE_LABELS.get(drive, drive)
	label.position = base_pos + Vector3(0, -0.35, 0)
	label.pixel_size = 0.008
	label.font_size = 22
	label.outline_size = 5
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	add_child(label)
	_labels[drive] = label


func _on_snapshot_updated(data: Dictionary) -> void:
	for drive in DRIVES:
		var value: float = clampf(data.get("drive_%s" % drive, 0.0), 0.0, 1.0)
		var bar: MeshInstance3D = _bars[drive]
		# Fixed-height cylinder scaled down to a thin sliver by default
		# (see `_create_bar`'s `scale.y = 0.02`) - repositioned each update
		# so it visibly grows *up* from a shared floor line rather than
		# from its own center, the same trick a bar chart needs whenever
		# the mesh's own height is what's being scaled.
		var scale_y: float = max(value, 0.02)
		bar.scale.y = scale_y
		bar.position.y = _base_pos.y + (MAX_BAR_HEIGHT * scale_y) / 2.0

		var label: Label3D = _labels[drive]
		label.text = "%s\n%.2f" % [DRIVE_LABELS.get(drive, drive), value]


## Copied from `working_memory_ring.gd::_orient_between` (also reused by
## `memory_silos.gd`/`tier_indicator.gd`) - see that function's doc comment
## for why cross-product/angle-axis, not a hand-derived basis.
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
