extends Node3D
## Predictive-processing affect, made visible: `EngineSnapshot.affect_valence`
## (`steps::affect::AffectTracker::valence()`, hard-bounded [-1,1], EMA-
## smoothed ~10%/tick - a slow trend, never a per-tick jump) and
## `precision_gain` (derived from valence, [0.1,1.3], 1.0 neutral - and
## genuinely fed back into the engine: `loop_actor.rs` multiplies it into
## that tick's surprise score, so this isn't just cosmetic). A bead sliding
## along a track reads naturally as a signed value where a bar chart
## wouldn't - this is the first signed-value encoding in the viz, no prior
## pattern to match. Self-positions at Compare (where Compare actually
## produces these numbers) the same way memory_silos.gd hangs off Learn.

const TRACK_LENGTH := 1.2
const TRACK_HEIGHT := 1.0
const NEGATIVE_COLOR := Color(1.0, 0.3, 0.3)
const POSITIVE_COLOR := Color(0.3, 1.0, 0.5)
const FLASH_DURATION_SEC := 0.5

@onready var _pipeline: Node3D = get_parent().get_node("PipelineView")

var _bead: MeshInstance3D
var _label: Label3D
var _panel_pos: Vector3 = Vector3.ZERO
## `EpochMillis` of the last `last_prediction_error` this panel has already
## flashed for - `-1` means "none seen yet" (no flash on the first sample,
## same "first observation just establishes baseline" reasoning as
## `goal_stack_view.gd`'s `_seen_before`).
var _previous_error_at: int = -1


func _ready() -> void:
	ApiClient.snapshot_updated.connect(_on_snapshot_updated)

	var anchor_x: float = _pipeline.station_x("compare") if _pipeline.has_station("compare") else 0.0
	_panel_pos = Vector3(anchor_x, TRACK_HEIGHT, 0)

	var track := MeshInstance3D.new()
	var box := BoxMesh.new()
	box.size = Vector3(TRACK_LENGTH, 0.02, 0.02)
	track.mesh = box
	track.position = _panel_pos
	var track_material := StandardMaterial3D.new()
	track_material.albedo_color = Color(0.4, 0.4, 0.45)
	track.set_surface_override_material(0, track_material)
	add_child(track)

	_bead = MeshInstance3D.new()
	_bead.mesh = SphereMesh.new()
	_bead.mesh.radius = 0.08
	_bead.mesh.height = 0.16
	_bead.position = _panel_pos
	var bead_material := StandardMaterial3D.new()
	bead_material.shading_mode = BaseMaterial3D.SHADING_MODE_PER_PIXEL
	bead_material.emission_enabled = true
	_bead.set_surface_override_material(0, bead_material)
	add_child(_bead)

	_label = Label3D.new()
	_label.pixel_size = 0.01
	_label.font_size = 28
	_label.outline_size = 6
	_label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	_label.position = _panel_pos + Vector3(0, -0.3, 0)
	_label.text = "Affect: +0.00  gain 1.00x"
	add_child(_label)


func _on_snapshot_updated(data: Dictionary) -> void:
	var valence: float = clampf(data.get("affect_valence", 0.0), -1.0, 1.0)
	var gain: float = data.get("precision_gain", 1.0)

	var half := TRACK_LENGTH / 2.0
	_bead.position.x = _panel_pos.x + remap(valence, -1.0, 1.0, -half, half)

	var color: Color = NEGATIVE_COLOR.lerp(POSITIVE_COLOR, remap(valence, -1.0, 1.0, 0.0, 1.0))
	var material: StandardMaterial3D = _bead.get_surface_override_material(0)
	material.albedo_color = color
	material.emission = color
	material.emission_energy_multiplier = gain

	var error = data.get("last_prediction_error", null)
	if error == null:
		_label.text = "Affect: %+.2f  gain %.2fx" % [valence, gain]
	else:
		var error_dict: Dictionary = error
		var magnitude: float = error_dict.get("error_magnitude", 0.0)
		_label.text = "Affect: %+.2f  gain %.2fx  surprise %.2f" % [valence, gain, magnitude]

		var at: int = int(error_dict.get("at", 0))
		if _previous_error_at >= 0 and at != _previous_error_at:
			_flash(magnitude, gain)
		_previous_error_at = at


## Flares the bead brighter than its resting `precision_gain` energy,
## scaled by how surprising this tick's fresh comparison actually was -
## the same "show the moment it happened" idiom as `working_memory_ring.gd`'s
## `_pulse_all_nodes`, applied to a fresh prediction-error sample instead
## of an impasse/escalation event.
func _flash(magnitude: float, resting_gain: float) -> void:
	var material: StandardMaterial3D = _bead.get_surface_override_material(0)
	var tween := create_tween()
	tween.tween_property(material, "emission_energy_multiplier", resting_gain + magnitude * 3.0, FLASH_DURATION_SEC * 0.3)
	tween.tween_property(material, "emission_energy_multiplier", resting_gain, FLASH_DURATION_SEC * 0.7)
