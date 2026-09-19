extends Node3D
## Composes the whole architecture view: positions each sub-view (Working
## Memory ring, tier indicator, memory silos, goal stack label, Knowledge
## Library label) relative to the pipeline's fixed station layout, and aims
## the light. The camera
## manages its own orientation (see orbit_camera.gd) - this script must not
## also touch it, or the two would fight over the transform every frame.
## Each sub-view otherwise builds and updates its own geometry
## independently by listening to ApiClient directly - this script only
## wires up *where* things sit relative to each other.

@onready var _pipeline: Node3D = $PipelineView
@onready var _working_memory_ring: Node3D = $WorkingMemoryRing
@onready var _goal_stack_view: Node3D = $GoalStackView
@onready var _knowledge_library_view: Node3D = $KnowledgeLibraryView
@onready var _executive_decision_view: Node3D = $ExecutiveDecisionView


func _ready() -> void:
	var light := get_node("DirectionalLight3D") as DirectionalLight3D
	if light != null:
		# The light sits directly above the origin, so its look-at
		# direction is parallel to Vector3.UP - using UP as the reference
		# "up" there is ambiguous (colinear target/up vectors); any
		# non-parallel reference resolves it, and a directional light's
		# roll has no visible effect on the scene anyway.
		light.look_at(Vector3.ZERO, Vector3.FORWARD)

	var broadcast_x: float = _pipeline.station_x("broadcast")
	var executive_x: float = _pipeline.station_x("executive")

	_working_memory_ring.position = Vector3(broadcast_x, -2.8, 0)
	_pipeline.connect_down("broadcast", 2.8)

	# TierIndicator (the LLM chain) self-positions relative to Coalition and
	# Executive - see tier_indicator.gd - rather than getting one fixed
	# offset the way every other sub-view here does, since it spans two
	# stations at once and sits *above* the line (Working Memory already
	# occupies the space below, near enough to Coalition to visually
	# overlap it). No connect_down grounding pipe here, unlike every other
	# station-anchored view - connect_down only draws downward, and
	# TierIndicator's own per-model beams already connect each node to its
	# station.

	# MemorySilos self-positions relative to Learn the same way - see
	# memory_silos.gd - rather than the old fixed off-to-the-side offset.

	_goal_stack_view.position = Vector3(executive_x - 0.8, 1.9, 0)

	# Mirrors the Goal Stack label on the opposite side of the same
	# station - Consult is an Executive-proposed operator too, and the
	# Knowledge Library's corpus size is exactly the kind of standing,
	# rarely-changing status this station-adjacent label style already
	# suits.
	_knowledge_library_view.position = Vector3(executive_x + 0.8, 1.9, 0)

	# Centered above the Goal Stack / Knowledge Library pair rather than a
	# third flanking point - the executive's last selected operator is a
	# single standing readout, not something that needs its own side slot.
	_executive_decision_view.position = Vector3(executive_x, 2.6, 0)

	# AffectPanel (Compare) and DriveCluster (Predict) self-position off
	# `_pipeline.station_x(...)` in their own `_ready()`, the same way
	# TierIndicator and MemorySilos already do above - no fixed offset
	# needed here.
