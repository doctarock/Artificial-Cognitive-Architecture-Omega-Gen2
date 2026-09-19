extends Node3D
## Purely decorative: a soft grounding floor and drifting ambient dust, so
## the architecture reads as a real space rather than shapes floating in
## a void. No engine data involved - this is the "fancier looking" pass,
## separate from anything that visualizes real state.

func _ready() -> void:
	_build_floor()
	_build_dust()


func _build_floor() -> void:
	var floor_mesh := MeshInstance3D.new()
	var plane := PlaneMesh.new()
	plane.size = Vector2(60, 60)
	floor_mesh.mesh = plane
	floor_mesh.position = Vector3(0, -4.2, 2)

	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.05, 0.06, 0.09)
	material.metallic = 0.35
	material.roughness = 0.25
	material.emission_enabled = true
	material.emission = Color(0.08, 0.12, 0.22)
	material.emission_energy_multiplier = 0.3
	# A faint reflection of the scene's own glow sells the "floor" read far
	# better than a flat matte plane would, at negligible cost for a single
	# large low-poly quad.
	material.clearcoat_enabled = true
	material.clearcoat = 0.6
	floor_mesh.set_surface_override_material(0, material)
	add_child(floor_mesh)


func _build_dust() -> void:
	var particles := GPUParticles3D.new()
	particles.amount = 220
	particles.lifetime = 14.0
	particles.position = Vector3(0, 2, 2)

	var mesh := SphereMesh.new()
	mesh.radius = 0.02
	mesh.height = 0.04
	var dust_material := StandardMaterial3D.new()
	dust_material.emission_enabled = true
	dust_material.emission = Color(0.5, 0.7, 1.0)
	dust_material.emission_energy_multiplier = 0.8
	# So each particle picks up process_material's per-particle color
	# (used below for a very slight variation) instead of one flat albedo.
	dust_material.vertex_color_use_as_albedo = true
	mesh.surface_set_material(0, dust_material)
	particles.draw_pass_1 = mesh

	var process_material := ParticleProcessMaterial.new()
	process_material.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_BOX
	process_material.emission_box_extents = Vector3(16, 6, 12)
	process_material.direction = Vector3(0, 1, 0)
	process_material.spread = 180.0
	process_material.gravity = Vector3(0, 0.05, 0)
	process_material.initial_velocity_min = 0.02
	process_material.initial_velocity_max = 0.12
	process_material.scale_min = 0.5
	process_material.scale_max = 1.8
	process_material.color = Color(0.5, 0.7, 1.0, 0.5)
	particles.process_material = process_material

	add_child(particles)
