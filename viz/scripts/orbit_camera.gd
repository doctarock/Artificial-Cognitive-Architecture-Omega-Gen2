extends Camera3D
## An orbit-and-pan camera: hold the right mouse button and drag to orbit,
## scroll to zoom, WASD (+ E/Space up, Q/Shift down) to pan the pivot itself
## around the scene. The architecture view has too much in it (an 8-station
## pipeline, three memory silos, a tier indicator, a Working Memory ring) to
## frame legibly from one fixed angle or reach by orbiting alone - panning
## lets the viewer actually travel to a different part of the layout, not
## just spin around the same starting point. Fully manual - no
## auto-rotation - the viewer decides what to look at and when.

@export var target := Vector3(0, -0.8, 3.0)
@export var distance := 30.0
@export var yaw := 0.3
@export var pitch := 0.5

const MIN_DISTANCE := 6.0
const MAX_DISTANCE := 55.0
const MIN_PITCH := 0.05
const MAX_PITCH := 1.5
const ORBIT_SPEED := 0.006
const ZOOM_STEP := 2.0
# A fraction of the current zoom distance per second held, not a fixed
# world-space speed - so panning still feels reasonable whether zoomed in
# close or all the way out, rather than crawling or rocketing off-screen.
const PAN_SPEED_FACTOR := 0.7

var _dragging := false


func _ready() -> void:
	_update_transform()


func _process(delta: float) -> void:
	# Without this, typing "was" or "sad" into the chat input box would also
	# pan the camera on every overlapping keystroke - raw Input.is_key_pressed
	# reads hardware state regardless of what has keyboard focus.
	if get_viewport().gui_get_focus_owner() is TextEdit:
		return
	var forward := Vector3(-sin(yaw), 0.0, -cos(yaw))
	var right := Vector3(cos(yaw), 0.0, -sin(yaw))
	var move := Vector3.ZERO
	if Input.is_key_pressed(KEY_W):
		move += forward
	if Input.is_key_pressed(KEY_S):
		move -= forward
	if Input.is_key_pressed(KEY_D):
		move += right
	if Input.is_key_pressed(KEY_A):
		move -= right
	if Input.is_key_pressed(KEY_E) or Input.is_key_pressed(KEY_SPACE):
		move += Vector3.UP
	if Input.is_key_pressed(KEY_Q) or Input.is_key_pressed(KEY_SHIFT):
		move -= Vector3.UP
	if move != Vector3.ZERO:
		target += move.normalized() * distance * PAN_SPEED_FACTOR * delta
		_update_transform()


func _unhandled_input(event: InputEvent) -> void:
	# Godot doesn't defocus the input box just because you clicked somewhere
	# else - only clicking *another* focusable Control does that. Clicking
	# the 3D viewport reaches here precisely because nothing else claimed
	# it, which makes this the right place to release focus explicitly;
	# without it, the input box holds focus forever and WASD/orbit clicks
	# stop working until you manually click back into a text field to tab
	# away, which most people won't think to do.
	if event is InputEventMouseButton and event.pressed:
		var focus_owner := get_viewport().gui_get_focus_owner()
		if focus_owner is TextEdit:
			focus_owner.release_focus()
	if event is InputEventKey and event.pressed and event.keycode == KEY_ESCAPE:
		var focus_owner := get_viewport().gui_get_focus_owner()
		if focus_owner is TextEdit:
			focus_owner.release_focus()

	if event is InputEventMouseButton:
		var mouse_event := event as InputEventMouseButton
		if mouse_event.button_index == MOUSE_BUTTON_RIGHT:
			_dragging = mouse_event.pressed
		elif mouse_event.button_index == MOUSE_BUTTON_WHEEL_UP and mouse_event.pressed:
			distance = clampf(distance - ZOOM_STEP, MIN_DISTANCE, MAX_DISTANCE)
			_update_transform()
		elif mouse_event.button_index == MOUSE_BUTTON_WHEEL_DOWN and mouse_event.pressed:
			distance = clampf(distance + ZOOM_STEP, MIN_DISTANCE, MAX_DISTANCE)
			_update_transform()
	elif event is InputEventMouseMotion and _dragging:
		var motion_event := event as InputEventMouseMotion
		yaw -= motion_event.relative.x * ORBIT_SPEED
		pitch = clampf(pitch - motion_event.relative.y * ORBIT_SPEED, MIN_PITCH, MAX_PITCH)
		_update_transform()


func _update_transform() -> void:
	var offset := Vector3(
		cos(pitch) * sin(yaw),
		sin(pitch),
		cos(pitch) * cos(yaw)
	) * distance
	position = target + offset
	look_at(target, Vector3.UP)
