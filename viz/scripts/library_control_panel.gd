extends Control
## The manual replacement for the old unattended drip timer: press "Deliver
## a Book" to advance the library's "currently reading" cursor by exactly
## one chunk (`POST /library/deliver` - see omega_acad::library's module doc
## comment for why this stopped being timer-driven). Mirrors
## test_status_panel.gd's caller-triggered button pattern.

@onready var _deliver_button: Button = %DeliverBookButton
@onready var _status_label: Label = %LibraryStatusLabel

const COLOR_DELIVERED := Color(0.4, 1.0, 0.5)
const COLOR_EXHAUSTED := Color(0.95, 0.85, 0.4)
const COLOR_FAILED := Color(1.0, 0.35, 0.35)
const COLOR_IDLE := Color(0.7, 0.7, 0.75)


func _ready() -> void:
	_deliver_button.pressed.connect(_on_deliver_pressed)
	ApiClient.book_delivered.connect(_on_book_delivered)
	ApiClient.book_delivery_failed.connect(_on_book_delivery_failed)
	_status_label.text = "Library: idle"
	_status_label.modulate = COLOR_IDLE


func _on_deliver_pressed() -> void:
	_deliver_button.disabled = true
	_status_label.text = "Delivering..."
	_status_label.modulate = COLOR_IDLE
	ApiClient.deliver_book()


func _on_book_delivered(data: Dictionary) -> void:
	_deliver_button.disabled = false
	var status: String = str(data.get("status", ""))
	if status == "exhausted":
		_status_label.text = "Library: every book has been read"
		_status_label.modulate = COLOR_EXHAUSTED
		return
	var file: String = str(data.get("file", "?"))
	_status_label.text = "Delivered: %s" % file.get_file()
	_status_label.modulate = COLOR_DELIVERED


func _on_book_delivery_failed(reason: String) -> void:
	_deliver_button.disabled = false
	_status_label.text = "Delivery failed: %s" % reason
	_status_label.modulate = COLOR_FAILED
