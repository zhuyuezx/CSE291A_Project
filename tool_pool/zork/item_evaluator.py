"""Score Zork states higher when useful items appear in the room or inventory."""

__TOOL_META__ = {
    "name": "item_evaluator",
    "type": "state_evaluator",
    "description": "Score higher when high-value items (lantern, sword, treasure) are present in the room or inventory.",
}

_HIGH_VALUE_ITEMS = ["lantern", "sword", "trophy", "jewel", "coin", "gold", "silver", "diamond"]
_MEDIUM_VALUE_ITEMS = ["leaflet", "mailbox", "door", "key", "bottle"]


def run(state) -> float:
    text = state.text.lower() if hasattr(state, "text") else str(state).lower()
    score = 0.0
    for item in _HIGH_VALUE_ITEMS:
        if item in text:
            score += 0.2
    for item in _MEDIUM_VALUE_ITEMS:
        if item in text:
            score += 0.05
    return max(-1.0, min(1.0, score - 0.5))  # center around 0
