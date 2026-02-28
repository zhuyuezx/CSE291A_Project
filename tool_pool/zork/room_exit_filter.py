"""Filter Zork actions to only include directions mentioned in the room text."""

__TOOL_META__ = {
    "name": "room_exit_filter",
    "type": "action_filter",
    "description": "Only keep directional actions (go north/south/etc.) that are mentioned in the current room description.",
}

_DIRECTIONS = {"north", "south", "east", "west", "up", "down"}


def run(state, legal_actions: list[int]) -> list[int]:
    from src.games.zork_adapter import _VOCAB
    text_lower = state.text.lower() if hasattr(state, "text") else str(state).lower()

    mentioned = {d for d in _DIRECTIONS if d in text_lower}

    filtered = []
    for a in legal_actions:
        cmd = _VOCAB[a % len(_VOCAB)]
        parts = cmd.split()
        if parts[0] == "go":
            direction = parts[-1]
            if direction in mentioned:
                filtered.append(a)
        else:
            filtered.append(a)  # keep all non-directional actions

    return filtered if filtered else legal_actions
