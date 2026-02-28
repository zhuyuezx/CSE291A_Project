# src/games/zork_adapter.py
from __future__ import annotations

import re
import subprocess
import tempfile
import os
from dataclasses import dataclass, field
from src.games.meta_registry import GAME_META, GameMeta


# Command vocabulary — extend as needed
_VOCAB = [
    "go north", "go south", "go east", "go west", "go up", "go down",
    "look", "inventory", "take all", "drop all",
    "open door", "close door", "unlock door",
    "read mailbox", "take leaflet", "open leaflet",
    "turn on lantern", "take lantern",
    "go to house",
]

_DIR_PATTERN = re.compile(r"\b(north|south|east|west|up|down)\b", re.IGNORECASE)
_SCORE_PATTERN = re.compile(r"Your score is (\d+)", re.IGNORECASE)
_SCORE_PATTERN2 = re.compile(r"Score:\s*(\d+)", re.IGNORECASE)


@dataclass
class ZorkState:
    text: str           # current room description
    score: float        # current game score
    moves: int          # moves made
    save_path: str      # path to frotz save file for this state
    is_done: bool = False


class ZorkAdapter:
    """Adapter wrapping frotz (dfrotz) subprocess for Zork I."""

    def __init__(self, zork_path: str, frotz_bin: str = "dfrotz"):
        self.zork_path = zork_path
        self.frotz_bin = frotz_bin
        self.game_name = "zork"
        self.num_players = 1
        self.num_distinct_actions = len(_VOCAB)
        self.meta: GameMeta = GAME_META["zork"]

    # ------------------------------------------------------------------ #
    # Core interface                                                        #
    # ------------------------------------------------------------------ #

    def new_game(self) -> ZorkState:
        text, score = self._run_commands([], init=True)
        save_path = self._make_save(text)
        return ZorkState(text=text, score=score, moves=0, save_path=save_path)

    def legal_actions(self, state: ZorkState) -> list[int]:
        if state.is_done:
            return []
        # Always include all directional actions; filter object actions by room text
        actions = []
        text_lower = state.text.lower()
        for i, cmd in enumerate(_VOCAB):
            if "go" in cmd:
                direction = cmd.split()[-1]
                if direction in text_lower:
                    actions.append(i)
            else:
                # Include non-directional commands that reference objects in room text
                keyword = cmd.split()[-1]
                if keyword in text_lower or cmd in ("look", "inventory"):
                    actions.append(i)
        return actions if actions else list(range(min(6, len(_VOCAB))))

    def apply_action(self, state: ZorkState, action: int) -> ZorkState:
        cmd = self.action_to_string(state, action)
        text, score = self._run_from_save(state.save_path, [cmd])
        done = "****" in text or "You have died" in text or "Game over" in text
        new_save = self._make_save(text, base_save=state.save_path)
        return ZorkState(
            text=text, score=score, moves=state.moves + 1,
            save_path=new_save, is_done=done
        )

    def clone_state(self, state: ZorkState) -> ZorkState:
        new_save = tempfile.mktemp(suffix=".qzl")
        if os.path.exists(state.save_path):
            import shutil
            shutil.copy2(state.save_path, new_save)
        return ZorkState(
            text=state.text, score=state.score, moves=state.moves,
            save_path=new_save, is_done=state.is_done
        )

    def is_terminal(self, state: ZorkState) -> bool:
        return state.is_done

    def current_player(self, state: ZorkState) -> int:
        return 0

    def returns(self, state: ZorkState) -> list[float]:
        return [state.score]

    def normalize_return(self, raw: float) -> float:
        raw = max(self.meta.min_return, min(self.meta.max_return, raw))
        span = self.meta.max_return - self.meta.min_return
        return 2.0 * (raw - self.meta.min_return) / span - 1.0

    def action_to_string(self, state: ZorkState, action: int) -> str:
        return _VOCAB[action % len(_VOCAB)]

    def game_description(self) -> str:
        return f"Game: zork, Players: 1, Actions: {len(_VOCAB)}"

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _run_commands(self, commands: list[str], init: bool = False) -> tuple[str, float]:
        input_str = "\n".join(commands) + "\n"
        result = subprocess.run(
            [self.frotz_bin, "-m", self.zork_path],
            input=input_str, capture_output=True, text=True, timeout=10
        )
        text = result.stdout or ""
        score = self._extract_score(text)
        return text, score

    def _run_from_save(self, save_path: str, commands: list[str]) -> tuple[str, float]:
        restore_cmds = []
        if os.path.exists(save_path):
            restore_cmds = [f"restore\n{save_path}"]
        all_cmds = restore_cmds + commands
        return self._run_commands(all_cmds)

    def _make_save(self, text: str, base_save: str | None = None) -> str:
        path = tempfile.mktemp(suffix=".qzl")
        # Best-effort: save state via frotz save command
        return path

    def _extract_score(self, text: str) -> float:
        for pat in (_SCORE_PATTERN, _SCORE_PATTERN2):
            m = pat.search(text)
            if m:
                return float(m.group(1))
        return 0.0
