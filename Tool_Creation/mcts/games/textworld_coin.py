"""
Pure-Python symbolic TextWorld coin game for MCTS.

This is a lightweight planning model inspired by the coin task used in
hw2_part2.ipynb. It avoids the Java-backed TextWorldExpress runtime so
states can be cloned cheaply for MCTS.

Supported parameters:
    numLocations=<int>
    includeDoors=<0|1>
    numDistractorItems=<int>   # currently parsed but ignored

Model assumptions:
    - Rooms are arranged in a 1-D corridor from west to east.
    - The agent starts in room 0.
    - The coin is in the final room.
    - If includeDoors=1, each corridor edge has a door that starts closed.
    - Goal is to take the coin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .game_interface import Game, GameState


LOOK_AROUND = "look around"
INVENTORY = "inventory"
MOVE_WEST = "move west"
MOVE_EAST = "move east"
OPEN_WEST = "open door to west"
OPEN_EAST = "open door to east"
CLOSE_WEST = "close door to west"
CLOSE_EAST = "close door to east"
TAKE_COIN = "take coin"


def _parse_game_params(game_params: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for chunk in game_params.split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


@dataclass(frozen=True)
class CoinConfig:
    num_locations: int = 5
    include_doors: bool = True
    num_distractor_items: int = 0
    max_steps: int = 50

    @classmethod
    def from_game_params(cls, game_params: str, max_steps: int = 50) -> "CoinConfig":
        params = _parse_game_params(game_params)
        return cls(
            num_locations=params.get("numLocations", 5),
            include_doors=bool(params.get("includeDoors", 1)),
            num_distractor_items=params.get("numDistractorItems", 0),
            max_steps=max_steps,
        )


class TextWorldCoinState(GameState):
    """
    Single-player symbolic coin-game state.
    """

    def __init__(self, config: CoinConfig):
        self.config = config
        self.room = 0
        self.coin_room = config.num_locations - 1
        self.coin_taken = False
        self.steps = 0
        self.inventory_items: tuple[str, ...] = ()
        self.doors_open = [not config.include_doors] * (config.num_locations - 1)

    # ---------- GameState interface ----------

    def clone(self) -> "TextWorldCoinState":
        s = TextWorldCoinState.__new__(TextWorldCoinState)
        s.config = self.config
        s.room = self.room
        s.coin_room = self.coin_room
        s.coin_taken = self.coin_taken
        s.steps = self.steps
        s.inventory_items = tuple(self.inventory_items)
        s.doors_open = list(self.doors_open)
        return s

    def current_player(self) -> int:
        return 0

    def legal_actions(self) -> list[Any]:
        if self.is_terminal():
            return []

        actions = [LOOK_AROUND, INVENTORY]

        if self.room > 0:
            left_open = self.doors_open[self.room - 1]
            if self.config.include_doors and not left_open:
                actions.append(OPEN_WEST)
            if self.config.include_doors and left_open:
                actions.append(CLOSE_WEST)
            if left_open:
                actions.append(MOVE_WEST)

        if self.room < self.config.num_locations - 1:
            right_open = self.doors_open[self.room]
            if self.config.include_doors and not right_open:
                actions.append(OPEN_EAST)
            if self.config.include_doors and right_open:
                actions.append(CLOSE_EAST)
            if right_open:
                actions.append(MOVE_EAST)

        if self.room == self.coin_room and not self.coin_taken:
            actions.append(TAKE_COIN)

        return actions

    def apply_action(self, action: Any) -> None:
        action = str(action)
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action!r} from state {self.state_key()}")

        if action == MOVE_WEST:
            self.room -= 1
        elif action == MOVE_EAST:
            self.room += 1
        elif action == OPEN_WEST and self.room > 0:
            self.doors_open[self.room - 1] = True
        elif action == OPEN_EAST and self.room < self.config.num_locations - 1:
            self.doors_open[self.room] = True
        elif action == CLOSE_WEST and self.room > 0:
            self.doors_open[self.room - 1] = False
        elif action == CLOSE_EAST and self.room < self.config.num_locations - 1:
            self.doors_open[self.room] = False
        elif action == TAKE_COIN and self.room == self.coin_room and not self.coin_taken:
            self.coin_taken = True
            self.inventory_items = tuple(sorted(self.inventory_items + ("coin",)))
        elif action in (LOOK_AROUND, INVENTORY):
            pass

        self.steps += 1

    def is_terminal(self) -> bool:
        return self.coin_taken or self.steps >= self.config.max_steps

    def returns(self) -> list[float]:
        if self.coin_taken:
            return [1.0]
        # Shaped return to help single-player MCTS distinguish progress.
        progress = self.room / max(1, self.config.num_locations - 1)
        door_bonus = 0.0
        if self.room < self.config.num_locations - 1 and self.doors_open[self.room]:
            door_bonus = 0.1
        return [max(0.0, min(0.9, 0.75 * progress + door_bonus))]

    def state_key(self) -> str:
        doors = "".join("1" if d else "0" for d in self.doors_open)
        inv = ",".join(self.inventory_items)
        return f"room={self.room}|coin={int(self.coin_taken)}|doors={doors}|inv={inv}|steps={self.steps}"

    def __str__(self) -> str:
        lines = [self.look_text(), self.inventory_text()]
        return "\n".join(lines)

    # ---------- Text helpers for LLM heuristics ----------

    def look_text(self) -> str:
        parts = [f"You are in room {self.room} out of {self.config.num_locations - 1}."]
        if self.room == self.coin_room and not self.coin_taken:
            parts.append("You see a coin here.")
        if self.room > 0:
            west_open = self.doors_open[self.room - 1]
            parts.append(f"There is a {'open' if west_open else 'closed'} door to the west.")
        if self.room < self.config.num_locations - 1:
            east_open = self.doors_open[self.room]
            parts.append(f"There is a {'open' if east_open else 'closed'} door to the east.")
        return " ".join(parts)

    def inventory_text(self) -> str:
        if not self.inventory_items:
            return "Inventory: empty"
        return f"Inventory: {', '.join(self.inventory_items)}"

    def observation_text(self) -> str:
        return f"{self.look_text()}\n{self.inventory_text()}"

    def distance_to_coin(self) -> int:
        return 0 if self.coin_taken else self.coin_room - self.room


class TextWorldCoin(Game):
    def __init__(self, game_params: str = "numLocations=5,includeDoors=1,numDistractorItems=0", max_steps: int = 50):
        self.game_params = game_params
        self.max_steps = max_steps
        self.config = CoinConfig.from_game_params(game_params, max_steps=max_steps)

    def new_initial_state(self) -> TextWorldCoinState:
        return TextWorldCoinState(self.config)

    def num_players(self) -> int:
        return 1

    def name(self) -> str:
        return f"TextWorldCoin({self.game_params})"
