"""
Pure-Python symbolic benchmark aligned with hw2_part2.ipynb case grid.

This is not a byte-for-byte recreation of TextWorld-Express. It is a
planning-friendly symbolic benchmark that mirrors the homework's test
structure:
    - game types: coin, mapreader
    - environment variants: deterministic, stochastic, punishment
    - seeds: 0, 1, 2
    - parameter grids matching hw2_part2.ipynb
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .game_interface import Game, GameState


LOOK_AROUND = "look around"
INVENTORY = "inventory"
TASK = "task"
MOVE_WEST = "move west"
MOVE_EAST = "move east"
MOVE_SOUTH = "move south"
MOVE_NORTH = "move north"
OPEN_WEST = "open door to west"
OPEN_EAST = "open door to east"
OPEN_SOUTH = "open door to south"
OPEN_NORTH = "open door to north"
CLOSE_WEST = "close door to west"
CLOSE_EAST = "close door to east"
CLOSE_SOUTH = "close door to south"
CLOSE_NORTH = "close door to north"
TAKE_COIN = "take coin"
TAKE_MAP = "take map"
READ_MAP = "read map"
PUT_MAP_IN_BOX = "put map in box"
PUT_COIN_IN_BOX = "put coin in box"

DIRS = {
    "west": (-1, 0),
    "east": (1, 0),
    "south": (0, -1),
    "north": (0, 1),
}
MOVE_ACTIONS = {
    MOVE_WEST: "west",
    MOVE_EAST: "east",
    MOVE_SOUTH: "south",
    MOVE_NORTH: "north",
}
OPEN_ACTIONS = {
    OPEN_WEST: "west",
    OPEN_EAST: "east",
    OPEN_SOUTH: "south",
    OPEN_NORTH: "north",
}
CLOSE_ACTIONS = {
    CLOSE_WEST: "west",
    CLOSE_EAST: "east",
    CLOSE_SOUTH: "south",
    CLOSE_NORTH: "north",
}


def _parse_params(game_params: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in game_params.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


@dataclass(frozen=True)
class BenchmarkConfig:
    game_type: str
    game_params: str
    seed: int = 0
    variant: str = "deterministic"  # deterministic | stochastic | punishment
    max_steps: int = 50

    @property
    def params(self) -> dict[str, int]:
        return _parse_params(self.game_params)


class TextWorldBenchmarkState(GameState):
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.params = config.params
        self.rng = random.Random(config.seed)
        self.steps = 0
        self.last_reward = 0.0
        self.reward_accumulator = 0.0
        self.task_completed = False
        self.partial_credit = False
        self.map_read = False
        self.known_goal_room: int | None = None
        self.inventory_items: tuple[str, ...] = ()
        self._build_world()
        self.room = 0
        self.previous_look = self.look_text()

    # ---------- world generation ----------

    def _build_world(self) -> None:
        if self.config.game_type == "coin":
            self._build_coin_world()
        elif self.config.game_type == "mapreader":
            self._build_mapreader_world()
        else:
            raise ValueError(f"Unsupported game_type {self.config.game_type!r}")

    def _build_coin_world(self) -> None:
        n = self.params.get("numLocations", 5)
        include_doors = bool(self.params.get("includeDoors", 1))
        self.coords = {i: (i, 0) for i in range(n)}
        self.graph: dict[int, dict[str, int]] = {i: {} for i in range(n)}
        self.doors: dict[tuple[int, str], bool] = {}
        for i in range(n - 1):
            self.graph[i]["east"] = i + 1
            self.graph[i + 1]["west"] = i
            open_default = not include_doors
            self.doors[(i, "east")] = open_default
            self.doors[(i + 1, "west")] = open_default
        self.coin_room = n - 1
        self.map_room = None
        self.box_room = n // 2
        self.goal_room = self.coin_room

    def _build_mapreader_world(self) -> None:
        n = self.params.get("numLocations", 5)
        max_dist = self.params.get("maxDistanceApart", 3)
        self.coords = {0: (0, 0)}
        self.graph = {0: {}}
        frontier = [0]
        next_id = 1
        while next_id < n:
            parent = self.rng.choice(frontier)
            px, py = self.coords[parent]
            dirs = list(DIRS.items())
            self.rng.shuffle(dirs)
            placed = False
            for dname, (dx, dy) in dirs:
                nx, ny = px + dx, py + dy
                if abs(nx) + abs(ny) > max_dist:
                    continue
                if (nx, ny) in self.coords.values():
                    continue
                self.coords[next_id] = (nx, ny)
                self.graph[next_id] = {}
                self.graph[parent][dname] = next_id
                opposite = {"west": "east", "east": "west", "north": "south", "south": "north"}[dname]
                self.graph[next_id][opposite] = parent
                frontier.append(next_id)
                next_id += 1
                placed = True
                break
            if not placed:
                # fallback: attach linearly
                last = max(self.coords)
                lx, ly = self.coords[last]
                self.coords[next_id] = (lx + 1, ly)
                self.graph[next_id] = {"west": last}
                self.graph[last]["east"] = next_id
                frontier.append(next_id)
                next_id += 1
        self.doors = {(room, d): True for room, nbrs in self.graph.items() for d in nbrs}
        # mapreader task: map at start, read it, then move to hidden goal room
        dists = {room: abs(x) + abs(y) for room, (x, y) in self.coords.items()}
        self.goal_room = max(dists, key=lambda r: (dists[r], r))
        self.map_room = 0
        self.coin_room = None
        self.box_room = None

    # ---------- GameState interface ----------

    def clone(self) -> "TextWorldBenchmarkState":
        s = TextWorldBenchmarkState.__new__(TextWorldBenchmarkState)
        s.config = self.config
        s.params = dict(self.params)
        s.rng = random.Random()
        s.rng.setstate(self.rng.getstate())
        s.steps = self.steps
        s.last_reward = self.last_reward
        s.reward_accumulator = self.reward_accumulator
        s.task_completed = self.task_completed
        s.partial_credit = self.partial_credit
        s.map_read = self.map_read
        s.known_goal_room = self.known_goal_room
        s.inventory_items = tuple(self.inventory_items)
        s.coords = dict(self.coords)
        s.graph = {k: dict(v) for k, v in self.graph.items()}
        s.doors = dict(self.doors)
        s.room = self.room
        s.previous_look = self.previous_look
        s.goal_room = self.goal_room
        s.map_room = self.map_room
        s.coin_room = self.coin_room
        s.box_room = self.box_room
        return s

    def current_player(self) -> int:
        return 0

    def legal_actions(self) -> list[Any]:
        if self.is_terminal():
            return []
        actions = [LOOK_AROUND, INVENTORY, TASK]
        for d, nxt in self.graph[self.room].items():
            open_flag = self.doors.get((self.room, d), True)
            move_action = {
                "west": MOVE_WEST, "east": MOVE_EAST, "north": MOVE_NORTH, "south": MOVE_SOUTH
            }[d]
            open_action = {
                "west": OPEN_WEST, "east": OPEN_EAST, "north": OPEN_NORTH, "south": OPEN_SOUTH
            }[d]
            close_action = {
                "west": CLOSE_WEST, "east": CLOSE_EAST, "north": CLOSE_NORTH, "south": CLOSE_SOUTH
            }[d]
            if open_flag:
                actions.append(move_action)
                actions.append(close_action)
            else:
                actions.append(open_action)
        if self.config.game_type == "coin":
            if self.room == self.coin_room and "coin" not in self.inventory_items:
                actions.append(TAKE_COIN)
            if "coin" in self.inventory_items and self.box_room is not None and self.room == self.box_room:
                actions.append(PUT_COIN_IN_BOX)
        elif self.config.game_type == "mapreader":
            if self.room == self.map_room and "map" not in self.inventory_items:
                actions.append(TAKE_MAP)
            if "map" in self.inventory_items:
                actions.append(READ_MAP)
                if self.box_room is not None and self.room == self.box_room:
                    actions.append(PUT_MAP_IN_BOX)
        return actions

    def apply_action(self, action: Any) -> None:
        intended = str(action)
        legal = self.legal_actions()
        if intended not in legal:
            raise ValueError(f"Illegal action {intended!r} from state {self.state_key()}")

        executed = intended
        if self.config.variant == "stochastic":
            meaningful = [a for a in legal if a not in (LOOK_AROUND, INVENTORY, TASK)]
            if meaningful and self.rng.random() < 0.25:
                executed = self.rng.choice(meaningful)

        old_look = self.look_text()
        self.last_reward = 0.0

        if executed in MOVE_ACTIONS:
            d = MOVE_ACTIONS[executed]
            if self.doors.get((self.room, d), True):
                self.room = self.graph[self.room][d]
        elif executed in OPEN_ACTIONS:
            d = OPEN_ACTIONS[executed]
            self.doors[(self.room, d)] = True
            other = self.graph[self.room][d]
            opposite = {"west": "east", "east": "west", "north": "south", "south": "north"}[d]
            self.doors[(other, opposite)] = True
        elif executed in CLOSE_ACTIONS:
            d = CLOSE_ACTIONS[executed]
            self.doors[(self.room, d)] = False
            other = self.graph[self.room][d]
            opposite = {"west": "east", "east": "west", "north": "south", "south": "north"}[d]
            self.doors[(other, opposite)] = False
        elif executed == TAKE_COIN and self.config.game_type == "coin" and self.room == self.coin_room:
            self.inventory_items = tuple(sorted(self.inventory_items + ("coin",)))
            self.task_completed = True
            self.last_reward = 1.0
        elif executed == TAKE_MAP and self.config.game_type == "mapreader" and self.room == self.map_room:
            self.inventory_items = tuple(sorted(self.inventory_items + ("map",)))
        elif executed == READ_MAP and "map" in self.inventory_items:
            self.map_read = True
            self.known_goal_room = self.goal_room
            self.partial_credit = True
            self.last_reward = max(self.last_reward, 0.5)
        elif executed == PUT_MAP_IN_BOX and "map" in self.inventory_items:
            self.inventory_items = tuple(x for x in self.inventory_items if x != "map")
        elif executed == PUT_COIN_IN_BOX and "coin" in self.inventory_items:
            self.inventory_items = tuple(x for x in self.inventory_items if x != "coin")

        if self.config.game_type == "mapreader" and self.map_read and self.room == self.goal_room:
            self.task_completed = True
            self.last_reward = 1.0

        new_look = self.look_text()
        if self.config.variant == "punishment" and new_look == old_look:
            self.last_reward = min(self.last_reward, -1.0)

        self.reward_accumulator += self.last_reward
        self.steps += 1
        self.previous_look = new_look

    def is_terminal(self) -> bool:
        return self.task_completed or self.steps >= self.config.max_steps

    def returns(self) -> list[float]:
        if self.task_completed:
            return [1.0]
        if self.partial_credit:
            return [0.5]
        if self.reward_accumulator < 0:
            return [max(-1.0, self.reward_accumulator)]
        return [0.0]

    def state_key(self) -> str:
        doors = ",".join(f"{k[0]}:{k[1]}={int(v)}" for k, v in sorted(self.doors.items()))
        inv = ",".join(self.inventory_items)
        rng_tag = hash(str(self.rng.getstate()[1][:5])) if self.config.variant == "stochastic" else 0
        return (
            f"type={self.config.game_type}|variant={self.config.variant}|room={self.room}|"
            f"goal={self.goal_room}|map_read={int(self.map_read)}|done={int(self.task_completed)}|"
            f"inv={inv}|doors={doors}|steps={self.steps}|rng={rng_tag}"
        )

    # ---------- helpers exposed to LLM ----------

    def look_text(self) -> str:
        exits = []
        for d in sorted(self.graph[self.room]):
            state = "open" if self.doors.get((self.room, d), True) else "closed"
            exits.append(f"{state} exit to the {d}")
        parts = [f"You are in room {self.room}.", "There is " + ", ".join(exits) + "." if exits else "There are no exits."]
        if self.config.game_type == "coin" and self.room == self.coin_room and "coin" not in self.inventory_items:
            parts.append("You see a coin here.")
        if self.config.game_type == "mapreader" and self.room == self.map_room and "map" not in self.inventory_items:
            parts.append("You see a map here.")
        if self.map_read and self.known_goal_room is not None:
            parts.append(f"The map says the goal is room {self.known_goal_room}.")
        return " ".join(parts)

    def inventory_text(self) -> str:
        if not self.inventory_items:
            return "Inventory: empty"
        return "Inventory: " + ", ".join(self.inventory_items)

    def observation_text(self) -> str:
        return f"{self.look_text()}\n{self.inventory_text()}"

    def distance_to_goal(self) -> int:
        return self._shortest_path_len(self.room, self.goal_room)

    def _shortest_path_len(self, src: int, dst: int) -> int:
        if src == dst:
            return 0
        frontier = [(src, 0)]
        seen = {src}
        while frontier:
            room, dist = frontier.pop(0)
            for nxt in self.graph[room].values():
                if nxt == dst:
                    return dist + 1
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append((nxt, dist + 1))
        return 999


class TextWorldBenchmark(Game):
    def __init__(self, game_type: str, game_params: str, seed: int = 0, variant: str = "deterministic", max_steps: int = 50):
        self.config = BenchmarkConfig(
            game_type=game_type,
            game_params=game_params,
            seed=seed,
            variant=variant,
            max_steps=max_steps,
        )

    def new_initial_state(self) -> TextWorldBenchmarkState:
        return TextWorldBenchmarkState(self.config)

    def num_players(self) -> int:
        return 1

    def name(self) -> str:
        return f"TextWorldBenchmark({self.config.variant}, {self.config.game_type}, {self.config.game_params}, seed={self.config.seed})"
