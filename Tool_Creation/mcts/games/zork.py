"""
Pure-Python symbolic Zork I benchmark for MCTS.

A simplified but faithful recreation of key areas from Zork I,
designed for planning-friendly MCTS rollouts.  The world includes
~20 iconic rooms, items, puzzles (dark rooms, troll combat,
locked passages), and a treasure-collection scoring system.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .game_interface import Game, GameState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Room IDs
WEST_OF_HOUSE = "west_of_house"
NORTH_OF_HOUSE = "north_of_house"
SOUTH_OF_HOUSE = "south_of_house"
BEHIND_HOUSE = "behind_house"
KITCHEN = "kitchen"
LIVING_ROOM = "living_room"
ATTIC = "attic"
FOREST_WEST = "forest_west"
FOREST_EAST = "forest_east"
CLEARING = "clearing"
CANYON_VIEW = "canyon_view"
ROCKY_CRAWL = "rocky_crawl"
CELLAR = "cellar"
TROLL_ROOM = "troll_room"
EW_PASSAGE = "ew_passage"
ROUND_ROOM = "round_room"
DOME_ROOM = "dome_room"
TORCH_ROOM = "torch_room"
TEMPLE = "temple"
ALTAR = "altar"
LOUD_ROOM = "loud_room"
TREASURE_ROOM = "treasure_room"

# Rooms ordered by "distance from start" — used for num_rooms slicing.
# First rooms are surface/easy, deeper rooms are underground/harder.
ALL_ROOMS = [
    # --- surface layer (rooms 1-10) ---
    WEST_OF_HOUSE,      # 1  (start)
    NORTH_OF_HOUSE,     # 2
    SOUTH_OF_HOUSE,     # 3
    BEHIND_HOUSE,       # 4
    KITCHEN,            # 5
    LIVING_ROOM,        # 6  (trophy case + lantern + sword)
    ATTIC,              # 7
    FOREST_WEST,        # 8
    FOREST_EAST,        # 9
    CLEARING,           # 10
    # --- underground layer 1 (rooms 11-14) ---
    CELLAR,             # 11 (dark)
    TROLL_ROOM,         # 12 (dark, troll blocks west)
    EW_PASSAGE,         # 13 (dark)
    CANYON_VIEW,        # 14
    # --- underground layer 2 (rooms 15-18) ---
    ROCKY_CRAWL,        # 15 (dark)
    ROUND_ROOM,         # 16 (dark)
    DOME_ROOM,          # 17 (dark)
    TORCH_ROOM,         # 18 (dark, has ivory torch)
    # --- deep underground (rooms 19-22) ---
    TEMPLE,             # 19 (dark)
    ALTAR,              # 20 (dark, has chalice)
    LOUD_ROOM,          # 21 (dark, has sceptre)
    TREASURE_ROOM,      # 22 (dark, has gold coffin + platinum bar)
]

# Items
LANTERN = "brass lantern"
SWORD = "elvish sword"
LEAFLET = "leaflet"
JEWELED_EGG = "jeweled egg"
GOLD_COFFIN = "gold coffin"
CHALICE = "silver chalice"
BAR = "platinum bar"
TORCH = "ivory torch"
SCEPTRE = "sceptre"
ROPE = "rope"

TREASURES = [JEWELED_EGG, GOLD_COFFIN, CHALICE, BAR, TORCH, SCEPTRE]
TREASURE_POINTS = {
    JEWELED_EGG: 15,
    GOLD_COFFIN: 20,
    CHALICE: 15,
    BAR: 20,
    TORCH: 15,
    SCEPTRE: 15,
}
MAX_SCORE = sum(TREASURE_POINTS.values())  # 100

# Directions
DIRECTIONS = ["north", "south", "east", "west", "up", "down"]

# Room descriptions
ROOM_DESCRIPTIONS: dict[str, str] = {
    WEST_OF_HOUSE: "You are standing in an open field west of a white house, with a boarded front door.",
    NORTH_OF_HOUSE: "You are facing the north side of a white house. There is no door here, and all the windows are boarded.",
    SOUTH_OF_HOUSE: "You are facing the south side of a white house.",
    BEHIND_HOUSE: "You are behind the white house. A path leads into the forest to the east. In one corner of the house there is a small window which is slightly ajar.",
    KITCHEN: "You are in the kitchen of the white house. A table seems to have been used recently for the preparation of food.",
    LIVING_ROOM: "You are in the living room. There is a doorway to the east, a wooden door with strange gothic lettering to the west, which appears to be nailed shut, a trophy case, and a large oriental rug in the center of the room.",
    ATTIC: "This is the attic. The only exit is a stairway leading down. A nasty-looking knife is lying here.",
    FOREST_WEST: "This is a forest, with trees in all directions. To the east, there appears to be sunlight.",
    FOREST_EAST: "This is a dimly lit forest, with large trees all around.",
    CLEARING: "You are in a clearing, with a forest surrounding you on all sides. A path leads south.",
    CANYON_VIEW: "You are at the top of the Great Canyon, on the west wall. From here there is a marvelous view of the canyon and parts of the Frigid River below.",
    ROCKY_CRAWL: "You are in a rocky crawlway. A narrow passage leads east.",
    CELLAR: "You are in a dark and damp cellar with a narrow passageway leading north, and a crawlway to the south.",
    TROLL_ROOM: "This is a small room with passages to the east and south and a forbidding hole leading west. Bloodstains and deep scratches (perhaps made by straining fingers) parsing the walls.",
    EW_PASSAGE: "You are in a narrow east-west passageway. There is a narrow stairway leading up at the north end of the room.",
    ROUND_ROOM: "You are in a circular room with passages to the east, west, and south.",
    DOME_ROOM: "You are at the top of a dome-shaped room. Far below you can see a room with passages leading off in several directions.",
    TORCH_ROOM: "This is a large room with a prominent doorway leading to a down staircase. Above you is a large dome. In the center of the room there is a small white pedestal.",
    TEMPLE: "This is the north end of a large temple. On the east wall is an ancient inscription. In the center of the room is a large wooden door. Below you is a dark and winding stairway.",
    ALTAR: "This is the south end of a large temple. In front of you is what appears to be an altar.",
    LOUD_ROOM: "This is a large room with an extremely loud waterfall in the background. The acoustics here are such that any spoken word is drowned out by the roar.",
    TREASURE_ROOM: "This is a small treasure chamber. The walls are covered with dust and cobwebs.",
}

# Which rooms are dark (need lantern)?
DARK_ROOMS = {
    CELLAR, TROLL_ROOM, EW_PASSAGE, ROUND_ROOM, DOME_ROOM,
    TORCH_ROOM, TEMPLE, ALTAR, LOUD_ROOM, TREASURE_ROOM,
    ROCKY_CRAWL,
}

# Base connectivity graph: room -> {direction: room}
BASE_GRAPH: dict[str, dict[str, str]] = {
    WEST_OF_HOUSE: {"north": NORTH_OF_HOUSE, "south": SOUTH_OF_HOUSE, "west": FOREST_WEST},
    NORTH_OF_HOUSE: {"south": WEST_OF_HOUSE, "east": BEHIND_HOUSE, "west": FOREST_WEST},
    SOUTH_OF_HOUSE: {"north": WEST_OF_HOUSE, "east": BEHIND_HOUSE, "west": FOREST_EAST},
    BEHIND_HOUSE: {"north": NORTH_OF_HOUSE, "south": SOUTH_OF_HOUSE, "west": KITCHEN, "east": CLEARING},
    KITCHEN: {"east": BEHIND_HOUSE, "west": LIVING_ROOM, "up": ATTIC},
    LIVING_ROOM: {"east": KITCHEN, "down": CELLAR},
    ATTIC: {"down": KITCHEN},
    FOREST_WEST: {"east": WEST_OF_HOUSE, "south": FOREST_EAST, "north": CLEARING},
    FOREST_EAST: {"north": FOREST_WEST, "east": SOUTH_OF_HOUSE, "south": CANYON_VIEW},
    CLEARING: {"south": FOREST_WEST, "west": BEHIND_HOUSE},
    CANYON_VIEW: {"north": FOREST_EAST, "down": ROCKY_CRAWL},
    ROCKY_CRAWL: {"up": CANYON_VIEW, "east": CELLAR},
    CELLAR: {"north": TROLL_ROOM, "south": ROCKY_CRAWL, "up": LIVING_ROOM},
    TROLL_ROOM: {"south": CELLAR, "east": EW_PASSAGE, "west": ROUND_ROOM},
    EW_PASSAGE: {"west": TROLL_ROOM, "east": ROUND_ROOM, "up": DOME_ROOM},
    ROUND_ROOM: {"west": EW_PASSAGE, "east": LOUD_ROOM, "south": TREASURE_ROOM},
    DOME_ROOM: {"down": TORCH_ROOM},
    TORCH_ROOM: {"up": DOME_ROOM, "south": TEMPLE},
    TEMPLE: {"south": ALTAR, "north": TORCH_ROOM},
    ALTAR: {"north": TEMPLE},
    LOUD_ROOM: {"west": ROUND_ROOM},
    TREASURE_ROOM: {"north": ROUND_ROOM},
}

# Default item placements: room -> [items]
DEFAULT_ITEM_PLACEMENT: dict[str, list[str]] = {
    LIVING_ROOM: [LANTERN, SWORD],
    WEST_OF_HOUSE: [LEAFLET],
    BEHIND_HOUSE: [JEWELED_EGG],
    TORCH_ROOM: [TORCH],
    TREASURE_ROOM: [GOLD_COFFIN, BAR],
    ALTAR: [CHALICE],
    LOUD_ROOM: [SCEPTRE],
    ATTIC: [ROPE],
}

LANTERN_MAX_FUEL = 200


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZorkConfig:
    difficulty: str = "medium"  # easy | medium | hard
    seed: int = 0
    variant: str = "deterministic"  # deterministic | stochastic | punishment
    max_steps: int = 100
    num_rooms: int = 0  # 0 = use default for difficulty


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ZorkState(GameState):
    """Symbolic Zork I game state for MCTS."""

    def __init__(self, config: ZorkConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.steps = 0
        self.last_reward = 0.0
        self.reward_accumulator = 0.0
        self.dead = False
        self.death_message = ""

        # World state
        self.room: str = WEST_OF_HOUSE
        self.inventory: list[str] = []
        self.room_items: dict[str, list[str]] = {}
        self.trophy_case: list[str] = []
        self.troll_alive: bool = True
        self.lantern_on: bool = False
        self.lantern_fuel: int = LANTERN_MAX_FUEL
        self.rug_moved: bool = False  # reveals trapdoor in living room

        # Build world based on difficulty
        self._build_world()

    def _build_world(self) -> None:
        # num_rooms is the PRIMARY difficulty axis (like TextWorld's numLocations).
        # ALL_ROOMS is ordered surface-first → deep-underground-last, so slicing
        # to N rooms gives progressively harder maps.
        #
        # If num_rooms is 0, fall back to difficulty presets:
        #   easy   -> 10 rooms (surface only)
        #   medium -> 16 rooms (surface + underground layer 1-2)
        #   hard   -> 22 rooms (full map)
        num_rooms = self.config.num_rooms
        if num_rooms <= 0:
            num_rooms = {"easy": 10, "medium": 16, "hard": 22}.get(
                self.config.difficulty, 16
            )
        num_rooms = min(num_rooms, len(ALL_ROOMS))
        self.active_rooms = ALL_ROOMS[:num_rooms]

        # Troll: alive only if troll room is reachable
        self.troll_alive = TROLL_ROOM in self.active_rooms

        # Lantern fuel scales with map size (more rooms = more steps needed)
        base_fuel = {"easy": 9999, "medium": 200, "hard": 120}.get(
            self.config.difficulty, 200
        )
        # If num_rooms was explicitly set, scale fuel proportionally
        if self.config.num_rooms > 0:
            base_fuel = max(60, num_rooms * 10)
        self.lantern_fuel = base_fuel

        # Build connectivity (filtered to active rooms)
        active_set = set(self.active_rooms)
        self.graph: dict[str, dict[str, str]] = {}
        for room in self.active_rooms:
            exits = {}
            for direction, target in BASE_GRAPH.get(room, {}).items():
                if target in active_set:
                    exits[direction] = target
            self.graph[room] = exits

        # Place items — only in active rooms
        for room in self.active_rooms:
            items = DEFAULT_ITEM_PLACEMENT.get(room, [])
            self.room_items[room] = list(items)

        # Redistribute treasures from inactive rooms into the deepest active rooms
        # so there's always something to collect
        inactive_treasures: list[str] = []
        for room, items in DEFAULT_ITEM_PLACEMENT.items():
            if room not in active_set:
                for item in items:
                    if item in TREASURES:
                        inactive_treasures.append(item)

        if inactive_treasures:
            # Place them in the deepest reachable rooms (end of active_rooms)
            rng = random.Random(self.config.seed)
            eligible = [r for r in self.active_rooms if r not in (WEST_OF_HOUSE, LIVING_ROOM)]
            for treasure in inactive_treasures:
                target = rng.choice(eligible) if eligible else self.active_rooms[-1]
                if target not in self.room_items:
                    self.room_items[target] = []
                self.room_items[target].append(treasure)

        # Seed-based variety: shuffle one treasure to a different room
        if self.config.seed != 0:
            rng = random.Random(self.config.seed + 1000)
            treasure_rooms = [
                r for r in self.active_rooms
                if any(t in self.room_items.get(r, []) for t in TREASURES)
            ]
            eligible = [
                r for r in self.active_rooms
                if r not in (WEST_OF_HOUSE, LIVING_ROOM)
            ]
            if len(treasure_rooms) >= 2 and len(eligible) >= 2:
                rng.shuffle(eligible)
                src = rng.choice(treasure_rooms)
                items = self.room_items.get(src, [])
                treasures_here = [i for i in items if i in TREASURES]
                if treasures_here:
                    t = treasures_here[0]
                    items.remove(t)
                    dst = eligible[0] if eligible[0] != src else eligible[1]
                    if dst not in self.room_items:
                        self.room_items[dst] = []
                    self.room_items[dst].append(t)

    # ---------- GameState interface ----------

    def clone(self) -> "ZorkState":
        s = ZorkState.__new__(ZorkState)
        s.config = self.config
        s.rng = random.Random()
        s.rng.setstate(self.rng.getstate())
        s.steps = self.steps
        s.last_reward = self.last_reward
        s.reward_accumulator = self.reward_accumulator
        s.dead = self.dead
        s.death_message = self.death_message
        s.room = self.room
        s.inventory = list(self.inventory)
        s.room_items = {k: list(v) for k, v in self.room_items.items()}
        s.trophy_case = list(self.trophy_case)
        s.troll_alive = self.troll_alive
        s.lantern_on = self.lantern_on
        s.lantern_fuel = self.lantern_fuel
        s.rug_moved = self.rug_moved
        s.active_rooms = list(self.active_rooms)
        s.graph = {k: dict(v) for k, v in self.graph.items()}
        return s

    def current_player(self) -> int:
        return 0

    def legal_actions(self) -> list[Any]:
        if self.is_terminal():
            return []

        actions: list[str] = ["look", "inventory"]

        # Movement
        for direction, target in self.graph.get(self.room, {}).items():
            # Troll blocks passage west from troll room
            if self.room == TROLL_ROOM and direction == "west" and self.troll_alive:
                continue
            actions.append(f"go {direction}")

        # Special: living room trapdoor
        if self.room == LIVING_ROOM and not self.rug_moved:
            actions.append("move rug")

        # Take items from room
        visible_items = self._visible_room_items()
        for item in visible_items:
            actions.append(f"take {item}")

        # Drop items
        for item in self.inventory:
            actions.append(f"drop {item}")

        # Lantern controls
        if LANTERN in self.inventory:
            if not self.lantern_on:
                actions.append("turn on lantern")
            else:
                actions.append("turn off lantern")

        # Attack troll
        if self.room == TROLL_ROOM and self.troll_alive and SWORD in self.inventory:
            actions.append("attack troll with sword")

        # Trophy case (in living room)
        if self.room == LIVING_ROOM:
            for item in self.inventory:
                if item in TREASURES:
                    actions.append(f"put {item} in trophy case")

        # Read leaflet
        if LEAFLET in self.inventory:
            actions.append("read leaflet")

        return actions

    def apply_action(self, action: Any) -> None:
        action = str(action)
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(f"Illegal action {action!r}")

        # Stochastic variant: random action replacement
        executed = action
        if self.config.variant == "stochastic":
            meaningful = [a for a in legal if a not in ("look", "inventory")]
            if meaningful and self.rng.random() < 0.25:
                executed = self.rng.choice(meaningful)

        old_obs = self.look_text()
        self.last_reward = 0.0

        self._execute(executed)

        # Lantern fuel consumption
        if self.lantern_on:
            self.lantern_fuel -= 1
            if self.lantern_fuel <= 0:
                self.lantern_on = False
                # If in dark room, grue death
                if self.room in DARK_ROOMS:
                    self.dead = True
                    self.death_message = "Your lantern has run out of power. It is pitch black. You are likely to be eaten by a grue."

        # Punishment variant
        new_obs = self.look_text()
        if self.config.variant == "punishment" and new_obs == old_obs:
            self.last_reward = min(self.last_reward, -1.0)

        self.reward_accumulator += self.last_reward
        self.steps += 1

    def _execute(self, action: str) -> None:
        if action == "look" or action == "inventory":
            return

        if action.startswith("go "):
            direction = action[3:]
            target = self.graph.get(self.room, {}).get(direction)
            if target is not None:
                self.room = target
                # Dark room check
                if self.room in DARK_ROOMS and not self.lantern_on:
                    self.dead = True
                    self.death_message = "Oh no! You have walked into the slavering fangs of a lurking grue!"
            return

        if action == "move rug" and self.room == LIVING_ROOM:
            self.rug_moved = True
            # Reveal cellar access (already in graph via 'down')
            return

        if action.startswith("take "):
            item = action[5:]
            visible = self._visible_room_items()
            if item in visible:
                self.room_items[self.room].remove(item)
                self.inventory.append(item)
                if item in TREASURES:
                    self.last_reward += 0.1  # small reward for picking up treasures
            return

        if action.startswith("drop "):
            item = action[5:]
            if item in self.inventory:
                self.inventory.remove(item)
                if self.room not in self.room_items:
                    self.room_items[self.room] = []
                self.room_items[self.room].append(item)
            return

        if action == "turn on lantern" and LANTERN in self.inventory:
            self.lantern_on = True
            return

        if action == "turn off lantern" and LANTERN in self.inventory:
            self.lantern_on = False
            return

        if action == "attack troll with sword":
            # Combat success depends on RNG, but with sword it's reliable
            if self.rng.random() < 0.8:  # 80% success
                self.troll_alive = False
                self.last_reward += 0.1
            # else: miss, troll still alive
            return

        if action.startswith("put ") and action.endswith(" in trophy case"):
            item = action[4 : -len(" in trophy case")]
            if item in self.inventory and item in TREASURES:
                self.inventory.remove(item)
                self.trophy_case.append(item)
                points = TREASURE_POINTS.get(item, 10)
                self.last_reward += points / MAX_SCORE
            return

        if action == "read leaflet" and LEAFLET in self.inventory:
            # Flavor text, no game effect
            return

    def _visible_room_items(self) -> list[str]:
        """Items visible in the current room (respects darkness)."""
        if self.room in DARK_ROOMS and not self.lantern_on:
            return []
        return list(self.room_items.get(self.room, []))

    def is_terminal(self) -> bool:
        if self.dead:
            return True
        if self.steps >= self.config.max_steps:
            return True
        # Win: all treasures in trophy case
        if len(self.trophy_case) == len(TREASURES):
            return True
        return False

    def returns(self) -> list[float]:
        if self.dead:
            return [-1.0]
        score = self.score()
        normalized = score / MAX_SCORE
        if len(self.trophy_case) == len(TREASURES):
            return [1.0]
        return [normalized]

    def state_key(self) -> str:
        inv = ",".join(sorted(self.inventory))
        trophy = ",".join(sorted(self.trophy_case))
        room_items_key = "|".join(
            f"{r}:{','.join(sorted(items))}"
            for r, items in sorted(self.room_items.items())
            if items
        )
        rng_tag = hash(str(self.rng.getstate()[1][:5])) if self.config.variant == "stochastic" else 0
        return (
            f"room={self.room}|inv={inv}|trophy={trophy}|"
            f"troll={int(self.troll_alive)}|lantern={int(self.lantern_on)}|"
            f"fuel={self.lantern_fuel}|rug={int(self.rug_moved)}|"
            f"dead={int(self.dead)}|steps={self.steps}|"
            f"items={room_items_key}|rng={rng_tag}"
        )

    def __str__(self) -> str:
        return f"{self.observation_text()}\nScore: {self.score()}/{MAX_SCORE} | Steps: {self.steps}/{self.config.max_steps}"

    # ---------- helpers exposed to LLM heuristics ----------

    def score(self) -> int:
        """Current score based on treasures in trophy case."""
        return sum(TREASURE_POINTS.get(t, 0) for t in self.trophy_case)

    def look_text(self) -> str:
        """Description of current location."""
        if self.room in DARK_ROOMS and not self.lantern_on:
            return "It is pitch black. You are likely to be eaten by a grue."

        desc = ROOM_DESCRIPTIONS.get(self.room, f"You are in {self.room}.")
        parts = [desc]

        # Exits
        exits = list(self.graph.get(self.room, {}).keys())
        if exits:
            parts.append("Exits: " + ", ".join(sorted(exits)) + ".")

        # Items on ground
        items = self.room_items.get(self.room, [])
        if items:
            for item in items:
                parts.append(f"There is a {item} here.")

        # Troll
        if self.room == TROLL_ROOM and self.troll_alive:
            parts.append("A nasty-looking troll, brandishing a bloody axe, blocks all passages out of the room to the west.")

        # Trophy case contents
        if self.room == LIVING_ROOM and self.trophy_case:
            parts.append("The trophy case contains: " + ", ".join(self.trophy_case) + ".")
        elif self.room == LIVING_ROOM:
            parts.append("The trophy case is empty.")

        # Rug
        if self.room == LIVING_ROOM and not self.rug_moved:
            parts.append("Under the rug you might find something interesting.")

        return " ".join(parts)

    def inventory_text(self) -> str:
        """Description of items in inventory."""
        if not self.inventory:
            return "You are empty-handed."
        items_desc = ", ".join(self.inventory)
        extra = ""
        if LANTERN in self.inventory:
            status = "on" if self.lantern_on else "off"
            extra = f" (lantern is {status}, fuel: {self.lantern_fuel})"
        return f"You are carrying: {items_desc}.{extra}"

    def observation_text(self) -> str:
        """Full observation: look + inventory."""
        return f"{self.look_text()}\n{self.inventory_text()}"

    def treasures_remaining(self) -> list[str]:
        """Treasures not yet in trophy case."""
        in_case = set(self.trophy_case)
        return [t for t in TREASURES if t not in in_case]

    def distance_to_room(self, target: str) -> int:
        """BFS shortest path length from current room to target."""
        if self.room == target:
            return 0
        frontier = [(self.room, 0)]
        seen = {self.room}
        while frontier:
            room, dist = frontier.pop(0)
            for nxt in self.graph.get(room, {}).values():
                if nxt == target:
                    return dist + 1
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append((nxt, dist + 1))
        return 999


# ---------------------------------------------------------------------------
# Game factory
# ---------------------------------------------------------------------------


class Zork(Game):
    """Zork I game factory for the MCTS framework."""

    def __init__(
        self,
        difficulty: str = "medium",
        seed: int = 0,
        variant: str = "deterministic",
        max_steps: int = 100,
        num_rooms: int = 0,
    ):
        self.config = ZorkConfig(
            difficulty=difficulty,
            seed=seed,
            variant=variant,
            max_steps=max_steps,
            num_rooms=num_rooms,
        )

    def new_initial_state(self) -> ZorkState:
        return ZorkState(self.config)

    def num_players(self) -> int:
        return 1

    def name(self) -> str:
        return (
            f"Zork({self.config.difficulty}, variant={self.config.variant}, "
            f"seed={self.config.seed})"
        )
