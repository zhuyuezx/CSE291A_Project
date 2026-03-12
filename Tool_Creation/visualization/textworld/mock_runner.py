#!/usr/bin/env python3
"""
TextWorld visualization mock runner.

Creates a tiny mock trajectory and exports a GIF to:
  Tool_Creation/visualization/output/textworld/
"""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
OUTPUT_DIR = PROJECT_ROOT / "Tool_Creation" / "visualization" / "output" / "textworld"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Tool_Creation"))

from visualization.textworld import TrajectoryVisualizer


def create_mock_textworld_game_state():
    return {
        "world_graph": {0: [1, 4], 1: [0, 2], 2: [1, 3], 3: [2], 4: [0, 5], 5: [4]},
        "room_descriptions": {
            0: "Kitchen with a counter and sink",
            1: "Living room with couch and TV",
            2: "Bedroom with bed and dresser",
            3: "Bathroom with shower and toilet",
            4: "Hallway connecting rooms",
            5: "Garden with flowers and trees",
        },
    }


def create_mock_trajectory():
    return [
        {
            "observation": {
                "room": 0,
                "description": "You are in the kitchen. There is a counter with a key on it.",
                "inventory": [],
                "quest_progress": {"items": "0/1"},
            },
            "action": "take key",
            "reward": 1.0,
            "done": False,
            "legal_actions": ["go east", "take key"],
        },
        {
            "observation": {
                "room": 0,
                "description": "You are in the kitchen. You have a key.",
                "inventory": ["key"],
                "quest_progress": {"items": "1/1"},
            },
            "action": "go east",
            "reward": 0.0,
            "done": False,
            "legal_actions": ["go east", "go north"],
        },
        {
            "observation": {
                "room": 1,
                "description": "You are in the living room. There is a door to the north.",
                "inventory": ["key"],
                "quest_progress": {"items": "1/1"},
            },
            "action": "go north",
            "reward": 0.0,
            "done": False,
            "legal_actions": ["go west", "go north"],
        },
        {
            "observation": {
                "room": 2,
                "description": "You are in the bedroom. There is a treasure chest.",
                "inventory": ["key"],
                "quest_progress": {"items": "1/1"},
            },
            "action": "open chest",
            "reward": 10.0,
            "done": True,
            "legal_actions": ["go south", "open chest"],
        },
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    game_state = create_mock_textworld_game_state()
    trajectory = create_mock_trajectory()
    viz = TrajectoryVisualizer(trajectory, game_state=game_state)
    out = OUTPUT_DIR / "mock_textworld_trajectory.gif"
    viz.visualize(output_path=out, fps=1)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

