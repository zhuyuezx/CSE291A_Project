# src/games/meta_registry.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GameMeta:
    name: str
    is_single_player: bool
    min_return: float
    max_return: float
    metric_name: str   # "win_rate" | "avg_score" | "success_rate"
    max_sim_depth: int


GAME_META: dict[str, GameMeta] = {
    "connect_four":      GameMeta("connect_four",      False, -1.0,    1.0,   "win_rate",      42),
    "tic_tac_toe":       GameMeta("tic_tac_toe",        False, -1.0,    1.0,   "win_rate",       9),
    "quoridor":          GameMeta("quoridor",           False, -1.0,    1.0,   "win_rate",      200),
    "chess":             GameMeta("chess",              False, -1.0,    1.0,   "win_rate",      200),
    "pathfinding":       GameMeta("pathfinding",        True,   0.0,    1.0,   "success_rate",  500),
    "morpion_solitaire": GameMeta("morpion_solitaire",  True,   0.0,   35.0,  "avg_score",      35),
    "2048":              GameMeta("2048",                True,   0.0, 20000.0, "avg_score",    1000),
    "zork":              GameMeta("zork",               True,   0.0,  350.0,  "avg_score",     500),
}
