"""
Game-aware visualization configuration helpers.

Reads the active pipeline configuration from MCTS_tools/hyperparams and
training_logic so visualization scripts can auto-target the current game.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional


_TOOL_CREATION_ROOT = Path(__file__).resolve().parent.parent
_MCTS_TOOLS_DIR = _TOOL_CREATION_ROOT / "MCTS_tools"
_HYPERPARAMS_PATH = _MCTS_TOOLS_DIR / "hyperparams" / "default_hyperparams.py"
_TRAINING_LOGIC_DIR = _MCTS_TOOLS_DIR / "training_logic"
_SIMULATION_DIR = _MCTS_TOOLS_DIR / "simulation"
_RECORDS_DIR = _TOOL_CREATION_ROOT / "mcts" / "records"
_SUMMARY_PATH = _RECORDS_DIR / "optimization_summary.json"


def _load_module_from_file(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_optimized_tool_path(game_name: str) -> Optional[Path]:
    """Resolve simulation tool for the active game (from default_hyperparams)."""
    candidates = [
        _SIMULATION_DIR / f"{game_name}_simulation.py",
        _SIMULATION_DIR / f"{game_name}_mcts.py",
        _SIMULATION_DIR / "simulation.py",
        _SIMULATION_DIR / "default_simulation.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _resolve_optimized_tool_path_for_game(game_key: str) -> Optional[Path]:
    """
    Resolve simulation tool for an explicit game (sokoban or rush_hour).
    Avoids loading Rush Hour–specific simulation.py when game is Sokoban.
    """
    if game_key == "sokoban":
        candidates = [
            _SIMULATION_DIR / "sokoban_simulation.py",
            _SIMULATION_DIR / "sokoban_mcts.py",
            _SIMULATION_DIR / "simulation.py",  # LLM-generated (pipeline writes here)
            _SIMULATION_DIR / "default_simulation.py",
        ]
    else:
        candidates = [
            _SIMULATION_DIR / f"{game_key}_simulation.py",
            _SIMULATION_DIR / f"{game_key}_mcts.py",
            _SIMULATION_DIR / "simulation.py",
            _SIMULATION_DIR / "default_simulation.py",
        ]
    for path in candidates:
        if path.exists():
            return path
    return None


def get_game_config() -> Dict[str, Any]:
    """Return active game configuration from default_hyperparams.py."""
    hp_mod = _load_module_from_file("viz_default_hyperparams", _HYPERPARAMS_PATH)
    game_module_name = getattr(hp_mod, "GAME_MODULE", "mcts.games")
    game_class_name = getattr(hp_mod, "GAME_CLASS", "Sokoban")
    game_name = str(getattr(hp_mod, "GAME_NAME", "sokoban"))
    constructor_kwargs = dict(getattr(hp_mod, "CONSTRUCTOR_KWARGS", {}))
    training_logic_name = str(getattr(hp_mod, "TRAINING_LOGIC", "sokoban_training"))
    training_logic_path = _TRAINING_LOGIC_DIR / f"{training_logic_name}.py"

    game_module = importlib.import_module(game_module_name)
    game_class = getattr(game_module, game_class_name)
    training_mod = _load_module_from_file(
        f"viz_training_{training_logic_name}",
        training_logic_path,
    )
    levels = list(getattr(training_mod, "LEVELS", []))
    hyperparams = dict(getattr(hp_mod, "get_hyperparams")())

    return {
        "tool_creation_root": _TOOL_CREATION_ROOT,
        "records_dir": _RECORDS_DIR,
        "summary_path": _SUMMARY_PATH,
        "mcts_tools_dir": _MCTS_TOOLS_DIR,
        "game_name": game_name,
        "game_module_name": game_module_name,
        "game_class_name": game_class_name,
        "game_class": game_class,
        "constructor_kwargs": constructor_kwargs,
        "training_logic": training_logic_name,
        "levels": levels,
        "hyperparams": hyperparams,
        "optimized_tool_path": _resolve_optimized_tool_path(game_name),
    }


# Known games for explicit override (e.g. --game sokoban when active config is rush_hour)
_GAME_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "sokoban": {
        "game_name": "Sokoban",
        "game_module_name": "mcts.games",
        "game_class_name": "Sokoban",
        "training_logic_name": "sokoban_training",
        "constructor_kwargs": {"max_steps": 200},
        "default_level": "level3",
    },
    "rush_hour": {
        "game_name": "Rush Hour",
        "game_module_name": "mcts.games",
        "game_class_name": "RushHour",
        "training_logic_name": "rush_hour_training",
        "constructor_kwargs": {"max_moves": 80},
        "default_level": "easy1",
    },
}


def get_game_config_for(game_name: str) -> Dict[str, Any]:
    """
    Return config for a specific game (e.g. 'sokoban' or 'rush_hour') without
    reading the active pipeline game from default_hyperparams.
    Use this to generate trajectory/GIF for a game other than the current config.
    """
    key = (game_name or "").strip().lower().replace(" ", "_")
    if key not in _GAME_OVERRIDES:
        return get_game_config()
    over = _GAME_OVERRIDES[key]
    game_module = importlib.import_module(over["game_module_name"])
    game_class = getattr(game_module, over["game_class_name"])
    training_path = _TRAINING_LOGIC_DIR / f"{over['training_logic_name']}.py"
    training_mod = _load_module_from_file(
        f"viz_training_{over['training_logic_name']}",
        training_path,
    )
    levels = list(getattr(training_mod, "LEVELS", []))
    hp_path = _MCTS_TOOLS_DIR / "hyperparams" / "default_hyperparams.py"
    hp_mod = _load_module_from_file("viz_hp_override", hp_path)
    hyperparams = dict(getattr(hp_mod, "get_hyperparams", lambda: {})())

    return {
        "tool_creation_root": _TOOL_CREATION_ROOT,
        "records_dir": _RECORDS_DIR,
        "summary_path": _SUMMARY_PATH,
        "mcts_tools_dir": _MCTS_TOOLS_DIR,
        "game_name": over["game_name"],
        "game_module_name": over["game_module_name"],
        "game_class_name": over["game_class_name"],
        "game_class": game_class,
        "constructor_kwargs": dict(over["constructor_kwargs"]),
        "training_logic": over["training_logic_name"],
        "levels": levels,
        "hyperparams": hyperparams,
        "optimized_tool_path": _resolve_optimized_tool_path_for_game(key),
    }

