# src/tools/base.py
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class ToolType(str, Enum):
    STATE_EVALUATOR = "state_evaluator"
    ACTION_FILTER = "action_filter"
    ROLLOUT_POLICY = "rollout_policy"
    SELECTION_PRIOR = "selection_prior"
    REWARD_SHAPER = "reward_shaper"
    MACRO_ACTION = "macro_action"


@dataclass
class ToolMeta:
    name: str
    type: ToolType
    description: str


def validate_tool_meta(meta: dict) -> ToolMeta:
    required = {"name", "type", "description"}
    missing = required - set(meta.keys())
    if missing:
        raise ValueError(f"Missing required fields in __TOOL_META__: {missing}")

    try:
        tool_type = ToolType(meta["type"])
    except ValueError:
        valid = [t.value for t in ToolType]
        raise ValueError(
            f"Invalid tool type '{meta['type']}'. Must be one of: {valid}"
        )

    return ToolMeta(name=meta["name"], type=tool_type, description=meta["description"])


def load_tool_from_file(filepath: str) -> tuple[ToolMeta, Callable]:
    spec = importlib.util.spec_from_file_location("tool_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "__TOOL_META__"):
        raise ValueError(
            f"Tool file {filepath} missing __TOOL_META__ dict"
        )
    if not hasattr(module, "run"):
        raise ValueError(f"Tool file {filepath} missing run() function")

    meta = validate_tool_meta(module.__TOOL_META__)
    return meta, module.run
