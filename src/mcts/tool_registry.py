# src/mcts/tool_registry.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from src.tools.base import ToolType, load_tool_from_file


@dataclass
class RegisteredTool:
    name: str
    tool_type: ToolType
    run_fn: Callable


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        tool_type: ToolType,
        run_fn: Callable,
    ) -> None:
        self._tools[name] = RegisteredTool(
            name=name, tool_type=tool_type, run_fn=run_fn
        )

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get_tools(self, tool_type: ToolType) -> list[RegisteredTool]:
        return [t for t in self._tools.values() if t.tool_type == tool_type]

    def list_all(self) -> list[str]:
        return list(self._tools.keys())

    def load_from_directory(self, directory: str) -> None:
        for filename in os.listdir(directory):
            if not filename.endswith(".py"):
                continue
            filepath = os.path.join(directory, filename)
            try:
                meta, run_fn = load_tool_from_file(filepath)
                self.register(meta.name, meta.type, run_fn)
            except (ValueError, Exception):
                continue  # skip invalid files
