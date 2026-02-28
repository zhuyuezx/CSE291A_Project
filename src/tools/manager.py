# src/tools/manager.py
from __future__ import annotations

import json
import os
import shutil


class ToolPoolManager:
    def __init__(self, pool_dir: str = "tool_pool"):
        self.pool_dir = pool_dir
        os.makedirs(pool_dir, exist_ok=True)

    def save_tool(self, game_name: str, tool_name: str, code: str) -> str:
        game_dir = os.path.join(self.pool_dir, game_name)
        os.makedirs(game_dir, exist_ok=True)
        path = os.path.join(game_dir, f"{tool_name}.py")
        with open(path, "w") as f:
            f.write(code)
        return path

    def load_metadata(self) -> dict:
        meta_path = os.path.join(self.pool_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return {}
        with open(meta_path) as f:
            return json.load(f)

    def update_metadata(self, tool_name: str, info: dict) -> None:
        meta = self.load_metadata()
        meta[tool_name] = info
        meta_path = os.path.join(self.pool_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def list_tools_for_game(self, game_name: str) -> list[str]:
        game_dir = os.path.join(self.pool_dir, game_name)
        if not os.path.exists(game_dir):
            return []
        return [f for f in os.listdir(game_dir) if f.endswith(".py")]

    def promote_to_global(self, game_name: str, tool_name: str) -> None:
        src = os.path.join(self.pool_dir, game_name, f"{tool_name}.py")
        dst_dir = os.path.join(self.pool_dir, "global")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"{tool_name}.py")
        shutil.copy2(src, dst)

    def get_all_tools_for_game(self, game_name: str) -> list[str]:
        """Get paths for global + game-specific tools."""
        paths = []
        global_dir = os.path.join(self.pool_dir, "global")
        if os.path.exists(global_dir):
            for f in os.listdir(global_dir):
                if f.endswith(".py"):
                    paths.append(os.path.join(global_dir, f))

        game_dir = os.path.join(self.pool_dir, game_name)
        if os.path.exists(game_dir):
            for f in os.listdir(game_dir):
                if f.endswith(".py"):
                    paths.append(os.path.join(game_dir, f))

        return paths
