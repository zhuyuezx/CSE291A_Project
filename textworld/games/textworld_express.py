"""
TextWorld-Express adapter for the MCTS framework.

This wraps the deterministic TextWorld-Express homework environments
used in hw2_part2.ipynb behind the GameState/Game interface expected by
the MCTS engine.

Key design choice:
    TextWorld-Express does not expose a lightweight clone API here, so
    cloned states reconstruct the underlying environment lazily by
    resetting from the original seed and replaying the action history.
    This is slower than native board games but keeps the integration
    correct and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .game_interface import Game, GameState


DEFAULT_FILTERED_ACTIONS = {"inventory", "look around"}


def _require_textworld():
    try:
        from textworld_express import TextWorldExpressEnv
    except ImportError as exc:
        raise ImportError(
            "textworld-express is required for the TextWorld MCTS adapter. "
            "Install it in the environment running the notebook/script."
        ) from exc
    return TextWorldExpressEnv


def _normalize_valid_actions(valid_actions: list[str]) -> list[str]:
    return [a for a in valid_actions if a not in DEFAULT_FILTERED_ACTIONS]


def _format_state_text(look: str, inventory: str) -> str:
    if "Your inventory is currently empty" in inventory:
        inventory = "Inventory: empty"
    if "(maximum capacity is 2 items)" in inventory:
        inventory = inventory.replace("(maximum capacity is 2 items)", "")
    return f"{look}\n{inventory}".strip()


@dataclass(frozen=True)
class TextWorldConfig:
    game_type: str = "coin"
    game_params: str = "numLocations=5,includeDoors=1,numDistractorItems=0"
    seed: int = 3
    env_step_limit: int = 100
    game_fold: str = "train"
    generate_gold_path: bool = True


class TextWorldExpressState(GameState):
    """
    Single-player TextWorld state.

    Actions are strings (e.g. "move east", "take coin").
    Rewards are the cumulative environment rewards seen so far.
    """

    def __init__(self, config: TextWorldConfig, max_steps: int = 50):
        self.config = config
        self.max_steps = max_steps
        self.steps = 0
        self.history: list[str] = []
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.done = False

        self._env = None
        self.observation = ""
        self.inventory = ""
        self.valid_actions: list[str] = []
        self._reset_fresh()

    # ------------------------------------------------------------------
    # GameState interface
    # ------------------------------------------------------------------

    def clone(self) -> "TextWorldExpressState":
        new = TextWorldExpressState.__new__(TextWorldExpressState)
        new.config = self.config
        new.max_steps = self.max_steps
        new.steps = self.steps
        new.history = list(self.history)
        new.total_reward = self.total_reward
        new.last_reward = self.last_reward
        new.done = self.done
        new.observation = self.observation
        new.inventory = self.inventory
        new.valid_actions = list(self.valid_actions)
        new._env = None
        return new

    def current_player(self) -> int:
        return 0

    def legal_actions(self) -> list[Any]:
        return list(self.valid_actions)

    def apply_action(self, action: Any) -> None:
        if self.done:
            return
        self._ensure_env()
        _, reward, done, infos = self._env.step(str(action))
        self.history.append(str(action))
        self.steps += 1
        self.last_reward = float(reward)
        self.total_reward += float(reward)
        self.done = bool(done) or self.steps >= self.max_steps
        self._update_from_infos(infos)

    def is_terminal(self) -> bool:
        return self.done or self.steps >= self.max_steps

    def returns(self) -> list[float]:
        # Use cumulative reward as the single-player return. In these
        # homework environments, success is typically signaled by 1.0.
        return [float(self.total_reward)]

    def state_key(self) -> str:
        return "||".join(
            [
                self.config.game_type,
                self.config.game_params,
                self.observation.strip(),
                self.inventory.strip(),
                ",".join(sorted(self.valid_actions)),
                str(self.steps),
            ]
        )

    def __str__(self) -> str:
        return _format_state_text(self.observation, self.inventory)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_env(self):
        TextWorldExpressEnv = _require_textworld()
        env = TextWorldExpressEnv(envStepLimit=self.config.env_step_limit)
        env.load(gameName=self.config.game_type, gameParams=self.config.game_params)
        return env

    def _reset_fresh(self) -> None:
        self._env = self._make_env()
        _, infos = self._env.reset(
            seed=self.config.seed,
            gameFold=self.config.game_fold,
            generateGoldPath=self.config.generate_gold_path,
        )
        self._update_from_infos(infos)

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        self._env = self._make_env()
        _, infos = self._env.reset(
            seed=self.config.seed,
            gameFold=self.config.game_fold,
            generateGoldPath=self.config.generate_gold_path,
        )
        for action in self.history:
            _, _, _, infos = self._env.step(action)
        self._update_from_infos(infos)

    def _update_from_infos(self, infos: dict[str, Any]) -> None:
        self.observation = infos.get("look", infos.get("observation", ""))
        self.inventory = infos.get("inventory", "")
        self.valid_actions = _normalize_valid_actions(list(infos.get("validActions", [])))


class TextWorldExpressGame(Game):
    def __init__(
        self,
        game_type: str = "coin",
        game_params: str = "numLocations=5,includeDoors=1,numDistractorItems=0",
        seed: int = 3,
        env_step_limit: int = 100,
        max_steps: int = 50,
        game_fold: str = "train",
        generate_gold_path: bool = True,
    ):
        self.config = TextWorldConfig(
            game_type=game_type,
            game_params=game_params,
            seed=seed,
            env_step_limit=env_step_limit,
            game_fold=game_fold,
            generate_gold_path=generate_gold_path,
        )
        self.max_steps = max_steps

    def new_initial_state(self) -> TextWorldExpressState:
        return TextWorldExpressState(self.config, max_steps=self.max_steps)

    def num_players(self) -> int:
        return 1

    def name(self) -> str:
        return f"TextWorldExpress({self.config.game_type}, {self.config.game_params})"
