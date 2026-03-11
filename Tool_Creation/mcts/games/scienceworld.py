"""
ScienceWorld adapter for the MCTS framework.

Wraps the ScienceWorld text-based science experiment environment behind
the GameState/Game interface expected by the MCTS engine.

ScienceWorld runs on a JVM via py4j. Creating multiple JVM instances is
expensive, so we use a shared environment with action-history replay.
Cloned states store their history and reconstruct env state lazily
when needed.  This is correct for single-threaded MCTS.

Install:  pip install scienceworld
Requires: Java 1.8+
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from .game_interface import Game, GameState


TASK_ID_TO_NAME: dict[str, str] = {
    "1-1": "boil",
    "1-2": "melt",
    "1-3": "freeze",
    "1-4": "change-the-state-of-matter-of",
    "2-1": "use-thermometer",
    "2-2": "measure-melting-point-known-substance",
    "2-3": "measure-melting-point-unknown-substance",
    "3-1": "power-component",
    "3-2": "power-component-renewable-vs-nonrenewable-energy",
    "3-3": "test-conductivity",
    "3-4": "test-conductivity-of-unknown-substances",
    "4-1": "find-living-thing",
    "4-2": "find-non-living-thing",
    "4-3": "find-plant",
    "4-4": "find-animal",
    "5-1": "grow-plant",
    "5-2": "grow-fruit",
    "6-1": "chemistry-mix",
    "6-2": "chemistry-mix-paint-secondary-color",
    "6-3": "chemistry-mix-paint-tertiary-color",
    "7-1": "lifespan-longest-lived",
    "7-2": "lifespan-shortest-lived",
    "7-3": "lifespan-longest-lived-then-shortest-lived",
    "8-1": "identify-life-stages-1",
    "8-2": "identify-life-stages-2",
    "9-1": "inclined-plane-determine-angle",
    "9-2": "inclined-plane-friction-named-surfaces",
    "9-3": "inclined-plane-friction-unnamed-surfaces",
    "10-1": "mendelian-genetics-known-plant",
    "10-2": "mendelian-genetics-unknown-plant",
}

EASY_SIMPLIFICATIONS = (
    "teleportAction,openDoors,selfWateringFlowerPots,noElectricalAction"
)


def _require_scienceworld():
    try:
        from scienceworld import ScienceWorldEnv
    except ImportError as exc:
        raise ImportError(
            "scienceworld is required for the ScienceWorld MCTS adapter. "
            "Install it with:  pip install scienceworld\n"
            "Java 1.8+ must also be installed."
        ) from exc
    return ScienceWorldEnv


def get_simplifications(task_id: str, preset: str = "easy") -> str:
    """Return simplification string for a task, respecting the preset."""
    if preset != "easy":
        return ""
    parts = ["teleportAction", "openDoors", "selfWateringFlowerPots"]
    if not task_id.startswith("3-"):
        parts.append("noElectricalAction")
    return ",".join(parts)


def resolve_task_name(task: str) -> str:
    """Accept either a task ID ('1-1') or a task name ('boil')."""
    return TASK_ID_TO_NAME.get(task, task)


# ── Shared JVM environment pool ──────────────────────────────────────

class _SharedEnvPool:
    """
    Manages a single ScienceWorldEnv instance shared across all states.

    The pool tracks which (task, variation, history) it was last synced to,
    so sequential steps on the same branch (the common MCTS simulation
    case) are O(1) instead of requiring a full replay.
    """

    _lock = threading.Lock()
    _env: Any = None
    _synced_task: str | None = None
    _synced_variation: int | None = None
    _synced_simplifications: str | None = None
    _synced_history: list[str] | None = None

    # Default max score (ScienceWorld often reports score 0-100)
    DEFAULT_MAX_SCORE = 100.0

    @classmethod
    def _get_env(cls, step_limit: int = 100):
        if cls._env is None:
            ScienceWorldEnv = _require_scienceworld()
            # ScienceWorldEnv(task_name, jar_path, envStepLimit=...)
            cls._env = ScienceWorldEnv("", "", envStepLimit=step_limit)
        return cls._env

    @classmethod
    def _full_reset(
        cls, task_name: str, variation: int, simplifications: str,
        step_limit: int = 100,
    ):
        env = cls._get_env(step_limit)
        # load(taskName, variationIdx, simplificationStr) - positional
        env.load(task_name, variation, simplifications)
        env.reset()
        cls._synced_task = task_name
        cls._synced_variation = variation
        cls._synced_simplifications = simplifications
        cls._synced_history = []
        return env

    @classmethod
    def _get_valid_actions(cls, env: Any) -> list[str]:
        """Return list of action strings from env (snake_case API)."""
        try:
            combos = env.get_valid_action_object_combinations_with_templates()
        except AttributeError:
            combos = getattr(
                env, "getValidActionObjectCombinations",
                lambda: [],
            )()
        if not combos:
            return []
        actions = []
        for item in combos:
            if isinstance(item, dict) and "action" in item:
                actions.append(str(item["action"]).strip())
            else:
                actions.append(str(item).strip())
        return actions

    @classmethod
    def _get_task_description(cls, env: Any) -> str:
        try:
            return str(env.get_task_description() or "")
        except AttributeError:
            return str(getattr(env, "getTaskDescription", lambda: "")() or "")

    @classmethod
    def init_state(
        cls,
        task_name: str,
        variation: int,
        simplifications: str,
        step_limit: int = 100,
        max_actions: int = 0,
    ) -> tuple[str, float, float, list[str], str]:
        """Load + reset, return (obs, score, max_score, valid_actions, task_desc)."""
        with cls._lock:
            env = cls._full_reset(
                task_name, variation, simplifications, step_limit=step_limit
            )
            # reset() returns (observation, info_dict)
            obs, info = env.reset()
            obs = str(obs or "")
            score = float(info.get("score", 0.0))
            max_score = float(
                info.get("maxScore", info.get("max_score", cls.DEFAULT_MAX_SCORE))
            )
            if max_score <= 0:
                max_score = cls.DEFAULT_MAX_SCORE
            valid_actions = cls._get_valid_actions(env)
            if max_actions > 0 and len(valid_actions) > max_actions:
                valid_actions = valid_actions[:max_actions]
            task_desc = cls._get_task_description(env)
            if not task_desc and obs:
                task_desc = obs[:500]
            return obs, score, max_score, valid_actions, task_desc

    @classmethod
    def sync_and_step(
        cls,
        task_name: str,
        variation: int,
        simplifications: str,
        history: list[str],
        action: str,
        step_limit: int = 100,
        max_actions: int = 0,
    ) -> tuple[str, float, bool, float, float, list[str]]:
        """
        Ensure env is at *history*, step with *action*.

        Returns (obs, reward, done, score, max_score, valid_actions).
        """
        with cls._lock:
            env = cls._get_env(step_limit)

            can_extend = (
                cls._synced_task == task_name
                and cls._synced_variation == variation
                and cls._synced_simplifications == simplifications
                and cls._synced_history is not None
                and len(history) >= len(cls._synced_history)
                and history[: len(cls._synced_history)] == cls._synced_history
            )

            if can_extend:
                for a in history[len(cls._synced_history):]:
                    env.step(a)
            else:
                cls._full_reset(
                    task_name, variation, simplifications, step_limit=step_limit
                )
                for a in history:
                    env.step(a)

            # step() returns (observation, reward, isCompleted, info)
            obs, reward, done, info = env.step(action)
            cls._synced_history = list(history) + [action]

            score = float(info.get("score", 0.0))
            max_score = float(
                info.get("maxScore", info.get("max_score", cls.DEFAULT_MAX_SCORE))
            )
            if max_score <= 0:
                max_score = cls.DEFAULT_MAX_SCORE
            if not done:
                valid_actions = cls._get_valid_actions(env)
                if max_actions > 0 and len(valid_actions) > max_actions:
                    valid_actions = valid_actions[:max_actions]
            else:
                valid_actions = []

            return obs, float(reward), bool(done), score, max_score, valid_actions

    @classmethod
    def close(cls):
        with cls._lock:
            if cls._env is not None:
                try:
                    cls._env.close()
                except Exception:
                    pass
                cls._env = None
            cls._synced_task = None
            cls._synced_variation = None
            cls._synced_simplifications = None
            cls._synced_history = None


# ── Config ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScienceWorldConfig:
    task_name: str
    variation: int = 0
    simplifications: str = ""
    max_steps: int = 100
    max_actions: int = 0  # 0 = no cap; else cap legal_actions to this many (reduces MCTS branching)


# ── GameState ─────────────────────────────────────────────────────────

class ScienceWorldState(GameState):
    """
    Single-player ScienceWorld state backed by action-history replay.

    Public helper attributes exposed for LLM heuristics:
        observation, task_description, score, max_score, valid_actions
    """

    def __init__(self, config: ScienceWorldConfig):
        self.config = config
        self.history: list[str] = []
        self.steps: int = 0
        self.done: bool = False
        self.score: float = 0.0
        self.max_score: float = 1.0
        self.observation: str = ""
        self.task_description: str = ""
        self.valid_actions: list[str] = []
        self._init_from_env()

    def _init_from_env(self) -> None:
        obs, score, max_score, valid_actions, task_desc = _SharedEnvPool.init_state(
            self.config.task_name,
            self.config.variation,
            self.config.simplifications,
            step_limit=self.config.max_steps,
            max_actions=self.config.max_actions,
        )
        self.observation = obs
        self.score = score
        self.max_score = max_score if max_score > 0 else 1.0
        self.valid_actions = valid_actions
        self.task_description = task_desc

    # -- GameState interface -------------------------------------------

    def clone(self) -> ScienceWorldState:
        new = ScienceWorldState.__new__(ScienceWorldState)
        new.config = self.config
        new.history = list(self.history)
        new.steps = self.steps
        new.done = self.done
        new.score = self.score
        new.max_score = self.max_score
        new.observation = self.observation
        new.task_description = self.task_description
        new.valid_actions = list(self.valid_actions)
        return new

    def current_player(self) -> int:
        return 0

    def legal_actions(self) -> list[Any]:
        return list(self.valid_actions)

    def apply_action(self, action: Any) -> None:
        if self.done:
            return
        obs, reward, done, score, max_score, valid_actions = (
            _SharedEnvPool.sync_and_step(
                self.config.task_name,
                self.config.variation,
                self.config.simplifications,
                self.history,
                str(action),
                step_limit=self.config.max_steps,
                max_actions=self.config.max_actions,
            )
        )
        self.history.append(str(action))
        self.steps += 1
        self.observation = obs
        self.done = done or self.steps >= self.config.max_steps
        self.score = score
        self.max_score = max_score if max_score > 0 else 1.0
        self.valid_actions = valid_actions if not self.done else []

    def is_terminal(self) -> bool:
        return self.done or self.steps >= self.config.max_steps

    def returns(self) -> list[float]:
        if self.max_score > 0:
            return [self.score / self.max_score]
        return [0.0]

    def state_key(self) -> str:
        return "||".join([
            self.config.task_name,
            str(self.config.variation),
            self.observation.strip()[:200],
            ",".join(sorted(self.valid_actions[:10])),
            str(self.steps),
            f"score={self.score}",
        ])

    def __str__(self) -> str:
        return self.observation

    # -- Helpers for LLM heuristics ------------------------------------

    def observation_text(self) -> str:
        return self.observation

    def normalized_score(self) -> float:
        if self.max_score > 0:
            return self.score / self.max_score
        return 0.0


# ── Game factory ──────────────────────────────────────────────────────

class ScienceWorldGame(Game):
    """
    ScienceWorld game factory for the MCTS engine.

    Accepts either a task ID ('1-1') or a task name ('boil').
    """

    def __init__(
        self,
        task_name: str = "boil",
        variation: int = 0,
        simplifications: str = "",
        max_steps: int = 100,
        max_actions: int = 0,
    ):
        resolved = resolve_task_name(task_name)
        self.config = ScienceWorldConfig(
            task_name=resolved,
            variation=variation,
            simplifications=simplifications,
            max_steps=max_steps,
            max_actions=max_actions,
        )

    def new_initial_state(self) -> ScienceWorldState:
        return ScienceWorldState(self.config)

    def num_players(self) -> int:
        return 1

    def name(self) -> str:
        return (
            f"ScienceWorld({self.config.task_name}, "
            f"var={self.config.variation})"
        )
