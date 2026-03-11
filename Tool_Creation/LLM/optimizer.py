"""
MCTS Tool Optimizer — LLM-based heuristic improvement.

The Optimizer is **decoupled from gameplay**.  It does NOT own the MCTS
engine or the game loop.  External logic (e.g. a notebook) is
responsible for:

  1. Playing the game → producing trace files
  2. Collecting tool sources from the engine
  3. Calling ``optimizer.run(record_files, tool_list, state_factory)``
  4. Hot-swapping the returned function into the engine for evaluation

The Optimizer only handles:
  • Building LLM prompts (analysis → code generation)
  • Querying the LLM
  • Parsing / validating / installing the generated code
  • Smoke-testing with an optional repair loop

Usage::

    from LLM.optimizer import Optimizer

    opt = Optimizer(game="sokoban", target_phase="simulation")
    result = opt.run(
        record_files=["traces/game_001.txt"],
        tool_list=engine.get_tool_source(),          # dict[str, str]
        state_factory=lambda: game.new_initial_state(),
    )
    if result["smoke_test"]:
        engine.set_tool("simulation", result["function"])
"""

from __future__ import annotations

import importlib.util
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

NUM_SMOKE_SCENARIOS = 4

# Ensure Tool_Creation is importable
_TOOL_CREATION = Path(__file__).resolve().parent.parent
if str(_TOOL_CREATION) not in sys.path:
    sys.path.insert(0, str(_TOOL_CREATION))

from .prompt_builder import PromptBuilder
from .llm_querier import LLMQuerier
from .tool_manager import ToolManager, EXPECTED_SIGNATURES


class Optimizer:
    """
    LLM-driven tool optimizer — decoupled from the game engine.

    Parameters
    ----------
    game : str
        Game name (must match ``game_infos/<game>.txt``).
    target_phase : str
        MCTS phase to improve (``"simulation"``, ``"selection"``, etc.).
    max_moves_per_trace : int
        How many moves from each trace to include in the prompt.
    two_step : bool
        Use two-step analyse-then-generate flow (default True).
    three_step : bool
        Use three-step analyse → draft → critique+finalize flow.
        Overrides ``two_step`` when True.
    max_repair_attempts : int
        Repair-loop iterations if the smoke test fails.
    verbose : bool
        Print progress messages.
    """

    def __init__(
        self,
        game: str = "sokoban",
        target_phase: str = "simulation",
        max_moves_per_trace: int = 30,
        two_step: bool = True,
        three_step: bool = False,
        max_repair_attempts: int = 5,
        verbose: bool = True,
    ):
        self.game = game
        self.target_phase = target_phase
        self.max_moves_per_trace = max_moves_per_trace
        self.two_step = two_step
        self.three_step = three_step
        self.max_repair_attempts = max_repair_attempts
        self.verbose = verbose

        # Lazily initialized components
        self._builder: PromptBuilder | None = None
        self._querier: LLMQuerier | None = None
        self._manager: ToolManager | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        record_files: list[str],
        tool_list: dict[str, str],
        state_factory: Callable[[], Any] | None = None,
        additional_context: str | None = None,
        session_tag: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the LLM optimisation pipeline.

        Parameters
        ----------
        record_files : list[str]
            Paths to MCTS trace / log files for the LLM to analyse.
        tool_list : dict[str, str]
            Mapping of MCTS phase names to their current source code,
            e.g. ``{"simulation": "def default_simulation(...): ..."}``
        state_factory : callable, optional
            Zero-arg callable returning a fresh initial game state for
            smoke testing.  If *None*, smoke testing is skipped.
        additional_context : str, optional
            Free-form text included in the prompt (e.g. previous
            iteration results, improvement history).
        session_tag : str, optional
            Override the debug session tag for this run (e.g.
            ``"iter2_level5_simulation"``).  Each call with a new tag
            creates a separate debug log folder.

        Returns
        -------
        dict with keys:
            llm_result     — raw LLM query result (includes step1_analysis
                             when two_step=True)
            parsed         — parsed structured response
            installed_path — Path to the installed tool file
            smoke_test     — bool, whether smoke test passed
            function       — the loaded callable (or None)
            error          — str or None
        """
        # Start a fresh debug session for this run
        tag = session_tag or f"{self.game}_{self.target_phase}"
        self.querier.new_session(tag)

        out: dict[str, Any] = {
            "llm_result": None,
            "parsed": None,
            "installed_path": None,
            "smoke_test": False,
            "function": None,
            "error": None,
        }

        try:
            tool_source = tool_list.get(self.target_phase)

            # 1. Build prompts & query LLM
            if self.three_step:
                llm_result = self._query_three_step(
                    record_files, tool_source, tool_list, additional_context
                )
            elif self.two_step:
                self._log("Step 1/4: Querying LLM (step 1 — analysis)…")
                self._log("Step 2/4: Querying LLM (step 2 — code generation)…")
                llm_result = self._query_two_step(
                    record_files, tool_source, tool_list, additional_context
                )
            else:
                self._log("Step 1/4: Building prompt…")
                self._log("Step 2/4: Querying LLM (single-step)…")
                llm_result = self._query_single_step(
                    record_files, tool_source, tool_list
                )
            out["llm_result"] = llm_result

            if llm_result["status"] == "error":
                out["error"] = f"LLM query failed: {llm_result.get('error')}"
                return out

            self._log(
                f"  LLM query: status={llm_result['status']}  "
                f"elapsed={llm_result['elapsed_seconds']}s"
            )
            if (self.two_step or self.three_step) and llm_result.get("step1_analysis"):
                self._log(
                    f"  Step-1 analysis length: "
                    f"{len(llm_result['step1_analysis'])} chars"
                )

            # 2. Parse & validate
            n = 6 if self.three_step else 4
            self._log(f"Step {n-1}/{n}: Parsing & validating response…")
            parsed = llm_result.get("parsed")
            if parsed is None:
                parsed = self.manager.parse_response(llm_result["response"])
            out["parsed"] = parsed

            validation = self.manager.validate(parsed, phase=self.target_phase)
            if not validation["valid"]:
                out["error"] = (
                    f"Validation failed: {validation['errors']}"
                )
                return out
            self._log("  Validation passed ✓")

            # 3. Install
            installed_path = self.manager.install(
                parsed, phase=self.target_phase, overwrite=True
            )
            out["installed_path"] = installed_path
            self._log(f"  Installed to: {installed_path}")

            # 4. Smoke test (with repair loop)
            self._log(f"Step {n}/{n}: Smoke testing…")
            fn, smoke_ok = self._smoke_test_with_repair(
                installed_path, parsed, llm_result, state_factory
            )
            out["smoke_test"] = smoke_ok
            out["function"] = fn
            if not smoke_ok:
                out["error"] = "Smoke test failed after repair attempts."
                return out
            self._log("  Smoke test passed ✓")

        except Exception as e:
            out["error"] = f"Pipeline error: {e}\n{traceback.format_exc()}"
            self._log(f"  ERROR: {e}")

        return out

    # ------------------------------------------------------------------
    # Properties for lazy initialization
    # ------------------------------------------------------------------

    @property
    def builder(self) -> PromptBuilder:
        if self._builder is None:
            self._builder = PromptBuilder(
                game=self.game, target_phase=self.target_phase
            )
        return self._builder

    @property
    def querier(self) -> LLMQuerier:
        if self._querier is None:
            self._querier = LLMQuerier(
                session_tag=f"{self.game}_{self.target_phase}"
            )
        return self._querier

    @property
    def manager(self) -> ToolManager:
        if self._manager is None:
            self._manager = ToolManager()
        return self._manager

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _query_single_step(
        self,
        record_files: list[str],
        tool_source: str | None,
        all_sources: dict[str, str],
    ) -> dict[str, Any]:
        """Legacy single-prompt query."""
        prompt = self.builder.build(
            record_files=record_files or None,
            tool_source=tool_source,
            all_tool_sources=all_sources,
            max_moves_per_trace=self.max_moves_per_trace,
        )
        func_name = self._expected_func_name()
        return self.querier.query(prompt, required_func_name=func_name)

    def _query_two_step(
        self,
        record_files: list[str],
        tool_source: str | None,
        all_sources: dict[str, str],
        additional_context: str | None = None,
    ) -> dict[str, Any]:
        """Two-step analyse-then-generate query."""
        analysis_prompt = self.builder.build_analysis_prompt(
            record_files=record_files or None,
            tool_source=tool_source,
            all_tool_sources=all_sources,
            max_moves_per_trace=self.max_moves_per_trace,
            additional_context=additional_context,
        )

        def gen_prompt_fn(analysis_text: str) -> str:
            return self.builder.build_generation_prompt(
                analysis=analysis_text,
                tool_source=tool_source,
                all_tool_sources=all_sources,
                additional_context=additional_context,
            )

        func_name = self._expected_func_name()
        return self.querier.query_two_step(
            analysis_prompt=analysis_prompt,
            generation_prompt_fn=gen_prompt_fn,
            required_func_name=func_name,
        )

    def _query_three_step(
        self,
        record_files: list[str],
        tool_source: str | None,
        all_sources: dict[str, str],
        additional_context: str | None = None,
    ) -> dict[str, Any]:
        """Three-step analyse → draft → critique+finalize query."""
        self._log("Step 1/6: Building analysis prompt…")
        analysis_prompt = self.builder.build_analysis_prompt(
            record_files=record_files or None,
            tool_source=tool_source,
            all_tool_sources=all_sources,
            max_moves_per_trace=self.max_moves_per_trace,
            additional_context=additional_context,
        )

        def gen_prompt_fn(analysis_text: str) -> str:
            self._log("Step 2/6: Querying LLM (step 2 — draft code)…")
            return self.builder.build_generation_prompt(
                analysis=analysis_text,
                tool_source=tool_source,
                all_tool_sources=all_sources,
                additional_context=additional_context,
            )

        def critique_prompt_fn(analysis_text: str, draft_code: str) -> str:
            self._log("Step 3/6: Querying LLM (step 3 — critique & finalize)…")
            return self.builder.build_critique_prompt(
                analysis=analysis_text,
                draft_code=draft_code,
                tool_source=tool_source,
                # Skip all_tool_sources to keep critique prompt compact.
                # The draft already incorporated that context.
            )

        func_name = self._expected_func_name()
        return self.querier.query_three_step(
            analysis_prompt=analysis_prompt,
            generation_prompt_fn=gen_prompt_fn,
            critique_prompt_fn=critique_prompt_fn,
            required_func_name=func_name,
        )

    def _expected_func_name(self) -> str:
        """Derive the expected function name from ``target_phase``."""
        if self.target_phase == "hyperparams":
            return "get_hyperparams"
        return f"default_{self.target_phase}"

    def _load_function(self, installed_path: Path, func_name: str):
        """Dynamically import the function from the installed file."""
        spec = importlib.util.spec_from_file_location(
            installed_path.stem, str(installed_path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, func_name)

    def _build_smoke_test_scenarios(
        self,
        state_factory: Callable[[], Any],
        num_scenarios: int = NUM_SMOKE_SCENARIOS,
    ) -> list[tuple[list, str]]:
        """
        Build (args_list, label) for each scenario. Uses varied states:
        s0=initial, s1=after 1 move, s2=after 2 moves, etc.
        """
        rng = random.Random(42)
        scenarios: list[tuple[list, str]] = []
        sig_params = EXPECTED_SIGNATURES.get(self.target_phase, [])

        if not sig_params:
            # hyperparams: no args, single scenario
            return [([], "get_hyperparams")]

        states: list[Any] = []
        s0 = state_factory()
        states.append(s0)

        for i in range(1, num_scenarios):
            prev = states[-1]
            if prev.is_terminal():
                break
            actions = prev.legal_actions()
            if not actions:
                break
            s = prev.clone()
            s.apply_action(rng.choice(actions))
            states.append(s)

        rewards = [0.5, 0.0, 1.0, 0.3]
        for i, st in enumerate(states):
            label = "initial" if i == 0 else f"after_{i}_moves"
            args = self._build_smoke_args(sig_params, st, reward=rewards[i % len(rewards)])
            scenarios.append((args, label))

        return scenarios

    def _smoke_test(
        self,
        installed_path: Path,
        func_name: str,
        state_factory: Callable[[], Any] | None,
    ):
        """
        Load and call the function on multiple test scenarios.

        Returns (fn, passed: bool, error: str | None, tb: str | None).
        """
        try:
            fn = self._load_function(installed_path, func_name)
        except Exception as e:
            return None, False, str(e), traceback.format_exc()

        if state_factory is None:
            return fn, True, None, None

        scenarios = self._build_smoke_test_scenarios(state_factory)
        for args, label in scenarios:
            try:
                result = fn(*args)
                if self.target_phase == "simulation":
                    assert isinstance(result, (int, float)), (
                        f"Expected numeric, got {type(result)}"
                    )
                if self.target_phase == "hyperparams":
                    assert isinstance(result, dict), (
                        f"Expected dict from get_hyperparams, got {type(result)}"
                    )
            except Exception as e:
                err_msg = f"[scenario '{label}'] {e}"
                return None, False, err_msg, traceback.format_exc()

        return fn, True, None, None

    @staticmethod
    def _build_smoke_args(
        sig_params: list[str], test_state, reward: float = 0.5
    ) -> list:
        """Create plausible test arguments from parameter names."""
        args: list = []
        for p in sig_params:
            if p in ("state",):
                args.append(test_state)
            elif p in ("perspective_player", "player"):
                args.append(0)
            elif p in ("max_depth", "depth"):
                args.append(50)
            elif p in ("root", "node"):
                try:
                    from mcts.node import MCTSNode
                    args.append(MCTSNode(test_state))
                except Exception:
                    args.append(test_state)
            elif p in ("exploration_weight",):
                args.append(1.41)
            elif p in ("reward",):
                args.append(reward)
            else:
                args.append(None)
        return args

    def _smoke_test_with_repair(
        self,
        installed_path: Path,
        parsed: dict[str, Any],
        llm_result: dict[str, Any],
        state_factory: Callable[[], Any] | None,
    ) -> tuple[Any, bool]:
        """
        Smoke test with an optional LLM repair loop.

        Returns (fn, passed).
        """
        func_name = parsed["function_name"]
        fn, passed, error, tb = self._smoke_test(
            installed_path, func_name, state_factory
        )
        if passed:
            return fn, True

        # Attempt repair
        for attempt in range(self.max_repair_attempts):
            self._log(
                f"  Smoke test failed: {error}. "
                f"Attempting repair ({attempt + 1}/{self.max_repair_attempts})…"
            )
            repair_result = self._repair(parsed, func_name, tb, state_factory, attempt=attempt)
            if repair_result is None:
                continue

            repair_parsed = repair_result["parsed"]
            if repair_parsed is None:
                repair_parsed = self.manager.parse_response(
                    repair_result["response"]
                )
            # Fill missing headers from original
            if repair_parsed["code"] and not repair_parsed.get("action"):
                repair_parsed["action"] = parsed.get("action", "modify")
            if repair_parsed["code"] and not repair_parsed.get("file_name"):
                repair_parsed["file_name"] = parsed.get(
                    "file_name", f"{self.target_phase}.py"
                )
            if repair_parsed["code"] and not repair_parsed.get("function_name"):
                repair_parsed["function_name"] = func_name
            # Strip spurious parse errors about missing optional headers
            repair_parsed["parse_errors"] = [
                e for e in repair_parsed.get("parse_errors", [])
                if "Missing ACTION" not in e
                and "Missing FILE_NAME" not in e
                and "Missing FUNCTION_NAME" not in e
            ]

            validation = self.manager.validate(
                repair_parsed, phase=self.target_phase
            )
            if not validation["valid"]:
                self._log(f"  Repair validation failed: {validation['errors']}")
                continue

            installed_path = self.manager.install(
                repair_parsed, phase=self.target_phase, overwrite=True
            )
            fn, passed, error, tb = self._smoke_test(
                installed_path, func_name, state_factory
            )
            if passed:
                # Update outer parsed so the caller sees the fixed version
                parsed.update(repair_parsed)
                return fn, True

        return None, False

    def _repair(
        self,
        parsed: dict[str, Any],
        func_name: str,
        tb_text: str | None,
        state_factory: Callable[[], Any] | None,
        attempt: int = 0,
    ) -> dict[str, Any] | None:
        """Send a targeted repair prompt to the LLM."""
        broken_code = parsed.get("code", "")

        # Build state-API reference if a state factory is available
        attr_lines = "(state API unavailable)"
        if state_factory is not None:
            test_state = state_factory()
            state_attrs = [
                (a, type(getattr(test_state, a)).__name__)
                for a in dir(test_state) if not a.startswith("_")
            ]
            attr_lines = "\n".join(f"  {n}: {t}" for n, t in state_attrs)

        # For node-based phases: MCTSNode uses __slots__, no arbitrary attributes
        node_api_section = ""
        if self.target_phase in ("selection", "expansion", "backpropagation"):
            node_api_section = (
                "\n== MCTSNode API (node parameter) ==\n"
                "MCTSNode uses __slots__. You CANNOT add new attributes.\n"
                "Use ONLY these: node.state, node.parent, node.parent_action,\n"
                "node.children, node._untried_actions, node.visits, node.value.\n"
                "  - node.state = GameState (use for legal_actions, clone, etc.)\n"
                "  - node._untried_actions = list of actions to expand\n"
                "  - node.children = dict[action, child_node]\n"
                "  - node.visits, node.value = backpropagated stats\n"
                "Do NOT assign node._mcts_root_key or any attribute not listed above.\n"
            )

        constraints = (
            "CRITICAL: The function must be SELF-CONTAINED. Define ALL helper "
            "functions inline. Do NOT use undefined names (e.g. _ag, _bfs_distance). "
            "Fix ONLY the broken parts. Keep the heuristic strategy the same.\n"
        )

        repair_prompt = (
            f"You previously generated the following {self.game} MCTS "
            f"{self.target_phase} function, but it raised a runtime error.\n\n"
            f"== BROKEN CODE ==\n```python\n{broken_code}\n```\n\n"
            f"== RUNTIME ERROR ==\n{tb_text}\n\n"
            f"== ACTUAL GameState PUBLIC API (for node.state or state param) ==\n"
            f"{attr_lines}\n"
            f"{node_api_section}\n"
            f"{constraints}"
            f"Return using the SAME structured format.\n\n"
            f"ACTION: modify\n"
            f"FILE_NAME: {parsed.get('file_name', self.target_phase + '.py')}\n"
            f"FUNCTION_NAME: {func_name}\n"
            f"DESCRIPTION: <one-line description of what you fixed>\n"
            f"```python\n<complete corrected function here>\n```"
        )
        result = self.querier.query(
            repair_prompt,
            required_func_name=func_name,
            step_name=f"repair_{attempt + 1}",
        )
        if result["status"] == "error":
            return None
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
