"""
Tests for LLM.prompt_builder.PromptBuilder.

Covers:
  - Game info loading (sokoban.txt)
  - Trace record loading & formatting
  - Prompt assembly (all sections present)
  - Tool source injection
  - max_moves_per_trace truncation
  - save() to file
  - Error paths (missing game info, missing record, bad phase)
  - Multiple traces
  - Edge cases (empty trace, no moves)
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest

# ── Fixtures ─────────────────────────────────────────────────────────

# Locate the real sokoban game info
_LLM_DIR = Path(__file__).resolve().parent.parent / "LLM"
_GAME_INFOS_DIR = _LLM_DIR / "game_infos"
_RECORDS_DIR = Path(__file__).resolve().parent.parent / "mcts" / "records"

# We import after path setup so Python can find the LLM package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from LLM.prompt_builder import PromptBuilder


# ── Helpers ──────────────────────────────────────────────────────────

def _make_trace(
    game: str = "Sokoban (level1)",
    solved: bool = True,
    num_moves: int = 3,
    records_dir: Path | None = None,
) -> Path:
    """Write a minimal but valid trace JSON and return its path."""
    if records_dir is None:
        raise ValueError("records_dir required")
    records_dir.mkdir(parents=True, exist_ok=True)
    trace = {
        "metadata": {
            "game": game,
            "timestamp": "2026-03-05T10:00:00.000000",
            "iterations": 100,
            "max_rollout_depth": 50,
            "exploration_weight": 1.41,
            "tools": {
                "selection": "MCTS_tools/selection/default_selection.py",
                "expansion": "MCTS_tools/expansion/default_expansion.py",
                "simulation": "MCTS_tools/simulation/default_simulation.py",
                "backpropagation": "MCTS_tools/backpropagation/default_backpropagation.py",
            },
        },
        "moves": [
            {
                "move_number": i + 1,
                "player": 0,
                "state_before": f"Step {i}/200 | Boxes on target: 0/1 | Total distance: 1\n#####\n# @$#\n#  .#\n#####",
                "state_key": f"P(1,1)B((1,2),)_step{i}",
                "legal_actions": ["0", "1", "2", "3"],
                "action_chosen": str(i % 4),
                "root_visits": 100,
                "children_stats": {
                    "0": {"visits": 50, "value": 25.0, "avg_value": 0.5},
                    "1": {"visits": 20, "value": 5.0, "avg_value": 0.25},
                    "2": {"visits": 15, "value": 3.0, "avg_value": 0.2},
                    "3": {"visits": 15, "value": 2.0, "avg_value": 0.1333},
                },
            }
            for i in range(num_moves)
        ],
        "outcome": {
            "solved": solved,
            "steps": num_moves,
            "returns": [1.0] if solved else [0.0],
            "final_state": "#####\n# @*#\n#  .#\n#####" if solved else "#####\n#$@ #\n#  .#\n#####",
        },
    }
    filename = f"{game}_test.json"
    filepath = records_dir / filename
    with open(filepath, "w") as f:
        json.dump(trace, f, indent=2)
    return filepath


# =====================================================================
# Test classes
# =====================================================================


class TestPromptBuilderInit:
    """Constructor validation."""

    def test_valid_init(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        assert pb.game == "sokoban"
        assert pb.target_phase == "simulation"

    def test_game_name_normalized_to_lower(self):
        pb = PromptBuilder(game="Sokoban", target_phase="selection")
        assert pb.game == "sokoban"

    def test_invalid_phase_raises(self):
        with pytest.raises(ValueError, match="Invalid target_phase"):
            PromptBuilder(game="sokoban", target_phase="magic")

    def test_all_valid_phases(self):
        for phase in ("selection", "expansion", "simulation", "backpropagation"):
            pb = PromptBuilder(game="sokoban", target_phase=phase)
            assert pb.target_phase == phase

    def test_custom_dirs(self, tmp_path):
        gi_dir = tmp_path / "gi"
        gi_dir.mkdir()
        (gi_dir / "test.txt").write_text("rules")
        rec_dir = tmp_path / "rec"
        rec_dir.mkdir()
        pb = PromptBuilder(
            game="test",
            target_phase="simulation",
            game_infos_dir=gi_dir,
            records_dir=rec_dir,
        )
        assert pb.game_infos_dir == gi_dir
        assert pb.records_dir == rec_dir


class TestGameInfoLoading:
    """Loading game description files."""

    def test_loads_sokoban_info(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build()
        assert "Sokoban" in prompt
        assert "push" in prompt.lower()

    def test_game_rules_section_present(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build()
        assert "GAME RULES" in prompt

    def test_missing_game_info_raises(self, tmp_path):
        gi_dir = tmp_path / "empty_infos"
        gi_dir.mkdir()
        pb = PromptBuilder(
            game="nonexistent",
            target_phase="simulation",
            game_infos_dir=gi_dir,
        )
        with pytest.raises(FileNotFoundError, match="Game info file not found"):
            pb.build()

    def test_custom_game_info(self, tmp_path):
        gi_dir = tmp_path / "infos"
        gi_dir.mkdir()
        (gi_dir / "mygame.txt").write_text("MyGame is a great game.\nRules here.")
        pb = PromptBuilder(
            game="mygame",
            target_phase="simulation",
            game_infos_dir=gi_dir,
        )
        prompt = pb.build()
        assert "MyGame is a great game" in prompt
        assert "Rules here" in prompt


class TestTraceLoading:
    """Loading and formatting trace records."""

    def test_build_with_no_traces(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build(record_files=None)
        assert "No gameplay traces provided" in prompt

    def test_build_with_empty_trace_list(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build(record_files=[])
        assert "No gameplay traces provided" in prompt

    def test_load_trace_absolute_path(self, tmp_path):
        rec_dir = tmp_path / "records"
        trace_path = _make_trace(records_dir=rec_dir)
        pb = PromptBuilder(
            game="sokoban",
            target_phase="simulation",
            records_dir=rec_dir,
        )
        prompt = pb.build(record_files=[trace_path])
        assert "Trace #1" in prompt

    def test_load_trace_relative_to_records_dir(self, tmp_path):
        rec_dir = tmp_path / "records"
        trace_path = _make_trace(records_dir=rec_dir)
        pb = PromptBuilder(
            game="sokoban",
            target_phase="simulation",
            records_dir=rec_dir,
        )
        # Use just the filename (relative)
        prompt = pb.build(record_files=[trace_path.name])
        assert "Trace #1" in prompt

    def test_missing_record_raises(self, tmp_path):
        pb = PromptBuilder(
            game="sokoban",
            target_phase="simulation",
            records_dir=tmp_path,
        )
        with pytest.raises(FileNotFoundError, match="Record file not found"):
            pb.build(record_files=["totally_fake.json"])

    def test_multiple_traces(self, tmp_path):
        rec_dir = tmp_path / "records"
        t1 = _make_trace(game="Sokoban (level1)", solved=True, records_dir=rec_dir)
        t2 = _make_trace(game="Sokoban (level2)", solved=False, records_dir=rec_dir)
        pb = PromptBuilder(
            game="sokoban",
            target_phase="simulation",
            records_dir=rec_dir,
        )
        prompt = pb.build(record_files=[t1, t2])
        assert "Trace #1" in prompt
        assert "Trace #2" in prompt

    def test_real_sokoban_record(self):
        """Use a real trace from mcts/records/ if available."""
        record_files = list(_RECORDS_DIR.glob("Sokoban*"))
        if not record_files:
            pytest.skip("No Sokoban records found in mcts/records/")
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build(record_files=[record_files[0]])
        assert "GAMEPLAY TRACES" in prompt
        assert "Trace #1" in prompt


class TestTraceFormatting:
    """Trace content appears correctly in prompt."""

    def test_trace_contains_metadata(self, tmp_path):
        rec_dir = tmp_path / "records"
        trace_path = _make_trace(records_dir=rec_dir)
        pb = PromptBuilder(game="sokoban", target_phase="simulation", records_dir=rec_dir)
        prompt = pb.build(record_files=[trace_path])
        assert "Sokoban (level1)" in prompt
        assert "Iterations: 100" in prompt

    def test_trace_contains_outcome(self, tmp_path):
        rec_dir = tmp_path / "records"
        trace_path = _make_trace(solved=False, records_dir=rec_dir)
        pb = PromptBuilder(game="sokoban", target_phase="simulation", records_dir=rec_dir)
        prompt = pb.build(record_files=[trace_path])
        assert "Solved:     False" in prompt

    def test_trace_contains_moves(self, tmp_path):
        rec_dir = tmp_path / "records"
        trace_path = _make_trace(num_moves=5, records_dir=rec_dir)
        pb = PromptBuilder(game="sokoban", target_phase="simulation", records_dir=rec_dir)
        prompt = pb.build(record_files=[trace_path])
        assert "Move 1:" in prompt
        assert "Move 5:" in prompt

    def test_children_stats_in_moves(self, tmp_path):
        rec_dir = tmp_path / "records"
        trace_path = _make_trace(num_moves=1, records_dir=rec_dir)
        pb = PromptBuilder(game="sokoban", target_phase="simulation", records_dir=rec_dir)
        prompt = pb.build(record_files=[trace_path])
        # Should show visit counts and avg values
        assert "v=50" in prompt
        assert "0.500" in prompt

    def test_final_state_shown(self, tmp_path):
        rec_dir = tmp_path / "records"
        trace_path = _make_trace(solved=True, records_dir=rec_dir)
        pb = PromptBuilder(game="sokoban", target_phase="simulation", records_dir=rec_dir)
        prompt = pb.build(record_files=[trace_path])
        assert "Final state:" in prompt

    def test_max_moves_truncation(self, tmp_path):
        rec_dir = tmp_path / "records"
        trace_path = _make_trace(num_moves=10, records_dir=rec_dir)
        pb = PromptBuilder(game="sokoban", target_phase="simulation", records_dir=rec_dir)
        prompt = pb.build(record_files=[trace_path], max_moves_per_trace=3)
        assert "Move 1:" in prompt
        assert "Move 3:" in prompt
        assert "Move 4:" not in prompt
        assert "7 more moves omitted" in prompt

    def test_max_moves_none_shows_all(self, tmp_path):
        rec_dir = tmp_path / "records"
        trace_path = _make_trace(num_moves=5, records_dir=rec_dir)
        pb = PromptBuilder(game="sokoban", target_phase="simulation", records_dir=rec_dir)
        prompt = pb.build(record_files=[trace_path], max_moves_per_trace=None)
        assert "Move 5:" in prompt
        assert "omitted" not in prompt


class TestToolSource:
    """Tool source code injection."""

    def test_no_tool_source_placeholder(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build(tool_source=None)
        assert "TARGET HEURISTIC TO IMPROVE" in prompt
        assert "No heuristic code provided" in prompt

    def test_tool_source_included(self):
        code = textwrap.dedent("""\
            def my_simulation(state, player, max_depth):
                return 0.5
        """)
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build(tool_source=code)
        assert "my_simulation" in prompt
        assert "```python" in prompt

    def test_tool_source_section_label_matches_phase(self):
        pb = PromptBuilder(game="sokoban", target_phase="backpropagation")
        prompt = pb.build(tool_source="def bp(node, r): pass")
        assert "TARGET HEURISTIC TO IMPROVE (backpropagation)" in prompt


class TestPromptStructure:
    """Overall prompt structure and section ordering."""

    def test_all_five_sections_present(self, tmp_path):
        rec_dir = tmp_path / "records"
        trace_path = _make_trace(records_dir=rec_dir)
        pb = PromptBuilder(game="sokoban", target_phase="simulation", records_dir=rec_dir)
        prompt = pb.build(
            record_files=[trace_path],
            tool_source="def sim(s, p, d): return 0.0",
        )
        assert "SYSTEM: MCTS Heuristic Improvement" in prompt
        assert "GAME RULES" in prompt
        assert "TARGET HEURISTIC TO IMPROVE" in prompt
        assert "GAMEPLAY TRACES" in prompt
        assert "TASK" in prompt

    def test_sections_in_order(self, tmp_path):
        rec_dir = tmp_path / "records"
        trace_path = _make_trace(records_dir=rec_dir)
        pb = PromptBuilder(game="sokoban", target_phase="simulation", records_dir=rec_dir)
        prompt = pb.build(
            record_files=[trace_path],
            tool_source="def sim(s, p, d): return 0.0",
        )
        idx_system = prompt.index("SYSTEM")
        idx_rules = prompt.index("GAME RULES")
        idx_code = prompt.index("TARGET HEURISTIC TO IMPROVE")
        idx_traces = prompt.index("GAMEPLAY TRACES")
        idx_task = prompt.index("TASK")
        assert idx_system < idx_rules < idx_code < idx_traces < idx_task

    def test_task_section_contains_instructions(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build()
        assert "Same function signature" in prompt or "SAME function signature" in prompt or "function signature" in prompt
        assert "standalone" in prompt or "Standalone" in prompt
        assert "```python" in prompt

    def test_phase_mentioned_in_task(self):
        pb = PromptBuilder(game="sokoban", target_phase="expansion")
        prompt = pb.build()
        assert "'expansion'" in prompt


class TestSave:
    """Saving prompts to disk."""

    def test_save_default_path(self, tmp_path, monkeypatch):
        # Monkeypatch the default drafts dir to tmp
        import LLM.prompt_builder as pb_mod
        monkeypatch.setattr(pb_mod, "_DRAFTS_DIR", tmp_path / "drafts")
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build()
        path = pb.save(prompt)
        assert path.exists()
        assert path.name == "sokoban_simulation_prompt.txt"
        content = path.read_text()
        assert "SYSTEM" in content

    def test_save_custom_path(self, tmp_path):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build()
        out = tmp_path / "custom" / "my_prompt.txt"
        path = pb.save(prompt, filepath=out)
        assert path == out
        assert path.exists()
        assert path.read_text() == prompt

    def test_save_creates_directory(self, tmp_path):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build()
        out = tmp_path / "deep" / "nested" / "dir" / "prompt.txt"
        path = pb.save(prompt, filepath=out)
        assert path.exists()

    def test_saved_file_matches_prompt(self, tmp_path):
        rec_dir = tmp_path / "records"
        trace_path = _make_trace(records_dir=rec_dir)
        pb = PromptBuilder(game="sokoban", target_phase="simulation", records_dir=rec_dir)
        prompt = pb.build(
            record_files=[trace_path],
            tool_source="def sim(s, p, d): return 0.0",
        )
        path = pb.save(prompt, filepath=tmp_path / "out.txt")
        assert path.read_text(encoding="utf-8") == prompt


class TestEdgeCases:
    """Edge case handling."""

    def test_trace_with_zero_moves(self, tmp_path):
        rec_dir = tmp_path / "records"
        trace_path = _make_trace(num_moves=0, records_dir=rec_dir)
        pb = PromptBuilder(game="sokoban", target_phase="simulation", records_dir=rec_dir)
        prompt = pb.build(record_files=[trace_path])
        assert "Trace #1" in prompt
        # No "Move" lines since 0 moves
        assert "Move 1:" not in prompt

    def test_max_moves_larger_than_actual(self, tmp_path):
        rec_dir = tmp_path / "records"
        trace_path = _make_trace(num_moves=2, records_dir=rec_dir)
        pb = PromptBuilder(game="sokoban", target_phase="simulation", records_dir=rec_dir)
        prompt = pb.build(record_files=[trace_path], max_moves_per_trace=100)
        assert "Move 1:" in prompt
        assert "Move 2:" in prompt
        assert "omitted" not in prompt

    def test_prompt_is_nonempty_string(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_build_is_deterministic(self, tmp_path):
        rec_dir = tmp_path / "records"
        trace_path = _make_trace(records_dir=rec_dir)
        pb = PromptBuilder(game="sokoban", target_phase="simulation", records_dir=rec_dir)
        p1 = pb.build(record_files=[trace_path], tool_source="def x(): pass")
        p2 = pb.build(record_files=[trace_path], tool_source="def x(): pass")
        assert p1 == p2


class TestIntegrationWithEngine:
    """End-to-end: engine trace -> prompt builder."""

    def test_engine_trace_to_prompt(self, tmp_path):
        """Play a game with logging, then build a prompt from its trace."""
        from mcts import MCTSEngine
        from mcts.games import Sokoban

        rec_dir = tmp_path / "records"
        engine = MCTSEngine(
            Sokoban("level1"),
            iterations=50,
            logging=True,
            records_dir=rec_dir,
        )
        result = engine.play_game()
        log_file = result["log_file"]
        assert log_file is not None

        pb = PromptBuilder(
            game="sokoban",
            target_phase="simulation",
            records_dir=rec_dir,
        )
        # Use get_tool_source from the engine
        sources = engine.get_tool_source()
        prompt = pb.build(
            record_files=[log_file],
            tool_source=sources["simulation"],
        )
        assert "GAME RULES" in prompt
        assert "default_simulation" in prompt
        assert "Trace #1" in prompt
        assert "Sokoban" in prompt

    def test_engine_play_many_traces(self, tmp_path):
        """Play multiple games, feed all traces to prompt builder."""
        from mcts import MCTSEngine
        from mcts.games import Sokoban

        rec_dir = tmp_path / "records"
        engine = MCTSEngine(
            Sokoban("level1"),
            iterations=50,
            logging=True,
            records_dir=rec_dir,
        )
        stats = engine.play_many(num_games=3)
        log_files = [r["log_file"] for r in stats["results"] if r.get("log_file")]
        assert len(log_files) == 3

        pb = PromptBuilder(
            game="sokoban",
            target_phase="simulation",
            records_dir=rec_dir,
        )
        prompt = pb.build(record_files=log_files)
        assert "Trace #1" in prompt
        assert "Trace #2" in prompt
        assert "Trace #3" in prompt


class TestAllToolSources:
    """Tests for the all_tool_sources parameter."""

    _SAMPLE_SOURCES = {
        "selection": "def select(node, w): return node",
        "expansion": "def expand(node): return node",
        "simulation": "def simulate(s, p, d): return 0.5",
        "backpropagation": "def backprop(n, r): pass",
    }

    def test_all_tools_section_present(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build(all_tool_sources=self._SAMPLE_SOURCES)
        assert "MCTS TOOL FUNCTIONS (all 4 phases)" in prompt

    def test_all_four_phases_shown(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build(all_tool_sources=self._SAMPLE_SOURCES)
        assert "--- selection ---" in prompt
        assert "--- expansion ---" in prompt
        assert "--- backpropagation ---" in prompt
        # simulation is the target, so it gets a marker
        assert "--- simulation" in prompt

    def test_target_phase_marked(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build(all_tool_sources=self._SAMPLE_SOURCES)
        assert "simulation \u25c0 TARGET" in prompt

    def test_target_marker_changes_with_phase(self):
        pb = PromptBuilder(game="sokoban", target_phase="expansion")
        prompt = pb.build(all_tool_sources=self._SAMPLE_SOURCES)
        assert "expansion \u25c0 TARGET" in prompt
        # Other phases should NOT have the marker
        assert "simulation \u25c0 TARGET" not in prompt

    def test_source_code_included(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build(all_tool_sources=self._SAMPLE_SOURCES)
        assert "def select(node, w):" in prompt
        assert "def expand(node):" in prompt
        assert "def simulate(s, p, d):" in prompt
        assert "def backprop(n, r):" in prompt

    def test_no_all_tools_section_when_omitted(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build(all_tool_sources=None)
        assert "MCTS TOOL FUNCTIONS" not in prompt

    def test_all_tools_before_target_heuristic(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build(
            all_tool_sources=self._SAMPLE_SOURCES,
            tool_source="def sim(s, p, d): return 0.0",
        )
        idx_all = prompt.index("MCTS TOOL FUNCTIONS")
        idx_target = prompt.index("TARGET HEURISTIC TO IMPROVE")
        assert idx_all < idx_target

    def test_integration_with_engine_sources(self, tmp_path):
        """all_tool_sources from engine.get_tool_source()."""
        from mcts import MCTSEngine
        from mcts.games import Sokoban

        rec_dir = tmp_path / "records"
        engine = MCTSEngine(
            Sokoban("level1"), iterations=50, logging=True, records_dir=rec_dir,
        )
        result = engine.play_game()

        sources = engine.get_tool_source()
        pb = PromptBuilder(game="sokoban", target_phase="simulation", records_dir=rec_dir)
        prompt = pb.build(
            record_files=[result["log_file"]],
            tool_source=sources["simulation"],
            all_tool_sources=sources,
        )
        assert "MCTS TOOL FUNCTIONS" in prompt
        assert "default_selection" in prompt
        assert "default_expansion" in prompt
        assert "default_simulation" in prompt
        assert "default_backpropagation" in prompt


# =====================================================================
# Critique prompt (3-step pipeline)
# =====================================================================


class TestCritiquePrompt:
    """Tests for build_critique_prompt() – used in the 3-step pipeline."""

    _ANALYSIS = (
        "The current heuristic only considers box-target distance. "
        "It ignores deadlocks and player position."
    )
    _DRAFT_CODE = textwrap.dedent("""\
        def custom_simulation(state, player_id, depth=50):
            s = state.clone()
            for _ in range(depth):
                if s.is_terminal():
                    break
                actions = s.legal_actions()
                s.apply_action(actions[0])
            return s.returns()
    """)

    def test_contains_system_section(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build_critique_prompt(
            analysis=self._ANALYSIS, draft_code=self._DRAFT_CODE,
        )
        assert "SYSTEM:" in prompt
        assert "sokoban" in prompt

    def test_contains_game_rules(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build_critique_prompt(
            analysis=self._ANALYSIS, draft_code=self._DRAFT_CODE,
        )
        assert "GAME RULES" in prompt

    def test_contains_analysis_reference(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build_critique_prompt(
            analysis=self._ANALYSIS, draft_code=self._DRAFT_CODE,
        )
        assert "PRIOR ANALYSIS" in prompt
        assert "box-target distance" in prompt  # from our analysis text

    def test_contains_draft_code_section(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build_critique_prompt(
            analysis=self._ANALYSIS, draft_code=self._DRAFT_CODE,
        )
        assert "DRAFT CODE" in prompt
        assert "custom_simulation" in prompt

    def test_contains_critique_task(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build_critique_prompt(
            analysis=self._ANALYSIS, draft_code=self._DRAFT_CODE,
        )
        assert "CRITIQUE & FINALIZE" in prompt
        assert "BUGS" in prompt
        assert "SPEED" in prompt
        assert "REWARD SPREAD" in prompt
        # Should instruct minimal / no extra features
        assert "NOT add extra features" in prompt or "Do NOT add" in prompt

    def test_includes_all_tool_sources(self):
        sources = {
            "selection": "def sel(n): pass",
            "expansion": "def exp(n): pass",
            "simulation": "def sim(s, p, d): pass",
            "backpropagation": "def bp(n, r): pass",
        }
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build_critique_prompt(
            analysis=self._ANALYSIS,
            draft_code=self._DRAFT_CODE,
            all_tool_sources=sources,
        )
        assert "MCTS TOOL FUNCTIONS" in prompt
        assert "def sel(n): pass" in prompt

    def test_no_all_tools_when_omitted(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build_critique_prompt(
            analysis=self._ANALYSIS,
            draft_code=self._DRAFT_CODE,
            all_tool_sources=None,
        )
        assert "MCTS TOOL FUNCTIONS" not in prompt

    def test_includes_additional_context(self):
        ctx = "Iteration 3: solve_rate=0.33, avg_returns=0.72"
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build_critique_prompt(
            analysis=self._ANALYSIS,
            draft_code=self._DRAFT_CODE,
            additional_context=ctx,
        )
        assert "ADDITIONAL CONTEXT" in prompt
        assert "solve_rate=0.33" in prompt

    def test_no_additional_context_when_omitted(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build_critique_prompt(
            analysis=self._ANALYSIS,
            draft_code=self._DRAFT_CODE,
            additional_context=None,
        )
        assert "ADDITIONAL CONTEXT" not in prompt

    def test_section_ordering(self):
        """Critique prompt sections should appear in expected order."""
        sources = {
            "selection": "def sel(n): pass",
            "expansion": "def exp(n): pass",
            "simulation": "def sim(s, p, d): pass",
            "backpropagation": "def bp(n, r): pass",
        }
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build_critique_prompt(
            analysis=self._ANALYSIS,
            draft_code=self._DRAFT_CODE,
            all_tool_sources=sources,
            additional_context="iter history",
        )
        idx_system = prompt.index("SYSTEM:")
        idx_rules = prompt.index("GAME RULES")
        idx_tools = prompt.index("MCTS TOOL FUNCTIONS")
        idx_ctx = prompt.index("ADDITIONAL CONTEXT")
        idx_analysis = prompt.index("PRIOR ANALYSIS")
        idx_draft = prompt.index("DRAFT CODE")
        idx_task = prompt.index("CRITIQUE & FINALIZE")
        assert idx_system < idx_rules < idx_tools < idx_ctx < idx_analysis < idx_draft < idx_task

    def test_includes_tool_source_section(self):
        pb = PromptBuilder(game="sokoban", target_phase="simulation")
        prompt = pb.build_critique_prompt(
            analysis=self._ANALYSIS,
            draft_code=self._DRAFT_CODE,
            tool_source="def baseline_sim(s, p, d): return 0.5",
        )
        assert "TARGET HEURISTIC TO IMPROVE" in prompt
        assert "baseline_sim" in prompt
        assert "TARGET HEURISTIC TO IMPROVE" in prompt
