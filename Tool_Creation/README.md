# Tool_Creation — MCTS + LLM Heuristic Optimization

A modular Monte Carlo Tree Search (MCTS) framework with pluggable heuristic
tools and LLM-driven improvement. Built for CSE 291A.

## Project Structure

```
Tool_Creation/
├── run_sokoban.sh          # Quick-start shell script (see below)
├── run_sokoban.py          # Python entry point for Sokoban example
│
├── mcts/                   # Core MCTS framework
│   ├── __init__.py         # Exports: Game, GameState, MCTSNode, MCTSEngine
│   ├── node.py             # MCTSNode — tree node with UCB1 support
│   ├── mcts_engine.py      # MCTSEngine — pluggable 4-phase MCTS loop
│   ├── trace_logger.py     # Passive trace recorder (used by engine)
│   ├── tool_config.json    # Maps each MCTS phase → its tool file
│   ├── records/            # JSON trace files from logged games
│   └── games/
│       ├── game_interface.py   # Abstract Game / GameState contracts
│       ├── sokoban.py          # Sokoban (10 levels, easy → hard)
│       ├── tic_tac_toe.py      # Tic-Tac-Toe (3×3, 2-player)
│       ├── connect_four.py     # Connect Four (6×7, 2-player)
│       └── sliding_puzzle.py   # Sliding puzzle (N×N)
│
├── MCTS_tools/             # Pluggable MCTS phase implementations
│   ├── selection/          # UCB1 tree walk
│   ├── expansion/          # Child node creation
│   ├── simulation/         # Random rollout (default)
│   ├── backpropagation/    # Value backprop with sign-flip for adversarial games
│   ├── hyperparams/        # LLM-tunable MCTS engine parameters
│   │   └── default_hyperparams.py  # get_hyperparams() → dict
│   └── training_logic/     # Game-specific training strategies
│       └── sokoban_training.py     # Levels, mastery criteria, pick_next_level()
│
├── LLM/                    # LLM prompt building & querying pipeline
│   ├── __init__.py
│   ├── prompt_builder.py   # PromptBuilder — 3-step incremental prompts
│   ├── llm_querier.py      # LLMQuerier — async OpenAI-compatible queries
│   ├── optimizer.py        # Optimizer — end-to-end analysis→draft→critique loop
│   ├── tool_manager.py     # ToolManager — parse, validate, install tool code
│   ├── game_infos/         # Game description text files
│   │   └── sokoban.txt
│   ├── drafts/             # Saved prompt text files
│   └── results/            # Saved optimizer run outputs
│
├── orchestrator/           # Step 7–8: game-agnostic optimization orchestrator
│   ├── __init__.py         # Exports: Evaluator, OptimizationRunner
│   ├── evaluator.py        # Evaluator — multi-run eval, composite scoring, mastery
│   ├── runner.py           # OptimizationRunner — multi-phase iterative LLM loop
│   └── test_llm_pipeline.ipynb  # Thin notebook driver using orchestrator module
│
└── tests/                  # Pytest suite (192 tests)
    ├── test_mcts_engine.py     # 63 tests — Sokoban levels, engine, tool swap
    ├── test_tic_tac_toe.py     # 20 tests — state, MCTS quality, all phases
    ├── test_trace_logger.py    # 15 tests — logging lifecycle
    ├── test_prompt_builder.py  # 85 tests — prompt assembly, 3-step prompts, traces
    └── test_llm_querier.py     # 10 tests — async querier, 3-step pipeline
```

## Quick Start

```bash
# Run the default example (Sokoban level1, 200 iterations)
./run_sokoban.sh

# Choose a different level
./run_sokoban.sh --level level3

# More iterations, multiple games, verbose output
./run_sokoban.sh --level level5 --iterations 500 --games 3 --verbose

# Target a different MCTS phase for prompt building
./run_sokoban.sh --phase backpropagation

# Limit moves shown per trace in the prompt
./run_sokoban.sh --games 2 --max-moves 5

# Play without building a prompt
./run_sokoban.sh --no-prompt --verbose
```

All arguments are optional. Run `./run_sokoban.sh --help` for the full list.

## Architecture

### MCTS Engine

The engine runs a standard 4-phase loop:

```
Selection → Expansion → Simulation → Backpropagation
```

Each phase is a **standalone Python function** loaded from `MCTS_tools/<phase>/`.
The active tool for each phase is specified in `mcts/tool_config.json`, and can
be hot-swapped at runtime:

```python
from mcts import MCTSEngine
from mcts.games import Sokoban

engine = MCTSEngine(Sokoban("level1"), iterations=200)
action = engine.search(state)

# Hot-swap a tool
engine.set_tool("simulation", my_better_fn)
engine.load_tool("simulation", "MCTS_tools/simulation/guided_sim.py")
engine.reset_tool("simulation")   # back to default
```

### Trace Logging

When `logging=True`, the engine writes a JSON trace for every game to
`mcts/records/`. Each trace captures metadata, per-move search stats
(root visits, children visit counts and values), and the outcome.

```python
engine = MCTSEngine(game, iterations=200, logging=True)
result = engine.play_game()
print(result["log_file"])   # path to the JSON trace
```

### Prompt Builder

`PromptBuilder` assembles structured prompts for LLM-based heuristic
improvement using a **3-step incremental pipeline**:

1. **Analysis prompt** — game rules + traces + tool source → ask LLM to
   identify weaknesses and propose an approach (incremental or restructure)
2. **Generation prompt** — reference the analysis → ask LLM to write the code
3. **Critique prompt** — review the draft for bugs, speed issues, and reward
   spread → output a finalized version (or UNCHANGED if correct)

The 70/30 prompting strategy guides the LLM: ~70% incremental optimization
(targeted improvements building on existing code), ~30% paradigm shift
(larger restructure when the current approach is fundamentally limited).

```python
from LLM.prompt_builder import PromptBuilder

pb = PromptBuilder(game="sokoban", target_phase="simulation")
# Single-step (legacy)
prompt = pb.build(record_files=["mcts/records/Sokoban_xxx.json"])
# Three-step
analysis_prompt = pb.build_analysis_prompt(record_files=[...], all_tool_sources=tl)
gen_prompt      = pb.build_generation_prompt(analysis=llm_analysis, all_tool_sources=tl)
critique_prompt = pb.build_critique_prompt(analysis=llm_analysis, draft_code=draft)
```

### LLM Pipeline (`Optimizer`)

`Optimizer` is the end-to-end controller that decouples LLM querying from
gameplay. Each iteration:

1. Play a game with the current tool → collect trace
2. Call `Optimizer.run()` → 3-step prompting → parse → validate → smoke test
3. Evaluate the returned function over multiple games
4. Accept/reject based on per-level baseline comparison

```python
from LLM import Optimizer

opt = Optimizer(game="sokoban", target_phase="simulation", three_step=True, verbose=True)
result = opt.run(
    record_files=[trace_path],
    tool_list=engine.get_tool_source(),
    state_factory=lambda: Sokoban("level3", max_steps=200).new_initial_state(),
)
fn = result["function"]   # callable, or None if smoke test failed
```

### Orchestrator

`orchestrator/` encapsulates the full iterative optimization loop into
reusable, game-agnostic Python modules. All configuration lives in
``MCTS_tools/`` — no separate JSON config file:

1. **`MCTS_tools/hyperparams/default_hyperparams.py`** — single source of
   truth for game identity (class, module, constructor kwargs), optimization
   orchestration (num_iters, phases, history_window), and LLM-tunable MCTS
   engine parameters (iterations, max_rollout_depth, exploration_weight).
2. **`MCTS_tools/training_logic/<game>_training.py`** — game-specific
   training strategy: level list, start level, mastery criteria, and
   `pick_next_level()` function.

Key modules:

- **`Evaluator`** — runs MCTS games with a given tool function, computes
  composite scores (weighted solve_rate + avg_returns), tracks per-level
  baselines, handles mastery confirmation, and supports dynamic hyperparameter
  updates via `update_hyperparams()`.
- **`OptimizationRunner`** — multi-phase iterative loop. Supports optimizing
  both MCTS phase tools (e.g. simulation) and hyperparameters in the same run.
  Each iteration randomly selects which component to optimize from the
  configured `phases` list.

```python
from orchestrator import OptimizationRunner

runner = OptimizationRunner.from_config()
summary = runner.run()   # multi-phase iterative LLM optimization loop
print(summary["best_fn"], summary["mastered_levels"])
print(summary["current_hyperparams"])  # final tuned engine params
```

The notebook `orchestrator/test_llm_pipeline.ipynb` is a thin driver that
imports, configures, and runs the orchestrator.

## Running Tests

```bash
# All 192 tests
python -m pytest tests/ -v

# Single file
python -m pytest tests/test_prompt_builder.py -v
python -m pytest tests/test_llm_querier.py -v
```

## Roadmap

| Step | Status | Description |
|------|--------|-------------|
| 1 | ✅ | MCTS node + engine |
| 2 | ✅ | 10 Sokoban levels + pytest |
| 3 | ✅ | Test different heuristics as tools (hot-swap, smoke test, multi-eval) |
| 4 | ✅ | Trace logger (JSON records) |
| 5 | ✅ | Prompt builder (game rules + traces + tool source) |
| 6 | ✅ | LLM querying — 3-step pipeline (analysis → draft → critique), `Optimizer`, per-level baselines, mastery confirmation, 70/30 optimize/rewrite prompting, iterative loop in `test_llm_pipeline.ipynb` |
| 7 | ✅ | Orchestrator encapsulation — `Evaluator`, `OptimizationRunner`, `config.json`, game-agnostic design |
| 8 | ✅ | Hyperparams as LLM tool, game-specific training logic, multi-phase optimization |

## Requirements

- Python 3.12+
- pytest (for tests only)
- No external packages required for the MCTS framework itself
