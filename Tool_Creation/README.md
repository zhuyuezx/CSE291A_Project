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
│   └── backpropagation/    # Value backprop with sign-flip for adversarial games
│
├── LLM/                    # LLM prompt building
│   ├── __init__.py
│   ├── prompt_builder.py   # PromptBuilder — assembles 5-section prompts
│   ├── game_infos/         # Game description text files
│   │   └── sokoban.txt
│   └── drafts/             # Saved prompt text files
│
└── tests/                  # Pytest suite
    ├── test_mcts_engine.py     # 63 tests — Sokoban levels, engine, tool swap
    ├── test_tic_tac_toe.py     # 20 tests — state, MCTS quality, all phases
    ├── test_trace_logger.py    # 15 tests — logging lifecycle
    └── test_prompt_builder.py  # 40 tests — prompt assembly, traces, save
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
improvement. It combines five sections:

1. **System instruction** — role and objective
2. **Game rules** — loaded from `LLM/game_infos/<game>.txt`
3. **Current heuristic code** — the tool source being improved
4. **Gameplay traces** — from `mcts/records/*.json`
5. **Task instruction** — output format and constraints

```python
from LLM.prompt_builder import PromptBuilder

pb = PromptBuilder(game="sokoban", target_phase="simulation")
prompt = pb.build(
    record_files=["mcts/records/Sokoban_xxx.json"],
    tool_source=engine.get_tool_source()["simulation"],
    max_moves_per_trace=5,
)
pb.save(prompt)   # → LLM/drafts/sokoban_simulation_prompt.txt
```

## Running Tests

```bash
# All 138 tests
python -m pytest tests/ -v

# Single file
python -m pytest tests/test_prompt_builder.py -v
```

## Roadmap

| Step | Status | Description |
|------|--------|-------------|
| 1 | ✅ | MCTS node + engine |
| 2 | ✅ | 10 Sokoban levels + pytest |
| 3 | ⬜ | Test different heuristics as tools |
| 4 | ✅ | Trace logger (JSON records) |
| 5 | ✅ | Prompt builder (game rules + traces + tool source) |
| 6 | ⬜ | LLM querying (generate improved heuristics) |
| 7 | ⬜ | Single-loop test on 10 Sokoban levels |
| 8 | ⬜ | Multi-loop iterative improvement |

## Requirements

- Python 3.12+
- pytest (for tests only)
- No external packages required for the MCTS framework itself
