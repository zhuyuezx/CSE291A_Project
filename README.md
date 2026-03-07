# CSE 291A Winter 2025 Project

Exploring LLM-augmented game-playing agents across multiple domains.

## Repository Structure

| Directory | Description |
|-----------|-------------|
| **`Tool_Creation/`** | MCTS + LLM Heuristic Optimization — a modular Monte Carlo Tree Search framework with pluggable phase tools and an automated LLM-driven improvement pipeline. See [Tool_Creation/README.md](Tool_Creation/README.md) for full documentation. |
| `connect_four/` | Connect Four MCTS experiments (Jupyter notebook) |
| `tic_tac_toe/` | Tic-Tac-Toe MCTS experiments (Jupyter notebook) |
| `quoridor/` | Quoridor AI with C++ MCTS engine and GUI |
| `games/` | Standalone game implementations (Uno) |
| `play_zork/` | LLM-powered Zork agent using vision-based screen reading |

## Quick Start — MCTS + LLM Pipeline

```bash
cd Tool_Creation
./run_pipeline.sh            # run the full optimization loop
./run_pipeline.sh --iters 10 # override iteration count
```

See [Tool_Creation/README.md](Tool_Creation/README.md) for detailed setup and usage.

## Requirements

- Python 3.12+
- An OpenAI-compatible API key (set in `Tool_Creation/.env`) for the LLM pipeline
- pytest for running the test suite
