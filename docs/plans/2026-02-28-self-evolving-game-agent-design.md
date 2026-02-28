# Design: Self-Evolving Game Agent with Tool Integration

**Date:** 2026-02-28
**Project:** CSE 291A - AI Agents, UCSD Winter 2026

## Overview

A hybrid RL+LLM system where MCTS starts with zero heuristics on hard turn-based games, and an LLM periodically injects auto-discovered heuristic tools (as Python code plugins) to make MCTS viable on games with huge state spaces. Tools are game-agnostic in format and can transfer across games.

## Architecture

Three subsystems: MCTS Engine, Tool Creation Pipeline, Cross-Game Tool Manager.

### Subsystem A: MCTS Engine with Dynamic Tool Injection

**Core principle:** MCTS starts vanilla. During training, tools are hot-loaded as code plugins without restarting. Tools compose via phase-based dispatch.

**Implementation:** Custom Python MCTS using OpenSpiel for game state management. OpenSpiel's `MCTSBot` serves as the vanilla baseline for comparison (Approach 2 from exploration).

**Tool hook points (6 types):**

| Hook | MCTS Phase | Default (no tool) | With tool |
|------|-----------|-------------------|-----------|
| `selection_prior` | Selection | UCT (uniform prior) | PUCT with learned priors |
| `action_filter` | Expansion | All legal actions | Pruned action subset |
| `macro_actions` | Expansion | None | Compound strategic moves |
| `rollout_policy` | Simulation | Random action | Biased action selection |
| `state_evaluator` | Simulation | Play to terminal | Early cutoff with heuristic score |
| `reward_shaper` | Backpropagation | Raw win/loss (+1/-1) | Shaped intermediate rewards |

**Tool dispatch logic:**
- Multiple tools per hook: action_filters intersect, evaluators average, rollout_policies weighted-sample
- Tools are `.py` files loaded dynamically from `tool_pool/` directory
- Each tool declares its hook type via `__TOOL_META__` dict (Yunjue convention)
- Timeout enforcement: tools exceeding X ms/call are skipped

**Tool interface (game-agnostic):**

```python
# All tools receive pyspiel.State and use only the generic API:
# state.legal_actions(), state.returns(), state.current_player(),
# state.clone(), state.apply_action(), state.is_terminal(),
# state.observation_tensor(), str(state)

__TOOL_META__ = {
    "name": "evaluate_mobility",
    "type": "state_evaluator",  # one of the 6 hook types
    "description": "Score states by ratio of my legal actions to opponent's",
}

def run(state) -> float:
    """Return evaluation score for current player. Range [-1, 1]."""
    my_actions = len(state.legal_actions())
    state_copy = state.clone()
    # Simulate a pass to get opponent's perspective
    # ... game-agnostic mobility calculation
    return score
```

**Macro actions (Options framework):**
- A macro action is a Python function returning `list[int]` (sequence of primitive actions)
- During expansion, macro actions are added as children alongside primitive actions
- During simulation, the full sequence executes then normal rollout continues
- Backpropagation treats the macro as a single decision node
- Macro actions have a termination condition (abort if any step becomes illegal)

### Subsystem B: Tool Creation Pipeline

**Trigger:** Performance plateau detection. Rolling window of last N games (default 50). Trigger when:
- Win rate hasn't improved by >2% over 2 consecutive windows, OR
- Win rate dropped by >5% (regression)

**Pipeline stages:**

```
Stage 1: Trace Collection
  - Record full game states, actions, outcomes for all games
  - On trigger, select informative traces: losses, close games, games where
    the agent made clearly suboptimal moves

Stage 2: Trace Analysis (LLM)
  - Input: selected traces + current tool pool + game description
  - Prompt: "Why is the agent losing? What heuristic could help?"
  - Output: Tool specification (name, type, description, pseudocode)

Stage 3: Code Generation (LLM)
  - Input: tool spec + tool interface template + OpenSpiel API reference
  - Prompt: "Write a game-agnostic Python function implementing this heuristic"
  - Output: Complete .py file following __TOOL_META__ convention

Stage 4: Validation
  - Syntax: AST parse, check __TOOL_META__ exists, check run() signature
  - Runtime: Execute on 100 random game states, no crashes, outputs in expected range
  - Timeout: Must complete in <X ms per call on average
  - Up to 3 LLM retry attempts on failure (Yunjue's self-healing pattern)

Stage 5: A/B Testing
  - Play 50 games: MCTS + current tools + new tool
  - Play 50 games: MCTS + current tools (no new tool)
  - Keep if win rate improves or is neutral (within 2%)
  - Discard if win rate drops

Stage 6: Promotion
  - Add validated tool to tool_pool/<game_name>/
  - Update metadata.json with origin, performance stats
  - Log the tool generation for experiment tracking
```

**LLM configuration:** Follows Yunjue's `conf.yaml` pattern. Models are swappable:

```yaml
TRACE_ANALYZER:
  base_url: ...
  model: "deepseek-v3"
  temperature: 0.7

CODE_GENERATOR:
  base_url: ...
  model: "deepseek-v3"
  temperature: 0.2

TOOL_VALIDATOR:
  base_url: ...
  model: "qwen3-7b"  # cheaper model for validation retries
  temperature: 0.2
```

### Subsystem C: Cross-Game Tool Manager

**Tool pool directory structure:**

```
tool_pool/
├── global/                    # Tools validated across 2+ games
│   ├── evaluate_mobility.py
│   ├── prune_dominated.py
│   └── greedy_rollout.py
├── connect_four/
│   └── column_center_bias.py
├── quoridor/
│   └── wall_near_path.py
├── chess/
│   └── (populated during training)
└── metadata.json
```

**metadata.json schema:**

```json
{
  "tool_name": {
    "type": "state_evaluator",
    "origin_game": "connect_four",
    "games_tested": {"connect_four": +12.3, "quoridor": +5.1},
    "generation": 3,
    "created_by_model": "deepseek-v3",
    "created_at": "2026-03-01T12:00:00",
    "code_hash": "abc123"
  }
}
```

**Cross-game transfer protocol:**
1. Load all `global/` tools + tools from related games
2. Validate each tool on 100 random states of the new game (crash check)
3. A/B test each tool: 50 games with vs 50 without
4. Keep helpful/neutral tools, discard harmful ones
5. After training, promote tools that help on 2+ games to `global/`

**Tool merging (Yunjue's absorption pattern):**
- After accumulating many tools, run a deduplication pass
- LLM clusters semantically similar tools
- Merge each cluster into a single canonical tool
- Re-validate merged tools

## Game Ladder

| Game | Complexity | Purpose | Key tools expected |
|------|-----------|---------|-------------------|
| Connect Four | Low (4.5 x 10^12 states) | Sanity check, initial tool discovery | Column preference, threat detection, win-in-N eval |
| Quoridor | Medium (~10^20 states) | Main testbed, ground-truth comparison | Path evaluation, wall pruning, corridor blocking |
| Chess | High (~10^47 states) | Transfer evaluation, stretch goal | Piece mobility, material eval, king safety |

## Evaluation Plan

### Axis 1: Win Rate vs Baselines

For each game, at fixed simulation budgets (100, 500, 1000, 5000):

| Agent | 100 sims | 500 sims | 1000 sims | 5000 sims |
|-------|----------|----------|-----------|-----------|
| Random | - | - | - | - |
| Vanilla MCTS (OpenSpiel) | - | - | - | - |
| MCTS + tools (gen 1) | - | - | - | - |
| MCTS + tools (final) | - | - | - | - |
| MCTS + transfer tools | - | - | - | - |

### Axis 2: Sample Efficiency Curves

- X-axis: simulations per move (log scale)
- Y-axis: win rate vs vanilla MCTS at 1000 sims
- Lines: vanilla, +tools gen 1, +tools final, +transfer tools
- Target: tools curve reaches same win rate at 5-10x fewer simulations

### Axis 3: Cross-Game Transfer

- Train tools on Connect Four
- Apply to Quoridor: compare "from scratch" vs "with CF tools"
- Metric: games to reach X% win rate (learning speed)

## Project Structure

```
src/
├── mcts/
│   ├── engine.py           # Core MCTS with tool hooks
│   ├── node.py             # MCTS tree node
│   └── tool_registry.py    # Dynamic tool loading & dispatch
├── tools/
│   ├── base.py             # Tool interface definitions & __TOOL_META__ spec
│   ├── generator.py        # LLM tool generation pipeline
│   ├── validator.py        # Syntax, runtime, timeout, A/B validation
│   └── manager.py          # Cross-game tool pool manager
├── training/
│   ├── trainer.py          # Training loop with plateau detection
│   ├── trace_recorder.py   # Game trace collection & selection
│   └── evaluator.py        # Win rate, efficiency, transfer evaluation
├── llm/
│   ├── client.py           # Configurable LLM client
│   └── prompts/            # Prompt templates
│       ├── trace_analysis.md
│       ├── code_generation.md
│       ├── tool_validation.md
│       └── tool_merge.md
├── config.py               # Global configuration
└── games/
    └── adapter.py          # OpenSpiel game wrapper with trace hooks

tool_pool/                  # Persisted tools (versioned)
experiments/                # Evaluation scripts & results
conf.yaml                  # LLM model configuration
main.py                    # Entry point
```

## MVP Phases

**Phase 1:** MCTS engine + tool hooks + hand-written tools on Connect Four
**Phase 2:** LLM tool generation pipeline on Connect Four
**Phase 3:** Cross-game transfer to Quoridor, then Chess
**Phase 4:** Full evaluation suite
