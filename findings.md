# Findings: Self-Evolving Game Agent

## Source Documents Read

| Document | Key Contribution |
|----------|-----------------|
| `brainstorm.md` | Q&A covering all 10 design questions: tool types, tool interfaces, LLM pipeline, game progression, evaluation |
| `docs/plans/2026-02-28-implementation-plan.md` | 3000+ line TDD plan — 20 tasks across 4 phases with exact code, tests, and commit steps |
| `docs/plans/2026-02-28-self-evolving-game-agent-design.md` | Architecture spec: subsystems, tool hook table, pipeline stages, metadata schema |
| `CSE291A_Team09_02_03_Proposal.pdf` | Project overview: motivation (prompt-based tool use is insufficient), Yunjue background, cross-game transfer plan |

---

## Existing Codebase

| Directory | Contents |
|-----------|----------|
| `CSE291A_Project/connect_four/` | `MCTS_TT.ipynb` — exploratory MCTS notebook for Connect Four |
| `CSE291A_Project/tic_tac_toe/` | `MCTS.ipynb` — exploratory MCTS notebook |
| `CSE291A_Project/quoridor/` | Full C++ MCTS implementation (MCTS.hpp, QuoridorAI.hpp, AI.hpp, Board.hpp, etc.) + Python agent |
| `CSE291A_Project/play_zork/` | Zork LLM vision agent (unrelated to new system) |
| `Yunjue-Agent/` | Reference Yunjue agent: `conf.yaml.example`, `src/`, `scripts/`, `example/` |
| `open_spiel/` | OpenSpiel library (local copy or install) |

**Key insight:** The Quoridor C++ code has hand-crafted heuristics (BFS shortest path, wall pruning, 70/30 greedy rollout) that serve as ground-truth benchmarks for what the LLM should auto-discover.

---

## Tool Types (6 Hook Points)

| Type | MCTS Phase | Default | With Tool |
|------|-----------|---------|-----------|
| `state_evaluator` | Simulation (early cutoff) | Play to terminal | Heuristic score → early return |
| `action_filter` | Expansion | All legal actions | Pruned subset |
| `rollout_policy` | Simulation | Random | Biased selection |
| `selection_prior` | Selection | UCT (uniform prior) | PUCT with learned priors |
| `reward_shaper` | Backpropagation | Raw +1/-1 | Shaped intermediate reward |
| `macro_action` | Expansion | None | Compound strategic moves (Options framework) |

**Tool dispatch:** Multiple tools of same type compose as:
- `action_filter` → intersection of filtered sets
- `state_evaluator` → average of scores
- `rollout_policy` → weighted random sample from policies

---

## Tool File Convention (from Yunjue)

```python
__TOOL_META__ = {
    "name": "tool_name",
    "type": "state_evaluator",   # one of the 6 hook types
    "description": "...",
}

def run(state) -> float:
    """Game-agnostic: only use state.legal_actions(), state.clone(),
    state.apply_action(), state.is_terminal(), state.returns(),
    state.current_player(), state.observation_tensor(), str(state)"""
    ...
```

All tools use only OpenSpiel's generic API → game-agnostic by design.

---

## Project Structure (from implementation plan)

```
src/
├── mcts/
│   ├── engine.py          # MCTSEngine with 6 tool hooks
│   ├── node.py            # MCTSNode (UCT, PUCT, backprop)
│   └── tool_registry.py   # Dynamic tool loading & dispatch
├── tools/
│   ├── base.py            # ToolType, ToolMeta, load_tool_from_file
│   ├── generator.py       # LLM codegen pipeline
│   ├── validator.py       # Syntax, runtime, timeout, A/B validation
│   └── manager.py         # Cross-game tool pool manager
├── training/
│   ├── trainer.py         # Training loop + plateau detection
│   ├── trace_recorder.py  # Game trace collection
│   └── evaluator.py       # Win rate, efficiency, transfer metrics
├── llm/
│   ├── client.py          # Configurable LLM client (OpenAI-compatible)
│   └── prompts/           # trace_analysis.md, code_generation.md, ...
├── config.py
└── games/
    └── adapter.py         # GameAdapter wrapping pyspiel

tool_pool/
├── global/                # Tools valid on 2+ games
├── connect_four/          # Game-specific tools
├── quoridor/
├── chess/
└── metadata.json          # Tool lineage, performance stats

tests/                     # pytest; mirrors src/ structure
pyproject.toml             # open_spiel, pyyaml, openai deps
conf.yaml                  # LLM model config (3 separate sections)
main.py                    # CLI entry point
```

---

## Evaluation Axes

1. **Win rate vs baselines** at fixed sim budgets (100, 500, 1000, 5000): Random, Vanilla MCTS, MCTS+tools (gen 1), MCTS+tools (final), MCTS+transfer tools
2. **Sample efficiency curves**: win rate vs sims (log scale). Goal: tools reach same win rate at 5-10x fewer sims
3. **Cross-game transfer speed**: train on Connect Four, apply to Quoridor. Metric = games to reach X% win rate

---

## LLM Pipeline (Yunjue-inspired, 6 Stages)

1. Trace Collection — select losses, close games, clearly suboptimal moves
2. Trace Analysis (LLM, temperature=0.7) — "Why is the agent losing?"
3. Code Generation (LLM, temperature=0.2) — produce .py file with __TOOL_META__ + run()
4. Validation — AST parse, 100-state runtime check, timeout enforcement, up to 3 LLM retry attempts
5. A/B Testing — 50 games with vs 50 without new tool
6. Promotion — add to tool_pool/<game>/, update metadata.json

---

## Yunjue Reference Patterns

- `conf.yaml` with separate model configs per role (TRACE_ANALYZER, CODE_GENERATOR, TOOL_VALIDATOR)
- `__TOOL_META__` convention for self-describing tool files
- Spec → Codegen → Validate → Promote workflow
- Self-healing: up to 3 LLM retries on validation failure
- Tool merging/deduplication via semantic clustering after tool accumulation
- Absorption: merge new tools into global pool after multi-game validation

---

## Performance Plateau Detection

```
Rolling window: last N=50 games
Trigger evolution if:
  - win rate hasn't improved by >2% over 2 consecutive windows
  - OR win rate dropped by >5% (regression detection)
```
