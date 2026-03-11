# MCTS Phase Interaction

The four MCTS phases (selection, expansion, simulation, backpropagation) form a pipeline. Changing one phase affects the others. When optimizing tools via LLM, keep this interaction in mind.

## Pipeline Flow

```
Selection → Expansion → Simulation → Backpropagation
    ↑                                        │
    └────────────────────────────────────────┘
```

1. **Selection** — Picks which leaf node to expand using UCB1:
   - `score = value/visits + C * sqrt(log(parent_visits)/visits)`
   - Uses `value` and `visits` from backpropagation
   - Favors nodes with high value (promising) or low visits (under-explored)

2. **Expansion** — Adds one child to the selected node:
   - Orders untried actions (e.g. by heuristic)
   - Action order affects which branch is explored first
   - Produces children that selection will later choose among

3. **Simulation** — Rollout from the new leaf to a terminal state:
   - Returns a reward (e.g. 0–1 for Sokoban)
   - Reward flows into backpropagation

4. **Backpropagation** — Updates visits and value up the tree:
   - Propagates reward and visit count from leaf to root
   - Feeds the `value`/`visits` used by selection’s UCB1

## Synergy Requirements

For good performance, the phases should be coherent:

| Phase | Should produce | Consumed by |
|-------|----------------|-------------|
| Selection | Choice of promising leaf | — |
| Expansion | Children with sensible action order | Selection (indirectly) |
| Simulation | Rewards that reflect state quality | Backpropagation |
| Backpropagation | Value/visits that guide selection | Selection |

- **Selection** should favor branches that lead to good outcomes.
- **Expansion** should order actions so promising ones are tried early.
- **Simulation** should return rewards that reflect true state quality.
- **Backpropagation** should aggregate rewards in a way that guides selection.

## Optimization Implications

- Each phase is optimized **independently** on different levels.
- A selection tuned for level A may conflict with an expansion tuned for level B.
- Use **current_fns** (the coherent set after all iterations) for final evaluation, not a per-phase historical mix.
- Run **cross-level regression checks** when adopting a new tool to catch regressions on other levels.
