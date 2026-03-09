# Heuristic Tips For MCTS Phases

This document is a guide for proposing heuristics in the MCTS tool pipeline.
The key rule is that each heuristic should live in the phase where it has the
right scope, cost, and effect.

## Overview

The four MCTS phases in this project are:

1. `selection`
2. `expansion`
3. `simulation`
4. `backpropagation`

They serve different purposes:

- `selection` chooses which existing branch of the tree to follow.
- `expansion` decides which new child states are worth adding.
- `simulation` estimates how promising a state is by rolling forward.
- `backpropagation` updates ancestor node values using the rollout result.

For Sokoban, a good heuristic usually does one of these things:

- prefers states with more boxes on targets
- prefers states with smaller box-to-target distance
- rejects obvious deadlocks
- prefers useful pushes over wandering player movement
- reduces loops and stagnation

## Selection

### What this phase does

Selection walks down the existing search tree from the root to a leaf or
frontier node. It must balance exploration and exploitation.

In practice, this phase answers:

"Among nodes we already know about, which branch should we trust next?"

### Good heuristics for selection

- add a bonus for states with more boxes already on targets
- add a bonus for lower total box distance
- penalize states that look deadlocked or nearly deadlocked
- add a novelty bonus for states visited less often
- slightly prefer branches reached by meaningful pushes rather than only player repositioning

### Sokoban-specific examples

- prefer nodes where `boxes_on_targets()` is higher
- prefer nodes where `total_box_distance()` is lower
- deprioritize states where a box is pinned in a corner and not on a target
- deprioritize states that repeat common box layouts

### What not to put here

- expensive multi-step rollout logic
- deep deadlock analysis that must simulate many actions
- final reward shaping formulas meant for rollouts

Selection is called very often, so the heuristic should be cheap and mostly
rank nodes rather than perform heavy reasoning.

## Expansion

### What this phase does

Expansion creates new child nodes from a frontier node. This is the phase where
the search decides which actions are worth materializing into the tree.

In practice, this phase answers:

"Which next states should be added, and which are so bad that they should be
filtered or deprioritized immediately?"

### Good heuristics for expansion

- prune actions that create obvious deadlocks
- prefer pushes that move a box closer to a target
- prefer actions that increase future push opportunities
- avoid expanding duplicate or transposed states
- penalize no-op style movement that only repositions the player without helping box progress

### Sokoban-specific examples

- reject pushes into non-target corners
- reject wall-line deadlocks when no target exists on the relevant row or column
- prioritize pushes that reduce total Manhattan distance from boxes to targets
- prefer resulting states where the player is positioned behind boxes for future pushes
- merge equivalent box configurations when different player paths reach the same useful state

### What not to put here

- long rollout policies
- reward aggregation over many simulated steps
- node-value update rules

Expansion is the best place for hard constraints and pruning.

## Simulation

### What this phase does

Simulation rolls forward from a state to estimate how promising it is. This is
where random rollout can be replaced with heuristic-guided rollout.

In practice, this phase answers:

"If play continues from this state, how much progress does it appear to have?"

### Good heuristics for simulation

- assign intermediate reward for partial progress
- reward increases in boxes on targets
- reward reductions in box-to-target distance
- penalize deadlocks, loops, and long stagnation
- prefer pushes over wandering
- use weighted or softmax action choice instead of pure random sampling
- terminate early when the rollout is clearly stuck

### Sokoban-specific examples

- return `1.0` if solved, otherwise a shaped score based on:
  - fraction of boxes on targets
  - improvement in total box distance
  - deadlock penalties
  - loop or stagnation penalties
- add a bonus when a rollout action actually pushes a box
- stop rollout early if recent states repeat
- stop rollout early if many steps pass without improving any progress metric

### What not to put here

- global tree-level visit balancing
- acceptance or rejection criteria for candidate tools
- final evaluator policy for whether a puzzle counts as solved

If the goal is to provide intermediate rewards for Sokoban while keeping final
evaluation binary, this is the main phase to modify.

## Backpropagation

### What this phase does

Backpropagation sends the simulation result back up the visited path and updates
node statistics such as visit count and accumulated value.

In practice, this phase answers:

"How strongly should this rollout change the value of the nodes that led to it?"

### Good heuristics for backpropagation

- discount rewards by depth so faster progress matters more
- reduce the impact of noisy or weak rollout signals
- keep solved outcomes much more important than partial progress
- penalize very long paths
- combine shaped rollout score with terminal success in a controlled way

### Sokoban-specific examples

- give full weight to solved rollouts and lower weight to partial-progress rollouts
- apply depth discount so shorter successful plans dominate longer drifting ones
- reduce credit for rollouts that show minor distance improvement but no boxes placed
- separate "progress value" from "solved value" if partial rewards are too noisy

### What not to put here

- move generation logic
- deadlock pruning as a primary mechanism
- rollout action-choice policy

Backpropagation should control how evidence is accumulated, not invent new game
logic.

## Mapping Common Sokoban Heuristics To Phases

| Heuristic | Best phase |
|-----------|------------|
| UCB adjustment with progress bonus | `selection` |
| Corner deadlock pruning | `expansion` |
| Wall deadlock pruning | `expansion` |
| Prefer push actions | `expansion`, `simulation` |
| Boxes on targets as partial reward | `simulation` |
| Distance-to-target shaping | `selection`, `simulation` |
| Loop detection | `simulation` |
| Stagnation cutoff | `simulation` |
| Depth discount on reward | `backpropagation` |
| Weight solved outcomes above partial progress | `backpropagation` |

## Guidance For LLM Heuristic Proposals

When proposing a heuristic, the analyzer should state:

1. which phase it belongs to
2. what problem it addresses
3. why that phase is the correct place for it
4. whether it is cheap enough for that phase
5. what risk it introduces

Good proposal patterns:

- "Add cheap deadlock pruning in `expansion` because it filters obviously bad child states before they enter the tree."
- "Add shaped partial reward in `simulation` because rollout scoring is the correct place to evaluate incomplete Sokoban progress."
- "Add depth discount in `backpropagation` because value propagation should control how much long vs short paths are trusted."

Bad proposal patterns:

- "Put all Sokoban knowledge into `selection`."
- "Change final game reward to partial credit when the goal is binary final evaluation."
- "Do expensive multi-step lookahead inside `selection`."

## Recommended Default For Sokoban

If only one phase is being improved first, prioritize:

1. `expansion` for deadlock pruning
2. `simulation` for shaped progress reward
3. `selection` for cheap progress-aware ranking
4. `backpropagation` for reward calibration

This order usually gives the best return because Sokoban benefits heavily from
avoiding bad children and from more informative rollouts.
