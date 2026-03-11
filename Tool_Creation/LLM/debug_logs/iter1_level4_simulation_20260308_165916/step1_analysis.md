# step1_analysis

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-08 16:59:35 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 18.86s |
| Status     | success |
| Tokens     | 7180 |
| Validation | invalid — No code block found in response. |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban' (phase: simulation).

APPROACH — 70 / 30 RULE:
  ~70% of iterations: INCREMENTAL OPTIMIZATION
    • Start from the CURRENT code.
    • Make targeted, gradual improvements (add a check,
      tweak weights, add a heuristic factor, etc.).
    • Prefer building on what works rather than replacing it.

  ~30% of iterations: PARADIGM SHIFT (when warranted)
    • If the current approach is fundamentally flawed or
      plateauing, you may propose a larger restructure.
    • Explain clearly WHY a rewrite is needed.
    • Even rewrites should keep proven components that work.

GENERAL RULES:
  • Write clean, well-structured code — as long as the
    heuristic needs to be (no artificial line limit).
  • Each iteration builds on the previous version.
  • Complex heuristics with multiple factors are encouraged
    when they improve play quality.

------------------------------------------------------------
GAME RULES
------------------------------------------------------------
Game: Sokoban

== Overview ==
Sokoban is a single-player puzzle game. The player pushes boxes onto
target positions inside a grid-based warehouse. The puzzle is solved
when every box is on a target.

== Symbols ==
  #   wall (impassable)
  .   target position
  $   box
  *   box on a target
  @   player
  +   player standing on a target
  (space)  empty floor

== Rules ==
1. The player can move one step at a time: UP, DOWN, LEFT, or RIGHT.
2. The player can push a box by walking into it, but only if the cell
   on the far side of the box is empty floor or a target (not a wall
   and not another box).
3. The player cannot pull boxes — only push.
4. The puzzle is solved when ALL boxes are on target positions.
5. A game is lost (unsolvable) when a box is stuck in a position from
   which it can never reach any remaining target (deadlock). The
   simplest deadlock is a box pushed into a corner that is not a target.

== Actions ==
The action space is {0, 1, 2, 3} corresponding to:
  0 = UP    (row - 1)
  1 = DOWN  (row + 1)
  2 = LEFT  (col - 1)
  3 = RIGHT (col + 1)

== State Representation ==
Each state is described by:
  - The player's (row, col) position.
  - The set of (row, col) positions of all boxes.
  - A step counter (game terminates if max_steps is reached).
  - Derived metrics: boxes_on_targets count and total_box_distance
    (sum of Manhattan distance from each box to its nearest target).

== Reward ==
The state.returns() method returns a SHAPED continuous reward in [0, 1]:
  - Solved (all boxes on targets): 1.0
  - Deadlocked (box stuck in non-target corner): 0.0
  - Otherwise: a weighted combination of:
       70% — fraction of boxes on target  (boxes_on_targets / num_targets)
       30% — distance score  (1 − total_box_distance / max_dist)
    This gives a fine-grained gradient even when the puzzle is not yet solved.

== GameState API ==
Public attributes (not methods):
  walls     : frozenset[tuple[int,int]]   – wall positions
  targets   : frozenset[tuple[int,int]]   – target positions
  boxes     : set[tuple[int,int]]         – current box positions
  player    : tuple[int,int]              – current player position
  height, width : int                     – grid dimensions
  num_targets   : int                     – number of targets
  steps, max_steps : int                  – current / maximum step count

Public methods (call with parentheses):
  clone()            → new independent copy of the state
  legal_actions()    → list[int]  (subset of {0,1,2,3})
  apply_action(a)    → None (mutates the state in-place)
  is_terminal()      → bool
  returns()          → list[float]  (shaped reward, see above)
  current_player()   → int (always 0, single-player game)
  state_key()        → str (hashable key for transposition)
  boxes_on_targets() → int
  total_box_distance() → int  (sum of min Manhattan dist per box)

== Key Strategic Concepts ==
  - Avoid pushing boxes into corners (unless the corner IS a target).
  - Avoid pushing boxes against walls in directions where no target
    lies along that wall — the box becomes permanently stuck.
  - Plan pushes so that earlier pushes do not block corridors needed
    for later pushes.
  - Minimize total push count; fewer pushes usually means fewer
    opportunities for deadlock.
  - The order in which boxes are placed on targets matters:
    placing one box may block the path needed for another.


------------------------------------------------------------
MCTS TOOL FUNCTIONS (all 4 phases)
------------------------------------------------------------

--- selection ---
```python
"""
Default selection: UCB1 tree policy.

Walk down the tree choosing the child with the highest UCB1 score
until we reach a node that is either terminal or has untried actions
(then expand it).
"""

import math


def default_selection(node, exploration_weight: float = 1.41):
    """
    Pure UCB1 tree walk.

    Descends the tree choosing the child with the highest UCB1 score.
    Stops when reaching a node that has untried actions (needs expansion)
    or is terminal.

    Args:
        node:               Root MCTSNode to start selection from.
        exploration_weight:  UCB1 exploration constant C.

    Returns:
        An MCTSNode that is either terminal or has untried actions.
    """
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # hand off to expansion phase
        # UCB1 selection among fully-expanded children
        log_parent = math.log(node.visits)
        best, best_score = None, -math.inf
        for child in node.children.values():
            exploit = child.value / child.visits
            explore = exploration_weight * math.sqrt(log_parent / child.visits)
            score = exploit + explore
            if score > best_score:
                best, best_score = child, score
        node = best
    return node
```

--- expansion ---
```python
"""
Default expansion: expand one untried action.

Pops an untried action from the node, creates a child node
with the resulting state, and returns it.
"""


def default_expansion(node):
    """
    Expand one untried action from the given node.

    Args:
        node: MCTSNode with at least one untried action.

    Returns:
        The newly created child MCTSNode.
    """
    action = node._untried_actions.pop()
    child_state = node.state.clone()
    child_state.apply_action(action)
    from mcts.node import MCTSNode
    child = MCTSNode(child_state, parent=node, parent_action=action)
    node.children[action] = child
    return child
```

--- simulation ◀ TARGET ---
```python
"""
LLM-generated MCTS tool: simulation
Description: No changes needed; the function meets the intended incremental improvements.
Generated:   2026-03-08T16:30:23.952799
"""

import random
import math
from typing import Set, Tuple


def _manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def _is_corner_deadlock(pos: Tuple[int, int],
                        walls: Set[Tuple[int, int]],
                        targets: Set[Tuple[int, int]]) -> bool:
    """
    Detect the simplest dead‑lock: a box in a corner formed by two walls
    (or wall + boundary) that is **not** on a target.
    """
    if pos in targets:
        return False
    r, c = pos
    # four possible corner configurations
    combos = [((r - 1, c), (r, c - 1)),   # up & left
              ((r - 1, c), (r, c + 1)),   # up & right
              ((r + 1, c), (r, c - 1)),   # down & left
              ((r + 1, c), (r, c + 1))]   # down & right
    for a, b in combos:
        if a in walls and b in walls:
            return True
    return False


def _heuristic(state) -> float:
    """
    Compute the same shaped reward used by GameState.returns()
    but without the terminal check.  This provides a smooth estimate
    during rollouts.
    """
    num_targets = state.num_targets
    if num_targets == 0:
        return 0.0

    # fraction of boxes already on targets
    on_target = state.boxes_on_targets()
    frac = on_target / num_targets

    # total distance of each box to its nearest target
    total_dist = state.total_box_distance()

    # a safe upper bound for total distance:
    # max possible per box = height + width (Manhattan across whole board)
    max_dist = (state.height + state.width) * num_targets
    if max_dist == 0:
        dist_score = 0.0
    else:
        dist_score = 1.0 - (total_dist / max_dist)

    return 0.7 * frac + 0.3 * dist_score


def default_simulation(state,
                       perspective_player: int,
                       max_depth: int = 0,
                       epsilon: float = 0.2) -> float:
    """
    Heuristic‑guided simulation for Sokoban.

    * Uses an epsilon‑greedy policy that prefers actions improving the
      distance / target‑coverage heuristic.
    * Prunes actions that cause an immediate corner dead‑lock.
    * Stops after a modest horizon (default = 20 * num_targets) and
      returns the current heuristic estimate if the game is not solved.
    * Falls back to pure random when all remaining actions are dead‑locked.

    Args:
        state:               GameState to simulate from (will be cloned).
        perspective_player:  Player index whose reward we return.
        max_depth:           Upper bound on rollout length. If 0, a
                             problem‑size bound is used.
        epsilon:             Exploration probability for epsilon‑greedy.

    Returns:
        Float reward from perspective_player's viewpoint.
    """
    # Clone once – we will mutate this copy.
    sim_state = state.clone()

    # Choose a sensible depth if caller left the default.
    if max_depth <= 0:
        max_depth = max(30, state.num_targets * 20)

    depth = 0
    while not sim_state.is_terminal() and depth < max_depth:
        legal = sim_state.legal_actions()
        if not legal:
            break

        # Evaluate each legal action.
        best_h = -math.inf
        best_actions = []
        safe_actions = []  # actions that do NOT immediately dead‑lock

        for a in legal:
            # Peek at the result of action a.
            child = sim_state.clone()
            child.apply_action(a)

            # Simple corner dead‑lock detection.
            dead = False
            for bpos in child.boxes:
                if _is_corner_deadlock(bpos, child.walls, child.targets):
                    dead = True
                    break

            if not dead:
                safe_actions.append((a, child))
                h = _heuristic(child)
                if h > best_h:
                    best_h = h
                    best_actions = [(a, child)]
                elif h == best_h:
                    best_actions.append((a, child))

        # If every action leads to a dead‑lock, fall back to random.
        if not safe_actions:
            chosen_action, next_state = random.choice([
                (a, sim_state.clone()) for a in legal
            ])
            next_state.apply_action(chosen_action)
        else:
            # epsilon‑greedy selection among the safe actions.
            if random.random() < epsilon:
                # pure exploration among safe actions
                chosen_action, next_state = random.choice(safe_actions)
            else:
                # exploit: choose uniformly among the best‑heuristic actions
                chosen_action, next_state = random.choice(best_actions)

        # Advance simulation.
        sim_state = next_state
        depth += 1

    # If we reached a terminal state, use the exact returns().
    if sim_state.is_terminal():
        return sim_state.returns()[perspective_player]

    # Otherwise return the heuristic estimate for the current (non‑solved) state.
    return _heuristic(sim_state)
```

--- backpropagation ---
```python
"""
Default backpropagation: walk up the tree updating visits and value.

Value convention: each node's value is stored from the perspective of
the player who CHOSE the action leading to that node (= node.parent's
current_player). This lets UCB1 always maximize, which is correct for
both the searching player and the opponent.

For single-player games (e.g. Sokoban), all nodes share the same
perspective so the sign never flips.
"""


def default_backpropagation(node, reward: float) -> None:
    """
    Backpropagate a simulation result from leaf to root.

    Args:
        node:   The leaf MCTSNode where simulation started.
        reward: The simulation reward from the ROOT player's perspective.
    """
    # Find root's current player (= perspective of the reward)
    root = node
    while root.parent is not None:
        root = root.parent
    perspective = root.state.current_player()

    # Walk back up, flipping sign at opponent nodes
    while node is not None:
        node.visits += 1
        # Who chose the move that created this node?
        mover = node.parent.state.current_player() if node.parent else perspective
        node.value += reward if mover == perspective else -reward
        node = node.parent
```

------------------------------------------------------------
TARGET HEURISTIC TO IMPROVE (simulation)
------------------------------------------------------------
```python
"""
LLM-generated MCTS tool: simulation
Description: No changes needed; the function meets the intended incremental improvements.
Generated:   2026-03-08T16:30:23.952799
"""

import random
import math
from typing import Set, Tuple


def _manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def _is_corner_deadlock(pos: Tuple[int, int],
                        walls: Set[Tuple[int, int]],
                        targets: Set[Tuple[int, int]]) -> bool:
    """
    Detect the simplest dead‑lock: a box in a corner formed by two walls
    (or wall + boundary) that is **not** on a target.
    """
    if pos in targets:
        return False
    r, c = pos
    # four possible corner configurations
    combos = [((r - 1, c), (r, c - 1)),   # up & left
              ((r - 1, c), (r, c + 1)),   # up & right
              ((r + 1, c), (r, c - 1)),   # down & left
              ((r + 1, c), (r, c + 1))]   # down & right
    for a, b in combos:
        if a in walls and b in walls:
            return True
    return False


def _heuristic(state) -> float:
    """
    Compute the same shaped reward used by GameState.returns()
    but without the terminal check.  This provides a smooth estimate
    during rollouts.
    """
    num_targets = state.num_targets
    if num_targets == 0:
        return 0.0

    # fraction of boxes already on targets
    on_target = state.boxes_on_targets()
    frac = on_target / num_targets

    # total distance of each box to its nearest target
    total_dist = state.total_box_distance()

    # a safe upper bound for total distance:
    # max possible per box = height + width (Manhattan across whole board)
    max_dist = (state.height + state.width) * num_targets
    if max_dist == 0:
        dist_score = 0.0
    else:
        dist_score = 1.0 - (total_dist / max_dist)

    return 0.7 * frac + 0.3 * dist_score


def default_simulation(state,
                       perspective_player: int,
                       max_depth: int = 0,
                       epsilon: float = 0.2) -> float:
    """
    Heuristic‑guided simulation for Sokoban.

    * Uses an epsilon‑greedy policy that prefers actions improving the
      distance / target‑coverage heuristic.
    * Prunes actions that cause an immediate corner dead‑lock.
    * Stops after a modest horizon (default = 20 * num_targets) and
      returns the current heuristic estimate if the game is not solved.
    * Falls back to pure random when all remaining actions are dead‑locked.

    Args:
        state:               GameState to simulate from (will be cloned).
        perspective_player:  Player index whose reward we return.
        max_depth:           Upper bound on rollout length. If 0, a
                             problem‑size bound is used.
        epsilon:             Exploration probability for epsilon‑greedy.

    Returns:
        Float reward from perspective_player's viewpoint.
    """
    # Clone once – we will mutate this copy.
    sim_state = state.clone()

    # Choose a sensible depth if caller left the default.
    if max_depth <= 0:
        max_depth = max(30, state.num_targets * 20)

    depth = 0
    while not sim_state.is_terminal() and depth < max_depth:
        legal = sim_state.legal_actions()
        if not legal:
            break

        # Evaluate each legal action.
        best_h = -math.inf
        best_actions = []
        safe_actions = []  # actions that do NOT immediately dead‑lock

        for a in legal:
            # Peek at the result of action a.
            child = sim_state.clone()
            child.apply_action(a)

            # Simple corner dead‑lock detection.
            dead = False
            for bpos in child.boxes:
                if _is_corner_deadlock(bpos, child.walls, child.targets):
                    dead = True
                    break

            if not dead:
                safe_actions.append((a, child))
                h = _heuristic(child)
                if h > best_h:
                    best_h = h
                    best_actions = [(a, child)]
                elif h == best_h:
                    best_actions.append((a, child))

        # If every action leads to a dead‑lock, fall back to random.
        if not safe_actions:
            chosen_action, next_state = random.choice([
                (a, sim_state.clone()) for a in legal
            ])
            next_state.apply_action(chosen_action)
        else:
            # epsilon‑greedy selection among the safe actions.
            if random.random() < epsilon:
                # pure exploration among safe actions
                chosen_action, next_state = random.choice(safe_actions)
            else:
                # exploit: choose uniformly among the best‑heuristic actions
                chosen_action, next_state = random.choice(best_actions)

        # Advance simulation.
        sim_state = next_state
        depth += 1

    # If we reached a terminal state, use the exact returns().
    if sim_state.is_terminal():
        return sim_state.returns()[perspective_player]

    # Otherwise return the heuristic estimate for the current (non‑solved) state.
    return _heuristic(sim_state)
```

------------------------------------------------------------
GAMEPLAY TRACES
------------------------------------------------------------

--- Trace #1 ---
Game:       Sokoban (level4)
Timestamp:  2026-03-08T16:59:15.422128
Iterations: 200
Solved:     True
Steps:      10
Returns:    [1.0]

  Move 1: action=2, total_visits=200
    State: Step 0/200 | Boxes on target: 0/2 | Total distance: 2
    Children: [3(v=78, avg=0.423), 2(v=84, avg=0.381), 0(v=38, avg=0.210)]
  Move 2: action=2, total_visits=200
    State: Step 1/200 | Boxes on target: 0/2 | Total distance: 2
    Children: [3(v=36, avg=0.139), 2(v=141, avg=0.404), 0(v=23, avg=0.000)]
  Move 3: action=0, total_visits=200
    State: Step 2/200 | Boxes on target: 0/2 | Total distance: 2
    Children: [3(v=35, avg=0.200), 0(v=165, avg=0.503)]
  Move 4: action=0, total_visits=200
    State: Step 3/200 | Boxes on target: 0/2 | Total distance: 2
    Children: [3(v=22, avg=0.136), 1(v=41, avg=0.317), 0(v=137, avg=0.555)]
  Move 5: action=3, total_visits=200
    State: Step 4/200 | Boxes on target: 0/2 | Total distance: 2
    Children: [3(v=155, avg=0.626), 1(v=45, avg=0.400)]
  Move 6: action=1, total_visits=200
    State: Step 5/200 | Boxes on target: 0/2 | Total distance: 2
    Children: [3(v=54, avg=0.481), 2(v=41, avg=0.415), 1(v=105, avg=0.610)]
  Move 7: action=0, total_visits=200
    State: Step 6/200 | Boxes on target: 1/2 | Total distance: 1
    Children: [3(v=59, avg=0.712), 2(v=42, avg=0.595), 0(v=99, avg=0.768)]
  Move 8: action=3, total_visits=200
    State: Step 7/200 | Boxes on target: 1/2 | Total distance: 1
    Children: [3(v=93, avg=0.753), 2(v=75, avg=0.707), 1(v=32, avg=0.500)]
  Move 9: action=3, total_visits=200
    State: Step 8/200 | Boxes on target: 1/2 | Total distance: 1
    Children: [3(v=116, avg=0.897), 2(v=57, avg=0.754), 1(v=27, avg=0.556)]
  Move 10: action=1, total_visits=200
    State: Step 9/200 | Boxes on target: 1/2 | Total distance: 1
    Children: [3(v=49, avg=0.816), 2(v=31, avg=0.710), 1(v=120, avg=1.000)]

Final state:
Step 10/200 | Boxes on target: 2/2 | Total distance: 0
#######
#     #
#   @ #
# * * #
#######

------------------------------------------------------------
TASK — ANALYSIS ONLY (no code)
------------------------------------------------------------
Carefully study the game rules, the current 'simulation'
heuristic code, and the gameplay traces above.

Produce a focused analysis with these sections:

1. KEY WEAKNESSES
   What are the main problems causing poor play?
   Cite specific move numbers, Q-value patterns, or state
   observations as evidence. Be specific. Rank by impact.

2. ROOT CAUSE
   WHY does the current code produce this behaviour?
   Point to specific logic or missing logic in the code.

3. PROPOSED APPROACH
   Choose one of these strategies:

   A) INCREMENTAL (~70% of cases): Describe targeted
      modifications to the 'simulation' function
      that address the top weakness(es). Build on and
      extend the current code.

   B) RESTRUCTURE (~30% of cases): If the current approach
      is fundamentally limited, describe a different
      strategy. Explain why incremental changes won't
      suffice. Keep proven components that work.

   State which strategy (A or B) you recommend and why.

Keep your analysis under 500 words. Do NOT write code.

---

## Response

**1. KEY WEAKNESSES**  
| Rank | Symptom (as seen in the trace) | Why it hurts performance |
|------|--------------------------------|--------------------------|
| **1** | *Heuristic plateau*: From move 1‑5 the “total distance” stays at **2** and the “fraction on target” at **0**. The simulation therefore cannot distinguish any of the many legal pushes; the UCB1 scores are driven almost entirely by random visit noise (e.g., child 0 at step 2 has avg 0.000). | The distance term is too coarse – each box is measured to its *nearest* target, ignoring that two boxes may compete for the same target or that a box may be geometrically near a target but blocked by walls. This yields a flat reward landscape early in the search, causing wasteful exploration. |
| **2** | *Very limited dead‑lock detection*: Only the `_is_corner_deadlock` test is used. In the trace the algorithm never prunes moves that would push a box against a wall with no target along that wall, or create “freeze” deadlocks involving two boxes. | Such moves look harmless to the heuristic (distance may even improve) and are therefore kept as “safe”, inflating the search space and lowering the signal‑to‑noise ratio of roll‑outs. |
| **3** | *No push‑efficiency pressure*: All roll‑outs are evaluated solely on the static reward; there is no penalty for the number of pushes taken so far. Consequently the policy may wander with many unnecessary pushes before finally solving, which is reflected in the relatively modest Q‑values (≈0.6‑0.8) even at the final steps. | Sokoban solutions are usually short; encouraging fewer pushes makes the rollout policy more goal‑directed and improves the value estimate for promising branches. |
| **4** | *Fixed ε‑greedy*: ε = 0.2 is applied uniformly throughout the rollout. Early in the search the heuristic is uninformative, so a high exploration rate wastes visits; later when the heuristic is good it still forces needless randomness. | Inefficient use of the simulation budget and noisy value estimates. |

**2. ROOT CAUSE**  
The current `default_simulation` builds its roll‑out policy on two pillars:

1. **_heuristic** – a weighted sum of *fraction on target* (70 %) and a *naïve Manhattan distance* (30 %). The distance term uses `state.total_box_distance()`, which assigns each box to its *closest* target independently, ignoring contention and reachability. This produces a flat landscape when no box is on a target (as in moves 1‑5).  

2. **_is_corner_deadlock** – only detects deadlocks formed by two orthogonal walls. More subtle patterns (wall‑line deadlocks, frozen boxes, “2‑box” deadlocks) are never filtered, so the simulation keeps “safe” actions that are in fact hopeless.

Because the rollout never penalises long push sequences, it treats every safe continuation equally, letting the tree rely on random fluctuations in visit counts to break ties. The result is the noisy Q‑values observed in the trace.

**3. PROPOSED APPROACH – Strategy A (Incremental)**  
We can substantially lift the rollout quality while keeping the existing framework (clone‑apply‑evaluate loop). The following targeted extensions address the three highest‑impact weaknesses:

| Modification | What it adds | Expected effect |
|--------------|--------------|-----------------|
| **a) Better distance estimate – “matching distance”** | Compute a *minimum‑cost bipartite matching* between current boxes and all targets (Hungarian algorithm) and use that total cost as `matched_dist`. Replace the simple `total_box_distance()` term with `matched_dist`. | Captures competition for the same target and yields a monotonic, more discriminative distance signal. Early moves that bring a box closer to a *different* target will be rewarded, breaking the plateau. |
| **b) Simple wall‑line dead‑lock test** | Add `_is_wall_line_deadlock(box, walls, targets)` that checks: if a box is adjacent to a wall in direction *d* and there is **no** target in that same row/column beyond the wall, the box is irrevocably stuck when pushed against that wall. Apply this test to every box after a simulated move and treat the action as dead‑locked (exclude from `safe_actions`). | Prunes a large class of obvious futile pushes (e.g., pushing a box up against a vertical wall when no target is above), reducing the branching factor and focusing visits on promising lines. |
| **c) Push‑count penalty** | Track `pushes = sim_state.steps - state.steps` inside the rollout. When computing the heuristic value, subtract `λ * pushes / max_steps` (λ ≈ 0.05) from the weighted sum. | Gives a modest bias toward shorter solutions, encouraging the rollout to prefer actions that achieve the same heuristic improvement with fewer pushes. |
| **d) Adaptive ε** | Replace the constant ε with a schedule: `epsilon = max(0.2 * (1 - depth / max_depth), 0.05)`. Early in the rollout (when the heuristic is still noisy) we explore more; later we almost always exploit the best‑scoring actions. | Uses exploration where it matters and lets the refined heuristic dominate as the rollout approaches a promising state, reducing noise in the final value estimate. |

All four tweaks live inside the existing `default_simulation` loop: after generating `child`, run the extra dead‑lock check, compute the matching distance (still O(N³) but with N ≤ #boxes, tiny for typical levels), apply the push penalty, and use the adaptive ε when selecting among `safe_actions`. No structural changes to MCTS are required, preserving the proven selection/expansion/back‑propagation pipeline.

**Why incremental is enough:** The current rollout already performs cloning, deadlock pruning, and heuristic evaluation. By tightening the heuristic’s distance component, expanding dead‑lock detection to the low‑cost wall‑line case, and adding a light push‑penalty together with a depth‑aware ε, we directly attack the three dominant sources of noise without rewriting the whole simulation engine. This yields sharper Q‑value differentials, faster convergence, and more stable play, especially on harder levels where early progress is hard to signal.
