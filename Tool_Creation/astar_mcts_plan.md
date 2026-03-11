# A*-MCTS Implementation Plan

Map every structural component of the `sokoban_astar.ipynb` A* algorithm onto
the four MCTS heuristic phase files so that each MCTS iteration executes one
logical A* step.

---

## 1. A* Algorithm Decomposition

```
A* one iteration:
  pq:      priority queue keyed by f = g + h
  visited: dict[state_key → best_g_seen]

  Step A — POP:      f, g, key, state = heapq.heappop(pq)
  Step B — EXPAND:   for action in legal_actions(state):
                         next_state  = apply(action)
                         next_key    = state_key(next_state)
                         next_g      = g + 1
                         if deadlocked(next_state): skip
                         if visited[next_key] <= next_g: skip   ← visited check
                         visited[next_key] = next_g
                         h = heuristic(next_state)
                         heappush(pq, (next_g + h, next_g, ...))
  Step C — EVALUATE: h = player_dist + total_box_distance        ← heuristic
  Step D — RECORD:   visited[key] = g                            ← g-update
```

**Target heuristic** (from notebook):
```
h(state) = total_box_distance(state)
           + min(manhattan(player, box) for box in unplaced_boxes)
```

---

## 2. Phase Mapping

| A* component | MCTS phase | What changes |
|---|---|---|
| `heapq.heappop(pq)` — best-f node | **Selection** | Replace UCB1 with min-f = min(g+h) child walk |
| `visited` check + `heappush` for each successor | **Expansion** | Sort untried actions by h; skip visited states with equal/worse g |
| `h = heuristic(next_state)` | **Simulation** | Return shaped reward from h directly; no random rollout |
| `visited[key] = g` propagated upward | **Backpropagation** | Update shared visited dict with best g per state_key |

Each MCTS iteration = one A* pop-and-expand step.

---

## 3. Shared State Module

**File:** `MCTS_tools/shared/astar_globals.py`

All four phase files import from here. Module-level dicts persist across
phases within one MCTS search.

```python
# Shared A* state — one logical search (one call to engine.search())
_visited: dict[str, int] = {}   # state_key → best g-score seen so far
_root_key: str | None   = None  # detects when a new MCTS search begins

def reset(root_state_key: str) -> None:
    """Call at the start of every new MCTS search (new root state)."""
    global _visited, _root_key
    _visited  = {root_state_key: 0}
    _root_key = root_state_key

def get_visited() -> dict[str, int]:
    return _visited

def node_depth(node) -> int:
    """g-score = depth of node from root (walk parent chain)."""
    d = 0
    while node.parent is not None:
        node = node.parent
        d += 1
    return d

def h_sokoban(state) -> int:
    """
    A* heuristic: total_box_distance + player_to_closest_unplaced_box.
    Returns 0 when solved.
    """
    unplaced = [b for b in state.boxes if b not in state.targets]
    if not unplaced:
        return 0
    box_dist   = sum(min(abs(b[0]-t[0]) + abs(b[1]-t[1])
                         for t in state.targets)
                     for b in unplaced)
    pr, pc     = state.player
    player_dist = min(abs(pr-b[0]) + abs(pc-b[1]) for b in unplaced)
    return box_dist + player_dist
```

**Reset trigger**: Selection detects a new search when
`root.state.state_key() != _root_key` and calls `reset()`.

---

## 4. Phase Implementations

### Phase 1 — Selection  (`MCTS_tools/selection/selection.py`)

**A* role**: `heapq.heappop(pq)` — walk to the node with the lowest f-score.

```
Algorithm:
  1. If root key ≠ _root_key → call reset(root.state.state_key())
  2. Walk down tree:
       while node is fully-expanded and not terminal:
           g_child = node_depth(child) + 1
           h_child = h_sokoban(child.state)
           f_child = g_child + h_child
           select child = argmin(f_child)        ← A* pop, not UCB1
  3. Return node (to be expanded or simulated)
```

**Why min-f instead of max-UCB1**: A* always expands the cheapest-path node.
Replacing UCB1 with f-score makes MCTS walk toward the most promising
(lowest-cost) frontier, mirroring the pq-pop.

**Tie-breaking**: When multiple children share the same f-score, break ties
by fewest visits (encourages exploration of equally-promising branches).

---

### Phase 2 — Expansion  (`MCTS_tools/expansion/expansion.py`)

**A* role**: Loop over successors → visited check → `heapq.heappush(pq, ...)`.

```
Algorithm:
  1. Compute g_new = node_depth(node) + 1
  2. For each untried action, speculatively apply to get next_state:
       a. If next_state._is_deadlocked()          → discard action
       b. key = next_state.state_key()
       c. If visited[key] exists and visited[key] <= g_new → discard action
          (A* visited check: we already found this state on a shorter path)
  3. From surviving actions, pick action with lowest h_sokoban(next_state)
     (mirrors pushing to pq ordered by f = g + h)
  4. Apply chosen action, create MCTSNode child, record visited[key] = g_new
  5. Remove chosen action from node._untried_actions
```

**Key invariant maintained**: `visited[key] = g_new` is written here, matching
A*'s `visited[next_key] = next_g` before the heappush.

---

### Phase 3 — Simulation  (`MCTS_tools/simulation/simulation.py`)

**A* role**: Heuristic evaluation at the frontier — A* has no rollout; the
heuristic value IS the estimate.

```
Algorithm:
  1. If state is terminal: return state.returns()[perspective_player]
  2. h = h_sokoban(state)
  3. g = node depth (computed from simulation call context or state.steps)
  4. Return shaped reward:
       if h == 0:  1.0          (solved)
       else:       1.0 / (1.0 + h)
```

**No rollout**: Replacing the random rollout with a direct heuristic evaluation
is precisely what A* does — it never "plays out" a random game. The heuristic
directly estimates distance-to-goal.

**Reward range**: Always in (0, 1]. Solved state = 1.0. Larger h → smaller
reward → MCTS backpropagates lower value for distant states.

---

### Phase 4 — Backpropagation  (`MCTS_tools/backpropagation/backpropagation.py`)

**A* role**: `visited[key] = g` — ensure every ancestor records the best
g-score seen on the path through it, so expansion can prune stale paths.

```
Algorithm:
  1. Standard MCTS backprop: walk from leaf to root,
     node.visits += 1, node.value += reward
  2. Additionally, for each node on the walk:
       key = node.state.state_key()
       g   = node_depth(node)
       if key not in visited or visited[key] > g:
           visited[key] = g           ← A* g-score maintenance
```

**Why this matters**: Expansion (Phase 2) checks `visited[key] <= g_new` to
skip states already reached cheaply. Backprop keeps visited accurate even when
MCTS re-traverses paths, ensuring future expansion steps never waste iterations
on states already found at a shorter depth.

---

## 5. Data Flow Across One MCTS Iteration

```
engine._search_internal():
  for _ in range(iterations):

    ┌─ SELECTION ──────────────────────────────────────────────────┐
    │  detect new search → reset _visited                          │
    │  walk tree: at each level pick child with min f = g + h      │
    │  stop at node with untried actions                           │
    └──────────────────────────────────────────────────────────────┘
                              │  node (has untried actions)
    ┌─ EXPANSION ─────────────▼────────────────────────────────────┐
    │  filter untried actions: deadlock + visited-g check          │
    │  pick action with min h(next_state)                          │
    │  create child node                                           │
    │  write visited[child.state_key()] = g_new                    │
    └──────────────────────────────────────────────────────────────┘
                              │  child node (leaf)
    ┌─ SIMULATION ────────────▼────────────────────────────────────┐
    │  reward = 1.0 / (1.0 + h_sokoban(child.state))              │
    │  (no rollout — direct heuristic evaluation, like A*)         │
    └──────────────────────────────────────────────────────────────┘
                              │  reward ∈ (0, 1]
    ┌─ BACKPROPAGATION ───────▼────────────────────────────────────┐
    │  walk leaf → root:                                           │
    │    node.visits += 1                                          │
    │    node.value  += reward                                     │
    │    visited[node.state_key()] = min(g, existing)  ← A* sync  │
    └──────────────────────────────────────────────────────────────┘
```

---

## 6. Files to Create / Modify

```
MCTS_tools/
├── shared/
│   └── astar_globals.py         NEW — visited dict, node_depth, h_sokoban
│                                       (helper module, not a phase file)
├── selection/
│   └── selection.py             NEW — min-f walk (replaces UCB1)
│                                       picked up automatically by _load_installed_tools
├── expansion/
│   └── expansion.py             OVERWRITE — visited check + min-h action pick
├── simulation/
│   └── simulation.py            OVERWRITE — direct heuristic reward (no rollout)
└── backpropagation/
    └── backpropagation.py       NEW — standard backprop + visited g-sync
```

No changes needed to `mcts/node.py`, `mcts/mcts_engine.py`, or game files.
The existing `MCTSNode.__slots__` and engine loop are sufficient.

**Automatic pickup**: `_load_installed_tools` scans each phase directory for
the newest non-`default_*` file. Using `<phase>.py` as the filename means:
- `eval_heuristics.py` picks them up with no flags needed
- The LLM optimizer reads and overwrites the same files each run
- No manual `load_tool()` calls required

Existing files that will be overwritten:
- `expansion/expansion.py`   (currently LLM-generated Mar 8)
- `simulation/simulation.py` (currently player_prox heuristic Mar 10)

Both are preserved in `checkpoints/checkpoint_20260310_183842/` before
overwriting.

---

## 7. Design Decisions and Trade-offs

| Decision | Choice | Rationale |
|---|---|---|
| visited scope | Module-level in `astar_globals.py` | All 4 phase files share one dict without touching MCTSNode or engine |
| Reset trigger | Selection detects root key change | Selection always runs first; natural place to detect new search |
| g-score | `node_depth()` = parent-chain walk | Exact depth without adding fields to MCTSNode |
| h-score | `player_dist + total_box_distance` | Matches A* notebook exactly; admissible for Sokoban |
| Reward shape | `1.0 / (1.0 + h)` | Monotone in [0,1]; solved=1.0; compatible with existing backprop sign convention |
| Tie-breaking in selection | fewest visits | Encourages coverage of equally-promising frontier nodes |
| Visited check strictness | `visited[key] <= g_new` (skip equal) | Matches A*: no benefit re-expanding a state at the same depth |
| MCTS iterations meaning | Each iteration = one A* expansion | Allows comparing iterations vs A* node expansions directly |

---

## 8. Expected Behavior vs Pure A*

Pure A* (notebook) solves all 10 levels optimally because it exhaustively
explores the state graph. This A*-MCTS will converge to the same solution
given enough iterations, but with two differences:

1. **visit counts still accumulate** — `most_visited_child()` picks the final
   action, so after many iterations the most-expanded A* path wins.
2. **exploration term** — tie-breaking by fewest visits adds slight exploration
   that helps avoid getting stuck in very deep subtrees on harder levels.

The visited dict ensures no state is ever re-expanded on a shorter path, which
is the core correctness guarantee of A*.
