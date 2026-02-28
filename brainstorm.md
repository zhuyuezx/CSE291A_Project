# Brainstorm: Self-Evolving Game Agent with Tool Integration

## Q1: What is a "tool" in this system?

**Answer:** A tool is a Python function that plugs into specific MCTS phases to improve search quality. Tools span ALL four MCTS phases plus a new category of meta-actions:

1. **State evaluation functions** — Score a board state (material count, path distance, piece mobility). Used to replace or augment random rollout termination.
2. **Action filters/pruners** — Reduce the action space during expansion (e.g., "only consider walls near pawns"). Critical for games like Quoridor where most wall placements are useless.
3. **Rollout policies** — Bias simulation toward smarter play (e.g., "with 70% probability, follow shortest path"). Replace purely random playouts.
4. **Selection tweaks** — Modify UCT scoring (e.g., add domain-specific prior probabilities to PUCT formula).
5. **Reward shaping** — Adjust backpropagation values based on intermediate board features.
6. **Macro actions / compound moves** — Higher-level strategic actions (e.g., "build a wall corridor", "fork attack") that decompose into sequences of primitive moves. The RL agent can choose these alongside primitive actions, effectively expanding its action vocabulary with strategic concepts.

**Key insight from the proposal:** The Quoridor C++ implementation already has hand-crafted versions of tools #1-3 (BFS shortest path evaluation, probable-walls pruning, 70/30 greedy rollout). The project goal is to have the LLM *generate* these automatically AND discover new ones that humans wouldn't think of.

---

## Q2: How are tools represented?

**Answer:** Python functions operating on OpenSpiel game states. Each tool follows a standard interface:

```python
# Tool signature examples:
def evaluate_state(state: pyspiel.State) -> float:
    """Return a heuristic score for the current player."""

def filter_actions(state: pyspiel.State, legal_actions: list[int]) -> list[int]:
    """Return a subset of legal_actions worth exploring."""

def rollout_policy(state: pyspiel.State, legal_actions: list[int]) -> int:
    """Choose an action during simulation (biased rollout)."""

def macro_action(state: pyspiel.State) -> list[int]:
    """Return a sequence of primitive actions to execute."""
```

MCTS runs entirely in Python using OpenSpiel's API. This is slower than C++ but allows dynamic tool loading and LLM-generated code.

---

## Q3: How are tools created?

**Answer:** Yunjue-inspired custom pipeline (not full Yunjue dependency):

1. **Game Analysis** — LLM receives: game rules description, current tool pool, MCTS game traces (states, actions, outcomes), performance metrics
2. **Tool Specification** — LLM proposes a tool: name, type (eval/filter/rollout/macro), description, expected input/output
3. **Code Generation** — LLM writes the Python function body
4. **Validation** — Run the tool against sample game states, check it doesn't crash, verify outputs are in expected range
5. **Integration Test** — Run a small batch of MCTS games with the new tool, compare win rate to baseline
6. **Promotion** — If the tool improves performance, add to the tool pool. Otherwise discard.

This borrows Yunjue's patterns (spec -> codegen -> test -> promote) without requiring the full framework.

---

## Q4: When do tools get created/refined?

**Answer:** On performance plateau. The system monitors win rate during training. When improvement stalls (e.g., win rate hasn't improved by >X% over the last N game batches), the system triggers a tool evolution cycle:

1. Collect recent game traces (especially losses and close games)
2. Feed traces to LLM with prompt: "Analyze why the agent is losing. What heuristic or strategy could help?"
3. LLM generates candidate tool(s)
4. Validate and test candidates
5. Keep winners, discard losers

This is the most cost-efficient approach — only calls the LLM when there's a clear signal that current tools aren't enough.

---

## Q5: What is the game progression?

**Answer:** Connect Four -> Quoridor -> Chess

- **Connect Four**: Simple grid game, good for sanity-checking tool infrastructure. Tools should discover: column preference heuristics, threat detection, win-in-N evaluation.
- **Quoridor**: Medium complexity with wall mechanics. Tools should discover: shortest-path evaluation, wall placement heuristics, corridor-blocking strategies. We have ground truth from the C++ hand-crafted heuristics.
- **Chess**: Highly complex. Tests whether tools from simpler games (piece mobility, path analysis, material evaluation) transfer. We don't expect to match Stockfish, but tools should improve MCTS over vanilla at low sim counts.

---

## Q6: What LLM model to use?

**Answer:** Flexible/configurable system (like Yunjue's conf.yaml). Design supports model swapping:
- **Development**: Local Qwen3-7B/14B or DeepSeek API (cheap)
- **Final experiments**: GPT-5 or equivalent (best code quality)
- **Evaluation/analysis**: Cheaper models fine for trace analysis

---

## Q7: How do we evaluate success?

**Answer:** Three-pronged evaluation:

1. **Win rate vs baselines** — MCTS+tools vs vanilla MCTS vs random agent at fixed simulation budgets (100, 500, 1000, 5000 sims). Table format per game.
2. **Sample efficiency curves** — Plot win rate vs number of MCTS simulations. Show that tools shift the curve LEFT (same performance with fewer simulations). This is the core thesis.
3. **Cross-game transfer speed** — Train tools on Connect Four, apply to Quoridor. Compare learning speed vs starting from scratch. Then train on both, apply to Chess.

---

## Q8: What are the key technical challenges?

### Challenge 1: Tool-MCTS Integration Interface
How exactly do multiple tools compose? If we have 3 action filters and 2 rollout policies, how does MCTS use them?

**Proposed solution:** Tool registry with phase-based dispatch:
- Expansion phase: chain all action filters (intersection of allowed actions)
- Simulation phase: weighted ensemble of rollout policies
- Evaluation: weighted average of evaluation functions
- Selection: tools provide prior probabilities for PUCT

### Challenge 2: Tool Quality Assurance
LLM-generated code might be buggy, slow, or counterproductive.

**Proposed solution:**
- Syntactic validation (AST parse, type checking)
- Runtime validation (run on 100 random states, check no crashes, outputs in range)
- Performance validation (A/B test: 50 games with tool vs 50 without, must improve or break even)
- Timeout enforcement (tools that take >X ms per call are rejected)

### Challenge 3: Tool Generalization Across Games
A "shortest path" tool for Quoridor doesn't directly apply to Chess.

**Proposed solution:** Tools should be written at an abstraction level that maps to OpenSpiel's generic API:
- Instead of "BFS on Quoridor grid" -> "evaluate how close current player is to winning"
- LLM is prompted to write game-agnostic tools when possible
- Some tools are inherently game-specific (e.g., wall placement) — that's OK, the system should learn which transfer and which don't

### Challenge 4: Macro Action Representation
How do macro actions interact with MCTS tree structure?

**Proposed solution:** Options framework (from hierarchical RL):
- A macro action is a sequence of primitive actions with a termination condition
- During MCTS expansion, macro actions are treated as additional children alongside primitive actions
- Simulation of a macro action executes the full sequence, then continues with normal rollout
- Backpropagation treats the macro as a single decision node

### Challenge 5: Performance Plateau Detection
How to reliably detect when to trigger tool evolution?

**Proposed solution:** Rolling window statistics:
- Track win rate over last N games (e.g., N=50)
- If win rate hasn't improved by >2% over 2 consecutive windows, trigger evolution
- Also trigger if win rate drops by >5% (tool regression detection)

---

## Q9: What does the overall system architecture look like?

Three main subsystems:

### Subsystem A: MCTS Engine (with tool hooks)
- Python MCTS using OpenSpiel API
- Each MCTS phase has a "tool slot" that accepts registered tools
- Tool registry manages active tools per game
- Configurable: can run with no tools (vanilla), hand-picked tools, or full tool pool

### Subsystem B: Tool Creation Pipeline (LLM-powered)
- Yunjue-inspired: analyze -> spec -> codegen -> validate -> test -> promote
- Triggered on performance plateau
- Inputs: game traces, current tool pool, performance metrics
- Outputs: new Python tool functions
- LLM model is configurable

### Subsystem C: Cross-Game Tool Manager
- Maintains a global tool pool across games
- Tracks per-tool metadata: which games it was created for, which games it helped, performance impact
- When starting a new game: loads all tools, tests each one, keeps the beneficial ones
- After training on a new game: merges new tools into global pool (Yunjue's absorption mechanism)

### Data Flow:
```
Game (OpenSpiel) -> MCTS Engine (with tools) -> Game Traces
                                                    |
                                                    v
                                          Performance Monitor
                                                    |
                                          (plateau detected?)
                                                    |
                                                    v
                                        Tool Creation Pipeline (LLM)
                                                    |
                                                    v
                                          New Tool Candidates
                                                    |
                                          (validation & testing)
                                                    |
                                                    v
                                          Tool Pool (updated)
                                                    |
                                                    v
                                          MCTS Engine (with new tools)
```

---

## Q10: What's the minimum viable product (MVP)?

**Phase 1 (MVP):** MCTS + tool hooks + manual tool creation
- Python MCTS for Connect Four via OpenSpiel
- Tool interface defined (eval, filter, rollout, macro)
- Hand-write 2-3 tools to prove the interface works
- Show that hand-written tools beat vanilla MCTS

**Phase 2:** LLM tool generation
- Build the tool creation pipeline
- LLM generates tools for Connect Four
- Validate that LLM-generated tools match or approach hand-written ones

**Phase 3:** Cross-game transfer
- Apply Connect Four tools to Quoridor
- Train new Quoridor-specific tools
- Merge tool pools
- Apply to Chess

**Phase 4:** Full evaluation
- Win rate tables
- Sample efficiency curves
- Cross-game transfer analysis
