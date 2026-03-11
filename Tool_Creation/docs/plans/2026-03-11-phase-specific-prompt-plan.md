# Plan: Per-Phase System Prompt for LLM Guidance

## Goal

Inject phase-specific descriptions and optimization goals into the LLM prompt engine so the model receives tailored instructions depending on which MCTS phase is being optimized. This improves proposal quality by aligning the LLM with the phase's role, constraints, and best practices.

## Current State

- **PromptBuilder** (`LLM/prompt_builder.py`) builds prompts with:
  - `_build_system_section()` — generic 70/30 rule for all phases; only `target_phase` name is mentioned
  - `_build_task_section()` — generic task text; the "How the phase works" block is simulation-centric (talks about float reward, leaf node, etc.) and incorrect for selection/expansion/backpropagation
  - `_build_analysis_task_section()` — generic
  - `_build_critique_task_section()` — generic

- **Source material**: `docs/heuristictips.md` and `docs/MCTS_phase_interaction.md` contain the phase-specific guidance but are not wired into the prompt builder.

## Plan

### 1. Add Phase-Specific Content Module

Create `LLM/phase_prompts.py` (or add to `prompt_builder.py`) with:

- A dict `PHASE_DESCRIPTIONS: dict[str, str]` keyed by phase name
- Each value: a compact block of text (3–8 lines) covering:
  - **What this phase does** (one sentence)
  - **Optimization goal** (what to improve)
  - **Key constraints** (cost, scope, what to avoid)
  - **Good heuristic patterns** (2–3 bullets)

- Keep content game-agnostic where possible; Sokoban examples can be in a separate optional section or in heuristictips.md.

### 2. Insertion Point

Inject the phase-specific block into `_build_system_section()` **after** the role line and **before** the 70/30 rule, for phases `selection`, `expansion`, `simulation`, `backpropagation`. Skip for `hyperparams` (keep existing hyperparams logic).

Structure:

```
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function for the game 'X' (phase: Y).

[PHASE-SPECIFIC BLOCK — NEW]
  What this phase does: ...
  Optimization goal: ...
  Constraints: ...
  Good patterns: ...

APPROACH — 70 / 30 RULE:
  ...
```

### 3. Task Section Updates

- Fix `_build_task_section()` so the "How the phase works" block is **phase-specific** (current text is simulation-only and wrong for other phases):
  - **Selection**: receives `(root, exploration_weight)`, returns chosen child index; adjusts UCB scores or selection policy
  - **Expansion**: receives `(node)`, returns action to expand; orders/filters untried actions
  - **Simulation**: receives `(state, perspective_player, max_depth)`, returns float reward (current text is correct)
  - **Backpropagation**: receives `(node, reward)`, returns None; updates node.value and node.visits up the tree
- Add `_build_phase_mechanics_block()` that returns phase-appropriate "How it works" text.

### 4. Files to Modify

| File | Change |
|------|--------|
| `LLM/prompt_builder.py` | Add `_build_phase_guidance_section()`; call it from `_build_system_section()`; fix phase-specific task text |
| `LLM/phase_prompts.py` (new) | Define `PHASE_DESCRIPTIONS` dict |
| `docs/MCTS_phase_interaction.md` | Optional: add a brief note that phase prompts are used by the LLM |

### 5. Testing

- Unit test: `PromptBuilder(game="sokoban", target_phase="selection")` → system section contains selection-specific text
- Unit test: `PromptBuilder(game="sokoban", target_phase="simulation")` → system section contains simulation-specific text
- Sanity check: run a full pipeline for one phase and confirm output is coherent

## Draft Content: Per-Phase System Prompt Blocks

These blocks are inserted into the system section when building prompts for each phase.

---

### Selection

```
PHASE: selection
  • What it does: Walks down the tree from root to a leaf. Chooses which existing branch to explore next. Must balance exploration (UCB) and exploitation.
  • Optimization goal: Improve how we RANK existing nodes — favor promising branches, deprioritize dead ends. Your heuristic adjusts node scores used by UCB1.
  • Constraints: Called very often. Keep it CHEAP — no multi-step rollouts, no deep deadlock simulation. Rank nodes, don't simulate.
  • Good patterns: bonus for more boxes on targets, bonus for lower box distance, penalize obvious deadlocks, novelty bonus for under-visited nodes.
  • Avoid: expensive rollout logic, final reward shaping (that belongs in simulation).
```

---

### Expansion

```
PHASE: expansion
  • What it does: Creates new child nodes from a frontier node. Decides which actions to materialize into the tree and in what order.
  • Optimization goal: PRUNE bad actions and ORDER remaining actions so promising ones are tried first. Filter deadlocks before they enter the tree.
  • Constraints: Best place for hard constraints. Order actions; optionally filter some entirely. No rollout policies or value aggregation.
  • Good patterns: reject pushes into non-target corners, reject wall deadlocks, prefer pushes that reduce box distance, deprioritize no-op player movement.
  • Avoid: long rollout policies, reward aggregation, node-value update rules.
```

---

### Simulation

```
PHASE: simulation
  • What it does: Rolls forward from a leaf state to estimate how promising it is. Returns a reward (e.g. 0–1) that flows into backpropagation.
  • Optimization goal: Produce REWARDS that reflect true state quality. Shaped partial progress helps MCTS distinguish good from bad actions.
  • Constraints: Must return a FLOAT. Reward MUST vary across states — flat rewards ≈ random play. Called thousands of times per move — keep it fast.
  • Good patterns: shaped score (boxes on targets, distance improvement), penalize deadlocks/loops/stagnation, prefer pushes over wandering, early termination when stuck.
  • Avoid: tree-level visit balancing, acceptance criteria for tools — this phase only scores rollouts.
```

---

### Backpropagation

```
PHASE: backpropagation
  • What it does: Sends the simulation result back up the visited path. Updates node statistics (visits, value) that selection's UCB1 uses.
  • Optimization goal: Control HOW strongly rollout evidence affects node values. Calibrate depth discount, solved vs partial progress, path length.
  • Constraints: Only aggregates evidence — no move generation, no deadlock pruning, no rollout policy. Must stay coherent with selection's expectations.
  • Good patterns: depth discount so shorter plans dominate, weight solved outcomes above partial progress, reduce credit for noisy weak rollouts.
  • Avoid: move generation, deadlock pruning, rollout action-choice policy.
```

---

## Integration Sketch

In `prompt_builder.py`:

```python
# In _build_system_section(), after the role line:
if self.target_phase in self.VALID_PHASES:
    sections.append(PHASE_DESCRIPTIONS.get(self.target_phase, ""))
# Then append 70/30 rule, etc.
```
