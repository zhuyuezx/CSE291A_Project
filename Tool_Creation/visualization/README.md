# CSE291A Winter 2026 Project

## How to run trajectory visualization (`plot_game_trajectory.py`)

From the project root:

```bash
cd Tool_Creation
python -m visualization.plot_game_trajectory --game sokoban --level level4
```

**Arguments:**

| Option | Meaning |
|--------|--------|
| `--game` | Game to run: `sokoban` or `rush_hour`. If omitted, uses the active game from `MCTS_tools/hyperparams/default_hyperparams.py`. |
| `--level` | Level name (e.g. `level4`, `level5`, `easy1`, `easy2`). If omitted, uses the first level for the chosen game. |
| `--iterations` | MCTS iterations per move (default from config). |
| `--max-rollout-depth` | Max simulation depth (default from config). |
| `--max-steps` | Max game steps (default 200). |
| `--no-compare` | Skip baseline–vs–optimized comparison (only optimized trajectory is generated). |
| `--no-gif` | Do not export GIFs; only PNG timelines are saved. |
| `--from-records` | **Use latest trace JSONs** from `mcts/records` for baseline and optimized instead of rerunning MCTS. Baseline = trace whose tools are all `default_*.py`; optimized = trace with any `(set programmatically)` or non-default tool. If a matching trace is missing, that side falls back to a live MCTS run. |
| `--records-dir` | Override records directory for `--from-records` (default: `Tool_Creation/mcts/records`). |
| `--show` | Open plot windows interactively in addition to saving files. |

**Output (saved under `Tool_Creation/visualization/output/trajectory/`):**

- `{game}_trajectory_optimized_{level}.png` and `.gif`
- With comparison: `{game}_trajectory_compare_{level}.png` and `.gif` (baseline vs optimized)

**Examples:**

```bash
# Sokoban level4, run MCTS at runtime (baseline + optimized)
python -m visualization.plot_game_trajectory --game sokoban --level level4

# Same, but load baseline/optimized from latest JSONs in mcts/records (no MCTS run if both traces exist)
python -m visualization.plot_game_trajectory --game sokoban --level level4 --from-records

# Rush Hour easy2, no comparison, no GIF
python -m visualization.plot_game_trajectory --game rush_hour --level easy2 --no-compare --no-gif
```

---

## Baseline vs LLM-optimized performance

### Baseline vs optimized trajectory — Sokoban level 3 (`sokoban_trajectory_compare_level3`)

Compares baseline and optimized behavior on level3 side-by-side along sampled timesteps.

| Comparison PNG | Comparison GIF |
|----------------|----------------|
| ![Sokoban level 3 comparison](Tool_Creation/visualization/output/trajectory/sokoban_trajectory_compare_level3.png) | ![Sokoban level 3 comparison GIF](Tool_Creation/visualization/output/trajectory/sokoban_trajectory_compare_level3.gif) |

### Baseline vs optimized trajectory — Rush Hour (`rush_hour_trajectory_compare_easy2`)

Compares baseline and optimized MCTS on Rush Hour (easy2) side-by-side.

| Comparison PNG | Comparison GIF |
|----------------|----------------|
| ![Rush Hour easy2 comparison](Tool_Creation/visualization/output/trajectory/rush_hour_trajectory_compare_easy2.png) | ![Rush Hour easy2 comparison GIF](Tool_Creation/visualization/output/trajectory/rush_hour_trajectory_compare_easy2.gif) |

### Sokoban solve rate

Compares the solve rate (%) of baseline MCTS vs LLM-optimized MCTS across Sokoban levels 1–10. Both policies reach 100% on level1 and level2; on level3, level4, level5, and level9 the optimized policy reaches 100% while the baseline is lower (e.g. ~67% on level3/4/5, 0% on level9).

![Sokoban solve rate](Tool_Creation/visualization/output/sokoban_solve_rate.png)

### Sokoban steps to solve

Average steps to solve per level. The LLM-optimized policy uses fewer steps than the baseline, especially on harder levels (e.g. level3: ~34 vs ~145 steps, 4.3× faster; level9: ~56 vs ~200 steps, 3.5× faster).

![Sokoban steps](Tool_Creation/visualization/output/sokoban_steps.png)

### Rush Hour solve rate

Compares baseline vs LLM-optimized solve rate (%) across Rush Hour levels (easy1–3, medium1–2, hard1–3). Both policies achieve 100% on easy1, easy2, and easy3; neither solves medium or hard levels in this experiment (0% for those levels).

![Rush Hour solve rate](Tool_Creation/visualization/output/rush_hour_solve_rate.png)

### Rush Hour steps to solve

Average steps to solve on the easy levels where both policies solve. The optimized policy uses far fewer steps (e.g. easy1: ~2 vs ~41 steps, 20.3× faster; easy2: ~3 vs ~11, 3.8× faster; easy3: ~2 vs ~12, 6.2× faster).

![Rush Hour steps](Tool_Creation/visualization/output/rush_hour_steps.png)

---

## Level 3 Visualization

### 1) Root action statistics (`mcts_root_level3`)

Shows the action distribution from the **root state at t=0** for baseline vs optimized MCTS (visits and avg value).

![MCTS root level3](Tool_Creation/visualization/output/mcts_root/mcts_root_level3.png)

### 2) Tree snapshot (`mcts_tree_level3`)

Shows the MCTS tree for level3 from the same starting state, with the chosen path highlighted and deeper continuation indicated by `...`.

![MCTS tree level3](Tool_Creation/visualization/output/mcts_tree/mcts_tree_level3.png)


### 3) Optimized trajectory (`sokoban_trajectory_optimized_level3`)

Shows how optimized MCTS actually plays level3 over time (sampled timesteps from start to terminal).

![Optimized trajectory level3](Tool_Creation/visualization/output/trajectory/sokoban_trajectory_optimized_level3.png)

### 4) Principal variation (`principal_variation_level3`)

Shows the best path (top 10) selected by MCTS from the root by repeatedly following the most-visited child.

![Principal variation level3](Tool_Creation/visualization/output/principal_variation/principal_variation_level3.png)
