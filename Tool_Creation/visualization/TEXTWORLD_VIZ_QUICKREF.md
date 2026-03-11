# TextWorld Trajectory Visualization — Quick Reference

You asked to keep **only trajectory GIF visualization** and keep all TextWorld-viz code in one place:

- `Tool_Creation/visualization/textworld/`

## 30-second usage

From the project root:

```bash
python Tool_Creation/visualization/textworld/demo.py
```

This generates:
- `Tool_Creation/visualization/output/textworld/trajectory_baseline.gif`
- `Tool_Creation/visualization/output/textworld/trajectory_high_iter.gif`
- `Tool_Creation/visualization/output/textworld/trajectory_compare.gif`

## Minimal API (copy/paste)

```python
from visualization.textworld import TrajectoryVisualizer

trajectory = [
    {"observation": {"room": 0, "description": "..."}, "action": None, "reward": 0.0, "done": False},
    {"observation": {"room": 1, "description": "..."}, "action": "go east", "reward": 0.0, "done": False},
]

game_state = {
    "world_graph": {0: [1], 1: [0, 2], 2: [1]},
    # optional: "coords": {0: (0, 0), 1: (1, 0), 2: (2, 0)}
}

viz = TrajectoryVisualizer(trajectory, game_state=game_state)
viz.visualize("episode.gif", fps=2)
```

## Troubleshooting

- **World Map panel blank**: ensure `game_state` contains `world_graph` (dict of room -> neighbors) OR an object/dict with `graph`.
- **GIF not generated**: install Pillow: `pip install Pillow`
