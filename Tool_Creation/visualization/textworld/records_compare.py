#!/usr/bin/env python3
"""
Compare TextWorld trajectories from JSON records (no MCTS run).

Loads saved MCTS record JSONs from `textworld/mcts/records/`, detects newest baseline
vs newest optimized (or uses explicit paths), converts them into the
`TrajectoryVisualizer` format, and exports a side-by-side comparison GIF.

Outputs are written under: Tool_Creation/visualization/output/textworld/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
OUTPUT_DIR = PROJECT_ROOT / "Tool_Creation" / "visualization" / "output" / "textworld"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Tool_Creation"))

from visualization.textworld import TrajectoryVisualizer


@dataclass(frozen=True)
class RecordInfo:
    path: Path
    game: str
    timestamp: str
    tools: Dict[str, str]


_STATE_PART_RE = re.compile(r"([^=|]+)=([^|]*)")


def _parse_state(state_str: str) -> Dict[str, Any]:
    parts: Dict[str, str] = {}
    for m in _STATE_PART_RE.finditer(state_str):
        k, v = m.group(1).strip(), m.group(2)
        parts[k] = v

    def _to_int(x: Any, default: int = 0) -> int:
        try:
            return int(x)
        except Exception:
            return default

    room = _to_int(parts.get("room", 0))
    goal = _to_int(parts.get("goal", -1), default=-1)
    done = bool(_to_int(parts.get("done", 0)))

    inv_raw = str(parts.get("inv", "") or "")
    inventory = [s for s in inv_raw.split(",") if s] if inv_raw else []

    return {
        "raw": state_str,
        "room": room,
        "goal": goal,
        "done": done,
        "inventory": inventory,
        "doors_raw": str(parts.get("doors", "") or ""),
    }


def _infer_num_locations_from_state(state: Dict[str, Any]) -> int:
    doors_raw = state.get("doors_raw", "") or ""
    idxs: List[int] = []
    for token in doors_raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token.split(":")[0])
            idxs.append(idx)
        except Exception:
            continue
    if idxs:
        return max(idxs) + 1
    goal = state.get("goal", -1)
    if isinstance(goal, int) and goal >= 0:
        return goal + 1
    return 1


def _build_branching_world_graph(num_locations: int) -> Dict[int, List[int]]:
    """
    Build the same branching corridor shape used by the live coin visualizer.

    First 6 rooms match `mock_runner`:
        0: 1,4
        1: 0,2
        2: 1,3
        3: 2
        4: 0,5
        5: 4
    Additional rooms (if any) are attached as a tail from room 5.
    """
    base = {
        0: [1, 4],
        1: [0, 2],
        2: [1, 3],
        3: [2],
        4: [0, 5],
        5: [4],
    }

    if num_locations <= 6:
        return {i: list(base.get(i, [])) for i in range(num_locations)}

    g: Dict[int, List[int]] = {i: list(neigh) for i, neigh in base.items()}
    last = 5
    for room in range(6, num_locations):
        g.setdefault(last, []).append(room)
        g[room] = [last]
        last = room
    return g


def _record_to_trajectory(record: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    moves = record.get("moves", []) or []
    if not moves:
        raise ValueError("Record has no moves.")

    first_state = _parse_state(moves[0].get("state_before", ""))
    num_locations = _infer_num_locations_from_state(first_state)

    game_state = {
        "world_graph": _build_branching_world_graph(num_locations),
        "room_descriptions": {i: f"Room {i}" for i in range(num_locations)},
    }

    trajectory: List[Dict[str, Any]] = []

    trajectory.append(
        {
            "observation": {
                "room": first_state["room"],
                "description": f"Room {first_state['room']} (goal {first_state.get('goal', '?')})",
                "inventory": first_state["inventory"],
                "quest_progress": {
                    "goal_room": first_state.get("goal", None),
                    "coin_taken": "coin" in set(first_state["inventory"]),
                },
            },
            "action": None,
            "reward": 0.0,
            "done": False,
            "legal_actions": moves[0].get("legal_actions", []),
        }
    )

    for i, mv in enumerate(moves):
        action = mv.get("action_chosen")
        legal_actions = mv.get("legal_actions", [])

        if i + 1 < len(moves):
            next_state_str = moves[i + 1].get("state_before")
        else:
            next_state_str = record.get("outcome", {}).get("final_state") or mv.get("state_before")

        st = _parse_state(next_state_str or "")
        done = bool(record.get("outcome", {}).get("solved", False)) and (i == len(moves) - 1)

        trajectory.append(
            {
                "observation": {
                    "room": st["room"],
                    "description": f"Room {st['room']} (goal {st.get('goal', '?')})",
                    "inventory": st["inventory"],
                    "quest_progress": {
                        "goal_room": st.get("goal", None),
                        "coin_taken": "coin" in set(st["inventory"]),
                    },
                },
                "action": action,
                "reward": 0.0,
                "done": done,
                "legal_actions": legal_actions,
            }
        )

    return trajectory, game_state


def _is_baseline_tools(tools: Dict[str, str]) -> bool:
    if not tools:
        return False
    for v in tools.values():
        s = str(v)
        if s.strip() == "(set programmatically)":
            return False
        if "default_" not in s:
            return False
    return True


def _list_records(records_dir: Path, game_filter: Optional[str]) -> List[RecordInfo]:
    paths = sorted(records_dir.glob("*.json"))
    out: List[RecordInfo] = []
    for p in paths:
        try:
            data = json.loads(p.read_text())
            meta = data.get("metadata", {}) or {}
            game = str(meta.get("game", ""))
            if game_filter and game_filter not in game:
                continue
            out.append(
                RecordInfo(
                    path=p,
                    game=game,
                    timestamp=str(meta.get("timestamp", "")),
                    tools=dict(meta.get("tools", {}) or {}),
                )
            )
        except Exception:
            continue
    return out


def _pick_newest_pair(records: List[RecordInfo]) -> Tuple[RecordInfo, RecordInfo]:
    if not records:
        raise ValueError("No records found.")

    optimized = sorted([r for r in records if not _is_baseline_tools(r.tools)], key=lambda r: r.timestamp)[-1]
    baseline_same_game = [r for r in records if _is_baseline_tools(r.tools) and r.game == optimized.game]
    if baseline_same_game:
        baseline = sorted(baseline_same_game, key=lambda r: r.timestamp)[-1]
    else:
        baselines = [r for r in records if _is_baseline_tools(r.tools)]
        if not baselines:
            raise ValueError("No baseline records found (all records look optimized).")
        baseline = sorted(baselines, key=lambda r: r.timestamp)[-1]
    return baseline, optimized


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-records", action="store_true", help="Auto-pick newest baseline and optimized records.")
    ap.add_argument("--records-dir", type=str, default="textworld/mcts/records")
    ap.add_argument("--game-filter", type=str, default=None, help="Substring filter on metadata.game")
    ap.add_argument("--baseline-record", type=str, default=None, help="Explicit baseline record JSON path")
    ap.add_argument("--opt-record", type=str, default=None, help="Explicit optimized record JSON path")
    ap.add_argument("--output", type=str, default="record_compare.gif")
    ap.add_argument("--fps", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    baseline_path: Optional[Path] = Path(args.baseline_record) if args.baseline_record else None
    opt_path: Optional[Path] = Path(args.opt_record) if args.opt_record else None

    if args.from_records:
        records_dir = (PROJECT_ROOT / args.records_dir).resolve()
        records = _list_records(records_dir, args.game_filter)
        baseline_info, opt_info = _pick_newest_pair(records)
        baseline_path = baseline_info.path
        opt_path = opt_info.path
        print(f"Picked baseline: {baseline_path.name}")
        print(f"Picked optimized: {opt_path.name}")

    if not baseline_path or not opt_path:
        raise SystemExit("Need either --from-records or both --baseline-record and --opt-record.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / args.output

    baseline = json.loads(baseline_path.read_text())
    optimized = json.loads(opt_path.read_text())

    traj_base, gs_base = _record_to_trajectory(baseline)
    traj_opt, gs_opt = _record_to_trajectory(optimized)

    if args.max_steps is not None:
        traj_base = traj_base[: args.max_steps + 1]
        traj_opt = traj_opt[: args.max_steps + 1]

    viz = TrajectoryVisualizer(traj_base, game_state=gs_base)
    viz.compare_side_by_side(
        traj_opt,
        output_path=output_path,
        other_game_state=gs_opt,
        agent_names=("Baseline", "Optimized"),
        fps=args.fps,
    )
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

