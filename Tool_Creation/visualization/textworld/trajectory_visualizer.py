"""
Episode Trajectory Visualizer for TextWorld.

Creates decision-focused animated GIFs showing key steps of an episode with:
- World map & visited path
- Compact observation panel
- Inventory/progress panel
- Available actions panel (optionally with Q-values)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import networkx as nx


class TrajectoryVisualizer:
    """
    Visualizes an episode trajectory as an animated GIF with decision-focused layout.

    Trajectory format: list of step dicts with keys:
    - observation: dict with at least 'room' (int) and optional fields like
      'description', 'inventory', 'quest_progress'
    - action: action taken (any; shown via str())
    - reward: float
    - done: bool
    - legal_actions: list (optional)
    - q_values: list (optional; same length as legal_actions)
    """

    def __init__(
        self,
        trajectory: List[Dict[str, Any]],
        action_descriptions: Optional[Dict[str, str]] = None,
        game_state: Optional[Any] = None,
    ):
        self.trajectory = trajectory
        self.action_descriptions = action_descriptions or {}
        self.game_state = game_state
        self._extract_room_descriptions()

    def _extract_room_descriptions(self) -> None:
        self.room_descriptions: Dict[Any, str] = {}
        for step in self.trajectory:
            obs = step.get("observation", {})
            if not isinstance(obs, dict):
                continue
            room = obs.get("room")
            desc = obs.get("description", "Unknown location")
            if room is not None:
                self.room_descriptions[room] = str(desc)

    def _get_visited_rooms(self, step_idx: int) -> Set[Any]:
        visited: Set[Any] = set()
        for i in range(step_idx + 1):
            obs = self.trajectory[i].get("observation", {})
            if isinstance(obs, dict):
                room = obs.get("room")
                if room is not None:
                    visited.add(room)
        return visited

    def _infer_current_room(self, step_idx: int) -> Optional[Any]:
        for i in range(step_idx, -1, -1):
            obs = self.trajectory[i].get("observation", {})
            if isinstance(obs, dict):
                room = obs.get("room")
                if room is not None:
                    return room
        return None

    def _path_layout(
        self,
        G: nx.Graph,
        all_nodes: Set[Any],
        all_edges: Set[Tuple[Any, Any]],
    ) -> Optional[Dict[Any, Tuple[float, float]]]:
        """
        If the graph is a simple path (linear chain), return positions in a line with even spacing
        so nodes like 0 and 4 do not overlap. Otherwise return None.
        """
        if not all_nodes or len(all_edges) != len(all_nodes) - 1:
            return None
        degrees = dict(G.degree())
        if not all(d <= 2 for d in degrees.values()):
            return None
        if not nx.is_connected(G):
            return None
        # Order nodes along the path: start from a degree-1 node and walk the chain.
        start = min((n for n in all_nodes if degrees.get(n, 0) == 1), default=min(all_nodes))
        order: List[Any] = []
        seen: Set[Any] = set()
        u = start
        while u is not None:
            order.append(u)
            seen.add(u)
            next_u = None
            for v in G.neighbors(u):
                if v not in seen:
                    next_u = v
                    break
            u = next_u
        if len(order) != len(all_nodes):
            return None
        spacing = 1.0
        return {node: (i * spacing, 0.0) for i, node in enumerate(order)}

    def _extract_adjacency(self) -> Tuple[Dict[Any, Any], Dict[Any, Tuple[float, float]]]:
        """
        Extract adjacency + optional coordinates from game_state.

        Supported formats:
        - object with `.graph` and optional `.coords`
        - dict with keys: `graph` / `world_graph` and optional `coords`
        """
        gs = self.game_state
        if gs is None:
            return {}, {}

        if hasattr(gs, "graph"):
            graph = getattr(gs, "graph", {}) or {}
            coords = getattr(gs, "coords", {}) or {}
            return graph, coords

        if isinstance(gs, dict):
            graph = gs.get("graph") or gs.get("world_graph") or {}
            coords = gs.get("coords") or {}
            return graph, coords

        return {}, {}

    def _get_world_graph_data(
        self, step_idx: int, current_room: Optional[Any], step_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        if current_room is None:
            current_room = self._infer_current_room(step_idx)

        graph, coords = self._extract_adjacency()

        if not graph or current_room is None:
            node = 0 if current_room is None else current_room
            return {
                "nodes": [node],
                "edges": [],
                "positions": {node: (0.0, 0.0)},
                "current_room": node,
                "visited": {node},
                "room_descriptions": {node: self.room_descriptions.get(node, "Current location")},
                "path_edges": [],
            }

        visited_rooms = self._get_visited_rooms(step_idx)

        all_nodes: Set[Any] = set()
        all_edges: Set[Tuple[Any, Any]] = set()

        def add_edge(u: Any, v: Any) -> None:
            all_nodes.add(u)
            all_nodes.add(v)
            all_edges.add((u, v))

        for room, connections in graph.items():
            all_nodes.add(room)
            if connections is None:
                continue
            if isinstance(connections, dict):
                for _direction, next_room in connections.items():
                    add_edge(room, next_room)
            elif isinstance(connections, (list, tuple, set)):
                for next_room in connections:
                    add_edge(room, next_room)

        positions: Dict[Any, Tuple[float, float]] = {}
        if coords:
            positions.update(coords)
        else:
            G = nx.Graph()
            G.add_nodes_from(all_nodes)
            G.add_edges_from(all_edges)
            # Use linear layout for path graphs (e.g. TextWorld Coin corridor) so nodes stay well spaced.
            linear_pos = self._path_layout(G, all_nodes, all_edges)
            if linear_pos:
                positions = linear_pos
            else:
                # Force-directed layout for branching structures.
                layout = nx.spring_layout(G, k=1.2, iterations=50, seed=42)
                for node, (x, y) in layout.items():
                    positions[node] = (float(x), float(y))

        path_edges: List[Tuple[Any, Any]] = []
        for i in range(step_idx):
            obs_curr = self.trajectory[i].get("observation", {})
            obs_next = self.trajectory[i + 1].get("observation", {})
            if not isinstance(obs_curr, dict) or not isinstance(obs_next, dict):
                continue
            room_curr = obs_curr.get("room")
            room_next = obs_next.get("room")
            if room_curr is not None and room_next is not None:
                if room_curr in all_nodes and room_next in all_nodes:
                    path_edges.append((room_curr, room_next))

        return {
            "nodes": list(all_nodes),
            "edges": list(all_edges),
            "positions": positions,
            "current_room": current_room,
            "visited": visited_rooms,
            "chosen_action": step_data.get("action"),
            "path_edges": path_edges,
            "room_descriptions": self.room_descriptions,
        }

    def _draw_world_graph(self, ax: plt.Axes, graph_data: Dict[str, Any]) -> None:
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]
        positions = graph_data["positions"]
        current_room = graph_data["current_room"]
        visited = graph_data.get("visited", set())
        path_edges = graph_data.get("path_edges", [])
        room_descriptions = graph_data.get("room_descriptions", {})

        for room1, room2 in edges:
            if room1 in positions and room2 in positions:
                x1, y1 = positions[room1]
                x2, y2 = positions[room2]
                ax.plot([x1, x2], [y1, y2], color="#CCCCCC", linewidth=1.5, alpha=0.5, zorder=1)

        for room1, room2 in path_edges:
            if room1 in positions and room2 in positions:
                x1, y1 = positions[room1]
                x2, y2 = positions[room2]
                ax.plot([x1, x2], [y1, y2], color="#FF6B6B", linewidth=3, alpha=0.9, zorder=3)

        for room in nodes:
            if room not in positions:
                continue

            x, y = positions[room]
            if room == current_room:
                circle = mpatches.Circle(
                    (x, y), radius=0.18, facecolor="#FFD700", edgecolor="#FF6B6B", linewidth=3, zorder=5
                )
                ax.add_patch(circle)
            elif room in visited:
                circle = mpatches.Circle(
                    (x, y), radius=0.14, facecolor="#87CEEB", edgecolor="#4169E1", linewidth=2, zorder=2
                )
                ax.add_patch(circle)
            else:
                circle = mpatches.Circle(
                    (x, y), radius=0.12, facecolor="#E0E0E0", edgecolor="#999999", linewidth=1.5, zorder=1
                )
                ax.add_patch(circle)

            ax.text(x, y, str(room), ha="center", va="center", fontsize=8, fontweight="bold", zorder=6)

            desc = room_descriptions.get(room, "")
            if desc:
                desc = str(desc)
                if len(desc) > 40:
                    desc = desc[:37] + "..."
                ax.text(
                    x,
                    y - 0.32,
                    desc,
                    ha="center",
                    va="top",
                    fontsize=6,
                    wrap=True,
                    color="#333333",
                    zorder=4,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, linewidth=0),
                )

        margin = 0.5
        if positions:
            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            ax.set_xlim(min(xs) - margin, max(xs) + margin)
            ax.set_ylim(min(ys) - margin, max(ys) + margin)

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("World Map & Path", fontsize=11, fontweight="bold", pad=5)

    def _draw_observation_panel(self, ax: plt.Axes, obs: Dict[str, Any]) -> None:
        lines: List[str] = []
        if "room" in obs:
            lines.append(f"Room {obs['room']}")
        if "description" in obs:
            desc = str(obs["description"])
            if len(desc) > 60:
                desc = desc[:57] + "..."
            lines.append(desc)

        ax.text(
            0.05,
            0.95,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            family="sans-serif",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F4FD", alpha=0.9),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Observation", fontsize=10, fontweight="bold", pad=5)

    def _draw_inventory_panel(self, ax: plt.Axes, obs: Dict[str, Any], reward: float) -> None:
        lines: List[str] = ["INVENTORY:"]
        inventory = obs.get("inventory", [])
        if isinstance(inventory, list):
            if inventory:
                for item in inventory[:3]:
                    lines.append(f"• {item}")
                if len(inventory) > 3:
                    lines.append(f"• ... +{len(inventory) - 3} more")
            else:
                lines.append("(empty)")
        else:
            lines.append(str(inventory))

        lines += ["", "QUEST:"]
        quest_progress = obs.get("quest_progress", {})
        if isinstance(quest_progress, dict):
            for key, val in list(quest_progress.items())[:2]:
                lines.append(f"{key}: {val}")
        else:
            lines.append(str(quest_progress))

        # Only show reward when it is recorded (non-zero); records often have no reward data.
        if reward != 0.0:
            lines += ["", f"REWARD: {reward:+.3f}"]

        ax.text(
            0.05,
            0.95,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            family="sans-serif",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF2CC", alpha=0.9),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Inventory & Progress", fontsize=10, fontweight="bold", pad=5)

    def _draw_action_panel(self, ax: plt.Axes, step_data: Dict[str, Any]) -> None:
        action = step_data.get("action")
        legal_actions = step_data.get("legal_actions", []) or []
        q_values = step_data.get("q_values", []) or []

        action_q_map: Dict[Any, Any] = {}
        if q_values and len(q_values) == len(legal_actions):
            action_q_map = dict(zip(legal_actions, q_values))

        ax.text(
            0.05,
            0.95,
            "ACTIONS:",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            family="sans-serif",
        )

        for i, act in enumerate(legal_actions[:6]):
            act_str = self._format_action(act)
            chosen = str(act) == str(action) or act_str == str(action)
            prefix = "→" if chosen else "  "
            if act in action_q_map:
                line = f"{prefix} {act_str}: {float(action_q_map[act]):.3f}"
            else:
                line = f"{prefix} {act_str}"

            y_pos = 0.85 - i * 0.12
            bbox_color = "#FFE6E6" if chosen else "white"
            ax.text(
                0.05,
                y_pos,
                line,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                family="sans-serif",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=bbox_color, alpha=0.7),
            )

        if not legal_actions and action:
            ax.text(
                0.05,
                0.85,
                f"→ {self._format_action(action)}",
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                family="sans-serif",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFE6E6", alpha=0.7),
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Available Actions", fontsize=10, fontweight="bold", pad=5)

    def _format_action(self, action: Any) -> str:
        if action in self.action_descriptions:
            return self.action_descriptions[action]
        return str(action)

    def visualize(
        self,
        output_path: Path | str,
        figsize: Tuple[int, int] = (14, 10),
        dpi: int = 100,
        fps: int = 2,
        max_trajectory_length: Optional[int] = None,
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trajectory = self.trajectory[:max_trajectory_length] if max_trajectory_length else self.trajectory

        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, top=0.85, bottom=0.1, left=0.1, right=0.9)

        ax_map = fig.add_subplot(gs[0, 0])
        ax_obs = fig.add_subplot(gs[0, 1])
        ax_inventory = fig.add_subplot(gs[1, 0])
        ax_actions = fig.add_subplot(gs[1, 1])

        def create_frame(step_idx: int) -> None:
            ax_map.clear()
            ax_obs.clear()
            ax_inventory.clear()
            ax_actions.clear()

            step = trajectory[step_idx]
            obs = step.get("observation", {}) if isinstance(step.get("observation", {}), dict) else {}
            action = step.get("action")
            reward = float(step.get("reward", 0) or 0)
            done = bool(step.get("done", False))

            step_data = {
                "action": action,
                "legal_actions": step.get("legal_actions", []),
                "q_values": step.get("q_values", []),
                "done": done,
            }

            graph_data = self._get_world_graph_data(step_idx, obs.get("room"), step_data)
            self._draw_world_graph(ax_map, graph_data)
            self._draw_observation_panel(ax_obs, obs)
            self._draw_inventory_panel(ax_inventory, obs, reward)
            self._draw_action_panel(ax_actions, step_data)

            cumsum_reward = sum(float(t.get("reward", 0) or 0) for t in trajectory[: step_idx + 1])
            status = "DONE ✓" if done else "ONGOING"
            fig.suptitle(
                f"Decision Analysis - Step {step_idx + 1}/{len(trajectory)} | "
                f"Cumulative Reward: {cumsum_reward:.3f} | Status: {status}",
                fontsize=12,
                fontweight="bold",
            )

        writer = PillowWriter(fps=fps)
        with writer.saving(fig, str(output_path), dpi):
            for step_idx in range(len(trajectory)):
                create_frame(step_idx)
                writer.grab_frame()

        plt.close(fig)
        return str(output_path)

    def compare_side_by_side(
        self,
        other_trajectory: List[Dict[str, Any]],
        output_path: Path | str,
        other_game_state: Optional[Any] = None,
        other_action_descriptions: Optional[Dict[str, str]] = None,
        agent_names: Tuple[str, str] = ("Baseline", "Optimized"),
        figsize: Tuple[int, int] = (22, 10),
        dpi: int = 100,
        fps: int = 2,
        max_trajectory_length: Optional[int] = None,
    ) -> str:
        """
        Create a side-by-side comparison GIF for two trajectories.

        Layout per agent matches `visualize()` (map, obs, inventory, actions).
        Left = `self.trajectory`, Right = `other_trajectory`.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        left_viz = self
        right_viz = TrajectoryVisualizer(
            other_trajectory,
            action_descriptions=other_action_descriptions,
            game_state=other_game_state if other_game_state is not None else self.game_state,
        )

        left_traj = left_viz.trajectory[:max_trajectory_length] if max_trajectory_length else left_viz.trajectory
        right_traj = right_viz.trajectory[:max_trajectory_length] if max_trajectory_length else right_viz.trajectory
        n_frames = min(len(left_traj), len(right_traj))
        if n_frames <= 0:
            raise ValueError("Both trajectories must be non-empty for comparison.")

        fig = plt.figure(figsize=figsize, dpi=dpi)
        outer_gs = fig.add_gridspec(
            1,
            2,
            wspace=0.18,
            top=0.86,
            bottom=0.08,
            left=0.05,
            right=0.98,
        )

        # For each agent half, create:
        # - Left: one large axis for the world map spanning full height
        # - Right: three stacked axes for observation, inventory, actions
        def _make_axes(col_idx: int):
            sub = outer_gs[0, col_idx].subgridspec(2, 2, width_ratios=[2.0, 1.3], hspace=0.25, wspace=0.08)
            ax_map = fig.add_subplot(sub[:, 0])
            right = sub[:, 1].subgridspec(3, 1, hspace=0.25)
            ax_obs = fig.add_subplot(right[0, 0])
            ax_inv = fig.add_subplot(right[1, 0])
            ax_act = fig.add_subplot(right[2, 0])
            return ax_map, ax_obs, ax_inv, ax_act

        ax_map_l, ax_obs_l, ax_inv_l, ax_act_l = _make_axes(0)
        ax_map_r, ax_obs_r, ax_inv_r, ax_act_r = _make_axes(1)

        def draw_agent(
            viz: TrajectoryVisualizer,
            traj: List[Dict[str, Any]],
            step_idx: int,
            ax_map: plt.Axes,
            ax_obs: plt.Axes,
            ax_inv: plt.Axes,
            ax_act: plt.Axes,
            agent_title: str,
        ) -> Tuple[float, bool]:
            step = traj[step_idx]
            obs = step.get("observation", {}) if isinstance(step.get("observation", {}), dict) else {}
            action = step.get("action")
            reward = float(step.get("reward", 0) or 0)
            done = bool(step.get("done", False))

            step_data = {
                "action": action,
                "legal_actions": step.get("legal_actions", []),
                "q_values": step.get("q_values", []),
                "done": done,
            }

            graph_data = viz._get_world_graph_data(step_idx, obs.get("room"), step_data)
            viz._draw_world_graph(ax_map, graph_data)
            viz._draw_observation_panel(ax_obs, obs)
            viz._draw_inventory_panel(ax_inv, obs, reward)
            viz._draw_action_panel(ax_act, step_data)

            # Add per-agent titles (keep panel titles too)
            ax_map.set_title(f"{agent_title} — World Map & Path", fontsize=11, fontweight="bold", pad=5)
            ax_obs.set_title(f"{agent_title} — Observation", fontsize=10, fontweight="bold", pad=5)
            ax_inv.set_title(f"{agent_title} — Inventory & Progress", fontsize=10, fontweight="bold", pad=5)
            ax_act.set_title(f"{agent_title} — Available Actions", fontsize=10, fontweight="bold", pad=5)

            cumsum_reward = sum(float(t.get("reward", 0) or 0) for t in traj[: step_idx + 1])
            return cumsum_reward, done

        writer = PillowWriter(fps=fps)
        with writer.saving(fig, str(output_path), dpi):
            for step_idx in range(n_frames):
                for ax in (ax_map_l, ax_obs_l, ax_inv_l, ax_act_l, ax_map_r, ax_obs_r, ax_inv_r, ax_act_r):
                    ax.clear()

                cum_l, done_l = draw_agent(left_viz, left_traj, step_idx, ax_map_l, ax_obs_l, ax_inv_l, ax_act_l, agent_names[0])
                cum_r, done_r = draw_agent(right_viz, right_traj, step_idx, ax_map_r, ax_obs_r, ax_inv_r, ax_act_r, agent_names[1])

                status_l = "DONE ✓" if done_l else "ONGOING"
                status_r = "DONE ✓" if done_r else "ONGOING"
                fig.suptitle(
                    f"Trajectory Comparison — Step {step_idx + 1}/{n_frames} | "
                    f"{agent_names[0]}: {cum_l:.3f} ({status_l})  vs  {agent_names[1]}: {cum_r:.3f} ({status_r})",
                    fontsize=13,
                    fontweight="bold",
                )

                writer.grab_frame()

        plt.close(fig)
        return str(output_path)

