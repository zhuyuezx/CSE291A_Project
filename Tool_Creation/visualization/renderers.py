"""
Board renderers used by trajectory and animation visualizations.
"""

from __future__ import annotations

import math
from typing import Any

from matplotlib.patches import Circle, Rectangle


def _draw_sokoban_state(ax, state, step_idx: int, action_label: str = "") -> None:
    height, width = state.height, state.width
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(Rectangle((0, 0), width, height, facecolor="#f5f5f5", edgecolor="none", zorder=0))

    for r in range(height):
        for c in range(width):
            pos = (r, c)
            color = "#4b4b4b" if pos in state.walls else "#e9e9e9"
            ax.add_patch(
                Rectangle((c, r), 1, 1, facecolor=color, edgecolor="#d0d0d0", linewidth=0.6, zorder=1)
            )

    for (r, c) in state.targets:
        ax.add_patch(
            Circle((c + 0.5, r + 0.5), 0.17, facecolor="#ffeb3b", edgecolor="#b59f00", linewidth=0.8, zorder=2)
        )

    for (r, c) in state.boxes:
        on_target = (r, c) in state.targets
        face = "#4caf50" if on_target else "#f39c12"
        edge = "#2e7d32" if on_target else "#8a5a00"
        ax.add_patch(
            Rectangle((c + 0.12, r + 0.12), 0.76, 0.76, facecolor=face, edgecolor=edge, linewidth=1.2, zorder=3)
        )

    pr, pc = state.player
    ax.add_patch(Circle((pc + 0.5, pr + 0.5), 0.26, facecolor="#1e88e5", edgecolor="#0d47a1", linewidth=1.0, zorder=4))

    title = "t=0 (start)" if step_idx == 0 else f"t={step_idx} ({action_label})"
    ax.set_title(title, fontsize=8, pad=2)


def _draw_rush_hour_state(ax, state, step_idx: int, action_label: str = "") -> None:
    board_size = int(math.sqrt(len(getattr(state, "occupied", [])) or 36))
    ax.set_xlim(0, board_size)
    ax.set_ylim(board_size, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    target_idx = int(getattr(state, "target", board_size * board_size - 1))
    target_row = target_idx // board_size
    target_col = target_idx % board_size
    wall_cells = set(getattr(state, "walls", ()))
    for r in range(board_size):
        for c in range(board_size):
            idx = r * board_size + c
            if idx == target_idx:
                # Mark the goal exit cell clearly.
                cell_color = "#c8f7c5"
                edge_color = "#2e7d32"
                lw = 1.5
            elif idx in wall_cells:
                cell_color = "#2f2f2f"
                edge_color = "#c8c8c8"
                lw = 0.7
            else:
                cell_color = "#f0f0f0"
                edge_color = "#c8c8c8"
                lw = 0.7
            ax.add_patch(
                Rectangle((c, r), 1, 1, facecolor=cell_color, edgecolor=edge_color, linewidth=lw, zorder=1)
            )

    piece_colors = [
        "#d62728",  # Primary (A)
        "#1f77b4",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#8c564b",
        "#17becf",
        "#bcbd22",
    ]
    pieces = list(getattr(state, "pieces", []))
    for piece_idx, piece in enumerate(pieces):
        color = piece_colors[piece_idx % len(piece_colors)]
        if piece_idx == 0:
            color = "#d62728"
        for cell in piece.cells():
            r = cell // board_size
            c = cell % board_size
            ax.add_patch(
                Rectangle(
                    (c + 0.06, r + 0.06),
                    0.88,
                    0.88,
                    facecolor=color,
                    edgecolor="#202020",
                    linewidth=0.9,
                    zorder=2,
                )
            )
        if piece_idx == 0:
            # Label the primary target block explicitly.
            p0 = piece.cells()[0]
            r0 = p0 // board_size
            c0 = p0 % board_size
            ax.text(
                c0 + 0.5,
                r0 + 0.5,
                "A",
                fontsize=8,
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
                zorder=3,
            )

    # Exit direction hint (goal is to move A to the right edge target).
    ax.annotate(
        "EXIT",
        xy=(target_col + 0.5, target_row + 0.5),
        xytext=(board_size + 0.35, target_row + 0.5),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=7,
        color="#2e7d32",
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#2e7d32"},
        zorder=4,
    )
    ax.text(
        0.0,
        -0.08,
        "Goal: move red block A to EXIT",
        transform=ax.transAxes,
        fontsize=7,
        color="#333333",
        ha="left",
        va="top",
    )

    title = "t=0 (start)" if step_idx == 0 else f"t={step_idx} ({action_label})"
    ax.set_title(title, fontsize=8, pad=2)


def draw_state(ax, state: Any, game_name: str, step_idx: int, action_label: str = "") -> None:
    name = (game_name or "").lower().replace(" ", "_")
    if "rush_hour" in name or "rushhour" in name:
        _draw_rush_hour_state(ax, state, step_idx=step_idx, action_label=action_label)
        return
    _draw_sokoban_state(ax, state, step_idx=step_idx, action_label=action_label)

