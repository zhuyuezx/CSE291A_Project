"""
Torch-based Sokoban DQN module for PUCT integration.

Exports:
    - load_checkpoint(path, device)
    - q_model(model_input) -> list[float] (len=4)
    - encode_state_fn(state) -> torch.FloatTensor [1, C, H, W]
    - action_to_index_fn(action) -> int
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "PyTorch is required for dqn_sokoban_torch. "
        "Install torch in your active environment first."
    ) from e


@dataclass
class EncoderConfig:
    height: int = 10
    width: int = 10


class SokobanQNet(nn.Module):
    def __init__(self, in_channels: int = 5, height: int = 10, width: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * height * width, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


_MODEL: SokobanQNet | None = None
_DEVICE = torch.device("cpu")
_ENCODER_CFG = EncoderConfig()


def action_to_index_fn(action: Any) -> int:
    return int(action)


def _encode_planes(state: Any, cfg: EncoderConfig) -> torch.Tensor:
    h, w = cfg.height, cfg.width
    x = torch.zeros((5, h, w), dtype=torch.float32)

    # channels: walls, targets, boxes, player, floor
    for (r, c) in state.walls:
        if 0 <= r < h and 0 <= c < w:
            x[0, r, c] = 1.0
    for (r, c) in state.targets:
        if 0 <= r < h and 0 <= c < w:
            x[1, r, c] = 1.0
    for (r, c) in state.boxes:
        if 0 <= r < h and 0 <= c < w:
            x[2, r, c] = 1.0
    pr, pc = state.player
    if 0 <= pr < h and 0 <= pc < w:
        x[3, pr, pc] = 1.0

    for r in range(min(state.height, h)):
        for c in range(min(state.width, w)):
            if (r, c) not in state.walls:
                x[4, r, c] = 1.0
    return x


def encode_state_fn(state: Any) -> torch.Tensor:
    """Return [1, C, H, W] tensor suitable for model inference."""
    return _encode_planes(state, _ENCODER_CFG).unsqueeze(0).to(_DEVICE)


def load_checkpoint(path: str, device: str = "cpu") -> None:
    """
    Load trained DQN checkpoint.

    Expected keys:
        model_state_dict
        encoder: {height, width}
    """
    global _MODEL, _DEVICE, _ENCODER_CFG
    _DEVICE = torch.device(device)
    ckpt = torch.load(path, map_location=_DEVICE)
    enc = ckpt.get("encoder", {})
    _ENCODER_CFG = EncoderConfig(
        height=int(enc.get("height", 10)),
        width=int(enc.get("width", 10)),
    )
    model = SokobanQNet(
        in_channels=5,
        height=_ENCODER_CFG.height,
        width=_ENCODER_CFG.width,
    ).to(_DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _MODEL = model


def q_model(model_input: Any) -> list[float]:
    """
    Run Q inference and return 4 action values.

    model_input can be:
        - SokobanState (will be encoded)
        - tensor [1, C, H, W]
    """
    if _MODEL is None:
        raise RuntimeError(
            "No trained model loaded. Call load_checkpoint(path) first."
        )
    if isinstance(model_input, torch.Tensor):
        x = model_input.to(_DEVICE)
    else:
        x = encode_state_fn(model_input)
    with torch.no_grad():
        q = _MODEL(x).squeeze(0).detach().cpu().tolist()
    return [float(v) for v in q]
