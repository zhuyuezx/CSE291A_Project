"""
Evaluate PUCT with a trained Sokoban DQN checkpoint.

Example:
  python tools/eval_dqn_prior.py --checkpoint checkpoints/sokoban_dqn.pt --level level4 --games 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_sokoban_torch import (
    action_to_index_fn,
    encode_state_fn,
    load_checkpoint,
    q_model,
)
from mcts import MCTSEngine, make_dqn_prior_fn, make_puct_expansion, make_puct_selection
from mcts.games import Sokoban


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--level", default="level4")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--cpuct", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--expansion-strategy",
        choices=["greedy", "sample", "epsilon_greedy"],
        default="sample",
    )
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    load_checkpoint(args.checkpoint, device=args.device)
    prior_fn = make_dqn_prior_fn(
        q_model=q_model,
        encode_state_fn=encode_state_fn,
        action_to_index_fn=action_to_index_fn,
        temperature=args.temperature,
    )

    engine = MCTSEngine(
        Sokoban(args.level),
        iterations=args.iterations,
        logging=False,
    )
    engine.set_tool("selection", make_puct_selection(prior_fn, c_puct=args.cpuct))
    engine.set_tool(
        "expansion",
        make_puct_expansion(
            prior_fn,
            strategy=args.expansion_strategy,
            epsilon=args.epsilon,
        ),
    )
    stats = engine.play_many(num_games=args.games, verbose=args.verbose)
    print(
        f"PUCT+DQN level={args.level} solved={stats['solved']} "
        f"solve_rate={stats['solve_rate']} avg_steps={stats['avg_steps']}"
    )


if __name__ == "__main__":
    main()
