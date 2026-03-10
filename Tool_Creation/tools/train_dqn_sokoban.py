"""
Train a minimal DQN for Sokoban and save checkpoint.

Example:
  python tools/train_dqn_sokoban.py --episodes 3000 --save-path checkpoints/sokoban_dqn.pt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_sokoban_torch import EncoderConfig, SokobanQNet
from mcts import MCTSEngine, make_dqn_prior_fn, make_puct_expansion, make_puct_selection
from mcts.games import Sokoban
from mcts.games.sokoban import LEVELS


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self.buf.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buf, batch_size)

    def __len__(self) -> int:
        return len(self.buf)


def encode_state(state: Any, cfg: EncoderConfig, device: torch.device) -> torch.Tensor:
    x = torch.zeros((5, cfg.height, cfg.width), dtype=torch.float32, device=device)
    for (r, c) in state.walls:
        if 0 <= r < cfg.height and 0 <= c < cfg.width:
            x[0, r, c] = 1.0
    for (r, c) in state.targets:
        if 0 <= r < cfg.height and 0 <= c < cfg.width:
            x[1, r, c] = 1.0
    for (r, c) in state.boxes:
        if 0 <= r < cfg.height and 0 <= c < cfg.width:
            x[2, r, c] = 1.0
    pr, pc = state.player
    if 0 <= pr < cfg.height and 0 <= pc < cfg.width:
        x[3, pr, pc] = 1.0
    for r in range(min(state.height, cfg.height)):
        for c in range(min(state.width, cfg.width)):
            if (r, c) not in state.walls:
                x[4, r, c] = 1.0
    return x


def shaped_reward(prev_state: Any, next_state: Any, done: bool) -> float:
    reward = -0.01
    reward += 2.0 * (next_state.boxes_on_targets() - prev_state.boxes_on_targets())
    reward += 0.1 * (prev_state.total_box_distance() - next_state.total_box_distance())
    if done and next_state.returns()[0] > 0:
        reward += 10.0
    elif done:
        reward -= 2.0
    return float(reward)


def optimize(
    policy: SokobanQNet,
    target: SokobanQNet,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float:
    if len(replay) < batch_size:
        return 0.0
    batch = replay.sample(batch_size)

    s = torch.stack([t.state for t in batch]).to(device)
    a = torch.tensor([t.action for t in batch], dtype=torch.long, device=device).unsqueeze(1)
    r = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
    ns = torch.stack([t.next_state for t in batch]).to(device)
    d = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

    q = policy(s).gather(1, a).squeeze(1)
    with torch.no_grad():
        nq = target(ns).max(dim=1).values
        y = r + (1.0 - d) * gamma * nq

    loss = nn.SmoothL1Loss()(q, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())


def evaluate_performance(
    policy: SokobanQNet,
    cfg: EncoderConfig,
    device: torch.device,
    levels: list[str],
    iterations: int,
    games: int,
    cpuct: float,
    temperature: float,
    expansion_strategy: str,
    epsilon: float,
    max_steps: int,
) -> dict[str, Any]:
    """
    Evaluate current policy as PUCT prior against UCT baseline.
    Prints per-level progress and a compact summary table.
    """
    prev_training = policy.training
    policy.eval()

    def encode_state_fn(state: Any) -> torch.Tensor:
        return encode_state(state, cfg, device).unsqueeze(0)

    def q_model(model_input: Any):
        # Accept either tensor or raw state.
        if isinstance(model_input, torch.Tensor):
            x = model_input.to(device)
        else:
            x = encode_state_fn(model_input)
        with torch.no_grad():
            return policy(x).squeeze(0).detach().cpu().tolist()

    prior_fn = make_dqn_prior_fn(
        q_model=q_model,
        encode_state_fn=encode_state_fn,
        action_to_index_fn=lambda a: int(a),
        temperature=temperature,
    )

    rows = []
    for lv in levels:
        base_engine = MCTSEngine(
            Sokoban(level_name=lv, max_steps=max_steps),
            iterations=iterations,
            logging=False,
        )
        base = base_engine.play_many(num_games=games, verbose=False)

        opt_engine = MCTSEngine(
            Sokoban(level_name=lv, max_steps=max_steps),
            iterations=iterations,
            logging=False,
        )
        opt_engine.set_tool("selection", make_puct_selection(prior_fn, c_puct=cpuct))
        opt_engine.set_tool(
            "expansion",
            make_puct_expansion(
                prior_fn,
                strategy=expansion_strategy,
                epsilon=epsilon,
            ),
        )
        opt = opt_engine.play_many(num_games=games, verbose=False)

        base_rate = float(base["solve_rate"])
        opt_rate = float(opt["solve_rate"])
        print(
            f"Evaluating {lv}... baseline={base_rate:.3f} ({base_rate*100:.0f}%)  "
            f"optimized={opt_rate:.3f} ({opt_rate*100:.0f}%)",
            flush=True,
        )

        base_ret = sum(float(r["returns"][0]) for r in base["results"]) / max(1, len(base["results"]))
        opt_ret = sum(float(r["returns"][0]) for r in opt["results"]) / max(1, len(opt["results"]))
        rows.append(
            {
                "level": lv,
                "base_solve": base_rate,
                "opt_solve": opt_rate,
                "base_ret": base_ret,
                "opt_ret": opt_ret,
                "base_steps": float(base["avg_steps"]),
                "opt_steps": float(opt["avg_steps"]),
            }
        )

    print(
        f"{'Level':<8} {'Base Solve%':>10} {'Opt Solve%':>10} "
        f"{'Base AvgRet':>11} {'Opt AvgRet':>10} {'Base Steps':>10} {'Opt Steps':>9}",
        flush=True,
    )
    print("-" * 78, flush=True)
    for r in rows:
        print(
            f"{r['level']:<8} "
            f"{r['base_solve']*100:>9.0f}% "
            f"{r['opt_solve']*100:>9.0f}% "
            f"{r['base_ret']:>11.3f} "
            f"{r['opt_ret']:>10.3f} "
            f"{r['base_steps']:>10.1f} "
            f"{r['opt_steps']:>9.1f}",
            flush=True,
        )

    avg_opt_solve = sum(r["opt_solve"] for r in rows) / max(1, len(rows))
    avg_opt_steps = sum(r["opt_steps"] for r in rows) / max(1, len(rows))

    if prev_training:
        policy.train()

    return {
        "rows": rows,
        "avg_opt_solve": float(avg_opt_solve),
        "avg_opt_steps": float(avg_opt_steps),
    }


def _save_checkpoint(
    path: Path,
    policy: SokobanQNet,
    cfg: EncoderConfig,
    train_levels: list[str],
    episodes: int,
    max_steps: int,
    stage_name: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": policy.state_dict(),
        "encoder": {"height": cfg.height, "width": cfg.width},
        "train_levels": train_levels,
        "episodes": episodes,
        "max_steps": max_steps,
    }
    if stage_name is not None:
        payload["stage"] = stage_name
    torch.save(payload, str(path))


def _parse_levels(raw: str) -> list[str]:
    levels = [x.strip() for x in raw.split(",") if x.strip()]
    for lv in levels:
        if lv not in LEVELS:
            raise ValueError(f"Unknown level '{lv}'")
    return levels


def _run_training_stage(
    *,
    stage_name: str,
    levels: list[str],
    episodes: int,
    policy: SokobanQNet,
    target: SokobanQNet,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer,
    cfg: EncoderConfig,
    device: torch.device,
    args,
    total_steps_start: int,
    eval_levels: list[str],
    stage_max_steps: int,
    global_ep_start: int,
    global_total_episodes: int,
    stage_best_path: Path | None,
) -> int:
    total_steps = total_steps_start
    solved_recent = deque(maxlen=100)

    print(
        f"\n=== {stage_name} === levels={','.join(levels)} episodes={episodes}",
        flush=True,
    )
    best_score: tuple[float, float] | None = None  # (avg_opt_solve, -avg_opt_steps)

    for ep in range(1, episodes + 1):
        level = random.choice(levels)
        state = Sokoban(level_name=level, max_steps=stage_max_steps).new_initial_state()
        done = False
        loss_acc = 0.0
        loss_n = 0

        global_ep = global_ep_start + ep
        frac = min(1.0, global_ep / max(1, args.eps_decay_episodes))
        eps = args.eps_start + (args.eps_end - args.eps_start) * frac

        ep_steps = 0
        while not done:
            legal = state.legal_actions()
            if not legal:
                break
            if random.random() < eps:
                action = random.choice(legal)
            else:
                with torch.no_grad():
                    s = encode_state(state, cfg, device).unsqueeze(0)
                    q = policy(s).squeeze(0)
                    mask = torch.full((4,), -1e9, device=device)
                    for a in legal:
                        mask[int(a)] = 0.0
                    action = int(torch.argmax(q + mask).item())

            prev = state.clone()
            state.apply_action(action)
            done = state.is_terminal()
            r = shaped_reward(prev, state, done)

            replay.push(
                Transition(
                    state=encode_state(prev, cfg, device),
                    action=int(action),
                    reward=float(r),
                    next_state=encode_state(state, cfg, device),
                    done=1.0 if done else 0.0,
                )
            )
            total_steps += 1
            ep_steps += 1

            loss = optimize(
                policy,
                target,
                optimizer,
                replay,
                batch_size=args.batch_size,
                gamma=args.gamma,
                device=device,
            )
            if loss > 0:
                loss_acc += loss
                loss_n += 1

            if total_steps % args.target_update == 0:
                target.load_state_dict(policy.state_dict())

        solved = 1 if state.returns()[0] > 0 else 0
        solved_recent.append(solved)

        if ep % max(1, args.log_interval) == 0:
            avg_loss = (loss_acc / max(1, loss_n))
            sr = sum(solved_recent) / max(1, len(solved_recent))
            print(
                f"[{stage_name}] ep={ep}/{episodes} global_ep={global_ep}/{global_total_episodes} eps={eps:.3f} "
                f"recent_solve_rate={sr:.3f} avg_loss={avg_loss:.4f} "
                f"ep_steps={ep_steps} total_steps={total_steps}",
                flush=True,
            )

        if args.eval_interval > 0 and ep % args.eval_interval == 0:
            print(
                f"\n[Eval @ {stage_name} ep {ep}] "
                f"levels={','.join(eval_levels)} games={args.eval_games} "
                f"iters={args.eval_iterations}",
                flush=True,
            )
            eval_summary = evaluate_performance(
                policy=policy,
                cfg=cfg,
                device=device,
                levels=eval_levels,
                iterations=args.eval_iterations,
                games=args.eval_games,
                cpuct=args.eval_cpuct,
                temperature=args.eval_temperature,
                expansion_strategy=args.eval_expansion_strategy,
                epsilon=args.eval_epsilon,
                max_steps=stage_max_steps,
            )
            current_score = (
                float(eval_summary["avg_opt_solve"]),
                -float(eval_summary["avg_opt_steps"]),
            )
            if stage_best_path is not None and (best_score is None or current_score > best_score):
                best_score = current_score
                _save_checkpoint(
                    path=stage_best_path,
                    policy=policy,
                    cfg=cfg,
                    train_levels=levels,
                    episodes=ep,
                    max_steps=stage_max_steps,
                    stage_name=f"{stage_name}_best",
                )
                print(
                    f"[{stage_name}] new BEST checkpoint saved: {stage_best_path} "
                    f"(avg_opt_solve={current_score[0]:.3f}, avg_opt_steps={-current_score[1]:.1f})",
                    flush=True,
                )
            print("", flush=True)

    return total_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stage1-lr", type=float, default=None)
    parser.add_argument("--stage2-lr", type=float, default=5e-4)
    parser.add_argument("--stage3-lr", type=float, default=1e-4)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-episodes", type=int, default=2500)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--train-levels", default="level1,level2,level3,level4,level5")
    parser.add_argument("--save-path", default="checkpoints/sokoban_dqn.pt")
    parser.add_argument("--stage1-levels", default="level1,level2,level3,level4")
    parser.add_argument("--stage2-levels", default="level1,level2,level3,level4,level5,level6,level7")
    parser.add_argument("--stage3-levels", default="level1,level2,level3,level4,level5,level6,level7,level8,level9,level10")
    parser.add_argument("--stage1-max-steps", type=int, default=50)
    parser.add_argument("--stage2-max-steps", type=int, default=100)
    parser.add_argument("--stage3-max-steps", type=int, default=150)
    parser.add_argument("--stage1-episodes", type=int, default=50)
    parser.add_argument("--stage2-episodes", type=int, default=100)
    parser.add_argument("--stage3-episodes", type=int, default=150)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-levels", default="")
    parser.add_argument("--eval-games", type=int, default=3)
    parser.add_argument("--eval-iterations", type=int, default=200)
    parser.add_argument("--eval-cpuct", type=float, default=0.6)
    parser.add_argument("--eval-temperature", type=float, default=1.2)
    parser.add_argument(
        "--eval-expansion-strategy",
        choices=["greedy", "sample", "epsilon_greedy"],
        default="sample",
    )
    parser.add_argument("--eval-epsilon", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    cfg = EncoderConfig(height=args.height, width=args.width)
    levels = _parse_levels(args.train_levels)
    if args.eval_levels.strip():
        eval_levels_user = _parse_levels(args.eval_levels)
    else:
        eval_levels_user = []

    policy = SokobanQNet(5, cfg.height, cfg.width).to(device)
    target = SokobanQNet(5, cfg.height, cfg.width).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)

    total_steps = 0
    save_path = Path(args.save_path)

    if args.staged:
        stage_specs = [
            ("stage1", _parse_levels(args.stage1_levels), args.stage1_episodes, args.stage1_max_steps, args.stage1_lr),
            ("stage2", _parse_levels(args.stage2_levels), args.stage2_episodes, args.stage2_max_steps, args.stage2_lr),
            ("stage3", _parse_levels(args.stage3_levels), args.stage3_episodes, args.stage3_max_steps, args.stage3_lr),
        ]
        global_total_episodes = sum(s[2] for s in stage_specs)
        global_ep_start = 0
        for stage_name, stage_levels, stage_episodes, stage_max_steps, stage_lr in stage_specs:
            lr = args.lr if stage_lr is None else float(stage_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            print(f"\n[{stage_name}] optimizer lr={lr}", flush=True)

            eval_levels = eval_levels_user[:] if eval_levels_user else stage_levels[:]
            stage_best_ckpt = save_path.with_name(f"{save_path.stem}_{stage_name}_best{save_path.suffix}")
            total_steps = _run_training_stage(
                stage_name=stage_name,
                levels=stage_levels,
                episodes=stage_episodes,
                policy=policy,
                target=target,
                optimizer=optimizer,
                replay=replay,
                cfg=cfg,
                device=device,
                args=args,
                total_steps_start=total_steps,
                eval_levels=eval_levels,
                stage_max_steps=stage_max_steps,
                global_ep_start=global_ep_start,
                global_total_episodes=global_total_episodes,
                stage_best_path=stage_best_ckpt,
            )
            global_ep_start += stage_episodes
            stage_ckpt = save_path.with_name(f"{save_path.stem}_{stage_name}{save_path.suffix}")
            _save_checkpoint(
                path=stage_ckpt,
                policy=policy,
                cfg=cfg,
                train_levels=stage_levels,
                episodes=stage_episodes,
                max_steps=stage_max_steps,
                stage_name=stage_name,
            )
            print(f"saved checkpoint: {stage_ckpt}", flush=True)
    else:
        eval_levels = eval_levels_user[:] if eval_levels_user else levels[:]
        best_ckpt = save_path.with_name(f"{save_path.stem}_best{save_path.suffix}")
        total_steps = _run_training_stage(
            stage_name="single",
            levels=levels,
            episodes=args.episodes,
            policy=policy,
            target=target,
            optimizer=optimizer,
            replay=replay,
            cfg=cfg,
            device=device,
            args=args,
            total_steps_start=total_steps,
            eval_levels=eval_levels,
            stage_max_steps=args.max_steps,
            global_ep_start=0,
            global_total_episodes=args.episodes,
            stage_best_path=best_ckpt,
        )

    _save_checkpoint(
        path=save_path,
        policy=policy,
        cfg=cfg,
        train_levels=(stage_specs[-1][1] if args.staged else levels),
        episodes=(stage_specs[-1][2] if args.staged else args.episodes),
        max_steps=(stage_specs[-1][3] if args.staged else args.max_steps),
        stage_name=("stage3" if args.staged else "single"),
    )
    print(f"saved checkpoint: {save_path}", flush=True)


if __name__ == "__main__":
    main()
