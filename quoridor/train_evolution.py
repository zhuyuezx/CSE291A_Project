#!/usr/bin/env python3
"""
Train Evolution - Language-Based Gradient Descent Loop

Implements a self-evolving training loop where:
1. Agents play self-play games using current strategy
2. Losses are analyzed for patterns
3. New strategic insights are generated
4. Strategy guide is updated with new rules

Usage:
    python train_evolution.py [--cycles N] [--games N] [--simulations N]
"""

import sys
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from quoridor_agent import (
    QuoridorAgentFactory,
    ExperienceManager,
    HeuristicEvaluator,
    LLM_MCTS_Bot,
    load_game,
    estimate_path_lengths,
    DEFAULT_MEMORY_DB,
)


# ==============================================================================
# Configuration
# ==============================================================================

NUM_CYCLES = 5
GAMES_PER_CYCLE = 10
DEFAULT_SIMULATIONS = 30
STRATEGY_FILE = "strategy_guide.txt"

DEFAULT_STRATEGY = """# Quoridor Strategy Guide
# Read by the AI agent to inform decisions.

## Core Principles
1. Prioritize path length - shorter path = winning
2. Walls are precious - each should force 2+ extra steps
3. Center control provides more options
4. When ahead, push forward. When behind, wall strategically.

## Starting Insights
- Default: Prioritize path length over wall placement in early game.
"""


# ==============================================================================
# Strategy Management
# ==============================================================================

def load_strategy() -> str:
    """Load strategy from file, create default if missing."""
    if os.path.exists(STRATEGY_FILE):
        with open(STRATEGY_FILE, 'r') as f:
            return f.read()
    else:
        print(f"[Strategy] Creating default {STRATEGY_FILE}")
        with open(STRATEGY_FILE, 'w') as f:
            f.write(DEFAULT_STRATEGY)
        return DEFAULT_STRATEGY


def append_strategy_insight(insight: str):
    """Append a new insight to the strategy file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    with open(STRATEGY_FILE, 'a') as f:
        f.write(f"\n\n## Insight [{timestamp}]\n{insight}\n")
    
    print(f"[Strategy] Appended new insight to {STRATEGY_FILE}")


# ==============================================================================
# Insight Generation
# ==============================================================================

def generate_insight(losses: List[Dict[str, Any]], cycle: int) -> str:
    """
    Generate a strategic insight based on loss patterns.
    
    THIS IS A MOCKED IMPLEMENTATION.
    
    For real LLM integration, replace with:
    
    ```python
    import openai  # or anthropic, or transformers
    
    loss_summaries = [f"- {l['state_text'][:200]}..." for l in losses[-5:]]
    
    prompt = f'''
    Analyze these Quoridor losing positions and identify tactical mistakes:
    
    LOSSES:
    {chr(10).join(loss_summaries)}
    
    Generate ONE actionable strategic insight.
    Format: "Observation: [issue]. New Rule: [action]."
    '''
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    
    return response.choices[0].message.content
    ```
    """
    mock_insights = [
        "Observation: Early wall placement without clear benefit. "
        "New Rule: First 5 moves, advance unless opponent is within 3 steps of goal.",
        
        "Observation: Vertical walls at row 3 often ineffective. "
        "New Rule: Prioritize horizontal walls that cut main path corridors.",
        
        "Observation: Wall economy depleted before endgame. "
        "New Rule: Reserve 2+ walls for final 10 moves.",
        
        "Observation: Too much blocking, not enough advancing. "
        "New Rule: If path is 2+ longer than opponent's, move forward.",
        
        "Observation: Edge walls have low impact. "
        "New Rule: Focus walls in columns C-G where both players traverse.",
    ]
    
    # Analyze losses for patterns
    loss_analysis = ""
    if losses:
        wall_heavy = sum(1 for l in losses if "walls: 0" in l.get("state_text", ""))
        if wall_heavy > len(losses) / 2:
            loss_analysis = " (Pattern: Ran out of walls)"
    
    return mock_insights[cycle % len(mock_insights)] + loss_analysis


# ==============================================================================
# Self-Play
# ==============================================================================

def run_self_play_game(game, factory: QuoridorAgentFactory, 
                       simulations: int, game_num: int,
                       use_random_opponent: bool = False) -> float:
    """Run a single self-play game."""
    
    # Player 0's bot (records experience)
    bot0 = factory.create_bot(
        game=game,
        player_id=0,
        num_simulations=simulations,
        record_experience=True
    )
    
    # Player 1's bot (opponent)
    if use_random_opponent:
        # Use random bot to generate wins/losses
        from quoridor_agent import RandomBot
        bot1 = RandomBot(player_id=1, seed=game_num)
    else:
        # Use heuristic bot for more strategic games
        opponent_evaluator = HeuristicEvaluator(
            experience_manager=None,
            strategy_text=factory.strategy_text
        )
        bot1 = LLM_MCTS_Bot(
            game=game,
            player_id=1,
            num_simulations=max(10, simulations // 3),  # Weaker opponent
            exploration_constant=2.0,  # More random
            evaluator=opponent_evaluator,
            experience_manager=None,
        )
    
    bots = [bot0, bot1]
    state = game.new_initial_state()
    
    # Quoridor games should complete within ~100-200 moves
    max_turns = 300
    turn = 0
    
    while not state.is_terminal() and turn < max_turns:
        action = bots[state.current_player()].step(state)
        state.apply_action(action)
        turn += 1
    
    # Determine outcome - use ACTUAL game result
    if state.is_terminal():
        returns = state.returns()
        if returns[0] > returns[1]:
            return 1.0  # Player 0 wins
        elif returns[1] > returns[0]:
            return -1.0  # Player 0 loses
        return 0.0  # Draw
    else:
        # Timeout - use path difference as proxy
        # Player 0 controls '0' piece (P2 path), Player 1 controls '@' (P1 path)
        # P0 winning = P2 path < P1 path = (p1_path - p2_path) > 0
        p1_path, p2_path = estimate_path_lengths(state)
        path_diff = p1_path - p2_path  # Positive = P0 is winning (shorter path)
        
        if path_diff >= 3:
            return 0.8
        elif path_diff >= 1:
            return 0.4
        elif path_diff <= -3:
            return -0.8
        elif path_diff <= -1:
            return -0.4
        return 0.0


# ==============================================================================
# Main Training Loop
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Self-Evolving Quoridor Training")
    parser.add_argument("--cycles", type=int, default=NUM_CYCLES)
    parser.add_argument("--games", type=int, default=GAMES_PER_CYCLE)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  QUORIDOR: LANGUAGE-BASED GRADIENT DESCENT")
    print("=" * 60)
    print(f"\nCycles: {args.cycles}, Games/cycle: {args.games}, Simulations: {args.simulations}")
    print("=" * 60 + "\n")
    
    # Load game
    game = load_game()
    
    # Load strategy
    strategy = load_strategy()
    
    # Create factory with unified memory
    factory = QuoridorAgentFactory(
        memory_file=DEFAULT_MEMORY_DB,
        use_llm=False,
        strategy_text=strategy
    )
    
    print(f"[Memory] Using unified memory: {DEFAULT_MEMORY_DB}")
    
    total_wins, total_losses, total_draws = 0, 0, 0
    
    # Evolution loop
    for cycle in range(args.cycles):
        print("\n" + "=" * 60)
        print(f"CYCLE {cycle + 1}/{args.cycles}")
        print("=" * 60)
        
        cycle_wins, cycle_losses, cycle_draws = 0, 0, 0
        
        # Step A: Self-Play (mix of random and heuristic opponents)
        print(f"\n[Step A] Running {args.games} self-play games...")
        
        for game_num in range(args.games):
            factory.clear_buffer()
            
            # Use random opponent for half of games to generate varied outcomes
            use_random = (game_num % 2 == 0)
            
            outcome = run_self_play_game(
                game=game,
                factory=factory,
                simulations=args.simulations,
                game_num=game_num,
                use_random_opponent=use_random
            )
            
            factory.end_game(outcome)
            
            opp_type = "R" if use_random else "H"  # Random or Heuristic
            if outcome > 0:
                cycle_wins += 1
            elif outcome < 0:
                cycle_losses += 1
            else:
                cycle_draws += 1
            
            print(f"  Game {game_num + 1}/{args.games}: "
                  f"{'Win' if outcome > 0 else 'Loss' if outcome < 0 else 'Draw'}")
        
        print(f"\n[Step A] Results: W={cycle_wins}, L={cycle_losses}, D={cycle_draws}")
        total_wins += cycle_wins
        total_losses += cycle_losses
        total_draws += cycle_draws
        
        # Step B: Analyze losses (using SQLite query)
        print(f"\n[Step B] Analyzing losses...")
        losses = factory.experience_manager.get_recent_losses(limit=args.games * 10)
        print(f"  Found {len(losses)} losing positions in DB")
        
        # Step C: Generate insight
        print(f"\n[Step C] Generating insight...")
        if losses or cycle_losses > 0:
            insight = generate_insight(losses, cycle)
            print(f"  Insight: {insight[:80]}...")
            
            # Step D: Update strategy
            print(f"\n[Step D] Updating strategy...")
            append_strategy_insight(insight)
            strategy = load_strategy()
            
            # Update factory with new strategy
            factory.strategy_text = strategy
            factory._evaluator = None  # Reset evaluator
        else:
            print("  No losses - skipping")
        
        print(f"\n[Cycle {cycle + 1}] Win Rate: {cycle_wins / args.games * 100:.1f}%")
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    total_games = total_wins + total_losses + total_draws
    stats = factory.get_statistics()
    
    print(f"\nTotal Games: {total_games}")
    print(f"Wins: {total_wins} ({total_wins/total_games*100:.1f}%)")
    print(f"Losses: {total_losses} ({total_losses/total_games*100:.1f}%)")
    print(f"Total Experiences: {stats['total']}")
    print(f"Historical Win Rate: {stats['win_rate']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
