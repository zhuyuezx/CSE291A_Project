#!/usr/bin/env python3
"""
Human vs AI - Interactive Quoridor Game

Play against the LLM-MCTS agent in the terminal.

Usage:
    python human_vs_ai.py [--use-llm] [--simulations N]

Controls:
    - Move: Enter cell like "e2", "d3", etc.
    - Horizontal Wall: "h c3" (places horizontal wall at C3)
    - Vertical Wall: "v d4" (places vertical wall at D4)
    - Quit: "quit" or "q"
"""

import sys
import os
import argparse

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyspiel
from quoridor_agent import (
    QuoridorAgentFactory,
    load_game,
    DEFAULT_MEMORY_DB,
)


# ==============================================================================
# Input Parsing
# ==============================================================================

def parse_human_move(state: pyspiel.State, input_str: str) -> int:
    """
    Parse human input into a valid action.
    
    Formats:
        - Move: "e2", "d3" (column + row)
        - Horizontal Wall: "h c3" or "hc3"
        - Vertical Wall: "v d4" or "vd4"
    """
    input_str = input_str.strip().lower()
    
    if not input_str:
        return -1
    
    legal_actions = state.legal_actions()
    action_strings = {
        state.action_to_string(state.current_player(), a).lower(): a 
        for a in legal_actions
    }
    
    # Direct match
    if input_str in action_strings:
        return action_strings[input_str]
    
    # Wall commands with space: "h c3" -> "c3h"
    if input_str.startswith('h ') or input_str.startswith('v '):
        wall_type = input_str[0]
        pos = input_str[2:].strip()
        wall_str = f"{pos}{wall_type}"
        if wall_str in action_strings:
            return action_strings[wall_str]
    
    # Wall commands without space: "hc3" -> "c3h"
    if len(input_str) >= 3 and input_str[0] in ['h', 'v']:
        wall_type = input_str[0]
        pos = input_str[1:]
        wall_str = f"{pos}{wall_type}"
        if wall_str in action_strings:
            return action_strings[wall_str]
    
    return -1


def display_legal_moves(state: pyspiel.State):
    """Display categorized legal moves."""
    legal_actions = state.legal_actions()
    player = state.current_player()
    
    moves, h_walls, v_walls = [], [], []
    
    for action in legal_actions:
        action_str = state.action_to_string(player, action)
        if action_str.endswith('h'):
            h_walls.append(action_str)
        elif action_str.endswith('v'):
            v_walls.append(action_str)
        else:
            moves.append(action_str)
    
    print(f"\nMoves ({len(moves)}): {', '.join(sorted(moves))}")
    if h_walls:
        print(f"H-Walls ({len(h_walls)}): {', '.join(sorted(h_walls)[:10])}{'...' if len(h_walls) > 10 else ''}")
    if v_walls:
        print(f"V-Walls ({len(v_walls)}): {', '.join(sorted(v_walls)[:10])}{'...' if len(v_walls) > 10 else ''}")


# ==============================================================================
# Main Game Loop
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Play Quoridor against the AI")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use LLM for AI evaluation")
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS simulations per AI move")
    parser.add_argument("--ai-player", type=int, default=1, choices=[0, 1],
                        help="AI player (0=first, 1=second)")
    parser.add_argument("--show-moves", action="store_true",
                        help="Show legal moves each turn")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show memory retrieval debug info")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  QUORIDOR: HUMAN vs AI")
    print("=" * 60)
    print("\nControls:")
    print("  Move:    e2, d3, etc.")
    print("  H-Wall:  h c3 (horizontal wall at C3)")
    print("  V-Wall:  v d4 (vertical wall at D4)")
    print("  Help:    'help' or '?'")
    print("  Quit:    'quit' or 'q'")
    print("=" * 60 + "\n")
    
    # Load game
    print("Loading Quoridor...")
    game = load_game()
    
    # Create agent factory with unified memory
    factory = QuoridorAgentFactory(
        memory_file=DEFAULT_MEMORY_DB,
        use_llm=args.use_llm,
        verbose=args.verbose
    )
    
    # Create AI bot
    ai_bot = factory.create_bot(
        game=game,
        player_id=args.ai_player,
        num_simulations=args.simulations,
        record_experience=True
    )
    
    human_player = 1 - args.ai_player
    # OpenSpiel Quoridor: player 0 controls '0' piece, player 1 controls '@' piece!
    print(f"\nYou are Player {human_player + 1} ({'0' if human_player == 0 else '@'})")
    print(f"AI is Player {args.ai_player + 1} ({'0' if args.ai_player == 0 else '@'})")
    print(f"AI uses {args.simulations} MCTS simulations per move\n")
    
    # Game loop
    state = game.new_initial_state()
    turn = 0
    
    try:
        while not state.is_terminal():
            turn += 1
            current_player = state.current_player()
            
            print("\n" + "-" * 50)
            print(f"Turn {turn}")
            print("-" * 50)
            print(state)
            
            if current_player == human_player:
                print(f"\n>>> Your turn (Player {current_player + 1})")
                
                if args.show_moves:
                    display_legal_moves(state)
                
                while True:
                    try:
                        user_input = input("\nEnter move: ").strip()
                    except EOFError:
                        print("\n[Game ended]")
                        return
                    
                    if user_input.lower() in ['quit', 'q', 'exit']:
                        print("\n[Game ended by user]")
                        return
                    
                    if user_input.lower() in ['help', '?']:
                        display_legal_moves(state)
                        continue
                    
                    action = parse_human_move(state, user_input)
                    
                    if action == -1:
                        print(f"Invalid move: '{user_input}'")
                        print("Type 'help' or '?' to see legal moves")
                        continue
                    
                    action_str = state.action_to_string(current_player, action)
                    print(f"Playing: {action_str}")
                    state.apply_action(action)
                    break
            else:
                print(f"\n>>> AI thinking (Player {current_player + 1})...")
                action = ai_bot.step(state)
                action_str = state.action_to_string(current_player, action)
                print(f"AI plays: {action_str}")
                state.apply_action(action)
        
        # Game over
        print("\n" + "=" * 50)
        print("GAME OVER!")
        print("=" * 50)
        print(state)
        
        returns = state.returns()
        if returns[human_player] > returns[args.ai_player]:
            print("\n🎉 YOU WIN! 🎉")
            outcome = -1.0  # AI lost
        elif returns[args.ai_player] > returns[human_player]:
            print("\n🤖 AI WINS!")
            outcome = 1.0  # AI won
        else:
            print("\n🤝 DRAW!")
            outcome = 0.0
        
        factory.end_game(outcome)
        
        stats = factory.get_statistics()
        print(f"\nFinal: You={returns[human_player]}, AI={returns[args.ai_player]}")
        print(f"Memory: {stats['total']} experiences, {stats['win_rate']:.1f}% AI win rate")
    
    except KeyboardInterrupt:
        print("\n\n[Game interrupted]")
        factory.clear_buffer()


if __name__ == "__main__":
    main()
