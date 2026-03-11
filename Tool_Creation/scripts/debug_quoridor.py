"""Debug script to diagnose side-movement bug in Quoridor MCTS."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcts.games.quoridor import Quoridor, PawnMove, HWall, VWall
from mcts import MCTSEngine

game = Quoridor()
engine = MCTSEngine(game, iterations=100, max_rollout_depth=200, logging=False)
engine.load_tool("simulation", str(ROOT / "MCTS_tools" / "simulation" / "generic_simulation.py"))

result = engine.play_game(verbose=False)
moves = result["moves"]

# Replay to trace pawn positions
state = game.new_initial_state()
for i, action in enumerate(moves):
    p = state.current_player()
    state.apply_action(action)
    if i < 40 or i % 10 == 0:
        if isinstance(action, PawnMove):
            print(f"Move {i+1:3d}: P{p} -> ({action.row},{action.col})  "
                  f"pos=[P0@{state.pawn_pos[0]} P1@{state.pawn_pos[1]}]")
        elif isinstance(action, (HWall, VWall)):
            print(f"Move {i+1:3d}: P{p} wall {action}")

print(f"\nTotal moves: {len(moves)}")
print(f"Final: P0@{state.pawn_pos[0]} P1@{state.pawn_pos[1]}")
print(f"Goal rows: P0={state.goal_row[0]} P1={state.goal_row[1]}")
print(f"Returns: {state.returns()}")
if state._winner is not None:
    print(f"Winner: P{state._winner}")
else:
    print("Draw")
