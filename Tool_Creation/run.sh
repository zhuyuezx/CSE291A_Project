python -c "
from mcts import MCTSEngine
from mcts.games import Sokoban
engine = MCTSEngine(Sokoban('level1'), iterations=200, logging=True)
result = engine.play_game()
print('Log file:', result.get('log_file'))
print('Solved:', result['solved'])
" 2>&1

