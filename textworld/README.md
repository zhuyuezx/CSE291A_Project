# TextWorld Workspace

This folder groups the TextWorld-specific additions without removing the original files.

Structure:
- games/: symbolic TextWorld game implementations copied from mcts/games
- game_infos/: TextWorld rule descriptions copied from LLM/game_infos
- heuristics/: fixed-name best heuristics for each MCTS phase
- tools/: TextWorld-specific runners and comparison scripts

Notes:
- Original files remain in their canonical locations under `mcts/`, `LLM/`, `tools/`, and `MCTS_tools/`.
- The copied scripts in `textworld/tools/` are adjusted so they still import the project root modules.
