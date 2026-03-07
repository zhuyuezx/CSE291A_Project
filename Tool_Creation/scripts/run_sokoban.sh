#!/usr/bin/env bash
# run_sokoban.sh — Convenience wrapper to run the Sokoban MCTS example.
#
# Usage:
#   ./run_sokoban.sh                              # level1, 200 iters
#   ./run_sokoban.sh --level level3
#   ./run_sokoban.sh --level level5 --iterations 500 --games 3
#   ./run_sokoban.sh --no-prompt --verbose
#
# All arguments are forwarded to run_sokoban.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python3 run_sokoban.py "$@"
