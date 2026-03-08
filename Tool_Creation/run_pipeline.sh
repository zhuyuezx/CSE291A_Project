#!/usr/bin/env bash
# run_pipeline.sh — One-step caller for the full MCTS + LLM optimization pipeline.
#
# Usage (from Tool_Creation/):
#   ./run_pipeline.sh                  # default config
#   ./run_pipeline.sh --iters 20       # override iteration count
#   ./run_pipeline.sh --quiet          # minimal output
#
# All arguments are forwarded to scripts/run_pipeline.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python3 scripts/run_pipeline.py "$@"
