"""
Compatibility entrypoint for generating all visualization outputs.

This wraps `visualization.run_all` so you can run either:
- python -m visualization.run_all
- python -m visualization.run_app
"""

from __future__ import annotations

from .run_all import main


if __name__ == "__main__":
    main()

