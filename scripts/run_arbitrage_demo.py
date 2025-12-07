# scripts/run_arbitrage_demo.py
"""
CLI entrypoint to run the demo-trading arbitrage example.

Usage:
    python scripts/run_arbitrage_demo.py
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXAMPLE = ROOT / "example"

for p in (SRC, EXAMPLE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from arbitrage_demo_example import main  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
