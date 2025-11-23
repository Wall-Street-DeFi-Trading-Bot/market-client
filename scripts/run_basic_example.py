# scripts/run_basic_example.py
"""
CLI entrypoint to run the basic MarketDataClient example.

Usage:
    python scripts/run_basic_example.py
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

from basic_example import main  # type: ignore


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✅ Stopped basic example")
