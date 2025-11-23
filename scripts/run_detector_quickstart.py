# scripts/run_detector_quickstart.py
"""
CLI entrypoint to run the arbitrage detector quickstart example.

Usage:
    python scripts/run_detector_quickstart.py
or:
    PYTHONPATH=./src:./example python scripts/run_detector_quickstart.py
"""

import asyncio
import sys
from pathlib import Path

# Add src/ and example/ to sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXAMPLE = ROOT / "example"

for p in (SRC, EXAMPLE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from arbitrage_detector_quickstart import main  # type: ignore


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✅ Stopped detector quickstart example")
