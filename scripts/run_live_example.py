# scripts/run_live_example.py
"""
CLI entrypoint to run the live_example in ./example.

Usage:
    python scripts/run_live_example.py
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXAMPLE = ROOT / "example"

# Make src/ and example/ importable
for p in (SRC, EXAMPLE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from live_example import main  # type: ignore


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✅ Stopped live_example")
