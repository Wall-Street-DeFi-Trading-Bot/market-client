# src/market_data_client/arbitrage/__init__.py
from __future__ import annotations

"""
Arbitrage utilities for market_data_client.

Public API:
- ArbitrageDetector / ArbitrageOpportunity / run_arbitrage_detector
- BotConfig / ExecutionMode (for bot config)
"""

from .arbitrage_detector import (
    ArbitrageDetector,
    ArbitrageOpportunity,
    run_arbitrage_detector,
)
from .config import BotConfig, ExecutionMode

__all__ = [
    "ArbitrageDetector",
    "ArbitrageOpportunity",
    "run_arbitrage_detector",
    "BotConfig",
    "ExecutionMode",
]
