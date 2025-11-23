from __future__ import annotations

"""
Root package for the market_data_client library.

Re-exports the main market data client types for convenience.
"""

from .market_data_client import MarketDataClient, CexConfig, DexConfig

__all__ = ["MarketDataClient", "CexConfig", "DexConfig"]
