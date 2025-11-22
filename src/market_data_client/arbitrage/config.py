# src/market_data_client/arbitrage/config.py
"""
Configuration objects for the arbitrage bot.

This module only defines small dataclasses for configuration.
You can either construct them manually in your strategy scripts
or add helpers that load from environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class ExecutionMode(str, Enum):
    """Execution mode for the bot."""
    PAPER = "paper"   # no real orders, simulate fills in memory
    LIVE = "live"     # send real orders to exchanges


@dataclass
class BotConfig:
    """
    High-level configuration for the arbitrage bot.

    Attributes:
        symbols: List of trading pairs like ["BNBUSDT", "ETHUSDT"].
        exchanges: List of (exchange_name, instrument) tuples.
                   Example: [("Binance", "spot"), ("Binance", "perpetual"),
                             ("PancakeSwapV2", "swap")]
        min_profit_pct: Minimum net profit % to trade (already checked by detector,
                        but risk manager may re-check).
        trade_notional_usd: Target notional per trade (in quote currency, e.g. USDT).
        nats_url: NATS URL for MarketDataClient.
        scan_interval: Seconds between scans.
        mode: PAPER or LIVE.
    """
    symbols: List[str]
    exchanges: List[Tuple[str, str]]
    min_profit_pct: float = 0.1
    trade_notional_usd: float = 50.0
    nats_url: str = "nats://127.0.0.1:4222"
    scan_interval: float = 5.0
    mode: ExecutionMode = ExecutionMode.PAPER

    # Optional: per-exchange initial balances for paper trading
    # key = (exchange, instrument) -> {asset: balance}
    paper_initial_balances: Dict[Tuple[str, str], Dict[str, float]] = field(
        default_factory=dict
    )
