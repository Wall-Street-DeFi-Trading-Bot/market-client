from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ExecutionMode(str, Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"
    DEMO = "DEMO"


@dataclass
class BotConfig:
    """
    High-level configuration for the arbitrage bot.

    - symbols: symbols to scan, e.g. ["BTCUSDT", "BNBUSDT"]
    - exchanges: list of (exchange, instrument) tuples, e.g.
        [("Binance", "spot"), ("Binance", "perpetual"), ("PancakeSwapV2", "swap")]
    - mode: PAPER / LIVE / DEMO
    - nats_url: NATS endpoint where MarketDataClient subscribes
    - min_profit_pct: minimum net profit percentage to consider executing a trade
    - trade_notional_usd: target notional size per leg in USD terms
    - scan_interval: seconds between arbitrage scans
    - paper_initial_balances: seed balances per (exchange, instrument)
    - max_daily_loss_pct: optional circuit breaker
    - max_position_notional_usd: optional per-position size cap
    - enable_csv: if True, enable CSV logging in your strategy layer
    """

    symbols: List[str]
    exchanges: List[Tuple[str, str]]

    mode: ExecutionMode = ExecutionMode.PAPER
    nats_url: str = "nats://127.0.0.1:4222"

    min_profit_pct: float = 0.0
    trade_notional_usd: float = 100.0
    scan_interval: float = 0.5

    paper_initial_balances: Dict[Tuple[str, str], Dict[str, float]] = field(
        default_factory=dict
    )

    max_daily_loss_pct: Optional[float] = None
    max_position_notional_usd: Optional[float] = None

    enable_csv: bool = False
