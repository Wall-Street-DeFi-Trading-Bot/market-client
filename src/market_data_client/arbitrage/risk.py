# src/market_data_client/arbitrage/risk.py
"""
Simple risk management and position sizing for the arbitrage bot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import BotConfig
from .arbitrage_detector import ArbitrageOpportunity
from .exchange import ExchangeClient

logger = logging.getLogger(__name__)


@dataclass
class RiskManager:
    """
    Very small risk manager used by TradeExecutor.

    Responsibilities:
      - Decide whether an opportunity is tradable.
      - Decide how big the trade should be (position sizing).
    """

    config: BotConfig
    max_latency_ms: float = 800.0
    max_slippage_bps: float = 30.0

    def is_acceptable(self, opp: ArbitrageOpportunity) -> bool:
        """
        Check if this opportunity passes basic risk filters.
        """
        if opp.net_profit_pct < self.config.min_profit_pct:
            logger.debug(
                f"Rejecting opp {opp.symbol}: net {opp.net_profit_pct:.3f}% "
                f"< min {self.config.min_profit_pct:.3f}%"
            )
            return False

        if opp.buy_latency_ms and opp.buy_latency_ms > self.max_latency_ms:
            logger.debug(
                f"Rejecting opp {opp.symbol}: buy latency {opp.buy_latency_ms:.1f}ms "
                f"> max {self.max_latency_ms}ms"
            )
            return False

        if opp.sell_latency_ms and opp.sell_latency_ms > self.max_latency_ms:
            logger.debug(
                f"Rejecting opp {opp.symbol}: sell latency {opp.sell_latency_ms:.1f}ms "
                f"> max {self.max_latency_ms}ms"
            )
            return False

        total_slippage = opp.buy_slippage + opp.sell_slippage
        if total_slippage > self.max_slippage_bps:
            logger.debug(
                f"Rejecting opp {opp.symbol}: total slippage {total_slippage:.2f}bps "
                f"> max {self.max_slippage_bps}bps"
            )
            return False

        return True

    def compute_trade_quantity(self, opp: ArbitrageOpportunity) -> float:
        """
        Compute base asset quantity given target notional in quote currency.

        Example: notional 50 USDT, buy price 250 USDT => qty = 0.2.
        """
        price = opp.buy_price
        if price <= 0:
            return 0.0

        notional = self.config.trade_notional_usd
        qty = notional / price
        return max(qty, 0.0)
