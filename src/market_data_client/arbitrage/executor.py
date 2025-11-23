# src/market_data_client/arbitrage/excutor.py
"""
TradeExecutor: takes ArbitrageOpportunity objects and sends orders
through ExchangeClient implementations (LIVE or PAPER).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

from .arbitrage_detector import ArbitrageOpportunity
from .exchange import ExchangeClient, TradeResult
from .risk import RiskManager

logger = logging.getLogger(__name__)


class TradeExecutor:
    """
    Executes arbitrage trades given detected opportunities.

    It expects a mapping:
        key: (exchange_name, instrument)
        value: ExchangeClient instance (live or paper)
    """

    def __init__(
        self,
        exchange_clients: Dict[Tuple[str, str], ExchangeClient],
        risk_manager: RiskManager,
    ):
        self.exchange_clients = exchange_clients
        self.risk = risk_manager

    def _get_client(self, exchange: str, instrument: str) -> ExchangeClient:
        key = (exchange, instrument)
        if key not in self.exchange_clients:
            raise KeyError(f"No ExchangeClient registered for {exchange}({instrument})")
        return self.exchange_clients[key]

    async def execute_opportunity(self, opp: ArbitrageOpportunity) -> Tuple[TradeResult, TradeResult]:
        """
        Execute a single arbitrage: buy on buy_exchange, sell on sell_exchange.

        Returns:
            (buy_trade_result, sell_trade_result)

        Error handling:
          - If the opportunity is rejected by RiskManager, raises ValueError.
          - If BUY leg fails (no result / error flag), raises ValueError.
          - If SELL leg fails (no result / error flag), raises ValueError.
          In all failure cases this method returns early and the caller
          must treat the whole arbitrage as "not executed".
        """
        # 1) Risk check
        if not self.risk.is_acceptable(opp):
            logger.info(
                f"[EXECUTOR] Skipping opportunity {opp.symbol}: net={opp.net_profit_pct:.3f}% "
                f"below risk threshold."
            )
            raise ValueError("Opportunity rejected by risk manager")

        # 2) Compute quantity in base asset
        qty = self.risk.compute_trade_quantity(opp)
        if qty <= 0:
            logger.info(
                f"[EXECUTOR] Computed quantity is 0 for {opp.symbol}, skipping."
            )
            raise ValueError("Computed trade quantity <= 0")

        logger.info(
            f"[EXECUTOR] Executing arbitrage on {opp.symbol}: "
            f"qty={qty:.6f}, net={opp.net_profit_pct:.3f}% "
            f"{opp.buy_exchange}({opp.buy_instrument}) -> "
            f"{opp.sell_exchange}({opp.sell_instrument})"
        )

        buy_client = self._get_client(opp.buy_exchange, opp.buy_instrument)
        sell_client = self._get_client(opp.sell_exchange, opp.sell_instrument)

        # 3) BUY leg
        try:
            buy_trade = await buy_client.create_market_order(
                symbol=opp.symbol,
                side="buy",
                quantity=qty,
                price_hint=opp.buy_price,
                fee_rate=opp.buy_fee,
            )
        except Exception as exc:
            logger.warning(
                f"[EXECUTOR] BUY leg exception for {opp.symbol} on "
                f"{opp.buy_exchange}({opp.buy_instrument}): {exc}"
            )
            raise ValueError("BUY leg failed with exception") from exc

        # Treat None / False / ok=False as failure
        if not buy_trade:
            logger.warning(
                f"[EXECUTOR] BUY leg failed for {opp.symbol} on "
                f"{opp.buy_exchange}({opp.buy_instrument}): no trade result returned"
            )
            raise ValueError("BUY leg failed (no trade result)")

        if hasattr(buy_trade, "ok") and not getattr(buy_trade, "ok"):
            logger.warning(
                f"[EXECUTOR] BUY leg failed for {opp.symbol} on "
                f"{opp.buy_exchange}({opp.buy_instrument}): "
                f"{getattr(buy_trade, 'error', 'ok == False')}"
            )
            raise ValueError("BUY leg failed (ok == False)")

        if hasattr(buy_trade, "success") and not getattr(buy_trade, "success"):
            logger.warning(
                "[EXECUTOR] BUY leg failed for %s on %s(%s): %s",
                opp.symbol,
                opp.buy_exchange,
                opp.buy_instrument,
                getattr(buy_trade, "message", "success == False"),
            )
            raise ValueError("BUY leg failed (success == False)")
    
        # 4) SELL leg
        try:
            sell_trade = await sell_client.create_market_order(
                symbol=opp.symbol,
                side="sell",
                quantity=qty,
                price_hint=opp.sell_price,
                fee_rate=opp.sell_fee,
            )
        except Exception as exc:
            logger.warning(
                f"[EXECUTOR] SELL leg exception for {opp.symbol} on "
                f"{opp.sell_exchange}({opp.sell_instrument}): {exc}"
            )
            # In live trading you might want to hedge here.
            # For now: treat the whole arbitrage as failed.
            raise ValueError("SELL leg failed with exception") from exc

        if not sell_trade:
            logger.warning(
                f"[EXECUTOR] SELL leg failed for {opp.symbol} on "
                f"{opp.sell_exchange}({opp.sell_instrument}): no trade result returned"
            )
            raise ValueError("SELL leg failed (no trade result)")

        if hasattr(sell_trade, "ok") and not getattr(sell_trade, "ok"):
            logger.warning(
                f"[EXECUTOR] SELL leg failed for {opp.symbol} on "
                f"{opp.sell_exchange}({opp.sell_instrument}): "
                f"{getattr(sell_trade, 'error', 'ok == False')}"
            )
            raise ValueError("SELL leg failed (ok == False)")

        if hasattr(sell_trade, "success") and not getattr(sell_trade, "success"):
            logger.warning(
                "[EXECUTOR] SELL leg failed for %s on %s(%s): %s",
                opp.symbol,
                opp.sell_exchange,
                opp.sell_instrument,
                getattr(sell_trade, "message", "success == False"),
            )
            raise ValueError("SELL leg failed (success == False)")
        
        # 5) Both legs succeeded: log and return
        logger.info(
            f"[EXECUTOR] Done: buy @ {buy_trade.price:.4f} on {buy_trade.exchange}, "
            f"sell @ {sell_trade.price:.4f} on {sell_trade.exchange}"
        )

        return buy_trade, sell_trade

