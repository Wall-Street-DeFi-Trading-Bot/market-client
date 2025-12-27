# src/market_data_client/arbitrage/executor.py
"""
TradeExecutor: takes ArbitrageOpportunity objects and sends orders
through ExchangeClient implementations (LIVE / PAPER / DEMO).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

from .arbitrage_detector import ArbitrageOpportunity
from .exchange import ExchangeClient, TradeResult, OrderSide
from .risk import RiskManager

logger = logging.getLogger(__name__)


class TradeExecutor:
    """
    TradeExecutor wires ArbitrageOpportunity objects to concrete ExchangeClient
    implementations and runs a simple two-leg arbitrage (BUY then SELL).

    This class does not perform any scheduling or opportunity search.
    It only executes a single opportunity when asked.
    """

    def __init__(
        self,
        exchange_clients: Dict[Tuple[str, str], ExchangeClient],
        risk_manager: RiskManager,
    ) -> None:
        """
        Args:
            exchange_clients: Mapping (exchange, instrument) -> ExchangeClient.
            risk_manager: RiskManager instance used for sizing and filtering.
        """
        self._clients = exchange_clients
        self._risk = risk_manager

    async def execute_opportunity(
        self,
        opp: ArbitrageOpportunity,
    ) -> Tuple[TradeResult, TradeResult]:
        """
        Execute a single arbitrage opportunity.

        Flow:
            1) Ask risk manager if the opportunity is acceptable.
            2) Compute trade quantity via risk manager.
            3) Execute BUY leg on (buy_exchange, buy_instrument).
            4) Execute SELL leg on (sell_exchange, sell_instrument).

        Raises:
            ValueError: if the opportunity is rejected, sizing is invalid,
                        required clients are missing, or any leg fails.
        """
        if not self._risk.is_acceptable(opp):
            logger.info(
                "[EXECUTOR] Opportunity rejected by risk manager: "
                "%s(%s) -> %s(%s)",
                opp.buy_exchange,
                opp.buy_instrument,
                opp.sell_exchange,
                opp.sell_instrument,
            )
            raise ValueError("Opportunity rejected by risk manager")

        qty = self._risk.compute_trade_quantity(opp)
        if qty <= 0:
            raise ValueError("Risk manager returned non-positive trade size")

        buy_key = (opp.buy_exchange, opp.buy_instrument)
        sell_key = (opp.sell_exchange, opp.sell_instrument)

        buy_client = self._clients.get(buy_key)
        sell_client = self._clients.get(sell_key)

        if buy_client is None:
            raise ValueError(f"No exchange client configured for {buy_key}")
        if sell_client is None:
            raise ValueError(f"No exchange client configured for {sell_key}")

        logger.info(
            "[EXECUTOR] Executing arbitrage: %s | buy on %s(%s) @ %.4f, "
            "sell on %s(%s) @ %.4f, qty=%.6f",
            opp.symbol,
            opp.buy_exchange,
            opp.buy_instrument,
            opp.buy_price,
            opp.sell_exchange,
            opp.sell_instrument,
            opp.sell_price,
            qty,
        )

        # BUY leg
        try:
            buy_trade = await buy_client.create_market_order(
                symbol=opp.symbol,
                side=OrderSide.BUY,
                quantity=qty,
                price=opp.buy_price,
            )
        except Exception as exc:  # defensive
            logger.warning(
                "[EXECUTOR] BUY leg exception for %s on %s(%s): %s",
                opp.symbol,
                opp.buy_exchange,
                opp.buy_instrument,
                exc,
            )
            raise ValueError("BUY leg failed with exception") from exc

        if not buy_trade:
            logger.warning(
                "[EXECUTOR] BUY leg failed for %s on %s(%s): no trade result",
                opp.symbol,
                opp.buy_exchange,
                opp.buy_instrument,
            )
            raise ValueError("BUY leg failed (no trade result)")

        if hasattr(buy_trade, "ok") and not getattr(buy_trade, "ok"):
            logger.warning(
                "[EXECUTOR] BUY leg failed for %s on %s(%s): %s",
                opp.symbol,
                opp.buy_exchange,
                opp.buy_instrument,
                getattr(buy_trade, "error", "ok == False"),
            )
            raise ValueError("BUY leg failed (ok == False)")

        if hasattr(buy_trade, "success") and not getattr(buy_trade, "success"):
            logger.warning(
                "[EXECUTOR] BUY leg failed for %s on %s(%s): %s | %s",
                opp.symbol,
                opp.buy_exchange,
                opp.buy_instrument,
                getattr(buy_trade, "message", "success == False"),
                self._summarize_pancake_reverts(buy_trade),
            )
            raise ValueError("BUY leg failed (success == False)")


        # SELL leg
        try:
            sell_trade = await sell_client.create_market_order(
                symbol=opp.symbol,
                side=OrderSide.SELL,
                quantity=qty,
                price=opp.sell_price,
            )
        except Exception as exc:  # defensive
            logger.warning(
                "[EXECUTOR] SELL leg exception for %s on %s(%s): %s",
                opp.symbol,
                opp.sell_exchange,
                opp.sell_instrument,
                exc,
            )
            raise ValueError("SELL leg failed with exception") from exc

        if not sell_trade:
            logger.warning(
                "[EXECUTOR] SELL leg failed for %s on %s(%s): no trade result",
                opp.symbol,
                opp.sell_exchange,
                opp.sell_instrument,
            )
            raise ValueError("SELL leg failed (no trade result)")

        if hasattr(sell_trade, "ok") and not getattr(sell_trade, "ok"):
            logger.warning(
                "[EXECUTOR] SELL leg failed for %s on %s(%s): %s",
                opp.symbol,
                opp.sell_exchange,
                opp.sell_instrument,
                getattr(sell_trade, "error", "ok == False"),
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

        logger.info(
            "[EXECUTOR] Done: buy @ %.4f on %s, sell @ %.4f on %s",
            buy_trade.price,
            buy_trade.exchange,
            sell_trade.price,
            sell_trade.exchange,
        )

        return buy_trade, sell_trade

    @staticmethod
    def _summarize_pancake_reverts(trade: TradeResult) -> str:
        """
        Summarize per-block revert reasons from Pancake demo metadata.
        """
        meta = (trade.metadata or {}).get("pancake_demo") or {}
        rows = meta.get("per_block_results") or []
        fails = [r for r in rows if int(r.get("status", 0) or 0) != 1]

        if not fails:
            return "no per-block failures"

        parts = []
        for r in fails[:5]:
            reason = r.get("revert_reason")
            if isinstance(reason, str) and len(reason) > 180:
                reason = reason[:180] + "..."
            parts.append(
                f"block={r.get('fork_block')} status={r.get('status')} reason={reason}"
            )
        return " | ".join(parts)
