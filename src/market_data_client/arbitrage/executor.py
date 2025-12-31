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
    """

    def __init__(
        self,
        exchange_clients: Dict[Tuple[str, str], ExchangeClient],
        risk_manager: RiskManager,
    ) -> None:
        self._clients = exchange_clients
        self._risk = risk_manager

    async def execute_opportunity(
        self,
        opp: ArbitrageOpportunity,
    ) -> Tuple[TradeResult, TradeResult]:
        if not self._risk.is_acceptable(opp):
            logger.info(
                "[EXECUTOR] Opportunity rejected: %s(%s) -> %s(%s)",
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
            "[EXECUTOR] Executing arbitrage: %s | buy on %s(%s) @ %.6f, "
            "sell on %s(%s) @ %.6f, qty=%.8f",
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
                # For DEX demo clients (Pancake), this clarifies semantics:
                # "price_hint" is the reference price used for return_vs_hint.
                price_hint=opp.buy_price,
            )
        except Exception as exc:
            logger.warning(
                "[EXECUTOR] BUY leg exception for %s on %s(%s): %s",
                opp.symbol,
                opp.buy_exchange,
                opp.buy_instrument,
                exc,
            )
            raise ValueError("BUY leg failed with exception") from exc

        if not buy_trade:
            raise ValueError("BUY leg failed (no trade result)")

        if getattr(buy_trade, "success", True) is False:
            logger.warning(
                "[EXECUTOR] BUY leg failed for %s on %s(%s): %s | %s",
                opp.symbol,
                opp.buy_exchange,
                opp.buy_instrument,
                getattr(buy_trade, "message", "success == False"),
                self._summarize_pancake_reverts(buy_trade),
            )
            raise ValueError("BUY leg failed (success == False)")

        self._log_pancake_demo_summary("BUY", buy_trade)

        # SELL leg
        try:
            sell_trade = await sell_client.create_market_order(
                symbol=opp.symbol,
                side=OrderSide.SELL,
                quantity=qty,
                price=opp.sell_price,
                price_hint=opp.sell_price,
            )
        except Exception as exc:
            logger.warning(
                "[EXECUTOR] SELL leg exception for %s on %s(%s): %s",
                opp.symbol,
                opp.sell_exchange,
                opp.sell_instrument,
                exc,
            )
            raise ValueError("SELL leg failed with exception") from exc

        if not sell_trade:
            raise ValueError("SELL leg failed (no trade result)")

        if getattr(sell_trade, "success", True) is False:
            logger.warning(
                "[EXECUTOR] SELL leg failed for %s on %s(%s): %s | %s",
                opp.symbol,
                opp.sell_exchange,
                opp.sell_instrument,
                getattr(sell_trade, "message", "success == False"),
                self._summarize_pancake_reverts(sell_trade),
            )
            raise ValueError("SELL leg failed (success == False)")

        self._log_pancake_demo_summary("SELL", sell_trade)

        logger.info(
            "[EXECUTOR] Done: buy @ %.6f on %s, sell @ %.6f on %s",
            float(getattr(buy_trade, "price", 0.0) or 0.0),
            getattr(buy_trade, "exchange", "?"),
            float(getattr(sell_trade, "price", 0.0) or 0.0),
            getattr(sell_trade, "exchange", "?"),
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

    @staticmethod
    def _log_pancake_demo_summary(leg: str, trade: TradeResult) -> None:
        """
        Log a compact summary when the trade has pancake_demo metadata.

        If avg_* fields are missing, compute from per_block_results.
        """
        meta = (trade.metadata or {}).get("pancake_demo") or {}
        if not meta:
            return

        rows = meta.get("per_block_results") or []
        ok_rows = [r for r in rows if int(r.get("status", 0) or 0) == 1]

        # Counts
        ok = int(meta.get("ok_count", len(ok_rows)) or 0)
        fail = int(meta.get("fail_count", max(0, len(rows) - len(ok_rows))) or 0)

        # avg_fill: prefer meta, else compute from per-block fill_price
        avg_fill = meta.get("avg_fill_price", None)
        if not isinstance(avg_fill, (int, float)):
            fills = [
                float(r.get("fill_price"))
                for r in ok_rows
                if isinstance(r.get("fill_price"), (int, float))
            ]
            avg_fill = (sum(fills) / len(fills)) if fills else None

        # avg_return_vs_hint: prefer net return if present, else meta avg, else compute from per-block
        avg_rvh = None
        for k in (
            "avg_net_return_vs_hint_incl_gas",
            "avg_net_return_vs_hint_excl_gas",
            "avg_return_vs_hint",
        ):
            v = meta.get(k, None)
            if isinstance(v, (int, float)):
                avg_rvh = float(v)
                break

        if avg_rvh is None:
            rvhs = [
                float(r.get("return_vs_hint"))
                for r in ok_rows
                if isinstance(r.get("return_vs_hint"), (int, float))
            ]
            avg_rvh = (sum(rvhs) / len(rvhs)) if rvhs else None

        logger.info(
            "[EXECUTOR] %s pancake_demo: ok=%s fail=%s avg_fill=%s avg_return_vs_hint=%s",
            leg,
            ok,
            fail,
            f"{float(avg_fill):.8f}" if isinstance(avg_fill, (int, float)) else "n/a",
            f"{float(avg_rvh):+.6%}" if isinstance(avg_rvh, (int, float)) else "n/a",
        )
