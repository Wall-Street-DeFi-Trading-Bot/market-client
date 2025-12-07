import asyncio
from datetime import datetime
from typing import Any, Dict, Tuple

import sys
from pathlib import Path

# Ensure "src" is on sys.path when running directly with python.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_data_client.arbitrage.arbitrage_detector import ArbitrageOpportunity
from market_data_client.arbitrage.executor import TradeExecutor
from market_data_client.arbitrage.exchange import (
    ExchangeClient,
    OrderSide,
    TradeResult,
)


class DummyRiskManager:
    """Simple risk manager that always accepts and uses a fixed size."""

    def is_acceptable(self, opp: ArbitrageOpportunity) -> bool:
        return True

    def compute_trade_quantity(self, opp: ArbitrageOpportunity) -> float:
        return 1.0


class RecordingClient(ExchangeClient):
    """ExchangeClient that records create_market_order calls."""

    def __init__(self, name: str, instrument: str) -> None:
        super().__init__(name=name, instrument=instrument, state=object())
        self.calls: list[Dict[str, Any]] = []

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        order_type,
    ) -> TradeResult:
        raise NotImplementedError("create_order should not be called in this test")

    async def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
    ) -> TradeResult:
        self.calls.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
            }
        )
        return TradeResult(
            exchange=self.name,
            instrument=self.instrument,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            fee=0.0,
            base_delta=0.0,
            quote_delta=0.0,
            success=True,
        )


def test_executor_uses_orderside_and_price() -> None:
    buy_client = RecordingClient("BuyEx", "spot")
    sell_client = RecordingClient("SellEx", "spot")

    clients: Dict[Tuple[str, str], ExchangeClient] = {
        ("BuyEx", "spot"): buy_client,
        ("SellEx", "spot"): sell_client,
    }

    risk = DummyRiskManager()
    executor = TradeExecutor(exchange_clients=clients, risk_manager=risk)

    opp = ArbitrageOpportunity(
        symbol="ETHUSDT",
        buy_exchange="BuyEx",
        sell_exchange="SellEx",
        buy_price=100.0,
        sell_price=101.0,
        buy_instrument="spot",
        sell_instrument="spot",
        price_diff=1.0,
        price_diff_pct=1.0,
        buy_fee=0.0005,
        sell_fee=0.0007,
        buy_slippage=0.0,
        sell_slippage=0.0,
        net_profit_pct=0.5,
        net_profit_per_unit=0.5,
        buy_latency_ms=None,
        sell_latency_ms=None,
        timestamp=datetime.utcnow(),
    )

    asyncio.run(executor.execute_opportunity(opp))

    assert len(buy_client.calls) == 1
    assert len(sell_client.calls) == 1

    buy_call = buy_client.calls[0]
    sell_call = sell_client.calls[0]

    # Check that the executor uses OrderSide enum instead of raw strings.
    assert buy_call["side"] is OrderSide.BUY
    assert sell_call["side"] is OrderSide.SELL

    # Check that the executor forwards price to the client.
    assert buy_call["price"] == opp.buy_price
    assert sell_call["price"] == opp.sell_price


if __name__ == "__main__":
    test_executor_uses_orderside_and_price()
    print("test_executor_uses_orderside_and_price: OK")
