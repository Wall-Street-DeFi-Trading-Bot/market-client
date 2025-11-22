# src/market_data_client/arbitrage/exchange.py
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING, Tuple

from .config import ExecutionMode

if TYPE_CHECKING:
    # Only used for type hints – avoids circular import at runtime
    from .state import BotState

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """Order direction."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type (kept minimal for now)."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class TradeResult:
    """
    Result of a single executed order (paper or live).

    Fields:
        exchange: Exchange name, e.g. "Binance"
        instrument: Instrument type, e.g. "spot", "perpetual", "swap"
        symbol: Trading symbol, e.g. "BNBUSDT"
        side: "BUY" or "SELL" (stored as a plain string)
        quantity: Filled quantity in base asset units
        price: Executed price
        fee: Absolute fee amount in quote currency (e.g. USDT)
        base_delta: Change in base asset balance (e.g. +BNB / -BNB)
        quote_delta: Change in quote asset balance (e.g. -USDT / +USDT)
        success: Whether the order was successfully applied
        message: Human-readable status (e.g. "paper BUY filled" / error reason)
    """
    exchange: str
    instrument: str
    symbol: str
    side: str
    quantity: float
    price: float
    fee: float
    base_delta: float
    quote_delta: float
    success: bool
    message: str = ""


class ExchangeClient(abc.ABC):
    """
    Abstract base class for all exchange clients (Binance / PancakeSwap / Paper).

    TradeExecutor still calls create_market_order(...), so we provide a
    compatibility method that delegates to create_order(...).
    """

    def __init__(self, name: str, instrument: str, mode: ExecutionMode) -> None:
        self.name = name
        self.instrument = instrument
        self.mode = mode

    @abc.abstractmethod
    async def get_balance(self, asset: str) -> float:
        """Return current balance for the given asset."""
        raise NotImplementedError

    @abc.abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> TradeResult:
        """
        Create an order on the exchange.

        For PaperExchangeClient, this only updates in-memory balances.
        For real clients, this should call the actual exchange / chain API.
        """
        raise NotImplementedError

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        *,
        price: Optional[float] = None,
        price_hint: Optional[float] = None,
        quote_asset: str = "USDT",
        **_: object,
    ) -> TradeResult:
        """
        Backwards-compatible helper for older code that calls
        create_market_order(symbol, side, quantity, price_hint=...).

        Args:
            symbol: Trading symbol, e.g. "BNBUSDT"
            side: "buy" / "sell" (case-insensitive)
            quantity: Trade size in base asset
            price: Explicit price to use (if given)
            price_hint: Optional price hint (e.g. best bid/ask from detector)
            quote_asset: Currently unused here, kept only for compatibility

        This maps the string side → OrderSide enum and calls
        create_order(..., order_type=MARKET).
        """
        # Choose effective price: explicit price > price_hint
        eff_price = price if price is not None else price_hint

        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        return await self.create_order(
            symbol=symbol,
            side=side_enum,
            quantity=quantity,
            price=eff_price,
            order_type=OrderType.MARKET,
        )


class PaperExchangeClient(ExchangeClient):
    """
    In-memory paper trading client.

    - Uses BotState to manage balances per (exchange, instrument).
    - Records all trades into BotState.executed_trades.
    - Does NOT talk to real exchanges.
    """

    def __init__(
        self,
        name: str,
        instrument: str,
        state: "BotState",
        fee_rate: float = 0.0005,  # 5 bps default fee
    ) -> None:
        super().__init__(name=name, instrument=instrument, mode=ExecutionMode.PAPER)
        self._state = state
        self._fee_rate = fee_rate


    def _split_symbol(self, symbol: str) -> tuple[str, str]:
        """
        Naively split a symbol into (base, quote).

        Examples:
            "BNBUSDT" -> ("BNB", "USDT")

        You can extend this to handle more complex symbols
        (BUSD, USDC, etc.) if needed.
        """
        if symbol.endswith("USDT"):
            return symbol[:-4], "USDT"
        # Fallback: assume first 3 chars are base, the rest are quote
        return symbol[:3], symbol[3:]

    async def get_balance(self, asset: str) -> float:
        account = self._state.get_or_create_account(self.name, self.instrument)
        return account.balances.get(asset, 0.0)

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> TradeResult:
        """
        Simulate an order on a paper account.

        Behavior is different for spot/swap vs perpetual:

        - spot / swap (inventory-based):
            BUY:
              * withdraw quote (e.g. USDT) = quantity * price + fee
              * deposit base (e.g. BNB)    = quantity
            SELL:
              * withdraw base  = quantity
              * deposit quote  = quantity * price - fee

        - perpetual (margin-based, simplified):
            BUY  (open / increase long):
              * require quote balance as margin
              * withdraw quote = quantity * price + fee
              * base inventory is NOT required (no withdraw)
            SELL (open / increase short):
              * does NOT check base inventory
              * deposit quote = quantity * price - fee

        In both cases:
            - fee is charged in quote currency
            - TradeResult.base_delta / quote_delta reflect balance changes
              in the account's wallet (BotState account.balances).
        """
        if order_type is not OrderType.MARKET:
            # For now, we only simulate market-style fills.
            logger.warning(
                "[PAPER] Non-market order type requested (%s); treating as MARKET.",
                order_type.value,
            )

        if price is None or price <= 0.0:
            msg = "paper order failed: invalid or missing price"
            logger.warning("[PAPER] %s", msg)
            result = TradeResult(
                exchange=self.name,
                instrument=self.instrument,
                symbol=symbol,
                side=side.value,
                quantity=quantity,
                price=0.0,
                fee=0.0,
                base_delta=0.0,
                quote_delta=0.0,
                success=False,
                message=msg,
            )
            self._state.record_trade(result)
            return result

        base, quote = self._split_symbol(symbol)
        account = self._state.get_or_create_account(self.name, self.instrument)

        notional = quantity * price
        fee = notional * self._fee_rate

        base_delta = 0.0
        quote_delta = 0.0
        success = False
        msg = ""

        if self.instrument == "perpetual":
            if side is OrderSide.BUY:
                # Open / increase a long position:
                # - we only care about quote (USDT) as margin source.
                total_cost = notional + fee
                free_quote = account.balances.get(quote, 0.0)

                if free_quote + 1e-12 < total_cost:
                    fee = 0.0
                    msg = (
                        f"paper PERP BUY failed: insufficient {quote} "
                        f"(have {free_quote}, need {total_cost})"
                    )
                    logger.warning("[PAPER] %s", msg)
                    success = False
                else:
                    account.balances[quote] = free_quote - total_cost
                    # We do not touch base inventory; position is implicit.
                    base_delta = 0.0
                    quote_delta = -total_cost
                    success = True
                    msg = "paper PERP BUY filled (open/increase long)"

            else:  # side is SELL (open / increase short)
                # In perps you can short without holding the base asset.
                proceeds = notional - fee
                current_quote = account.balances.get(quote, 0.0)
                account.balances[quote] = current_quote + proceeds

                base_delta = 0.0
                quote_delta = proceeds
                success = True
                msg = "paper PERP SELL filled (open/increase short)"

        else:
            if side is OrderSide.BUY:
                total_cost = notional + fee
                free_quote = account.balances.get(quote, 0.0)

                if free_quote + 1e-12 < total_cost:
                    fee = 0.0
                    msg = (
                        f"paper SPOT BUY failed: insufficient {quote} "
                        f"(have {free_quote}, need {total_cost})"
                    )
                    logger.warning("[PAPER] %s", msg)
                    success = False
                else:
                    # Spend quote, receive base
                    account.balances[quote] = free_quote - total_cost
                    account.balances[base] = account.balances.get(base, 0.0) + quantity

                    base_delta = quantity
                    quote_delta = -total_cost
                    success = True
                    msg = "paper SPOT BUY filled"

            else:  # SELL on spot/swap
                current_base = account.balances.get(base, 0.0)
                if current_base + 1e-12 < quantity:
                    fee = 0.0
                    msg = (
                        f"paper SPOT SELL failed: insufficient {base} "
                        f"(have {current_base}, need {quantity})"
                    )
                    logger.warning("[PAPER] %s", msg)
                    success = False
                else:
                    # Reduce base inventory, receive quote
                    account.balances[base] = current_base - quantity
                    proceeds = notional - fee
                    account.balances[quote] = account.balances.get(quote, 0.0) + proceeds

                    base_delta = -quantity
                    quote_delta = proceeds
                    success = True
                    msg = "paper SPOT SELL filled"

        result = TradeResult(
            exchange=self.name,
            instrument=self.instrument,
            symbol=symbol,
            side=side.value,
            quantity=quantity,
            price=price,
            fee=fee if success else 0.0,
            base_delta=base_delta,
            quote_delta=quote_delta,
            success=success,
            message=msg,
        )

        # Record into BotState for later reporting / PnL analysis
        self._state.record_trade(result)

        logger.info(
            "[PAPER] %s(%s) %s %s qty=%.6f @ %.4f fee=%.6f Δbase=%.6f Δquote=%.6f success=%s",
            self.name,
            self.instrument,
            side.value,
            symbol,
            quantity,
            price,
            result.fee,
            base_delta,
            quote_delta,
            success,
        )
        logger.info(
            "[PAPER] New balances for %s(%s): %s",
            self.name,
            self.instrument,
            account.balances,
        )

        return result

class BinanceExchangeClient(ExchangeClient):
    """
    Skeleton for real Binance client.

    Right now this only defines the interface. You will later plug in
    ccxt / python-binance / binance-connector etc. to actually send orders.
    """

    def __init__(self, instrument: str, mode: ExecutionMode) -> None:
        super().__init__(name="Binance", instrument=instrument, mode=mode)
        # TODO: inject real Binance session / API key here

    async def get_balance(self, asset: str) -> float:
        raise NotImplementedError("BinanceExchangeClient.get_balance is not implemented yet")

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> TradeResult:
        raise NotImplementedError("BinanceExchangeClient.create_order is not implemented yet")


class PancakeSwapExchangeClient(ExchangeClient):
    """
    Skeleton for PancakeSwap client (on-chain DEX).

    For LIVE mode you will later integrate web3 / your swap executor
    (e.g. router.swapExactTokensForTokens).
    """

    def __init__(self, exchange_name: str, instrument: str, mode: ExecutionMode) -> None:
        super().__init__(name=exchange_name, instrument=instrument, mode=mode)
        # TODO: inject web3 client / router address etc.

    async def get_balance(self, asset: str) -> float:
        raise NotImplementedError("PancakeSwapExchangeClient.get_balance is not implemented yet")

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> TradeResult:
        raise NotImplementedError("PancakeSwapExchangeClient.create_order is not implemented yet")
