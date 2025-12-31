from __future__ import annotations

import abc
import asyncio
import logging
import time
import json
import json as _json
import time as _time
import hmac as _hmac
import hashlib as _hashlib
from urllib import parse as _parse, request as _request
from urllib import error as _error

from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, List
from decimal import Decimal, ROUND_DOWN

from hexbytes import HexBytes
from web3 import Web3
from .config import ExecutionMode
from eth_account import Account
from market_data_client.arbitrage.pairs import resolve_base_quote


# Relax extraData validation for PoA-style chains such as BSC.
# Web3's default block formatters and validation middleware enforce
# extraData length <= 32 bytes. BSC headers use a longer extraData field,
# so we override both the method formatters and the validation function.
try:
    # 1) Override block formatters (if used by this web3 version)
    try:
        from web3._utils.method_formatters import BLOCK_FORMATTERS

        if "extraData" in BLOCK_FORMATTERS:
            BLOCK_FORMATTERS["extraData"] = lambda v: v
    except Exception:
        pass

    # 2) Override validation module so that extraData is accepted as-is
    try:
        from web3.middleware import validation as w3_validation  # type: ignore[attr-defined]

        # Bump the allowed length to something large
        if hasattr(w3_validation, "MAX_EXTRADATA_LENGTH"):
            w3_validation.MAX_EXTRADATA_LENGTH = 1024

        # Replace the checker with a no-op that still returns HexBytes
        if hasattr(w3_validation, "_check_extradata_length"):

            def _no_op_extradata(value: Any) -> Any:
                if not isinstance(value, (str, bytes, int)):
                    return value
                try:
                    return HexBytes(value)
                except Exception:
                    return value

            w3_validation._check_extradata_length = _no_op_extradata  # type: ignore[attr-defined]
    except Exception:
        pass
except Exception:
    # If anything above fails, we simply skip the PoA override.
    pass


def _apply_poa_middleware(w3: Web3) -> None:
    """
    Inject PoA middleware for chains like BNB Chain / Polygon / geth --dev.

    web3.py v6: geth_poa_middleware
    web3.py v7+: ExtraDataToPOAMiddleware
    """
    # v6 style
    try:
        from web3.middleware import geth_poa_middleware

        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        return
    except Exception:
        pass

    # v7 style
    try:
        from web3.middleware import ExtraDataToPOAMiddleware

        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        return
    except Exception:
        pass

    # some installs expose it under proof_of_authority
    try:
        from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware  # type: ignore

        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        return
    except Exception:
        pass


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
ABI_DIR = ROOT / "../../contracts/abi"
ERC20_ABI_PATH = ABI_DIR / "ERC20.json"
ERC20_ABI = json.loads(ERC20_ABI_PATH.read_text())

PERMIT2_ABI_PATH = ABI_DIR / "Permit2.json"
PERMIT2_ABI = json.loads(PERMIT2_ABI_PATH.read_text())
if isinstance(PERMIT2_ABI, dict) and "abi" in PERMIT2_ABI:
    PERMIT2_ABI = PERMIT2_ABI["abi"]

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"


class InsufficientBalanceError(RuntimeError):
    """Raised when an account does not have enough balance for a simulated trade."""


@dataclass
class TradeResult:
    """
    Generic trade result used across all exchange clients.
    """

    exchange: str
    instrument: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    base_delta: float
    quote_delta: float
    success: bool
    message: str = ""
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BinanceDemoParams:
    """
    Demo params for Binance.

    This supports separate credentials for Spot testnet and Futures testnet.
    - Spot testnet:    https://testnet.binance.vision
    - Futures testnet: https://testnet.binancefuture.com

    If the matching credential is missing, testnet execution is skipped
    but the in-memory economic simulation still succeeds.
    """

    # Spot testnet
    spot_api_key: str = ""
    spot_api_secret: str = ""
    spot_base_url: str = "https://testnet.binance.vision"

    # Futures testnet
    futures_api_key: str = ""
    futures_api_secret: str = ""
    futures_base_url: str = "https://testnet.binancefuture.com"

    recv_window_ms: int = 5000
    http_timeout_sec: float = 10.0

    # If True, try to send a real order to testnet (best-effort)
    use_testnet_execution: bool = False

    # If True, mark the TradeResult as failed when testnet call fails.
    # Note: missing credentials are treated as "skipped", not a failure.
    fail_on_testnet_error: bool = True


@dataclass
class PancakeDemoParams:
    """
    Parameters for PancakeSwap demo mode.

    build_swap_tx:
        A callback that builds a single swap transaction dict.
        It should use Pancake router (or pool) and return a Web3-compatible
        transaction dict. The callback will be called multiple times for
        each block offset.

        Signature:
            build_swap_tx(web3, symbol, quantity, side_str) -> Dict[str, Any]

        where side_str is "BUY" or "SELL".

    default_fee_rate:
        IMPORTANT: This is an *extra* fee on top of DEX execution (LP fee/price impact/slippage).
        DEX execution effects are already reflected in fill_price derived from balance deltas.
        To avoid double-counting, keep this 0.0 unless you intentionally want an additional fee.
    """

    web3: Any
    account_address: str
    build_swap_tx: Callable[[Any, str, float, str], Dict[str, Any]]
    private_key: Optional[str] = None
    upstream_rpc_url: str = ""
    block_offsets: Tuple[int, int, int] = (1, 2, 3)

    # Extra fee rate (NOT Pancake LP fee). Default 0 to avoid double counting.
    default_fee_rate: float = 0.0

    # Gas token info
    native_symbol: str = "BNB"
    # If set, convert gas(native) -> quote using this price (e.g. BNB in USDT)
    native_price_in_quote: Optional[float] = None


class ExchangeClient(abc.ABC):
    """
    Abstract base class for all exchange clients (paper, demo, live).

    - name: exchange name, e.g. "Binance", "PancakeSwapV2"
    - instrument: e.g. "spot", "perpetual", "swap"
    - state: BotState instance that holds balances and executed trades
    """

    def __init__(self, name: str, instrument: str, state: "BotState") -> None:
        self.name = name
        self.instrument = instrument
        self._state = state

    def _split_symbol(self, symbol: str) -> Tuple[str, str]:
        """
        Split a symbol into (base, quote).

        This is a simple helper for symbols like "ETHUSDT" or "BTCUSDT".
        If the symbol ends with "USDT" we treat that as quote; otherwise
        we fall back to the last 3 characters as the quote asset.
        """
        if symbol.endswith("USDT"):
            return symbol[:-4], "USDT"
        return symbol[:-3], symbol[-3:]

    async def get_balance(self, asset: str) -> float:
        """
        Convenience helper that reads balance from BotState.
        """
        account = self._state.get_or_create_account(self.name, self.instrument)
        return account.balances.get(asset, 0.0)

    @abc.abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        order_type: OrderType = OrderType.MARKET,
        **kwargs: object,
    ) -> TradeResult:
        """
        Create an order on this exchange.

        Implementations may optionally support additional keyword arguments
        such as `fee_rate` or `price_hint`. Unknown kwargs can be safely
        ignored if they are not needed.
        """
        raise NotImplementedError

    async def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        **kwargs: object,
    ) -> TradeResult:
        """
        Convenience helper that always uses a MARKET order type.

        Any extra keyword arguments are forwarded to `create_order`.
        """
        return await self.create_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=OrderType.MARKET,
            **kwargs,
        )


class PaperExchangeClient(ExchangeClient):
    """
    Simple in-memory paper trading client.

    Balances are stored in BotState. This client only simulates fills
    and fees; it does not talk to any real exchange.
    """

    def __init__(
        self,
        name: str,
        instrument: str,
        state: "BotState",
        fee_rate: float = 0.0005,
    ) -> None:
        super().__init__(name=name, instrument=instrument, state=state)
        self._fee_rate = fee_rate

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        order_type: OrderType = OrderType.MARKET,
        **_: object,
    ) -> TradeResult:
        """
        Simulate a simple fill on a CEX-like order book.
        """
        base_asset, quote_asset = self._split_symbol(symbol)
        account = self._state.get_or_create_account(self.name, self.instrument)

        if side == OrderSide.BUY:
            cost_quote = quantity * price
            if account.balances.get(quote_asset, 0.0) < cost_quote:
                raise InsufficientBalanceError(
                    f"Not enough {quote_asset} balance to buy {symbol}: "
                    f"have={account.balances.get(quote_asset, 0.0)} need={cost_quote}"
                )
            account.balances[quote_asset] = account.balances.get(quote_asset, 0.0) - cost_quote
            account.balances[base_asset] = account.balances.get(base_asset, 0.0) + quantity
            base_delta = quantity
            quote_delta = -cost_quote

        elif side == OrderSide.SELL:
            if account.balances.get(base_asset, 0.0) < quantity:
                raise InsufficientBalanceError(
                    f"Not enough {base_asset} balance to sell {symbol}: "
                    f"have={account.balances.get(base_asset, 0.0)} need={quantity}"
                )
            proceeds_quote = quantity * price
            account.balances[base_asset] = account.balances.get(base_asset, 0.0) - quantity
            account.balances[quote_asset] = account.balances.get(quote_asset, 0.0) + proceeds_quote
            base_delta = -quantity
            quote_delta = proceeds_quote

        else:
            raise ValueError(f"Unsupported side for PaperExchangeClient: {side}")

        # Apply taker fee on quote asset
        fee = abs(quote_delta) * self._fee_rate
        account.balances[quote_asset] -= fee

        return TradeResult(
            exchange=self.name,
            instrument=self.instrument,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            fee=self._fee_rate,
            base_delta=base_delta,
            quote_delta=quote_delta - fee,
            success=True,
        )


class BinanceExchangeClient(ExchangeClient):
    """
    Placeholder for a future LIVE Binance client.

    For real trading, you would plug in python-binance or similar here.
    """

    def __init__(self, name: str, instrument: str, state: "BotState") -> None:
        super().__init__(name=name, instrument=instrument, state=state)

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        order_type: OrderType = OrderType.MARKET,
        **_: object,
    ) -> TradeResult:
        raise NotImplementedError("LIVE BinanceExchangeClient is not implemented yet")


class PancakeSwapExchangeClient(ExchangeClient):
    """
    Placeholder for a future LIVE PancakeSwap client.

    For real mainnet trading, you would build transactions here
    and send them through Web3.
    """

    def __init__(self, name: str, instrument: str, state: "BotState") -> None:
        super().__init__(name=name, instrument=instrument, state=state)

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        order_type: OrderType = OrderType.MARKET,
        **_: object,
    ) -> TradeResult:
        raise NotImplementedError("LIVE PancakeSwapExchangeClient is not implemented yet")


# BinanceDemoExchangeClient
# (unchanged from your paste)

class BinanceDemoExchangeClient(ExchangeClient):
    """
    Binance demo client with two layers:

      1) Economic layer (always on):
         - Uses theoretical/mainnet price passed as `price`
         - Applies deterministic slippage and taker fee on the quote asset
         - Updates BotState balances in memory
         - Attaches `binance_demo` metadata with full price/PnL breakdown

      2) Infrastructure layer (optional):
         - If BinanceDemoParams.use_testnet_execution is True, also sends
           a real MARKET order to Binance testnet REST.
         - The HTTP response (or error) is stored under `binance_testnet`
           in TradeResult.metadata.
         - Testnet fills NEVER affect balances or economic PnL.
    """

    def __init__(
        self,
        name: str,
        instrument: str,
        state: "BotState",
        params: Optional[BinanceDemoParams] = None,
        fee_rate: float = 0.0004,
        default_slippage_bps: float = 1.0,
    ) -> None:
        super().__init__(name=name, instrument=instrument, state=state)
        self._params: BinanceDemoParams = params or BinanceDemoParams()
        self._fee_rate = float(fee_rate)
        self._default_slippage_bps = float(default_slippage_bps)
        self._mode = ExecutionMode.DEMO
        self._symbol_filters_cache: Dict[Tuple[str, str, str], Dict[str, str]] = {}

        logger.info(
            "BinanceDemoExchangeClient initialized. "
            "Economic layer uses in-memory fills with mainnet-like prices. "
            "Testnet execution is %s.",
            "ENABLED" if self._params.use_testnet_execution else "DISABLED",
        )

    def _round_step_down_str(self, qty: float, step_size: str) -> str:
        """
        Round quantity down to a multiple of stepSize, returning a decimal string.
        """
        q = Decimal(str(qty))
        s = Decimal(str(step_size))

        if s <= 0:
            return format(q, "f")

        rounded = (q // s) * s
        rounded = rounded.quantize(s, rounding=ROUND_DOWN)
        return format(rounded, "f")

    def _get_testnet_symbol_filters_cached(
        self,
        mode_label: str,
        base_url: str,
        symbol: str,
        order_type: OrderType,
    ) -> Dict[str, str]:
        """
        Return cached symbol filters; fetch from exchangeInfo if missing.
        mode_label: "spot" or "futures"
        """
        for_market = (order_type == OrderType.MARKET)
        kind = "market" if for_market else "lot"

        cache_key = (mode_label, symbol, kind)
        if cache_key in self._symbol_filters_cache:
            return self._symbol_filters_cache[cache_key]

        if mode_label == "spot":
            filt = self._get_spot_symbol_filters_sync(base_url, symbol, for_market=for_market)
        else:
            filt = self._get_futures_symbol_filters_sync(base_url, symbol, for_market=for_market)

        self._symbol_filters_cache[cache_key] = filt
        return filt

    def _get_spot_symbol_filters_sync(self, base_url: str, symbol: str, for_market: bool) -> Dict[str, str]:
        url = base_url.rstrip("/") + "/api/v3/exchangeInfo?" + _parse.urlencode({"symbol": symbol})
        with _request.urlopen(url, timeout=self._params.http_timeout_sec) as resp:
            raw = resp.read().decode("utf-8")

        data = _json.loads(raw)
        info = data["symbols"][0]
        filt_map = {f["filterType"]: f for f in info.get("filters", [])}

        lot = filt_map.get("LOT_SIZE") or {}
        mlot = filt_map.get("MARKET_LOT_SIZE") or {}

        def _dec(x: Any) -> Decimal:
            try:
                return Decimal(str(x))
            except Exception:
                return Decimal("0")

        lot_step = _dec(lot.get("stepSize", "0"))
        mlot_step = _dec(mlot.get("stepSize", "0"))
        lot_min = _dec(lot.get("minQty", "0"))
        mlot_min = _dec(mlot.get("minQty", "0"))

        step_candidates = [s for s in (lot_step, mlot_step) if s > 0]
        min_candidates = [m for m in (lot_min, mlot_min) if m > 0]

        step = max(step_candidates) if step_candidates else Decimal("0")
        mn = max(min_candidates) if min_candidates else Decimal("0")

        min_notional = "0"
        mn_filter = filt_map.get("NOTIONAL") or filt_map.get("MIN_NOTIONAL") or {}
        if isinstance(mn_filter, dict):
            min_notional = str(
                mn_filter.get("minNotional")
                or mn_filter.get("notional")
                or mn_filter.get("minNotionalValue")
                or "0"
            )

        return {
            "stepSize": format(step, "f"),
            "minQty": format(mn, "f"),
            "minNotional": min_notional,
            "qtyPrecision": str(info.get("quantityPrecision") or info.get("baseAssetPrecision") or "0"),
        }

    def _get_futures_symbol_filters_sync(self, base_url: str, symbol: str, for_market: bool) -> Dict[str, str]:
        """
        Fetch futures symbol filters from /fapi/v1/exchangeInfo.

        IMPORTANT:
        Some Binance futures endpoints validate MARKET orders using LOT_SIZE as well.
        To be safe, we pick the *coarser* stepSize between LOT_SIZE and MARKET_LOT_SIZE.
        """
        url = base_url.rstrip("/") + "/fapi/v1/exchangeInfo?" + _parse.urlencode({"symbol": symbol})
        with _request.urlopen(url, timeout=self._params.http_timeout_sec) as resp:
            raw = resp.read().decode("utf-8")

        data = _json.loads(raw)
        info = data["symbols"][0]
        filt_map = {f["filterType"]: f for f in info.get("filters", [])}

        lot = filt_map.get("LOT_SIZE") or {}
        mlot = filt_map.get("MARKET_LOT_SIZE") or {}

        def _dec(x: Any) -> Decimal:
            try:
                return Decimal(str(x))
            except Exception:
                return Decimal("0")

        lot_step = _dec(lot.get("stepSize", "0"))
        mlot_step = _dec(mlot.get("stepSize", "0"))
        lot_min = _dec(lot.get("minQty", "0"))
        mlot_min = _dec(mlot.get("minQty", "0"))

        step = lot_step if lot_step > 0 else mlot_step
        if lot_step > 0 and mlot_step > 0:
            step = max(lot_step, mlot_step)

        mn = lot_min if lot_min > 0 else mlot_min
        if lot_min > 0 and mlot_min > 0:
            mn = max(lot_min, mlot_min)

        qty_prec = info.get("quantityPrecision", None)
        try:
            qty_prec = str(int(qty_prec)) if qty_prec is not None else None
        except Exception:
            qty_prec = None

        return {
            "stepSize": format(step, "f"),
            "minQty": format(mn, "f"),
            "minNotional": "0",
            "qtyPrecision": qty_prec or "",
        }

    def _normalize_qty_str(
        self,
        qty: float,
        step_size: str,
        min_qty: str,
        qty_precision: Optional[int] = None,
    ) -> str:
        """
        Normalize quantity to Binance stepSize/minQty and return a plain decimal string.

        - Floors qty to a multiple of stepSize
        - Enforces minQty
        - Formats with <= qty_precision decimals when available
        """
        q = Decimal(str(qty))
        step = Decimal(str(step_size))
        mn = Decimal(str(min_qty))

        if step > 0:
            q2 = (q // step) * step
        else:
            q2 = q

        if q2 < mn:
            return "0"

        if qty_precision is not None and qty_precision >= 0:
            decimals = int(qty_precision)
        else:
            step_norm = step.normalize() if step > 0 else Decimal("0")
            decimals = max(0, -step_norm.as_tuple().exponent)

        quant = Decimal("1").scaleb(-decimals)
        q2 = q2.quantize(quant, rounding=ROUND_DOWN)

        out = format(q2, "f").rstrip("0").rstrip(".")
        return out or "0"

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        order_type: OrderType = OrderType.MARKET,
        **kwargs: object,
    ) -> TradeResult:
        if price is None or price <= 0:
            raise ValueError("Binance demo client requires a positive theoretical price")

        theoretical_price = float(price)
        qty = float(quantity)

        fee_rate_val = float(kwargs.get("fee_rate", self._fee_rate))

        slippage_bps = float(kwargs.get("slippage_bps", self._default_slippage_bps))
        slip_factor = slippage_bps / 10_000.0

        if side == OrderSide.BUY:
            execution_price = theoretical_price * (1.0 + slip_factor)
        elif side == OrderSide.SELL:
            execution_price = theoretical_price * (1.0 - slip_factor)
        else:
            raise ValueError(f"Unsupported side for Binance demo client: {side}")

        base_asset, quote_asset = self._split_symbol(symbol)
        account = self._state.get_or_create_account(self.name, self.instrument)

        if side == OrderSide.BUY:
            cost_quote = qty * execution_price
            fee_preview = cost_quote * fee_rate_val
            need_quote = cost_quote + fee_preview
            if account.balances.get(quote_asset, 0.0) < need_quote:
                raise InsufficientBalanceError(
                    f"Not enough {quote_asset} balance to buy {symbol}: "
                    f"have={account.balances.get(quote_asset, 0.0)} need={need_quote}"
                )

            account.balances[quote_asset] = account.balances.get(quote_asset, 0.0) - cost_quote
            account.balances[base_asset] = account.balances.get(base_asset, 0.0) + qty

            base_delta = qty
            quote_delta_before_fee = -cost_quote

        elif side == OrderSide.SELL:
            if account.balances.get(base_asset, 0.0) < qty:
                raise InsufficientBalanceError(
                    f"Not enough {base_asset} balance to sell {symbol}: "
                    f"have={account.balances.get(base_asset, 0.0)} need={qty}"
                )

            proceeds_quote = qty * execution_price
            account.balances[base_asset] = account.balances.get(base_asset, 0.0) - qty
            account.balances[quote_asset] = account.balances.get(quote_asset, 0.0) + proceeds_quote

            base_delta = -qty
            quote_delta_before_fee = proceeds_quote
        else:
            raise ValueError(f"Unsupported side for Binance demo client: {side}")

        fee_amount = abs(quote_delta_before_fee) * fee_rate_val
        account.balances[quote_asset] -= fee_amount
        quote_delta_after_fee = quote_delta_before_fee - fee_amount

        trade = TradeResult(
            exchange=self.name,
            instrument=self.instrument,
            symbol=symbol,
            side=side,
            quantity=qty,
            price=execution_price,
            fee=fee_rate_val,
            base_delta=base_delta,
            quote_delta=quote_delta_after_fee,
            success=True,
        )

        demo_meta: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.value,
            "instrument": self.instrument,
            "theoretical_price": theoretical_price,
            "execution_price": execution_price,
            "slippage_bps": slippage_bps,
            "fee_rate": fee_rate_val,
            "fee_amount": fee_amount,
            "base_delta": base_delta,
            "quote_delta_before_fee": quote_delta_before_fee,
            "quote_delta_after_fee": quote_delta_after_fee,
        }

        trade.metadata = trade.metadata or {}
        trade.metadata["binance_demo"] = demo_meta

        if self._params.use_testnet_execution:
            try:
                loop = asyncio.get_running_loop()
                raw_result = await loop.run_in_executor(
                    None,
                    self._send_testnet_order_sync,
                    symbol,
                    side,
                    qty,
                    order_type,
                )
                trade.metadata["binance_testnet"] = raw_result

                if self._params.fail_on_testnet_error:
                    if raw_result.get("status") == "error":
                        trade.success = False
                        resp = raw_result.get("response")
                        trade.message = f"binance_testnet failed: {raw_result.get('error')} response={resp}"

            except Exception as exc:
                trade.metadata["binance_testnet_error"] = str(exc)
                if self._params.fail_on_testnet_error:
                    trade.success = False
                    trade.message = f"binance_testnet failed: {exc}"

        self._state.record_trade(trade)
        return trade

    def _select_testnet_endpoint_and_creds(self) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]:
        if self.instrument == "perpetual":
            return (
                (self._params.futures_base_url or "https://testnet.binancefuture.com").rstrip("/"),
                "/fapi/v1/order",
                self._params.futures_api_key or None,
                self._params.futures_api_secret or None,
                "futures",
            )

        return (
            (self._params.spot_base_url or "https://testnet.binance.vision").rstrip("/"),
            "/api/v3/order",
            self._params.spot_api_key or None,
            self._params.spot_api_secret or None,
            "spot",
        )

    def _send_testnet_order_sync(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType,
    ) -> Dict[str, Any]:
        base_url, path, api_key, api_secret, mode_label = self._select_testnet_endpoint_and_creds()

        if not base_url or not path:
            return {"status": "error", "error": "Missing base_url or path"}

        if not api_key or not api_secret:
            return {
                "status": "skipped",
                "mode": mode_label,
                "endpoint": base_url + path,
                "reason": "Missing credentials for this instrument",
            }

        url = base_url + path

        try:
            filters = self._get_testnet_symbol_filters_cached(mode_label, base_url, symbol, order_type)
            step = Decimal(str(filters.get("stepSize", "0") or "0"))
            mn = Decimal(str(filters.get("minQty", "0") or "0"))

            q_raw = Decimal(str(quantity))

            if step > 0:
                q_floor = (q_raw // step) * step
            else:
                q_floor = q_raw

            if q_floor < mn:
                return {
                    "status": "skipped",
                    "mode": mode_label,
                    "endpoint": url,
                    "reason": f"Quantity below minQty after step flooring (raw={quantity})",
                    "filters": filters,
                }

            step_decimals = max(0, -step.normalize().as_tuple().exponent) if step > 0 else 8

            candidates: List[str] = []
            seen = set()

            for d in range(step_decimals, -1, -1):
                quant = Decimal("1").scaleb(-d)
                q_try = (q_floor // quant) * quant

                if q_try < mn:
                    continue

                s = format(q_try.quantize(quant, rounding=ROUND_DOWN), "f").rstrip("0").rstrip(".") or "0"
                if s == "0":
                    continue
                if s in seen:
                    continue
                seen.add(s)
                candidates.append(s)

            if not candidates:
                return {
                    "status": "skipped",
                    "mode": mode_label,
                    "endpoint": url,
                    "reason": f"No valid qty candidates after normalization (raw={quantity})",
                    "filters": filters,
                }

        except Exception as exc:
            return {
                "status": "error",
                "mode": mode_label,
                "endpoint": url,
                "error": f"Failed to fetch/normalize symbol filters: {exc}",
            }

        def _do_request(qty_str: str) -> Dict[str, Any]:
            ts = int(_time.time() * 1000)
            params = {
                "symbol": symbol,
                "side": side.value,
                "type": "MARKET",
                "quantity": qty_str,
                "timestamp": str(ts),
                "recvWindow": str(self._params.recv_window_ms),
            }

            query_string = _parse.urlencode(params)
            signature = _hmac.new(
                api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                _hashlib.sha256,
            ).hexdigest()

            body = (query_string + "&signature=" + signature).encode("utf-8")

            headers = {
                "X-MBX-APIKEY": api_key,
                "Content-Type": "application/x-www-form-urlencoded",
            }

            req = _request.Request(url, data=body, headers=headers, method="POST")

            try:
                with _request.urlopen(req, timeout=self._params.http_timeout_sec) as resp:
                    status = resp.getcode()
                    raw_body = resp.read().decode("utf-8")
                try:
                    parsed_body: Any = _json.loads(raw_body)
                except Exception:
                    parsed_body = raw_body

                return {
                    "status": status,
                    "mode": mode_label,
                    "endpoint": url,
                    "request_params": params,
                    "response": parsed_body,
                }

            except _error.HTTPError as exc:
                raw_body = ""
                try:
                    raw_body = exc.read().decode("utf-8")
                except Exception:
                    pass

                try:
                    parsed_body: Any = _json.loads(raw_body) if raw_body else raw_body
                except Exception:
                    parsed_body = raw_body

                return {
                    "status": "error",
                    "mode": mode_label,
                    "endpoint": url,
                    "http_status": exc.code,
                    "request_params": params,
                    "response": parsed_body,
                    "error": f"HTTPError {exc.code}",
                }

        last_err: Optional[Dict[str, Any]] = None

        for qty_str in candidates:
            logger.info(
                "[binance-testnet] mode=%s symbol=%s raw_qty=%.18f step=%s min=%s try_qty=%s",
                mode_label,
                symbol,
                float(quantity),
                filters.get("stepSize"),
                filters.get("minQty"),
                qty_str,
            )

            res = _do_request(qty_str)

            if res.get("status") != "error":
                return res

            last_err = res

            code = None
            msg = ""
            resp_body = res.get("response")
            if isinstance(resp_body, dict):
                code = resp_body.get("code")
                msg = str(resp_body.get("msg", "") or "")

            if code == -1111:
                continue
            if code == -1013 and "LOT_SIZE" in msg:
                continue

            return res

        return {
            "status": "skipped",
            "mode": mode_label,
            "endpoint": url,
            "reason": "All qty candidates rejected (including LOT_SIZE/-1111)",
            "filters": filters,
            "last_error": last_err,
            "candidates": candidates,
        }


class PancakeSwapDemoExchangeClient(ExchangeClient):
    """
    Exchange client that actually pushes swaps to a forked BSC (Hardhat or Anvil).

    - If PancakeDemoParams.private_key is provided:
        -> sign locally and use eth_sendRawTransaction (no impersonate).
    - If private_key is None:
        -> impersonate account_address on fork node and use eth_sendTransaction.

    Profitability in metadata:
    - DEX execution effects (LP fee, slippage, price impact) are already reflected in fill_price.
    - fee_rate here is treated as an extra fee (platform/broker fee), default 0 to avoid double counting.
    - gas is computed from receipts (approve + swap).
    """

    def __init__(
        self,
        exchange_name: str,
        instrument: str,
        state: "BotState",
        params: PancakeDemoParams,
    ) -> None:
        super().__init__(exchange_name, instrument, state)
        self._mode = ExecutionMode.DEMO
        if params.web3 is None:
            raise ValueError("PancakeDemoParams.web3 is required in demo mode")
        self._web3 = params.web3
        _apply_poa_middleware(self._web3)
        self._params = params

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        quote_asset: Optional[str] = None,
        fee_rate: Optional[float] = None,
        price_hint: Optional[float] = None,
        **_: object,
    ) -> TradeResult:
        if price is None and price_hint is not None:
            price = price_hint
        if price is None:
            raise ValueError("Pancake demo client requires an explicit price or price_hint")

        theoretical_price = float(price)

        # Extra fee (NOT DEX LP fee). Default 0 to avoid double counting.
        fee_rate_val = float(fee_rate) if fee_rate is not None else float(self._params.default_fee_rate)

        account = self._state.get_or_create_account(self.name, self.instrument)

        per_block_meta: Optional[Dict[str, Any]] = None
        if self._params.block_offsets and len(self._params.block_offsets) > 0:
            per_block_meta = await self._run_multi_block_flow(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=theoretical_price,
                fee_rate=fee_rate_val,
            )

        if per_block_meta is not None:
            ok_count = int(per_block_meta.get("ok_count", 0) or 0)
            if ok_count < 1:
                return TradeResult(
                    exchange=self.name,
                    instrument=self.instrument,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=float(price),
                    fee=fee_rate_val,
                    base_delta=0.0,
                    quote_delta=0.0,
                    success=False,
                    message="pancake_demo swap failed (no successful fork swaps)",
                    metadata={"pancake_demo": per_block_meta},
                )

        # Prefer avg fill price; fallback to theoretical
        execution_price = theoretical_price
        if per_block_meta is not None:
            avg_fill = per_block_meta.get("avg_fill_price")
            if isinstance(avg_fill, (int, float)) and float(avg_fill) > 0:
                execution_price = float(avg_fill)

        if per_block_meta is not None:
            fee_rate_val = 0.0
        clean_symbol = symbol[5:] if (symbol or "").upper().startswith("DEMO_") else symbol
        base_asset, quote_asset2 = resolve_base_quote(clean_symbol)

        # If we have actual avg deltas from fork swaps, use them to update balances
        avg_base_qty = None
        avg_quote_qty = None
        avg_gas_cost_native = None

        if per_block_meta is not None:
            if isinstance(per_block_meta.get("avg_base_qty"), (int, float)):
                avg_base_qty = float(per_block_meta["avg_base_qty"])
            if isinstance(per_block_meta.get("avg_quote_qty"), (int, float)):
                avg_quote_qty = float(per_block_meta["avg_quote_qty"])
            if isinstance(per_block_meta.get("avg_gas_cost_native"), (int, float)):
                avg_gas_cost_native = float(per_block_meta["avg_gas_cost_native"])

        # Fallback to requested quantity if fork didn't provide base/quote quantities
        base_qty = float(avg_base_qty) if (avg_base_qty is not None and avg_base_qty > 0) else float(quantity)
        quote_qty = float(avg_quote_qty) if (avg_quote_qty is not None and avg_quote_qty > 0) else float(quantity) * float(execution_price)

        # Update balances based on deltas
        if side == OrderSide.BUY:
            # BUY: spend quote, receive base
            # Balance check includes extra fee preview (optional)
            fee_preview = quote_qty * fee_rate_val
            need_quote = quote_qty + fee_preview
            if account.balances.get(quote_asset2, 0.0) < need_quote:
                raise InsufficientBalanceError(
                    f"Not enough {quote_asset2} balance to buy {symbol}: "
                    f"have={account.balances.get(quote_asset2, 0.0)} need={need_quote}"
                )

            account.balances[quote_asset2] = account.balances.get(quote_asset2, 0.0) - quote_qty
            account.balances[base_asset] = account.balances.get(base_asset, 0.0) + base_qty

            base_delta = base_qty
            quote_delta_before_fee = -quote_qty

        elif side == OrderSide.SELL:
            # SELL: spend base, receive quote
            if account.balances.get(base_asset, 0.0) < base_qty:
                raise InsufficientBalanceError(
                    f"Not enough {base_asset} balance to sell {symbol}: "
                    f"have={account.balances.get(base_asset, 0.0)} need={base_qty}"
                )

            account.balances[base_asset] = account.balances.get(base_asset, 0.0) - base_qty
            account.balances[quote_asset2] = account.balances.get(quote_asset2, 0.0) + quote_qty

            base_delta = -base_qty
            quote_delta_before_fee = quote_qty

        else:
            raise ValueError(f"Unsupported side for Pancake demo client: {side}")

        # Apply extra fee on quote (once). This is NOT Pancake LP fee.
        fee_amount_quote = abs(quote_delta_before_fee) * fee_rate_val
        if fee_amount_quote > 0:
            account.balances[quote_asset2] -= fee_amount_quote
        quote_delta_after_fee = quote_delta_before_fee - fee_amount_quote

        # Deduct gas in native token (if we computed it)
        if avg_gas_cost_native is not None and avg_gas_cost_native > 0:
            native_sym = (self._params.native_symbol or "BNB").strip() or "BNB"
            account.balances[native_sym] = account.balances.get(native_sym, 0.0) - float(avg_gas_cost_native)

        trade = TradeResult(
            exchange=self.name,
            instrument=self.instrument,
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            price=float(execution_price),
            fee=fee_rate_val,
            base_delta=float(base_delta),
            quote_delta=float(quote_delta_after_fee),
            success=True,
        )

        if per_block_meta is not None:
            enriched_meta = dict(per_block_meta)
            enriched_meta.setdefault("theoretical_price", theoretical_price)
            enriched_meta.setdefault("price_hint", price_hint if price_hint is not None else theoretical_price)
            enriched_meta.setdefault("execution_price", execution_price)
            enriched_meta.setdefault("extra_fee_rate", fee_rate_val)
            enriched_meta.setdefault("extra_fee_amount_quote", fee_amount_quote)
            enriched_meta.setdefault("base_asset", base_asset)
            enriched_meta.setdefault("quote_asset", quote_asset2)

            trade.metadata = trade.metadata or {}
            trade.metadata["pancake_demo"] = enriched_meta

        self._state.record_trade(trade)
        return trade

    async def _run_multi_block_flow(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        fee_rate: float,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        meta = await loop.run_in_executor(
            None,
            self._run_multi_block_swaps_sync,
            symbol,
            side,
            quantity,
            price,
            fee_rate,
        )
        return meta

    def _run_multi_block_swaps_sync(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        fee_rate: float,
    ) -> Dict[str, Any]:
        web3 = self._web3
        offsets = tuple(self._params.block_offsets or (1, 2, 3))
        upstream_rpc_url = (getattr(self._params, "upstream_rpc_url", "") or "").strip()
        if not upstream_rpc_url:
            raise RuntimeError("Missing params.upstream_rpc_url for fork reset flow")

        upstream_web3 = Web3(Web3.HTTPProvider(self._params.upstream_rpc_url))
        _apply_poa_middleware(upstream_web3)

        base_block = int(upstream_web3.eth.block_number)

        provider = web3.provider
        if provider is None:
            raise RuntimeError("Web3 provider must not be None for Pancake demo client")

        priv_key = self._params.private_key
        use_local_signing = bool(priv_key)

        if use_local_signing:
            acct = Account.from_key(priv_key)  # type: ignore[arg-type]
            trader = Web3.to_checksum_address(acct.address)
        else:
            trader = Web3.to_checksum_address(self._params.account_address)

        fork_engine_pref = getattr(self._params, "fork_engine", "auto") or "auto"
        if fork_engine_pref == "auto":
            detected = self._detect_fork_engine(provider)
            if detected in ("anvil", "hardhat"):
                fork_engine_pref = detected

        def _get_nonce_safe(addr: str) -> int:
            try:
                return int(web3.eth.get_transaction_count(addr, "pending"))
            except Exception:
                return int(web3.eth.get_transaction_count(addr))

        def _avg_num(xs: List[Any]) -> Optional[float]:
            ys = [float(x) for x in xs if isinstance(x, (int, float))]
            return (sum(ys) / len(ys)) if ys else None

        def _sum_num(xs: List[Any]) -> Optional[float]:
            ys = [float(x) for x in xs if isinstance(x, (int, float))]
            return sum(ys) if ys else None

        def _wei_to_native(wei: Any) -> float:
            try:
                return float(int(wei)) / 1e18
            except Exception:
                return 0.0

        def _normalize_approve_meta(m: Any) -> Optional[Dict[str, Any]]:
            if not isinstance(m, dict):
                return None
            out = dict(m)

            # Normalize expected keys
            out.setdefault("needed", 0)
            out.setdefault("sent", False)
            out.setdefault("approve_gas_used", 0)
            out.setdefault("approve_gas_price_wei", 0)
            out.setdefault("approve_gas_cost_wei", 0)
            out.setdefault("tx_hashes", [])
            out.setdefault("erc20", {})
            out.setdefault("permit2", {})
            out.setdefault("error", "")

            # If builder provided nested metas only, derive totals.
            if int(out.get("approve_gas_cost_wei") or 0) == 0:
                erc20 = out.get("erc20") if isinstance(out.get("erc20"), dict) else {}
                p2 = out.get("permit2") if isinstance(out.get("permit2"), dict) else {}
                out["approve_gas_cost_wei"] = int(erc20.get("approve_gas_cost_wei", 0) or 0) + int(
                    p2.get("permit2_gas_cost_wei", 0) or 0
                )
                out["approve_gas_used"] = int(erc20.get("approve_gas_used", 0) or 0) + int(
                    p2.get("permit2_gas_used", 0) or 0
                )

            return out

        per_block_results: List[Dict[str, Any]] = []

        for offset in offsets:
            fork_block = base_block + offset
            self._wait_for_upstream_block(upstream_web3, fork_block)

            used_reset_method = ""
            used_impersonate_method = ""

            reset_config = {
                "forking": {
                    "jsonRpcUrl": self._params.upstream_rpc_url,
                    "blockNumber": fork_block,
                }
            }

            try:
                used_reset_method = self._fork_reset_any(provider, reset_config, prefer=fork_engine_pref)
            except Exception as exc:
                per_block_results.append(
                    {
                        "fork_block": fork_block,
                        "tx_hash": "",
                        "status": 0,
                        "gas_used": 0,
                        "reset_method": "",
                        "impersonate_method": "",
                        "revert_reason": f"fork_reset_failed: {exc}",
                    }
                )
                continue

            if not use_local_signing:
                try:
                    used_impersonate_method = self._impersonate_any(provider, trader, prefer=fork_engine_pref)
                except Exception as exc:
                    per_block_results.append(
                        {
                            "fork_block": fork_block,
                            "tx_hash": "",
                            "status": 0,
                            "gas_used": 0,
                            "reset_method": used_reset_method,
                            "impersonate_method": "",
                            "revert_reason": f"impersonate_failed: {exc}",
                        }
                    )
                    continue

            # Build swap tx (builder may attach _demo_* metadata keys)
            tx_dict = None
            try:
                tx_dict = self._params.build_swap_tx(
                    web3=web3,
                    symbol=symbol,
                    quantity=float(quantity),
                    side=side.value,
                    price=float(price),
                    trader=trader,
                    private_key=priv_key if use_local_signing else None,
                )
            except TypeError:
                tx_dict = self._params.build_swap_tx(
                    web3=web3,
                    symbol=symbol,
                    quantity=float(quantity),
                    side=side.value,
                    trader=trader,
                    private_key=priv_key if use_local_signing else None,
                )
            except Exception as exc:
                per_block_results.append(
                    {
                        "fork_block": fork_block,
                        "tx_hash": "",
                        "status": 0,
                        "gas_used": 0,
                        "reset_method": used_reset_method,
                        "impersonate_method": used_impersonate_method,
                        "revert_reason": f"build_swap_tx_failed: {exc}",
                    }
                )
                continue

            if tx_dict is None:
                per_block_results.append(
                    {
                        "fork_block": fork_block,
                        "tx_hash": "",
                        "status": 0,
                        "gas_used": 0,
                        "reset_method": used_reset_method,
                        "impersonate_method": used_impersonate_method,
                        "revert_reason": "build_swap_tx_returned_none",
                    }
                )
                continue

            if not isinstance(tx_dict, dict):
                try:
                    tx_dict = dict(tx_dict)
                except Exception as exc:
                    per_block_results.append(
                        {
                            "fork_block": fork_block,
                            "tx_hash": "",
                            "status": 0,
                            "gas_used": 0,
                            "reset_method": used_reset_method,
                            "impersonate_method": used_impersonate_method,
                            "revert_reason": f"build_swap_tx_non_dict: type={type(tx_dict)} err={exc}",
                        }
                    )
                    continue


            # Demo metadata keys from builder
            token_in = tx_dict.get("_demo_token_in")
            token_out = tx_dict.get("_demo_token_out")
            amount_in_wei = tx_dict.get("_demo_amount_in_wei")

            # Gas token symbol (native only; no quote conversion)
            gas_token_symbol = tx_dict.get("_demo_gas_token_symbol") or self._params.native_symbol or "BNB"

            permit2_addr = tx_dict.get("_demo_permit2") or tx_dict.get("_demo_spender")
            router_addr = tx_dict.get("to")

            # Strip builder metadata before sending/signing
            ALLOWED_TX_KEYS = {
                "from", "to", "value", "data", "nonce",
                "gas", "gasPrice", "maxFeePerGas", "maxPriorityFeePerGas",
                "chainId", "type", "accessList",
            }
            tx_send = {k: v for k, v in tx_dict.items() if k in ALLOWED_TX_KEYS}
            tx_send["from"] = trader
            tx_send.setdefault("chainId", int(web3.eth.chain_id))

            if "gas" not in tx_send:
                try:
                    est = int(web3.eth.estimate_gas(tx_send))
                    tx_send["gas"] = int(est * 12 // 10)
                except Exception:
                    tx_send["gas"] = 600_000

            if "gasPrice" not in tx_send and "maxFeePerGas" not in tx_send:
                try:
                    tx_send["gasPrice"] = int(web3.eth.gas_price)
                except Exception:
                    pass

            token_in_contract = None
            token_out_contract = None

            if ERC20_ABI and token_in and token_out:
                try:
                    token_in_contract = web3.eth.contract(
                        address=Web3.to_checksum_address(token_in),
                        abi=ERC20_ABI,
                    )
                    token_out_contract = web3.eth.contract(
                        address=Web3.to_checksum_address(token_out),
                        abi=ERC20_ABI,
                    )
                except Exception:
                    token_in_contract = None
                    token_out_contract = None

            bal_in_before = None
            bal_out_before = None
            decimals_in = None
            decimals_out = None

            if token_in_contract is not None and token_out_contract is not None:
                try:
                    bal_in_before = token_in_contract.functions.balanceOf(trader).call()
                    bal_out_before = token_out_contract.functions.balanceOf(trader).call()
                    decimals_in = int(token_in_contract.functions.decimals().call())
                    decimals_out = int(token_out_contract.functions.decimals().call())
                except Exception:
                    bal_in_before = None
                    bal_out_before = None
                    decimals_in = None
                    decimals_out = None

            # Fail fast: insufficient token_in
            if token_in_contract is not None and bal_in_before is not None and amount_in_wei is not None:
                try:
                    need = int(amount_in_wei)
                    have = int(bal_in_before)
                    if have < need:
                        per_block_results.append(
                            {
                                "fork_block": fork_block,
                                "tx_hash": "",
                                "status": 0,
                                "gas_used": 0,
                                "token_in": token_in,
                                "token_out": token_out,
                                "revert_reason": f"insufficient token_in balance before swap: have={have} need={need}",
                                "reset_method": used_reset_method,
                                "impersonate_method": used_impersonate_method,
                            }
                        )
                        continue
                except Exception:
                    pass

            # Ensure allowance (approve tx) and collect approve gas
            permit2_addr = tx_dict.get("_demo_permit2", None) or tx_dict.get("_demo_spender", None)
            router_addr = tx_dict.get("to")

            allowance_meta = _normalize_approve_meta(tx_dict.get("_demo_approve"))

            if allowance_meta is None:
                allowance_meta = {
                    "needed": int(amount_in_wei or 0),
                    "sent": False,
                    "approve_gas_used": 0,
                    "approve_gas_price_wei": 0,
                    "approve_gas_cost_wei": 0,
                    "tx_hashes": [],
                    "erc20": {},
                    "permit2": {},
                    "error": "",
                }

                try:
                    if token_in and amount_in_wei is not None and permit2_addr and router_addr and token_in_contract is not None:
                        # (1) ERC20 approve token_in -> Permit2
                        erc20_meta = self._ensure_allowance(
                            web3=web3,
                            owner=trader,
                            token=token_in,
                            spender=str(permit2_addr),
                            amount_wei=int(amount_in_wei),
                            priv_key=priv_key,
                            use_local_signing=use_local_signing,
                        )

                        # (2) Permit2 approve (owner, token_in, router) allowance
                        permit2_meta = self._ensure_permit2_allowance(
                            web3=web3,
                            owner=trader,
                            permit2=str(permit2_addr),
                            token=str(token_in),
                            spender=str(router_addr),
                            amount_wei=int(amount_in_wei),
                            priv_key=priv_key,
                            use_local_signing=use_local_signing,
                        )

                        allowance_meta["erc20"] = erc20_meta
                        allowance_meta["permit2"] = permit2_meta

                        erc20_cost = int((erc20_meta or {}).get("approve_gas_cost_wei", 0) or 0)
                        erc20_used = int((erc20_meta or {}).get("approve_gas_used", 0) or 0)

                        p2_cost = int((permit2_meta or {}).get("permit2_gas_cost_wei", 0) or 0)
                        p2_used = int((permit2_meta or {}).get("permit2_gas_used", 0) or 0)

                        allowance_meta["approve_gas_cost_wei"] = erc20_cost + p2_cost
                        allowance_meta["approve_gas_used"] = erc20_used + p2_used
                        allowance_meta["sent"] = bool((erc20_meta or {}).get("sent")) or bool((permit2_meta or {}).get("sent"))

                        allowance_meta["tx_hashes"] = list((erc20_meta or {}).get("tx_hashes", []) or []) + list(
                            (permit2_meta or {}).get("tx_hashes", []) or []
                        )
                except Exception as exc:
                    allowance_meta["error"] = f"ensure_allowance_failed: {exc}"
                    logger.warning("[pancake-demo] ensure_allowance failed: %s", exc)


            tx_hash_hex = ""
            status = 0
            gas_used = 0
            revert_reason = None

            swap_receipt = None
            swap_gas_used = 0
            swap_gas_price_wei = 0
            swap_gas_cost_wei = 0

            try:
                if use_local_signing:
                    tx_send["nonce"] = _get_nonce_safe(trader)
                    signed = Account.sign_transaction(tx_send, priv_key)  # type: ignore[arg-type]
                    raw_tx = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None)
                    tx_hash = web3.eth.send_raw_transaction(raw_tx)
                else:
                    tx_hash = web3.eth.send_transaction(tx_send)

                tx_hash_hex = tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash)
                swap_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

                if isinstance(swap_receipt, dict):
                    status = int(swap_receipt.get("status", 0))
                    gas_used = int(swap_receipt.get("gasUsed", 0))
                else:
                    status = int(getattr(swap_receipt, "status", 0))
                    gas_used = int(getattr(swap_receipt, "gasUsed", 0))

                swap_gas_used, swap_gas_price_wei, swap_gas_cost_wei = self._gas_cost_from_receipt(
                    web3,
                    swap_receipt,
                    fallback_gas_price_wei=int(tx_send.get("gasPrice", 0) or 0),
                )

                if status == 0:
                    try:
                        web3.eth.call(
                            {
                                "to": tx_send.get("to"),
                                "from": tx_send.get("from"),
                                "data": tx_send.get("data"),
                                "value": tx_send.get("value", 0),
                            },
                            block_identifier="latest",
                        )
                    except Exception as exc:
                        revert_reason = str(exc)

            except Exception as exc:
                revert_reason = f"send_or_wait_error: {exc}"

            delta_in = None
            delta_out = None

            base_qty = None
            quote_qty = None
            fill_price = None
            return_vs_hint = None

            # Derive base/quote qty from balance deltas (only on success)
            if (
                status == 1
                and token_in_contract is not None
                and token_out_contract is not None
                and bal_in_before is not None
                and bal_out_before is not None
                and decimals_in is not None
                and decimals_out is not None
            ):
                try:
                    bal_in_after = token_in_contract.functions.balanceOf(trader).call()
                    bal_out_after = token_out_contract.functions.balanceOf(trader).call()

                    delta_in = int(bal_in_after) - int(bal_in_before)
                    delta_out = int(bal_out_after) - int(bal_out_before)

                    scale_in = 10 ** int(decimals_in)
                    scale_out = 10 ** int(decimals_out)

                    # Mapping:
                    # - BUY  : token_in = quote(spent), token_out = base(received)
                    # - SELL : token_in = base(spent),  token_out = quote(received)
                    if side == OrderSide.BUY:
                        quote_qty = (-delta_in) / scale_in
                        base_qty = (delta_out) / scale_out
                    else:
                        base_qty = (-delta_in) / scale_in
                        quote_qty = (delta_out) / scale_out

                    if base_qty and base_qty > 0 and quote_qty is not None:
                        fill_price = float(quote_qty) / float(base_qty)
                        if price > 0:
                            return_vs_hint = float(fill_price) / float(price) - 1.0
                except Exception:
                    delta_in = None
                    delta_out = None
                    base_qty = None
                    quote_qty = None
                    fill_price = None
                    return_vs_hint = None


            expected_notional_quote: Optional[float] = None
            pnl_quote_vs_hint: Optional[float] = None
            pnl_rate_vs_hint: Optional[float] = None

            extra_fee_quote: Optional[float] = None

            net_pnl_quote_excl_gas: Optional[float] = None
            net_return_excl_gas: Optional[float] = None

            net_pnl_quote_incl_swap_gas: Optional[float] = None
            net_return_incl_swap_gas: Optional[float] = None

            net_pnl_quote_incl_total_gas: Optional[float] = None
            net_return_incl_total_gas: Optional[float] = None

            # Gas cost (native)
            approve_cost_wei = int(allowance_meta.get("approve_gas_cost_wei", 0) or 0)
            swap_cost_wei = int(swap_gas_cost_wei or 0)

            approve_gas_cost_native = float(approve_cost_wei) / 1e18 if approve_cost_wei > 0 else 0.0
            swap_gas_cost_native = float(swap_cost_wei) / 1e18 if swap_cost_wei > 0 else 0.0

            gas_total_cost_wei = approve_cost_wei + swap_cost_wei
            total_gas_cost_native = float(gas_total_cost_wei) / 1e18 if gas_total_cost_wei > 0 else 0.0

            approve_cost_native = approve_gas_cost_native
            swap_cost_native = swap_gas_cost_native
            gas_cost_native = total_gas_cost_native
            # Native gas symbol only (no implicit quote conversion)
            gas_sym = (str(gas_token_symbol or self._params.native_symbol or "BNB")).strip() or "BNB"

            # Optional: convert gas(native) -> quote when a price is provided.
            # If you want "native-only" strictly, keep native_price_in_quote=None.
            approve_cost_quote: Optional[float] = None
            swap_cost_quote: Optional[float] = None
            gas_cost_quote: Optional[float] = None

            native_px = getattr(self._params, "native_price_in_quote", None)
            if isinstance(native_px, (int, float)) and float(native_px) > 0:
                approve_cost_quote = float(approve_cost_native) * float(native_px)
                swap_cost_quote = float(swap_cost_native) * float(native_px)
                gas_cost_quote = float(gas_cost_native) * float(native_px)

            # Compute PnL/returns if we have fill quantities
            if base_qty is not None and quote_qty is not None and price > 0 and base_qty > 0:
                expected_notional_quote = float(base_qty) * float(price)

                # PnL vs hint in quote units
                if side == OrderSide.BUY:
                    pnl_quote_vs_hint = expected_notional_quote - float(quote_qty)
                else:
                    pnl_quote_vs_hint = float(quote_qty) - expected_notional_quote

                if expected_notional_quote > 0:
                    pnl_rate_vs_hint = float(pnl_quote_vs_hint) / float(expected_notional_quote)

                # Extra fee (NOT DEX LP fee): applied on quote amount
                extra_fee_quote = abs(float(quote_qty)) * float(fee_rate)

                net_pnl_quote_excl_gas = float(pnl_quote_vs_hint) - float(extra_fee_quote)
                if expected_notional_quote > 0:
                    net_return_excl_gas = float(net_pnl_quote_excl_gas) / float(expected_notional_quote)

                # Include gas only if we can convert native -> quote
                if swap_cost_quote is not None:
                    net_pnl_quote_incl_swap_gas = float(net_pnl_quote_excl_gas) - float(swap_cost_quote)
                    if expected_notional_quote > 0:
                        net_return_incl_swap_gas = float(net_pnl_quote_incl_swap_gas) / float(expected_notional_quote)

                if gas_cost_quote is not None:
                    net_pnl_quote_incl_total_gas = float(net_pnl_quote_excl_gas) - float(gas_cost_quote)
                    if expected_notional_quote > 0:
                        net_return_incl_total_gas = float(net_pnl_quote_incl_total_gas) / float(expected_notional_quote)

            # Backward-compatible: prefer "incl swap gas" if available, else "excl gas"
            net_return_vs_hint = (
                net_return_incl_swap_gas
                if net_return_incl_swap_gas is not None
                else net_return_excl_gas
            )

            # Aliases used later when building `result`
            swap_pnl_quote = pnl_quote_vs_hint
            swap_pnl_rate = pnl_rate_vs_hint

            net_pnl_excl_gas = net_pnl_quote_excl_gas
            net_ret_excl_gas = net_return_excl_gas

            net_pnl_incl_swap_gas = net_pnl_quote_incl_swap_gas
            net_ret_incl_swap_gas = net_return_incl_swap_gas

            net_pnl_incl_total_gas = net_pnl_quote_incl_total_gas
            net_ret_incl_total_gas = net_return_incl_total_gas

            # swap_prepare log (pre-send info)
            to_addr = tx_send.get("to")
            data = tx_send.get("data", b"")
            data_len = len(data) if isinstance(data, (bytes, bytearray)) else len(str(data))

            logger.info(
                "[pancake-demo] swap_prepare symbol=%s side=%s qty=%.8f hint=%.6f fork_block=%s reset=%s imp=%s "
                "from=%s to=%s value=%s gas=%s gasPrice=%s nonce=%s data_len=%s",
                symbol,
                side.value,
                float(quantity),
                float(price),
                fork_block,
                used_reset_method,
                used_impersonate_method,
                trader,
                to_addr,
                tx_send.get("value", 0),
                tx_send.get("gas"),
                tx_send.get("gasPrice", None),
                tx_send.get("nonce", None),
                data_len,
            )

            # Per-block metrics log
            logger.info(
                "[pancake-demo] swap_metrics fork_block=%s status=%s tx=%s side=%s qty=%.8f "
                "hint=%.6f fill=%s slip_bps=%s "
                "swap_pnl_quote=%s swap_pnl_rate=%s "
                "fee_rate=%.8f fee_quote=%s "
                "approve_gas_native=%s swap_gas_native=%s total_gas_native=%s "
                "approve_gas_quote=%s swap_gas_quote=%s total_gas_quote=%s "
                "net_excl_gas=%s net_incl_swap_gas=%s net_incl_total_gas=%s revert=%s",
                fork_block,
                status,
                (tx_hash_hex[:10] if tx_hash_hex else ""),
                side.value,
                float(quantity),
                float(price),
                (f"{fill_price:.10f}" if isinstance(fill_price, (int, float)) else None),
                (f"{(return_vs_hint * 10000.0):.3f}" if return_vs_hint is not None else None),
                (f"{swap_pnl_quote:.10f}" if isinstance(swap_pnl_quote, (int, float)) else None),
                (f"{(swap_pnl_rate * 100.0):.6f}%" if swap_pnl_rate is not None else None),
                float(fee_rate),
                (f"{extra_fee_quote:.10f}" if isinstance(extra_fee_quote, (int, float)) else None),
                (f"{approve_cost_native:.10f}" if isinstance(approve_cost_native, (int, float)) else None),
                (f"{swap_cost_native:.10f}" if isinstance(swap_cost_native, (int, float)) else None),
                (f"{gas_cost_native:.10f}" if isinstance(gas_cost_native, (int, float)) else None),
                (f"{approve_cost_quote:.10f}" if isinstance(approve_cost_quote, (int, float)) else None),
                (f"{swap_cost_quote:.10f}" if isinstance(swap_cost_quote, (int, float)) else None),
                (f"{gas_cost_quote:.10f}" if isinstance(gas_cost_quote, (int, float)) else None),
                (f"{(net_ret_excl_gas * 100.0):.6f}%" if net_ret_excl_gas is not None else None),
                (f"{(net_ret_incl_swap_gas * 100.0):.6f}%" if net_ret_incl_swap_gas is not None else None),
                (f"{(net_ret_incl_total_gas * 100.0):.6f}%" if net_ret_incl_total_gas is not None else None),
                revert_reason,
            )


            approve_gas_cost_wei = int(approve_cost_wei)
            result: Dict[str, Any] = {
                "fork_block": fork_block,
                "tx_hash": tx_hash_hex,
                "status": status,
                "gas_used": gas_used,
                "reset_method": used_reset_method,
                "impersonate_method": used_impersonate_method,
                "approve": allowance_meta,
                "swap_gas_used": int(swap_gas_used),
                "swap_gas_price_wei": int(swap_gas_price_wei),
                "swap_gas_cost_wei": int(swap_gas_cost_wei),

                "approve_gas_cost_wei": int(approve_gas_cost_wei),
                "swap_gas_cost_native": float(swap_gas_cost_native),
                "total_gas_cost_native": float(total_gas_cost_native),
                "total_gas_cost_wei": int(gas_total_cost_wei),

                # Gas costs in native token (always available if receipts exist)
                "approve_gas_cost_native": float(approve_cost_native),
                "swap_gas_cost_native": float(swap_cost_native),
                "total_gas_cost_native": float(gas_cost_native),

        
                "gas_token_symbol": gas_sym,

                "extra_fee_rate": float(fee_rate),
            }

            if revert_reason is not None:
                result["revert_reason"] = revert_reason
            if token_in:
                result["token_in"] = token_in
            if token_out:
                result["token_out"] = token_out
            if delta_in is not None:
                result["delta_in"] = delta_in
            if delta_out is not None:
                result["delta_out"] = delta_out

            if base_qty is not None:
                result["base_qty"] = float(base_qty)
            if quote_qty is not None:
                result["quote_qty"] = float(quote_qty)
            if fill_price is not None:
                result["fill_price"] = float(fill_price)
            if return_vs_hint is not None:
                result["return_vs_hint"] = float(return_vs_hint)

            if expected_notional_quote is not None:
                result["expected_notional_quote"] = float(expected_notional_quote)

            result["swap_pnl_quote_vs_hint"] = swap_pnl_quote
            result["swap_pnl_rate_vs_hint"] = swap_pnl_rate
            result["fee_quote"] = extra_fee_quote

            result["net_pnl_quote_excl_gas"] = net_pnl_excl_gas
            result["net_return_excl_gas"] = net_ret_excl_gas

            result["net_pnl_quote_incl_swap_gas"] = net_pnl_incl_swap_gas
            result["net_return_incl_swap_gas"] = net_ret_incl_swap_gas

            result["net_pnl_quote_incl_total_gas"] = net_pnl_incl_total_gas
            result["net_return_incl_total_gas"] = net_ret_incl_total_gas

            # Backward compatible "final" net return
            if net_return_vs_hint is not None:
                result["net_return_vs_hint"] = float(net_return_vs_hint)

            per_block_results.append(result)

            account = self._state.get_or_create_account(self.name, self.instrument)

            demo_meta: Dict[str, Any] = {
                "base_block": base_block,
                "block_offsets": list(offsets),
                "per_block_results": per_block_results,
                "account_snapshot": {"balances": dict(account.balances)},
            }

            ok_rows = [r for r in per_block_results if int(r.get("status", 0) or 0) == 1]
            ok_count = len(ok_rows)
            fail_count = len(per_block_results) - ok_count

            demo_meta["all_ok"] = (ok_count == len(per_block_results))
            demo_meta["ok_count"] = ok_count
            demo_meta["fail_count"] = fail_count

            # Avg fill price across success
            ok_fill_prices = [
                float(r.get("fill_price"))
                for r in ok_rows
                if isinstance(r.get("fill_price"), (int, float))
            ]
            demo_meta["avg_fill_price"] = (sum(ok_fill_prices) / len(ok_fill_prices)) if ok_fill_prices else None

            # Slippage vs hint (NOT net)
            ok_returns = [
                float(r.get("return_vs_hint"))
                for r in ok_rows
                if isinstance(r.get("return_vs_hint"), (int, float))
            ]
            demo_meta["avg_return_vs_hint"] = (sum(ok_returns) / len(ok_returns)) if ok_returns else None

            # Avg base/quote qty
            ok_base_qtys = [
                float(r.get("base_qty"))
                for r in ok_rows
                if isinstance(r.get("base_qty"), (int, float))
            ]
            ok_quote_qtys = [
                float(r.get("quote_qty"))
                for r in ok_rows
                if isinstance(r.get("quote_qty"), (int, float))
            ]
            demo_meta["avg_base_qty"] = (sum(ok_base_qtys) / len(ok_base_qtys)) if ok_base_qtys else None
            demo_meta["avg_quote_qty"] = (sum(ok_quote_qtys) / len(ok_quote_qtys)) if ok_quote_qtys else None

            # Optional: net return (keep if you use it elsewhere)
            ok_net_returns = [
                float(r.get("net_return_vs_hint"))
                for r in ok_rows
                if isinstance(r.get("net_return_vs_hint"), (int, float))
            ]
            demo_meta["avg_net_return_vs_hint"] = (sum(ok_net_returns) / len(ok_net_returns)) if ok_net_returns else None

            demo_meta["avg_swap_pnl_rate_vs_hint"] = _avg_num([r.get("swap_pnl_rate_vs_hint") for r in ok_rows])
            demo_meta["avg_fee_quote"] = _avg_num([r.get("fee_quote") for r in ok_rows])

            # Native gas metrics (approve/swap/total)
            demo_meta["avg_approve_gas_cost_native"] = _avg_num([r.get("approve_gas_cost_native") for r in ok_rows])
            demo_meta["avg_swap_gas_cost_native"] = _avg_num([r.get("swap_gas_cost_native") for r in ok_rows])
            demo_meta["avg_total_gas_cost_native"] = _avg_num([r.get("total_gas_cost_native") for r in ok_rows])

            demo_meta["sum_approve_gas_cost_native"] = _sum_num([r.get("approve_gas_cost_native") for r in ok_rows])
            demo_meta["sum_swap_gas_cost_native"] = _sum_num([r.get("swap_gas_cost_native") for r in ok_rows])
            demo_meta["sum_total_gas_cost_native"] = _sum_num([r.get("total_gas_cost_native") for r in ok_rows])

            # Backward-compatible aliases (create_order가 avg_gas_cost_native를 읽는 구조면 필수)
            demo_meta["avg_gas_cost_native"] = demo_meta["avg_total_gas_cost_native"]
            demo_meta["sum_gas_cost_native"] = demo_meta["sum_total_gas_cost_native"]

            # LEG summary log (native only)
            logger.info(
                "[pancake-demo][LEG] symbol=%s side=%s ok=%s fail=%s avg_fill=%s "
                "avg_slip_bps=%s gas_sym=%s "
                "approve_gas_native(avg=%s sum=%s) swap_gas_native(avg=%s sum=%s) total_gas_native(avg=%s sum=%s)",
                symbol,
                side.value,
                ok_count,
                fail_count,
                (f"{demo_meta.get('avg_fill_price'):.10f}" if isinstance(demo_meta.get("avg_fill_price"), (int, float)) else None),
                (f"{(demo_meta.get('avg_return_vs_hint') * 10000.0):.3f}" if isinstance(demo_meta.get("avg_return_vs_hint"), (int, float)) else None),
                (self._params.native_symbol or "BNB"),
                (f"{demo_meta.get('avg_approve_gas_cost_native'):.10f}" if isinstance(demo_meta.get("avg_approve_gas_cost_native"), (int, float)) else None),
                (f"{demo_meta.get('sum_approve_gas_cost_native'):.10f}" if isinstance(demo_meta.get("sum_approve_gas_cost_native"), (int, float)) else None),
                (f"{demo_meta.get('avg_swap_gas_cost_native'):.10f}" if isinstance(demo_meta.get("avg_swap_gas_cost_native"), (int, float)) else None),
                (f"{demo_meta.get('sum_swap_gas_cost_native'):.10f}" if isinstance(demo_meta.get("sum_swap_gas_cost_native"), (int, float)) else None),
                (f"{demo_meta.get('avg_total_gas_cost_native'):.10f}" if isinstance(demo_meta.get("avg_total_gas_cost_native"), (int, float)) else None),
                (f"{demo_meta.get('sum_total_gas_cost_native'):.10f}" if isinstance(demo_meta.get("sum_total_gas_cost_native"), (int, float)) else None),
            )

        return demo_meta


    def _wait_for_upstream_block(
        self,
        upstream_web3: Web3,
        target_block: int,
        timeout_seconds: float = 120.0,
        poll_interval: float = 2.0,
    ) -> None:
        start = time.time()
        while True:
            head = int(upstream_web3.eth.block_number)
            if head >= target_block:
                return

            if time.time() - start > timeout_seconds:
                raise TimeoutError(
                    f"Timed out waiting for upstream block {target_block}. "
                    f"Current head is {head}."
                )

            time.sleep(poll_interval)

    def _rpc_supports(self, provider: Any, method: str) -> bool:
        try:
            resp = provider.make_request(method, [])
            return isinstance(resp, dict) and ("error" not in resp)
        except Exception:
            return False

    def _fork_reset_any(
        self,
        provider: Any,
        reset_config: Dict[str, Any],
        prefer: str = "auto",
    ) -> str:
        if prefer == "hardhat":
            methods = ["hardhat_reset", "anvil_reset"]
        elif prefer == "anvil":
            methods = ["anvil_reset", "hardhat_reset"]
        else:
            methods = ["anvil_reset", "hardhat_reset"]

        errors: Dict[str, Any] = {}

        def _looks_unsupported(err_obj: Any) -> bool:
            try:
                if isinstance(err_obj, dict):
                    msg = str(err_obj.get("message", "") or "")
                    code = err_obj.get("code", None)
                else:
                    msg = str(err_obj)
                    code = None
                msg_l = msg.lower()
                return ("method not found" in msg_l) or ("resource not found" in msg_l) or (code == -32601)
            except Exception:
                return False

        for m in methods:
            try:
                resp = provider.make_request(m, [reset_config])
                if isinstance(resp, dict) and resp.get("error"):
                    err = resp["error"]
                    errors[m] = err
                    if _looks_unsupported(err):
                        continue
                    continue
                return m
            except Exception as exc:
                errors[m] = str(exc)
                continue

        hint = (
            "Hint: your web3 provider is probably NOT an Anvil/Hardhat fork RPC. "
            "Make sure params.web3 points to the local fork node (anvil/hardhat), "
            "not an upstream/public RPC."
        )
        raise RuntimeError(f"fork reset failed (tried {methods}). errors={errors}. {hint}")

    def _impersonate_any(
        self,
        provider: Any,
        addr: str,
        prefer: str = "auto",
    ) -> str:
        if prefer == "hardhat":
            methods = ["hardhat_impersonateAccount", "anvil_impersonateAccount"]
        elif prefer == "anvil":
            methods = ["anvil_impersonateAccount", "hardhat_impersonateAccount"]
        else:
            methods = ["anvil_impersonateAccount", "hardhat_impersonateAccount"]

        last_err: Optional[Any] = None
        for m in methods:
            try:
                resp = provider.make_request(m, [addr])
                if isinstance(resp, dict) and resp.get("error"):
                    last_err = resp["error"]
                    continue
                return m
            except Exception as exc:
                last_err = exc
                continue

        raise RuntimeError(f"impersonate failed (tried {methods}). last_error={last_err}")

    def _detect_fork_engine(self, provider: Any) -> str:
        if self._rpc_supports(provider, "anvil_nodeInfo") or self._rpc_supports(provider, "anvil_reset"):
            return "anvil"
        if self._rpc_supports(provider, "hardhat_metadata") or self._rpc_supports(provider, "hardhat_reset"):
            return "hardhat"
        return "unknown"

    def _receipt_int(self, receipt: Any, key: str, default: int = 0) -> int:
        if receipt is None:
            return default
        if isinstance(receipt, dict):
            return int(receipt.get(key, default) or default)
        return int(getattr(receipt, key, default) or default)

    def _gas_cost_from_receipt(
        self, 
        web3: Web3, 
        receipt: Any, 
        fallback_gas_price_wei: int = 0
    ) -> Tuple[int, int, int]:
        gas_used = int(receipt.get("gasUsed", 0)) if isinstance(receipt, dict) else int(getattr(receipt, "gasUsed", 0))
        gas_price = int(receipt.get("effectiveGasPrice", 0)) if isinstance(receipt, dict) else int(getattr(receipt, "effectiveGasPrice", 0))

        if gas_price <= 0:
            gas_price = int(fallback_gas_price_wei or 0)

        if gas_price <= 0:
            try:
                gas_price = int(web3.eth.gas_price)
            except Exception:
                gas_price = 0

        return gas_used, gas_price, gas_used * gas_price


    def _send_signed_and_wait(
        self, 
        web3: Web3, 
        tx, 
        private_key: str
    ) -> Dict[str, Any]:
        """
        Send a locally-signed tx and return gas usage/cost metadata.
        """
        tx2 = dict(tx)
        tx2.setdefault("chainId", int(web3.eth.chain_id))

        fallback_gas_price_wei = int(tx2.get("gasPrice", 0) or 0)

        signed = Account.sign_transaction(tx2, private_key)
        raw_tx = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None)
        tx_hash = web3.eth.send_raw_transaction(raw_tx)
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

        status = int(receipt.get("status", 0)) if isinstance(receipt, dict) else int(getattr(receipt, "status", 0))
        if status != 1:
            raise RuntimeError(f"tx reverted: hash={tx_hash.hex()}")

        gas_used, gas_price_wei, gas_cost_wei = self._gas_cost_from_receipt(web3, receipt, fallback_gas_price_wei=fallback_gas_price_wei)

        return {
            "tx_hash": tx_hash.hex(),
            "gas_used": int(gas_used),
            "gas_price_wei": int(gas_price_wei),
            "gas_cost_wei": int(gas_cost_wei),
        }

    def _ensure_allowance(
        self,
        web3: Web3,
        owner: str,
        token: Optional[str],
        spender: Optional[str],
        amount_wei: int,
        priv_key: Optional[str],
        use_local_signing: bool,
    ) -> Dict[str, Any]:
        """
        Ensure ERC20 allowance(owner -> spender) is at least amount_wei.

        Returns metadata including approve gas usage/cost.
        """
        meta: Dict[str, Any] = {
            "sent": False,
            "tx_hashes": [],
            "approve_gas_used": 0,
            "approve_gas_cost_wei": 0,
            "approve_gas_cost_native": 0.0, 
            "gas_token_symbol": "BNB",  
            "error": "",
        }

        if not token or not spender or amount_wei <= 0:
            return meta

        token_cs = Web3.to_checksum_address(token)
        spender_cs = Web3.to_checksum_address(spender)
        owner_cs = Web3.to_checksum_address(owner)

        erc20 = web3.eth.contract(address=token_cs, abi=ERC20_ABI)

        try:
            cur = int(erc20.functions.allowance(owner_cs, spender_cs).call())
        except Exception as exc:
            meta["error"] = f"allowance_read_failed: {exc}"
            return meta

        meta["current"] = cur
        if cur >= amount_wei:
            return meta

        def _get_nonce_pending(addr: str) -> int:
            try:
                return int(web3.eth.get_transaction_count(addr, "pending"))
            except Exception:
                return int(web3.eth.get_transaction_count(addr))

        def _send_tx(tx: Dict[str, Any]) -> Any:
            tx.setdefault("from", owner_cs)
            tx.setdefault("chainId", int(web3.eth.chain_id))
            tx.setdefault("gas", 150_000)

            if "gasPrice" not in tx and "maxFeePerGas" not in tx:
                try:
                    tx["gasPrice"] = int(web3.eth.gas_price)
                except Exception:
                    pass

            fallback_gas_price_wei = int(tx.get("gasPrice", 0) or 0)

            if use_local_signing:
                tx["nonce"] = _get_nonce_pending(owner_cs)
                signed = Account.sign_transaction(tx, priv_key)  # type: ignore[arg-type]
                raw = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None)
                tx_hash = web3.eth.send_raw_transaction(raw)
            else:
                tx_hash = web3.eth.send_transaction(tx)

            tx_hash_hex = tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash)
            meta["tx_hashes"].append(tx_hash_hex)

            receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

            gas_used, gas_price_wei, gas_cost_wei = self._gas_cost_from_receipt(
                web3, receipt, fallback_gas_price_wei=fallback_gas_price_wei
            )
            meta["sent"] = True
            meta["approve_gas_used"] += int(gas_used)
            meta["approve_gas_cost_wei"] += int(gas_cost_wei)
            meta["approve_gas_price_wei"] = int(gas_price_wei)

            return receipt

        try:
            if cur != 0:
                tx0 = erc20.functions.approve(spender_cs, 0).build_transaction({"from": owner_cs})
                _send_tx(tx0)
        except Exception:
            pass

        try:
            tx1 = erc20.functions.approve(spender_cs, int(amount_wei)).build_transaction({"from": owner_cs})
            _send_tx(tx1)
        except Exception as exc:
            meta["error"] = f"approve_failed: {exc}"

        return meta


    def _ensure_permit2_allowance(
        self,
        web3: Web3,
        owner: str,
        permit2: str,
        token: str,
        spender: str,
        amount_wei: int,
        priv_key: Optional[str],
        use_local_signing: bool,
    ) -> Dict[str, Any]:
        """
        Ensure Permit2 allowance(owner, token, spender) is sufficient and not expired.
        Returns metadata including gas usage/cost.
        """
        meta: Dict[str, Any] = {
            "sent": False,
            "tx_hashes": [],
            "permit2_gas_used": 0,
            "permit2_gas_price_wei": 0,
            "permit2_gas_cost_wei": 0,
            "current_amount": 0,
            "current_expiration": 0,
            "error": "",
        }

        if not permit2 or not token or not spender or amount_wei <= 0:
            return meta

        owner_cs = Web3.to_checksum_address(owner)
        permit2_cs = Web3.to_checksum_address(permit2)
        token_cs = Web3.to_checksum_address(token)
        spender_cs = Web3.to_checksum_address(spender)

        c = web3.eth.contract(address=permit2_cs, abi=PERMIT2_ABI)

        # Read current allowance
        try:
            res = c.functions.allowance(owner_cs, token_cs, spender_cs).call()
            cur_amount = int(res[0])
            cur_exp = int(res[1])
        except Exception as exc:
            meta["error"] = f"permit2_allowance_read_failed: {exc}"
            return meta

        meta["current_amount"] = cur_amount
        meta["current_expiration"] = cur_exp

        # Expiration check
        try:
            now_ts = int(web3.eth.get_block("latest")["timestamp"])
        except Exception:
            now_ts = 0

        need = (cur_amount < amount_wei) or (cur_exp <= now_ts)
        if not need:
            return meta

        max_uint160 = (1 << 160) - 1
        max_uint48 = (1 << 48) - 1

        tx = c.functions.approve(token_cs, spender_cs, max_uint160, max_uint48).build_transaction(
            {"from": owner_cs}
        )
        tx.setdefault("from", owner_cs)
        tx.setdefault("chainId", int(web3.eth.chain_id))
        tx.setdefault("gas", 250_000)

        if "gasPrice" not in tx and "maxFeePerGas" not in tx:
            try:
                tx["gasPrice"] = int(web3.eth.gas_price)
            except Exception:
                pass

        fallback_gas_price_wei = int(tx.get("gasPrice", 0) or 0)

        try:
            # Nonce
            try:
                nonce = int(web3.eth.get_transaction_count(owner_cs, "pending"))
            except Exception:
                nonce = int(web3.eth.get_transaction_count(owner_cs))
            tx["nonce"] = nonce

            # Send
            if use_local_signing:
                signed = Account.sign_transaction(tx, priv_key)  # type: ignore[arg-type]
                raw = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None)
                tx_hash = web3.eth.send_raw_transaction(raw)
            else:
                # Impersonated mode: node signs
                tx_hash = web3.eth.send_transaction(tx)

            tx_hash_hex = tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash)
            meta["tx_hashes"].append(tx_hash_hex)

            receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

            gas_used, gas_price_wei, gas_cost_wei = self._gas_cost_from_receipt(
                web3, receipt, fallback_gas_price_wei=fallback_gas_price_wei
            )
            meta["sent"] = True
            meta["permit2_gas_used"] += int(gas_used)
            meta["permit2_gas_cost_wei"] += int(gas_cost_wei)
            meta["permit2_gas_price_wei"] = int(gas_price_wei)

        except Exception as exc:
            meta["error"] = f"permit2_approve_failed: {exc}"

        return meta
