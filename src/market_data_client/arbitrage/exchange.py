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
    """

    web3: Any
    account_address: str
    build_swap_tx: Callable[[Any, str, float, str], Dict[str, Any]]
    private_key: Optional[str] = None
    block_offsets: Tuple[int, int, int] = (1, 2, 3)
    default_fee_rate: float = 0.0005


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
        from decimal import Decimal, ROUND_DOWN

        q = Decimal(str(qty))
        s = Decimal(str(step_size))

        if s <= 0:
            return format(q, "f")

        rounded = (q // s) * s

        # Ensure we don't produce scientific notation
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

        # Pick usable step/min (avoid MARKET_LOT_SIZE=0.00000000 trap)
        step_candidates = [s for s in (lot_step, mlot_step) if s > 0]
        min_candidates = [m for m in (lot_min, mlot_min) if m > 0]

        step = max(step_candidates) if step_candidates else Decimal("0")
        mn = max(min_candidates) if min_candidates else Decimal("0")

        # min notional (best-effort)
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

        # Choose the coarser (larger) step to satisfy both validators.
        step = lot_step if lot_step > 0 else mlot_step
        if lot_step > 0 and mlot_step > 0:
            step = max(lot_step, mlot_step)

        # Min qty: choose the stricter (larger) minQty if both present.
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
        from decimal import Decimal, ROUND_DOWN

        q = Decimal(str(qty))
        step = Decimal(str(step_size))
        mn = Decimal(str(min_qty))

        if step > 0:
            q2 = (q // step) * step
        else:
            q2 = q

        if q2 < mn:
            return "0"

        # Decide formatting precision
        if qty_precision is not None and qty_precision >= 0:
            decimals = int(qty_precision)
        else:
            step_norm = step.normalize() if step > 0 else Decimal("0")
            decimals = max(0, -step_norm.as_tuple().exponent)

        quant = Decimal("1").scaleb(-decimals)  # 10^-decimals
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
        """
        Create a demo order on Binance.

        Economic layer:
          - Uses `price` as theoretical/mainnet price.
          - Applies slippage_bps (default self._default_slippage_bps) to derive
            execution_price.
          - Applies taker fee on the quote asset (self._fee_rate).
          - Updates BotState balances (off-chain accounting).
          - Returns TradeResult with execution_price and deltas.

        Infrastructure layer (optional):
          - If self._params.use_testnet_execution is True AND both api_key and
            api_secret are set, sends a MARKET order to Binance testnet REST.
          - Stores the raw HTTP response or error under metadata["binance_testnet"].
        """
        if price is None or price <= 0:
            raise ValueError("Binance demo client requires a positive theoretical price")

        theoretical_price = float(price)
        qty = float(quantity)

        # Slippage in basis points (1 bps = 0.01%)
        slippage_bps = float(kwargs.get("slippage_bps", self._default_slippage_bps))
        slip_factor = slippage_bps / 10_000.0

        # Derive execution price from theoretical price + directional slippage.
        if side == OrderSide.BUY:
            # Buyer pays slightly worse price
            execution_price = theoretical_price * (1.0 + slip_factor)
        elif side == OrderSide.SELL:
            # Seller receives slightly worse price
            execution_price = theoretical_price * (1.0 - slip_factor)
        else:
            raise ValueError(f"Unsupported side for Binance demo client: {side}")

        # Split symbol into base / quote, e.g. "BNBUSDT" -> ("BNB", "USDT")
        base_asset, quote_asset = self._split_symbol(symbol)
        account = self._state.get_or_create_account(self.name, self.instrument)

        # --- Economic layer: update balances using execution_price ---
        if side == OrderSide.BUY:
            cost_quote = qty * execution_price
            if account.balances.get(quote_asset, 0.0) < cost_quote:
                raise InsufficientBalanceError(
                    f"Not enough {quote_asset} balance to buy {symbol}: "
                    f"have={account.balances.get(quote_asset, 0.0)} need={cost_quote}"
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
            # This should be unreachable due to the earlier branch
            raise ValueError(f"Unsupported side for Binance demo client: {side}")

        # Apply taker fee on the quote asset
        fee_amount = abs(quote_delta_before_fee) * self._fee_rate
        account.balances[quote_asset] -= fee_amount
        quote_delta_after_fee = quote_delta_before_fee - fee_amount

        # Build TradeResult (economic simulation only)
        trade = TradeResult(
            exchange=self.name,
            instrument=self.instrument,
            symbol=symbol,
            side=side,
            quantity=qty,
            price=execution_price,
            fee=self._fee_rate,
            base_delta=base_delta,
            quote_delta=quote_delta_after_fee,
            success=True,
        )

        # Attach demo economic metadata
        demo_meta: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.value,
            "instrument": self.instrument,
            "theoretical_price": theoretical_price,
            "execution_price": execution_price,
            "slippage_bps": slippage_bps,
            "fee_rate": self._fee_rate,
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
                # Do not crash the economic simulation due to testnet issues.
                trade.metadata["binance_testnet_error"] = str(exc)
                if self._params.fail_on_testnet_error:
                    trade.success = False
                    trade.message = f"binance_testnet failed: {exc}"

        self._state.record_trade(trade)
        return trade


    def _select_testnet_endpoint_and_creds(self) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]:
        """
        Returns (base_url, path, api_key, api_secret, mode_label).

        mode_label is one of: "spot", "futures".
        If creds are missing, api_key/api_secret can be None.
        """
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

        # --- Fetch filters & build candidate qty strings (coarser fallback) ---
        try:
            filters = self._get_testnet_symbol_filters_cached(mode_label, base_url, symbol, order_type)
            step = Decimal(str(filters.get("stepSize", "0") or "0"))
            mn = Decimal(str(filters.get("minQty", "0") or "0"))

            q_raw = Decimal(str(quantity))

            # Floor to stepSize first
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

            # Start with step-decimals, then try fewer decimals (coarser) to bypass -1111
            step_decimals = max(0, -step.normalize().as_tuple().exponent) if step > 0 else 8

            candidates: List[str] = []
            seen = set()

            for d in range(step_decimals, -1, -1):
                quant = Decimal("1").scaleb(-d)  # 10^-d
                # Coarsen by flooring to 10^-d (only affects testnet infra layer)
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

         # --- Try candidates until success ---
        last_err: Optional[Dict[str, Any]] = None

        for qty_str in candidates:
            logger.info(
                "[binance-testnet] mode=%s symbol=%s raw_qty=%.18f step=%s min=%s try_qty=%s",
                mode_label, symbol, float(quantity),
                filters.get("stepSize"), filters.get("minQty"),
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

            # Retry also on LOT_SIZE failures (common when first candidate isn't step-aligned)
            if code == -1111:
                continue
            if code == -1013 and "LOT_SIZE" in msg:
                continue

            # other errors: stop immediately
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
        → sign locally and use eth_sendRawTransaction (no impersonate).
    - If private_key is None:
        → impersonate account_address on fork node and use eth_sendTransaction.
    """

    def __init__(
        self,
        exchange_name: str,
        instrument: str,
        state: "BotState",
        params: PancakeDemoParams,
    ) -> None:
        super().__init__(exchange_name, instrument, state)
        self._mode = ExecutionMode.DEMO  # Demo mode for forked-chain swaps
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
        """
        Create an order on PancakeSwap in DEMO mode.

        - price / price_hint: theoretical expected price from the strategy.
        - Forked-chain simulation (_run_multi_block_flow) computes avg_fill_price.
        - BotState balances and TradeResult.price use that avg_fill_price
          when available; otherwise fall back to the theoretical price.
        """
        # 1) Resolve theoretical price from price or price_hint
        if price is None and price_hint is not None:
            price = price_hint
        if price is None:
            raise ValueError("Pancake demo client requires an explicit price or price_hint")

        theoretical_price = float(price)

        if fee_rate is None:
            fee_rate = self._params.default_fee_rate

        account = self._state.get_or_create_account(self.name, self.instrument)

        # 2) Run forked-chain multi-block simulation to get per-block metadata
        per_block_meta: Optional[Dict[str, Any]] = None
        if self._params.block_offsets and len(self._params.block_offsets) > 0:
            per_block_meta = await self._run_multi_block_flow(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=theoretical_price,
            )

        if per_block_meta is not None:
            all_ok = bool(per_block_meta.get("all_ok", True))
            avg_fill = per_block_meta.get("avg_fill_price", None)

            if (not all_ok) or (not isinstance(avg_fill, (int, float))) or (avg_fill <= 0):
                trade = TradeResult(
                    exchange=self.name,
                    instrument=self.instrument,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=float(price),
                    fee=fee_rate,
                    base_delta=0.0,
                    quote_delta=0.0,
                    success=False,
                    message="pancake_demo swap failed (one or more fork swaps reverted)",
                    metadata={"pancake_demo": per_block_meta},
                )
                return trade

        # 3) Determine actual execution price used for accounting:
        #    prefer avg_fill_price from fork, otherwise use theoretical price.
        execution_price = theoretical_price
        if per_block_meta is not None:
            avg_fill = per_block_meta.get("avg_fill_price")
            if isinstance(avg_fill, (int, float)) and avg_fill > 0:
                execution_price = float(avg_fill)

        base_asset, quote_asset = resolve_base_quote(symbol)

        # 4) Apply balance changes using execution_price
        if side == OrderSide.BUY:
            cost_quote = quantity * execution_price
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
            proceeds_quote = quantity * execution_price
            account.balances[base_asset] = account.balances.get(base_asset, 0.0) - quantity
            account.balances[quote_asset] = account.balances.get(quote_asset, 0.0) + proceeds_quote
            base_delta = -quantity
            quote_delta = proceeds_quote

        else:
            raise ValueError(f"Unsupported side for Pancake demo client: {side}")

        # 5) Apply taker fee on quote asset
        fee_amount = abs(quote_delta) * fee_rate
        account.balances[quote_asset] -= fee_amount
        quote_delta_after_fee = quote_delta - fee_amount

        trade = TradeResult(
            exchange=self.name,
            instrument=self.instrument,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=execution_price,
            fee=fee_rate,
            base_delta=base_delta,
            quote_delta=quote_delta_after_fee,
            success=True,
        )

        # 6) Attach demo metadata with both theoretical and execution price info
        if per_block_meta is not None:
            enriched_meta = dict(per_block_meta)
            enriched_meta.setdefault("theoretical_price", theoretical_price)
            enriched_meta.setdefault(
                "price_hint",
                price_hint if price_hint is not None else theoretical_price,
            )
            enriched_meta.setdefault("execution_price", execution_price)

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
    ) -> Dict[str, Any]:
        """
        Run a multi-block simulation on the forked chain.

        For each offset in block_offsets:
          - Reset the fork to base_block + offset
          - Impersonate or sign locally depending on params.private_key
          - Build and send one swap transaction
          - Collect receipt and any useful metadata
        """
        loop = asyncio.get_running_loop()
        meta = await loop.run_in_executor(
            None,
            self._run_multi_block_swaps_sync,
            symbol,
            side,
            quantity,
            price,
        )
        return meta
    
    def _run_multi_block_swaps_sync(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
    ) -> Dict[str, Any]:
        """
        Run the same swap on multiple forked blocks and collect per-block metadata.

        For each forked block this method:
        - Resets the fork to base_block + offset (with anvil/hardhat fallback)
        - Impersonates or signs locally depending on params.private_key
        - Builds and sends one swap transaction
        - Measures token balances before and after the swap
        - Records per-block fill price and return vs hint
        """
        web3 = self._web3
        offsets = tuple(self._params.block_offsets or (1, 2, 3))

        # Upstream (real chain) web3 for reading latest block height.
        upstream_web3 = Web3(Web3.HTTPProvider(self._params.upstream_rpc_url))
        _apply_poa_middleware(upstream_web3)

        base_block = int(upstream_web3.eth.block_number)

        provider = web3.provider
        if provider is None:
            raise RuntimeError("Web3 provider must not be None for Pancake demo client")

        # Decide signing mode and trader address once.
        priv_key = self._params.private_key
        use_local_signing = bool(priv_key)

        if use_local_signing:
            acct = Account.from_key(priv_key)  # type: ignore[arg-type]
            trader = Web3.to_checksum_address(acct.address)
        else:
            trader = Web3.to_checksum_address(self._params.account_address)

        # Treat fork_engine as preference only; always fallback if the preferred engine fails.
        fork_engine_pref = getattr(self._params, "fork_engine", "auto") or "auto"
        if fork_engine_pref == "auto":
            detected = self._detect_fork_engine(provider)
            # Not fatal: even if unknown, fallback reset will still try both.
            if detected in ("anvil", "hardhat"):
                fork_engine_pref = detected

        def _get_nonce_safe(addr: str) -> int:
            """Get a usable nonce; prefer pending if supported."""
            try:
                return int(web3.eth.get_transaction_count(addr, "pending"))
            except Exception:
                return int(web3.eth.get_transaction_count(addr))

        per_block_results: List[Dict[str, Any]] = []

        for offset in offsets:
            fork_block = base_block + offset
            self._wait_for_upstream_block(upstream_web3, fork_block)

            reset_config = {
                "forking": {
                    "jsonRpcUrl": self._params.upstream_rpc_url,
                    "blockNumber": fork_block,
                }
            }

            # --- Reset fork with fallback (anvil_reset <-> hardhat_reset) ---
            used_reset_method = ""
            try:
                used_reset_method = self._fork_reset_any(provider, reset_config, prefer=fork_engine_pref)
            except Exception as exc:
                raise RuntimeError(f"fork reset failed at block={fork_block}: {exc}") from exc

            # --- Impersonate with fallback (only if not local signing) ---
            used_impersonate_method = ""
            if not use_local_signing:
                try:
                    used_impersonate_method = self._impersonate_any(provider, trader, prefer=fork_engine_pref)
                except Exception as exc:
                    raise RuntimeError(f"impersonate failed for trader={trader}: {exc}") from exc

            # Build swap tx.
            # IMPORTANT: The builder must handle any funding/Permit2 allowance itself.
            try:
                # Some builders accept price, some do not. Try with price first.
                tx_dict = self._params.build_swap_tx(
                    web3=web3,
                    symbol=symbol,
                    quantity=float(quantity),
                    side=side.value,
                    price=float(price),
                )
            except TypeError:
                tx_dict = self._params.build_swap_tx(
                    web3=web3,
                    symbol=symbol,
                    quantity=float(quantity),
                    side=side.value,
                )

            # Read demo metadata BEFORE stripping _demo_* keys
            token_in = tx_dict.get("_demo_token_in", None)
            token_out = tx_dict.get("_demo_token_out", None)
            amount_in_wei = tx_dict.get("_demo_amount_in_wei", None)

            # Remove demo helper keys before sending.
            for k in list(tx_dict.keys()):
                if k.startswith("_demo_"):
                    tx_dict.pop(k, None)

            # Enforce sender + chainId for signing safety.
            tx_dict["from"] = trader
            tx_dict.setdefault("chainId", int(web3.eth.chain_id))

            # Provide reasonable defaults when builder didn't set them.
            if "gas" not in tx_dict:
                try:
                    est = int(web3.eth.estimate_gas(tx_dict))
                    tx_dict["gas"] = int(est * 12 // 10)  # +20% buffer
                except Exception:
                    tx_dict["gas"] = 600_000

            if "gasPrice" not in tx_dict and "maxFeePerGas" not in tx_dict:
                try:
                    tx_dict["gasPrice"] = int(web3.eth.gas_price)
                except Exception:
                    pass

            if use_local_signing:
                # Raw tx requires explicit nonce; do not override if builder set it.
                tx_dict.setdefault("nonce", _get_nonce_safe(trader))

            # Prepare ERC20 contracts for balance delta measurement.
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

            # --- FAIL FAST: avoid TRANSFER_FROM_FAILED by checking token_in balance first ---
            # This requires builder to provide `_demo_amount_in_wei`.
            if (
                token_in_contract is not None
                and bal_in_before is not None
                and amount_in_wei is not None
            ):
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
                                "revert_reason": (
                                    "insufficient token_in balance before swap: "
                                    f"have={have} need={need}"
                                ),
                                "reset_method": used_reset_method,
                                "impersonate_method": used_impersonate_method,
                            }
                        )
                        continue
                except Exception:
                    # Best-effort; if parsing fails, continue to normal send path.
                    pass

            tx_hash_hex = ""
            status = 0
            gas_used = 0
            revert_reason = None

            try:
                if use_local_signing:
                    signed = Account.sign_transaction(tx_dict, priv_key)  # type: ignore[arg-type]
                    raw_tx = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None)
                    tx_hash = web3.eth.send_raw_transaction(raw_tx)
                else:
                    tx_hash = web3.eth.send_transaction(tx_dict)

                tx_hash_hex = tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash)

                receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

                if isinstance(receipt, dict):
                    status = int(receipt.get("status", 0))
                    gas_used = int(receipt.get("gasUsed", 0))
                else:
                    status = int(getattr(receipt, "status", 0))
                    gas_used = int(getattr(receipt, "gasUsed", 0))

                if status == 0:
                    # Best-effort revert reason extraction.
                    try:
                        web3.eth.call(
                            {
                                "to": tx_dict.get("to"),
                                "from": tx_dict.get("from"),
                                "data": tx_dict.get("data"),
                                "value": tx_dict.get("value", 0),
                            },
                            block_identifier="latest",
                        )
                    except Exception as exc:
                        revert_reason = str(exc)

            except Exception as exc:
                revert_reason = f"send_or_wait_error: {exc}"

            delta_in = None
            delta_out = None
            fill_price = None
            return_vs_hint = None

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

                    # Assume builder mapping:
                    # - BUY  : token_in = quote (spent), token_out = base (received)
                    # - SELL : token_in = base  (spent), token_out = quote(received)
                    if side == OrderSide.BUY:
                        quote_qty = (-delta_in) / scale_in
                        base_qty = (delta_out) / scale_out
                    else:
                        base_qty = (-delta_in) / scale_in
                        quote_qty = (delta_out) / scale_out

                    if base_qty and base_qty > 0:
                        fill_price = quote_qty / base_qty
                        if price > 0:
                            return_vs_hint = fill_price / price - 1.0

                except Exception:
                    delta_in = None
                    delta_out = None
                    fill_price = None
                    return_vs_hint = None

            result: Dict[str, Any] = {
                "fork_block": fork_block,
                "tx_hash": tx_hash_hex,
                "status": status,
                "gas_used": gas_used,
                "reset_method": used_reset_method,
                "impersonate_method": used_impersonate_method,
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
            if fill_price is not None:
                result["fill_price"] = fill_price
            if return_vs_hint is not None:
                result["return_vs_hint"] = return_vs_hint

            per_block_results.append(result)

        account = self._state.get_or_create_account(self.name, self.instrument)

        demo_meta: Dict[str, Any] = {
            "base_block": base_block,
            "block_offsets": list(offsets),
            "per_block_results": per_block_results,
            "account_snapshot": {"balances": dict(account.balances)},
        }

        all_ok = all(int(r.get("status", 0) or 0) == 1 for r in per_block_results)
        demo_meta["all_ok"] = all_ok
        demo_meta["ok_count"] = sum(1 for r in per_block_results if int(r.get("status", 0) or 0) == 1)
        demo_meta["fail_count"] = len(per_block_results) - demo_meta["ok_count"]

        fill_prices = [r["fill_price"] for r in per_block_results if "fill_price" in r]
        if all_ok and len(fill_prices) == len(per_block_results):
            demo_meta["avg_fill_price"] = sum(fill_prices) / len(fill_prices)
        else:
            demo_meta["avg_fill_price"] = None

        returns = [r["return_vs_hint"] for r in per_block_results if "return_vs_hint" in r]
        if returns:
            demo_meta["avg_return_vs_hint"] = sum(returns) / len(returns)

        return demo_meta



    def _wait_for_upstream_block(
        self,
        upstream_web3: Web3,
        target_block: int,
        timeout_seconds: float = 120.0,
        poll_interval: float = 2.0,
    ) -> None:
        """
        Wait until the upstream chain has produced at least `target_block`.
        """
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
        """
        Try fork reset using Anvil/Hardhat methods with fallback.

        Returns:
            The RPC method name used for reset.
        """
        methods: List[str]
        if prefer == "hardhat":
            methods = ["hardhat_reset", "anvil_reset"]
        elif prefer == "anvil":
            methods = ["anvil_reset", "hardhat_reset"]
        else:
            methods = ["anvil_reset", "hardhat_reset"]

        last_err: Optional[Any] = None
        for m in methods:
            try:
                resp = provider.make_request(m, [reset_config])
                if isinstance(resp, dict) and resp.get("error"):
                    last_err = resp["error"]
                    continue
                return m
            except Exception as exc:
                last_err = exc
                continue

        raise RuntimeError(f"fork reset failed (tried {methods}). last_error={last_err}")

    def _impersonate_any(
        self,
        provider: Any,
        addr: str,
        prefer: str = "auto",
    ) -> str:
        """
        Try account impersonation using Anvil/Hardhat methods with fallback.

        Returns:
            The RPC method name used for impersonation.
        """
        methods: List[str]
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

    def _ensure_allowance(
        self,
        web3: Web3,
        owner: str,
        token: Optional[str],
        spender: Optional[str],
        amount_wei: int,
        priv_key: Optional[str],
        use_local_signing: bool,
    ) -> bool:
        """
        Ensure ERC20 allowance(owner -> spender) is at least amount_wei.

        Returns:
            True  if an approve transaction was sent (allowance updated),
            False if allowance was already enough or inputs were invalid.
        """
        if not token or not spender:
            return False
        if amount_wei <= 0:
            return False

        token = Web3.to_checksum_address(token)
        spender = Web3.to_checksum_address(spender)
        owner = Web3.to_checksum_address(owner)

        erc20 = web3.eth.contract(address=token, abi=ERC20_ABI)

        # Read current allowance
        try:
            cur = int(erc20.functions.allowance(owner, spender).call())
        except Exception:
            return False

        # If enough, nothing to do
        if cur >= amount_wei:
            return False

        def _send_tx(tx: Dict[str, Any]) -> None:
            """
            Send a transaction either by local signing (eth_sendRawTransaction)
            or by node-managed signing (eth_sendTransaction / impersonation).
            """
            tx.setdefault("from", owner)
            tx.setdefault("chainId", int(web3.eth.chain_id))
            tx.setdefault("gas", 150_000)

            # BSC forks often use legacy gasPrice
            if "gasPrice" not in tx and "maxFeePerGas" not in tx:
                try:
                    tx["gasPrice"] = int(web3.eth.gas_price)
                except Exception:
                    pass

            if use_local_signing:
                # Raw transactions require an explicit nonce
                tx.setdefault("nonce", web3.eth.get_transaction_count(owner))
                signed = Account.sign_transaction(tx, priv_key)  # type: ignore[arg-type]
                raw = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None)
                tx_hash = web3.eth.send_raw_transaction(raw)
            else:
                tx_hash = web3.eth.send_transaction(tx)

            web3.eth.wait_for_transaction_receipt(tx_hash)

        # Some tokens require setting allowance to 0 before changing it
        try:
            if cur != 0:
                tx0 = erc20.functions.approve(spender, 0).build_transaction({"from": owner})
                _send_tx(tx0)
        except Exception:
            # If 0-approve fails, still try direct approve below
            pass

        # Approve the required amount
        tx1 = erc20.functions.approve(spender, amount_wei).build_transaction({"from": owner})
        _send_tx(tx1)
        return True
