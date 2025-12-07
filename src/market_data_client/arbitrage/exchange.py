from __future__ import annotations

import abc
import asyncio
import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, List

from hexbytes import HexBytes
from web3 import Web3
from .config import ExecutionMode
from eth_account import Account 

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
    Parameters for Binance demo mode.

    For now this class is only stored and not used directly.
    You can later plug in Binance testnet APIs here if you want.
    """

    api_key: str
    api_secret: str
    base_url: str = "https://testnet.binance.vision"


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

class BinanceDemoExchangeClient(PaperExchangeClient):
    """
    Binance demo client.

    Right now this class behaves like PaperExchangeClient, but keeps
    a BinanceDemoParams instance so that you can later plug in
    Binance testnet API calls if you want to.
    """

    def __init__(
        self,
        name: str,
        instrument: str,
        state: "BotState",
        params: Optional[BinanceDemoParams] = None,
        fee_rate: float = 0.0005,
    ) -> None:
        super().__init__(name=name, instrument=instrument, state=state, fee_rate=fee_rate)
        self._params = params
        logger.info(
            "BinanceDemoExchangeClient initialized in in-memory mode. "
            "Wire real Binance testnet APIs here if needed."
        )


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
        self._params = params

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        quote_asset: str = "USDT",
        fee_rate: Optional[float] = None,
        price_hint: Optional[float] = None,
        **_: object,
    ) -> TradeResult:
        if price is None and price_hint is not None:
            price = price_hint
        if price is None:
            raise ValueError("Pancake demo client requires an explicit price or price_hint")

        if fee_rate is None:
            fee_rate = self._params.default_fee_rate

        account = self._state.get_or_create_account(self.name, self.instrument)

        per_block_meta: Optional[Dict[str, Any]] = None

        if self._params.block_offsets and len(self._params.block_offsets) > 0:
            per_block_meta = await self._run_multi_block_flow(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
            )

        base_asset = symbol.replace(quote_asset, "")

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
            raise ValueError(f"Unsupported side for Pancake demo client: {side}")

        # Apply taker fee on quote asset
        fee_amount = abs(quote_delta) * fee_rate
        account.balances[quote_asset] -= fee_amount
        quote_delta_after_fee = quote_delta - fee_amount

        trade = TradeResult(
            exchange=self.name,
            instrument=self.instrument,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            fee=fee_rate,
            base_delta=base_delta,
            quote_delta=quote_delta_after_fee,
            success=True,
        )

        if per_block_meta is not None:
            trade.metadata = trade.metadata or {}
            trade.metadata["pancake_demo"] = per_block_meta

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
          - Resets the local fork to base_block + offset
          - Builds a swap transaction via self._params.build_swap_tx(...)
          - Measures token_in and token_out balances before and after the swap
          - Records per-block balance deltas, fill price, and return vs price hint
        """
        web3 = self._web3

        try:
            web3.middleware_onion.clear()
        except Exception:
            pass

        if self._params.block_offsets is None:
            offsets = (1, 2, 3)
        else:
            offsets = tuple(self._params.block_offsets)

        upstream_web3 = Web3(Web3.HTTPProvider(self._params.upstream_rpc_url))
        try:
            upstream_web3.middleware_onion.clear()
        except Exception:
            pass

        base_block = int(upstream_web3.eth.block_number)

        provider = web3.provider
        if provider is None:
            raise RuntimeError("Web3 provider must not be None for Pancake demo client")

        fork_engine = getattr(self._params, "fork_engine", "anvil")

        priv_key = self._params.private_key
        use_local_signing = bool(priv_key)

        if use_local_signing:
            acct = Account.from_key(priv_key)  # type: ignore[arg-type]
            trader = Web3.to_checksum_address(acct.address)
        else:
            trader = Web3.to_checksum_address(self._params.account_address)

        per_block_results: List[Dict[str, Any]] = []

        for offset in offsets:
            fork_block = base_block + offset

            self._wait_for_upstream_block(upstream_web3, fork_block)

            if fork_engine == "hardhat":
                reset_config = {
                    "forking": {
                        "jsonRpcUrl": self._params.upstream_rpc_url,
                        "blockNumber": fork_block,
                    }
                }
                resp = provider.make_request("hardhat_reset", [reset_config])
                if isinstance(resp, dict) and "error" in resp:
                    raise RuntimeError(f"hardhat_reset failed: {resp['error']}")

                if not use_local_signing:
                    provider.make_request("hardhat_impersonateAccount", [trader])

            elif fork_engine == "anvil":
                reset_config = {
                    "forking": {
                        "jsonRpcUrl": self._params.upstream_rpc_url,
                        "blockNumber": fork_block,
                    }
                }
                resp = provider.make_request("anvil_reset", [reset_config])
                if isinstance(resp, dict) and "error" in resp:
                    raise RuntimeError(f"anvil_reset failed: {resp['error']}")

                if not use_local_signing:
                    provider.make_request("anvil_impersonateAccount", [trader])

            else:
                raise ValueError(f"Unknown fork engine: {fork_engine}")

            tx_dict = self._params.build_swap_tx(
                web3=web3,
                symbol=symbol,
                quantity=quantity,
                side=side.value,
            )

            # Demo-only metadata injected by build_swap_tx_router
            token_in = tx_dict.pop("_demo_token_in", None)
            token_out = tx_dict.pop("_demo_token_out", None)

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

            tx_dict["from"] = trader

            bal_in_before = None
            bal_out_before = None
            decimals_in = None
            decimals_out = None

            if token_in_contract is not None and token_out_contract is not None:
                try:
                    bal_in_before = token_in_contract.functions.balanceOf(trader).call()
                    bal_out_before = token_out_contract.functions.balanceOf(trader).call()
                    decimals_in = token_in_contract.functions.decimals().call()
                    decimals_out = token_out_contract.functions.decimals().call()
                except Exception:
                    bal_in_before = None
                    bal_out_before = None
                    decimals_in = None
                    decimals_out = None

            tx_hash_hex = ""
            status = 0
            gas_used = 0
            revert_reason = None

            try:
                if use_local_signing:
                    signed = Account.sign_transaction(tx_dict, priv_key)  # type: ignore[arg-type]
                    raw_tx = getattr(signed, "rawTransaction", None)
                    if raw_tx is None:
                        raw_tx = getattr(signed, "raw_transaction", None)
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
                    try:
                        web3.eth.call(
                            {
                                "to": tx_dict["to"],
                                "from": tx_dict["from"],
                                "data": tx_dict["data"],
                                "value": tx_dict.get("value", 0),
                            },
                            block_identifier=fork_block,
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
            ):
                try:
                    bal_in_after = token_in_contract.functions.balanceOf(trader).call()
                    bal_out_after = token_out_contract.functions.balanceOf(trader).call()

                    delta_in = int(bal_in_after) - int(bal_in_before)
                    delta_out = int(bal_out_after) - int(bal_out_before)

                    if decimals_in is None:
                        decimals_in = token_in_contract.functions.decimals().call()
                    if decimals_out is None:
                        decimals_out = token_out_contract.functions.decimals().call()

                    if decimals_in is not None and decimals_out is not None and delta_in != 0:
                        scale_in = 10 ** int(decimals_in)
                        scale_out = 10 ** int(decimals_out)

                        qty_in = -delta_in / scale_in if delta_in < 0 else delta_in / scale_in
                        qty_out = delta_out / scale_out

                        if qty_in != 0:
                            fill_price = qty_out / abs(qty_in)
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
            "account_snapshot": {
                "balances": dict(account.balances),
            },
        }

        fill_prices = [
            r["fill_price"] for r in per_block_results if "fill_price" in r
        ]
        returns = [
            r["return_vs_hint"] for r in per_block_results if "return_vs_hint" in r
        ]

        if fill_prices:
            demo_meta["avg_fill_price"] = sum(fill_prices) / len(fill_prices)
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
