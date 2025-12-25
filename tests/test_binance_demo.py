import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# Ensure "src" is on sys.path so that "market_data_client" can be imported
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_data_client.market_data_client import make_client_simple
from market_data_client.arbitrage.exchange import (
    OrderSide,
    BinanceDemoExchangeClient,
    BinanceDemoParams,
)
from market_data_client.arbitrage.state import BotState

# Environment variables required for this test (price feed only).
#
# MDC_HINT_* : used to fetch a real-time mid price from MarketDataClient.
# DEMO_BINANCE_* : optional; if provided and use_testnet_execution is enabled,
#                  the client will also send real orders to Binance testnet.
REQUIRED_ENVS = [
    "MDC_HINT_EXCHANGE",  # e.g. "Binance"
    "MDC_HINT_SYMBOL",    # e.g. "BNBUSDT"
]

MISSING_ENVS = [name for name in REQUIRED_ENVS if not os.getenv(name)]

print("[Binance demo env check]")
for name in REQUIRED_ENVS + ["DEMO_BINANCE_SPOT_API_KEY", "DEMO_BINANCE_API_SECRET"]:
    raw = os.getenv(name)
    # Do not print secrets in clear text
    if "SECRET" in name or "API_KEY" in name:
        display = "***hidden***" if raw else None
    else:
        display = raw
    print(f"  {name} = {display!r}")


async def _get_price_hint_from_market_data(side: str) -> Optional[float]:
    """
    Fetch a reference price from MarketDataClient and use it as price_hint
    for the Binance demo client.

    Env config:
      - NATS_URL              (optional, default: nats://127.0.0.1:4222)
      - MDC_HINT_EXCHANGE     e.g. "Binance"
      - MDC_HINT_SYMBOL       e.g. "BNBUSDT"
      - MDC_HINT_INSTRUMENT   e.g. "spot" or "perpetual" (default: "perpetual")
      - MDC_HINT_TIMEOUT_SEC  e.g. 20  (how long to wait for first price)
      - MDC_HINT_USE_JS       "0" to disable JetStream, anything else = use JS

    For CEX (Binance), we simply use the mid price as the theoretical price
    for both BUY and SELL; side is only used for logging.
    """
    exchange = os.getenv("MDC_HINT_EXCHANGE")
    symbol = os.getenv("MDC_HINT_SYMBOL")
    instrument = os.getenv("MDC_HINT_INSTRUMENT", "perpetual").lower()

    # If not configured, skip and fall back to static env hint
    if not exchange or not symbol:
        print(
            "[mdc-hint] MDC_HINT_EXCHANGE / MDC_HINT_SYMBOL not set, "
            "using static DEMO_BINANCE_TEST_PRICE_HINT"
        )
        return None

    nats_url = os.getenv("NATS_URL", "nats://127.0.0.1:4222")
    use_js = os.getenv("MDC_HINT_USE_JS", "1") != "0"
    timeout_sec = float(os.getenv("MDC_HINT_TIMEOUT_SEC", "5"))

    print(
        f"[mdc-hint-debug] EX= {exchange} SYM= {symbol} INST= {instrument} "
        f"TO= {timeout_sec} JS= {os.getenv('MDC_HINT_USE_JS')} NATS_URL= {os.getenv('NATS_URL')}"
    )

    print(
        f"[mdc-hint] Waiting up to {timeout_sec}s for market price "
        f"{exchange} {instrument} {symbol} via {nats_url} "
        f"(JetStream={use_js})"
    )

    # Build MarketDataClient for CEX symbols
    client = make_client_simple(
        nats_url=nats_url,
        use_jetstream=use_js,
        cex_exchange=exchange,
        cex_instruments=[instrument],
        cex_symbols=[symbol],
        dex_exchange=None,
        dex_chain=None,
        dex_pairs=None,
        enable_csv=False,
    )

    await client.start()
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_sec
        price: Optional[float] = None

        while loop.time() < deadline:
            pd = await client.get_latest_price_with_latency(symbol, exchange, instrument)
            if pd and pd.get("mid") is not None:
                price = float(pd["mid"])
                print(
                    "[mdc-hint] Using CEX mid price from market data: "
                    f"exchange={exchange}, instrument={instrument}, "
                    f"symbol={symbol}, side={side}, mid={price}"
                )
                break

            await asyncio.sleep(0.1)

        if price is None:
            print(
                f"[mdc-hint] No market price for {exchange} {instrument} {symbol} "
                f"within {timeout_sec}s, using static DEMO_BINANCE_TEST_PRICE_HINT"
            )

        return price

    finally:
        await client.stop()


def _inspect_binance_trade(label: str, trade) -> None:
    """
    Helper to inspect and log binance_demo metadata for a single trade.

    The BinanceDemoExchangeClient stores a compact summary under:
      trade.metadata["binance_demo"]
    which includes:
      - symbol, side, instrument
      - theoretical_price (hint from MarketDataClient or env)
      - execution_price (theoretical_price ± slippage)
      - slippage_bps
      - fee_rate
      - base_delta
      - quote_delta_before_fee
      - quote_delta_after_fee

    If testnet execution is enabled, the raw HTTP result (or error)
    is stored separately under trade.metadata["binance_testnet"].
    """
    meta = trade.metadata or {}
    assert "binance_demo" in meta, "Expected binance_demo metadata on TradeResult"

    demo_meta: Dict[str, Any] = meta["binance_demo"]

    theoretical_price = demo_meta.get("theoretical_price")
    execution_price = demo_meta.get("execution_price")
    slippage_bps = demo_meta.get("slippage_bps")
    fee_rate = demo_meta.get("fee_rate")
    base_delta = demo_meta.get("base_delta")
    q_before = demo_meta.get("quote_delta_before_fee")
    q_after = demo_meta.get("quote_delta_after_fee")

    implied_ret = None
    if theoretical_price and execution_price:
        implied_ret = execution_price / theoretical_price - 1.0

    print(f"\n=== {label} leg (Binance demo) ===")
    print(
        f"[binance-{label}] symbol={demo_meta.get('symbol')} "
        f"side={demo_meta.get('side')} instrument={demo_meta.get('instrument')}"
    )
    print(
        f"[binance-{label}] theoretical_price={theoretical_price} "
        f"execution_price={execution_price} slippage_bps={slippage_bps} "
        f"fee_rate={fee_rate}"
    )
    print(
        f"[binance-{label}] base_delta={base_delta} "
        f"quote_delta_before_fee={q_before} quote_delta_after_fee={q_after}"
    )
    print(
        f"[binance-{label}] implied_return_vs_hint={implied_ret}"
    )

    print(f"\n=== Binance demo metadata ({label} leg, compact) ===")
    meta_for_log = dict(demo_meta)
    print(json.dumps(meta_for_log, indent=2))

    # Optional: show testnet result if present
    if "binance_testnet" in meta:
        print(f"\n=== Binance TESTNET raw result ({label} leg) ===")
        print(json.dumps(meta["binance_testnet"], indent=2))
    if "binance_testnet_error" in meta:
        print(f"\n=== Binance TESTNET error ({label} leg) ===")
        print(meta["binance_testnet_error"])


async def _run_binance_demo_test() -> None:
    """
    Dynamic check for BinanceDemoExchangeClient.

    Economic layer:
        - Uses MarketDataClient mid price as a theoretical hint
        - Applies deterministic slippage and taker fees on the quote asset
        - Updates BotState balances for BUY and SELL legs
        - Attaches binance_demo metadata with price and PnL details

    Infrastructure layer (optional):
        - If DEMO_BINANCE_USE_TESTNET_EXECUTION is set (and API key/secret
          are present), also sends real MARKET orders to Binance testnet.
        - The HTTP result is stored under trade.metadata["binance_testnet"]
          but does NOT affect balances or PnL.
    """
    instrument = os.getenv("DEMO_BINANCE_INSTRUMENT", "perpetual")
    symbol = os.getenv("DEMO_BINANCE_SYMBOL", os.getenv("MDC_HINT_SYMBOL", "BNBUSDT"))

    # BotState and internal account snapshot (off-chain accounting)
    state = BotState()
    account = state.get_or_create_account("Binance", instrument)

    # Initial quote balance for the demo
    starting_quote = float(os.getenv("DEMO_BINANCE_STARTING_QUOTE_BALANCE", "10000.0"))
    # We assume the quote asset is USDT; if the symbol does not end with USDT,
    # the test adjusts later based on the split_symbol helper.
    quote_asset = "USDT"
    account.deposit(quote_asset, starting_quote)

    print(
        f"[binance-account-before] instrument={instrument} "
        f"quote_asset={quote_asset} balance={account.balances.get(quote_asset)}"
    )

    # Demo parameters for Binance TESTNET REST API (used only if execution enabled)
    params = BinanceDemoParams(
        api_key=os.getenv("DEMO_BINANCE_SPOT_API_KEY", ""),
        api_secret=os.getenv("DEMO_BINANCE_API_SECRET", ""),
        base_url=os.getenv("DEMO_BINANCE_BASE_URL", "https://testnet.binance.vision"),
        recv_window_ms=int(os.getenv("DEMO_BINANCE_RECV_WINDOW_MS", "5000")),
        http_timeout_sec=float(os.getenv("DEMO_BINANCE_HTTP_TIMEOUT_SEC", "10.0")),
        use_testnet_execution=os.getenv("DEMO_BINANCE_USE_TESTNET_EXECUTION", "0") != "0",
    )

    client = BinanceDemoExchangeClient(
        name="Binance",
        instrument=instrument,
        state=state,
        params=params,
        fee_rate=float(os.getenv("DEMO_BINANCE_FEE_RATE", "0.0004")),
        default_slippage_bps=float(os.getenv("DEMO_BINANCE_DEFAULT_SLIPPAGE_BPS", "1.0")),
    )

    quantity = float(os.getenv("DEMO_BINANCE_TEST_QUANTITY", "0.1"))
    price_hint_env = float(os.getenv("DEMO_BINANCE_TEST_PRICE_HINT", "300.0"))

    # --- BUY leg: use real-time snapshot (or env default) as hint ---
    mdc_hint_buy = await _get_price_hint_from_market_data("BUY")
    price_hint_buy = mdc_hint_buy if mdc_hint_buy is not None else price_hint_env

    print(
        f"[demo-price-hint-BUY] price_hint_buy={price_hint_buy} "
        f"(env_default={price_hint_env}, from_market_data={mdc_hint_buy})"
    )

    trade_buy = await client.create_market_order(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=quantity,
        price=price_hint_buy,
    )

    assert trade_buy.success is True
    assert trade_buy.exchange == "Binance"
    assert trade_buy.symbol == symbol

    _inspect_binance_trade("BUY", trade_buy)

    # Derive base/quote assets from the symbol and adjust the quote ticker if needed
    base_asset, split_quote_asset = client._split_symbol(symbol)  # type: ignore[attr-defined]
    if split_quote_asset != quote_asset:
        quote_asset = split_quote_asset

    base_balance = account.balances.get(base_asset, 0.0)
    sell_qty = min(base_balance, quantity)

    if sell_qty <= 0:
        raise AssertionError(
            f"No base asset balance to sell after BUY leg "
            f"(asset={base_asset}, balance={base_balance})"
        )

    # --- SELL leg: refresh hint with a new real-time snapshot (or env default) ---
    price_hint_env_sell = float(
        os.getenv(
            "DEMO_BINANCE_TEST_PRICE_HINT_SELL",
            os.getenv("DEMO_BINANCE_TEST_PRICE_HINT", "300.0"),
        )
    )
    mdc_hint_sell = await _get_price_hint_from_market_data("SELL")
    price_hint_sell = mdc_hint_sell if mdc_hint_sell is not None else price_hint_env_sell

    print(
        f"[demo-price-hint-SELL] price_hint_sell={price_hint_sell} "
        f"(env_default={price_hint_env_sell}, from_market_data={mdc_hint_sell})"
    )

    trade_sell = await client.create_market_order(
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=sell_qty,
        price=price_hint_sell,
    )

    assert trade_sell.success is True
    assert trade_sell.exchange == "Binance"
    assert trade_sell.symbol == symbol

    _inspect_binance_trade("SELL", trade_sell)

    print("\n=== Final BotState balances (Binance demo) ===")
    print(account.balances)


@pytest.mark.skipif(
    bool(MISSING_ENVS),
    reason=(
        "Binance demo environment variables for price feed are not fully configured. "
        f"Missing: {', '.join(MISSING_ENVS)}"
    ),
)
def test_binance_demo_perpetual_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Run the Binance demo round-trip test for perpetual (futures) instrument.
    """
    monkeypatch.setenv("DEMO_BINANCE_INSTRUMENT", "perpetual")
    monkeypatch.setenv("MDC_HINT_INSTRUMENT", "perpetual")
    asyncio.run(_run_binance_demo_test())


@pytest.mark.skipif(
    bool(MISSING_ENVS),
    reason=(
        "Binance demo environment variables for price feed are not fully configured. "
        f"Missing: {', '.join(MISSING_ENVS)}"
    ),
)
def test_binance_demo_spot_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Run the Binance demo round-trip test for spot instrument.
    """
    monkeypatch.setenv("DEMO_BINANCE_INSTRUMENT", "spot")
    monkeypatch.setenv("MDC_HINT_INSTRUMENT", "spot")
    asyncio.run(_run_binance_demo_test())


if __name__ == "__main__":
    # When run directly: respect whatever is in the environment
    asyncio.run(_run_binance_demo_test())
