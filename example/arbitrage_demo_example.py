import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Dict, Tuple

from web3 import HTTPProvider, Web3

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("DEMO_PANCAKE_ABI", str(ROOT / "contracts" / "abi"))

from market_data_client.arbitrage.bot import run_arbitrage_bot
from market_data_client.arbitrage.config import BotConfig, ExecutionMode
from market_data_client.arbitrage.exchange import BinanceDemoParams, PancakeDemoParams
from market_data_client.arbitrage.dex.web3_compat import disable_poa_extra_data_validation

from market_data_client.arbitrage.dex.routers.universal_router_v2 import (
    build_swap_tx_router_v3,
    build_swap_tx_router_v2,
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.getLogger("market_data_client.arbitrage.exchange").setLevel(logging.INFO)


def _set_first_existing(obj: object, names: list[str], value: object) -> None:
    for n in names:
        if hasattr(obj, n):
            setattr(obj, n, value)
            return
    raise AttributeError(f"{type(obj).__name__} has none of these attributes: {names}")


def _build_binance_params(prefix: str, kind: str, default_base_url: str) -> BinanceDemoParams:
    p = BinanceDemoParams()

    api_key = os.getenv(f"{prefix}_API_KEY", "")
    api_secret = os.getenv(f"{prefix}_API_SECRET", "")
    base_url = os.getenv(f"{prefix}_BASE_URL", default_base_url)

    recv_window_ms = int(os.getenv("DEMO_BINANCE_RECV_WINDOW_MS", "5000"))
    http_timeout_sec = float(os.getenv("DEMO_BINANCE_HTTP_TIMEOUT_SEC", "10.0"))
    use_testnet_execution = os.getenv("DEMO_BINANCE_USE_TESTNET_EXECUTION", "0") != "0"
    fail_on_testnet_error = os.getenv("DEMO_BINANCE_FAIL_ON_TESTNET_ERROR", "1") != "0"

    if kind == "spot":
        _set_first_existing(p, ["spot_api_key", "api_key", "apiKey", "key"], api_key)
        _set_first_existing(p, ["spot_api_secret", "api_secret", "apiSecret", "secret"], api_secret)
        _set_first_existing(p, ["spot_base_url", "base_url", "baseUrl"], base_url)
    elif kind == "futures":
        _set_first_existing(p, ["futures_api_key", "api_key", "apiKey", "key"], api_key)
        _set_first_existing(p, ["futures_api_secret", "api_secret", "apiSecret", "secret"], api_secret)
        _set_first_existing(p, ["futures_base_url", "base_url", "baseUrl"], base_url)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    _set_first_existing(p, ["recv_window_ms", "recvWindowMs"], recv_window_ms)
    _set_first_existing(p, ["http_timeout_sec", "httpTimeoutSec"], http_timeout_sec)
    _set_first_existing(p, ["use_testnet_execution", "useTestnetExecution"], use_testnet_execution)
    _set_first_existing(p, ["fail_on_testnet_error", "failOnTestnetError"], fail_on_testnet_error)

    return p


async def main() -> None:
    symbols = ["BNBUSDT"]

    exchanges = [
        ("Binance", "spot"),
        ("Binance", "perpetual"),
        ("PancakeSwapV3", "swap"),
        ("PancakeSwapV2", "swap"),
    ]

    paper_initial_balances = {
        ("Binance", "spot"): {"USDT": 10_000.0, "BNB": 100.0},
        ("Binance", "perpetual"): {"USDT": 10_000.0, "BNB": 100.0},
        ("PancakeSwapV3", "swap"): {"USDT": 10_000.0, "WBNB": 100.0, "BNB": 2.0},
        ("PancakeSwapV2", "swap"): {"USDT": 10_000.0, "WBNB": 100.0, "BNB": 2.0},
    }

    config = BotConfig(
        mode=ExecutionMode.DEMO,
        nats_url=os.getenv("NATS_URL", "nats://127.0.0.1:4222"),
        symbols=symbols,
        exchanges=exchanges,
        min_profit_pct=float(os.getenv("BOT_MIN_PROFIT_PCT", "0.1")),
        trade_notional_usd=float(os.getenv("BOT_TRADE_NOTIONAL_USD", "100.0")),
        scan_interval=float(os.getenv("BOT_SCAN_INTERVAL_SEC", "5.0")),
        paper_initial_balances=paper_initial_balances,
    )

    symbol_mapping = {
        "BNBUSDT": {
            "PancakeSwapV3": os.getenv("BOT_PANCAKE_PAIR", "USDTWBNB"),
            "PancakeSwapV2": os.getenv("BOT_PANCAKE_PAIR", "USDTWBNB"),
        }
    }

    demo_binance_spot = _build_binance_params(
        prefix="DEMO_BINANCE_SPOT",
        kind="spot",
        default_base_url="https://testnet.binance.vision",
    )

    demo_binance_futures = _build_binance_params(
        prefix="DEMO_BINANCE_FUTURES",
        kind="futures",
        default_base_url="https://testnet.binancefuture.com",
    )

    if not os.getenv("DEMO_BINANCE_FUTURES_API_KEY", "") or not os.getenv("DEMO_BINANCE_FUTURES_API_SECRET", ""):
        try:
            _set_first_existing(demo_binance_futures, ["use_testnet_execution", "useTestnetExecution"], False)
            _set_first_existing(demo_binance_futures, ["fail_on_testnet_error", "failOnTestnetError"], False)
        except Exception:
            pass

    demo_binance_params: Dict[Tuple[str, str], BinanceDemoParams] = {
        ("Binance", "spot"): demo_binance_spot,
        ("Binance", "perpetual"): demo_binance_futures,
    }

    fork_rpc_url = os.environ["DEMO_PANCAKE_FORK_RPC_URL"]
    upstream_rpc_url = os.environ["DEMO_PANCAKE_UPSTREAM_RPC_URL"]
    fork_engine = os.getenv("DEMO_PANCAKE_FORK_ENGINE", "anvil")
    private_key = os.environ["DEMO_PANCAKE_TRADER_PRIVATE_KEY"].strip()

    disable_poa_extra_data_validation()
    web3 = Web3(HTTPProvider(fork_rpc_url))

    # V3 uses V3 builder
    demo_pancake_v3 = PancakeDemoParams(
        web3=web3,
        account_address="",
        build_swap_tx=build_swap_tx_router_v3,
        private_key=private_key,
        block_offsets=(1,),
        default_fee_rate=float(os.getenv("DEMO_PANCAKE_DEFAULT_FEE_RATE", "0.0005")),
    )

    # V2 uses V2 builder
    demo_pancake_v2 = PancakeDemoParams(
        web3=web3,
        account_address="",
        build_swap_tx=build_swap_tx_router_v2,
        private_key=private_key,
        block_offsets=(1, ),
        default_fee_rate=float(os.getenv("DEMO_PANCAKE_DEFAULT_FEE_RATE", "0.0005")),
    )

    demo_pancake_v3.upstream_rpc_url = upstream_rpc_url  # type: ignore[attr-defined]
    demo_pancake_v3.fork_engine = fork_engine  # type: ignore[attr-defined]
    demo_pancake_v2.upstream_rpc_url = upstream_rpc_url  # type: ignore[attr-defined]
    demo_pancake_v2.fork_engine = fork_engine  # type: ignore[attr-defined]

    demo_pancake_params: Dict[Tuple[str, str], PancakeDemoParams] = {
        ("PancakeSwapV3", "swap"): demo_pancake_v3,
        ("PancakeSwapV2", "swap"): demo_pancake_v2,
    }

    stop_event = asyncio.Event()

    def _handle_sigint(signum, frame) -> None:
        if not stop_event.is_set():
            stop_event.set()
        else:
            raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _handle_sigint)

    await run_arbitrage_bot(
        config=config,
        symbol_mapping=symbol_mapping,
        stop_event=stop_event,
        demo_binance_params=demo_binance_params,
        demo_pancake_params=demo_pancake_params,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
