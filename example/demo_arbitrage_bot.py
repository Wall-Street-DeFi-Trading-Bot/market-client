"""
Demo trading example for cross-exchange arbitrage.

This script:
- Finds arbitrage opportunities using ArbitrageDetector
- Executes them in DEMO mode:
    * Binance: testnet via binance-connector
    * PancakeSwap: forked chain via web3 and your router
- Uses run_arbitrage_bot(config=..., mode=DEMO)
"""

import asyncio
import os
import signal

from web3 import Web3

from market_data_client.arbitrage.bot import run_arbitrage_bot
from market_data_client.arbitrage.config import BotConfig, ExecutionMode
from market_data_client.arbitrage.exchange import (
    BinanceDemoParams,
    PancakeDemoParams,
)


def build_pancake_v2_swap_tx(web3: Web3, symbol: str, quantity: float, side: str) -> dict:
    """
    User-defined callback to build a swap transaction for PancakeSwap V2.

    This function can capture router / token addresses via closure.
    The returned dict should be ready for sign_transaction.

    Example assumes:
      - router_v2 is a Contract instance
      - you swap between USDT and WBNB for BNBUSDT
      - quantity is in base asset units (BNB)
    """
    # ---------- customize below for your setup ----------
    router_v2 = build_pancake_router_v2(web3)  # your helper
    token_usdt = "0x..."  # fill with real address
    token_wbnb = "0x..."  # fill with real address

    base, quote = symbol[:-4], symbol[-4:]  # "BNBUSDT" -> "BNB", "USDT"

    # For simplicity, assume BUY = spend quote, receive base.
    # You probably want to convert quantity * price into quote units here.
    # This is intentionally left as "amount_in" stub.
    amount_in = 0  # TODO: convert quantity into quote token units

    if side.upper() == "BUY":
        path = [token_usdt, token_wbnb]
    else:
        path = [token_wbnb, token_usdt]

    deadline = web3.eth.get_block("latest")["timestamp"] + 60

    tx = router_v2.functions.swapExactTokensForTokens(
        amount_in,
        0,  # amountOutMin: for demo you can keep it 0
        path,
        os.environ["DEMO_ACCOUNT_ADDRESS"],
        deadline,
    ).build_transaction(
        {
            "to": router_v2.address,
            "gas": 800000,
            "gasPrice": web3.to_wei("5", "gwei"),
            # "value": 0 for ERC20 swap
        }
    )

    return tx
    # ---------- end of user customization ----------


def build_pancake_router_v2(web3: Web3):
    """
    Helper to build PancakeSwap V2 router contract.

    You need to provide:
      - router address
      - router ABI (loaded from JSON or hard-coded)
    """
    router_address = Web3.to_checksum_address("0x...")
    router_abi = [...]  # load from your JSON ABI
    return web3.eth.contract(address=router_address, abi=router_abi)


async def main() -> None:
    # Symbols to monitor
    symbols = ["BNBUSDT"]

    # (exchange, instrument) pairs
    exchanges = [
        ("Binance", "spot"),
        ("Binance", "perpetual"),
        ("PancakeSwapV2", "swap"),
        ("PancakeSwapV3", "swap"),
    ]

    # Starting balances used for DEMO (local mirror in BotState)
    paper_initial_balances = {
        ("Binance", "spot"): {
            "USDT": 10_000.0,
            "BNB": 100.0,
        },
        ("Binance", "perpetual"): {
            "USDT": 10_000.0,
        },
        ("PancakeSwapV2", "swap"): {
            "USDT": 10_000.0,
            "BNB": 100.0,
        },
        ("PancakeSwapV3", "swap"): {
            "USDT": 10_000.0,
            "BNB": 100.0,
        },
    }

    config = BotConfig(
        mode=ExecutionMode.DEMO,
        nats_url="nats://127.0.0.1:4222",
        symbols=symbols,
        exchanges=exchanges,
        min_profit_pct=0.1,        # execute only if net profit >= 0.1%
        trade_notional_usd=100.0,  # trade size per opportunity (in USDT)
        scan_interval=0.5,         # base interval; can be overridden below
        paper_initial_balances=paper_initial_balances,
    )

    # CEX symbol -> DEX pair mapping
    symbol_mapping = {
        "BNBUSDT": {
            "PancakeSwapV2": "USDTWBNB",
            "PancakeSwapV3": "USDTWBNB",
        },
    }

    # --- DEMO params injection ---

    # Binance demo: testnet keys and URLs from env
    binance_demo_params = {
        ("Binance", "spot"): BinanceDemoParams(
            api_key=os.environ["BINANCE_SPOT_TESTNET_KEY"],
            api_secret=os.environ["BINANCE_SPOT_TESTNET_SECRET"],
            base_url="https://testnet.binance.vision",
        ),
        ("Binance", "perpetual"): BinanceDemoParams(
            api_key=os.environ["BINANCE_PERP_TESTNET_KEY"],
            api_secret=os.environ["BINANCE_PERP_TESTNET_SECRET"],
            base_url="https://testnet.binancefuture.com",  # adjust if needed
        ),
    }

    # Web3 instance connected to your forked node (e.g. anvil / hardhat)
    web3 = Web3(Web3.HTTPProvider(os.environ["FORK_RPC_URL"]))

    pancake_demo_params = {
        ("PancakeSwapV2", "swap"): PancakeDemoParams(
            web3=web3,
            account_address=os.environ["DEMO_ACCOUNT_ADDRESS"],
            private_key=os.environ["DEMO_ACCOUNT_PRIVATE_KEY"],
            build_swap_tx=build_pancake_v2_swap_tx,
            block_offsets=(1, 2, 3),
            default_fee_rate=0.0005,
        ),
        # You can add a separate builder for V3 if needed
        ("PancakeSwapV3", "swap"): PancakeDemoParams(
            web3=web3,
            account_address=os.environ["DEMO_ACCOUNT_ADDRESS"],
            private_key=os.environ["DEMO_ACCOUNT_PRIVATE_KEY"],
            build_swap_tx=build_pancake_v2_swap_tx,
            block_offsets=(1, 2, 3),
            default_fee_rate=0.0005,
        ),
    }

    stop_event = asyncio.Event()

    def _handle_sigint(signum, frame):
        """
        Handle Ctrl+C (SIGINT).

        First Ctrl+C:
            - Set stop_event so the bot can exit its loop cleanly.
        Second Ctrl+C:
            - Raise KeyboardInterrupt to force exit.
        """
        if not stop_event.is_set():
            print("\n\n✅ Stopping demo arbitrage bot gracefully...")
            stop_event.set()
        else:
            print("\n\n⛔ Force exit.")
            raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _handle_sigint)

    # scan_sleep_seconds: time between scans (can be longer than config.scan_interval)
    scan_sleep_seconds = 12.0  # enough time to run fork tests 3 times per trade

    await run_arbitrage_bot(
        config=config,
        symbol_mapping=symbol_mapping,
        stop_event=stop_event,
        scan_sleep_seconds=scan_sleep_seconds,
        demo_binance_params=binance_demo_params,
        demo_pancake_params=pancake_demo_params,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✅ Stopped demo arbitrage bot")
